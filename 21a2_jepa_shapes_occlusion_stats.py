"""
21a2_jepa_shapes_occlusion_stats.py
[21a2] JEPA shapes + occlusion (distribution stats)

Goal:
  - Sample MANY random variants of circle/square/triangle
  - For each:
      full image, occluded image → CLS embeddings
  - Measure:
      * same-shape:  cos( full, occluded )
      * diff-shape:  cos( full(shape A), full(shape B) )
  - Report mean + std for each group.

Interpretation:
  - If mean_same(shape) >> mean_diff(shape vs others),
    then JEPA CLS is somewhat shape-identity aware under occlusion.
  - If they overlap heavily, 21a’s original narrative was too generous.
"""

import os
import math
import random
from collections import defaultdict

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "facebook/ijepa_vith14_1k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
N_SAMPLES_PER_SHAPE = 64  # you can bump this if GPU is fast
OCCLUSION_FRAC = 0.4      # fraction of width/height that occluder covers

SHAPES = ["circle", "square", "triangle"]
RNG = random.Random(42)


# -----------------------------
# Shape generator
# -----------------------------

def generate_shape_image(shape: str) -> Image.Image:
    """Generate a simple white-on-black image with one shape."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # random position and size (still centered-ish)
    margin = IMG_SIZE // 6
    x0 = RNG.randint(margin, IMG_SIZE // 2)
    y0 = RNG.randint(margin, IMG_SIZE // 2)
    x1 = RNG.randint(IMG_SIZE // 2, IMG_SIZE - margin)
    y1 = RNG.randint(IMG_SIZE // 2, IMG_SIZE - margin)

    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=(255, 255, 255))
    elif shape == "square":
        # enforce square-ish
        side = min(x1 - x0, y1 - y0)
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=(255, 255, 255))
    elif shape == "triangle":
        # triangle using three vertices
        x_mid = (x0 + x1) // 2
        points = [(x_mid, y0), (x0, y1), (x1, y1)]
        draw.polygon(points, fill=(255, 255, 255))
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return img


def apply_occlusion(img: Image.Image) -> Image.Image:
    """Apply a random rectangular occluder."""
    img_occ = img.copy()
    draw = ImageDraw.Draw(img_occ)

    occ_w = int(IMG_SIZE * OCCLUSION_FRAC)
    occ_h = int(IMG_SIZE * OCCLUSION_FRAC)

    x0 = RNG.randint(0, IMG_SIZE - occ_w)
    y0 = RNG.randint(0, IMG_SIZE - occ_h)
    x1 = x0 + occ_w
    y1 = y0 + occ_h

    # occluder color: mid-gray
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))

    return img_occ


# -----------------------------
# JEPA embedding helper
# -----------------------------

def load_jepa():
    print(f"[21a2] Device: {DEVICE}")
    print(f"[21a2] Loading I-JEPA processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    print("[21a2] Processor loaded.")
    print(f"[21a2] Loading I-JEPA encoder on {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print("[21a2] Encoder loaded.")
    return processor, model


@torch.no_grad()
def embed_image(pil_img: Image.Image, processor, model) -> torch.Tensor:
    """
    Return a (D,) CLS embedding tensor for JEPA.
    """
    pil_img = pil_img.convert("RGB")
    inputs = processor(pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)

    # I-JEPA ViT: use CLS token from last_hidden_state
    cls = outputs.last_hidden_state[:, 0, :]  # (1, D)
    return cls.squeeze(0)  # (D,)


# -----------------------------
# Main
# -----------------------------

def main():
    processor, model = load_jepa()

    same_shape_cos = defaultdict(list)  # shape -> list of cos(full, occ)
    diff_shape_cos = defaultdict(list)  # "A_vs_B" -> list of cos(fullA, fullB)

    print(f"[21a2] Sampling {N_SAMPLES_PER_SHAPE} variants per shape...")

    for i in range(N_SAMPLES_PER_SHAPE):
        # store full embeddings for this iteration to build diff-shape comparisons
        full_embeddings = {}

        for shape in SHAPES:
            full_img = generate_shape_image(shape)
            occ_img = apply_occlusion(full_img)

            z_full = embed_image(full_img, processor, model)      # (D,)
            z_occ = embed_image(occ_img, processor, model)        # (D,)

            full_embeddings[shape] = z_full

            cos_same = cosine_similarity(
                z_full.unsqueeze(0), z_occ.unsqueeze(0)
            ).item()
            same_shape_cos[shape].append(cos_same)

        # across shapes, compare full vs full
        for a in SHAPES:
            for b in SHAPES:
                if a >= b:
                    continue  # avoid duplicates + self
                key = f"{a}_vs_{b}"
                za = full_embeddings[a]
                zb = full_embeddings[b]
                cos_diff = cosine_similarity(
                    za.unsqueeze(0), zb.unsqueeze(0)
                ).item()
                diff_shape_cos[key].append(cos_diff)

    # -------------------------
    # Report statistics
    # -------------------------
    def mean_std(xs):
        if not xs:
            return float("nan"), float("nan")
        t = torch.tensor(xs)
        return t.mean().item(), t.std(unbiased=False).item()

    print("\n[21a2] Cosine similarity: full vs occluded (same shape)")
    for shape in SHAPES:
        m, s = mean_std(same_shape_cos[shape])
        print(f"  {shape:8s}: mean = {m:.4f}, std = {s:.4f}, n = {len(same_shape_cos[shape])}")

    print("\n[21a2] Cosine similarity: full vs full (different shapes)")
    for key in sorted(diff_shape_cos.keys()):
        m, s = mean_std(diff_shape_cos[key])
        print(f"  {key:15s}: mean = {m:.4f}, std = {s:.4f}, n = {len(diff_shape_cos[key])}")

    # Optional quick textual interpretation
    print("\n[21a2] Interpretation (rough):")
    print("  - If mean_same(shape) is NOT clearly higher than mean_diff(A_vs_B),")
    print("    JEPA CLS is not providing clean shape identity under occlusion for this toy setup.")
    print("  - Use this as a *probe* of representation quality, not as proof of object understanding.")


if __name__ == "__main__":
    main()
