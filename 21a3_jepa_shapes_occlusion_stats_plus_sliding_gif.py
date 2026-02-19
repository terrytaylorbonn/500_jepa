"""
21a3_jepa_shapes_occlusion_stats_plus_sliding_gif.py
[21a3] JEPA shapes + occlusion (distribution stats) + sliding occluder GIF

This is 21a2 + a visual sanity-check:
  - Generate a single base image for a chosen shape
  - Slide a rectangular occluder left->right
  - Print cosine(z_full, z_occ_frame) while generating frames
  - Save a GIF with cosine overlaid in the title
"""

import os
import math
import random
from collections import defaultdict

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# For GIF
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError:
    plt = None
    FuncAnimation = None
    PillowWriter = None

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
# GIF demo config (NEW)
# -----------------------------
MAKE_SLIDING_GIF = True
GIF_SHAPE = "circle"           # choose from SHAPES
GIF_OUT = "21a3_sliding_occluder.gif"
GIF_FRAMES = 40
GIF_FPS = 8

# Sliding occluder size/placement (relative to image)
SLIDE_OCC_W_FRAC = 0.25        # occluder width as fraction of IMG_SIZE
SLIDE_OCC_H_FRAC = 0.35        # occluder height as fraction of IMG_SIZE
SLIDE_OCC_Y_FRAC = 0.30        # occluder top y position as fraction of IMG_SIZE
SLIDE_OCC_COLOR = (128, 128, 128)  # mid-gray to match your random occluder


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
# Sliding occluder helper (NEW)
# -----------------------------
def apply_sliding_occluder(img: Image.Image, x0: int, y0: int, w: int, h: int, color=(128, 128, 128)) -> Image.Image:
    """
    Apply a rectangular occluder at (x0,y0) with size (w,h).
    x0 can be negative or beyond bounds; we clip safely.
    """
    img_occ = img.copy()
    draw = ImageDraw.Draw(img_occ)

    x1 = x0 + w
    y1 = y0 + h

    # Clip to image bounds
    x0c = max(0, min(IMG_SIZE, x0))
    y0c = max(0, min(IMG_SIZE, y0))
    x1c = max(0, min(IMG_SIZE, x1))
    y1c = max(0, min(IMG_SIZE, y1))

    if x1c > x0c and y1c > y0c:
        draw.rectangle([x0c, y0c, x1c, y1c], fill=color)

    return img_occ


# -----------------------------
# JEPA embedding helper
# -----------------------------

def load_jepa():
    print(f"[21a3] Device: {DEVICE}")
    print(f"[21a3] Loading I-JEPA processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    print("[21a3] Processor loaded.")
    print(f"[21a3] Loading I-JEPA encoder on {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print("[21a3] Encoder loaded.")
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
# Sliding GIF maker (NEW)
# -----------------------------
def make_sliding_occluder_gif(shape: str, processor, model, out_gif: str):
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("[21a3] matplotlib animation not available; skipping GIF.")
        return

    if shape not in SHAPES:
        raise ValueError(f"GIF_SHAPE must be one of {SHAPES}, got {shape}")

    # Make a deterministic "base" image (still random w.r.t RNG state)
    base_img = generate_shape_image(shape)

    # Baseline embedding (THIS is your z_full for the GIF demo)
    z_full = embed_image(base_img, processor, model)  # (D,)

    occ_w = int(IMG_SIZE * SLIDE_OCC_W_FRAC)
    occ_h = int(IMG_SIZE * SLIDE_OCC_H_FRAC)
    occ_y = int(IMG_SIZE * SLIDE_OCC_Y_FRAC)

    # Slide occluder left -> right; allow starting off-canvas
    x_positions = torch.linspace(-occ_w, IMG_SIZE, GIF_FRAMES).to(torch.int32).tolist()

    frames = []
    cos_vals = []

    print(f"\n[21a3] Sliding occluder GIF for shape='{shape}'")
    print("[21a3] Live cosine(z_full, z_occ_frame):")
    for k, x0 in enumerate(x_positions):
        occ_img = apply_sliding_occluder(
            base_img,
            x0=int(x0),
            y0=occ_y,
            w=occ_w,
            h=occ_h,
            color=SLIDE_OCC_COLOR
        )
        z_occ = embed_image(occ_img, processor, model)  # (D,)

        cos = cosine_similarity(z_full.unsqueeze(0), z_occ.unsqueeze(0)).item()
        cos_vals.append(cos)
        frames.append(occ_img)

        print(f"  frame {k+1:02d}/{GIF_FRAMES:02d}  x0={int(x0):4d}  cos={cos:.4f}")

    # Animate
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_axis_off()

    im = ax.imshow(frames[0])
    title = ax.set_title(f"{shape} | cos={cos_vals[0]:.4f} (1/{GIF_FRAMES})")

    def update(i):
        im.set_data(frames[i])
        title.set_text(f"{shape} | cos={cos_vals[i]:.4f} ({i+1}/{GIF_FRAMES})")
        return [im, title]

    ani = FuncAnimation(fig, update, frames=GIF_FRAMES, interval=int(1000 / GIF_FPS), blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=GIF_FPS))
    plt.close(fig)

    print(f"[21a3] Saved GIF: {out_gif}\n")


# -----------------------------
# Main
# -----------------------------

def main():
    processor, model = load_jepa()

    # Optional: make sliding occluder GIF once (NEW)
    if MAKE_SLIDING_GIF:
        make_sliding_occluder_gif(GIF_SHAPE, processor, model, GIF_OUT)

    same_shape_cos = defaultdict(list)  # shape -> list of cos(full, occ)
    diff_shape_cos = defaultdict(list)  # "A_vs_B" -> list of cos(fullA, fullB)

    print(f"[21a3] Sampling {N_SAMPLES_PER_SHAPE} variants per shape...")

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

    print("\n[21a3] Cosine similarity: full vs occluded (same shape)")
    for shape in SHAPES:
        m, s = mean_std(same_shape_cos[shape])
        print(f"  {shape:8s}: mean = {m:.4f}, std = {s:.4f}, n = {len(same_shape_cos[shape])}")

    print("\n[21a3] Cosine similarity: full vs full (different shapes)")
    for key in sorted(diff_shape_cos.keys()):
        m, s = mean_std(diff_shape_cos[key])
        print(f"  {key:15s}: mean = {m:.4f}, std = {s:.4f}, n = {len(diff_shape_cos[key])}")

    # Optional quick textual interpretation
    print("\n[21a3] Interpretation (rough):")
    print("  - If mean_same(shape) is NOT clearly higher than mean_diff(A_vs_B),")
    print("    JEPA CLS is not providing clean shape identity under occlusion for this toy setup.")
    print("  - Use this as a *probe* of representation quality, not as proof of object understanding.")


if __name__ == "__main__":
    main()
