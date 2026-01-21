"""
21a_jepa_shapes_occlusion_demo.py

JEPA occlusion robustness demo (synthetic shapes).

What it does:
- Generate simple RGB images with one shape: circle, square, triangle.
- Generate occluded versions by overlaying a gray rectangle.
- Encode both with I-JEPA (facebook/ijepa_vith14_1k).
- Print cosine similarities between:
    - full vs occluded of same shape
    - full vs full of different shapes (as a sanity contrast)

Interpretation:
- High sim(full, occluded same shape) and lower sim(full, other shapes)
  suggests JEPA encodes object identity robustly to occlusion.
"""

import math
import argparse
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor


# -------------------------------------------------------------
# Image generation: simple shapes + occlusion
# -------------------------------------------------------------

def make_blank(img_size: int = 224, color=(255, 255, 255)) -> Image.Image:
    return Image.new("RGB", (img_size, img_size), color=color)


def draw_shape(
    shape: str,
    img_size: int = 224,
    shape_color=(0, 0, 0),
) -> Image.Image:
    """
    shape: "circle", "square", "triangle"
    """
    img = make_blank(img_size)
    draw = ImageDraw.Draw(img)

    # Simple centered shapes
    margin = img_size // 4
    x0, y0 = margin, margin
    x1, y1 = img_size - margin, img_size - margin

    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=shape_color)
    elif shape == "square":
        draw.rectangle([x0, y0, x1, y1], fill=shape_color)
    elif shape == "triangle":
        # Equilateral-ish triangle pointing up
        top = (img_size // 2, margin)
        left = (margin, img_size - margin)
        right = (img_size - margin, img_size - margin)
        draw.polygon([top, left, right], fill=shape_color)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    return img


def add_occlusion(
    img: Image.Image,
    frac: float = 0.4,
    orientation: str = "vertical",
    occ_color=(180, 180, 180),
) -> Image.Image:
    """
    Overlays a gray rectangle occluding part of the image.

    frac: fraction of width/height to occlude
    orientation: "vertical" or "horizontal"
    """
    img_occ = img.copy()
    draw = ImageDraw.Draw(img_occ)
    W, H = img_occ.size

    if orientation == "vertical":
        occ_w = int(W * frac)
        x0 = (W - occ_w) // 2
        x1 = x0 + occ_w
        y0 = 0
        y1 = H
    else:  # horizontal
        occ_h = int(H * frac)
        y0 = (H - occ_h) // 2
        y1 = y0 + occ_h
        x0 = 0
        x1 = W

    draw.rectangle([x0, y0, x1, y1], fill=occ_color)
    return img_occ


# -------------------------------------------------------------
# I-JEPA helpers
# -------------------------------------------------------------

def load_ijepa(model_id: str, device: torch.device):
    print(f"[21a] Loading I-JEPA processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print("[21a] Processor loaded.", flush=True)

    print(f"[21a] Loading I-JEPA encoder on {device}...", flush=True)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    print(f"[21a] Encoder loaded on {device}.", flush=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, processor


@torch.no_grad()
def encode_image_to_latent(
    img: Image.Image,
    encoder,
    processor,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a (D,) latent vector (CLS).
    """
    img = img.convert("RGB")
    inputs = processor(img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = encoder(**inputs)
    z = outputs.last_hidden_state[:, 0, :]  # (1, D)
    return z.squeeze(0)  # (D,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[21a] Device: cuda", flush=True)
        return torch.device("cuda")
    else:
        print("[21a] Device: cpu", flush=True)
        return torch.device("cpu")


# -------------------------------------------------------------
# Main demo
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="I-JEPA model id.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image resolution for synthetic shapes.",
    )
    parser.add_argument(
        "--occlusion-frac",
        type=float,
        default=0.4,
        help="Fraction of image to occlude.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default="vertical",
        choices=["vertical", "horizontal"],
        help="Occlusion orientation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (for completeness, though shapes are deterministic).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    encoder, processor = load_ijepa(args.model_id, device)

    shapes = ["circle", "square", "triangle"]
    print("\n[21a] Generating shapes and occlusions...\n", flush=True)

    # Store full + occluded latents
    z_full = {}
    z_occ = {}

    for shape in shapes:
        # Full image
        img_full = draw_shape(shape, img_size=args.img_size, shape_color=(0, 0, 0))
        # Occluded image
        img_occ = add_occlusion(
            img_full,
            frac=args.occlusion_frac,
            orientation=args.orientation,
            occ_color=(180, 180, 180),
        )

        # Encode both
        z_f = encode_image_to_latent(img_full, encoder, processor, device)
        z_o = encode_image_to_latent(img_occ, encoder, processor, device)
        z_full[shape] = z_f
        z_occ[shape] = z_o

    # ---------------------------------------------------------
    # Cosine similarities
    # ---------------------------------------------------------
    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    print("[21a] Cosine similarity: full vs occluded (same shape)")
    for shape in shapes:
        s = cos(z_full[shape], z_occ[shape])
        print(f"  {shape:8s}: cos(z_full, z_occ) = {s:.4f}")
    print()

    print("[21a] Cosine similarity: full vs full (different shapes)")
    for i, s1 in enumerate(shapes):
        for j, s2 in enumerate(shapes):
            if i >= j:
                continue
            c_ff = cos(z_full[s1], z_full[s2])
            print(f"  {s1:8s} vs {s2:8s}: cos = {c_ff:.4f}")
    print()

    print("[21a] Interpretation:")
    print("  - If cos(full, occluded same shape) is high (e.g. > 0.8),")
    print("    and cos(full, full other shape) is lower,")
    print("    then JEPA is encoding object identity robustly under occlusion.")
    print("")


if __name__ == "__main__":
    main()
