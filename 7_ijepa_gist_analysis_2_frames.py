# 7_ijepa_gist_analysis_2_frames.py

#!/usr/bin/env python3
"""
ijepa_gist_analysis_2_frames.py

Goal:
- Understand the runtime workflow for TWO observations (frame t, frame t+1)
- Inspect what data is produced at each step
- Verify that:
    global belief (CLS) changes less than local evidence (patch tokens)

No training. No predictors. No decoders. Pure analysis.
"""

import argparse
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def extract_latents(model, processor, img, device):
    """Return CLS token and patch tokens for one image."""
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs, return_dict=True)
    h = outputs.last_hidden_state  # (1, 1+N, D)

    z_cls = h[:, 0, :]      # (1, D)
    z_patches = h[:, 1:, :] # (1, N, D)

    return z_cls, z_patches


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image1", required=True, help="Frame t image")
    ap.add_argument("--image2", required=True, help="Frame t+1 image")
    ap.add_argument("--model_id", default="facebook/ijepa_vith14_1k")
    args = ap.parse_args()

    # ------------------------------------------------------------
    # [0] DEVICE
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[0] DEVICE\n  device = {device}\n")

    # ------------------------------------------------------------
    # [1] LOAD MODEL + PROCESSOR
    # ------------------------------------------------------------
    print("[1] LOAD\n  loading processor + model...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    print("  done.\n")

    # ------------------------------------------------------------
    # [2] INPUT OBSERVATIONS
    # ------------------------------------------------------------
    print("[2] INPUT\n  reading two images...")
    img1 = Image.open(args.image1).convert("RGB")
    img2 = Image.open(args.image2).convert("RGB")
    print(f"  frame t   : size={img1.size}")
    print(f"  frame t+1 : size={img2.size}\n")

    # ------------------------------------------------------------
    # [3] ENCODER → LATENT TOKENS (frame t)
    # ------------------------------------------------------------
    print("[3] ENCODE FRAME t")
    z_cls_1, z_patches_1 = extract_latents(model, processor, img1, device)
    print(f"  z_cls(t)     shape={tuple(z_cls_1.shape)}")
    print(f"  z_patches(t) shape={tuple(z_patches_1.shape)}\n")

    # ------------------------------------------------------------
    # [4] ENCODER → LATENT TOKENS (frame t+1)
    # ------------------------------------------------------------
    print("[4] ENCODE FRAME t+1")
    z_cls_2, z_patches_2 = extract_latents(model, processor, img2, device)
    print(f"  z_cls(t+1)     shape={tuple(z_cls_2.shape)}")
    print(f"  z_patches(t+1) shape={tuple(z_patches_2.shape)}\n")

    # ------------------------------------------------------------
    # [5] TEMPORAL COMPARISON — GLOBAL BELIEF
    # ------------------------------------------------------------
    print("[5] GLOBAL BELIEF CHANGE (CLS)")
    cls_sim = float(
        cosine_similarity(
            l2_normalize(z_cls_1), l2_normalize(z_cls_2)
        ).item()
    )
    print(f"  cosine( CLS_t , CLS_t+1 ) = {cls_sim:.4f}\n")

    # ------------------------------------------------------------
    # [6] TEMPORAL COMPARISON — LOCAL EVIDENCE
    # ------------------------------------------------------------
    print("[6] LOCAL EVIDENCE CHANGE (PATCH TOKENS)")
    p1 = l2_normalize(z_patches_1.squeeze(0))  # (N, D)
    p2 = l2_normalize(z_patches_2.squeeze(0))  # (N, D)

    patch_sims = cosine_similarity(p1, p2, dim=-1)  # (N,)
    ps = patch_sims.detach().cpu()

    print(f"  patch similarity stats:")
    print(f"    min    = {float(ps.min()):.4f}")
    print(f"    mean   = {float(ps.mean()):.4f}")
    print(f"    median = {float(ps.median()):.4f}")
    print(f"    max    = {float(ps.max()):.4f}")
    print(f"    std    = {float(ps.std()):.4f}\n")

    # ------------------------------------------------------------
    # [7] INTERPRETATION
    # ------------------------------------------------------------
    print("[7] INTERPRETATION")
    if cls_sim > float(ps.median()):
        print("  ✅ Global belief (CLS) changes LESS than typical patch tokens.")
    else:
        print("  ⚠️  Global belief (CLS) does NOT change less than patches.")

    frac = float((ps < cls_sim).float().mean())
    print(f"  Fraction of patches changing more than CLS: {frac:.2%}")

    print("\nDONE.\n")


if __name__ == "__main__":
    main()
