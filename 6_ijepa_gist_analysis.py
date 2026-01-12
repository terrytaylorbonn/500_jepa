# 6_jepa_gist_analysis.py

#!/usr/bin/env python3
"""
ijepa_gist_analysis.py

Goal: Understand the gist of runtime inference for I-JEPA (ViT encoder):
- Image (PIL) → processor → pixel_values tensor
- Model → last_hidden_state tokens
- Split tokens into:
    CLS token (global belief summary)
    Patch tokens (local evidence representations)
- Make a simple global readout:
    CLS global
    Mean-of-patches global
- Compute simple similarities to analyze behavior

This is runtime-only: no training, no gradients.
"""

import argparse
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--model_id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--topk", type=int, default=8, help="How many top aligned patches to show")
    args = ap.parse_args()

    # 0) Device (where computation runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[0] DEVICE\n  device = {device}\n")

    # 1) Load processor + model
    print("[1] LOAD\n  loading processor + model...")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()
    print("  done.\n")

    # 2) INPUT (raw observation)
    print("[2] INPUT\n  reading image...")
    img = Image.open(args.image).convert("RGB")
    print(f"  PIL image mode={img.mode}, size={img.size} (W,H)\n")

    # 3) PREPROCESS (translate image → tensor)
    print("[3] PREPROCESS\n  processor(images=img) → inputs dict")
    inputs = processor(images=img, return_tensors="pt")
    # Usually contains pixel_values for vision models
    for k, v in inputs.items():
        print(f"  inputs['{k}'] shape={tuple(v.shape)} dtype={v.dtype}")
    print()

    # Move tensors to GPU/CPU device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(f"  moved inputs to device={device}\n")

    # 4) ENCODER FORWARD PASS (inference)
    print("[4] ENCODER\n  outputs = model(**inputs)")
    outputs = model(**inputs, return_dict=True)
    h = outputs.last_hidden_state  # (B, 1+N, D)
    print(f"  outputs.last_hidden_state shape={tuple(h.shape)} dtype={h.dtype}")
    print("  meaning: (batch, tokens, hidden_dim)\n")

    # 5) LATENT STATE TOKENS (split CLS vs patch tokens)
    print("[5] LATENT TOKENS\n  split tokens into CLS and patches")
    z_cls = h[:, 0, :]      # (B, D)
    z_patches = h[:, 1:, :] # (B, N, D)
    print(f"  z_cls shape={tuple(z_cls.shape)}   (global token)")
    print(f"  z_patches shape={tuple(z_patches.shape)} (patch tokens)")
    B, N, D = z_patches.shape
    print(f"  inferred: B={B}, N={N} patches, D={D} dims\n")

    # 6) GLOBAL READOUTS (two ways to summarize)
    print("[6] GLOBAL READOUTS\n  make two global vectors")
    g_cls = z_cls                              # (B, D)
    g_mean = z_patches.mean(dim=1)             # (B, D)
    print(f"  g_cls shape={tuple(g_cls.shape)}   (CLS global)")
    print(f"  g_mean shape={tuple(g_mean.shape)} (mean of patches global)\n")

    # 7) Compare these two globals
    print("[7] GLOBAL COMPARISON\n  cosine(g_cls, g_mean)")
    sim_cls_mean = float(
        cosine_similarity(l2_normalize(g_cls), l2_normalize(g_mean)).item()
    )
    print(f"  cosine(CLS_global, mean_patch_global) = {sim_cls_mean:.4f}\n")

    # 8) Patch-to-CLS alignment (analysis probe)
    print("[8] PATCH→CLS ALIGNMENT (PROBE)\n  score each patch against CLS")
    patches = l2_normalize(z_patches.squeeze(0))   # (N, D) (assuming batch=1)
    cls_vec = l2_normalize(g_cls.squeeze(0))       # (D,)
    patch_sims = cosine_similarity(patches, cls_vec.unsqueeze(0), dim=-1)  # (N,)
    print(f"  patch_sims shape={tuple(patch_sims.shape)} (one score per patch)")

    # Show stats
    ps = patch_sims.detach().cpu()
    print(f"  stats: min={float(ps.min()):.4f} mean={float(ps.mean()):.4f} "
          f"median={float(ps.median()):.4f} max={float(ps.max()):.4f} std={float(ps.std()):.4f}")

    # Show top-k aligned patches
    k = min(args.topk, ps.numel())
    vals, idxs = torch.topk(ps, k=k)
    print(f"\n  top-{k} patches aligned with CLS (proxy):")
    for rank, (i, v) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
        print(f"   {rank:>2}. patch#{i:<4} sim={v:.4f}")

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
