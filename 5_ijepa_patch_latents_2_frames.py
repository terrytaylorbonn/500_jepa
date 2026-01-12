# 5_ijepa_patch_latents_2_frames.py

#!/usr/bin/env python3
"""
ijepa_patch_latents_demo_v2.py

Runtime-only demo for facebook/ijepa_vith14_1k:
- Extract CLS + patch token latents from ONE image (frame t)
- Optionally extract from a SECOND image (frame t+1) and compare temporal stability:
    "CLS changes less than patch tokens"  ⇢  cosine(CLS_t, CLS_t1) > median(cos(patch_i_t, patch_i_t1))

Also prints:
- shapes
- CLS vs mean(patches) similarity
- top-K patches aligned with CLS (proxy "contributors")
- patch↔CLS similarity stats
"""

import argparse
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from PIL import Image, ImageDraw
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor


# -----------------------------
# Conceptual struct for outputs
# -----------------------------
@dataclass
class PatchLatents:
    z_cls: torch.Tensor                 # (1, D)
    z_patches: torch.Tensor             # (1, N, D)
    patch_sims_to_global: torch.Tensor  # (N,)
    global_from_cls: torch.Tensor       # (1, D)
    global_from_mean: torch.Tensor      # (1, D)
    cls_vs_mean_sim: float


def apply_occlusion(
    img: Image.Image,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Return a copy of img with a filled rectangle occlusion."""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, fill=fill)
    return out


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def extract_patch_latents(
    model,
    processor,
    pil_img: Image.Image,
    device: torch.device,
) -> PatchLatents:
    """
    Extract:
      - CLS token latent
      - Patch token latents
      - Two global embeddings: CLS-based and mean-of-patches
      - Patch-to-CLS cosine similarities (which patches align with global)
    """
    # B) [PREPROCESS]
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # C) [ENCODER] — request token-level outputs
    outputs = model(**inputs, output_hidden_states=False, return_dict=True)

    # D) [LATENT STATE] token latents
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        h = outputs.last_hidden_state  # (1, 1+N, D)
    elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        h = outputs.hidden_states[-1]
    else:
        raise RuntimeError(
            "Model output does not contain last_hidden_state or hidden_states."
        )

    z_cls = h[:, 0, :]      # (1, D)
    z_patches = h[:, 1:, :] # (1, N, D)

    # E) [POOLING] two ways to make a global embedding
    global_from_cls = z_cls
    global_from_mean = z_patches.mean(dim=1)

    # Compare the two globals
    cls_vs_mean_sim = float(
        cosine_similarity(
            l2_normalize(global_from_cls), l2_normalize(global_from_mean)
        ).item()
    )

    # [METRIC] per-patch alignment to CLS global summary (proxy)
    z_patches_n = l2_normalize(z_patches.squeeze(0))         # (N, D)
    global_n = l2_normalize(global_from_cls.squeeze(0))      # (D,)
    patch_sims = cosine_similarity(z_patches_n, global_n.unsqueeze(0), dim=-1)  # (N,)

    return PatchLatents(
        z_cls=z_cls,
        z_patches=z_patches,
        patch_sims_to_global=patch_sims.detach().cpu(),
        global_from_cls=global_from_cls,
        global_from_mean=global_from_mean,
        cls_vs_mean_sim=cls_vs_mean_sim,
    )


def infer_patch_grid(num_patches: int) -> Tuple[int, int]:
    """
    Simple inference: if num_patches is a perfect square, assume square grid.
    Otherwise return (num_patches, 1) and do not claim 2D mapping.
    """
    side = int(math.sqrt(num_patches))
    if side * side == num_patches:
        return side, side
    return num_patches, 1


def topk_patch_report(patch_sims: torch.Tensor, k: int = 10) -> str:
    """
    Report top-k patches by similarity to CLS.
    If the patch grid is square, also report (row, col).
    """
    sims = patch_sims
    n = sims.numel()
    h, w = infer_patch_grid(n)

    vals, idxs = torch.topk(sims, k=min(k, n))
    lines = []
    for rank, (v, i) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        if w > 1:
            r, c = divmod(i, w)
            lines.append(f"{rank:>2}. patch#{i:<4} (r={r:>2}, c={c:>2})  sim={v:.4f}")
        else:
            lines.append(f"{rank:>2}. patch#{i:<4}               sim={v:.4f}")
    return "\n".join(lines)


@torch.no_grad()
def compare_two_frames(lat_a: PatchLatents, lat_b: PatchLatents) -> None:
    """
    Print cosine similarities between:
      - CLS(t) vs CLS(t+1)
      - patch_i(t) vs patch_i(t+1) for all i
    Interprets "CLS changes less than patches" as:
      cosine(CLS_t, CLS_t1) > median(cos(patch_t, patch_t1))
    """
    # CLS similarity (stability of global belief)
    cls_sim = float(
        cosine_similarity(
            l2_normalize(lat_a.z_cls), l2_normalize(lat_b.z_cls)
        ).item()
    )

    # Patch similarities (local evidence stability)
    za = lat_a.z_patches.squeeze(0)  # (N, D)
    zb = lat_b.z_patches.squeeze(0)  # (N, D)

    if za.shape != zb.shape:
        raise ValueError(f"Patch shapes differ: {za.shape} vs {zb.shape}")

    patch_sims = cosine_similarity(l2_normalize(za), l2_normalize(zb), dim=-1)  # (N,)
    patch_sims_cpu = patch_sims.detach().cpu()

    patch_mean = float(patch_sims_cpu.mean())
    patch_median = float(patch_sims_cpu.median())
    patch_min = float(patch_sims_cpu.min())
    patch_max = float(patch_sims_cpu.max())
    patch_std = float(patch_sims_cpu.std())

    print("\n=== Temporal comparison (frame_t vs frame_t+1) ===")
    print(f"Δ_cls   = cosine(CLS_t, CLS_t1)          = {cls_sim:.4f}")
    print("Δ_patch = cosine(patch_i_t, patch_i_t1)  stats:")
    print(f"         min={patch_min:.4f}  mean={patch_mean:.4f}  median={patch_median:.4f}  max={patch_max:.4f}  std={patch_std:.4f}")

    print("\nInterpretation:")
    if cls_sim > patch_median:
        print(f"✅ CLS changes LESS than typical patch tokens (CLS_sim {cls_sim:.4f} > patch_median {patch_median:.4f})")
    else:
        print(f"⚠️  CLS not more stable than typical patch tokens (CLS_sim {cls_sim:.4f} <= patch_median {patch_median:.4f})")

    frac_less = float((patch_sims_cpu < cls_sim).float().mean())
    print(f"Fraction of patches with sim < CLS_sim: {frac_less:.2%}")


def print_single_image_report(lat: PatchLatents, topk: int) -> None:
    """Print the same diagnostics you already used for a single image."""
    print("\n=== Token latent shapes ===")
    print(f"z_cls:     {tuple(lat.z_cls.shape)}")
    print(f"z_patches: {tuple(lat.z_patches.shape)}")

    n_patches = lat.z_patches.shape[1]
    grid_h, grid_w = infer_patch_grid(n_patches)
    print(f"num_patches: {n_patches}   inferred_grid: {grid_h}x{grid_w}")

    print("\n=== Global summary comparison ===")
    print(f"cosine( CLS_global , mean_patch_global ) = {lat.cls_vs_mean_sim:.4f}")

    print("\n=== Top contributing patches (proxy) ===")
    print(topk_patch_report(lat.patch_sims_to_global, k=topk))

    sims = lat.patch_sims_to_global
    print("\n=== Patch↔Global similarity stats ===")
    print(f"min={float(sims.min()):.4f}  mean={float(sims.mean()):.4f} max={float(sims.max()):.4f}  std={float(sims.std()):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image file (frame t)")
    ap.add_argument("--image2", default=None, help="Optional: second image file (frame t+1) for temporal comparison")
    ap.add_argument("--occlude", action="store_true", help="Apply a rectangular occlusion to --image only")
    ap.add_argument("--box", type=int, nargs=4, default=[80, 80, 160, 160],
                    help="Occlusion box: x1 y1 x2 y2 (in original image coords)")
    ap.add_argument("--model_id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # [ENCODER LOAD] runtime-only
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    model = AutoModel.from_pretrained(args.model_id).to(device).eval()

    # Frame t
    img1 = Image.open(args.image).convert("RGB")
    if args.occlude:
        img1 = apply_occlusion(img1, tuple(args.box))
        print(f"Applied occlusion box (on image1): {args.box}")

    lat1 = extract_patch_latents(model, processor, img1, device)

    # Print diagnostics for image1
    print_single_image_report(lat1, topk=args.topk)

    # Optional frame t+1
    if args.image2 is not None:
        img2 = Image.open(args.image2).convert("RGB")
        lat2 = extract_patch_latents(model, processor, img2, device)

        # Temporal comparison: CLS stability vs patch stability
        compare_two_frames(lat1, lat2)


if __name__ == "__main__":
    main()
