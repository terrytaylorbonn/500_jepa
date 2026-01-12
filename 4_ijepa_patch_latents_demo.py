# 4_ijepa_patch_latents_demo.py

import argparse
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
      - Patch-to-global cosine similarities (which patches align with global)
    """
    # B) [PREPROCESS]
    inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # C) [ENCODER] — request token-level outputs
    # Many ViT-like models expose last_hidden_state (B, seq, D)
    outputs = model(**inputs, output_hidden_states=False, return_dict=True)

    if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
        raise RuntimeError(
            "Model output does not contain last_hidden_state. "
            "Try output_hidden_states=True and use hidden_states[-1]."
        )

    # D) [LATENT STATE] token latents
    h = outputs.last_hidden_state               # (1, 1+N, D)
    z_cls = h[:, 0, :]                          # (1, D)
    z_patches = h[:, 1:, :]                     # (1, N, D)

    # E) [POOLING] two ways to make a global embedding
    global_from_cls = z_cls                     # (1, D)
    global_from_mean = z_patches.mean(dim=1)    # (1, D)

    # Compare the two globals (are they “similar summaries”?)
    cls_vs_mean_sim = float(
        cosine_similarity(l2_normalize(global_from_cls), l2_normalize(global_from_mean)).item()
    )

    # F) [METRIC] per-patch alignment to the chosen global summary
    # (here we pick CLS as the global reference)
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
    Many ViT configs produce square patch grids (e.g., 16x16 = 256).
    We'll infer a square if possible.
    """
    side = int(math.sqrt(num_patches))
    if side * side == num_patches:
        return side, side
    # fallback: return (num_patches, 1) if non-square
    return num_patches, 1


def topk_patch_report(patch_sims: torch.Tensor, k: int = 10) -> str:
    """
    Report top-k patches by similarity to global embedding.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an image file")
    ap.add_argument("--occlude", action="store_true", help="Apply a rectangular occlusion")
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

    # A) [INPUT]
    img = Image.open(args.image).convert("RGB")

    # optional occlusion
    if args.occlude:
        img = apply_occlusion(img, tuple(args.box))
        print(f"Applied occlusion box: {args.box}")

    # Extract patch-level latents
    lat = extract_patch_latents(model, processor, img, device)

    # Print shapes
    print("\n=== Token latent shapes ===")
    print(f"z_cls:     {tuple(lat.z_cls.shape)}")
    print(f"z_patches: {tuple(lat.z_patches.shape)}")

    n_patches = lat.z_patches.shape[1]
    grid_h, grid_w = infer_patch_grid(n_patches)
    print(f"num_patches: {n_patches}   inferred_grid: {grid_h}x{grid_w}")

    # Compare CLS global vs mean-pooled global
    print("\n=== Global summary comparison ===")
    print(f"cosine( CLS_global , mean_patch_global ) = {lat.cls_vs_mean_sim:.4f}")

    # Patch contribution proxy: patch ↔ global similarity
    print("\n=== Top contributing patches (proxy) ===")
    print(topk_patch_report(lat.patch_sims_to_global, k=args.topk))

    # Extra: simple stats for intuition
    sims = lat.patch_sims_to_global
    print("\n=== Patch↔Global similarity stats ===")
    print(f"min={float(sims.min()):.4f}  mean={float(sims.mean()):.4f} max={float(sims.max()):.4f}  std={float(sims.std()):.4f}")


if __name__ == "__main__":
    main()
