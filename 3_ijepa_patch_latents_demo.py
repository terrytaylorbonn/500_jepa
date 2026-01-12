# 3_ijepa_patch_latents_demo.py

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device, torch.cuda.get_device_name(0) if device.type=="cuda" else "")

model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModel.from_pretrained(model_id).to(device).eval()

def load_img(path):
    return Image.open(path).convert("RGB")

def occlude_and_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * 0.70), int(h * 0.70)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch)).copy()

    ow, oh = cropped.size
    occ = Image.new("RGB", (int(ow*0.40), int(oh*0.35)), (0, 0, 0))
    cropped.paste(occ, (int(ow*0.55), int(oh*0.10)))
    return cropped

@torch.no_grad()
def patch_latents(pil_img: Image.Image) -> torch.Tensor:
    """
    Returns patch latents (no pooling).
    Shape: (N, D) where N ~ num_patches (+ maybe CLS).
    """
    inputs = processor(pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs).last_hidden_state  # (1, N, D)
    out = out.squeeze(0)                     # (N, D)
    out = F.normalize(out, dim=-1)
    return out

def patch_cosine_matrix(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between every token in A and every token in B.
    A: (NA, D), B: (NB, D) → (NA, NB)
    """
    return A @ B.T  # because rows are normalized

def summarize_top_matches(sim: torch.Tensor, k=10, label=""):
    # sim: (NA, NB)
    vals, idx = torch.topk(sim.flatten(), k)
    NA, NB = sim.shape
    print(f"\nTop {k} patch-to-patch similarities {label}:")
    for rank in range(k):
        flat_i = idx[rank].item()
        a_i = flat_i // NB
        b_i = flat_i % NB
        print(f"  #{rank+1:02d}  cos={vals[rank].item():.4f}   A_patch={a_i}  B_patch={b_i}")

# ---- Load images ----
img1 = load_img("image1.jpg")
img2 = load_img("image2.jpg")
img1_partial = occlude_and_crop(img1)

# ---- Compute patch latents ----
A = patch_latents(img1)
B = patch_latents(img2)
P = patch_latents(img1_partial)

# ---- Similarity matrices ----
sim_1_vs_2 = patch_cosine_matrix(A, B)
sim_1_vs_partial = patch_cosine_matrix(A, P)

summarize_top_matches(sim_1_vs_2, k=10, label="(image1 vs image2)")
summarize_top_matches(sim_1_vs_partial, k=10, label="(image1 vs image1_partial)")

print("\nDone.")
