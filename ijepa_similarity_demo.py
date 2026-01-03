import os
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoProcessor

# -----------------------------
# Device selection (GPU → CPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️  Using CPU")

# -----------------------------
# Load model + processor
# -----------------------------
model_id = "facebook/ijepa_vith14_1k"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModel.from_pretrained(model_id).to(device).eval()

def embed_image(pil_img: Image.Image) -> torch.Tensor:
    """Return a (1, D) embedding tensor."""
    pil_img = pil_img.convert("RGB")
    inputs = processor(pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs)

    # Simple global embedding: mean pool over tokens/patches
    emb = out.last_hidden_state.mean(dim=1)  # (1, D)
    # Normalize so cosine similarity is well-behaved
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(cosine_similarity(a, b).item())

def make_occluded_variant(img: Image.Image) -> Image.Image:
    """
    Create a 'partial view' variant:
    - center crop to 70%
    - then paste a black rectangle over part of it (occlusion)
    """
    w, h = img.size

    # Center crop (70%)
    cw, ch = int(w * 0.70), int(h * 0.70)
    left = (w - cw) // 2
    top = (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch)).copy()

    # Occlude: black rectangle over ~25% area
    ow, oh = cropped.size
    occ_left = int(ow * 0.55)
    occ_top = int(oh * 0.10)
    occ_right = int(ow * 0.95)
    occ_bottom = int(oh * 0.45)

    occluded = cropped.copy()
    black = Image.new("RGB", (occ_right - occ_left, occ_bottom - occ_top), (0, 0, 0))
    occluded.paste(black, (occ_left, occ_top))
    return occluded

# -----------------------------
# Ensure two images exist
# -----------------------------
# image1.jpg should already exist from your earlier wget
img1_path = "image1.jpg"
if not os.path.exists(img1_path):
    raise FileNotFoundError("image1.jpg not found in current directory.")

# Download a second image (different dog)
img2_path = "image2.jpg"
if not os.path.exists(img2_path):
    # Another Wikimedia dog photo (small)
    import urllib.request
    url = "https://upload.wikimedia.org/wikipedia/commons/5/5f/Golden_Retriever_medium-to-light-coat.jpg"
    print("Downloading image2.jpg ...")
    urllib.request.urlretrieve(url, img2_path)

# -----------------------------
# Load images
# -----------------------------
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

img1_partial = make_occluded_variant(img1)

# -----------------------------
# Compute embeddings
# -----------------------------
emb1 = embed_image(img1)
emb2 = embed_image(img2)
emb1_partial = embed_image(img1_partial)

# -----------------------------
# Similarities
# -----------------------------
print("\nSimilarity scores (cosine, higher = more similar):")
print(f"image1 vs image2 (different dog photos):   {cosine(emb1, emb2):.4f}")
print(f"image1 vs image1_partial (crop+occlusion): {cosine(emb1, emb1_partial):.4f}")

print("\nDone.")
