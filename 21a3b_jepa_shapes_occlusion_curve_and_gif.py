"""
21a3b_jepa_shapes_occlusion_curve_and_gif.py
[21a3b] JEPA shapes occlusion:
    - 21a2 stats
    - 21a3 sliding occluder GIF
    - NEW: cosine vs occluder-position curve + CSV
    - Introduces reusable occlusion_sweep() for 21a4
"""

import csv
import random
from collections import defaultdict

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# plotting / GIF
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError:
    plt = None
    FuncAnimation = None
    PillowWriter = None


# ============================================================
# Config
# ============================================================

MODEL_ID = "facebook/ijepa_vith14_1k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
N_SAMPLES_PER_SHAPE = 64
OCCLUSION_FRAC = 0.4
SHAPES = ["circle", "square", "triangle"]
RNG = random.Random(42)

# Sliding occluder demo
MAKE_GIF = True
GIF_SHAPE = "circle"
GIF_OUT = "21a3b_sliding_occluder.gif"
GIF_FRAMES = 40
GIF_FPS = 8

# Curve outputs
CURVE_PNG = "21a3b_cosine_curve.png"
CURVE_CSV = "21a3b_cosine_curve.csv"


# ============================================================
# Shape generation
# ============================================================

def generate_shape_image(shape: str) -> Image.Image:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = IMG_SIZE // 6
    x0 = RNG.randint(margin, IMG_SIZE // 2)
    y0 = RNG.randint(margin, IMG_SIZE // 2)
    x1 = RNG.randint(IMG_SIZE // 2, IMG_SIZE - margin)
    y1 = RNG.randint(IMG_SIZE // 2, IMG_SIZE - margin)

    if shape == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=(255, 255, 255))
    elif shape == "square":
        side = min(x1 - x0, y1 - y0)
        draw.rectangle([x0, y0, x0 + side, y0 + side], fill=(255, 255, 255))
    elif shape == "triangle":
        x_mid = (x0 + x1) // 2
        draw.polygon([(x_mid, y0), (x0, y1), (x1, y1)], fill=(255, 255, 255))
    else:
        raise ValueError(shape)

    return img


def apply_sliding_occluder(img, x0, y0, w, h, color=(128, 128, 128)):
    img_occ = img.copy()
    draw = ImageDraw.Draw(img_occ)

    x1, y1 = x0 + w, y0 + h
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(IMG_SIZE, x1), min(IMG_SIZE, y1)

    if x1c > x0c and y1c > y0c:
        draw.rectangle([x0c, y0c, x1c, y1c], fill=color)

    return img_occ


# ============================================================
# JEPA helpers
# ============================================================

def load_jepa():
    print("[21a3b] Device:", DEVICE)
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def embed_image(img, processor, model):
    inputs = processor(img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze(0)


# ============================================================
# CORE: occlusion sweep (reused in 21a4)
# ============================================================

def occlusion_sweep(base_img, processor, model):
    z_full = embed_image(base_img, processor, model)

    occ_w = int(IMG_SIZE * 0.25)
    occ_h = int(IMG_SIZE * 0.35)
    occ_y = int(IMG_SIZE * 0.30)

    xs = torch.linspace(-occ_w, IMG_SIZE, GIF_FRAMES).to(torch.int32).tolist()

    frames, cos_vals = [], []

    print("\n[21a3b] Sliding occluder — live cosine:")
    for i, x0 in enumerate(xs):
        occ_img = apply_sliding_occluder(base_img, int(x0), occ_y, occ_w, occ_h)
        z_occ = embed_image(occ_img, processor, model)

        cos = cosine_similarity(z_full.unsqueeze(0), z_occ.unsqueeze(0)).item()
        print(f"  frame {i+1:02d}/{GIF_FRAMES}  x0={int(x0):4d}  cos={cos:.4f}")

        frames.append(occ_img)
        cos_vals.append(cos)

    return xs, frames, cos_vals


# ============================================================
# GIF + curve
# ============================================================

def make_gif_and_curve(xs, frames, cos_vals):

    # ---- GIF ----
    if MAKE_GIF and plt:
        fig, ax = plt.subplots()
        ax.set_axis_off()

        im = ax.imshow(frames[0])
        title = ax.set_title(f"cos={cos_vals[0]:.4f}")

        def update(i):
            im.set_data(frames[i])
            title.set_text(f"cos={cos_vals[i]:.4f}  frame {i+1}/{len(frames)}")
            return [im, title]

        ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000/GIF_FPS))
        ani.save(GIF_OUT, writer=PillowWriter(fps=GIF_FPS))
        plt.close(fig)

        print("[21a3b] Saved GIF:", GIF_OUT)

    # ---- Curve ----
    if plt:
        plt.figure()
        plt.plot(xs, cos_vals)
        plt.xlabel("Occluder X position")
        plt.ylabel("cos(z_full, z_occ)")
        plt.title("JEPA Occlusion Sensitivity")
        plt.grid(True)
        plt.savefig(CURVE_PNG, dpi=120)
        plt.close()
        print("[21a3b] Saved curve:", CURVE_PNG)

    # ---- CSV ----
    with open(CURVE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_pos", "cosine"])
        for x, c in zip(xs, cos_vals):
            writer.writerow([x, c])

    print("[21a3b] Saved CSV:", CURVE_CSV)


# ============================================================
# MAIN
# ============================================================

def main():
    processor, model = load_jepa()

    # --- Sliding occluder demo ---
    base_img = generate_shape_image(GIF_SHAPE)
    xs, frames, cos_vals = occlusion_sweep(base_img, processor, model)
    make_gif_and_curve(xs, frames, cos_vals)

    # --- Original 21a2 statistics ---
    same_shape_cos = defaultdict(list)
    diff_shape_cos = defaultdict(list)

    print("\n[21a3b] Running distribution probe...")

    for _ in range(N_SAMPLES_PER_SHAPE):
        full_embeddings = {}

        for shape in SHAPES:
            img = generate_shape_image(shape)
            img_occ = apply_sliding_occluder(img, 40, 40, 60, 60)

            z_full = embed_image(img, processor, model)
            z_occ = embed_image(img_occ, processor, model)

            full_embeddings[shape] = z_full

            cos = cosine_similarity(z_full.unsqueeze(0), z_occ.unsqueeze(0)).item()
            same_shape_cos[shape].append(cos)

        for a in SHAPES:
            for b in SHAPES:
                if a >= b:
                    continue
                cos = cosine_similarity(
                    full_embeddings[a].unsqueeze(0),
                    full_embeddings[b].unsqueeze(0)
                ).item()
                diff_shape_cos[f"{a}_vs_{b}"].append(cos)

    print("\n[21a3b] SAME shape stats:")
    for k, v in same_shape_cos.items():
        t = torch.tensor(v)
        print(f"{k:8s} mean={t.mean():.4f} std={t.std():.4f}")

    print("\n[21a3b] DIFF shape stats:")
    for k, v in diff_shape_cos.items():
        t = torch.tensor(v)
        print(f"{k:12s} mean={t.mean():.4f} std={t.std():.4f}")


if __name__ == "__main__":
    main()
