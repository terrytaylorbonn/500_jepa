"""
21a4_jepa_occlusion_identity_probe.py
[21a4] JEPA occlusion identity probe

What this adds beyond 21a3b:
  - Identity stress test with baselines:
      For shape A:
        curve(x)  = cos( z_full(A), z_occ(A, x) )
        base(B)   = cos( z_full(A), z_full(B) ) for B != A
  - Plots curve + baseline lines, saves PNG + CSV
  - Optional GIF per shape (same as 21a3b, but now "identity-aware")
  - Optional repeat over multiple random base instances and average curves

Outputs (by default):
  - 21a4_circle_curve.png / .csv (+ optional gif)
  - 21a4_square_curve.png / .csv (+ optional gif)
  - 21a4_triangle_curve.png / .csv (+ optional gif)
"""

import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
SHAPES = ["circle", "square", "triangle"]

# Determinism (same spirit as your ladder)
SEED = 42
RNG = random.Random(SEED)

# Occluder sweep
SWEEP_FRAMES = 40
SWEEP_FPS = 8
OCC_COLOR = (128, 128, 128)
OCC_W_FRAC = 0.25
OCC_H_FRAC = 0.35
OCC_Y_FRAC = 0.30

# How many random base instances per shape to average over
# (1 = fastest; 5-10 = more stable)
NUM_BASE_INSTANCES = 3

# Save GIFs too?
MAKE_GIFS = True

# Output prefix
OUT_PREFIX = "21a4"


# ============================================================
# Shape generation
# ============================================================

def generate_shape_image(shape: str, rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = IMG_SIZE // 6
    x0 = rng.randint(margin, IMG_SIZE // 2)
    y0 = rng.randint(margin, IMG_SIZE // 2)
    x1 = rng.randint(IMG_SIZE // 2, IMG_SIZE - margin)
    y1 = rng.randint(IMG_SIZE // 2, IMG_SIZE - margin)

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


def apply_rect_occluder(img: Image.Image, x0: int, y0: int, w: int, h: int, color=(128, 128, 128)) -> Image.Image:
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
    print(f"[21a4] Device: {DEVICE}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def embed_image(img: Image.Image, processor, model) -> torch.Tensor:
    inputs = processor(img.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    out = model(**inputs)
    return out.last_hidden_state[:, 0, :].squeeze(0)  # (D,)


# ============================================================
# Core sweep primitive
# ============================================================

@dataclass
class SweepResult:
    xs: List[int]
    cos_curve: List[float]
    frames: List[Image.Image]  # occluded frames (for optional GIF)


def occlusion_sweep(base_img: Image.Image, z_full: torch.Tensor, processor, model) -> SweepResult:
    occ_w = int(IMG_SIZE * OCC_W_FRAC)
    occ_h = int(IMG_SIZE * OCC_H_FRAC)
    occ_y = int(IMG_SIZE * OCC_Y_FRAC)

    xs = torch.linspace(-occ_w, IMG_SIZE, SWEEP_FRAMES).to(torch.int32).tolist()

    frames: List[Image.Image] = []
    cos_curve: List[float] = []

    for x0 in xs:
        occ_img = apply_rect_occluder(base_img, int(x0), occ_y, occ_w, occ_h, color=OCC_COLOR)
        z_occ = embed_image(occ_img, processor, model)
        cos = cosine_similarity(z_full.unsqueeze(0), z_occ.unsqueeze(0)).item()
        frames.append(occ_img)
        cos_curve.append(cos)

    return SweepResult(xs=[int(x) for x in xs], cos_curve=cos_curve, frames=frames)


# ============================================================
# Plot / save utilities
# ============================================================

def save_curve_csv(path: str, xs: List[int], cos_curve: List[float], baselines: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # header includes baselines for convenience
        w.writerow(["x_pos", "cos_full_vs_occ", "baseline_name", "baseline_value"])
        # write curve rows (baseline columns left blank)
        for x, c in zip(xs, cos_curve):
            w.writerow([x, c, "", ""])
        # append baselines
        for k, v in baselines.items():
            w.writerow(["", "", k, v])


def plot_curve(path_png: str, xs: List[int], cos_curve: List[float], title: str, baselines: Dict[str, float]) -> None:
    if plt is None:
        print("[21a4] matplotlib not installed; skipping plot.")
        return

    plt.figure()
    plt.plot(xs, cos_curve, label="cos(full, occluded(x))")
    for name, val in baselines.items():
        plt.axhline(val, linestyle="--", label=f"baseline: {name} = {val:.4f}")

    plt.xlabel("Occluder X position")
    plt.ylabel("Cosine similarity")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path_png, dpi=130)
    plt.close()


def save_gif(path_gif: str, frames: List[Image.Image], cos_curve: List[float], title_prefix: str) -> None:
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("[21a4] matplotlib animation not available; skipping GIF.")
        return

    fig, ax = plt.subplots()
    ax.set_axis_off()

    im = ax.imshow(frames[0])
    t = ax.set_title(f"{title_prefix} | cos={cos_curve[0]:.4f} (1/{len(frames)})")

    def update(i):
        im.set_data(frames[i])
        t.set_text(f"{title_prefix} | cos={cos_curve[i]:.4f} ({i+1}/{len(frames)})")
        return [im, t]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / SWEEP_FPS), blit=False)
    ani.save(path_gif, writer=PillowWriter(fps=SWEEP_FPS))
    plt.close(fig)


# ============================================================
# Identity probe
# ============================================================

def identity_probe_one_instance(shape_a: str, rng: random.Random, processor, model) -> Tuple[SweepResult, Dict[str, float]]:
    """
    Create one random base instance for shape_a, and one random full instance
    for each other shape, compute baselines, then run occlusion sweep.
    """
    base_a = generate_shape_image(shape_a, rng)
    z_a = embed_image(base_a, processor, model)

    baselines: Dict[str, float] = {}
    for shape_b in SHAPES:
        if shape_b == shape_a:
            continue
        img_b = generate_shape_image(shape_b, rng)
        z_b = embed_image(img_b, processor, model)
        baselines[f"{shape_a}_vs_{shape_b}"] = cosine_similarity(z_a.unsqueeze(0), z_b.unsqueeze(0)).item()

    sweep = occlusion_sweep(base_a, z_a, processor, model)
    return sweep, baselines


def average_curves(curves: List[List[float]]) -> List[float]:
    t = torch.tensor(curves, dtype=torch.float32)  # (K, T)
    return t.mean(dim=0).tolist()


def main():
    processor, model = load_jepa()

    print(f"[21a4] Running identity probe with NUM_BASE_INSTANCES={NUM_BASE_INSTANCES}, SWEEP_FRAMES={SWEEP_FRAMES}")
    print(f"[21a4] Outputs prefix: {OUT_PREFIX}_*.png/.csv (+ optional .gif)\n")

    # Use a deterministic RNG stream but avoid coupling between shapes too much
    rng_master = random.Random(SEED)

    for shape_a in SHAPES:
        print(f"======================================================")
        print(f"[21a4] Shape A = {shape_a}")
        print(f"======================================================")

        instance_curves: List[List[float]] = []
        instance_baselines: List[Dict[str, float]] = []
        first_frames: List[Image.Image] = []

        for k in range(NUM_BASE_INSTANCES):
            # Create a per-instance rng derived from master for repeatability
            seed_k = rng_master.randint(0, 10_000_000)
            rng_k = random.Random(seed_k)

            sweep, baselines = identity_probe_one_instance(shape_a, rng_k, processor, model)

            instance_curves.append(sweep.cos_curve)
            instance_baselines.append(baselines)

            # Keep frames from the first instance for GIF (optional)
            if k == 0:
                first_frames = sweep.frames

            # Print a compact summary
            btxt = "  ".join([f"{n}={v:.4f}" for n, v in baselines.items()])
            print(f"  instance {k+1}/{NUM_BASE_INSTANCES}  curve_mean={sum(sweep.cos_curve)/len(sweep.cos_curve):.4f}  baselines: {btxt}")

        # Average curve over instances
        xs = sweep.xs  # same for all instances
        mean_curve = average_curves(instance_curves)

        # Average baselines over instances
        baseline_keys = list(instance_baselines[0].keys())
        mean_baselines: Dict[str, float] = {}
        for key in baseline_keys:
            vals = [b[key] for b in instance_baselines]
            mean_baselines[key] = float(torch.tensor(vals).mean().item())

        # Save outputs
        out_png = f"{OUT_PREFIX}_{shape_a}_curve.png"
        out_csv = f"{OUT_PREFIX}_{shape_a}_curve.csv"
        out_gif = f"{OUT_PREFIX}_{shape_a}_sliding.gif"

        plot_curve(
            out_png,
            xs,
            mean_curve,
            title=f"21a4 Identity probe: {shape_a} | mean over {NUM_BASE_INSTANCES} instances",
            baselines=mean_baselines,
        )
        save_curve_csv(out_csv, xs, mean_curve, mean_baselines)

        print(f"[21a4] Saved: {out_png}")
        print(f"[21a4] Saved: {out_csv}")

        if MAKE_GIFS:
            save_gif(out_gif, first_frames, instance_curves[0], title_prefix=f"{shape_a} occlusion sweep")
            print(f"[21a4] Saved: {out_gif}")

        # Quick interpretation print
        # (compare curve min to baselines)
        curve_min = min(mean_curve)
        curve_mean = sum(mean_curve) / len(mean_curve)
        best_baseline = max(mean_baselines.values()) if mean_baselines else float("nan")

        print(f"[21a4] curve_mean={curve_mean:.4f}  curve_min={curve_min:.4f}  best_diffshape_baseline(max)={best_baseline:.4f}")
        if curve_min > best_baseline:
            print("[21a4] ✅ Occluded(A) stays above diff-shape baselines → identity looks preserved under this occluder (for this probe).")
        else:
            print("[21a4] ⚠️ Occluded(A) overlaps/crosses diff-shape baselines → CLS is not strongly shape-identifying here (for this probe).")

        print()

    print("[21a4] Done.")


if __name__ == "__main__":
    main()
