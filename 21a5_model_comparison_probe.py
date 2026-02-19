"""
21a5_model_comparison_probe.py
[21a5] Model comparison probe: occlusion identity curves across encoders

What it does (per model, per shape A):
  - Sample NUM_BASE_INSTANCES random base images of shape A
  - For each instance:
      curve(x) = cos( z_full(A), z_occ(A, x) )   (sliding occluder)
      baselines: cos( z_full(A), z_full(B) ) for B != A
  - Average curve & baselines across instances
  - Save:
      PNG: curve + baseline lines
      CSV: curve values + baselines
      GIF: (optional) first instance sliding occluder with cosine overlay

Supported encoders (default):
  - I-JEPA: facebook/ijepa_vith14_1k
  - DINOv2: facebook/dinov2-base  (you can swap to small/large/giant)
  - CLIP: openai/clip-vit-base-patch32

Notes:
  - Repo "staleness" isn't relevant here: we compare representation behavior empirically.
  - This is representation-level identity persistence, NOT belief-state object permanence.
"""

import os
import csv
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
)

# plotting / GIF
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
except ImportError:
    plt = None
    FuncAnimation = None
    PillowWriter = None


# ============================================================
# Config / Defaults
# ============================================================

IMG_SIZE = 224
SHAPES = ["circle", "square", "triangle"]

DEFAULT_SEED = 42

# Sliding occluder sweep
DEFAULT_SWEEP_FRAMES = 40
DEFAULT_SWEEP_FPS = 8
OCC_COLOR = (128, 128, 128)
OCC_W_FRAC = 0.25
OCC_H_FRAC = 0.35
OCC_Y_FRAC = 0.30

# Averaging
DEFAULT_NUM_BASE_INSTANCES = 3

# Output
DEFAULT_OUT_DIR = "out_21a5"

# Models to compare
# Feel free to add more model IDs here later.
MODEL_SPECS = {
    "ijepa": {
        "model_id": "facebook/ijepa_vith14_1k",
        "kind": "vit_cls",  # use last_hidden_state CLS
    },
    "dinov2": {
        "model_id": "facebook/dinov2-base",
        "kind": "vit_cls",  # use last_hidden_state CLS
    },
    "clip": {
        "model_id": "openai/clip-vit-base-patch32",
        "kind": "clip_image_embeds",  # use outputs.image_embeds
    },
}


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
# Encoder wrapper
# ============================================================

class Embedder:
    def __init__(self, name: str, model_id: str, kind: str, device: torch.device):
        self.name = name
        self.model_id = model_id
        self.kind = kind
        self.device = device

        print(f"[21a5] Loading model '{name}' ({model_id}) on {device}...")

        # Processor
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

        # Model
        if kind == "clip_image_embeds":
            self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        else:
            self.model = AutoModel.from_pretrained(model_id).to(device).eval()

        print(f"[21a5] Loaded '{name}' ✅")

    @torch.no_grad()
    def embed(self, img: Image.Image) -> torch.Tensor:
        """
        Returns a 1D embedding tensor (D,) on self.device.
        Normalization is applied to make cosine comparisons stable across model scales.
        """
        inputs = self.processor(img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)

        if self.kind == "vit_cls":
            # ViT-like models: use CLS from last_hidden_state
            z = out.last_hidden_state[:, 0, :].squeeze(0)
        elif self.kind == "clip_image_embeds":
            # CLIP: use image_embeds (already pooled + projected)
            z = out.image_embeds.squeeze(0)
        else:
            raise ValueError(f"Unknown embed kind: {self.kind}")

        # Normalize so cosine is meaningful and stable
        z = z / (z.norm() + 1e-8)
        return z


# ============================================================
# Sweep + probe
# ============================================================

@dataclass
class SweepResult:
    xs: List[int]
    cos_curve: List[float]
    frames: List[Image.Image]


def occlusion_sweep(base_img: Image.Image, z_full: torch.Tensor, embedder: Embedder, sweep_frames: int) -> SweepResult:
    occ_w = int(IMG_SIZE * OCC_W_FRAC)
    occ_h = int(IMG_SIZE * OCC_H_FRAC)
    occ_y = int(IMG_SIZE * OCC_Y_FRAC)

    xs = torch.linspace(-occ_w, IMG_SIZE, sweep_frames).to(torch.int32).tolist()

    frames: List[Image.Image] = []
    cos_curve: List[float] = []

    for x0 in xs:
        occ_img = apply_rect_occluder(base_img, int(x0), occ_y, occ_w, occ_h, color=OCC_COLOR)
        z_occ = embedder.embed(occ_img)
        cos = cosine_similarity(z_full.unsqueeze(0), z_occ.unsqueeze(0)).item()
        frames.append(occ_img)
        cos_curve.append(cos)

    return SweepResult(xs=[int(x) for x in xs], cos_curve=cos_curve, frames=frames)


def identity_probe_one_instance(shape_a: str, rng: random.Random, embedder: Embedder, sweep_frames: int) -> Tuple[SweepResult, Dict[str, float]]:
    base_a = generate_shape_image(shape_a, rng)
    z_a = embedder.embed(base_a)

    baselines: Dict[str, float] = {}
    for shape_b in SHAPES:
        if shape_b == shape_a:
            continue
        img_b = generate_shape_image(shape_b, rng)
        z_b = embedder.embed(img_b)
        baselines[f"{shape_a}_vs_{shape_b}"] = cosine_similarity(z_a.unsqueeze(0), z_b.unsqueeze(0)).item()

    sweep = occlusion_sweep(base_a, z_a, embedder, sweep_frames=sweep_frames)
    return sweep, baselines


def average_curves(curves: List[List[float]]) -> List[float]:
    t = torch.tensor(curves, dtype=torch.float32)
    return t.mean(dim=0).tolist()


# ============================================================
# Output helpers
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_curve_csv(path: str, xs: List[int], cos_curve: List[float], baselines: Dict[str, float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x_pos", "cos_full_vs_occ"])
        for x, c in zip(xs, cos_curve):
            w.writerow([x, c])
        w.writerow([])
        w.writerow(["baseline_name", "baseline_value"])
        for k, v in baselines.items():
            w.writerow([k, v])


def plot_curve(path_png: str, xs: List[int], cos_curve: List[float], title: str, baselines: Dict[str, float]) -> None:
    if plt is None:
        print("[21a5] matplotlib not installed; skipping plot.")
        return

    plt.figure()
    plt.plot(xs, cos_curve, label="cos(full, occluded(x))")
    for name, val in baselines.items():
        plt.axhline(val, linestyle="--", label=f"{name} = {val:.4f}")
    plt.xlabel("Occluder X position")
    plt.ylabel("Cosine similarity")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(path_png, dpi=130)
    plt.close()


def save_gif(path_gif: str, frames: List[Image.Image], cos_curve: List[float], title_prefix: str, fps: int) -> None:
    if plt is None or FuncAnimation is None or PillowWriter is None:
        print("[21a5] matplotlib animation not available; skipping GIF.")
        return

    fig, ax = plt.subplots()
    ax.set_axis_off()

    im = ax.imshow(frames[0])
    t = ax.set_title(f"{title_prefix} | cos={cos_curve[0]:.4f} (1/{len(frames)})")

    def update(i):
        im.set_data(frames[i])
        t.set_text(f"{title_prefix} | cos={cos_curve[i]:.4f} ({i+1}/{len(frames)})")
        return [im, t]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=False)
    ani.save(path_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(os.environ.get("SEED", DEFAULT_SEED))
    rng_master = random.Random(seed)

    out_dir = os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)
    ensure_dir(out_dir)

    # Which models to run: set env MODELS="ijepa,dinov2,clip" to control
    models_env = os.environ.get("MODELS", "ijepa,dinov2,clip")
    model_names = [m.strip() for m in models_env.split(",") if m.strip()]

    num_instances = int(os.environ.get("NUM_BASE_INSTANCES", DEFAULT_NUM_BASE_INSTANCES))
    sweep_frames = int(os.environ.get("SWEEP_FRAMES", DEFAULT_SWEEP_FRAMES))
    sweep_fps = int(os.environ.get("SWEEP_FPS", DEFAULT_SWEEP_FPS))
    make_gifs = os.environ.get("MAKE_GIFS", "1") != "0"

    print(f"[21a5] Device: {device}")
    print(f"[21a5] MODELS={model_names}")
    print(f"[21a5] NUM_BASE_INSTANCES={num_instances}  SWEEP_FRAMES={sweep_frames}  MAKE_GIFS={make_gifs}")
    print(f"[21a5] Output dir: {out_dir}\n")

    # Load embedders
    embedders: List[Embedder] = []
    for name in model_names:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model name '{name}'. Options: {list(MODEL_SPECS.keys())}")
        spec = MODEL_SPECS[name]
        embedders.append(Embedder(name=name, model_id=spec["model_id"], kind=spec["kind"], device=device))

    # Run probe
    for embedder in embedders:
        print("\n" + "=" * 70)
        print(f"[21a5] Encoder = {embedder.name}  ({embedder.model_id})")
        print("=" * 70)

        for shape_a in SHAPES:
            print(f"\n[21a5] Shape A = {shape_a}")

            instance_curves: List[List[float]] = []
            instance_baselines: List[Dict[str, float]] = []
            first_frames: Optional[List[Image.Image]] = None
            xs_ref: Optional[List[int]] = None

            for k in range(num_instances):
                # per-instance deterministic seed stream
                seed_k = rng_master.randint(0, 10_000_000)
                rng_k = random.Random(seed_k)

                sweep, baselines = identity_probe_one_instance(shape_a, rng_k, embedder, sweep_frames=sweep_frames)

                if xs_ref is None:
                    xs_ref = sweep.xs
                instance_curves.append(sweep.cos_curve)
                instance_baselines.append(baselines)

                if k == 0:
                    first_frames = sweep.frames

                btxt = "  ".join([f"{n}={v:.4f}" for n, v in baselines.items()])
                print(f"  instance {k+1}/{num_instances}  curve_mean={sum(sweep.cos_curve)/len(sweep.cos_curve):.4f}  baselines: {btxt}")

            assert xs_ref is not None
            mean_curve = average_curves(instance_curves)

            # Average baselines over instances
            baseline_keys = list(instance_baselines[0].keys())
            mean_baselines: Dict[str, float] = {}
            for key in baseline_keys:
                vals = [b[key] for b in instance_baselines]
                mean_baselines[key] = float(torch.tensor(vals).mean().item())

            # Save
            tag = f"{embedder.name}_{shape_a}"
            out_png = os.path.join(out_dir, f"{tag}_curve.png")
            out_csv = os.path.join(out_dir, f"{tag}_curve.csv")
            out_gif = os.path.join(out_dir, f"{tag}_sliding.gif")

            plot_curve(
                out_png,
                xs_ref,
                mean_curve,
                title=f"21a5 {embedder.name}: identity probe ({shape_a}) | mean over {num_instances}",
                baselines=mean_baselines,
            )
            save_curve_csv(out_csv, xs_ref, mean_curve, mean_baselines)

            print(f"[21a5] Saved: {out_png}")
            print(f"[21a5] Saved: {out_csv}")

            if make_gifs and first_frames is not None:
                # Use first instance curve for GIF overlay (nice for intuition)
                save_gif(
                    out_gif,
                    first_frames,
                    instance_curves[0],
                    title_prefix=f"{embedder.name}:{shape_a}",
                    fps=sweep_fps,
                )
                print(f"[21a5] Saved: {out_gif}")

            # Quick interpretation
            curve_min = min(mean_curve)
            best_baseline = max(mean_baselines.values()) if mean_baselines else float("nan")
            margin = curve_min - best_baseline

            print(f"[21a5] curve_min={curve_min:.4f}  best_diffshape_baseline(max)={best_baseline:.4f}  margin={margin:+.4f}")
            if margin > 0:
                print("[21a5] ✅ Occluded(A) stays above diff-shape baselines → identity looks preserved (for this probe).")
            else:
                print("[21a5] ⚠️ Overlaps/crosses baselines → weak identity separation (for this probe).")

    print("\n[21a5] Done.")


if __name__ == "__main__":
    main()
