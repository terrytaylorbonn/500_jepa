"""
22_d6_temporal_imagination_demo.py
[22_d6] Temporal imagination in JEPA latent space (skeleton)

Idea:
  - Create simple synthetic sequences of a moving shape:
        x0 -> x1 -> x2 -> x3
  - Use frozen I-JEPA encoder to get CLS embeddings:
        z0, z1, z2, z3
  - Train a small MLP predictor:
        g([z0, z1, z2]) -> ẑ3
  - Evaluate:
        cos(ẑ3, z3) and compare vs cos(z2, z3) etc.

Notes:
  - This is NOT JEPA "predicting" the future;
    it's a tiny learned predictor operating IN JEPA's latent space.
"""

import math
import random
from typing import List

import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "facebook/ijepa_vith14_1k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
SEQ_LEN = 4                     # x0, x1, x2, x3
N_TRAIN_SEQS = 256              # toy-size; bump if you want
N_TEST_SEQS = 32
BATCH_SIZE = 16
EPOCHS = 5                      # skeleton: small; adjust as needed
LR = 1e-3
RNG = random.Random(123)

# -----------------------------
# Synthetic sequence generator
# -----------------------------

def generate_moving_square_sequence(seq_len: int = 4) -> List[Image.Image]:
    """
    Very simple "world":
      - Black background.
      - One white square moving horizontally over time.

    Returns:
      list of PIL.Images [x0, x1, x2, x3]
    """
    imgs = []
    img_w, img_h = IMG_SIZE, IMG_SIZE

    # Choose a random vertical band and size
    square_size = RNG.randint(img_w // 8, img_w // 5)
    y_top = RNG.randint(img_h // 4, img_h // 2)
    y_bottom = y_top + square_size

    # Horizontal motion: start and end positions
    margin = square_size
    x_start = RNG.randint(margin, img_w // 3)
    x_end = RNG.randint(2 * img_w // 3, img_w - margin)

    for t in range(seq_len):
        alpha = t / (seq_len - 1) if seq_len > 1 else 0.0
        x_center = int((1 - alpha) * x_start + alpha * x_end)
        x_left = x_center - square_size // 2
        x_right = x_center + square_size // 2

        img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([x_left, y_top, x_right, y_bottom], fill=(255, 255, 255))
        imgs.append(img)

    return imgs

# -----------------------------
# JEPA encoder helpers
# -----------------------------

def load_jepa():
    print(f"[22_d6] Device: {DEVICE}")
    print(f"[22_d6] Loading I-JEPA processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    print("[22_d6] Processor loaded.")
    print(f"[22_d6] Loading I-JEPA encoder on {DEVICE}...")
    model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    print("[22_d6] Encoder loaded.")
    return processor, model


@torch.no_grad()
def embed_images(imgs: List[Image.Image], processor, model) -> torch.Tensor:
    """
    Embed a list of PIL images with JEPA and return CLS embeddings.

    Returns:
      z: (T, D) tensor, where T=len(imgs)
    """
    inputs = processor([im.convert("RGB") for im in imgs], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)

    # CLS token: (B, T_tokens, D) -> take index 0 along token dimension
    cls = outputs.last_hidden_state[:, 0, :]     # (T, D)
    return cls


# -----------------------------
# Tiny predictor MLP
# -----------------------------

class TemporalPredictor(nn.Module):
    """
    g([z0, z1, z2]) -> z3
    Simple 2-layer MLP on concatenated latents.
    """

    def __init__(self, dim: int, n_steps: int = 3, hidden_factor: int = 2):
        super().__init__()
        in_dim = dim * n_steps
        hidden_dim = hidden_factor * dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        """
        z_seq: (B, n_steps, D)
        returns: (B, D)
        """
        B, T, D = z_seq.shape
        assert T >= 3, "Need at least 3 steps for [z0, z1, z2]"
        x = z_seq[:, :3, :].reshape(B, 3 * D)
        return self.net(x)


# -----------------------------
# Dataset creation
# -----------------------------

def build_dataset(n_seqs: int, processor, model):
    """
    Generate sequences, encode with JEPA, and create (input, target) pairs.

    Returns:
      inputs:  (N, 3, D)  tensor with [z0, z1, z2]
      targets: (N, D)     tensor with z3
    """
    all_inputs = []
    all_targets = []

    print(f"[22_d6] Building dataset with {n_seqs} sequences...")
    for _ in range(n_seqs):
        imgs = generate_moving_square_sequence(SEQ_LEN)   # [x0..x3]
        z_seq = embed_images(imgs, processor, model)      # (4, D)

        z0, z1, z2, z3 = z_seq[0], z_seq[1], z_seq[2], z_seq[3]
        inp = torch.stack([z0, z1, z2], dim=0)            # (3, D)
        tgt = z3                                          # (D,)

        all_inputs.append(inp)
        all_targets.append(tgt)

    inputs = torch.stack(all_inputs, dim=0)   # (N, 3, D)
    targets = torch.stack(all_targets, dim=0) # (N, D)
    return inputs, targets


# -----------------------------
# Training loop (skeleton)
# -----------------------------

def train_predictor(model_p, train_inputs, train_targets):
    model_p.train()
    optimizer = torch.optim.Adam(model_p.parameters(), lr=LR)

    dataset_size = train_inputs.size(0)
    steps_per_epoch = math.ceil(dataset_size / BATCH_SIZE)

    print(f"[22_d6] Training predictor for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(dataset_size)
        train_inputs_shuf = train_inputs[perm]
        train_targets_shuf = train_targets[perm]

        running_loss = 0.0

        for step in range(steps_per_epoch):
            start = step * BATCH_SIZE
            end = min(start + BATCH_SIZE, dataset_size)

            batch_inp = train_inputs_shuf[start:end].to(DEVICE)   # (B, 3, D)
            batch_tgt = train_targets_shuf[start:end].to(DEVICE)  # (B, D)

            optimizer.zero_grad()

            pred = model_p(batch_inp)               # (B, D)
            # cosine loss = 1 - cos
            cos = cosine_similarity(pred, batch_tgt, dim=-1)
            loss = (1.0 - cos).mean()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * (end - start)

        avg_loss = running_loss / dataset_size
        print(f"  [epoch {epoch:02d}] loss = {avg_loss:.4f}")


# -----------------------------
# Evaluation (skeleton)
# -----------------------------

@torch.no_grad()
def evaluate_predictor(model_p, test_inputs, test_targets):
    model_p.eval()
    test_inputs = test_inputs.to(DEVICE)
    test_targets = test_targets.to(DEVICE)

    pred = model_p(test_inputs)              # (N, D)
    cos_pred_true = cosine_similarity(pred, test_targets, dim=-1)

    # Baseline: just use z2 as a trivial predictor of z3
    baseline = test_inputs[:, 2, :]         # (N, D)
    cos_baseline = cosine_similarity(baseline, test_targets, dim=-1)

    print("\n[22_d6] Evaluation:")
    print(f"  mean cos(ẑ3, z3)      = {cos_pred_true.mean().item():.4f}")
    print(f"  mean cos(z2, z3) (base)= {cos_baseline.mean().item():.4f}")

    # Show a few examples
    for i in range(min(5, test_inputs.size(0))):
        print(f"    sample {i:02d}: cos(pred, true) = {cos_pred_true[i].item():.4f}, "
              f"cos(z2, true) = {cos_baseline[i].item():.4f}")


# -----------------------------
# Main
# -----------------------------

def main():
    # 1) Load JEPA
    processor, model = load_jepa()

    # 2) Build train/test sets in JEPA latent space
    train_inputs, train_targets = build_dataset(N_TRAIN_SEQS, processor, model)
    test_inputs, test_targets = build_dataset(N_TEST_SEQS, processor, model)

    _, _, dim = train_inputs.shape
    print(f"[22_d6] Latent dim D = {dim}")

    # 3) Initialize temporal predictor
    predictor = TemporalPredictor(dim=dim, n_steps=3, hidden_factor=2).to(DEVICE)
    print(f"[22_d6] Predictor parameters: {sum(p.numel() for p in predictor.parameters())}")

    # 4) Train
    train_predictor(predictor, train_inputs, train_targets)

    # 5) Evaluate temporal imagination
    evaluate_predictor(predictor, test_inputs, test_targets)


if __name__ == "__main__":
    main()
