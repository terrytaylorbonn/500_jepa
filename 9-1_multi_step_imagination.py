#!/usr/bin/env python3
"""
9-1_multi_step_imagination.py (RENAMED 26.0218)

Minimal "action-conditioned predictor" training demo (JEPA-style), runnable on a laptop.

We:
- Freeze the I-JEPA encoder (facebook/ijepa_vith14_1k)
- Synthesize "time" and "actions" from ONE image by generating pairs:
    x_t   = mild augmentation of base image
    x_t+1 = augmentation whose severity depends on a fake action a_t
- Encode both frames:
    z_t   = encoder(x_t)      (CLS embedding)
    z_t+1 = encoder(x_t+1)    (CLS embedding)  ← target latent
- Train a small predictor:
    ẑ_t+1 = predictor(z_t, a_t)
- Loss compares predicted latent to target latent (cosine loss)

THEN (Level 2):
- Starting from a single latent belief z_0, we repeatedly apply the predictor:
    z_0 → ẑ_1 → ẑ_2 → ... → ẑ_T
- For each action a, we measure cosine(z_0, ẑ_t) over t = 1..T
- This shows how different actions define different imagined trajectories in latent space.

Run:
  python 9_multi_step_imagination.py --image ./image1.jpg

Optional:
  --steps 400 --batch 16 --actions 3 --rollout_steps 5
"""

import argparse
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import AutoModel, AutoProcessor
from torch.nn.functional import cosine_similarity

# -----------------------------
# Utilities
# -----------------------------
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity, averaged over batch."""
    pred_n = l2_normalize(pred)
    tgt_n = l2_normalize(target)
    sim = (pred_n * tgt_n).sum(dim=-1)  # (B,)
    return (1.0 - sim).mean()


def random_crop(img: Image.Image, strength: float) -> Image.Image:
    """
    strength in [0, 1]. Higher = more aggressive crop.
    """
    w, h = img.size
    # crop ratio from 1.0 down to ~0.6 depending on strength
    min_ratio = 0.60
    ratio = 1.0 - strength * (1.0 - min_ratio)
    cw, ch = int(w * ratio), int(h * ratio)
    if cw < 2 or ch < 2:
        return img

    x0 = random.randint(0, max(0, w - cw))
    y0 = random.randint(0, max(0, h - ch))
    cropped = img.crop((x0, y0, x0 + cw, y0 + ch))
    return cropped.resize((w, h), resample=Image.BICUBIC)


def random_occlude(img: Image.Image, strength: float) -> Image.Image:
    """
    strength in [0, 1]. Higher = bigger occlusion box.
    """
    out = img.copy()
    w, h = out.size
    # occlusion area up to ~25% depending on strength
    max_frac = 0.25
    frac = strength * max_frac
    box_w = max(1, int(w * (frac ** 0.5)))
    box_h = max(1, int(h * (frac ** 0.5)))

    x0 = random.randint(0, max(0, w - box_w))
    y0 = random.randint(0, max(0, h - box_h))

    # draw black rectangle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(out)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(0, 0, 0))
    return out


def random_color_jitter(img: Image.Image, strength: float) -> Image.Image:
    """
    strength in [0, 1]. Higher = stronger brightness/contrast jitter.
    """
    out = img
    # brightness range: 1 ± 0.4*strength
    b = 1.0 + (random.uniform(-0.4, 0.4) * strength)
    c = 1.0 + (random.uniform(-0.4, 0.4) * strength)
    out = ImageEnhance.Brightness(out).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    return out


def make_augmented(img: Image.Image, strength: float) -> Image.Image:
    """Compose a few simple augmentations."""
    out = img
    out = random_crop(out, strength=strength)
    out = random_color_jitter(out, strength=strength)
    out = random_occlude(out, strength=strength)
    return out


# -----------------------------
# Encoder wrapper (CLS only)
# -----------------------------
@torch.no_grad()
def encode_cls(model, processor, pil_images, device: torch.device) -> torch.Tensor:
    """
    Encode a list of PIL images -> CLS embeddings (B, D)
    """
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    h = outputs.last_hidden_state  # (B, 1+N, D)
    z_cls = h[:, 0, :]             # (B, D)
    return z_cls


# -----------------------------
# Fake action-conditioned predictor
# -----------------------------
class ActionConditionedPredictor(nn.Module):
    """
    Minimal predictor:
      input: z_t (B,D) and action id a_t (B,)
      output: z_hat_t1 (B,D)

    We embed action -> vector and concatenate with z_t, then MLP.
    """
    def __init__(self, dim: int, num_actions: int, action_emb_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(dim + action_emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_emb(a_t)  # (B, A)
        x = torch.cat([z_t, a_emb], dim=-1)
        return self.net(x)


# -----------------------------
# Training batch generator
# -----------------------------
@dataclass
class Batch:
    x_t: list          # list[PIL.Image]
    x_t1: list         # list[PIL.Image]
    a_t: torch.Tensor  # (B,)


def sample_batch(base_img: Image.Image, batch_size: int, num_actions: int) -> Batch:
    """
    Fake "actions" = discrete augmentation severity levels:
      a=0 -> mild change
      a=max -> strong change
    """
    x_t = []
    x_t1 = []
    actions = []
    for _ in range(batch_size):
        a = random.randint(0, num_actions - 1)
        actions.append(a)

        # frame t: always mild-ish observation
        strength_t = 0.15

        # frame t+1: severity depends on action
        # map action id to strength in [0.15, 0.95]
        if num_actions == 1:
            strength_t1 = 0.50
        else:
            strength_t1 = 0.15 + (a / (num_actions - 1)) * (0.95 - 0.15)

        x_t.append(make_augmented(base_img, strength=strength_t))
        x_t1.append(make_augmented(base_img, strength=strength_t1))

    return Batch(x_t=x_t, x_t1=x_t1, a_t=torch.tensor(actions, dtype=torch.long))


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Base image to synthesize frame pairs from")
    ap.add_argument("--model_id", default="facebook/ijepa_vith14_1k")
    ap.add_argument("--actions", type=int, default=3, help="Number of fake actions (severity levels)")
    ap.add_argument("--steps", type=int, default=300, help="Training steps")
    ap.add_argument("--batch", type=int, default=12, help="Batch size")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    ap.add_argument("--print_every", type=int, default=50)
    # NEW: rollout steps for multi-step imagination
    ap.add_argument("--rollout_steps", type=int, default=5,
                    help="Number of imagination steps per action for multi-step rollout")
    args = ap.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load encoder (frozen)
    print("\n[1] Load frozen encoder + processor")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model_id).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print("  encoder frozen ✅")

    # Load base image
    base_img = Image.open(args.image).convert("RGB")
    print(f"\n[2] Base image loaded: size={base_img.size}")

    # Infer embedding dim from one forward pass
    with torch.no_grad():
        z0 = encode_cls(encoder, processor, [base_img], device)
    dim = z0.shape[-1]
    print(f"\n[3] CLS embedding dim D = {dim}")

    # Predictor (trainable)
    predictor = ActionConditionedPredictor(dim=dim, num_actions=args.actions).to(device).train()
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    print("\n[4] Training predictor to map (z_t, a_t) → ẑ_t+1")
    print("    Target is z_t+1 = encoder(x_t+1)  (stop-grad / frozen encoder)")
    print("    Loss = 1 - cosine(ẑ_t+1, z_t+1)\n")

    # Train loop
    for step in range(1, args.steps + 1):
        batch = sample_batch(base_img, args.batch, args.actions)

        # Encode both frames (frozen encoder)
        with torch.no_grad():
            z_t = encode_cls(encoder, processor, batch.x_t, device)      # (B,D)
            z_t1 = encode_cls(encoder, processor, batch.x_t1, device)    # (B,D)

        a_t = batch.a_t.to(device)  # (B,)

        # Predict
        z_hat_t1 = predictor(z_t, a_t)

        # Train loss in latent space
        loss = cosine_loss(z_hat_t1, z_t1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.print_every == 0 or step == 1 or step == args.steps:
            with torch.no_grad(): sim = (l2_normalize(z_hat_t1) * l2_normalize(z_t1)).sum(dim=-1).mean().item()
            print(f"step {step:>4}/{args.steps}  loss={loss.item():.4f}  mean_cos(ẑ_t+1, z_t+1)={sim:.4f}")

    # Demonstrate "imagination": same z_t, different actions → different predicted ẑ
    print("\n[5] Imagination demo (same current belief, vary action)")
    predictor.eval()
    with torch.no_grad():
        # pick one fresh x_t
        x_t = make_augmented(base_img, strength=0.15)
        z_t_single = encode_cls(encoder, processor, [x_t], device)  # (1,D)

        preds = []
        for a in range(args.actions):
            a_t = torch.tensor([a], device=device)
            z_hat = predictor(z_t_single, a_t)  # (1,D)
            preds.append(z_hat)

        # Compare predicted future beliefs between actions
        print("  pairwise cosine similarities between predicted ẑ_t+1 for different actions:")
        for i in range(args.actions):
            for j in range(i + 1, args.actions):
                sim_ij = cosine_similarity(
                    l2_normalize(preds[i]), l2_normalize(preds[j])
                ).item()
                print(f"    cos(ẑ | a={i}, ẑ | a={j}) = {sim_ij:.4f}")

    # NEW: [6] Multi-step imagination rollout (Level 2)
    print(f"\n[6] Multi-step imagination rollouts (Level 2, steps={args.rollout_steps})")

    def rollout_latent(z0: torch.Tensor, action_id: int, steps: int) -> list[float]:
        """
        Starting from latent z0, repeatedly apply the predictor with a fixed action.
        Returns the cosine similarity cos(z0, z_t) for t=1..steps.
        """
        z = z0.clone()
        sims = []
        with torch.no_grad():
            for t in range(steps):
                a_t = torch.tensor([action_id], device=device, dtype=torch.long)
                z = predictor(z, a_t)  # one imagination step: z_t -> ẑ_{t+1}

                sim = float(
                    cosine_similarity(
                        l2_normalize(z0), l2_normalize(z)
                    ).item()
                )
                sims.append(sim)
        return sims

    # Use a fresh mild observation as the starting belief z0
    with torch.no_grad():
        x0 = make_augmented(base_img, strength=0.15)
        z0_rollout = encode_cls(encoder, processor, [x0], device)  # (1,D)

    for a in range(args.actions):
        sims = rollout_latent(z0_rollout, action_id=a, steps=args.rollout_steps)
        sims_str = ", ".join(f"{s:.4f}" for s in sims)
        print(f"  action = {a}: cos(z_0, z_t) over {args.rollout_steps} steps -> [{sims_str}]")

    print("\nDONE.\n")
    print("Interpretation:")
    print("- Encoder produces z_t and z_t+1 (belief states) from observations.")
    print("- Predictor learns an action-conditioned mapping in latent space.")
    print("- Here actions are fake (augmentation severity). In real V-JEPA-2-AC, actions are robot controls.")
    print("- [Level 2] Re-applying the predictor shows how different actions define different imagined")
    print("            trajectories in latent belief space over multiple steps.")


if __name__ == "__main__":
    main()

