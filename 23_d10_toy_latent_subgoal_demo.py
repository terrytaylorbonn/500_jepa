"""
[23_d10] Toy latent: goal-conditioned subgoal inference
23_d10_toy_latent_subgoal_demo.py

Idea:
  - Latent space z ∈ R^2 (just a 2D point).
  - Given (z_start, z_goal), learn fθ that outputs a subgoal z_sub:
        z_sub = fθ(z_start, z_goal)
  - Ground truth is simple linear interpolation toward goal:
        z_sub* = z_start + α (z_goal - z_start), with α in (0, 0.5)
  - After training:
        - we can iteratively apply fθ:
              z_{t+1} = fθ(z_t, z_goal)
          and watch z_t move toward z_goal in latent space.

This is a minimal toy for:
  - goal-conditioned imagination in latent space
  - setting up later connection to true dynamics + planning (CEM etc.)
"""

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT_DIM = 2
TRAIN_SAMPLES = 4096
TEST_SAMPLES = 512
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3

ALPHA = 0.25  # how far the ground-truth subgoal is between start and goal
RNG = random.Random(123)


# -----------------------------
# Data generation
# -----------------------------

def sample_pair() -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (z_start, z_goal) from uniform box in R^2.

    Returns:
      z_start, z_goal: tensors of shape (LATENT_DIM,)
    """
    z_start = torch.empty(LATENT_DIM).uniform_(-1.0, 1.0)
    z_goal = torch.empty(LATENT_DIM).uniform_(-1.0, 1.0)
    return z_start, z_goal


def build_dataset(n_samples: int):
    """
    Build dataset of (z_start, z_goal, z_sub_target).
    z_sub_target is a linear interpolation between start and goal:
        z_sub_target = z_start + ALPHA * (z_goal - z_start)
    """
    starts = []
    goals = []
    sub_targets = []

    for _ in range(n_samples):
        z_s, z_g = sample_pair()
        z_sub = z_s + ALPHA * (z_g - z_s)
        starts.append(z_s)
        goals.append(z_g)
        sub_targets.append(z_sub)

    starts = torch.stack(starts, dim=0)       # (N, D)
    goals = torch.stack(goals, dim=0)         # (N, D)
    subs = torch.stack(sub_targets, dim=0)    # (N, D)
    return starts, goals, subs


# -----------------------------
# Model
# -----------------------------

class GoalConditionedSubgoalNet(nn.Module):
    """
    fθ(z_start, z_goal) -> z_sub

    Simple MLP taking concatenated [z_start, z_goal] as input.
    """

    def __init__(self, dim: int):
        super().__init__()
        in_dim = 2 * dim
        hidden = 64
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, z_start: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        z_start: (B, D)
        z_goal:  (B, D)
        returns z_sub: (B, D)
        """
        x = torch.cat([z_start, z_goal], dim=-1)
        return self.net(x)


# -----------------------------
# Training loop
# -----------------------------

def train_model(model, train_starts, train_goals, train_subs):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    dataset_size = train_starts.size(0)
    n_batches = math.ceil(dataset_size / BATCH_SIZE)

    print(f"[23_d10] Training on {dataset_size} samples for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(dataset_size)
        starts_shuf = train_starts[perm].to(DEVICE)
        goals_shuf = train_goals[perm].to(DEVICE)
        subs_shuf = train_subs[perm].to(DEVICE)

        running_loss = 0.0

        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, dataset_size)

            z_s = starts_shuf[start:end]  # (B, D)
            z_g = goals_shuf[start:end]   # (B, D)
            z_tgt = subs_shuf[start:end]  # (B, D)

            optimizer.zero_grad()
            z_pred = model(z_s, z_g)

            loss = F.mse_loss(z_pred, z_tgt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * (end - start)

        avg_loss = running_loss / dataset_size
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [epoch {epoch:02d}] MSE = {avg_loss:.6f}")


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate_model(model, test_starts, test_goals, test_subs):
    model.eval()

    test_starts = test_starts.to(DEVICE)
    test_goals = test_goals.to(DEVICE)
    test_subs = test_subs.to(DEVICE)

    z_pred = model(test_starts, test_goals)
    mse = F.mse_loss(z_pred, test_subs).item()

    # Distance improvement when applying one subgoal step
    # For a sample:
    #   d0 = ||z_start - z_goal||
    #   d1 = ||z_pred  - z_goal||
    # Ideally d1 < d0.
    d0 = (test_starts - test_goals).norm(dim=-1)
    d1 = (z_pred - test_goals).norm(dim=-1)

    improvement = d0 - d1  # positive means closer to goal
    frac_improved = (improvement > 0).float().mean().item()
    mean_improvement = improvement.mean().item()

    print("\n[23_d10] Evaluation:")
    print(f"  Test MSE(z_sub)          = {mse:.6f}")
    print(f"  Mean distance improvement = {mean_improvement:.4f}")
    print(f"  Fraction improved         = {frac_improved * 100:.2f}%")

    # Show a few samples
    print("\n  Few sample trajectories:")
    for i in range(min(5, test_starts.size(0))):
        zs = test_starts[i].cpu().numpy()
        zg = test_goals[i].cpu().numpy()
        zp = z_pred[i].cpu().numpy()
        d0_i = d0[i].item()
        d1_i = d1[i].item()
        print(f"    sample {i:02d}:")
        print(f"      z_start = {zs}")
        print(f"      z_goal  = {zg}")
        print(f"      z_sub   = {zp}")
        print(f"      ||start-goal|| = {d0_i:.4f}")
        print(f"      ||sub-goal||   = {d1_i:.4f}")


@torch.no_grad()
def rollout_demo(model, n_steps: int = 5):
    """
    Pick a random (z_start, z_goal), iteratively apply the subgoal net:
        z_{t+1} = f(z_t, z_goal)
    and print the latent trajectory.
    """
    model.eval()

    z_start, z_goal = sample_pair()
    z_start = z_start.to(DEVICE)
    z_goal = z_goal.to(DEVICE)

    z_t = z_start.clone()

    print("\n[23_d10] Rollout demo:")
    print(f"  z_start = {z_start.cpu().numpy()}")
    print(f"  z_goal  = {z_goal.cpu().numpy()}")

    for t in range(n_steps):
        d = (z_t - z_goal).norm().item()
        print(f"    t={t}: z_t = {z_t.cpu().numpy()}, ||z_t - z_goal|| = {d:.4f}")
        z_t = model(z_t.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)

    # final distance
    d = (z_t - z_goal).norm().item()
    print(f"    t={n_steps}: z_t = {z_t.cpu().numpy()}, ||z_t - z_goal|| = {d:.4f}")


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[23_d10] Device: {DEVICE}")

    # 1) Build datasets
    train_starts, train_goals, train_subs = build_dataset(TRAIN_SAMPLES)
    test_starts, test_goals, test_subs = build_dataset(TEST_SAMPLES)

    print(f"[23_d10] Train samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")

    # 2) Init model
    model = GoalConditionedSubgoalNet(dim=LATENT_DIM).to(DEVICE)
    print(f"[23_d10] Model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3) Train
    train_model(model, train_starts, train_goals, train_subs)

    # 4) Evaluate
    evaluate_model(model, test_starts, test_goals, test_subs)

    # 5) Rollout a single trajectory
    rollout_demo(model, n_steps=6)


if __name__ == "__main__":
    main()
