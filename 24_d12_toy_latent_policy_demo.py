"""
[24_d12] Toy latent: distill planner into latent policy
24_d12_toy_latent_policy_demo.py

Idea:
  - Latent space z ∈ R^2 (same as D10).
  - We define a simple "planner" in latent space:
        z_sub = z_start + ALPHA * (z_goal - z_start)
    This is like a tiny one-step move toward the goal.
  - We treat this planner as the "expert".
  - We then train a POLICY network πθ(z_start, z_goal) to predict
        a = z_sub - z_start
    i.e., the action that moves z_start toward the subgoal.
  - At rollout time we use:
        z_{t+1} = z_t + πθ(z_t, z_goal)
    and compare against the expert planner rollout:
        z_{t+1}^expert = z_start + ALPHA * (z_goal - z_start)

This is a minimal demo of:
  - Distilling a planner (subgoals) into a policy (actions)
  - Using a simple latent dynamics model:
        z_{t+1} = z_t + a_t
"""

import math
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

ALPHA = 0.25  # fractional step toward goal


# -----------------------------
# Ground-truth planner
# -----------------------------

def planner_subgoal(z_start: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """
    Expert "planner" in latent space:
      z_sub = z_start + ALPHA * (z_goal - z_start)
    Works for both shape (D,) and (B, D).
    """
    return z_start + ALPHA * (z_goal - z_start)


# -----------------------------
# Dataset creation
# -----------------------------

def sample_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (z_start, z_goal) from uniform box in R^2.
    Returns 1D tensors of shape (LATENT_DIM,).
    """
    z_start = torch.empty(LATENT_DIM).uniform_(-1.0, 1.0)
    z_goal = torch.empty(LATENT_DIM).uniform_(-1.0, 1.0)
    return z_start, z_goal


def build_policy_dataset(n_samples: int):
    """
    Build dataset of (z_start, z_goal, action_target) pairs.

    For each sample:
      z_sub       = planner_subgoal(z_start, z_goal)
      action_tgt  = z_sub - z_start

    So the policy learns:
      π(z_start, z_goal) ≈ action_tgt
    """
    starts = []
    goals = []
    actions = []

    for _ in range(n_samples):
        z_s, z_g = sample_pair()
        z_sub = planner_subgoal(z_s, z_g)
        a = z_sub - z_s   # one-step move toward the subgoal
        starts.append(z_s)
        goals.append(z_g)
        actions.append(a)

    starts = torch.stack(starts, dim=0)    # (N, D)
    goals = torch.stack(goals, dim=0)      # (N, D)
    actions = torch.stack(actions, dim=0)  # (N, D)
    return starts, goals, actions


# -----------------------------
# Policy network
# -----------------------------

class LatentPolicyNet(nn.Module):
    """
    πθ(z_start, z_goal) -> action in R^2

    Input:  concat [z_start, z_goal] (4D)
    Output: action vector a (2D)
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
        returns: action: (B, D)
        """
        x = torch.cat([z_start, z_goal], dim=-1)
        return self.net(x)


# -----------------------------
# Training loop
# -----------------------------

def train_policy(policy: LatentPolicyNet,
                 train_starts: torch.Tensor,
                 train_goals: torch.Tensor,
                 train_actions: torch.Tensor):
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    dataset_size = train_starts.size(0)
    n_batches = math.ceil(dataset_size / BATCH_SIZE)

    print(f"[24_d12] Training policy on {dataset_size} samples for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(dataset_size)
        starts_shuf = train_starts[perm].to(DEVICE)
        goals_shuf = train_goals[perm].to(DEVICE)
        actions_shuf = train_actions[perm].to(DEVICE)

        running_loss = 0.0

        for b in range(n_batches):
            s = b * BATCH_SIZE
            e = min(s + BATCH_SIZE, dataset_size)

            z_s = starts_shuf[s:e]     # (B, D)
            z_g = goals_shuf[s:e]      # (B, D)
            a_tgt = actions_shuf[s:e]  # (B, D)

            optimizer.zero_grad()
            a_pred = policy(z_s, z_g)

            loss = F.mse_loss(a_pred, a_tgt)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * (e - s)

        avg_loss = running_loss / dataset_size
        if epoch % 5 == 0 or epoch == 1:
            print(f"  [epoch {epoch:02d}] MSE(action) = {avg_loss:.6f}")


# -----------------------------
# Evaluation
# -----------------------------

@torch.no_grad()
def evaluate_policy(policy: LatentPolicyNet,
                    test_starts: torch.Tensor,
                    test_goals: torch.Tensor,
                    test_actions: torch.Tensor):
    policy.eval()

    test_starts = test_starts.to(DEVICE)
    test_goals = test_goals.to(DEVICE)
    test_actions = test_actions.to(DEVICE)

    a_pred = policy(test_starts, test_goals)
    mse = F.mse_loss(a_pred, test_actions).item()

    # Check how much one step of expert vs policy reduces distance to goal.
    # Expert: z_next_exp  = planner_subgoal(z_start, z_goal)
    # Policy: z_next_pol  = z_start + a_pred
    z_next_exp = planner_subgoal(test_starts, test_goals)
    z_next_pol = test_starts + a_pred

    d0 = (test_starts - test_goals).norm(dim=-1)  # start distance
    d_exp = (z_next_exp - test_goals).norm(dim=-1)
    d_pol = (z_next_pol - test_goals).norm(dim=-1)

    # Improvements (positive means closer to goal)
    imp_exp = d0 - d_exp
    imp_pol = d0 - d_pol

    frac_improved_pol = (imp_pol > 0).float().mean().item()
    mean_imp_exp = imp_exp.mean().item()
    mean_imp_pol = imp_pol.mean().item()

    print("\n[24_d12] Evaluation (one-step):")
    print(f"  Test MSE(action)          = {mse:.6f}")
    print(f"  Expert mean improvement   = {mean_imp_exp:.4f}")
    print(f"  Policy mean improvement   = {mean_imp_pol:.4f}")
    print(f"  Policy fraction improved  = {frac_improved_pol * 100:.2f}%")

    print("\n  Few sample one-step comparisons:")
    for i in range(min(5, test_starts.size(0))):
        zs = test_starts[i].cpu().numpy()
        zg = test_goals[i].cpu().numpy()
        a_t = test_actions[i].cpu().numpy()
        a_p = a_pred[i].cpu().numpy()
        print(f"    sample {i:02d}:")
        print(f"      z_start     = {zs}")
        print(f"      z_goal      = {zg}")
        print(f"      a_target    = {a_t}")
        print(f"      a_pred      = {a_p}")
        print(f"      ||start-goal||    = {d0[i].item():.4f}")
        print(f"      ||exp_next-goal|| = {d_exp[i].item():.4f}")
        print(f"      ||pol_next-goal|| = {d_pol[i].item():.4f}")


@torch.no_grad()
def rollout_comparison(policy: LatentPolicyNet, n_steps: int = 6):
    """
    Compare expert planner vs learned policy over multiple steps.

    Expert rollout (latent planner):
      z_{t+1}^exp = planner_subgoal(z_t^exp, z_goal)

    Policy rollout (latent policy + dynamics):
      a_t         = π(z_t^pol, z_goal)
      z_{t+1}^pol = z_t^pol + a_t
    """
    policy.eval()

    # Sample one start/goal pair
    z_start, z_goal = sample_pair()
    z_start = z_start.to(DEVICE)
    z_goal = z_goal.to(DEVICE)

    z_exp = z_start.clone()
    z_pol = z_start.clone()

    print("\n[24_d12] Rollout comparison (expert vs policy):")
    print(f"  z_start = {z_start.cpu().numpy()}")
    print(f"  z_goal  = {z_goal.cpu().numpy()}")

    for t in range(n_steps + 1):
        d_exp = (z_exp - z_goal).norm().item()
        d_pol = (z_pol - z_goal).norm().item()
        print(f"    t={t}:")
        print(f"      expert z_t = {z_exp.cpu().numpy()}, ||z_exp - z_goal|| = {d_exp:.4f}")
        print(f"      policy z_t = {z_pol.cpu().numpy()}, ||z_pol - z_goal|| = {d_pol:.4f}")

        # Next step (skip after last print)
        if t == n_steps:
            break

        # Expert uses planner subgoal directly:
        z_exp = planner_subgoal(z_exp, z_goal)

        # Policy produces an action, then we apply simple dynamics:
        a_pol = policy(z_pol.unsqueeze(0), z_goal.unsqueeze(0)).squeeze(0)
        z_pol = z_pol + a_pol


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"[24_d12] Device: {DEVICE}")

    # 1) Build dataset from ground-truth planner
    train_starts, train_goals, train_actions = build_policy_dataset(TRAIN_SAMPLES)
    test_starts, test_goals, test_actions = build_policy_dataset(TEST_SAMPLES)
    print(f"[24_d12] Train samples: {TRAIN_SAMPLES}, Test samples: {TEST_SAMPLES}")

    # 2) Init policy
    policy = LatentPolicyNet(dim=LATENT_DIM).to(DEVICE)
    print(f"[24_d12] Policy parameters: {sum(p.numel() for p in policy.parameters())}")

    # 3) Train policy to mimic planner's one-step behavior
    train_policy(policy, train_starts, train_goals, train_actions)

    # 4) Evaluate one-step performance
    evaluate_policy(policy, test_starts, test_goals, test_actions)

    # 5) Multi-step rollout comparison (expert vs policy)
    rollout_comparison(policy, n_steps=6)


if __name__ == "__main__":
    main()
