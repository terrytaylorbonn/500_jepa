#43_49b2B_online_residual_measurement_only.py

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 43_49b2B: Online residual learning (measurement-only)
#
# State s = [x, y, vx, vy]
# Action a = [ax, ay]
#
# - True env uses f_true (slightly mismatched friction vs prior).
# - Prior model f_prior is hand-written physics (imperfect).
# - Residual net r_theta learns corrections to prior *only* from
#   measurement residuals (no direct access to s_{t+1} in the loss).
# - Measurement y = h(s) = [x, y] + noise.
#
# Belief b_t is the model's internal state estimate.
# It evolves via:
#   1) Prediction: b_pred = f_prior(b_t, a_t) + r_theta(b_t, a_t)
#   2) Measurement update (simple fixed gain):
#        e = y_obs - h(b_pred)
#        b_{t+1} = b_pred + K_meas * [e_x, e_y, 0, 0]
# -----------------------------


DT = 0.1


@dataclass
class Config:
    steps: int = 400
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    meas_noise_std: float = 0.05
    meas_gain: float = 0.5
    log_interval: int = 20
    seed: int = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def f_true(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    True environment dynamics (unknown to the model).
    s: (..., 4) [x, y, vx, vy]
    a: (..., 2) [ax, ay]
    """
    x, y, vx, vy = torch.unbind(s, dim=-1)
    ax, ay = torch.unbind(a, dim=-1)

    # Slightly stronger friction and a bias to make prior wrong.
    friction = 0.93
    vx_next = friction * vx + 0.5 * ax * DT
    vy_next = friction * vy + 0.5 * ay * DT
    x_next = x + vx_next * DT
    y_next = y + vy_next * DT

    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


def f_prior(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Imperfect prior dynamics used by the agent.
    """
    x, y, vx, vy = torch.unbind(s, dim=-1)
    ax, ay = torch.unbind(a, dim=-1)

    friction = 0.98  # Wrong friction
    vx_next = friction * vx + 0.5 * ax * DT
    vy_next = friction * vy + 0.5 * ay * DT
    x_next = x + vx_next * DT
    y_next = y + vy_next * DT

    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


def h_measure(s: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
    """
    Measurement model: observe position only.
    s: (..., 4)
    returns y: (..., 2) [x, y] + noise
    """
    pos = s[..., :2]
    if noise_std > 0.0:
        noise = noise_std * torch.randn_like(pos)
        pos = pos + noise
    return pos


class ResidualNet(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


def generate_action(t: int) -> np.ndarray:
    """
    Simple open-loop policy to excite dynamics:
    - Slowly rotating acceleration vector.
    """
    angle = 0.02 * t
    ax = math.cos(angle)
    ay = math.sin(angle)
    return np.array([ax, ay], dtype=np.float32)


def run_demo(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[49b2B] Device: {device}")

    set_seed(cfg.seed)

    residual_net = ResidualNet().to(device)
    optimizer = optim.Adam(residual_net.parameters(), lr=cfg.lr)

    # Initial true state and belief (start them equal for clarity)
    s_true = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    b = s_true.clone().detach()

    # Logging buffers
    true_traj = []
    belief_traj = []
    meas_traj = []
    losses = []

    print("[49b2B] Starting online residual learning (measurement-only)")
    print(f"  steps           = {cfg.steps}")
    print(f"  meas_noise_std  = {cfg.meas_noise_std}")
    print(f"  meas_gain       = {cfg.meas_gain}")
    print(f"  lr              = {cfg.lr}")
    print("")

    for t in range(cfg.steps):
        # 1) Choose action
        a_np = generate_action(t)
        a = torch.tensor(a_np, device=device).unsqueeze(0)  # (1, 2)

        # 2) True environment step (hidden from the learner)
        s_true = f_true(s_true, a)

        # 3) Measurement from true state
        y_obs = h_measure(s_true, noise_std=cfg.meas_noise_std)

        # 4) Prediction using prior + residual model
        with torch.no_grad():
            s_prior = f_prior(b, a)
        s_prior.requires_grad_(False)  # keep as leaf for clarity

        s_pred = s_prior + residual_net(b, a)  # belief dynamics prior

        # 5) Predicted measurement from belief
        y_pred = h_measure(s_pred)

        # 6) Measurement-only loss: match y_pred to y_obs
        loss = torch.mean((y_pred - y_obs) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) Simple measurement update on belief (fixed gain)
        with torch.no_grad():
            y_pred_detached = h_measure(s_pred.detach())
            e = (y_obs - y_pred_detached)  # (1, 2)
            # Expand to state correction: only position gets corrected
            dx = cfg.meas_gain * e[..., 0:1]
            dy = cfg.meas_gain * e[..., 1:2]
            dvx = torch.zeros_like(dx)
            dvy = torch.zeros_like(dy)
            correction = torch.cat([dx, dy, dvx, dvy], dim=-1)
            b = s_pred.detach() + correction

        # 8) Logging
        true_traj.append(s_true.detach().cpu().numpy()[0])
        belief_traj.append(b.detach().cpu().numpy()[0])
        meas_traj.append(y_obs.detach().cpu().numpy()[0])
        losses.append(loss.item())

        if (t % cfg.log_interval) == 0 or (t == cfg.steps - 1):
            pos_true = s_true[0, :2].detach().cpu().numpy()
            pos_belief = b[0, :2].detach().cpu().numpy()
            pos_err = np.linalg.norm(pos_true - pos_belief)
            meas_err = np.linalg.norm(
                y_obs[0].detach().cpu().numpy() - y_pred[0].detach().cpu().numpy()
            )
            print(
                f"[t={t:03d}] "
                f"loss={loss.item():.5f}  "
                f"|pos_err|={pos_err:.3f}  "
                f"|meas_err|={meas_err:.3f}"
            )

    true_traj = np.stack(true_traj, axis=0)
    belief_traj = np.stack(belief_traj, axis=0)
    meas_traj = np.stack(meas_traj, axis=0)
    losses = np.array(losses)

    rms_pos_err = np.sqrt(
        np.mean(np.sum((true_traj[:, :2] - belief_traj[:, :2]) ** 2, axis=-1))
    )
    rms_meas_err = np.sqrt(
        np.mean(np.sum((meas_traj - belief_traj[:, :2]) ** 2, axis=-1))
    )
    print("")
    print("[49b2B] Finished.")
    print(f"  RMS position error (true vs belief)   = {rms_pos_err:.4f}")
    print(f"  RMS measurement error (belief vs y)   = {rms_meas_err:.4f}")
    print(f"  Final loss (last step)                = {losses[-1]:.6f}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="49b2B: Online residual learning (measurement-only)"
    )
    parser.add_argument("--steps", type=int, default=400, help="Number of time steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--meas-noise-std", type=float, default=0.05, help="Measurement noise std"
    )
    parser.add_argument(
        "--meas-gain", type=float, default=0.5, help="Measurement update gain"
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"
    cfg = Config(
        steps=args.steps,
        device=device,
        lr=args.lr,
        meas_noise_std=args.meas_noise_std,
        meas_gain=args.meas_gain,
        seed=args.seed,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    run_demo(cfg)


