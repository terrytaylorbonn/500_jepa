"""
50_D55_confidence_measurement_head_A.py

D55 VERSION A (working demo):
- Measurement head predicts (x,y) AND uncertainty (sigma)  <-- NEW in D55
- Belief update uses uncertainty-aware normalized innovation gate
- Residual is trained only from innovation (no s_true in residual loss)  <-- keep D54 rule
- Adds "real-world garbage" toggles: dropout/outliers/partial/delay (optional)

This is intentionally a minimal, self-contained Toy2D demo:
- True state exists only for simulation + head supervision (like having a labeled sensor model).
- Residual never uses s_true in its own loss.
"""

import argparse
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility
# -----------------------------
def device_select():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dev


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_gate(d, good, bad):
    """
    D54-style soft trust gate.
    Returns w in [0,1].
    w=1 when d<=good, w=0 when d>=bad, linear between.
    """
    if bad <= good:
        # safety: avoid divide by zero / inverted params
        return float(d <= good)
    if d <= good:
        return 1.0
    if d >= bad:
        return 0.0
    return float(1.0 - (d - good) / (bad - good))


def huber(x, delta=1.0):
    # Smooth L1 / Huber
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta, reduction="none")


# -----------------------------
# Toy 2D dynamics + measurement generator
# -----------------------------
class Toy2DEnv:
    """
    Simple 2D point mass with velocity.
    State: [x, y, vx, vy]
    Control: [ax, ay]
    """
    def __init__(self, dt=0.1, friction=0.97):
        self.dt = dt
        self.friction = friction
        self.reset()

    def reset(self):
        self.s = np.zeros(4, dtype=np.float32)
        self.s[0] = 0.0
        self.s[1] = 0.0
        self.s[2] = 1.0
        self.s[3] = 0.5
        return self.s.copy()

    def step(self, a):
        x, y, vx, vy = self.s
        ax, ay = a
        dt = self.dt

        # semi-implicit-ish update; simple, stable
        vx_next = self.friction * vx + ax * dt
        vy_next = self.friction * vy + ay * dt
        x_next = x + vx_next * dt
        y_next = y + vy_next * dt

        self.s = np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)
        return self.s.copy()


def sample_action(t):
    # small smooth-ish acceleration signal
    ax = 0.7 * math.sin(0.07 * t) + 0.15 * math.sin(0.011 * t)
    ay = 0.6 * math.cos(0.05 * t) + 0.10 * math.sin(0.017 * t)
    return np.array([ax, ay], dtype=np.float32)


class MeasurementChannel:
    """
    Produces a "raw measurement feature" from the true position, with optional garbage:
      - dropout: no measurement sometimes
      - outliers: occasionally big jump
      - partial obs: observe only x or only y
      - delay: measurement arrives late (k steps)

    For simplicity, the raw feature is always a 2D vector plus a mask:
      feat = [mx, my, mask_x, mask_y]
    """
    def __init__(
        self,
        noise_std=0.05,
        p_drop=0.0,
        p_out=0.0,
        outlier_scale=10.0,
        partial_mode="none",  # "none" | "x" | "y" | "random"
        delay_k=0,
        seed=0,
    ):
        self.noise_std = noise_std
        self.p_drop = p_drop
        self.p_out = p_out
        self.outlier_scale = outlier_scale
        self.partial_mode = partial_mode
        self.delay_k = int(delay_k)
        self.rng = np.random.RandomState(seed)

        self.delay_buf = deque(maxlen=max(1, self.delay_k + 1))
        # Initialize delay buffer with "no measurement"
        for _ in range(self.delay_k + 1):
            self.delay_buf.append(None)

    def _make_partial_mask(self):
        if self.partial_mode == "none":
            return np.array([1.0, 1.0], dtype=np.float32)
        if self.partial_mode == "x":
            return np.array([1.0, 0.0], dtype=np.float32)
        if self.partial_mode == "y":
            return np.array([0.0, 1.0], dtype=np.float32)
        if self.partial_mode == "random":
            return np.array([1.0, 0.0], dtype=np.float32) if self.rng.rand() < 0.5 else np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 1.0], dtype=np.float32)

    def observe(self, true_pos_xy):
        # dropout
        if self.rng.rand() < self.p_drop:
            meas_now = None
        else:
            # base noisy measurement
            noise = self.noise_std * self.rng.randn(2).astype(np.float32)
            meas = true_pos_xy.astype(np.float32) + noise

            # outlier
            if self.rng.rand() < self.p_out:
                meas = meas + (self.outlier_scale * self.noise_std) * self.rng.randn(2).astype(np.float32)

            # partial
            mask = self._make_partial_mask()
            # for missing dims, keep value but mark mask=0 (head can learn to ignore)
            meas_now = (meas, mask)

        # delay handling
        self.delay_buf.append(meas_now)
        return self.delay_buf[0]  # delayed output


# -----------------------------
# Models
# -----------------------------
class MeasHead(nn.Module):
    """
    D55 CHANGE: Head outputs both mu (x,y) and log_sigma (x,y)
    Input: feat = [mx, my, mask_x, mask_y]
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.log_sigma = nn.Linear(hidden, 2)

    def forward(self, feat):
        h = self.net(feat)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        # clamp for numerical sanity
        log_sigma = torch.clamp(log_sigma, min=-6.0, max=3.0)
        return mu, log_sigma


class ResidualNet(nn.Module):
    """
    Residual predicts correction delta from innovation features.
    D54 rule preserved: residual loss target is innovation only (no s_true inside residual loss).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),  # [innov_x, innov_y, logσx, logσy, mask_x, mask_y]
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Training / main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.97)

    ap.add_argument("--meas_noise_std", type=float, default=0.05)

    # D55 garbage toggles
    ap.add_argument("--p_drop", type=float, default=0.0)
    ap.add_argument("--p_out", type=float, default=0.0)
    ap.add_argument("--outlier_scale", type=float, default=10.0)
    ap.add_argument("--partial_mode", type=str, default="none", choices=["none", "x", "y", "random"])
    ap.add_argument("--delay_k", type=int, default=0)

    # D54-ish knobs
    ap.add_argument("--head_warmup_steps", type=int, default=200)
    ap.add_argument("--head_train_steps_per_env_step", type=int, default=2)
    ap.add_argument("--innov_clip", type=float, default=2.0)

    # Trust gate (now in normalized innovation units)
    ap.add_argument("--gate_distance_good", type=float, default=1.5)
    ap.add_argument("--gate_distance_bad", type=float, default=5.0)

    # Learning rates
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_residual", type=float, default=1e-3)

    # Robust loss
    ap.add_argument("--huber_k", type=float, default=1.0)

    # logging
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_select()
    print(f"[D55-A] Device: {dev}")

    env = Toy2DEnv(dt=args.dt, friction=args.friction)
    meas = MeasurementChannel(
        noise_std=args.meas_noise_std,
        p_drop=args.p_drop,
        p_out=args.p_out,
        outlier_scale=args.outlier_scale,
        partial_mode=args.partial_mode,
        delay_k=args.delay_k,
        seed=args.seed + 123,
    )

    head = MeasHead(hidden=64).to(dev)
    resnet = ResidualNet(hidden=64).to(dev)

    opt_head = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    opt_res = torch.optim.Adam(resnet.parameters(), lr=args.lr_residual)

    # Belief state (our estimate): position + velocity
    s = env.reset()
    pos_bel = s[:2].copy()
    vel_bel = s[2:].copy()

    # Running stats for “belief acceptance”
    accepted_ema = 0.0
    w_ema = 0.0

    for t in range(args.steps):
        # --- simulate true dynamics ---
        a = sample_action(t)
        s_true = env.step(a)
        pos_true = s_true[:2]

        # --- predictor prior (simple kinematics using belief vel) ---
        pos_pred = pos_bel + vel_bel * args.dt

        # --- raw measurement feature from channel ---
        m = meas.observe(pos_true)
        if m is None:
            # no measurement
            mx, my = 0.0, 0.0
            mask = np.array([0.0, 0.0], dtype=np.float32)
            drop_flag = 1.0
        else:
            (mxy, mask) = m
            mx, my = float(mxy[0]), float(mxy[1])
            drop_flag = 0.0

        feat = torch.tensor([[mx, my, mask[0], mask[1]]], dtype=torch.float32, device=dev)

        # --- head predicts measurement + uncertainty ---
        mu, log_sigma = head(feat)
        sigma = torch.exp(log_sigma) + 1e-6  # sigma floor
        xy_hat = mu[0].detach().cpu().numpy()

        # Innovation (in state space)
        innov = xy_hat - pos_pred

        # D55 CHANGE: normalized innovation distance uses sigma
        sigma_np = sigma[0].detach().cpu().numpy()
        # if partial obs: ignore missing dims by setting sigma huge for masked dims
        sigma_eff = sigma_np.copy()
        for i in range(2):
            if mask[i] < 0.5:
                sigma_eff[i] = 1e6

        dn = float(math.sqrt((innov[0] / sigma_eff[0]) ** 2 + (innov[1] / sigma_eff[1]) ** 2))

        # D54 warmup logic (still)
        if t < args.head_warmup_steps:
            w = 0.0
        else:
            w = soft_gate(dn, args.gate_distance_good, args.gate_distance_bad)

        # Residual net input: innov + log_sigma + mask
        innov_t = torch.tensor([[innov[0], innov[1]]], dtype=torch.float32, device=dev)
        log_sigma_t = log_sigma.detach()  # keep residual independent of head gradients (optional; keeps roles clean)
        mask_t = torch.tensor([[mask[0], mask[1]]], dtype=torch.float32, device=dev)
        res_in = torch.cat([innov_t, log_sigma_t, mask_t], dim=1)

        # Residual prediction
        delta_hat = resnet(res_in)[0].detach().cpu().numpy()

        # Belief update:
        # - If you want pure innovation, use innov.
        # - If you want learned correction, use delta_hat.
        # Here we use learned correction but it is trained toward innovation.
        pos_corr = pos_pred + w * delta_hat

        # Update belief velocity from correction (simple; keeps demo stable)
        vel_bel = (pos_corr - pos_bel) / args.dt
        pos_bel = pos_corr

        # --- TRAINING: head + residual ---
        # We train head and residual multiple steps per env step, like D54.
        head_loss_val = 0.0
        res_loss_val = 0.0

        for _ in range(args.head_train_steps_per_env_step):
            # HEAD TRAIN:
            # D55 CHANGE: heteroscedastic NLL (mu + sigma)
            mu2, log_sigma2 = head(feat)
            sigma2 = torch.exp(log_sigma2) + 1e-6

            # target for head is the TRUE position (this is the "labeled sensor model" assumption)
            # Missing dims are masked out.
            y = torch.tensor([[pos_true[0], pos_true[1]]], dtype=torch.float32, device=dev)
            mask2 = mask_t

            # NLL per-dim: (err^2 / (2σ^2) + logσ)
            err = (y - mu2)
            nll = (err * err) / (2.0 * sigma2 * sigma2) + log_sigma2
            nll = nll * mask2  # apply partial observation mask
            head_loss = nll.mean()

            opt_head.zero_grad(set_to_none=True)
            head_loss.backward()
            opt_head.step()
            head_loss_val += float(head_loss.detach().cpu().item())

            # RESIDUAL TRAIN:
            # D54 rule preserved: residual target is innovation only (no s_true in residual loss).
            # Target innov is (xy_hat - pos_pred). Here we use mu2 (head output) as measurement estimate.
            with torch.no_grad():
                xy_hat2 = mu2[0].detach().cpu().numpy()
            innov2 = xy_hat2 - pos_pred
            innov2 = np.clip(innov2, -args.innov_clip, args.innov_clip)

            innov2_t = torch.tensor([[innov2[0], innov2[1]]], dtype=torch.float32, device=dev)
            res_in2 = torch.cat([innov2_t, log_sigma2.detach(), mask2], dim=1)
            delta2 = resnet(res_in2)

            # Huber on (delta - innov) for robustness to outliers
            diff = delta2 - innov2_t
            per = huber(diff, delta=args.huber_k)  # returns per-element
            per = per * mask2  # if partial obs, ignore missing dims
            res_loss = per.mean()

            opt_res.zero_grad(set_to_none=True)
            res_loss.backward()
            opt_res.step()
            res_loss_val += float(res_loss.detach().cpu().item())

        head_loss_val /= max(1, args.head_train_steps_per_env_step)
        res_loss_val /= max(1, args.head_train_steps_per_env_step)

        # Diagnostics
        pos_err = float(np.linalg.norm(pos_bel - pos_true))
        d_state = float(np.linalg.norm(xy_hat - pos_pred))

        accepted = 1.0 if w > 0.1 else 0.0
        accepted_ema = 0.98 * accepted_ema + 0.02 * accepted
        w_ema = 0.98 * w_ema + 0.02 * w

        if (t % args.print_every) == 0:
            sig_mean = float(np.mean(sigma_np))
            print(
                f"[D55-A][t={t:04d}] "
                f"res_loss={res_loss_val:.5f}  head_loss={head_loss_val:.5f}  "
                f"d=|xy_hat-pos_pred|={d_state:.3f}  "
                f"dn={dn:.3f}  w={w:.2f}  "
                f"|pos_err|={pos_err:.3f}  "
                f"sigma_mean={sig_mean:.3f}  "
                f"drop={int(drop_flag)}  "
                f"acc%~={100.0*accepted_ema:.1f}  w_ema={w_ema:.2f}"
            )

    print("[D55-A] Done.")


if __name__ == "__main__":
    main()
