"""
51_D55B_confidence_measurement_head_B.py

D55 VERSION B — "real-world garbage" + uncertainty becomes meaningful

DIFFS vs D55-A1:
B-1) Default dropout/outliers enabled (mild) so sigma has a reason to inflate
B-2) Add sigma regularizer to prevent sigma collapsing to the floor
B-3) Log outlier/drop flags separately
"""

import argparse
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def device_select():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_gate(d: float, good: float, bad: float) -> float:
    """Soft trust gate: 1 below good, 0 above bad, linear in between."""
    if bad <= good:
        return float(d <= good)
    if d <= good:
        return 1.0
    if d >= bad:
        return 0.0
    return float(1.0 - (d - good) / (bad - good))


def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Per-element Huber loss (smooth_l1) toward zero."""
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta, reduction="none")


class Toy2DEnv:
    """Simple 2D point-mass with velocity and friction."""
    def __init__(self, dt=0.1, friction=0.97):
        self.dt = float(dt)
        self.friction = float(friction)
        self.reset()

    def reset(self):
        self.s = np.zeros(4, dtype=np.float32)  # [x,y,vx,vy]
        self.s[0] = 0.0
        self.s[1] = 0.0
        self.s[2] = 1.0
        self.s[3] = 0.5
        return self.s.copy()

    def step(self, a: np.ndarray):
        x, y, vx, vy = self.s
        ax, ay = float(a[0]), float(a[1])
        dt = self.dt

        vx_next = self.friction * vx + ax * dt
        vy_next = self.friction * vy + ay * dt
        x_next = x + vx_next * dt
        y_next = y + vy_next * dt

        self.s = np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)
        return self.s.copy()


def sample_action(t: int) -> np.ndarray:
    ax = 0.7 * math.sin(0.07 * t) + 0.15 * math.sin(0.011 * t)
    ay = 0.6 * math.cos(0.05 * t) + 0.10 * math.sin(0.017 * t)
    return np.array([ax, ay], dtype=np.float32)


class MeasurementChannel:
    """
    Returns delayed measurement (mxy, mask) or None.
    Also returns flags for dropout/outlier for logging.

    observe(true_pos_xy) -> (meas_or_none, drop_flag, out_flag)
      meas_or_none is None OR (mxy, mask)
      where mxy is (2,) float32 and mask is (2,) float32 in {0,1}
    """
    def __init__(
        self,
        noise_std=0.05,
        p_drop=0.10,         # B-1 default ON
        p_out=0.03,          # B-1 default ON
        outlier_scale=20.0,  # B-1: outlier magnitude multiplier on noise_std
        partial_mode="none", # none|x|y|random
        delay_k=0,
        seed=0,
    ):
        self.noise_std = float(noise_std)
        self.p_drop = float(p_drop)
        self.p_out = float(p_out)
        self.outlier_scale = float(outlier_scale)
        self.partial_mode = str(partial_mode)
        self.delay_k = int(delay_k)
        self.rng = np.random.RandomState(seed)

        # Buffer stores tuples: (meas_now_or_none, drop_flag, out_flag)
        self.delay_buf = deque(maxlen=max(1, self.delay_k + 1))
        for _ in range(self.delay_k + 1):
            self.delay_buf.append((None, 0, 0))

    def _make_partial_mask(self) -> np.ndarray:
        if self.partial_mode == "none":
            return np.array([1.0, 1.0], dtype=np.float32)
        if self.partial_mode == "x":
            return np.array([1.0, 0.0], dtype=np.float32)
        if self.partial_mode == "y":
            return np.array([0.0, 1.0], dtype=np.float32)
        if self.partial_mode == "random":
            return np.array([1.0, 0.0], dtype=np.float32) if self.rng.rand() < 0.5 else np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 1.0], dtype=np.float32)

    def observe(self, true_pos_xy: np.ndarray):
        drop_flag = 0
        out_flag = 0

        if self.rng.rand() < self.p_drop:
            meas_now = None
            drop_flag = 1
        else:
            noise = (self.noise_std * self.rng.randn(2)).astype(np.float32)
            meas = true_pos_xy.astype(np.float32) + noise

            if self.rng.rand() < self.p_out:
                meas = meas + (self.outlier_scale * self.noise_std) * self.rng.randn(2).astype(np.float32)
                out_flag = 1

            mask = self._make_partial_mask()
            meas_now = (meas, mask)

        self.delay_buf.append((meas_now, drop_flag, out_flag))
        return self.delay_buf[0]


class MeasHead(nn.Module):
    """Measurement head: feat -> (mu_xy, log_sigma_xy)."""
    def __init__(self, hidden=64, log_sigma_min=-2.5, log_sigma_max=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),  # [mx, my, maskx, masky]
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.log_sigma = nn.Linear(hidden, 2)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def forward(self, feat: torch.Tensor):
        h = self.net(feat)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)
        return mu, log_sigma


class ResidualNet(nn.Module):
    """Residual: [innov_xy, log_sigma_xy, mask_xy] -> delta_xy."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)

    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.97)

    ap.add_argument("--meas_noise_std", type=float, default=0.05)

    # B-1: defaults ON (garbage)
    ap.add_argument("--p_drop", type=float, default=0.10)
    ap.add_argument("--p_out", type=float, default=0.03)
    ap.add_argument("--outlier_scale", type=float, default=20.0)
    ap.add_argument("--partial_mode", type=str, default="none", choices=["none", "x", "y", "random"])
    ap.add_argument("--delay_k", type=int, default=0)

    ap.add_argument("--head_warmup_steps", type=int, default=200)
    ap.add_argument("--head_train_steps_per_env_step", type=int, default=2)
    ap.add_argument("--innov_clip", type=float, default=2.0)

    ap.add_argument("--gate_distance_good", type=float, default=1.5)
    ap.add_argument("--gate_distance_bad", type=float, default=6.0)

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_residual", type=float, default=1e-3)

    ap.add_argument("--huber_k", type=float, default=1.0)

    ap.add_argument("--sigma_floor", type=float, default=0.08)
    ap.add_argument("--log_sigma_min", type=float, default=-2.5)
    ap.add_argument("--log_sigma_max", type=float, default=3.0)

    # B-2: sigma regularizer
    ap.add_argument("--sigma_reg", type=float, default=0.002)

    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_select()
    print(f"[D55-B] Device: {dev}")

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

    head = MeasHead(hidden=64, log_sigma_min=args.log_sigma_min, log_sigma_max=args.log_sigma_max).to(dev)
    resnet = ResidualNet(hidden=64).to(dev)
    opt_head = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    opt_res = torch.optim.Adam(resnet.parameters(), lr=args.lr_residual)

    s = env.reset()
    pos_bel = s[:2].copy()
    vel_bel = s[2:].copy()

    accepted_ema = 0.0
    w_ema = 0.0
    out_ema = 0.0
    drop_ema = 0.0

    for t in range(args.steps):
        a = sample_action(t)
        s_true = env.step(a)
        pos_true = s_true[:2]

        # Predictor (same dynamics as env)
        vel_pred = args.friction * vel_bel + a * args.dt
        pos_pred = pos_bel + vel_pred * args.dt

        (m, drop_flag, out_flag) = meas.observe(pos_true)
        if m is None:
            mx, my = 0.0, 0.0
            mask = np.array([0.0, 0.0], dtype=np.float32)
        else:
            (mxy, mask) = m
            mx, my = float(mxy[0]), float(mxy[1])

        feat = torch.tensor([[mx, my, mask[0], mask[1]]], dtype=torch.float32, device=dev)

        # Head inference (interpreted measurement + uncertainty)
        mu, log_sigma = head(feat)
        sigma = torch.exp(log_sigma) + float(args.sigma_floor)
        xy_hat = mu[0].detach().cpu().numpy()

        # Innovation
        innov = xy_hat - pos_pred

        # Effective sigma for dn (ignore missing dims)
        sigma_np = sigma[0].detach().cpu().numpy()
        sigma_eff = sigma_np.copy()
        for i in range(2):
            if mask[i] < 0.5:
                sigma_eff[i] = 1e6

        # Normalized innovation distance (diagonal Mahalanobis distance)
        dn = float(math.sqrt((innov[0] / sigma_eff[0]) ** 2 + (innov[1] / sigma_eff[1]) ** 2))


        # Warmup disables trust until head is trained a bit
        if t < args.head_warmup_steps:
            w = 0.0
        else:
            # B-FIX: if measurement is missing (dropout / no observed dims), do NOT correct
            if mask[0] < 0.5 and mask[1] < 0.5:
                w = 0.0
            else:
                w = soft_gate(dn, args.gate_distance_good, args.gate_distance_bad)


        # # Warmup disables trust until head is trained a bit
        # if t < args.head_warmup_steps:
        #     w = 0.0
        # else:
        #     w = soft_gate(dn, args.gate_distance_good, args.gate_distance_bad)

        # Residual correction proposal
        innov_t = torch.tensor([[innov[0], innov[1]]], dtype=torch.float32, device=dev)
        mask_t = torch.tensor([[mask[0], mask[1]]], dtype=torch.float32, device=dev)
        res_in = torch.cat([innov_t, log_sigma.detach(), mask_t], dim=1)
        delta_hat = resnet(res_in)[0].detach().cpu().numpy()

        # Belief update
        pos_corr = pos_pred + w * delta_hat
        pos_bel = pos_corr
        vel_bel = vel_pred.copy()

        # Online learning (K gradient steps per env step)
        head_loss_val = 0.0
        res_loss_val = 0.0

        for _ in range(args.head_train_steps_per_env_step):
            # --- train head ---
            mu2, log_sigma2 = head(feat)
            sigma2 = torch.exp(log_sigma2) + float(args.sigma_floor)

            y = torch.tensor([[pos_true[0], pos_true[1]]], dtype=torch.float32, device=dev)
            mask2 = mask_t

            err = (y - mu2)

            # Gaussian NLL (diagonal)
            nll = (err * err) / (2.0 * sigma2 * sigma2) + torch.log(sigma2)
            nll = (nll * mask2).mean()

            # B-2: sigma regularizer (prevents sigma collapsing too aggressively)
            sigma_reg = float(args.sigma_reg) * (1.0 / sigma2).mean()

            head_loss = nll + sigma_reg

            opt_head.zero_grad(set_to_none=True)
            head_loss.backward()
            opt_head.step()
            head_loss_val += float(head_loss.detach().cpu().item())

            # --- train residual ---
            # Target is innovation only (uses current head output mu2; D54 rule preserved)
            with torch.no_grad():
                xy_hat2 = mu2[0].detach().cpu().numpy()
            innov2 = xy_hat2 - pos_pred
            innov2 = np.clip(innov2, -args.innov_clip, args.innov_clip)

            innov2_t = torch.tensor([[innov2[0], innov2[1]]], dtype=torch.float32, device=dev)
            res_in2 = torch.cat([innov2_t, log_sigma2.detach(), mask2], dim=1)
            delta2 = resnet(res_in2)

            diff = delta2 - innov2_t
            per = huber(diff, delta=args.huber_k)
            per = (per * mask2).mean()

            opt_res.zero_grad(set_to_none=True)
            per.backward()
            opt_res.step()
            res_loss_val += float(per.detach().cpu().item())

        head_loss_val /= max(1, args.head_train_steps_per_env_step)
        res_loss_val /= max(1, args.head_train_steps_per_env_step)

        # Diagnostics (sim-only)
        pos_err = float(np.linalg.norm(pos_bel - pos_true))
        d_state = float(np.linalg.norm(xy_hat - pos_pred))

        accepted = 1.0 if w > 0.1 else 0.0
        accepted_ema = 0.98 * accepted_ema + 0.02 * accepted
        w_ema = 0.98 * w_ema + 0.02 * w
        out_ema = 0.98 * out_ema + 0.02 * float(out_flag)
        drop_ema = 0.98 * drop_ema + 0.02 * float(drop_flag)

        if (t % args.print_every) == 0:
            sig_mean = float(np.mean(sigma_np))
            print(
                f"[D55-B][t={t:04d}] "
                f"res_loss={res_loss_val:.5f}  head_loss={head_loss_val:.5f}  "
                f"d=|xy_hat-pos_pred|={d_state:.3f}  dn={dn:.3f}  w={w:.2f}  "
                f"|pos_err|={pos_err:.3f}  sigma_mean={sig_mean:.3f}  "
                f"drop={int(drop_flag)} out={int(out_flag)} "
                f"acc%~={100.0*accepted_ema:.1f} w_ema={w_ema:.2f} "
                f"drop%~={100.0*drop_ema:.1f} out%~={100.0*out_ema:.1f}"
            )

    print("[D55-B] Done.")

if __name__ == "__main__":
    main()


