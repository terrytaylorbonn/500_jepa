"""
52_D55C_confidence_measurement_head_C.py

D55 VERSION C — "remove cheat3 for uncertainty"
Core idea:
  - We do NOT use pos_true to train sigma (uncertainty).
  - Sigma is trained from SELF-CONSISTENCY / HISTORY:
      sigma should match the typical innovation magnitude (mu - pos_pred) over time.
  - Gate uses dn = normalized innovation distance (diagonal Mahalanobis using sigma).

What is still supervised (because this is still a toy sim demo):
  - mu (the cleaned measurement estimate) is trained toward pos_true (so the head learns denoising).
What is NOT supervised anymore:
  - sigma (uncertainty) is NOT trained with NLL to pos_true.

DIFFS vs D55-B:
C-1) Replace NLL head loss with:
      head_loss = MSE(mu, pos_true)  +  sigma_self_w * SmoothL1(log_sigma, log(target_sigma_from_innov_ema))
C-2) target_sigma is computed from an EMA of innovation^2 (no pos_true)
C-3) Gate uses dn built from sigma, so learned sigma affects trust without "cheat" signals.
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


def soft_gate(d, good, bad):
    # linear ramp: 1 -> 0 between good..bad
    if bad <= good:
        return float(d <= good)
    if d <= good:
        return 1.0
    if d >= bad:
        return 0.0
    return float(1.0 - (d - good) / (bad - good))


def huber(x, delta=1.0):
    # elementwise smooth L1 vs 0
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta, reduction="none")


class Toy2DEnv:
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

        vx_next = self.friction * vx + ax * dt
        vy_next = self.friction * vy + ay * dt
        x_next = x + vx_next * dt
        y_next = y + vy_next * dt

        self.s = np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)
        return self.s.copy()


def sample_action(t):
    ax = 0.7 * math.sin(0.07 * t) + 0.15 * math.sin(0.011 * t)
    ay = 0.6 * math.cos(0.05 * t) + 0.10 * math.sin(0.017 * t)
    return np.array([ax, ay], dtype=np.float32)


class MeasurementChannel:
    """
    Returns delayed measurement (mxy, mask) or None.
    Also returns flags for dropout/outlier for logging only.
    """
    def __init__(
        self,
        noise_std=0.05,
        p_drop=0.10,
        p_out=0.03,
        outlier_scale=20.0,
        partial_mode="none",
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
        for _ in range(self.delay_k + 1):
            self.delay_buf.append((None, 0, 0))

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
        drop_flag = 0
        out_flag = 0

        if self.rng.rand() < self.p_drop:
            meas_now = None
            drop_flag = 1
        else:
            noise = self.noise_std * self.rng.randn(2).astype(np.float32)
            meas = true_pos_xy.astype(np.float32) + noise

            if self.rng.rand() < self.p_out:
                meas = meas + (self.outlier_scale * self.noise_std) * self.rng.randn(2).astype(np.float32)
                out_flag = 1

            mask = self._make_partial_mask()
            meas_now = (meas, mask)

        self.delay_buf.append((meas_now, drop_flag, out_flag))
        return self.delay_buf[0]


class MeasHead(nn.Module):
    def __init__(self, hidden=64, log_sigma_min=-2.5, log_sigma_max=3.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, 2)
        self.log_sigma = nn.Linear(hidden, 2)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)

    def forward(self, feat):
        h = self.net(feat)
        mu = self.mu(h)
        log_sigma = self.log_sigma(h)
        log_sigma = torch.clamp(log_sigma, min=self.log_sigma_min, max=self.log_sigma_max)
        return mu, log_sigma


class ResidualNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # input = [innov(2), log_sigma(2), mask(2)] = 6
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)

    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--friction", type=float, default=0.97)

    ap.add_argument("--meas_noise_std", type=float, default=0.05)

    # "real-world garbage" still ON by default
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

    # D55C core knobs: sigma learned from innovation history (no pos_true)
    ap.add_argument("--innov_ema_beta", type=float, default=0.995)   # closer to 1 => slower sigma adaptation
    ap.add_argument("--sigma_self_w", type=float, default=0.50)      # weight on sigma self-consistency loss
    ap.add_argument("--sigma_target_min", type=float, default=0.10)  # avoid absurdly tiny targets
    ap.add_argument("--sigma_target_max", type=float, default=10.0)  # avoid absurdly huge targets

    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    dev = device_select()
    print(f"[D55-C] Device: {dev}")

    print("\nAbbrev:")
    print("  r_l = res_loss")
    print("  h_l = head_loss")
    print("  d   = |xy_hat - pos_pred|")
    print("  dn  = normalized innovation distance")
    print("  w   = gate weight")
    print("  p_e = |pos_err|")
    print("  s_m = sigma_mean")
    print("  t_s = targ_sigma")
    print("  acc = accepted_ema (%)")
    print("  w_e = w_ema")
    print("  drp = drop_ema (%)")
    print("  out = out_ema (%)\n")


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

    # Innovation variance EMA (2D, per-axis). This is the "self-consistency / history" signal.
    innov_var_ema = np.array([1.0, 1.0], dtype=np.float32) * (args.sigma_target_min ** 2)

    accepted_ema = 0.0
    w_ema = 0.0
    out_ema = 0.0
    drop_ema = 0.0

    for t in range(args.steps):
        # 0a) world advances
        a = sample_action(t)
        s_true = env.step(a)
        pos_true = s_true[:2]

        # 0b) predictor (prior)
        vel_pred = args.friction * vel_bel + a * args.dt
        pos_pred = pos_bel + vel_pred * args.dt

        # 1) sensor
        (m, drop_flag, out_flag) = meas.observe(pos_true)
        if m is None:
            mx, my = 0.0, 0.0
            mask = np.array([0.0, 0.0], dtype=np.float32)
        else:
            (mxy, mask) = m
            mx, my = float(mxy[0]), float(mxy[1])

        # 2) feature
        feat = torch.tensor([[mx, my, mask[0], mask[1]]], dtype=torch.float32, device=dev)

        # 3) head predicts mu + sigma
        mu, log_sigma = head(feat)
        sigma = torch.exp(log_sigma) + float(args.sigma_floor)
        xy_hat = mu[0].detach().cpu().numpy()

# ===========================
# FIXED BLOCK (MARKED)
# Drop this into 52_D55C_confidence_measurement_head_C.py
# replacing your current section:
#   "effective sigma for gating"  -> through gate computation
# ===========================

        # innovation
        innov = xy_hat - pos_pred

        # effective sigma for gating (mask -> ignore missing axis)
        sigma_np = sigma[0].detach().cpu().numpy()

        # ---------------------------------------------------------
        # FIX #1: Compute dn only over PRESENT axes (cleaner + correct)
        #         (Avoid "dn dilution" via sigma_eff=1e6 trick)
        # FIX #2: If BOTH axes missing, force w=0 and dn=1e9
        # ---------------------------------------------------------
        mask_sum = float(mask[0] + mask[1])
        if mask_sum < 0.5:
            dn = 1e9
            w = 0.0
        else:
            dn2 = 0.0
            if mask[0] > 0.5:
                dn2 += float((innov[0] / sigma_np[0]) ** 2)
            if mask[1] > 0.5:
                dn2 += float((innov[1] / sigma_np[1]) ** 2)
            dn = float(math.sqrt(dn2))

            # warmup disables trust until head stabilizes
            w = 0.0 if t < args.head_warmup_steps else soft_gate(dn, args.gate_distance_good, args.gate_distance_bad)

        # FIX: define dn_str right here, after dn is final
        dn_str = "NAxxx" if mask_sum < 0.5 else f"{dn:.3f}"

        # residual correction proposal (unchanged)
        innov_t = torch.tensor([[innov[0], innov[1]]], dtype=torch.float32, device=dev)
        mask_t = torch.tensor([[mask[0], mask[1]]], dtype=torch.float32, device=dev)
        res_in = torch.cat([innov_t, log_sigma.detach(), mask_t], dim=1)
        delta_hat = resnet(res_in)[0].detach().cpu().numpy()

        # 5) commit belief
        pos_corr = pos_pred + w * delta_hat
        pos_bel = pos_corr
        vel_bel = vel_pred.copy()

        # 6) online learning
        head_loss_val = 0.0
        res_loss_val = 0.0

        for _ in range(args.head_train_steps_per_env_step):
            mu2, log_sigma2 = head(feat)
            sigma2 = torch.exp(log_sigma2) + float(args.sigma_floor)

            # --- Head loss for mu (supervised denoising) ---
            y = torch.tensor([[pos_true[0], pos_true[1]]], dtype=torch.float32, device=dev)
            mask2 = mask_t
            err_mu = (y - mu2)
            mse_mu = (err_mu * err_mu) * mask2
            mse_mu = mse_mu.mean()

            # --- D55C: sigma trained from self-consistency / history (NO pos_true) ---
            # innovation for this head forward pass
            with torch.no_grad():
                xy_hat2 = mu2[0].detach().cpu().numpy()
            innov2 = xy_hat2 - pos_pred

            # update EMA of innovation^2 using only available axes
            # (this is deliberately "online stats" style, not backprop)
            beta = float(args.innov_ema_beta)
            for i in range(2):
                if mask[i] > 0.5:
                    innov_var_ema[i] = beta * innov_var_ema[i] + (1.0 - beta) * float(innov2[i] * innov2[i])

            # target sigma from EMA variance (clamped), plus floor for stability
            sigma_target_np = np.sqrt(np.maximum(innov_var_ema, 1e-12)).astype(np.float32)
            sigma_target_np = np.clip(sigma_target_np, args.sigma_target_min, args.sigma_target_max)
            sigma_target_t = torch.tensor([[sigma_target_np[0], sigma_target_np[1]]], dtype=torch.float32, device=dev)

            # compare in log-space (more stable; also matches how we parameterize sigma)
            log_sigma_target = torch.log(sigma_target_t + 1e-8)

            # self-consistency loss only on present axes
            sigma_diff = (log_sigma2 - log_sigma_target) * mask2
            sigma_self = huber(sigma_diff, delta=0.5).mean()

            head_loss = mse_mu + float(args.sigma_self_w) * sigma_self

            opt_head.zero_grad(set_to_none=True)
            head_loss.backward()
            opt_head.step()
            head_loss_val += float(head_loss.detach().cpu().item())

            # --- Residual training (D54 rule preserved): target = innovation only ---
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

        # 7) diagnostics + smoothing
        pos_err = float(np.linalg.norm(pos_bel - pos_true))
        d_state = float(np.linalg.norm(xy_hat - pos_pred))

        accepted = 1.0 if w > 0.1 else 0.0
        accepted_ema = 0.98 * accepted_ema + 0.02 * accepted
        w_ema = 0.98 * w_ema + 0.02 * w
        out_ema = 0.98 * out_ema + 0.02 * float(out_flag)
        drop_ema = 0.98 * drop_ema + 0.02 * float(drop_flag)

        # 8) printing
        if (t % args.print_every) == 0:
            
# ===========================
# FIXED PRINT BLOCK (MARKED)
# Drop this into the printing section (t % print_every == 0)
# replacing your current targ_mean computation.
# ===========================

            sig_mean = float(np.mean(sigma_np))

            # ---------------------------------------------------------
            # FIX #3: Print t_s that MATCHES training target:
            #         use the CLIPPED sigma_target (same as training)
            # ---------------------------------------------------------
            sigma_target_np = np.sqrt(np.maximum(innov_var_ema, 1e-12)).astype(np.float32)
            sigma_target_np = np.clip(sigma_target_np, args.sigma_target_min, args.sigma_target_max)
            targ_mean = float(np.mean(sigma_target_np))

            # FIX: pretty-print dn when no measurement
            dn_str = "NA---" if mask_sum < 0.5 else f"{dn:.3f}"

            print(
                f"t={t:04d} "
                f"r_l={res_loss_val:.5f} "
                f"h_l={head_loss_val:.5f} "
                f"d={d_state:.3f} "
                f"dn={dn_str} "
                # f"dn={dn:.3f} "
                f"m={int(mask[0])}{int(mask[1])} " ######
                f"w={w:.2f} "
                f"p_e={pos_err:.3f} "
                f"s_m={sig_mean:.3f} "
                f"t_s={targ_mean:.3f} "
                f"acc={100.0*accepted_ema:.1f} "
                f"w_e={w_ema:.2f} "
                f"drp={100.0*drop_ema:.1f} "
                f"out={100.0*out_ema:.1f}"
            )

    print("[D55-C] Done.")


if __name__ == "__main__":
    main()


