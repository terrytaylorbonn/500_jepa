# 49_D54_measurement_only_residual_FINAL.py
import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoProcessor

# -------------------------------------------------
# D54 FINAL: Measurement-only residual learning (innovation-supervised)
#
# Key properties:
# - ResidualNet is trained from innovation e = pos_meas - pos_pred (NO s_true in residual loss).
# - Belief update uses pos_meas (NO s_true fallback).
# - Trust/gating uses ONLY self-consistency: compare xy_hat vs pos_pred (NO s_true).
#
# Remaining demo scaffold:
# - The head (JEPA->XY) is trained supervised from replay using s_true[:2] labels. (cheat2)
#   This is intentional: D54 focuses on removing cheat1 (residual supervision).
# -------------------------------------------------


DT = 0.1


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    steps: int = 300
    log_interval: int = 10

    # World / rendering
    world_size: float = 5.0
    img_size: int = 128
    robot_radius: int = 4
    goal_radius: int = 4

    # JEPA
    jepa_model_id: str = "facebook/ijepa_vith14_1k"

    # Belief update gain
    meas_gain_pos: float = 0.6

    # Residual learning
    lr_residual: float = 1e-3
    vel_reg: float = 0.01  # keep vx/vy residual small (stability)

    # Head learning (JEPA -> XY)
    lr_head: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 64
    head_train_steps_per_env_step: int = 2
    head_warmup_steps: int = 200
    store_buffer_on_cpu: bool = True

    # Self-consistency trust gate (NO s_true)
    # We judge measurement by how far it is from the model prediction:
    #   d = ||xy_hat - pos_pred||
    # then compute a weight w in [0..1] used to blend:
    #   pos_meas = (1-w)*pos_pred + w*xy_hat
    #
    # gate_distance_good: if d <= good => w ~ 1 (trust measurement)
    # gate_distance_bad : if d >= bad  => w ~ 0 (ignore measurement)
    gate_distance_good: float = 0.5
    gate_distance_bad: float = 2.0

    # Optional clamp on innovation to prevent huge updates
    innov_clip: float = 2.0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Dynamics
# -------------------------
def f_true(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """True simulator dynamics."""
    x, y, vx, vy = torch.unbind(s, dim=-1)
    ax, ay = torch.unbind(a, dim=-1)
    friction = 0.93
    vx_next = friction * vx + 0.5 * ax * DT
    vy_next = friction * vy + 0.5 * ay * DT
    x_next = x + vx_next * DT
    y_next = y + vy_next * DT
    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


def f_prior(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Imperfect prior dynamics."""
    x, y, vx, vy = torch.unbind(s, dim=-1)
    ax, ay = torch.unbind(a, dim=-1)
    friction = 0.98
    vx_next = friction * vx + 0.5 * ax * DT
    vy_next = friction * vy + 0.5 * ay * DT
    x_next = x + vx_next * DT
    y_next = y + vy_next * DT
    return torch.stack([x_next, y_next, vx_next, vy_next], dim=-1)


# -------------------------
# Residual dynamics net
# -------------------------
class ResidualNet(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden_dim=64):
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


# -------------------------
# JEPA latent -> XY head
# -------------------------
class JepaToXY(nn.Module):
    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -------------------------
# Replay buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, z_dim: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.z_dim = int(z_dim)
        self.z = torch.empty((self.capacity, self.z_dim), dtype=torch.float32, device=device)
        self.xy = torch.empty((self.capacity, 2), dtype=torch.float32, device=device)
        self.size = 0
        self.ptr = 0

    def add(self, z: torch.Tensor, xy: torch.Tensor):
        if z.ndim == 2:
            z = z[0]
        if xy.ndim == 2:
            xy = xy[0]
        self.z[self.ptr].copy_(z)
        self.xy[self.ptr].copy_(xy)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        bs = min(int(batch_size), self.size)
        idx = torch.randint(0, self.size, (bs,), device=self.device)
        return self.z[idx], self.xy[idx]


# -------------------------
# Action generator
# -------------------------
def generate_action(t: int) -> np.ndarray:
    angle = 0.02 * t
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


# -------------------------
# Rendering: state -> image
# -------------------------
def state_to_image(
    s: torch.Tensor,
    goal_xy: torch.Tensor,
    world_size: float,
    img_size: int,
    robot_radius: int,
    goal_radius: int,
) -> Image.Image:
    x, y = s[0, 0].item(), s[0, 1].item()
    gx, gy = goal_xy[0, 0].item(), goal_xy[0, 1].item()

    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    def world_to_pixel(wx, wy):
        u = int((wx + world_size) / (2 * world_size) * (img_size - 1))
        v = int((wy + world_size) / (2 * world_size) * (img_size - 1))
        v = img_size - 1 - v
        return u, v

    gx_pix, gy_pix = world_to_pixel(gx, gy)
    draw.ellipse(
        [gx_pix - goal_radius, gy_pix - goal_radius, gx_pix + goal_radius, gy_pix + goal_radius],
        fill=(255, 0, 0),
    )

    rx_pix, ry_pix = world_to_pixel(x, y)
    draw.ellipse(
        [rx_pix - robot_radius, ry_pix - robot_radius, rx_pix + robot_radius, ry_pix + robot_radius],
        fill=(0, 128, 255),
    )
    return img


# -------------------------
# JEPA encoder
# -------------------------
class JepaEncoder(nn.Module):
    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def embed(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        inputs = self.processor(img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]


# -------------------------
# Trust gate (self-consistency)
# -------------------------
def gate_weight(d: float, good: float, bad: float) -> float:
    """
    d = ||xy_hat - pos_pred||
    returns w in [0,1].
    - if d <= good -> w=1 (trust measurement)
    - if d >= bad  -> w=0 (ignore measurement)
    - else linear ramp between
    """
    if d <= good:
        return 1.0
    if d >= bad:
        return 0.0
    # linear interpolation
    return float((bad - d) / (bad - good))


# -------------------------
# Main demo loop
# -------------------------
def run_demo(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[D54-FINAL] Device: {device}")
    print(f"[D54-FINAL] Loading JEPA model: {cfg.jepa_model_id}")
    jepa = JepaEncoder(cfg.jepa_model_id, device)
    print("[D54-FINAL] JEPA loaded.\n")

    residual_net = ResidualNet().to(device)
    opt_res = optim.Adam(residual_net.parameters(), lr=cfg.lr_residual)

    # init true + belief
    s_true = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    b = s_true.clone().detach()

    goal_xy = torch.tensor([[2.0, 2.0]], device=device)
    mse = nn.MSELoss()

    # z_dim
    with torch.no_grad():
        img0 = state_to_image(s_true, goal_xy, cfg.world_size, cfg.img_size, cfg.robot_radius, cfg.goal_radius)
        z0 = jepa.embed(img0, device)
        z_dim = int(z0.shape[-1])

    head = JepaToXY(z_dim).to(device)
    opt_head = optim.Adam(head.parameters(), lr=cfg.lr_head)

    buffer_device = torch.device("cpu") if cfg.store_buffer_on_cpu else device
    replay = ReplayBuffer(cfg.buffer_size, z_dim=z_dim, device=buffer_device)

    # logs
    res_losses, head_losses, pos_errs, jepa_losses = [], [], [], []
    gate_ds, gate_ws = [], []

    print("[D54-FINAL] Starting measurement-only residual learning demo")
    print(f"  steps                        = {cfg.steps}")
    print(f"  lr_residual                   = {cfg.lr_residual}")
    print(f"  lr_head                       = {cfg.lr_head}")
    print(f"  meas_gain_pos                 = {cfg.meas_gain_pos}")
    print(f"  buffer_size                   = {cfg.buffer_size}")
    print(f"  batch_size                    = {cfg.batch_size}")
    print(f"  head_train_steps_per_env_step = {cfg.head_train_steps_per_env_step}")
    print(f"  head_warmup_steps             = {cfg.head_warmup_steps}")
    print(f"  store_buffer_on_cpu           = {cfg.store_buffer_on_cpu}")
    print(f"  gate_distance_good            = {cfg.gate_distance_good}")
    print(f"  gate_distance_bad             = {cfg.gate_distance_bad}")
    print(f"  innov_clip                    = {cfg.innov_clip}")
    print(f"  vel_reg                       = {cfg.vel_reg}")
    print("")

    for t in range(cfg.steps):
        # S1) action + true env step
        a = torch.tensor(generate_action(t), device=device).unsqueeze(0)
        s_true = f_true(s_true, a)

        # S2) prediction
        with torch.no_grad():
            s_prior = f_prior(b, a)
        r = residual_net(b, a)              # (1,4)
        s_pred = s_prior + r                # (1,4)

        # S3) sensor latent
        with torch.no_grad():
            img_true = state_to_image(s_true, goal_xy, cfg.world_size, cfg.img_size, cfg.robot_radius, cfg.goal_radius)
            img_pred = state_to_image(s_pred.detach(), goal_xy, cfg.world_size, cfg.img_size, cfg.robot_radius, cfg.goal_radius)
            z_obs = jepa.embed(img_true, device)
            z_pred = jepa.embed(img_pred, device)
            jepa_loss = torch.mean((z_pred - z_obs) ** 2).item()

        # store replay (cheat2: labels from s_true)
        with torch.no_grad():
            z_store = z_obs.detach()[0].float()
            xy_store = s_true.detach()[0, :2].float()
            if cfg.store_buffer_on_cpu:
                z_store = z_store.cpu()
                xy_store = xy_store.cpu()
            replay.add(z_store, xy_store)

        # S4) train head from replay (after warmup buffer size)
        head_loss_val = 0.0
        if (t >= cfg.head_warmup_steps) and (replay.size >= max(cfg.batch_size, 8)):
            for _ in range(cfg.head_train_steps_per_env_step):
                zb, xyb = replay.sample(cfg.batch_size)
                if cfg.store_buffer_on_cpu:
                    zb = zb.to(device)
                    xyb = xyb.to(device)
                pred_xy = head(zb)
                head_loss = mse(pred_xy, xyb)
                opt_head.zero_grad()
                head_loss.backward()
                opt_head.step()
                head_loss_val = head_loss.item()
        else:
            # purely for a meaningful printed number during warmup
            with torch.no_grad():
                pred_xy = head(z_obs.detach())
                head_loss_val = mse(pred_xy, s_true[..., :2]).item()

        # S5) compute innovation from measurement (NO s_true)
        # - xy_hat from head
        # - compare xy_hat to pos_pred to get self-consistency distance d
        # - compute weight w(d)
        # - pos_meas = blend(pos_pred, xy_hat)
        # - innovation e = pos_meas - pos_pred
        # - train residual to predict e (measurement-only)
        xy_hat = head(z_obs.detach())        # (1,2)
        pos_pred = s_pred[0, :2]             # (2,)
        d = torch.norm(xy_hat[0] - pos_pred).item()
        # w = gate_weight(d, cfg.gate_distance_good, cfg.gate_distance_bad)

        # NEW: never trust measurement during warmup ###################
        if t < cfg.head_warmup_steps:
            w = 0.0
        else:
            w = gate_weight(d, cfg.gate_distance_good, cfg.gate_distance_bad)


        with torch.no_grad():
            pos_meas = (1.0 - w) * pos_pred + w * xy_hat[0]
            e = pos_meas - pos_pred
            if cfg.innov_clip > 0.0:
                e = torch.clamp(e, -cfg.innov_clip, cfg.innov_clip)

        # Train residual from innovation (NO s_true)
        target_e = e.detach().unsqueeze(0)       # (1,2)
        res_loss = mse(r[..., :2], target_e)
        if cfg.vel_reg > 0.0:
            res_loss = res_loss + float(cfg.vel_reg) * mse(r[..., 2:], torch.zeros_like(r[..., 2:]))

        opt_res.zero_grad()
        res_loss.backward()
        opt_res.step()

        # Belief update using same innovation e
        with torch.no_grad():
            b = s_pred.detach().clone()
            b[0, 0:1] += cfg.meas_gain_pos * e[0:1]
            b[0, 1:2] += cfg.meas_gain_pos * e[1:2]

            pos_err = torch.norm(b[0, :2] - s_true[0, :2]).item()

        # logs
        res_losses.append(res_loss.item())
        head_losses.append(head_loss_val)
        pos_errs.append(pos_err)
        jepa_losses.append(jepa_loss)
        gate_ds.append(d)
        gate_ws.append(w)

        if (t % cfg.log_interval) == 0 or t == cfg.steps - 1:
            warm_flag = "W" if (t < cfg.head_warmup_steps) else " "
            print(
                f"[D54-FINAL][t={t:03d}]{warm_flag} "
                f"res_loss={res_loss.item():.5f}  "
                f"head_loss={head_loss_val:.5f}  "
                f"d=|xy_hat-pos_pred|={d:.3f}  "
                f"w={w:.2f}  "
                f"|pos_err|={pos_err:.3f}  "
                f"jepa_loss={jepa_loss:.3e}  "
                f"buf={replay.size}"
            )

    print("\n[D54-FINAL] Finished.")
    print(f"  Mean res_loss  = {np.mean(res_losses):.6f}")
    print(f"  Mean head_loss = {np.mean(head_losses):.6f}")
    print(f"  Mean d         = {np.mean(gate_ds):.4f}")
    print(f"  Mean w         = {np.mean(gate_ws):.4f}")
    print(f"  Mean |pos_err| = {np.mean(pos_errs):.4f}")
    print(f"  Mean JEPA loss = {np.mean(jepa_losses):.6f}")


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="D54 FINAL: measurement-only residual learning (innovation-supervised)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--meas-gain-pos", type=float, default=0.6)

    p.add_argument("--lr-residual", type=float, default=1e-3)
    p.add_argument("--lr-head", type=float, default=1e-3)

    p.add_argument("--buffer-size", type=int, default=50_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--head-train-steps-per-env-step", type=int, default=2)
    p.add_argument("--head-warmup-steps", type=int, default=200)
    p.add_argument("--store-buffer-on-gpu", action="store_true")

    p.add_argument("--gate-distance-good", type=float, default=0.5)
    p.add_argument("--gate-distance-bad", type=float, default=2.0)
    p.add_argument("--innov-clip", type=float, default=2.0)
    p.add_argument("--vel-reg", type=float, default=0.01)

    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"

    cfg = Config(
        device=device,
        seed=args.seed,
        steps=args.steps,
        log_interval=args.log_interval,
        meas_gain_pos=args.meas_gain_pos,
        lr_residual=args.lr_residual,
        lr_head=args.lr_head,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        head_train_steps_per_env_step=args.head_train_steps_per_env_step,
        head_warmup_steps=args.head_warmup_steps,
        store_buffer_on_cpu=(not args.store_buffer_on_gpu),
        gate_distance_good=args.gate_distance_good,
        gate_distance_bad=args.gate_distance_bad,
        innov_clip=args.innov_clip,
        vel_reg=args.vel_reg,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    run_demo(cfg)
