#46_D52bbb_ijepa_latent_measurement_demo.py
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
# 52_D52: JEPA as measurement / sensor channel
#
# (Base D52)
# - Residual model trained in STATE space on position
# - JEPA used only for logging
#
# ###### D52 NEW:
# - Add small head: (JEPA z_obs) -> (x_hat, y_hat)
# - Train head online using true position supervision
# - Use (x_hat, y_hat) for measurement update (no more "pos_true cheat")
# -------------------------------------------------


DT = 0.1


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    steps: int = 300
    lr: float = 1e-3
    log_interval: int = 10

    # World / rendering
    world_size: float = 5.0         # coordinates in [-L, L]
    img_size: int = 128
    robot_radius: int = 4
    goal_radius: int = 4

    # JEPA
    jepa_model_id: str = "facebook/ijepa_vith14_1k"

    # Measurement update gain (in state space)
    meas_gain_pos: float = 0.2      # how much we trust “measurement” vs prediction

    ###### D52 NEW
    lr_head: float = 1e-3           # learning rate for JEPA->XY head
    ##################


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
    """True dynamics (unknown to learner)."""
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

    friction = 0.98  # slightly wrong
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


###### D52 NEW
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
##################


# -------------------------
# Action generator
# -------------------------
def generate_action(t: int) -> np.ndarray:
    angle = 0.02 * t
    ax = math.cos(angle)
    ay = math.sin(angle)
    return np.array([ax, ay], dtype=np.float32)


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
    """
    Render a simple top-down view:
    - robot at (x, y) as blue circle
    - goal at (gx, gy) as red circle
    s: (1, 4)
    goal_xy: (1, 2)
    """
    x, y = s[0, 0].item(), s[0, 1].item()
    gx, gy = goal_xy[0, 0].item(), goal_xy[0, 1].item()

    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    def world_to_pixel(wx, wy):
        # map [-world_size, world_size] -> [0, img_size-1]
        u = int((wx + world_size) / (2 * world_size) * (img_size - 1))
        v = int((wy + world_size) / (2 * world_size) * (img_size - 1))
        # flip y-axis
        v = img_size - 1 - v
        return u, v

    # Draw goal
    gx_pix, gy_pix = world_to_pixel(gx, gy)
    draw.ellipse(
        [
            gx_pix - goal_radius,
            gy_pix - goal_radius,
            gx_pix + goal_radius,
            gy_pix + goal_radius,
        ],
        fill=(255, 0, 0),
    )

    # Draw robot
    rx_pix, ry_pix = world_to_pixel(x, y)
    draw.ellipse(
        [
            rx_pix - robot_radius,
            ry_pix - robot_radius,
            rx_pix + robot_radius,
            ry_pix + robot_radius,
        ],
        fill=(0, 128, 255),
    )

    return img


# -------------------------
# JEPA embedding
# -------------------------
class JepaEncoder(nn.Module):
    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def embed(self, img: Image.Image, device: torch.device) -> torch.Tensor:
        """
        Returns CLS embedding: (1, D)
        """
        inputs = self.processor(img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # For I-JEPA ViT, last_hidden_state[:, 0, :] is CLS
        z = outputs.last_hidden_state[:, 0, :]
        return z


# -------------------------
# Main demo loop
# -------------------------
def run_demo(cfg: Config):
    device = torch.device(cfg.device)
    print(f"[D52] Device: {device}")
    print(f"[D52] Loading JEPA model: {cfg.jepa_model_id}")
    jepa = JepaEncoder(cfg.jepa_model_id, device)
    print("[D52] JEPA loaded.")

    residual_net = ResidualNet().to(device)
    optimizer = optim.Adam(residual_net.parameters(), lr=cfg.lr)

    # Initial true state and belief
    s_true = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    b = s_true.clone().detach()

    # Fixed goal in world
    goal_xy = torch.tensor([[2.0, 2.0]], device=device)

    state_losses = []
    pos_errs = []
    jepa_losses = []

    ###### D52 NEW
    head_losses = []
    meas_errs = []   # |(x_hat,y_hat) - true_pos|
    ##################

    print("")
    print("[D52] Starting JEPA-measurement world-model demo (state-supervised)")
    print(f"  steps         = {cfg.steps}")
    print(f"  lr            = {cfg.lr}")
    print(f"  meas_gain_pos = {cfg.meas_gain_pos}")
    ###### D52 NEW
    print(f"  lr_head       = {cfg.lr_head}")
    ##################
    print("")

    mse = nn.MSELoss()

    ###### D52 NEW
    # Build JEPA->XY head after we know z_dim
    with torch.no_grad():
        img0 = state_to_image(
            s_true, goal_xy, cfg.world_size, cfg.img_size,
            cfg.robot_radius, cfg.goal_radius
        )
        z0 = jepa.embed(img0, device)
        z_dim = z0.shape[-1]

    head = JepaToXY(z_dim).to(device)
    opt_head = optim.Adam(head.parameters(), lr=cfg.lr_head)
    ##################

    for t in range(cfg.steps):
        # 1) Action
        a_np = generate_action(t)
        a = torch.tensor(a_np, device=device).unsqueeze(0)  # (1, 2)

        # 2) True env step
        s_true = f_true(s_true, a)

        # 3) Prediction: prior + residual on belief
        with torch.no_grad():
            s_prior = f_prior(b, a)
        s_prior.requires_grad_(False)

        s_pred = s_prior + residual_net(b, a)  # (1, 4)

        # 4) Train residual in STATE space: match true position
        state_loss = mse(s_pred[..., :2], s_true[..., :2])

        optimizer.zero_grad()
        state_loss.backward()
        optimizer.step()

        # 5) JEPA measurements (no grad) – for logging
        with torch.no_grad():
            img_true = state_to_image(
                s_true, goal_xy, cfg.world_size, cfg.img_size,
                cfg.robot_radius, cfg.goal_radius
            )
            img_pred = state_to_image(
                s_pred.detach(), goal_xy, cfg.world_size, cfg.img_size,
                cfg.robot_radius, cfg.goal_radius
            )
            z_obs = jepa.embed(img_true, device)
            z_pred = jepa.embed(img_pred, device)
            jepa_loss = torch.mean((z_pred - z_obs) ** 2).item()

        ###### D52 NEW
        # 5b) Train JEPA->XY head (this is the "vision sensor")
        # NOTE: z_obs is from JEPA, computed under no_grad above (frozen JEPA).
        # We DO allow gradients into the head, so we feed z_obs as a detached tensor.
        z_obs_detached = z_obs.detach()
        xy_hat = head(z_obs_detached)  # (1,2)

        head_loss = mse(xy_hat, s_true[..., :2])

        opt_head.zero_grad()
        head_loss.backward()
        opt_head.step()

        with torch.no_grad():
            meas_err = torch.norm(xy_hat.detach()[0] - s_true[0, :2]).item()
        ##################

        # 6) Measurement update (in state space)
        #    Here we still "cheat" a bit by using true position to correct belief.
        # with torch.no_grad():
        #     pos_true = s_true[0, :2]
        #     pos_pred = s_pred[0, :2]
        #     pos_diff = pos_true - pos_pred  # (2,)
        #     b = s_pred.detach().clone()
        #     b[0, 0:1] += cfg.meas_gain_pos * pos_diff[0:1]
        #     b[0, 1:2] += cfg.meas_gain_pos * pos_diff[1:2]
        #     # velocities unchanged

        ###### D52 NEW
        # 6) Measurement update using JEPA-decoded position (no more pos_true cheat)
        with torch.no_grad():
            pos_pred = s_pred[0, :2]
            pos_meas = xy_hat.detach()[0]   # (2,) from JEPA head
            e = (pos_meas - pos_pred)       # innovation in XY

            b = s_pred.detach().clone()
            b[0, 0:1] += cfg.meas_gain_pos * e[0:1]
            b[0, 1:2] += cfg.meas_gain_pos * e[1:2]
            # velocities unchanged
        ##################

        # 7) Logging
        state_losses.append(state_loss.item())
        ###### D52 NEW
        head_losses.append(head_loss.item())
        meas_errs.append(meas_err)
        ##################
        # pos_true used only for metrics now (not for updating belief)
        pos_true = s_true[0, :2]
        pos_err = torch.norm(pos_true - b[0, :2]).item()
        pos_errs.append(pos_err)
        jepa_losses.append(jepa_loss)

        if (t % cfg.log_interval) == 0 or t == cfg.steps - 1:
            ###### D52 NEW
            print(
                f"[D52][t={t:03d}] "
                f"state_loss={state_loss.item():.5f}  "
                f"head_loss={head_loss.item():.5f}  "
                f"|meas_err|={meas_err:.3f}  "
                f"|pos_err|={pos_err:.3f}  "
                f"jepa_loss={jepa_loss:.3e}"
            )
            ##################
            # OLD print kept for reference
            # print(
            #     f"[D52][t={t:03d}] "
            #     f"state_loss={state_loss.item():.5f}  "
            #     f"|pos_err|={pos_err:.3f}  "
            #     f"jepa_loss={jepa_loss:.5f}"
            # )

    print("")
    print("[D52] Finished.")
    print(f"  Mean state_loss = {np.mean(state_losses):.6f}")
    ###### D52 NEW
    print(f"  Mean head_loss  = {np.mean(head_losses):.6f}")
    print(f"  Mean |meas_err| = {np.mean(meas_errs):.4f}")
    ##################
    print(f"  Mean |pos_err|  = {np.mean(pos_errs):.4f}")
    print(f"  Mean JEPA loss  = {np.mean(jepa_losses):.6f}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="D52: JEPA as measurement channel for a simple world model (state-supervised)"
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--meas-gain-pos", type=float, default=0.2)
    ###### D52 NEW
    parser.add_argument("--lr-head", type=float, default=1e-3)
    ##################
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda"

    cfg = Config(
        device=device,
        seed=args.seed,
        steps=args.steps,
        lr=args.lr,
        meas_gain_pos=args.meas_gain_pos,
        ###### D52 NEW
        lr_head=args.lr_head,
        ##################
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    run_demo(cfg)
