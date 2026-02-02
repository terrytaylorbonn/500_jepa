#44_49c_latent_transition_autoencoder.py

import argparse
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------
# 44_49c: Latent transition with autoencoder
#
# Stage 1 (offline):
#   - Train encoder E(s) -> z and decoder D(z) -> s
#   - Reconstruction loss: ||s - D(E(s))||^2
#
# Stage 2 (online):
#   - Belief lives in latent space: z_b
#   - True env:       s_true(t+1) = f_true(s_true(t), a_t)
#   - Prior in latent:
#         s_b      = D(z_b)
#         s_prior  = f_prior(s_b, a_t)
#         z_prior  = E(s_prior)
#   - Residual latent dynamics:
#         z_pred   = z_prior + r_theta(z_b, a_t)
#   - Decode to state for measurement:
#         s_pred   = D(z_pred)
#         y_pred   = h_measure(s_pred)
#   - Measurement-only loss: ||y_obs - y_pred||^2
#   - Measurement update:
#         s_corr   = s_pred
#         s_corr.xy += K * (y_obs - y_pred)
#         z_b      = E(s_corr)
#
# This is a JEPA-style "latent world model" but using a
# toy autoencoder instead of a vision encoder.
# ---------------------------------------------


DT = 0.1


@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # State / latent dims
    state_dim: int = 4   # [x, y, vx, vy]
    action_dim: int = 2  # [ax, ay]
    latent_dim: int = 8

    # Offline AE training
    ae_samples: int = 5000
    ae_batch_size: int = 128
    ae_epochs: int = 20
    ae_lr: float = 1e-3

    # Online residual training
    steps: int = 400
    lr_residual: float = 1e-3
    meas_noise_std: float = 0.05
    meas_gain: float = 0.5
    log_interval: int = 20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------
# True and prior dynamics
# -------------------------
def f_true(s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    True environment dynamics (unknown to the model).
    s: (..., 4) [x, y, vx, vy]
    a: (..., 2) [ax, ay]
    """
    x, y, vx, vy = torch.unbind(s, dim=-1)
    ax, ay = torch.unbind(a, dim=-1)

    # Slightly stronger friction and small bias
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


# -------------------------
# Measurement model
# -------------------------
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


# -------------------------
# Networks
# -------------------------
class Encoder(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentResidualNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)
        return self.net(x)


# -------------------------
# Action generator
# -------------------------
def generate_action(t: int) -> np.ndarray:
    """
    Simple open-loop policy to excite dynamics:
    - Slowly rotating acceleration vector.
    """
    angle = 0.02 * t
    ax = math.cos(angle)
    ay = math.sin(angle)
    return np.array([ax, ay], dtype=np.float32)


# -------------------------
# Offline AE training
# -------------------------
def collect_ae_dataset(cfg: Config, device: torch.device) -> torch.Tensor:
    """
    Collect a dataset of true states by rolling out f_true with random-ish actions.
    Returns tensor of shape (N, state_dim).
    """
    s = torch.zeros((1, cfg.state_dim), device=device)  # start at origin
    states = []

    with torch.no_grad():
        for t in range(cfg.ae_samples):
            a_np = generate_action(t)
            a = torch.tensor(a_np, device=device).unsqueeze(0)
            s = f_true(s, a)
            states.append(s.detach().cpu().numpy()[0])

    states = np.stack(states, axis=0)
    return torch.tensor(states, dtype=torch.float32, device=device)


def train_autoencoder(cfg: Config, device: torch.device):
    print("[49c] Offline autoencoder training...")
    states = collect_ae_dataset(cfg, device)
    dataset = states  # (N, state_dim)

    encoder = Encoder(cfg.state_dim, cfg.latent_dim).to(device)
    decoder = Decoder(cfg.latent_dim, cfg.state_dim).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=cfg.ae_lr)
    loss_fn = nn.MSELoss()

    N = dataset.shape[0]
    num_batches = max(1, N // cfg.ae_batch_size)

    for epoch in range(cfg.ae_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for bi in range(num_batches):
            idx = perm[bi * cfg.ae_batch_size : (bi + 1) * cfg.ae_batch_size]
            batch = dataset[idx]  # (B, state_dim)

            z = encoder(batch)
            recon = decoder(z)
            loss = loss_fn(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= num_batches
        print(f"[49c][AE] epoch {epoch+1:02d}/{cfg.ae_epochs:02d}  recon_loss={epoch_loss:.6f}")

    # Final reconstruction error on whole dataset
    with torch.no_grad():
        z_all = encoder(dataset)
        recon_all = decoder(z_all)
        final_loss = loss_fn(recon_all, dataset).item()
    print(f"[49c][AE] Final reconstruction loss on dataset: {final_loss:.6f}")
    print("")

    return encoder, decoder


# -------------------------
# Online latent filter + residual loop
# -------------------------
def run_online_latent_demo(cfg: Config, device: torch.device, encoder: Encoder, decoder: Decoder):
    print("[49c] Online latent residual world model (measurement-only)")
    print(f"  device          = {device}")
    print(f"  steps           = {cfg.steps}")
    print(f"  meas_noise_std  = {cfg.meas_noise_std}")
    print(f"  meas_gain       = {cfg.meas_gain}")
    print(f"  lr_residual     = {cfg.lr_residual}")
    print("")

    # Freeze AE params
    encoder.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    residual_net = LatentResidualNet(cfg.latent_dim, cfg.action_dim).to(device)
    optimizer = optim.Adam(residual_net.parameters(), lr=cfg.lr_residual)

    # Initial true state
    s_true = torch.zeros((1, cfg.state_dim), device=device)
    with torch.no_grad():
        z_b = encoder(s_true)  # initial latent belief

    true_traj = []
    belief_traj = []      # decoded belief in state space
    meas_traj = []
    losses = []

    for t in range(cfg.steps):
        # 1) Choose action
        a_np = generate_action(t)
        a = torch.tensor(a_np, device=device).unsqueeze(0)  # (1, 2)

        # 2) True environment step
        s_true = f_true(s_true, a)

        # 3) Measurement from true state
        y_obs = h_measure(s_true, noise_std=cfg.meas_noise_std)

        # 4) Latent prior: decode belief → prior dynamics → encode
        with torch.no_grad():
            s_b = decoder(z_b)            # belief state
            s_prior = f_prior(s_b, a)     # prior next state
            z_prior = encoder(s_prior)    # prior latent

        # 5) Latent residual dynamics
        z_pred = z_prior + residual_net(z_b, a)  # latent prediction

        # 6) Decode predicted latent to state, then to measurement
        s_pred = decoder(z_pred)
        y_pred = h_measure(s_pred)  # no noise

        # 7) Measurement-only loss
        loss = torch.mean((y_pred - y_obs) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 8) Measurement update in state space, then re-encode
        with torch.no_grad():
            s_pred_det = s_pred.detach()
            y_pred_det = h_measure(s_pred_det)
            e = y_obs - y_pred_det  # (1, 2)

            s_corr = s_pred_det.clone()
            # Correct position only
            s_corr[..., 0:1] += cfg.meas_gain * e[..., 0:1]
            s_corr[..., 1:2] += cfg.meas_gain * e[..., 1:2]
            # velocities left unchanged

            # Re-encode corrected state as new latent belief
            z_b = encoder(s_corr)

        # 9) Logging
        true_traj.append(s_true.detach().cpu().numpy()[0])
        belief_traj.append(s_corr.detach().cpu().numpy()[0])   # decoded belief
        meas_traj.append(y_obs.detach().cpu().numpy()[0])
        losses.append(loss.item())

        if (t % cfg.log_interval) == 0 or (t == cfg.steps - 1):
            pos_true = s_true[0, :2].detach().cpu().numpy()
            pos_belief = s_corr[0, :2].detach().cpu().numpy()
            pos_err = np.linalg.norm(pos_true - pos_belief)
            meas_err = np.linalg.norm(
                y_obs[0].detach().cpu().numpy() - y_pred[0].detach().cpu().numpy()
            )
            print(
                f"[49c][t={t:03d}] "
                f"loss={loss.item():.5f}  "
                f"|pos_err|={pos_err:.3f}  "
                f"|meas_err|={meas_err:.3f}"
            )

    # Summary metrics
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
    print("[49c] Finished online latent demo.")
    print(f"  RMS position error (true vs decoded belief) = {rms_pos_err:.4f}")
    print(f"  RMS measurement error (belief vs y)         = {rms_meas_err:.4f}")
    print(f"  Final loss (last step)                      = {losses[-1]:.6f}")


# -------------------------
# Argparse / main
# -------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="44_49c: Latent transition with autoencoder + online residual filter"
    )
    parser.add_argument("--steps", type=int, default=400, help="Online steps")
    parser.add_argument("--ae-samples", type=int, default=5000, help="AE dataset size")
    parser.add_argument("--ae-epochs", type=int, default=20, help="AE epochs")
    parser.add_argument("--ae-lr", type=float, default=1e-3, help="AE learning rate")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--lr-residual", type=float, default=1e-3, help="Residual LR")
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
        device=device,
        seed=args.seed,
        latent_dim=args.latent_dim,
        ae_samples=args.ae_samples,
        ae_epochs=args.ae_epochs,
        ae_lr=args.ae_lr,
        steps=args.steps,
        lr_residual=args.lr_residual,
        meas_noise_std=args.meas_noise_std,
        meas_gain=args.meas_gain,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # Stage 1: offline autoencoder training
    encoder, decoder = train_autoencoder(cfg, device)

    # Stage 2: online latent residual + filtering
    run_online_latent_demo(cfg, device, encoder, decoder)
