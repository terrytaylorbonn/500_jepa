#!/usr/bin/env python
"""
13b_train_toy2d_latent_predictor_embed.py

Train a JEPA-style latent dynamics model on the 2D toy world (grid + agent + goal).

Pipeline:
  - Roll out random episodes in Toy2DWorld.
  - For each transition:
      img_t, action_idx, img_{t+1}
  - Encode images with frozen I-JEPA encoder:
      z_t, z_{t+1}
  - Train LatentDynamicsWithActionEmbedding:
      (z_t, action_idx) -> ẑ_{t+1}
  - Loss: 1 - cosine(ẑ_{t+1}, z_{t+1})
  - Save checkpoint: ./checkpoints/predictor_2d_embed.pt

This sets up Chapter 13 for:
  - learned action embeddings,
  - latent imagination in 2D visual world,
  - later planning (random shooting, CEM, MPC).
"""

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# 1. 2D Toy World (same as your random GIF demo)
# ------------------------------------------------------------

@dataclass
class Toy2DWorldConfig:
    img_size: int = 96        # image pixels (square)
    grid_size: int = 10       # logical grid cells per side
    margin: int = 8           # padding around grid
    agent_color: Tuple[int, int, int] = (0, 0, 255)    # blue
    goal_color: Tuple[int, int, int] = (255, 0, 0)     # red
    bg_color: Tuple[int, int, int] = (240, 240, 240)   # light gray
    grid_color: Tuple[int, int, int] = (200, 200, 200) # grid lines
    agent_radius: int = 3
    goal_radius: int = 3


class Toy2DWorld:
    """
    Simple 2D gridworld:
      - agent moves in a grid (x, y)
      - goal is at a fixed grid cell
      - actions: 0=up, 1=down, 2=left, 3=right
    """

    def __init__(self, cfg: Toy2DWorldConfig):
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.grid_size = cfg.grid_size
        self.margin = cfg.margin

        # world state
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)

    def reset(
        self,
        agent_pos: Tuple[int, int] = None,
        goal_pos: Tuple[int, int] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Reset agent + goal (optionally random)."""
        if agent_pos is None:
            ax = random.randint(0, self.grid_size - 1)
            ay = random.randint(0, self.grid_size - 1)
            agent_pos = (ax, ay)

        if goal_pos is None:
            # ensure goal != agent
            while True:
                gx = random.randint(0, self.grid_size - 1)
                gy = random.randint(0, self.grid_size - 1)
                if (gx, gy) != agent_pos:
                    break
            goal_pos = (gx, gy)

        self.agent_pos = agent_pos
        self.goal_pos = goal_pos
        return self.agent_pos, self.goal_pos

    def step(self, action_idx: int) -> Tuple[Tuple[int, int], bool]:
        """
        Apply discrete action to agent.
        Actions:
          0 = up
          1 = down
          2 = left
          3 = right

        Returns:
          new_agent_pos, done
        """
        x, y = self.agent_pos
        if action_idx == 0:      # up
            y -= 1
        elif action_idx == 1:    # down
            y += 1
        elif action_idx == 2:    # left
            x -= 1
        elif action_idx == 3:    # right
            x += 1

        # clamp to grid
        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        self.agent_pos = (x, y)

        done = (self.agent_pos == self.goal_pos)
        return self.agent_pos, done

    def render(self) -> Image.Image:
        """
        Render current state as a PIL.Image.
        - grid background
        - goal dot
        - agent dot
        """
        cfg = self.cfg
        W = H = cfg.img_size
        img = Image.new("RGB", (W, H), color=cfg.bg_color)
        draw = ImageDraw.Draw(img)

        g = cfg.grid_size
        left = cfg.margin
        right = W - cfg.margin
        top = cfg.margin
        bottom = H - cfg.margin

        cell_w = (right - left) / g
        cell_h = (bottom - top) / g

        # vertical lines
        for i in range(g + 1):
            x = left + i * cell_w
            draw.line([(x, top), (x, bottom)], fill=cfg.grid_color, width=1)

        # horizontal lines
        for j in range(g + 1):
            y = top + j * cell_h
            draw.line([(left, y), (right, y)], fill=cfg.grid_color, width=1)

        # convert (grid_x, grid_y) to pixel center
        def cell_center(ix: int, iy: int) -> Tuple[int, int]:
            cx = left + (ix + 0.5) * cell_w
            cy = top + (iy + 0.5) * cell_h
            return int(cx), int(cy)

        # draw goal
        gx, gy = self.goal_pos
        gx_pix, gy_pix = cell_center(gx, gy)
        r = cfg.goal_radius
        draw.ellipse(
            [(gx_pix - r, gy_pix - r), (gx_pix + r, gy_pix + r)],
            fill=cfg.goal_color,
        )

        # draw agent
        ax, ay = self.agent_pos
        ax_pix, ay_pix = cell_center(ax, ay)
        r = cfg.agent_radius
        draw.ellipse(
            [(ax_pix - r, ay_pix - r), (ax_pix + r, ay_pix + r)],
            fill=cfg.agent_color,
        )

        return img

# ------------------------------------------------------------
# 2. Frozen I-JEPA encoder
# ------------------------------------------------------------

def load_frozen_encoder(model_id: str = "facebook/ijepa_vith14_1k"):
    """
    Load frozen I-JEPA encoder + processor.
    """
    print(f"[Encoder] Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print("[Encoder] Loaded and frozen.")
    return model, processor


@torch.no_grad()
def encode_image_to_latent(
    img: Image.Image,
    model: AutoModel,
    processor: AutoProcessor,
) -> torch.Tensor:
    """
    Return a single latent z of shape (D,) on device.

    This uses CLS token from last_hidden_state[:, 0]
    and L2-normalizes it.
    """
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)

    z = outputs.last_hidden_state[:, 0]   # (1, D)
    z = F.normalize(z, dim=-1)
    return z.squeeze(0)                   # (D,)

# ------------------------------------------------------------
# 3. Dataset generation in latent space
# ------------------------------------------------------------

@torch.no_grad()
def generate_dataset_2d_latent(
    num_samples: int,
    encoder: AutoModel,
    processor: AutoProcessor,
    world: Toy2DWorld,
    max_steps_per_episode: int = 30,
    num_actions: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset of transitions in *latent* space for the 2D world.
    """
    print(f"[Data] Generating {num_samples} transitions (2D world, latent) ...")

    zs_t: List[torch.Tensor] = []
    zs_tp1: List[torch.Tensor] = []
    actions_idx: List[int] = []

    total = 0
    while total < num_samples:
        world.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode and total < num_samples:
            img_t = world.render()
            z_t = encode_image_to_latent(img_t, encoder, processor)  # (D,)

            a_idx = random.randint(0, num_actions - 1)
            _, done = world.step(a_idx)

            img_tp1 = world.render()
            z_tp1 = encode_image_to_latent(img_tp1, encoder, processor)  # (D,)

            zs_t.append(z_t.cpu())
            zs_tp1.append(z_tp1.cpu())
            actions_idx.append(a_idx)

            total += 1
            steps += 1

            # More frequent progress updates
            if total % 100 == 0 or total == num_samples:
                print(f"[Data] {total}/{num_samples} done")

    zs_t = torch.stack(zs_t, dim=0)          # (N, D)
    zs_tp1 = torch.stack(zs_tp1, dim=0)      # (N, D)
    actions_idx = torch.tensor(actions_idx)  # (N,)

    print("[Data] Shapes:",
          "z_t", zs_t.shape,
          "a_idx", actions_idx.shape,
          "z_tp1", zs_tp1.shape)

    return zs_t, actions_idx, zs_tp1

# @torch.no_grad()
# def generate_dataset_2d_latent(
#     num_samples: int,
#     encoder: AutoModel,
#     processor: AutoProcessor,
#     world: Toy2DWorld,
#     max_steps_per_episode: int = 30,
#     num_actions: int = 4,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Generate a dataset of transitions in *latent* space for the 2D world.

#     For each sample:
#       - pick a random state (world.reset)
#       - render img_t
#       - sample random action_idx ~ Uniform{0..num_actions-1}
#       - step in world -> img_{t+1}
#       - encode both images: z_t, z_{t+1}
#       - store (z_t, action_idx, z_{t+1})

#     We keep rolling episodes until we gather num_samples transitions.
#     """
#     print(f"[Data] Generating {num_samples} transitions (2D world, latent) ...")

#     zs_t: List[torch.Tensor] = []
#     zs_tp1: List[torch.Tensor] = []
#     actions_idx: List[int] = []

#     cfg = world.cfg

#     while len(zs_t) < num_samples:
#         # new episode
#         world.reset()
#         done = False
#         steps = 0

#         while not done and steps < max_steps_per_episode and len(zs_t) < num_samples:
#             img_t = world.render()
#             z_t = encode_image_to_latent(img_t, encoder, processor)  # (D,)

#             # random discrete action
#             a_idx = random.randint(0, num_actions - 1)
#             _, done = world.step(a_idx)

#             img_tp1 = world.render()
#             z_tp1 = encode_image_to_latent(img_tp1, encoder, processor)  # (D,)

#             zs_t.append(z_t.cpu())
#             zs_tp1.append(z_tp1.cpu())
#             actions_idx.append(a_idx)

#             steps += 1

#         # optional: could re-randomize goal positions here if you want variety

#         if len(zs_t) % max(1, num_samples // 10) == 0:
#             print(f"[Data] {len(zs_t)}/{num_samples} done")

#     zs_t = torch.stack(zs_t, dim=0)          # (N, D)
#     zs_tp1 = torch.stack(zs_tp1, dim=0)      # (N, D)
#     actions_idx = torch.tensor(actions_idx)  # (N,)

#     print("[Data] Shapes:",
#           "z_t", zs_t.shape,
#           "a_idx", actions_idx.shape,
#           "z_tp1", zs_tp1.shape)

#     return zs_t, actions_idx, zs_tp1

# ------------------------------------------------------------
# 4. Latent dynamics with learned action embeddings
# ------------------------------------------------------------

class LatentDynamicsWithActionEmbedding(nn.Module):
    """
    MLP on [z, a_emb] with residual output:
      z_next = normalize(z + f([z, a_emb])).

    Discrete actions are mapped to continuous action embeddings
    via nn.Embedding(num_actions, action_embed_dim).
    """

    def __init__(
        self,
        z_dim: int,
        num_actions: int,
        action_embed_dim: int = 8,
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.action_embed_dim = action_embed_dim

        self.action_embedding = nn.Embedding(num_actions, action_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(z_dim + action_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, z: torch.Tensor, action_idx: torch.Tensor) -> torch.Tensor:
        """
        z:          (..., D)
        action_idx: (...,)  int64 / long
        returns:    (..., D)
        """
        a_emb = self.action_embedding(action_idx)  # (..., action_embed_dim)
        x = torch.cat([z, a_emb], dim=-1)
        dz = self.net(x)
        z_next = z + dz
        z_next = F.normalize(z_next, dim=-1)
        return z_next

# ------------------------------------------------------------
# 5. Training loop
# ------------------------------------------------------------

def train_latent_dynamics(
    zs_t: torch.Tensor,
    actions_idx: torch.Tensor,
    zs_tp1: torch.Tensor,
    num_actions: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    hidden_dim: int,
    action_embed_dim: int,
    ckpt_path: str,
):
    """
    Train LatentDynamicsWithActionEmbedding on (z_t, action_idx, z_{t+1})
    with cosine similarity loss.
    """
    dataset = TensorDataset(zs_t, actions_idx, zs_tp1)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    z_dim = zs_t.shape[-1]

    model = LatentDynamicsWithActionEmbedding(
        z_dim=z_dim,
        num_actions=num_actions,
        action_embed_dim=action_embed_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    print("[Train] Starting training ...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for z_t, a_idx, z_tp1 in loader:
            z_t = z_t.to(device)
            a_idx = a_idx.to(device).long()
            z_tp1 = z_tp1.to(device)

            pred_z_tp1 = model(z_t, a_idx)  # (B, D)

            # cosine similarity loss: 1 - cos(pred, target)
            cos_sim = F.cosine_similarity(pred_z_tp1, z_tp1, dim=-1)  # (B,)
            loss = (1.0 - cos_sim).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"[Train] Epoch {epoch:03d}/{num_epochs:03d}  loss={avg_loss:.4f}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Train] Saved predictor checkpoint to: {ckpt_path}")

# ------------------------------------------------------------
# 6. CLI / main
# ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="13b: Train 2D JEPA latent predictor with action embeddings.",
    )

    ap.add_argument(
        "--out_ckpt",
        type=str,
        default="./checkpoints/predictor_2d_embed.pt",
        help="Where to save the trained predictor checkpoint.",
    )

    ap.add_argument(
        "--model_id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HF model id for I-JEPA encoder.",
    )

    ap.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Number of transitions to generate.",
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    ap.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )

    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension in the MLP.",
    )

    ap.add_argument(
        "--action_embed_dim",
        type=int,
        default=8,
        help="Dimensionality of learned action embeddings.",
    )

    ap.add_argument(
        "--grid_size",
        type=int,
        default=10,
        help="Grid size for Toy2DWorld.",
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=96,
        help="Image size (pixels) for Toy2DWorld.",
    )

    ap.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=30,
        help="Max steps per random episode during data collection.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    # 1) Setup toy world
    cfg = Toy2DWorldConfig(
        img_size=args.img_size,
        grid_size=args.grid_size,
    )
    world = Toy2DWorld(cfg)

    # 2) Load frozen encoder
    encoder, processor = load_frozen_encoder(args.model_id)

    # 3) Generate latent dataset
    with torch.no_grad():
        zs_t, actions_idx, zs_tp1 = generate_dataset_2d_latent(
            num_samples=args.num_samples,
            encoder=encoder,
            processor=processor,
            world=world,
            max_steps_per_episode=args.max_steps_per_episode,
            num_actions=4,
        )

    # 4) Train latent dynamics model with action embeddings
    train_latent_dynamics(
        zs_t=zs_t,
        actions_idx=actions_idx,
        zs_tp1=zs_tp1,
        num_actions=4,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        action_embed_dim=args.action_embed_dim,
        ckpt_path=args.out_ckpt,
    )


if __name__ == "__main__":
    main()

