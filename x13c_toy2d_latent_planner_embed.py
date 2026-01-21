#!/usr/bin/env python
"""
13c_toy2d_latent_planner_embed.py

Use the trained 2D latent dynamics model (with action embeddings)
to do goal-conditioned planning in JEPA latent space using random shooting.

Pipeline:
  - Create Toy2DWorld.
  - Sample random start agent/goal positions.
  - Render + encode start and goal images -> z_start, z_goal.
  - Sample many candidate action sequences (discrete actions 0..3).
  - Roll out each sequence in latent space with the predictor.
  - Score each final latent vs z_goal via cosine similarity.
  - Pick the best sequence.
  - Replay that sequence in the real Toy2DWorld and save a GIF.

This is the 2D analogue of the 10a 1D planner, but with:
  - learned action embeddings,
  - 2D gridworld images,
  - JEPA latents.
"""

import os
import argparse
from dataclasses import dataclass
from typing import Tuple, List

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# 1. 2D Toy World (same as in 13b)
# ------------------------------------------------------------

@dataclass
class Toy2DWorldConfig:
    img_size: int = 96
    grid_size: int = 10
    margin: int = 8
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
        self.agent_pos = (0, 0)
        self.goal_pos = (self.grid_size - 1, self.grid_size - 1)

    def reset(
        self,
        agent_pos: Tuple[int, int] = None,
        goal_pos: Tuple[int, int] = None,
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if agent_pos is None:
            ax = random.randint(0, self.grid_size - 1)
            ay = random.randint(0, self.grid_size - 1)
            agent_pos = (ax, ay)

        if goal_pos is None:
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
        x, y = self.agent_pos
        if action_idx == 0:      # up
            y -= 1
        elif action_idx == 1:    # down
            y += 1
        elif action_idx == 2:    # left
            x -= 1
        elif action_idx == 3:    # right
            x += 1

        x = max(0, min(self.grid_size - 1, x))
        y = max(0, min(self.grid_size - 1, y))
        self.agent_pos = (x, y)
        done = (self.agent_pos == self.goal_pos)
        return self.agent_pos, done

    def render(self) -> Image.Image:
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

        # grid lines
        for i in range(g + 1):
            x = left + i * cell_w
            draw.line([(x, top), (x, bottom)], fill=cfg.grid_color, width=1)
        for j in range(g + 1):
            y = top + j * cell_h
            draw.line([(left, y), (right, y)], fill=cfg.grid_color, width=1)

        def cell_center(ix: int, iy: int) -> Tuple[int, int]:
            cx = left + (ix + 0.5) * cell_w
            cy = top + (iy + 0.5) * cell_h
            return int(cx), int(cy)

        # goal
        gx, gy = self.goal_pos
        gx_pix, gy_pix = cell_center(gx, gy)
        r = cfg.goal_radius
        draw.ellipse(
            [(gx_pix - r, gy_pix - r), (gx_pix + r, gy_pix + r)],
            fill=cfg.goal_color,
        )

        # agent
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
    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    z = outputs.last_hidden_state[:, 0]   # (1, D)
    z = F.normalize(z, dim=-1)
    return z.squeeze(0)                   # (D,)

# ------------------------------------------------------------
# 3. Latent dynamics with action embeddings (same arch as 13b)
# ------------------------------------------------------------

class LatentDynamicsWithActionEmbedding(nn.Module):
    def __init__(
        self,
        z_dim: int,
        num_actions: int,
        action_embed_dim: int = 8,
        hidden_dim: int = 1024,
    ):
        super().__init__()
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
        z:          (N, D)
        action_idx: (N,)  int64
        """
        a_emb = self.action_embedding(action_idx)  # (N, E)
        x = torch.cat([z, a_emb], dim=-1)
        dz = self.net(x)
        z_next = z + dz
        z_next = F.normalize(z_next, dim=-1)
        return z_next


def load_trained_predictor(
    ckpt_path: str,
    z_dim: int,
    num_actions: int,
    action_embed_dim: int,
    hidden_dim: int,
) -> LatentDynamicsWithActionEmbedding:
    print(f"[Predictor] Loading checkpoint from {ckpt_path} ...")
    model = LatentDynamicsWithActionEmbedding(
        z_dim=z_dim,
        num_actions=num_actions,
        action_embed_dim=action_embed_dim,
        hidden_dim=hidden_dim,
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    print("[Predictor] Loaded and set to eval().")
    return model

# ------------------------------------------------------------
# 4. Random shooting in latent space
# ------------------------------------------------------------

@torch.no_grad()
def sample_action_sequences_discrete(
    num_candidates: int,
    horizon: int,
    num_actions: int = 4,
) -> torch.Tensor:
    """
    Sample discrete action sequences:
      shape: (N, H), entries in {0, ..., num_actions-1}
    """
    # numpy for convenience, then to torch
    arr = np.random.randint(0, num_actions, size=(num_candidates, horizon))
    actions_idx = torch.from_numpy(arr).long().to(device)
    return actions_idx


@torch.no_grad()
def rollout_latent_sequences_discrete(
    z0: torch.Tensor,                     # (D,)
    actions_idx: torch.Tensor,            # (N, H)
    predictor: LatentDynamicsWithActionEmbedding,
) -> torch.Tensor:
    """
    Roll out all candidate sequences in parallel in latent space.

    Returns:
      z_final: (N, D)
    """
    N, H = actions_idx.shape
    D = z0.shape[-1]

    z = z0.unsqueeze(0).expand(N, D)  # (N, D)

    for t in range(H):
        a_t_idx = actions_idx[:, t]   # (N,)
        z = predictor(z, a_t_idx)     # (N, D)

    return z


@torch.no_grad()
def score_candidates(
    z_final: torch.Tensor,   # (N, D)
    z_goal: torch.Tensor,    # (D,)
) -> torch.Tensor:
    """
    Cosine similarity between final latent z_T of each candidate and z_goal.
    Returns scores of shape (N,).
    """
    z_goal = z_goal.unsqueeze(0)      # (1, D)
    z_goal = F.normalize(z_goal, dim=-1)
    z_final = F.normalize(z_final, dim=-1)
    scores = (z_final * z_goal).sum(dim=-1)
    return scores


@torch.no_grad()
def random_shooting_plan_latent(
    z_start: torch.Tensor,         # (D,)
    z_goal: torch.Tensor,          # (D,)
    predictor: LatentDynamicsWithActionEmbedding,
    num_candidates: int,
    horizon: int,
    num_actions: int = 4,
) -> Tuple[torch.Tensor, float]:
    """
    Goal-conditioned random shooting in latent space.

    Returns:
      best_action_seq: (H,)  int indices
      best_score: float
    """
    # 1) sample sequences
    actions_idx = sample_action_sequences_discrete(
        num_candidates=num_candidates,
        horizon=horizon,
        num_actions=num_actions,
    )  # (N, H)

    # 2) rollout
    z_final = rollout_latent_sequences_discrete(
        z0=z_start,
        actions_idx=actions_idx,
        predictor=predictor,
    )  # (N, D)

    # 3) score
    scores = score_candidates(z_final, z_goal)  # (N,)

    # 4) pick best
    best_idx = torch.argmax(scores).item()
    best_seq = actions_idx[best_idx]           # (H,)
    best_score = scores[best_idx].item()
    return best_seq, best_score

# ------------------------------------------------------------
# 5. Full demo: plan + replay + GIF
# ------------------------------------------------------------

@torch.no_grad()
def demo_toy2d_planning(
    predictor_ckpt: str,
    model_id: str,
    z_dim: int,
    action_embed_dim: int,
    hidden_dim: int,
    num_candidates: int,
    horizon: int,
    grid_size: int,
    img_size: int,
    out_gif: str,
    seed: int = 0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Worlds: one for latent images, one for replay
    cfg = Toy2DWorldConfig(img_size=img_size, grid_size=grid_size)
    world_latent = Toy2DWorld(cfg)
    world_real = Toy2DWorld(cfg)

    # Sample shared start/goal
    start_agent, goal_pos = world_latent.reset()
    start_agent = world_latent.agent_pos
    goal_pos = world_latent.goal_pos
    print(f"[Toy2D] Start agent={start_agent}, goal={goal_pos}")

    # Mirror into replay world
    world_real.reset(agent_pos=start_agent, goal_pos=goal_pos)

    # Load encoder + predictor
    encoder, processor = load_frozen_encoder(model_id)
    predictor = load_trained_predictor(
        ckpt_path=predictor_ckpt,
        z_dim=z_dim,
        num_actions=4,
        action_embed_dim=action_embed_dim,
        hidden_dim=hidden_dim,
    )

    # Render + encode start and goal
    img_start = world_latent.render()
    world_latent.agent_pos, world_latent.goal_pos = start_agent, goal_pos
    img_goal = world_latent.render()

    print("[Toy2D] Encoding start/goal images to latents ...")
    z_start = encode_image_to_latent(img_start, encoder, processor)
    z_goal = encode_image_to_latent(img_goal, encoder, processor)

    # Plan
    print("[Planner] Running goal-conditioned random shooting (2D latent) ...")
    best_seq, best_score = random_shooting_plan_latent(
        z_start=z_start,
        z_goal=z_goal,
        predictor=predictor,
        num_candidates=num_candidates,
        horizon=horizon,
        num_actions=4,
    )

    print(f"[Planner] Best score (cosine with z_goal): {best_score:.4f}")
    print("[Planner] Best action sequence (H):")
    print(best_seq.cpu().numpy())

    # Replay best plan in the real toy world
    print("[Toy2D] Replaying best plan and saving frames ...")
    frames: List[Image.Image] = []
    world_real.reset(agent_pos=start_agent, goal_pos=goal_pos)
    frames.append(world_real.render())

    done = False
    for t in range(horizon):
        a_idx = best_seq[t].item()
        _, done = world_real.step(a_idx)
        frames.append(world_real.render())
        if done:
            print(f"[Toy2D] Reached goal at step t={t}")
            # keep going for GIF continuity, or break; here we just continue

    # Save GIF
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
    )
    print(f"[Toy2D] Saved GIF to {out_gif}")


# ------------------------------------------------------------
# 6. CLI / main
# ------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="13c: Goal-conditioned random shooting in 2D JEPA latent space.",
    )
    ap.add_argument(
        "--predictor_ckpt",
        type=str,
        required=True,
        help="Path to trained latent predictor checkpoint (.pt).",
    )
    ap.add_argument(
        "--model_id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HF model id for I-JEPA encoder.",
    )
    ap.add_argument(
        "--z_dim",
        type=int,
        default=1280,
        help="Latent dimension D (must match encoder CLS dim).",
    )
    ap.add_argument(
        "--action_embed_dim",
        type=int,
        default=8,
        help="Action embedding dimension (must match training).",
    )
    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Hidden dimension for MLP (must match training).",
    )
    ap.add_argument(
        "--num_candidates",
        type=int,
        default=256,
        help="Number of random action sequences.",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Planning horizon (sequence length).",
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
        help="Image size for Toy2DWorld.",
    )
    ap.add_argument(
        "--out_gif",
        type=str,
        default="toy2d_best_plan.gif",
        help="Path to output GIF.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    demo_toy2d_planning(
        predictor_ckpt=args.predictor_ckpt,
        model_id=args.model_id,
        z_dim=args.z_dim,
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
        num_candidates=args.num_candidates,
        horizon=args.horizon,
        grid_size=args.grid_size,
        img_size=args.img_size,
        out_gif=args.out_gif,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
