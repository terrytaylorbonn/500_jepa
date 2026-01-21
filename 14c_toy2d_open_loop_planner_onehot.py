"""
14c_toy2d_open_loop_planner_onehot.py

Open-loop latent planning in a simple Toy2D environment using:

- Frozen I-JEPA encoder (facebook/ijepa_vith14_1k)
- One-hot latent predictor trained in 14b

Steps:
  1) Reset Toy2D env → start image x_start.
  2) Sample a random goal grid cell, render a goal image x_goal.
  3) Encode x_start -> z_start, x_goal -> z_goal via I-JEPA.
  4) Random-shooting planner in latent space chooses best action sequence.
  5) Execute the full sequence open-loop in env, record frames → GIF.
"""

import os
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoProcessor


# -------------------------------------------------------------
#  Toy2DEnv (same as 14a, duplicated here for clarity)
# -------------------------------------------------------------

class Toy2DEnv:
    """
    Minimal 2D grid world with a single moving agent.

    - Grid: grid_size x grid_size (logical)
    - Render: img_size x img_size RGB
    - Actions: 0=up, 1=down, 2=left, 3=right
    """

    def __init__(self, img_size: int = 96, grid_size: int = 12, num_actions: int = 4):
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_actions = num_actions

        # agent (x, y) in grid coordinates
        self.agent_pos: Tuple[int, int] = (0, 0)

        # Colors
        self.bg_color = np.array([255, 255, 255], dtype=np.uint8)   # white
        self.agent_color = np.array([0, 0, 255], dtype=np.uint8)    # blue

        # Pixel size of each grid cell
        self.cell_px = img_size // grid_size

    def reset(self):
        """Randomize agent position and return initial RGB observation."""
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)
        self.agent_pos = (x, y)
        return self.render_rgb()

    def step(self, action: int):
        """
        Take one step in the grid.

        action: int in [0, num_actions-1]
          0 = up, 1 = down, 2 = left, 3 = right

        Returns:
          obs : (H, W, 3) uint8
          reward : float   (always 0.0 for now)
          done   : bool    (always False)
          info   : dict    (empty)
        """
        x, y = self.agent_pos

        if action == 0:      # up
            y -= 1
        elif action == 1:    # down
            y += 1
        elif action == 2:    # left
            x -= 1
        elif action == 3:    # right
            x += 1

        # Clamp to grid bounds
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))

        self.agent_pos = (x, y)
        obs = self.render_rgb()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def render_rgb(self) -> np.ndarray:
        """
        Render the current state as a uint8 RGB image of shape (H, W, 3).
        """
        H = W = self.img_size
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

        x, y = self.agent_pos
        px = self.cell_px
        x0 = x * px
        y0 = y * px
        x1 = x0 + px
        y1 = y0 + px

        canvas[y0:y1, x0:x1] = self.agent_color
        return canvas

    def render_goal_image(self, goal_pos: Tuple[int, int]) -> np.ndarray:
        """
        Render an image with the agent at goal_pos (without changing current env state).
        """
        H = W = self.img_size
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white

        x, y = goal_pos
        px = self.cell_px
        x0 = x * px
        y0 = y * px
        x1 = x0 + px
        y1 = y0 + px

        canvas[y0:y1, x0:x1] = self.agent_color
        return canvas


# -------------------------------------------------------------
#  LatentPredictorOneHot (same as 14b)
# -------------------------------------------------------------

class LatentPredictorOneHot(nn.Module):
    def __init__(self, z_dim: int, num_actions: int, hidden_dim: int = 1024):
        super().__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(z_dim + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, z_dim)

    def forward(self, z: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a_idx, num_classes=self.num_actions).float()
        x = torch.cat([z, a_onehot], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        z_next = self.fc_out(x)
        return z_next


# -------------------------------------------------------------
#  I-JEPA helpers
# -------------------------------------------------------------

def load_ijepa_encoder(model_id: str, device: torch.device):
    print(f"[14c] Loading I-JEPA processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print("[14c] Processor loaded.", flush=True)

    print(f"[14c] Loading I-JEPA encoder on {device}...", flush=True)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    print(f"[14c] Encoder loaded on {device}.", flush=True)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, processor


@torch.no_grad()
def encode_image_to_latent(
    img: Image.Image,
    encoder,
    processor,
    device: torch.device,
) -> torch.Tensor:
    img = img.convert("RGB")
    inputs = processor(img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = encoder(**inputs)
    z = outputs.last_hidden_state[:, 0, :]  # CLS
    return z  # (1, z_dim)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("[14c] Device: cuda", flush=True)
        return torch.device("cuda")
    else:
        print("[14c] Device: cpu", flush=True)
        return torch.device("cpu")


# -------------------------------------------------------------
#  Random-shooting planner in latent space
# -------------------------------------------------------------

@torch.no_grad()
def plan_open_loop_random_shooting(
    z_start: torch.Tensor,     # (1, D)
    z_goal: torch.Tensor,      # (1, D)
    predictor: nn.Module,
    num_actions: int,
    num_candidates: int,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns best action sequence: (horizon,) integer tensor.
    """
    # Sample candidate action sequences: (N, H)
    candidate_actions = torch.randint(
        low=0,
        high=num_actions,
        size=(num_candidates, horizon),
        device=device,
    )

    # Tile latent start: (N, D)
    z = z_start.expand(num_candidates, -1).contiguous()
    z_goal_expanded = z_goal.expand(num_candidates, -1).contiguous()

    for t in range(horizon):
        a_t_idx = candidate_actions[:, t]   # (N,)
        z = predictor(z, a_t_idx)           # (N, D)

    # Cost = squared distance to goal latent
    costs = torch.sum((z - z_goal_expanded) ** 2, dim=-1)  # (N,)
    best_idx = torch.argmin(costs)
    best_seq = candidate_actions[best_idx]                 # (H,)
    return best_seq


# -------------------------------------------------------------
#  GIF helper
# -------------------------------------------------------------

def save_gif(frames: List[Image.Image], out_path: str, fps: int = 10):
    if len(frames) == 0:
        print("[14c] No frames to save.")
        return
    durations = [1.0 / fps] * len(frames)
    imageio.mimsave(out_path, frames, duration=durations)
    print(f"[14c] Saved GIF to {out_path} (frames={len(frames)}, fps={fps})")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor-ckpt",
        type=str,
        required=True,
        help="Checkpoint from 14b_train_latent_predictor_onehot.py",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HuggingFace model id for I-JEPA.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="Image size for Toy2DEnv.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=12,
        help="Grid size for Toy2DEnv.",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=4,
        help="Number of discrete actions.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=256,
        help="Number of candidate sequences for random shooting.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Planning horizon (latent steps).",
    )
    parser.add_argument(
        "--max-env-steps",
        type=int,
        default=30,
        help="Maximum steps to execute in the environment.",
    )
    parser.add_argument(
        "--out-gif",
        type=str,
        default="14c_toy2d_open_loop.gif",
        help="Output GIF filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()

    # 1) Load I-JEPA encoder
    encoder, processor = load_ijepa_encoder(args.model_id, device=device)

    # 2) Load predictor
    state = torch.load(args.predictor_ckpt, map_location=device)
    z_dim = state["z_dim"]
    num_actions = state["num_actions"]
    print(f"[14c] Loaded predictor ckpt: z_dim={z_dim}, num_actions={num_actions}", flush=True)

    predictor = LatentPredictorOneHot(
        z_dim=z_dim,
        num_actions=num_actions,
        hidden_dim=1024,
    ).to(device)
    predictor.load_state_dict(state["predictor"])
    predictor.eval()

    # 3) Create environment
    env = Toy2DEnv(
        img_size=args.img_size,
        grid_size=args.grid_size,
        num_actions=num_actions,
    )

    # 4) Sample start state and goal state
    start_obs = env.reset()  # (H, W, 3)
    start_img = Image.fromarray(start_obs)

    # Choose a random goal position on the grid
    gx = np.random.randint(0, env.grid_size)
    gy = np.random.randint(0, env.grid_size)
    goal_pos = (gx, gy)
    print(f"[14c] Goal grid position: {goal_pos}", flush=True)

    goal_obs = env.render_goal_image(goal_pos)
    goal_img = Image.fromarray(goal_obs)

    # 5) Encode z_start and z_goal
    with torch.no_grad():
        z_start = encode_image_to_latent(start_img, encoder, processor, device)
        z_goal = encode_image_to_latent(goal_img, encoder, processor, device)

    # 6) Plan in latent space (open-loop)
    best_seq = plan_open_loop_random_shooting(
        z_start=z_start,
        z_goal=z_goal,
        predictor=predictor,
        num_actions=num_actions,
        num_candidates=args.num_candidates,
        horizon=args.horizon,
        device=device,
    )

    print("[14c] Best action sequence:", best_seq.cpu().numpy(), flush=True)

    # 7) Execute sequence in env and record frames
    frames: List[Image.Image] = []
    frames.append(start_img.copy())

    step = 0
    for a in best_seq:
        if step >= args.max_env_steps:
            print("[14c] Reached max_env_steps, stopping.", flush=True)
            break

        a_int = int(a.item())
        obs, reward, done, info = env.step(a_int)

        frame = Image.fromarray(obs)
        frames.append(frame.copy())

        step += 1
        if done:
            print(f"[14c] Env returned done at step {step}", flush=True)
            break

    # 8) Save GIF
    save_gif(frames, args.out_gif, fps=5)


if __name__ == "__main__":
    main()
