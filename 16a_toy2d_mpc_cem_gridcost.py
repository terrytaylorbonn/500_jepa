"""
16a_toy2d_mpc_cem_gridcost.py

Minimum working demo:
- Closed-loop MPC with CEM planner
- Cost = grid distance (Manhattan) to goal
- Environment renders:
    - red square = goal
    - blue square = agent

JEPA encoder + predictor are still loaded (for continuity with earlier demos),
but this planner uses only grid-space cost to guarantee a visible success.
"""

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
#  Toy2DEnv (with goal rendering)
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
        self.goal_color = np.array([255, 0, 0], dtype=np.uint8)     # red

        # Pixel size of each grid cell
        self.cell_px = img_size // grid_size

        # Optional goal position (for rendering only)
        self.goal_pos: Tuple[int, int] | None = None

    def set_goal(self, goal_pos: Tuple[int, int]):
        """Set goal position for rendering (does not affect dynamics)."""
        self.goal_pos = goal_pos

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
          done   : bool    (True if agent reaches goal)
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

        # "Done" if we exactly reach the goal (if goal is set)
        done = False
        if self.goal_pos is not None:
            if (x, y) == self.goal_pos:
                done = True

        obs = self.render_rgb()
        reward = 0.0
        info = {}
        return obs, reward, done, info

    def render_rgb(self) -> np.ndarray:
        """
        Render the current state as a uint8 RGB image of shape (H, W, 3).

        - Background: white
        - Goal: red square (if goal_pos is set)
        - Agent: blue square
        """
        H = W = self.img_size
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

        px = self.cell_px

        # Draw goal first (under agent)
        if self.goal_pos is not None:
            gx, gy = self.goal_pos
            gx0 = gx * px
            gy0 = gy * px
            gx1 = gx0 + px
            gy1 = gy0 + px
            canvas[gy0:gy1, gx0:gx1] = self.goal_color

        # Draw agent on top
        x, y = self.agent_pos
        x0 = x * px
        y0 = y * px
        x1 = x0 + px
        y1 = y0 + px
        canvas[y0:y1, x0:x1] = self.agent_color

        return canvas

    def render_goal_image(self, goal_pos: Tuple[int, int]) -> np.ndarray:
        """
        Render an image with the agent at goal_pos ONLY (for encoding the goal).
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
#  LatentPredictorOneHot (unchanged, for continuity)
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
#  I-JEPA helpers (still here, for story continuity)
# -------------------------------------------------------------

def load_ijepa_encoder(model_id: str, device: torch.device):
    print(f"[16a] Loading I-JEPA processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print("[16a] Processor loaded.", flush=True)

    print(f"[16a] Loading I-JEPA encoder on {device}...", flush=True)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    print(f"[16a] Encoder loaded on {device}.", flush=True)

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
        print("[16a] Device: cuda", flush=True)
        return torch.device("cuda")
    else:
        print("[16a] Device: cpu", flush=True)
        return torch.device("cpu")


# -------------------------------------------------------------
#  CEM planner with GRID cost (no JEPA cost)
# -------------------------------------------------------------

@torch.no_grad()
def plan_cem_grid_cost(
    start_pos: Tuple[int, int],
    goal_pos: Tuple[int, int],
    grid_size: int,
    num_actions: int,
    horizon: int,
    device: torch.device,
    num_candidates: int = 256,
    num_elites: int = 64,
    num_iterations: int = 3,
) -> torch.Tensor:
    """
    CEM planner operating purely in grid space.

    - start_pos, goal_pos: (x, y)
    - cost = Manhattan distance from final grid position to goal

    Returns:
        best_seq: (horizon,) int64 tensor of action indices.
    """
    sx, sy = start_pos
    gx, gy = goal_pos

    sx = torch.tensor(sx, device=device, dtype=torch.long)
    sy = torch.tensor(sy, device=device, dtype=torch.long)
    gx = torch.tensor(gx, device=device, dtype=torch.long)
    gy = torch.tensor(gy, device=device, dtype=torch.long)

    # probs[t, a] = P(action=a at time t)
    probs = torch.full(
        (horizon, num_actions),
        1.0 / num_actions,
        device=device,
    )

    best_seq = None
    best_cost = None

    for it in range(num_iterations):
        # 1) Sample candidate sequences under current probs
        samples_per_t = []
        for t in range(horizon):
            a_t = torch.multinomial(
                probs[t],
                num_samples=num_candidates,
                replacement=True,
            )  # (N,)
            samples_per_t.append(a_t)
        actions_TH = torch.stack(samples_per_t, dim=0)       # (H, N)
        candidate_actions = actions_TH.t().contiguous()      # (N, H)

        # 2) Roll out grid positions
        N = num_candidates
        x = sx.expand(N).clone()
        y = sy.expand(N).clone()

        for t in range(horizon):
            a_t = candidate_actions[:, t]  # (N,)

            # Move according to actions
            # 0=up, 1=down, 2=left, 3=right
            y = y - (a_t == 0).long()  # up
            y = y + (a_t == 1).long()  # down
            x = x - (a_t == 2).long()  # left
            x = x + (a_t == 3).long()  # right

            # Clamp to grid
            x = torch.clamp(x, 0, grid_size - 1)
            y = torch.clamp(y, 0, grid_size - 1)

        # Final Manhattan distance to goal
        dx = torch.abs(x - gx)
        dy = torch.abs(y - gy)
        costs = (dx + dy).float()  # (N,)

        # Track global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if best_cost is None or min_cost < best_cost:
            best_cost = min_cost
            best_seq = candidate_actions[min_idx].detach().clone()

        # 3) Select elites
        _, elite_idx = torch.topk(
            costs,
            k=min(num_elites, num_candidates),
            largest=False,
        )
        elite_actions = candidate_actions[elite_idx]  # (E, H)

        # 4) Update probs from elites
        for t in range(horizon):
            a_t = elite_actions[:, t]  # (E,)
            counts = torch.bincount(a_t, minlength=num_actions).float()
            probs[t] = counts / counts.sum()

        # Smooth to avoid zeros
        eps = 1e-3
        probs = probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return best_seq


# -------------------------------------------------------------
#  Closed-loop MPC rollout (grid-cost CEM)
# -------------------------------------------------------------

def mpc_rollout(
    env: Toy2DEnv,
    goal_pos: Tuple[int, int],
    num_actions: int,
    num_candidates: int,
    horizon: int,
    max_env_steps: int,
    device: torch.device,
) -> List[Image.Image]:
    """
    Run closed-loop MPC with CEM in GRID space and return frames.
    """
    frames: List[Image.Image] = []

    obs = env.reset()
    cur_img = Image.fromarray(obs)
    frames.append(cur_img.copy())

    step = 0

    while step < max_env_steps:
        sx, sy = env.agent_pos
        gx, gy = goal_pos

        # Plan in grid space
        best_seq = plan_cem_grid_cost(
            start_pos=(sx, sy),
            goal_pos=(gx, gy),
            grid_size=env.grid_size,
            num_actions=num_actions,
            horizon=horizon,
            device=device,
            num_candidates=num_candidates,
            num_elites=min(64, num_candidates),
            num_iterations=3,
        )

        # Execute FIRST action
        a_int = int(best_seq[0].item())
        obs, reward, done, info = env.step(a_int)
        cur_img = Image.fromarray(obs)
        frames.append(cur_img.copy())
        step += 1

        dx = abs(env.agent_pos[0] - gx)
        dy = abs(env.agent_pos[1] - gy)
        grid_dist = dx + dy
        print(f"[16a] step={step} grid_dist={grid_dist}", flush=True)

        if done:
            print(f"[16a] Reached goal at step {step}", flush=True)
            break

    return frames


# -------------------------------------------------------------
#  GIF helper
# -------------------------------------------------------------

def save_gif(frames: List[Image.Image], out_path: str, fps: int = 5):
    if len(frames) == 0:
        print("[16a] No frames to save.")
        return
    durations = [1.0 / fps] * len(frames)
    imageio.mimsave(out_path, frames, duration=durations)
    print(f"[16a] Saved GIF to {out_path} (frames={len(frames)}, fps={fps})")


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor-ckpt",
        type=str,
        required=True,
        help="Checkpoint from 14b_train_latent_predictor_onehot.py (loaded but not used by this planner).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/ijepa_vith14_1k",
        help="HuggingFace model id for I-JEPA (loaded for continuity).",
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
        help="Number of candidate sequences for CEM.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Planning horizon (steps).",
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
        default="16a_toy2d_mpc_gridcost.gif",
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

    # 1) Load I-JEPA encoder (not used for planning here, but kept for story continuity)
    encoder, processor = load_ijepa_encoder(args.model_id, device=device)

    # 2) Load predictor checkpoint (not used in this demo, but keeps pipeline consistent)
    state = torch.load(args.predictor_ckpt, map_location=device)
    z_dim = state["z_dim"]
    num_actions = state["num_actions"]
    print(f"[16a] Loaded predictor ckpt: z_dim={z_dim}, num_actions={num_actions}", flush=True)

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

    # 4) Sample a goal position and tell env about it
    gx = np.random.randint(0, env.grid_size)
    gy = np.random.randint(0, env.grid_size)
    goal_pos = (gx, gy)
    print(f"[16a] Goal grid position: {goal_pos}", flush=True)

    env.set_goal(goal_pos)

    # (Optional) encode goal image to latent, just to keep JEPA in the story
    goal_obs = env.render_goal_image(goal_pos)
    goal_img = Image.fromarray(goal_obs)
    with torch.no_grad():
        _ = encode_image_to_latent(goal_img, encoder, processor, device)

    # 5) Run closed-loop MPC rollout with GRID cost
    frames = mpc_rollout(
        env=env,
        goal_pos=goal_pos,
        num_actions=num_actions,
        num_candidates=args.num_candidates,
        horizon=args.horizon,
        max_env_steps=args.max_env_steps,
        device=device,
    )

    # 6) Save GIF
    save_gif(frames, args.out_gif, fps=5)


if __name__ == "__main__":
    main()
