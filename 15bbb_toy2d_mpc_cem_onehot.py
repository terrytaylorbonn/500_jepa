"""
15bbb_toy2d_mpc_cem_onehot.py
fix the gif

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
#  Toy2DEnv (same as 14a/14c)
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
        #123.1a #################################
        self.goal_color  = np.array([255, 0, 0], dtype=np.uint8)    # red
        ##################################
              
        # Pixel size of each grid cell
        self.cell_px = img_size // grid_size

        #123.1a #################################
        # Optional goal position (grid coords)
        self.goal_pos: Tuple[int, int] | None = None
        ##################################
        
    #123.1b #################################
    def set_goal(self, goal_pos: Tuple[int, int]):
        """Set goal position for rendering (does not affect dynamics)."""
        self.goal_pos = goal_pos
    ##################################    

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

    #123.1c #################################
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
    ##################################

    # def render_rgb(self) -> np.ndarray:
    #     """
    #     Render the current state as a uint8 RGB image of shape (H, W, 3).
    #     """
    #     H = W = self.img_size
    #     canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    #     x, y = self.agent_pos
    #     px = self.cell_px
    #     x0 = x * px
    #     y0 = y * px
    #     x1 = x0 + px
    #     y1 = y0 + px

    #     canvas[y0:y1, x0:x1] = self.agent_color
    #     return canvas

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
#  LatentPredictorOneHot (same as 14b/14c)
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
    print(f"[14d] Loading I-JEPA processor: {model_id}", flush=True)
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    print("[14d] Processor loaded.", flush=True)

    print(f"[14d] Loading I-JEPA encoder on {device}...", flush=True)
    encoder = AutoModel.from_pretrained(model_id).to(device)
    print(f"[14d] Encoder loaded on {device}.", flush=True)

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
        print("[14d] Device: cuda", flush=True)
        return torch.device("cuda")
    else:
        print("[14d] Device: cpu", flush=True)
        return torch.device("cpu")


# -------------------------------------------------------------
#  Random-shooting planner (same as 14c)
# -------------------------------------------------------------

@torch.no_grad()
def plan_random_shooting(
    z_start: torch.Tensor,
    z_goal: torch.Tensor,
    predictor: nn.Module,
    num_actions: int,
    num_candidates: int,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns best action sequence: (horizon,) integer tensor.
    """
    candidate_actions = torch.randint(
        low=0,
        high=num_actions,
        size=(num_candidates, horizon),
        device=device,
    )

    z = z_start.expand(num_candidates, -1).contiguous()
    z_goal_expanded = z_goal.expand(num_candidates, -1).contiguous()

    for t in range(horizon):
        a_t_idx = candidate_actions[:, t]
        z = predictor(z, a_t_idx)

    costs = torch.sum((z - z_goal_expanded) ** 2, dim=-1)
    best_idx = torch.argmin(costs)
    best_seq = candidate_actions[best_idx]
    return best_seq


############################################3
# #122 1. CEM planner (drop-in function)
############################################3


def plan_cem_latent(
    z_start: torch.Tensor,      # (1, D)
    z_goal: torch.Tensor,       # (1, D)
    predictor: torch.nn.Module,
    num_actions: int,
    horizon: int,
    device: torch.device,
    num_candidates: int = 256,
    num_elites: int = 64,
    num_iterations: int = 3,
) -> torch.Tensor:
    """
    CEM planner in JEPA latent space.

    Returns:
        best_seq: (horizon,) int64 tensor of action indices.
    """
    z_start = z_start.to(device)
    z_goal = z_goal.to(device)
    predictor = predictor.to(device)
    predictor.eval()

    # probs[t, a] = P(action=a at time t)
    probs = torch.full(
        (horizon, num_actions),
        1.0 / num_actions,
        device=device,
    )

    best_seq = None
    best_cost = None

    for it in range(num_iterations):
        # -------------------------------
        # 1) Sample candidate sequences
        # -------------------------------
        # actions_TH: (H, N) then transpose → (N, H)
        samples_per_t = []
        for t in range(horizon):
            # sample num_candidates actions at time t from probs[t]
            a_t = torch.multinomial(
                probs[t], 
                num_samples=num_candidates,
                replacement=True,
            )  # (N,)
            samples_per_t.append(a_t)
        actions_TH = torch.stack(samples_per_t, dim=0)       # (H, N)
        candidate_actions = actions_TH.t().contiguous()      # (N, H)

        # -------------------------------
        # 2) Roll out in latent space
        # -------------------------------
        with torch.no_grad():
            z = z_start.expand(num_candidates, -1).contiguous()        # (N, D)
            z_goal_exp = z_goal.expand(num_candidates, -1).contiguous()# (N, D)

            for t in range(horizon):
                a_t_idx = candidate_actions[:, t]   # (N,)
                z = predictor(z, a_t_idx)           # (N, D)

            # terminal cost: squared distance to goal
            costs = torch.sum((z - z_goal_exp) ** 2, dim=-1)  # (N,)

        # track global best
        min_cost, min_idx = torch.min(costs, dim=0)
        if best_cost is None or min_cost < best_cost:
            best_cost = min_cost
            best_seq = candidate_actions[min_idx].detach().clone()

        # -------------------------------
        # 3) Select elites
        # -------------------------------
        elite_costs, elite_idx = torch.topk(
            costs,
            k=min(num_elites, num_candidates),
            largest=False,   # smallest cost = best
        )
        elite_actions = candidate_actions[elite_idx]   # (E, H)

        # -------------------------------
        # 4) Update probs from elites
        # -------------------------------
        for t in range(horizon):
            a_t = elite_actions[:, t]  # (E,)
            counts = torch.bincount(
                a_t,
                minlength=num_actions,
            ).float()                  # (num_actions,)
            probs[t] = counts / counts.sum()

        # small smoothing to avoid zero-prob actions
        eps = 1e-3
        probs = probs + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)

    return best_seq


# -------------------------------------------------------------
#  Closed-loop MPC rollout
# -------------------------------------------------------------

def mpc_rollout(
    env: Toy2DEnv,
    encoder,
    processor,
    predictor: nn.Module,
    z_goal: torch.Tensor,
    num_actions: int,
    num_candidates: int,
    horizon: int,
    max_env_steps: int,
    device: torch.device,
    stop_threshold: float = 0.1,
    num_elites: int = 64,
    cem_iters: int = 3,
) -> List[Image.Image]:
    """
    Run closed-loop MPC (with CEM) and return list of frames (PIL Images).
    """
    frames: List[Image.Image] = []

    # Reset env and grab initial frame
    obs = env.reset()
    cur_img = Image.fromarray(obs)
    frames.append(cur_img.copy())

    step = 0

    while step < max_env_steps:
        # 1) Encode current obs → z_t
        with torch.no_grad():
            z_t = encode_image_to_latent(cur_img, encoder, processor, device)

        # 2) Plan from z_t using CEM (NO args.* here)
        best_seq = plan_cem_latent(
            z_start=z_t,
            z_goal=z_goal,
            predictor=predictor,
            num_actions=num_actions,
            horizon=horizon,
            device=device,
            num_candidates=num_candidates,
            num_elites=num_elites,
            num_iterations=cem_iters,
        )

        # 3) Execute FIRST action
        a_int = int(best_seq[0].item())
        obs, reward, done, info = env.step(a_int)
        cur_img = Image.fromarray(obs)
        frames.append(cur_img.copy())
        step += 1

        # 4) Check latent distance to goal for early stop
        with torch.no_grad():
            z_cur = encode_image_to_latent(cur_img, encoder, processor, device)
            dist = torch.norm(z_cur - z_goal, dim=-1).item()
        print(f"[15a] step={step} latent_dist={dist:.4f}", flush=True)

        if dist < stop_threshold:
            print(f"[15a] Stopping: latent distance {dist:.4f} < {stop_threshold}", flush=True)
            break

        if done:
            print(f"[15a] Env done at step {step}", flush=True)
            break

    return frames



# -------------------------------------------------------------
#  GIF helper
# -------------------------------------------------------------

def save_gif(frames: List[Image.Image], out_path: str, fps: int = 5):
    if len(frames) == 0:
        print("[14d] No frames to save.")
        return
    durations = [1.0 / fps] * len(frames)
    imageio.mimsave(out_path, frames, duration=durations)
    print(f"[14d] Saved GIF to {out_path} (frames={len(frames)}, fps={fps})")


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
        "--stop-threshold",
        type=float,
        #123.3 OPTIONAL ######################################
        #For a more obvious “success” in this imperfect latent geometry, you could try:
        default=0.1,
        #default=1.3,
        #default=3.3,
        ######################################
        help="Latent-space distance threshold for success.",
    )
    parser.add_argument(
        "--out-gif",
        type=str,
        default="14d_toy2d_mpc.gif",
        help="Output GIF filename.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    
##############################################################
# #122 3. Add CEM-specific CLI args
##############################################################
    parser.add_argument(
        "--num-elites",
        type=int,
        default=64,
        help="Number of elite sequences used to update CEM distribution.",
    )
    parser.add_argument(
        "--cem-iters",
        type=int,
        default=3,
        help="Number of CEM refinement iterations per MPC step.",
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
    print(f"[14d] Loaded predictor ckpt: z_dim={z_dim}, num_actions={num_actions}", flush=True)

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

    # 4) Sample a goal position and render goal image
    gx = np.random.randint(0, env.grid_size)
    gy = np.random.randint(0, env.grid_size)
    goal_pos = (gx, gy)
    print(f"[14d] Goal grid position: {goal_pos}", flush=True)
    
    # #123.2 ###################################
    # Tell env to render the goal in red
    env.set_goal(goal_pos)
    ###################################

    goal_obs = env.render_goal_image(goal_pos)
    goal_img = Image.fromarray(goal_obs)

    # Encode goal to latent
    with torch.no_grad():
        z_goal = encode_image_to_latent(goal_img, encoder, processor, device)

    # 5) Run closed-loop MPC rollout
    frames = mpc_rollout(
        env=env,
        encoder=encoder,
        processor=processor,
        predictor=predictor,
        z_goal=z_goal,
        num_actions=num_actions,
        num_candidates=args.num_candidates,
        horizon=args.horizon,
        max_env_steps=args.max_env_steps,
        device=device,
        stop_threshold=args.stop_threshold,
        ############################################
        num_elites=args.num_elites,
        cem_iters=args.cem_iters,
    )

    # 6) Save GIF
    save_gif(frames, args.out_gif, fps=5)


if __name__ == "__main__":
    main()
