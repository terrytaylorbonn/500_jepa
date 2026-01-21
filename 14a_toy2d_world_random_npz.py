"""
14a_toy2d_world_random_npz.py

Fresh, self-contained Toy2D environment + random data collection.

- Does NOT touch or import any 13_* files.
- Defines its own Toy2DEnv.
- Collects random (obs_t, action, obs_tp1) transitions.
- Saves to .npz for later JEPA latent training (14b) and planning (14c/14d).

Saved arrays in .npz:
  obs_t   : (N, H, W, 3) uint8
  obs_tp1 : (N, H, W, 3) uint8
  actions : (N,) int64

Usage:

  python 14a_toy2d_world_random_npz.py \\
      --out ./data/toy2d_14a_data.npz \\
      --num-transitions 1000 \\
      --num-actions 4
"""

import os
import argparse
from typing import Tuple, List

import numpy as np


# -------------------------------------------------------------
#  Simple self-contained Toy2DEnv
# -------------------------------------------------------------

class Toy2DEnv:
    """
    Minimal 2D grid world with a single moving agent.

    - Grid: grid_size x grid_size (logical)
    - Render: img_size x img_size RGB
    - Actions: 0=up, 1=down, 2=left, 3=right (num_actions=4)
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
          done   : bool    (always False; planning controls termination)
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


# -------------------------------------------------------------
#  Random data collection
# -------------------------------------------------------------

def collect_random_data(
    env: Toy2DEnv,
    num_transitions: int,
    num_actions: int,
    max_episode_len: int = 100,
    seed: int = 0,
):
    """
    Run a random policy to collect num_transitions transitions.

    Each transition:
      (obs_t, action, obs_tp1)

    Returns:
      obs_t_list, obs_tp1_list, actions_list
    """
    rng = np.random.default_rng(seed)

    obs_t_list: List[np.ndarray] = []
    obs_tp1_list: List[np.ndarray] = []
    actions_list: List[int] = []

    transitions_collected = 0

    while transitions_collected < num_transitions:
        obs = env.reset()
        obs_t = obs.copy()
        done = False
        episode_len = 0

        while not done and transitions_collected < num_transitions:
            a = int(rng.integers(0, num_actions))  # random int in [0, num_actions-1]

            obs_tp1, reward, done, info = env.step(a)

            obs_t_list.append(obs_t)
            obs_tp1_list.append(obs_tp1.copy())
            actions_list.append(a)

            transitions_collected += 1
            episode_len += 1
            obs_t = obs_tp1.copy()

            if episode_len >= max_episode_len:
                # force new episode to keep diversity
                break

        print(f"Collected {transitions_collected}/{num_transitions} transitions", end="\r")

    print()  # newline
    return obs_t_list, obs_tp1_list, actions_list


# -------------------------------------------------------------
#  Main
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npz path, e.g., ./data/toy2d_14a_data.npz",
    )
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=1000,
        help="Number of transitions (s, a, s') to collect.",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=4,
        help="Number of discrete actions (0..num_actions-1).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="Image size (pixels) for observations (H=W=img_size).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=12,
        help="Logical grid size (grid_size x grid_size).",
    )
    parser.add_argument(
        "--max-episode-len",
        type=int,
        default=100,
        help="Max steps per episode before reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data collection.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 1) Create environment (self-contained; does not use 13_* files)
    env = Toy2DEnv(
        img_size=args.img_size,
        grid_size=args.grid_size,
        num_actions=args.num_actions,
    )

    # 2) Collect random data
    obs_t_list, obs_tp1_list, actions_list = collect_random_data(
        env=env,
        num_transitions=args.num_transitions,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len,
        seed=args.seed,
    )

    # 3) Stack to arrays
    obs_t_arr = np.stack(obs_t_list, axis=0).astype(np.uint8)      # (N, H, W, 3)
    obs_tp1_arr = np.stack(obs_tp1_list, axis=0).astype(np.uint8)  # (N, H, W, 3)
    actions_arr = np.array(actions_list, dtype=np.int64)           # (N,)

    print("Final shapes:")
    print("  obs_t   :", obs_t_arr.shape, obs_t_arr.dtype)
    print("  obs_tp1 :", obs_tp1_arr.shape, obs_tp1_arr.dtype)
    print("  actions :", actions_arr.shape, actions_arr.dtype)

    # 4) Save to .npz
    np.savez_compressed(
        args.out,
        obs_t=obs_t_arr,
        obs_tp1=obs_tp1_arr,
        actions=actions_arr,
    )
    print(f"Saved data to {args.out}")


if __name__ == "__main__":
    main()

