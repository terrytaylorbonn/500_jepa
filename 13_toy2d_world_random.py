"""
13_toy2d_world_random.py

Collect random transitions from the Toy2D environment and save them as an .npz file.

Each transition:
  (obs_t, action, obs_tp1)

Saved arrays:
  - obs_t   : (N, H, W, 3) uint8
  - obs_tp1 : (N, H, W, 3) uint8
  - actions : (N,) int64

Usage example:

  python 13_toy2d_world_random.py \
      --out ./data/toy2d_data.npz \
      --num-transitions 1000 \
      --num-actions 4

"""

import os
import argparse
import numpy as np

from PIL import Image

# TODO: adjust this import to your actual env module
from toy2d_env import Toy2DEnv  # <-- change if needed


def to_uint8_rgb(obs):
    """
    Convert an observation to a uint8 (H, W, 3) RGB array.
    Handles:
      - NumPy arrays (H, W, 3)
      - PIL Images
    """
    if isinstance(obs, Image.Image):
        arr = np.array(obs)
    else:
        arr = np.array(obs)

    if arr.dtype != np.uint8:
        # assume 0..1 float, or something similar
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        # grayscale → RGB
        arr = np.stack([arr, arr, arr], axis=-1)

    assert arr.ndim == 3 and arr.shape[2] in (1, 3, 4), \
        f"Unexpected obs shape: {arr.shape}"

    if arr.shape[2] == 4:
        # RGBA → RGB (drop alpha)
        arr = arr[:, :, :3]

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    return arr


def collect_random_data(
    env,
    num_transitions: int,
    num_actions: int,
    max_episode_len: int = 100,
    seed: int = 0,
):
    """
    Run a random policy in env to collect num_transitions transitions.

    Returns:
      obs_t_list, obs_tp1_list, actions_list
    """
    rng = np.random.default_rng(seed)

    obs_t_list = []
    obs_tp1_list = []
    actions_list = []

    transitions_collected = 0

    while transitions_collected < num_transitions:
        obs = env.reset()
        obs_arr = to_uint8_rgb(obs)

        episode_len = 0
        done = False

        while not done and transitions_collected < num_transitions:
            # sample random action
            a = int(rng.integers(0, num_actions))  # 0 .. num_actions-1

            obs_next, reward, done, info = env.step(a)
            obs_next_arr = to_uint8_rgb(obs_next)

            obs_t_list.append(obs_arr)
            obs_tp1_list.append(obs_next_arr)
            actions_list.append(a)

            transitions_collected += 1
            episode_len += 1

            obs_arr = obs_next_arr

            if episode_len >= max_episode_len:
                # force reset to avoid super-long episodes
                break

        print(f"Collected {transitions_collected}/{num_transitions} transitions", end="\r")

    print()  # newline
    return obs_t_list, obs_tp1_list, actions_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output .npz path, e.g., ./data/toy2d_data.npz",
    )
    parser.add_argument(
        "--num-transitions",
        type=int,
        default=1000,
        help="Number of (s, a, s') transitions to collect.",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=4,
        help="Number of discrete actions in Toy2D (0..num_actions-1).",
    )
    parser.add_argument(
        "--max-episode-len",
        type=int,
        default=100,
        help="Max steps per episode before forced reset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data collection.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        help="(Optional) image size argument to Toy2DEnv, if applicable.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---------------------------------------------------------
    # 1) Create environment
    # ---------------------------------------------------------
    # TODO: adjust constructor args to match your Toy2DEnv
    env = Toy2DEnv(img_size=args.img_size)

    # ---------------------------------------------------------
    # 2) Collect transitions with random policy
    # ---------------------------------------------------------
    obs_t_list, obs_tp1_list, actions_list = collect_random_data(
        env=env,
        num_transitions=args.num_transitions,
        num_actions=args.num_actions,
        max_episode_len=args.max_episode_len,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # 3) Stack and save to .npz
    # ---------------------------------------------------------
    obs_t_arr = np.stack(obs_t_list, axis=0).astype(np.uint8)      # (N, H, W, 3)
    obs_tp1_arr = np.stack(obs_tp1_list, axis=0).astype(np.uint8)  # (N, H, W, 3)
    actions_arr = np.array(actions_list, dtype=np.int64)           # (N,)

    print("Final shapes:")
    print("  obs_t   :", obs_t_arr.shape, obs_t_arr.dtype)
    print("  obs_tp1 :", obs_tp1_arr.shape, obs_tp1_arr.dtype)
    print("  actions :", actions_arr.shape, actions_arr.dtype)

    np.savez_compressed(
        args.out,
        obs_t=obs_t_arr,
        obs_tp1=obs_tp1_arr,
        actions=actions_arr,
    )
    print(f"Saved data to {args.out}")


if __name__ == "__main__":
    main()
