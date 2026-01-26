"""
34_D42a_belief_1d_occlusion.py

D42a — 1D belief + occlusion (CPU only)

Toy 1D world:
    - discrete cells [0..N-1]
    - robot at index r
    - object at index o (static)
    - wall at index w (occluder)

Robot:
    - scans left/right over time (predefined pattern)

Observation:
    - robot has finite field-of-view (FOV radius)
    - can only see cells within FOV that are NOT behind wall
    - if wall is between robot and a cell, that cell is occluded

Belief:
    - array B[i] = belief object is at cell i
    - at each step:
        1) prediction: small diffusion (uncertainty)
        2) update: incorporate observation:
            - if object seen at cell k in visible FOV → boost belief at k,
              decay beliefs in visible cells != k
            - if object not seen in visible FOV → decay beliefs in visible cells
        3) renormalize

Plot:
    - top: world (robot, wall, object)
    - bottom: belief distribution across cells
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ================================================================
# 1. World1D // World definition
# ================================================================

@dataclass
class World1D:
    num_cells: int
    robot_idx: int
    object_idx: int
    wall_idx: int
    fov_radius: int


# ================================================================
# 2. robot_scan_policy // Robot scan policy
# ================================================================

def robot_scan_policy(t: int, world: World1D) -> int:
    """
    Simple scan pattern:
        - start near left
        - move right until near end
        - then move left, repeat

    This is a deterministic pattern over time for the demo.
    """

    # Effective range robot scans over
    left = 3
    right = world.num_cells - 4
    span = right - left

    # Sawtooth pattern over [left, right]
    period = 2 * span
    m = t % period
    if m < span:
        idx = left + m
    else:
        idx = right - (m - span)

    return int(idx)


# ================================================================
# 3. observe_object // Observation model with occlusion
# ================================================================

def visible_cells(world: World1D) -> List[int]:
    """
    Return the list of cells robot can SEE given its FOV and the wall occluder.

    Rules:
        - cells within |i - r| <= fov_radius FIELD OF VIEW
        - if wall lies strictly between robot and a cell, that cell is occluded
    """
    r = world.robot_idx
    w = world.wall_idx
    N = world.num_cells
    R = world.fov_radius

    vis = []
    for i in range(N):
        if abs(i - r) > R:
            continue

        # occlusion: if robot and cell on same side of wall → visible
        # if wall between them → occluded
        if (r < w <= i) or (i <= w < r):
            # wall is between robot and cell
            continue

        vis.append(i)
    return vis


def observe_object(world: World1D) -> Optional[int]:
    """
    Return observed cell index if object is visible, else None.
    """
    # If object cell is in visible set → we see it
    vis = visible_cells(world)
    if world.object_idx in vis:
        return world.object_idx
    return None


# ================================================================
# 4. Belief1D, predict, update // Belief representation and update
# ================================================================

class Belief1D:
    """
    Belief over object position in 1D cells.

    B[i] = belief object is at cell i, sum_i B[i] = 1.
    """

    def __init__(self, num_cells: int):
        # Start uniform
        self.belief = np.ones(num_cells, dtype=np.float64)
        self.belief /= self.belief.sum()

    def predict(self, diffusion_prob: float = 0.1):
        """
        Prediction step:
            - small diffusion to neighbors to model uncertainty.
        """
        B = self.belief
        N = len(B)
        new_B = np.zeros_like(B)

        stay_prob = 1.0 - diffusion_prob
        move_prob = diffusion_prob / 2.0

        for i in range(N):
            # stay
            new_B[i] += stay_prob * B[i]
            # move left
            if i - 1 >= 0:
                new_B[i - 1] += move_prob * B[i]
            else:
                new_B[i] += move_prob * B[i]  # reflect at boundary
            # move right
            if i + 1 < N:
                new_B[i + 1] += move_prob * B[i]
            else:
                new_B[i] += move_prob * B[i]  # reflect at boundary

        self.belief = new_B / new_B.sum()

    def update(
        self,
        world: World1D,
        observation: Optional[int],
        visible_indices: List[int],
        hit_boost: float = 4.0,
        miss_decay: float = 0.2,
    ):
        """
        Update step:

        Inputs:
            - observation: cell index where object is seen, or None
            - visible_indices: cells currently visible (FOV without occlusion)

        Rules:
            - If observation is not None:
                * multiply belief at observed cell by hit_boost
                * multiply beliefs at OTHER visible cells by miss_decay
            - If observation is None:
                * multiply beliefs at ALL visible cells by miss_decay
                * cells outside visible_indices unchanged

        Then renormalize.
        """
        B = self.belief
        if observation is not None:
            for i in visible_indices:
                if i == observation:
                    B[i] *= hit_boost
                else:
                    B[i] *= miss_decay
        else:
            for i in visible_indices:
                B[i] *= miss_decay

        # Renormalize
        s = B.sum()
        if s <= 0.0:
            # fallback: uniform
            B[:] = 1.0 / len(B)
        else:
            B /= s

        self.belief = B


# ================================================================
# 5. plot_world_and_belief // Plotting
# ================================================================

def plot_world_and_belief(
    world: World1D,
    belief: Belief1D,
    t: int,
    observation: Optional[int],
    ax_world,
    ax_belief,
):
    if plt is None:
        return

    N = world.num_cells
    cells = np.arange(N)

    # clear old frame
    ax_world.cla()
    ax_belief.cla()

    # --- World row ---
    colors = []
    for i in range(N):
        if i == world.robot_idx:
            colors.append("blue")
        elif i == world.object_idx:
            colors.append("red")
        elif i == world.wall_idx:
            colors.append("black")
        else:
            colors.append("lightgray")

    ax_world.bar(cells, 1.0, color=colors, edgecolor="k")
    ax_world.set_ylim(0, 1.2)
    ax_world.set_yticks([])
    ax_world.set_title(
        f"D42a — 1D Belief + Occlusion   t={t}   "
        f"robot={world.robot_idx}, object={world.object_idx}, wall={world.wall_idx}, obs={observation}"
    )

    # highlight visible cells
    vis = visible_cells(world)
    for i in vis:
        ax_world.text(i, 0.6, "V", ha="center", va="center", fontsize=8, color="green")

    if observation is not None:
        ax_world.text(
            observation,
            0.9,
            "OBS",
            ha="center",
            va="center",
            fontsize=8,
            color="magenta",
        )

    # --- Belief row ---
    ax_belief.bar(cells, belief.belief, color="orange", edgecolor="k")
    ax_belief.set_ylabel("Belief")
    ax_belief.set_xlabel("Cell index")
    ax_belief.set_ylim(0, max(0.25, belief.belief.max() * 1.2))

    plt.tight_layout()
    plt.pause(0.05)



# ================================================================
# 6. run_sim (robot_scan_policy, observe_object, predict, update, plot_world_and_belief) 
# ================================================================

def run_sim(
    num_steps: int = 120,
    num_cells: int = 30,
    object_idx: int = 20,
    wall_idx: int = 15,
    fov_radius: int = 3,
    plot: bool = True,
):
    # Initial world
    world = World1D(
        num_cells=num_cells,
        robot_idx=5,
        object_idx=object_idx,
        wall_idx=wall_idx,
        fov_radius=fov_radius,
    )

    belief = Belief1D(num_cells=num_cells)

    print("[D42a] Starting 1D belief + occlusion demo")
    print(f"  num_cells   = {num_cells}")
    print(f"  object_idx  = {object_idx}")
    print(f"  wall_idx    = {wall_idx}")
    print(f"  fov_radius  = {fov_radius}")
    print()

    if plot and plt is not None:
        plt.ion()
        fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True, num=1)
        ax_world, ax_belief = axes

    for t in range(num_steps):
        # 1) Move robot according to scan policy
        world.robot_idx = robot_scan_policy(t, world)

        # 2) Observation (with occlusion)
        vis = visible_cells(world)
        obs = observe_object(world)

        # 3) Belief prediction (small diffusion)
        belief.predict(diffusion_prob=0.1)

        # 4) Belief update based on observation
        belief.update(world, observation=obs, visible_indices=vis)

        # 5) Logging
        # Top-3 most likely cells
        top_indices = np.argsort(-belief.belief)[:3]
        top_str = ", ".join(f"{i}:{belief.belief[i]:.2f}" for i in top_indices)

        # obs_str = "None" if obs is None else f"{obs:02d}"
        # print(
        #     f"[t={t:03d}] robot={world.robot_idx:02d} "
        #     f"obs={obs_str}  "
        #     f"top belief cells: {top_str}"
        # )

        print(
            f"[t={t:03d}] robot={world.robot_idx:02d} "
            f"obs={obs if obs is None else f'{obs:02d}'}  "
            # f"obs={'None' if obs is None else obs:02d}  "
            f"top belief cells: {top_str}"
        )

        # 6) Plot
        if plot and plt is not None:
            plot_world_and_belief(world, belief, t, obs, ax_world, ax_belief)

    if plot and plt is not None:
        plt.ioff()
        plt.show()

    if plot and plt is not None:
        plt.ioff()
        plt.show()

    print("\n[D42a] Final belief (top 5 cells):")
    top_indices = np.argsort(-belief.belief)[:5]
    for i in top_indices:
        print(f"  cell {i:02d}: belief={belief.belief[i]:.3f}")
    print(f"True object index = {world.object_idx}")


# ================================================================
# 7. (run_sim) Main 
# ================================================================

def main():
    run_sim(
        num_steps=120,
        num_cells=30,
        object_idx=20,
        wall_idx=15,
        fov_radius=3,
        plot=True,
    )


if __name__ == "__main__":
    main()
