"""
36_D42c_belief_2d_planner.py

D42c — 2D belief + occlusion + tiny planner (CPU only)

This extends D42b:

    - SAME:
        * 2D grid belief over object position
        * cone-shaped FOV with range + half-angle
        * occlusion from a rectangular obstacle
        * Bayesian-style update (prediction + sensor likelihood)
        * matplotlib visualization of:
            - world (robot, object, obstacle, FOV)
            - belief heatmap

    - NEW in D42c:
        * A tiny "planner" that uses the belief map:
            - When belief is LOW-confidence → robot does the old scan pattern
            - When belief is HIGH-confidence → robot moves toward the MAP cell
              (the cell with maximum belief probability).

    Intuition:
        D42b = "pure perception": robot scans, belief updates
        D42c = "perception + control": belief drives motion
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Wedge
except ImportError:
    plt = None


# ================================================================
# 1. World2D definition
# ================================================================

@dataclass
class World2D:
    width: int
    height: int

    robot_x: float
    robot_y: float
    robot_theta: float  # radians, 0 = +x

    object_x: int
    object_y: int

    # obstacle rectangle as (x_min, y_min, x_max, y_max)
    obstacle_rect: Tuple[int, int, int, int]

    fov_range: float
    fov_half_angle_deg: float


# ================================================================
# 2. Geometry helpers: FOV + occlusion
# ================================================================

def angle_diff(a: float, b: float) -> float:
    """
    Smallest signed angle difference a - b in [-pi, pi].
    """
    d = a - b
    while d > np.pi:
        d -= 2.0 * np.pi
    while d < -np.pi:
        d += 2.0 * np.pi
    return d


def line_intersects_rect(
    x0: float, y0: float,
    x1: float, y1: float,
    rect: Tuple[int, int, int, int],
    num_samples: int = 32,
) -> bool:
    """
    Approximate occlusion test:
        - sample points along the segment from (x0, y0) to (x1, y1)
        - if any sample lies inside the rectangle → treat as occluded.

    rect = (xmin, ymin, xmax, ymax)
    """
    xmin, ymin, xmax, ymax = rect
    ts = np.linspace(0.0, 1.0, num_samples)
    xs = x0 + (x1 - x0) * ts
    ys = y0 + (y1 - y0) * ts

    inside = (xs >= xmin) & (xs <= xmax) & (ys >= ymin) & (ys <= ymax)
    return bool(np.any(inside))


def is_cell_visible(world: World2D, ix: int, iy: int) -> bool:
    """
    Check if cell (ix, iy) is inside FOV cone and not occluded by obstacle.

    FOV:
        - distance <= fov_range
        - angular difference |angle_to_cell - robot_theta| <= fov_half_angle
    """
    rx, ry = world.robot_x, world.robot_y
    dx = ix - rx
    dy = iy - ry
    dist = np.hypot(dx, dy)

    if dist <= 1e-6:
        return True  # it's where the robot is

    # range check
    if dist > world.fov_range:
        return False

    # angle check
    ang = np.arctan2(dy, dx)
    d_ang = angle_diff(ang, world.robot_theta)
    half = np.deg2rad(world.fov_half_angle_deg)
    if abs(d_ang) > half:
        return False

    # occlusion by rectangular obstacle
    if line_intersects_rect(rx, ry, ix, iy, world.obstacle_rect):
        return False

    return True


def visible_cells(world: World2D) -> List[Tuple[int, int]]:
    """
    Return list of (iy, ix) cells robot can SEE given FOV + obstacle.
    """
    vis = []
    for iy in range(world.height):
        for ix in range(world.width):
            if is_cell_visible(world, ix, iy):
                vis.append((iy, ix))
    return vis


def observe_object(world: World2D) -> Optional[Tuple[int, int]]:
    """
    Return observed object cell if visible, else None.
    """
    if is_cell_visible(world, world.object_x, world.object_y):
        return (world.object_y, world.object_x)  # (iy, ix)
    return None


# ================================================================
# 3. Belief2D
# ================================================================

class Belief2D:
    """
    2D belief over object position.

    B[iy, ix] = probability object is in cell (ix, iy).
    sum over all cells = 1.
    """

    def __init__(self, height: int, width: int):
        self.belief = np.ones((height, width), dtype=np.float64)
        self.belief /= self.belief.sum()

    def predict(self, diffusion_prob: float = 0.1):
        """
        Prediction step:
            - diffuse a little mass to 4-neighbors (up, down, left, right)
            - reflect at boundaries
        """
        B = self.belief
        H, W = B.shape
        new_B = np.zeros_like(B)

        stay_prob = 1.0 - diffusion_prob
        move_prob = diffusion_prob / 4.0

        for iy in range(H):
            for ix in range(W):
                # stay
                new_B[iy, ix] += stay_prob * B[iy, ix]

                # up
                if iy - 1 >= 0:
                    new_B[iy - 1, ix] += move_prob * B[iy, ix]
                else:
                    new_B[iy, ix] += move_prob * B[iy, ix]  # reflect

                # down
                if iy + 1 < H:
                    new_B[iy + 1, ix] += move_prob * B[iy, ix]
                else:
                    new_B[iy, ix] += move_prob * B[iy, ix]

                # left
                if ix - 1 >= 0:
                    new_B[iy, ix - 1] += move_prob * B[iy, ix]
                else:
                    new_B[iy, ix] += move_prob * B[iy, ix]

                # right
                if ix + 1 < W:
                    new_B[iy, ix + 1] += move_prob * B[iy, ix]
                else:
                    new_B[iy, ix] += move_prob * B[iy, ix]

        # normalize
        s = new_B.sum()
        if s <= 0.0:
            new_B[:] = 1.0 / (H * W)
        else:
            new_B /= s

        self.belief = new_B

    def update(
        self,
        world: World2D,
        observation: Optional[Tuple[int, int]],
        visible_indices: List[Tuple[int, int]],
        hit_boost: float = 5.0,
        miss_decay: float = 0.3,
    ):
        """
        Sensor update:

        Inputs:
            - observation: (iy, ix) if object seen, else None
            - visible_indices: cells currently visible

        Rules:
            - If observation is not None:
                * B[obs] *= hit_boost
                * For other visible cells v != obs: B[v] *= miss_decay
            - If observation is None:
                * For all visible cells v: B[v] *= miss_decay
            - Then renormalize.
        """
        B = self.belief
        if observation is not None:
            oy, ox = observation
            for (iy, ix) in visible_indices:
                if (iy, ix) == (oy, ox):
                    B[iy, ix] *= hit_boost
                else:
                    B[iy, ix] *= miss_decay
        else:
            for (iy, ix) in visible_indices:
                B[iy, ix] *= miss_decay

        s = B.sum()
        if s <= 0.0:
            H, W = B.shape
            B[:] = 1.0 / (H * W)
        else:
            B /= s

        self.belief = B

    # NEW in D42c: helper to get the MAP cell
    def map_cell(self) -> Tuple[int, int]:
        """
        Return (iy, ix) of the maximum a posteriori (MAP) cell.
        """
        B = self.belief
        H, W = B.shape
        k = int(np.argmax(B))
        iy, ix = divmod(k, W)
        return iy, ix


# ================================================================
# 4. Robot motion: scan policy (old) + planner (new)
# ================================================================

def robot_scan_policy_2d(t: int, world: World2D) -> Tuple[float, float, float]:
    """
    D42b-style deterministic scan:
        - robot moves along x from left to right, then right to left
        - y stays fixed
        - theta = 0 when moving right, pi when moving left
    """
    left_x = 4
    right_x = world.width - 5  # e.g., 25 if width=30
    span = right_x - left_x
    period = 2 * span

    m = t % period
    if m < span:
        x = left_x + m
        theta = 0.0
    else:
        x = right_x - (m - span)
        theta = np.pi

    y = 4.0
    return float(x), float(y), float(theta)


# NEW in D42c: planner that moves toward MAP cell
def move_toward_map_cell(
    world: World2D,
    belief: Belief2D,
    max_step: float = 0.7,
) -> Tuple[float, float, float]:
    """
    Tiny planner:
        - find MAP cell (iy*, ix*)
        - move robot a small step toward that cell
        - set heading theta toward that cell
    """
    iy_star, ix_star = belief.map_cell()

    rx, ry = world.robot_x, world.robot_y
    tx, ty = float(ix_star), float(iy_star)

    dx = tx - rx
    dy = ty - ry
    dist = np.hypot(dx, dy)
    if dist < 1e-6:
        # already at target cell center
        theta = world.robot_theta
        return rx, ry, theta

    step = min(max_step, dist)
    ux = dx / dist
    uy = dy / dist

    new_x = rx + step * ux
    new_y = ry + step * uy
    new_theta = np.arctan2(uy, ux)
    return new_x, new_y, new_theta


# NEW in D42c: policy that switches from scan → planner
def robot_policy_2d_with_belief(
    t: int,
    world: World2D,
    belief: Belief2D,
    confidence_threshold: float = 0.3,
) -> Tuple[float, float, float]:
    """
    If belief is still low-confidence (max(B) < threshold) → scan.
    If belief is high-confidence                   → move toward MAP cell.
    """
    max_prob = float(belief.belief.max())

    if max_prob < confidence_threshold:
        # Exploration (same behavior as D42b)
        return robot_scan_policy_2d(t, world)
    else:
        # Exploit belief: move toward the most likely cell
        return move_toward_map_cell(world, belief)


# ================================================================
# 5. Plotting
# ================================================================

def plot_world_and_belief(
    world: World2D,
    belief: Belief2D,
    t: int,
    observation: Optional[Tuple[int, int]],
    ax_world,
    ax_belief,
):
    if plt is None:
        return

    ax_world.cla()
    ax_belief.cla()

    # --- Belief heatmap ---
    B = belief.belief
    im = ax_belief.imshow(
        B,
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.25 * B.max(), B.max()),
        extent=[0, world.width, 0, world.height],
        interpolation="nearest",
    )
    ax_belief.set_title("Belief over object position")
    ax_belief.set_xlabel("x")
    ax_belief.set_ylabel("y")
    # NOTE (D42c): no colorbar per frame to avoid Tk/toolbar crashes
    # If you really want a colorbar, create it once outside the loop.


    # # --- Belief heatmap ---
    # B = belief.belief
    # im = ax_belief.imshow(
    #     B,
    #     origin="lower",
    #     cmap="viridis",
    #     vmin=0.0,
    #     vmax=max(0.25 * B.max(), B.max()),
    #     extent=[0, world.width, 0, world.height],
    #     interpolation="nearest",
    # )
    # ax_belief.set_title("Belief over object position")
    # ax_belief.set_xlabel("x")
    # ax_belief.set_ylabel("y")
    # plt.colorbar(im, ax=ax_belief, fraction=0.046, pad=0.04)

    # --- World view ---
    ax_world.set_xlim(0, world.width)
    ax_world.set_ylim(0, world.height)
    ax_world.set_aspect("equal")

    # obstacle rectangle
    oxmin, oymin, oxmax, oymax = world.obstacle_rect
    rect = Rectangle(
        (oxmin, oymin),
        oxmax - oxmin,
        oymax - oymin,
        linewidth=1.5,
        edgecolor="k",
        facecolor="gray",
        alpha=0.7,
    )
    ax_world.add_patch(rect)

    # object
    ax_world.scatter(
        world.object_x + 0.5,
        world.object_y + 0.5,
        s=80,
        c="red",
        marker="x",
        label="object",
    )

    # robot
    ax_world.scatter(
        world.robot_x,
        world.robot_y,
        s=60,
        c="blue",
        marker="o",
        label="robot",
    )

    # FOV wedge (for visualization only, not exact sampling)
    fov_th = world.robot_theta
    half = world.fov_half_angle_deg
    wedge = Wedge(
        center=(world.robot_x, world.robot_y),
        r=world.fov_range,
        theta1=np.degrees(fov_th) - half,
        theta2=np.degrees(fov_th) + half,
        color="blue",
        alpha=0.1,
    )
    ax_world.add_patch(wedge)

    # observation marker
    if observation is not None:
        oy, ox = observation
        ax_world.scatter(
            ox + 0.5,
            oy + 0.5,
            s=60,
            c="magenta",
            marker="o",
            label="obs",
        )

    ax_world.set_title(
        f"D42c — 2D Belief + Occlusion + Planner   t={t}   "
        f"robot=({world.robot_x:.1f}, {world.robot_y:.1f}), "
        f"theta={np.degrees(world.robot_theta):.1f} deg"
    )
    ax_world.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.pause(0.05)


# ================================================================
# 6. run_sim
# ================================================================

def run_sim(
    num_steps: int = 80,
    width: int = 30,
    height: int = 30,
    object_xy: Tuple[int, int] = (22, 20),
    obstacle_rect: Tuple[int, int, int, int] = (14, 17, 10, 22),
    fov_range: float = 25.0,
    fov_half_angle_deg: float = 80.0,
    plot: bool = True,
):
    ox, oy = object_xy

    world = World2D(
        width=width,
        height=height,
        robot_x=4.0,
        robot_y=4.0,
        robot_theta=0.0,
        object_x=ox,
        object_y=oy,
        obstacle_rect=obstacle_rect,
        fov_range=fov_range,
        fov_half_angle_deg=fov_half_angle_deg,
    )

    belief = Belief2D(height=height, width=width)

    print("[D42c] Starting 2D belief + occlusion + planner demo")
    print(f"  grid size      = {width} x {height}")
    print(f"  object_xy      = ({ox}, {oy})")
    print(f"  obstacle_rect  = {obstacle_rect}")
    print(f"  fov_range      = {fov_range}")
    print(f"  fov_half_angle = {fov_half_angle_deg:.1f} deg\n")

    if plot and plt is not None:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), num=1)
        ax_world, ax_belief = axes

    for t in range(num_steps):
        # 1) Move robot based on policy that uses BELIEF (NEW in D42c)
        new_x, new_y, new_theta = robot_policy_2d_with_belief(t, world, belief)
        world.robot_x = new_x
        world.robot_y = new_y
        world.robot_theta = new_theta

        # 2) Observe
        vis = visible_cells(world)
        obs = observe_object(world)

        # 3) Predict
        belief.predict(diffusion_prob=0.1)

        # 4) Update
        belief.update(world, observation=obs, visible_indices=vis)

        # 5) Logging (top 3 cells)
        flat = belief.belief.ravel()
        H, W = belief.belief.shape
        idx_sorted = np.argsort(-flat)
        top3 = idx_sorted[:3]
        top_str_parts = []
        for k in top3:
            iy, ix = divmod(int(k), W)
            top_str_parts.append(f"({ix:02d},{iy:02d}):{belief.belief[iy, ix]:.2f}")
        top_str = ", ".join(top_str_parts)

        print(
            f"[t={t:03d}] robot=({world.robot_x:4.1f},{world.robot_y:4.1f}) "
            f"theta={np.degrees(world.robot_theta):5.2f}  "
            f"obs={obs}  "
            f"top belief cells: {top_str}"
        )

        # 6) Plot
        if plot and plt is not None:
            plot_world_and_belief(world, belief, t, obs, ax_world, ax_belief)

    if plot and plt is not None:
        plt.ioff()
        plt.show()

    # Final summary
    print("\n[D42c] Final belief (top 5 cells):")
    flat = belief.belief.ravel()
    H, W = belief.belief.shape
    idx_sorted = np.argsort(-flat)
    top5 = idx_sorted[:5]
    for k in top5:
        iy, ix = divmod(int(k), W)
        print(f"  cell ({ix:02d},{iy:02d}): belief={belief.belief[iy, ix]:.3f}")
    print(f"True object index = ({ox}, {oy})")


# ================================================================
# 7. Main
# ================================================================

def main():
    run_sim(
        num_steps=80,
        width=30,
        height=30,
        object_xy=(22, 20),
        obstacle_rect=(14, 17, 24, 22),  # (xmin, ymin, xmax, ymax)
        fov_range=25.0,
        fov_half_angle_deg=80.0,
        plot=True,
    )


if __name__ == "__main__":
    main()
