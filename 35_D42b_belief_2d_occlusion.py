"""
35_D42b_belief_2d_occlusion.py

D42b — 2D belief + occlusion cones (CPU only)

Toy 2D world:
    - discrete grid (width x height)
    - robot at (x, y) with heading theta
    - one rectangular obstacle that can occlude line-of-sight
    - one static object

Robot:
    - scans left/right along the bottom of the grid (predefined pattern)
    - heading points along motion direction (right or left)

Observation:
    - robot has finite FOV (range + half-angle)
    - a cell is visible if:
        1) inside FOV cone (distance + angle test)
        2) line-of-sight ray from robot to cell does NOT intersect obstacle

Belief:
    - grid B[y, x] = belief object is at cell (x, y)
    - at each step:
        1) prediction: small diffusion to neighbors (uncertainty)
        2) update with observation:
            - if object seen at (ox, oy) → boost belief there,
              decay beliefs on other visible cells
            - if object not seen in visible FOV → decay beliefs on all visible cells
        3) renormalize

Plot:
    - left: world (robot, FOV-visible cells, wall, object)
    - right: belief heatmap
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ================================================================
# 1. Geometry helpers
# ================================================================

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class RectObstacle:
    """Axis-aligned rectangular obstacle."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)


# ================================================================
# 2. World2D definition
# ================================================================

@dataclass
class World2D:
    width: int
    height: int

    robot_x: float
    robot_y: float
    robot_theta: float  # radians

    object_x: float
    object_y: float

    obstacles: List[RectObstacle]

    fov_range: float
    fov_half_angle: float  # radians


# ================================================================
# 3. Robot scan policy in 2D
# ================================================================

def robot_scan_policy_2d(t: int, world: World2D) -> None:
    """
    Simple 2D scan pattern:

    - Robot moves horizontally along a line (near the bottom of the grid):
        x in [x_left, x_right], y fixed
    - It bounces left/right, like a sawtooth.
    - Heading is 0 when moving right, pi when moving left.
    """
    x_left = 4
    x_right = world.width - 5
    span = x_right - x_left

    base_y = 4  # row near bottom

    period = 2 * span
    m = t % period
    if m < span:
        # move right
        x = x_left + m
        theta = 0.0  # facing right
    else:
        # move left
        x = x_right - (m - span)
        theta = np.pi  # facing left

    world.robot_x = float(x)
    world.robot_y = float(base_y)
    world.robot_theta = theta


# ================================================================
# 4. Visibility (FOV cone + occlusion)
# ================================================================

def cell_center(ix: int, iy: int) -> Tuple[float, float]:
    """Convert integer grid indices to continuous cell center coordinates."""
    return float(ix) + 0.5, float(iy) + 0.5


def is_cell_in_fov(world: World2D, ix: int, iy: int) -> bool:
    """Check distance + angle (FOV cone) for cell (ix, iy)."""
    rx, ry = world.robot_x, world.robot_y
    cx, cy = cell_center(ix, iy)

    dx = cx - rx
    dy = cy - ry
    dist = np.sqrt(dx * dx + dy * dy)
    if dist <= 1e-6:
        return False
    if dist > world.fov_range:
        return False

    angle_to_cell = np.arctan2(dy, dx)
    diff = wrap_to_pi(angle_to_cell - world.robot_theta)
    if abs(diff) > world.fov_half_angle:
        return False

    return True


def is_line_occluded(world: World2D, ix: int, iy: int, num_steps: int = 20) -> bool:
    """
    Check if line from robot to cell center intersects any obstacle.

    Simple ray-marching:
        - sample a few points along the line (excluding the robot position)
        - if any sample lies inside an obstacle → occluded
    """
    rx, ry = world.robot_x, world.robot_y
    cx, cy = cell_center(ix, iy)

    for obs in world.obstacles:
        # sample along the line
        for s in np.linspace(0.05, 0.95, num_steps):
            px = rx + s * (cx - rx)
            py = ry + s * (cy - ry)
            if obs.contains(px, py):
                return True

    return False


def visible_cells_2d(world: World2D) -> List[Tuple[int, int]]:
    """
    Return list of (ix, iy) for cells visible from robot:

        - inside FOV cone (range + angle)
        - NOT occluded by obstacles (line-of-sight check)
    """
    vis: List[Tuple[int, int]] = []
    for iy in range(world.height):
        for ix in range(world.width):
            if not is_cell_in_fov(world, ix, iy):
                continue
            if is_line_occluded(world, ix, iy):
                continue
            vis.append((ix, iy))
    return vis


def observe_object_2d(world: World2D, visible_cells: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Return observed cell (ix, iy) if object is visible, else None.
    """
    ox = int(world.object_x)
    oy = int(world.object_y)
    if (ox, oy) in visible_cells:
        return (ox, oy)
    return None


# ================================================================
# 5. Belief2D representation and update
# ================================================================

class Belief2D:
    """
    Belief over object position in 2D grid.

    B[yy, xx] = belief object is at cell (xx, yy), sum over all cells = 1.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
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
                mass = B[iy, ix]
                # stay
                new_B[iy, ix] += stay_prob * mass
                # up
                if iy + 1 < H:
                    new_B[iy + 1, ix] += move_prob * mass
                else:
                    new_B[iy, ix] += move_prob * mass
                # down
                if iy - 1 >= 0:
                    new_B[iy - 1, ix] += move_prob * mass
                else:
                    new_B[iy, ix] += move_prob * mass
                # right
                if ix + 1 < W:
                    new_B[iy, ix + 1] += move_prob * mass
                else:
                    new_B[iy, ix] += move_prob * mass
                # left
                if ix - 1 >= 0:
                    new_B[iy, ix - 1] += move_prob * mass
                else:
                    new_B[iy, ix] += move_prob * mass

        new_B_sum = new_B.sum()
        if new_B_sum <= 0.0:
            new_B[:] = 1.0 / (H * W)
        else:
            new_B /= new_B_sum

        self.belief = new_B

    def update(
        self,
        observation: Optional[Tuple[int, int]],
        visible_indices: List[Tuple[int, int]],
        hit_boost: float = 4.0,
        miss_decay: float = 0.2,
    ):
        """
        Update step:

        - observation: (ox, oy) if object seen, else None
        - visible_indices: [(ix, iy)] currently visible cells (FOV, no occlusion)

        Rules:
            - If observation is not None:
                * multiply belief at observed cell by hit_boost
                * multiply beliefs at OTHER visible cells by miss_decay
            - If observation is None:
                * multiply beliefs at ALL visible cells by miss_decay
            - cells outside visible_indices unchanged

        Then renormalize.
        """
        B = self.belief

        if observation is not None:
            ox, oy = observation
            for (ix, iy) in visible_indices:
                if ix == ox and iy == oy:
                    B[iy, ix] *= hit_boost
                else:
                    B[iy, ix] *= miss_decay
        else:
            for (ix, iy) in visible_indices:
                B[iy, ix] *= miss_decay

        s = B.sum()
        if s <= 0.0:
            H, W = B.shape
            B[:] = 1.0 / (H * W)
        else:
            B /= s

        self.belief = B


# ================================================================
# 6. Plotting
# ================================================================

def render_world_grid(
    world: World2D,
    visible_indices: List[Tuple[int, int]],
    observation: Optional[Tuple[int, int]],
) -> np.ndarray:
    """
    Build an integer grid encoding for visualization:

        0 = empty
        1 = obstacle
        2 = FOV-visible cell
        3 = object
        4 = robot
        5 = observed object cell
    """
    H = world.height
    W = world.width
    grid = np.zeros((H, W), dtype=np.int32)

    # obstacles
    for obs in world.obstacles:
        x_min = max(0, int(obs.x_min))
        x_max = min(W - 1, int(obs.x_max))
        y_min = max(0, int(obs.y_min))
        y_max = min(H - 1, int(obs.y_max))
        grid[y_min:y_max + 1, x_min:x_max + 1] = 1

    # FOV visible cells
    for (ix, iy) in visible_indices:
        if grid[iy, ix] == 0:
            grid[iy, ix] = 2

    # object
    ox = int(world.object_x)
    oy = int(world.object_y)
    if 0 <= ox < W and 0 <= oy < H:
        grid[oy, ox] = 3

    # robot
    rx = int(world.robot_x)
    ry = int(world.robot_y)
    if 0 <= rx < W and 0 <= ry < H:
        grid[ry, rx] = 4

    # observed object
    if observation is not None:
        ox_obs, oy_obs = observation
        if 0 <= ox_obs < W and 0 <= oy_obs < H:
            grid[oy_obs, ox_obs] = 5

    return grid


def plot_world_and_belief(
    world: World2D,
    belief: Belief2D,
    t: int,
    visible_indices: List[Tuple[int, int]],
    observation: Optional[Tuple[int, int]],
    ax_world,
    ax_belief,
):
    if plt is None:
        return

    H = world.height
    W = world.width

    # clear old frame
    ax_world.cla()
    ax_belief.cla()

    # --- World plot ---
    grid = render_world_grid(world, visible_indices, observation)

    im1 = ax_world.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
    )
    ax_world.set_title(
        f"D42b — 2D Belief + Occlusion   t={t}   "
        f"robot=({world.robot_x:.1f},{world.robot_y:.1f}), "
        f"theta={world.robot_theta:.2f} rad, obs={observation}"
    )
    ax_world.set_xticks(range(0, W, 5))
    ax_world.set_yticks(range(0, H, 5))

    # --- Belief plot ---
    im2 = ax_belief.imshow(
        belief.belief,
        origin="lower",
        interpolation="nearest",
    )
    ax_belief.set_title("Belief over object position")
    ax_belief.set_xticks(range(0, W, 5))
    ax_belief.set_yticks(range(0, H, 5))

    plt.tight_layout()
    plt.pause(0.05)


# ================================================================
# 7. Simulation
# ================================================================

def run_sim_2d(
    num_steps: int = 80,
    width: int = 30,
    height: int = 30,
    object_xy: Tuple[int, int] = (22, 20),
    obstacle_rect: Tuple[int, int, int, int] = (14, 17, 10, 22),
    fov_range: float = 25.0,
    fov_half_angle_deg: float = 80.0, ###############40.0,
    plot: bool = True,
):
    # World setup
    ox, oy = object_xy
    (oxmin, oxmax, oymin, oymax) = obstacle_rect

    world = World2D(
        width=width,
        height=height,
        robot_x=5.0,
        robot_y=4.0,
        robot_theta=0.0,
        object_x=float(ox),
        object_y=float(oy),
        obstacles=[
            RectObstacle(
                x_min=float(oxmin),
                x_max=float(oxmax),
                y_min=float(oymin),
                y_max=float(oymax),
            )
        ],
        fov_range=fov_range,
        fov_half_angle=np.deg2rad(fov_half_angle_deg),
    )

    belief = Belief2D(width=width, height=height)

    print("[D42b] Starting 2D belief + occlusion demo")
    print(f"  grid size      = {width} x {height}")
    print(f"  object_xy      = {object_xy}")
    print(f"  obstacle_rect  = {obstacle_rect}")
    print(f"  fov_range      = {fov_range}")
    print(f"  fov_half_angle = {fov_half_angle_deg} deg")
    print()

    ax_world = ax_belief = None
    if plot and plt is not None:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), num=1)
        ax_world, ax_belief = axes

    for t in range(num_steps):
        # 1) Move robot
        robot_scan_policy_2d(t, world)

        # 2) Visible cells (cone + occlusion)
        vis_cells = visible_cells_2d(world)

        # 3) Observation
        obs = observe_object_2d(world, vis_cells)

        # 4) Belief prediction
        belief.predict(diffusion_prob=0.05)

        # 5) Belief update
        belief.update(observation=obs, visible_indices=vis_cells)

        # 6) Logging: top-3 belief cells
        flat = belief.belief.ravel()
        top_indices = np.argsort(-flat)[:3]
        top_str_parts = []
        H = height
        W = width
        for idx in top_indices:
            iy = idx // W
            ix = idx % W
            top_str_parts.append(f"({ix},{iy}):{belief.belief[iy, ix]:.2f}")
        top_str = ", ".join(top_str_parts)

        print(
            f"[t={t:03d}] robot=({world.robot_x:4.1f},{world.robot_y:4.1f}) "
            f"theta={world.robot_theta:5.2f}  "
            f"obs={obs}  "
            f"top belief cells: {top_str}"
        )

        # 7) Plot
        if plot and plt is not None:
            plot_world_and_belief(world, belief, t, vis_cells, obs, ax_world, ax_belief)

    if plot and plt is not None:
        plt.ioff()
        plt.show()

    # Final belief summary
    print("\n[D42b] Final belief (top 5 cells):")
    flat = belief.belief.ravel()
    top_indices = np.argsort(-flat)[:5]
    for idx in top_indices:
        iy = idx // width
        ix = idx % width
        print(f"  cell ({ix:02d},{iy:02d}): belief={belief.belief[iy, ix]:.3f}")
    print(f"True object index = ({int(world.object_x)}, {int(world.object_y)})")


# ================================================================
# 8. Main
# ================================================================

def main():
    run_sim_2d(
        num_steps=80,
        width=30,
        height=30,
        object_xy=(22, 20),
        obstacle_rect=(14, 17, 10, 22),
        fov_range=25.0,
        fov_half_angle_deg=80.0, ########## 40.0,
        plot=True,
    )


if __name__ == "__main__":
    main()
