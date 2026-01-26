"""
32bbb_D36c_subgoal_latent_mpc_gpu.py

D36c — Latent MPC on GPU + JEPA perception (improved)

Goal of this demo:
    - keep the same 2D toy world as D36a/D36b
    - keep the same "subgoal" idea (doorway vs direct goal)
    - BUT:
        * still infer subgoal in simple CPU geometry
        * use a GPU-based MPC-style planner in a low-dimensional latent state
          (here: the 2D (x, y) position treated as a latent vector)
        * run many candidate action sequences in parallel on GPU
        * choose the best sequence by cost and execute the first action

    - also keep JEPA perception on GPU (like D36b) to show:
        "scene → high-D JEPA z", even if planning uses a simpler 2D latent.

This version is slightly tuned so that:
    - candidate actions are BIASED toward the subgoal direction
    - the path is less squiggly and reliably reaches the doorway
    - we usually switch to the final goal within 500 steps

Dependencies:
    - Python 3.10+
    - torch
    - transformers (for JEPA)
    - pillow
    - matplotlib (optional, for plotting)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting is optional

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

Vec2 = Tuple[float, float]


# ================================================================
# Geometry helpers (same as D36a/D36b)
# ================================================================

def v_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def v_scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def v_len(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def v_norm(a: Vec2) -> Vec2:
    length = v_len(a)
    if length == 0.0:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def segment_point_dist(p: Vec2, a: Vec2, b: Vec2) -> float:
    """Distance from point p to segment a-b."""
    ax, ay = a
    bx, by = b
    px, py = p

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab_len2 = abx * abx + aby * aby
    if ab_len2 == 0.0:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab_len2
    t = max(0.0, min(1.0, t))

    closest_x = ax + abx * t
    closest_y = ay + aby * t
    return math.hypot(px - closest_x, py - closest_y)


def segment_intersects_circle(a: Vec2, b: Vec2, center: Vec2, radius: float, margin: float = 0.0) -> bool:
    return segment_point_dist(center, a, b) <= (radius + margin)


# ================================================================
# World objects
# ================================================================

@dataclass
class Obstacle:
    center: Vec2
    radius: float


@dataclass
class Robot:
    """
    Robot for the CPU "ground truth" world.
    For D36c, we treat robot.pos as the low-dimensional latent state.
    """
    pos: Vec2
    max_speed: float = 2.0  # max speed magnitude (used as action bound)


@dataclass
class World:
    """
    Simple 2D world:
        - robot position
        - final goal position
        - circular obstacles
        - waypoint candidates (e.g., doorway + final goal)
    """
    robot: Robot
    goal_final: Vec2
    obstacles: List[Obstacle]
    waypoint_candidates: List[Vec2]
    dt: float = 0.05


# ================================================================
# JEPA-based perception (GPU, same idea as D36b)
# ================================================================

class JEPAPerception:
    """
    Minimal wrapper around facebook/ijepa_vith14_1k:

    Responsibilities:
        - render a top-down image of the World (2D toy world → pixels)
        - encode it to a JEPA latent z
        - we log this latent; we do not yet use it for planning directly.

    This keeps the "scene → JEPA z" piece consistent across D36b/D36c.
    """

    def __init__(
        self,
        model_id: str = "facebook/ijepa_vith14_1k",
        device: str = "cuda",
    ):
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"[D36c] JEPA device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def embed_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Return a (1, D) JEPA latent from a PIL image."""
        pil_img = pil_img.convert("RGB")
        inputs = self.processor(pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model(**inputs)

        # CLS embedding: shape (1, D)
        z = out.last_hidden_state[:, 0, :]
        return z

    def render_world_to_image(self, world: "World", size: int = 256) -> Image.Image:
        """
        Very simple top-down rasterization:
            - white background
            - obstacles as gray circles
            - robot as blue disk
            - final goal as green disk
            - waypoint candidates as magenta squares
        """
        img = Image.new("RGB", (size, size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # World bounds: chosen to include start, goal, and doorway
        xmin, xmax = -10.0, 10.0
        ymin, ymax = -2.0, 8.0

        def world_to_px(p: Vec2) -> Tuple[int, int]:
            x, y = p
            u = (x - xmin) / (xmax - xmin)
            v = 1.0 - (y - ymin) / (ymax - ymin)
            return (int(u * (size - 1)), int(v * (size - 1)))

        # Obstacles
        for obs in world.obstacles:
            cx, cy = world_to_px(obs.center)
            r_px = int(obs.radius / (xmax - xmin) * size * 3.0)
            bbox = [cx - r_px, cy - r_px, cx + r_px, cy + r_px]
            draw.ellipse(bbox, outline=(128, 128, 128), width=2)

        # Waypoint candidates (magenta)
        for cand in world.waypoint_candidates:
            cx, cy = world_to_px(cand)
            r = 4
            draw.rectangle(
                [cx - r, cy - r, cx + r, cy + r],
                outline=(255, 0, 255),
                width=1,
            )

        # Final goal (green)
        gx, gy = world_to_px(world.goal_final)
        r = 5
        draw.ellipse(
            [gx - r, gy - r, gx + r, gy + r],
            outline=(0, 180, 0),
            width=2,
        )

        # Robot (blue)
        rx, ry = world_to_px(world.robot.pos)
        r = 5
        draw.ellipse(
            [rx - r, ry - r, rx + r, ry + r],
            outline=(0, 0, 255),
            width=2,
        )

        return img


# ================================================================
# CPU subgoal inference (same idea as D36a/D36b)
# ================================================================

def path_cost_with_penalties(start: Vec2, via: Vec2, goal: Vec2, obstacles: List[Obstacle]) -> float:
    """
    Cost = path length start->via + via->goal
           + big penalties if line-of-sight passes too close to obstacles.

    This is only used to choose which *subgoal* to aim for (doorway vs direct goal).
    """
    segments = [(start, via), (via, goal)]
    cost = 0.0

    for s, t in segments:
        seg_len = v_len(v_sub(t, s))
        cost += seg_len

        for obs in obstacles:
            if segment_intersects_circle(s, t, obs.center, obs.radius, margin=0.05):
                # Simple huge penalty if the segment intersects / grazes obstacle
                cost += 1000.0

    return cost


def infer_best_subgoal(
    robot_pos: Vec2,
    goal_pos: Vec2,
    obstacles: List[Obstacle],
    candidates: List[Vec2],
) -> Vec2:
    """
    Choose the subgoal that minimizes:
        length(robot -> subgoal -> goal) + obstacle penalties.

    This is still done in explicit geometry (CPU),
    and we feed the chosen subgoal to the latent MPC.
    """
    best: Optional[Vec2] = None
    best_cost = float("inf")

    for cand in candidates:
        c = path_cost_with_penalties(robot_pos, cand, goal_pos, obstacles)
        if c < best_cost:
            best_cost = c
            best = cand

    assert best is not None
    return best


# ================================================================
# GPU latent MPC planner (tuned)
# ================================================================

class LatentMPCPlanner:
    """
    Simple batched MPC in a low-dimensional latent space.

    For D36c:
        - latent state = (x, y) robot position (treated as a 2D vector)
        - we sample many candidate action sequences (velocities) on GPU
        - samples are BIASED toward the subgoal direction, with small noise
        - we roll out each candidate in parallel
        - we compute a cost for each trajectory:
            * final distance to current subgoal
            * penalty if any state comes too close to an obstacle
            * small penalty on action magnitude
        - we choose the best candidate and return its first action.

    This version is tuned so that:
        - path is less squiggly
        - we actually reach the doorway and then the goal in a reasonable time
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        num_candidates: int = 256,
        horizon: int = 15,
        action_scale: float = 2.0,    # approximate max speed magnitude
        noise_scale: float = 0.4,     # how much we jitter around the main direction
        collision_margin: float = 0.05,
        collision_penalty: float = 1000.0,
        action_penalty: float = 0.05,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_candidates = num_candidates
        self.horizon = horizon
        self.action_scale = action_scale
        self.noise_scale = noise_scale
        self.collision_margin = collision_margin
        self.collision_penalty = collision_penalty
        self.action_penalty = action_penalty

        print(
            f"[D36c] LatentMPCPlanner device={self.device}, "
            f"candidates={self.num_candidates}, horizon={self.horizon}"
        )

    def _obstacle_tensors(self, obstacles: List[Obstacle]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of Obstacle to batched tensors on the planner device.

        Returns:
            centers: (num_obs, 2)
            radii:   (num_obs,)
        """
        if len(obstacles) == 0:
            centers = torch.zeros(0, 2, device=self.device)
            radii = torch.zeros(0, device=self.device)
            return centers, radii

        centers_list = [list(obs.center) for obs in obstacles]
        radii_list = [obs.radius for obs in obstacles]

        centers = torch.tensor(centers_list, dtype=torch.float32, device=self.device)  # (O, 2)
        radii = torch.tensor(radii_list, dtype=torch.float32, device=self.device)       # (O,)
        return centers, radii

    def plan(
        self,
        robot_pos: Vec2,
        current_subgoal: Vec2,
        obstacles: List[Obstacle],
        dt: float,
    ) -> Vec2:
        """
        Run a single MPC step in latent state space.

        Inputs:
            - robot_pos: current robot position (x, y)
            - current_subgoal: where we want to end up this horizon
            - obstacles: list of Obstacle
            - dt: simulation time step

        Returns:
            - action: (dx, dy) to apply for ONE step
        """
        # ---------------------------------------------
        # 1. Build tensors for initial state and subgoal
        # ---------------------------------------------
        start_state = torch.tensor(robot_pos, dtype=torch.float32, device=self.device)        # (2,)
        subgoal_state = torch.tensor(current_subgoal, dtype=torch.float32, device=self.device)  # (2,)

        # Expand to (N, 2) so each candidate starts from the same initial pos
        s0 = start_state.unsqueeze(0).expand(self.num_candidates, 2)  # (N, 2)

        # Direction from robot → subgoal (unit vector)
        dir_vec = subgoal_state - start_state
        dir_norm = torch.linalg.norm(dir_vec) + 1e-8
        dir_unit = dir_vec / dir_norm  # (2,)

        # Base action (constant velocity toward subgoal)
        base_action = dir_unit * self.action_scale  # (2,)

        # ---------------------------------------------
        # 2. Sample candidate action sequences (biased)
        # ---------------------------------------------
        # Start from a constant-velocity sequence toward the subgoal
        # and add Gaussian noise (small jitter).
        # base_seq: (N, H, 2)
        base_seq = base_action.unsqueeze(0).unsqueeze(0).expand(self.num_candidates, self.horizon, 2)

        noise = torch.randn(self.num_candidates, self.horizon, 2, device=self.device)
        a_seq = base_seq + self.noise_scale * noise

        # ---------------------------------------------
        # 3. Roll out candidate trajectories
        # ---------------------------------------------
        # states: we store states at H+1 steps (including initial)
        # shape: (N, H+1, 2)
        states = torch.zeros(self.num_candidates, self.horizon + 1, 2, device=self.device)
        states[:, 0, :] = s0

        for t in range(self.horizon):
            states[:, t + 1, :] = states[:, t, :] + a_seq[:, t, :] * dt

        # ---------------------------------------------
        # 4. Compute costs
        # ---------------------------------------------
        # 4.1 final distance to subgoal
        final_states = states[:, -1, :]  # (N, 2)
        diff_goal = final_states - subgoal_state.unsqueeze(0)
        dist_goal = torch.linalg.norm(diff_goal, dim=-1)  # (N,)

        # 4.2 obstacle collision penalty
        centers, radii = self._obstacle_tensors(obstacles)
        if centers.shape[0] > 0:
            # states: (N, H+1, 2)
            # centers: (O, 2)
            # pairwise distances (N, H+1, O)
            diff = states.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)  # (N, H+1, O, 2)
            dists = torch.linalg.norm(diff, dim=-1)  # (N, H+1, O)

            # min distance to any obstacle along trajectory
            min_dist, _ = dists.min(dim=-1)         # (N, H+1)
            min_dist, _ = min_dist.min(dim=-1)      # (N,)

            collision_cost = torch.zeros_like(dist_goal)
            # penalty if we get closer than (max radius + margin)
            if radii.numel() > 0:
                threshold = radii.max() + self.collision_margin
                collision_mask = min_dist < threshold
                collision_cost[collision_mask] += self.collision_penalty
        else:
            collision_cost = torch.zeros_like(dist_goal)

        # 4.3 action magnitude penalty (to discourage wild moves)
        action_mag = torch.linalg.norm(a_seq, dim=-1)  # (N, H)
        action_cost = self.action_penalty * action_mag.sum(dim=-1)  # (N,)

        # Total cost
        total_cost = dist_goal + collision_cost + action_cost

        # ---------------------------------------------
        # 5. Choose best candidate
        # ---------------------------------------------
        best_idx = torch.argmin(total_cost)
        best_first_action = a_seq[best_idx, 0, :]  # (2,)

        # Convert back to Python tuple (dx, dy)
        dx, dy = best_first_action.detach().cpu().tolist()
        return (dx, dy)


# ================================================================
# Simulation loop using latent MPC
# ================================================================

def run_sim(
    world: World,
    max_steps: int = 600,
    subgoal_switch_epsilon: float = 0.6,
    perception: Optional[JEPAPerception] = None,
    planner: Optional[LatentMPCPlanner] = None,
) -> Tuple[Vec2, List[Vec2], List[Vec2]]:
    """
    Main simulation loop.

    Differences vs D36a/D36b:
        - We still infer subgoal (doorway vs goal) on CPU.
        - BUT robot motion is now controlled by a latent GPU MPC planner.
        - At each step, MPC chooses an action (dx, dy) for one time-step.

    Returns:
        final_robot_pos, robot_path, subgoal_sequence
    """
    robot = world.robot
    goal_final = world.goal_final
    obstacles = world.obstacles
    candidates = world.waypoint_candidates

    # 1) Optional JEPA perception (scene → z)
    if perception is not None:
        img = perception.render_world_to_image(world)
        z = perception.embed_image(img)
        with torch.no_grad():
            z_norm = torch.linalg.norm(z).item()
        print(f"[D36c] JEPA latent shape = {tuple(z.shape)}, ||z||={z_norm:.3f}")

    # 2) Latent MPC planner
    if planner is None:
        planner = LatentMPCPlanner()

    # 3) Initial subgoal from explicit geometry
    current_subgoal = infer_best_subgoal(robot.pos, goal_final, obstacles, candidates)
    subgoal_sequence = [current_subgoal]
    robot_path = [robot.pos]

    print("[D36c] Starting simulation")
    print(f"  robot_start = {robot.pos}")
    print(f"  goal_final  = {goal_final}")
    print(f"  candidates  = {candidates}")
    print(f"  chosen_subgoal_0 = {current_subgoal}")
    print()

    last_t = 0

    for t in range(max_steps):
        last_t = t

        # If we've reached the current subgoal, switch to the final goal.
        if v_len(v_sub(current_subgoal, robot.pos)) < subgoal_switch_epsilon:
            if current_subgoal != goal_final:
                current_subgoal = goal_final
                subgoal_sequence.append(current_subgoal)
                print(f"[t={t}] Reached intermediate subgoal, switching to final goal {goal_final}")
            else:
                print(f"[t={t}] Reached final goal (within epsilon).")
                break

        # MPC step: compute action from robot.pos → current_subgoal
        dx, dy = planner.plan(
            robot_pos=robot.pos,
            current_subgoal=current_subgoal,
            obstacles=obstacles,
            dt=world.dt,
        )

        # Clip action magnitude to robot.max_speed for realism
        mag = math.hypot(dx, dy)
        if mag > robot.max_speed:
            scale = robot.max_speed / max(mag, 1e-8)
            dx *= scale
            dy *= scale

        # Apply action as "velocity" for one dt step
        robot.pos = (robot.pos[0] + dx * world.dt, robot.pos[1] + dy * world.dt)
        robot_path.append(robot.pos)

        d_to_subgoal = v_len(v_sub(current_subgoal, robot.pos))
        d_to_final = v_len(v_sub(goal_final, robot.pos))

        if t % 10 == 0:
            print(
                f"[t={t:03d}] pos={robot.pos!r} d_sub={d_to_subgoal:.3f} "
                f"d_goal={d_to_final:.3f} current_subgoal={current_subgoal!r}"
            )

        # stop if very close to final goal
        if current_subgoal == goal_final and d_to_final < subgoal_switch_epsilon:
            print(f"[t={t}] Final goal reached (epsilon).")
            break

    print()
    print(f"[D36c] Simulation ended at t={last_t}, final_pos={robot.pos}")
    print(f"[D36c] Distance to final goal = {v_len(v_sub(goal_final, robot.pos)):.3f}")

    return robot.pos, robot_path, subgoal_sequence


# ================================================================
# Visualization (same spirit as D36a/D36b)
# ================================================================

def plot_trajectory(world: World, robot_path: List[Vec2], subgoal_seq: List[Vec2]) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plot.")
        return

    xs = [p[0] for p in robot_path]
    ys = [p[1] for p in robot_path]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", "box")

    # Obstacles
    for obs in world.obstacles:
        circle = plt.Circle(obs.center, obs.radius, fill=False, linestyle="--")
        ax.add_patch(circle)
        ax.text(obs.center[0], obs.center[1], "obs", ha="center", va="center")

    # Path
    ax.plot(xs, ys, marker=".", linewidth=1.0)
    ax.scatter(xs[0], ys[0], marker="o", label="start")
    ax.scatter(xs[-1], ys[-1], marker="x", label="end")

    # Final goal
    gx, gy = world.goal_final
    ax.scatter([gx], [gy], marker="*", s=120, label="goal_final")

    # Subgoals
    for i, sg in enumerate(subgoal_seq):
        ax.scatter([sg[0]], [sg[1]], marker="s", s=60, label=f"subgoal_{i}" if i == 0 else None)

    ax.set_title("D36c — Latent MPC (GPU, biased) + JEPA perception")
    ax.legend()
    ax.grid(True)
    plt.show()


# ================================================================
# Main
# ================================================================

def main():
    """
    Scenario:
        - robot at left
        - final goal at right
        - two obstacles forming a "wall" with a doorway above
        - one candidate doorway waypoint above the obstacles
        - subgoal inference chooses doorway vs direct
        - latent MPC on GPU plans the path step-by-step
    """
    robot_start: Vec2 = (-8.0, 0.0)
    goal_final: Vec2 = (8.0, 0.0)

    obstacles = [
        Obstacle(center=(-1.5, 0.0), radius=2.0),
        Obstacle(center=(1.5, 0.0), radius=2.0),
    ]

    # Doorway waypoint above the obstacles (hand-designed)
    doorway: Vec2 = (0.0, 5.0)

    # Candidates include both doorway and direct-goal:
    waypoint_candidates = [doorway, goal_final]

    world = World(
        robot=Robot(pos=robot_start, max_speed=2.0),
        goal_final=goal_final,
        obstacles=obstacles,
        waypoint_candidates=waypoint_candidates,
        dt=0.05,
    )

    # JEPA perception + latent MPC
    perception = JEPAPerception()
    planner = LatentMPCPlanner(device=perception.device)

    final_pos, robot_path, subgoal_seq = run_sim(
        world,
        perception=perception,
        planner=planner,
    )
    plot_trajectory(world, robot_path, subgoal_seq)


if __name__ == "__main__":
    main()

