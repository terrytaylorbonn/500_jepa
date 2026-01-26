"""
31_D36b_subgoal_jepa_perception.py

D36b — Hybrid:
    - CPU-only world, subgoal inference, and planner
    - GPU-based JEPA perception:
        * render a simple top-down image of the scene
        * encode it with facebook/ijepa_vith14_1k
        * log latent shape and norm

Behavior:
    - same path and subgoal logic as D36a
    - additional JEPA latent computed once at the start
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


# ------------------------------
# 1 Geometry helpers
# ------------------------------

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


# ------------------------------
# 2 World objects
# ------------------------------

@dataclass
class Obstacle:
    center: Vec2
    radius: float


@dataclass
class Robot:
    pos: Vec2
    max_speed: float = 0.2


@dataclass
class World:
    robot: Robot
    goal_final: Vec2
    obstacles: List[Obstacle]
    waypoint_candidates: List[Vec2]  # includes doorway + possibly goal itself
    dt: float = 0.1


# ------------------------------
# 2b JEPA-based perception (GPU)
# ------------------------------

class JEPAPerception:
    """
    Minimal wrapper around facebook/ijepa_vith14_1k.

    Responsibilities:
        - render a top-down image of the World
        - encode it to a latent z with JEPA
        - no effect on planning; we just log the latent
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

        print(f"[D36b] JEPA device: {self.device}")

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

        # World bounds: chosen to include start, goal, doorway
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
            # crude scaling from world radius to pixels
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


# ------------------------------
# 3 Subgoal inference (CPU-only) SAME
# ------------------------------

def path_cost_with_penalties(start: Vec2, via: Vec2, goal: Vec2, obstacles: List[Obstacle]) -> float:
    """
    Cost = path length start->via + via->goal
           + big penalties if line-of-sight passes too close to obstacles.
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
    This is the "subgoal inference from explicit scene state".
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


# ------------------------------
# 4 Control loop (greedy steering) SAME
# ------------------------------

def step_robot_towards(robot: Robot, target: Vec2, dt: float) -> None:
    """Greedy controller: move directly towards target, clamped by max_speed."""
    direction = v_sub(target, robot.pos)
    dist = v_len(direction)
    if dist == 0.0:
        return

    vmax_step = robot.max_speed * dt
    step_dist = min(dist, vmax_step)
    step_vec = v_scale(v_norm(direction), step_dist)
    robot.pos = v_add(robot.pos, step_vec)


# ------------------------------
# 5 Simulation
# ------------------------------

def run_sim(
    world: World,
    max_steps: int = 500,
    subgoal_switch_epsilon: float = 0.3,
    perception: Optional[JEPAPerception] = None,
) -> Tuple[Vec2, List[Vec2], List[Vec2]]:
    """
    Run a simple sim until robot reaches final goal or max_steps.

    Returns:
        final_robot_pos, robot_path, subgoal_path
    """
    robot = world.robot
    goal_final = world.goal_final
    obstacles = world.obstacles
    candidates = world.waypoint_candidates

    # One-time JEPA perception pass (independent of planner)
    if perception is not None:
        img = perception.render_world_to_image(world)
        z = perception.embed_image(img)
        with torch.no_grad():
            z_norm = torch.linalg.norm(z).item()
        print(f"[D36b] JEPA latent shape = {tuple(z.shape)}, ||z||={z_norm:.3f}")

    # First subgoal: inferred from scene
    current_subgoal = infer_best_subgoal(robot.pos, goal_final, obstacles, candidates)
    subgoal_sequence = [current_subgoal]
    robot_path = [robot.pos]

    print("[D36b] Starting simulation")
    print(f"  robot_start = {robot.pos}")
    print(f"  goal_final  = {goal_final}")
    print(f"  candidates  = {candidates}")
    print(f"  chosen_subgoal_0 = {current_subgoal}")
    print()

    last_t = 0

    for t in range(max_steps):
        last_t = t

        # Switch to final goal when near current subgoal and current is not final goal.
        if v_len(v_sub(current_subgoal, robot.pos)) < subgoal_switch_epsilon:
            if current_subgoal != goal_final:
                current_subgoal = goal_final
                subgoal_sequence.append(current_subgoal)
                print(f"[t={t}] Reached intermediate subgoal, switching to final goal {goal_final}")
            else:
                print(f"[t={t}] Reached final goal (within epsilon).")
                break

        # Take one control step
        step_robot_towards(robot, current_subgoal, world.dt)
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
    print(f"[D36b] Simulation ended at t={last_t}, final_pos={robot.pos}")
    print(f"[D36b] Distance to final goal = {v_len(v_sub(goal_final, robot.pos)):.3f}")

    return robot.pos, robot_path, subgoal_sequence


# ------------------------------
# 6 Visualization (optional)
# ------------------------------

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

    ax.set_title("D36b — CPU subgoal inference + JEPA perception")
    ax.legend()
    ax.grid(True)
    plt.show()


# ------------------------------
# 7 Main
# ------------------------------

def main():
    # Toy scenario:
    #   - robot at left
    #   - final goal at right
    #   - two obstacles forming a "wall" with a doorway above
    #   - one candidate doorway waypoint above the obstacles
    robot_start: Vec2 = (-8.0, 0.0)
    goal_final: Vec2 = (8.0, 0.0)

    obstacles = [
        Obstacle(center=(-1.5, 0.0), radius=2.0),
        Obstacle(center=(1.5, 0.0), radius=2.0),
    ]

    # Doorway waypoint above the obstacles (hand-designed for D36b)
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

    perception = JEPAPerception()

    final_pos, robot_path, subgoal_seq = run_sim(world, perception=perception)
    plot_trajectory(world, robot_path, subgoal_seq)


if __name__ == "__main__":
    main()
