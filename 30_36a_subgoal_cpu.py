"""
30_36a_subgoal_cpu.py
D36a — CPU-only subgoal inference from explicit scene state.

- 2D world
- robot = point mass
- circular obstacles
- final goal
- one candidate "doorway" subgoal
- rule-based subgoal selection:
    pick the waypoint (doorway or direct goal) with lowest
    line-of-sight collision-penalized path cost.
- simple greedy controller (no MPC) that steers to the current subgoal.

Dependencies:
    - Python 3.10+
    - matplotlib (for visualization only; the sim runs without showing)
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting is optional


Vec2 = Tuple[float, float]


# ------------------------------
# Geometry helpers
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
# World objects
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
# Subgoal inference (CPU-only)
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


def infer_best_subgoal(robot_pos: Vec2,
                       goal_pos: Vec2,
                       obstacles: List[Obstacle],
                       candidates: List[Vec2]) -> Vec2:
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
# Control loop (greedy steering)
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
# Simulation
# ------------------------------

def run_sim(world: World,
            max_steps: int = 500,
            subgoal_switch_epsilon: float = 0.3) -> Tuple[Vec2, List[Vec2], List[Vec2]]:
    """
    Run a simple sim until robot reaches final goal or max_steps.

    Returns:
        final_robot_pos, robot_path, subgoal_path
    """
    robot = world.robot
    goal_final = world.goal_final
    obstacles = world.obstacles
    candidates = world.waypoint_candidates

    # First subgoal: inferred from scene
    current_subgoal = infer_best_subgoal(robot.pos, goal_final, obstacles, candidates)
    subgoal_sequence = [current_subgoal]
    robot_path = [robot.pos]

    print("[D36a] Starting simulation")
    print(f"  robot_start = {robot.pos}")
    print(f"  goal_final  = {goal_final}")
    print(f"  candidates  = {candidates}")
    print(f"  chosen_subgoal_0 = {current_subgoal}")
    print()

    for t in range(max_steps):
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
                f"[t={t:03d}] pos={robot.pos!r} d_sub={d_to_subgoal:.3f} d_goal={d_to_final:.3f} "
                f"current_subgoal={current_subgoal!r}"
            )

        # stop if very close to final goal
        if current_subgoal == goal_final and d_to_final < subgoal_switch_epsilon:
            print(f"[t={t}] Final goal reached (epsilon).")
            break

    print()
    print(f"[D36a] Simulation ended at t={t}, final_pos={robot.pos}")
    print(f"[D36a] Distance to final goal = {v_len(v_sub(goal_final, robot.pos)):.3f}")

    return robot.pos, robot_path, subgoal_sequence


# ------------------------------
# Visualization (optional)
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

    ax.set_title("D36a — CPU subgoal inference demo")
    ax.legend()
    ax.grid(True)
    plt.show()


# ------------------------------
# Main
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

    # Doorway waypoint above the obstacles (hand-designed for D36a)
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

    final_pos, robot_path, subgoal_seq = run_sim(world)
    plot_trajectory(world, robot_path, subgoal_seq)


if __name__ == "__main__":
    main()
