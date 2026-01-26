"""
33_D40_unicycle_mpc.py

D40 — Control / Embodiment:
    - Same 2D world as D36 (start, doorway, goal, obstacles).
    - But robot is now a UNICYCLE:
        state  s = (x, y, theta)
        action a = (v, omega)   (forward speed, turn rate)

    Unicycle dynamics:
        x_{t+1}     = x_t + v * cos(theta) * dt
        y_{t+1}     = y_t + v * sin(theta) * dt
        theta_{t+1} = theta_t + omega * dt

    Planner:
        - Batched MPC on GPU (if available) or CPU.
        - Samples many (v, omega) sequences.
        - Biases actions toward the subgoal direction.
        - Rolls out trajectories with unicycle dynamics.
        - Costs:
            * distance to subgoal
            * collision penalty
            * action/energy penalty
        - Returns the first control of the best trajectory.

    Subgoal:
        - Same doorway-vs-direct heuristic as D36a.
        - Geometry picks doorway vs direct path.
        - Unicycle MPC does the actual steering.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import torch

Vec2 = Tuple[float, float]


# ================================================================
# 1. Geometry helpers
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
    """
    Distance from point p to segment a-b.
    Used for approximate collision penalty along straight segments.
    """
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


def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap_angle_tensor(theta: torch.Tensor) -> torch.Tensor:
    """
    Wrap angles in a tensor to [-pi, pi].
    """
    return torch.atan2(torch.sin(theta), torch.cos(theta))


# ================================================================
# 2. World objects
# ================================================================

@dataclass
class Obstacle:
    center: Vec2
    radius: float


@dataclass
class Robot:
    """
    Unicycle robot:
        state: (x, y, theta)
        controls: (v, omega)
    """
    pos: Vec2
    theta: float
    max_speed: float = 2.0       # max forward speed v
    max_turn_rate: float = 2.0   # max turn rate omega


@dataclass
class World:
    """
    Simple 2D world:
        - unicycle robot
        - final goal
        - obstacles
        - waypoint candidates (doorway, final goal)
        - dt time step
    """
    robot: Robot
    goal_final: Vec2
    obstacles: List[Obstacle]
    waypoint_candidates: List[Vec2]
    dt: float = 0.05


# ================================================================
# 3. Subgoal inference (same spirit as D36a)
# ================================================================

def path_cost_with_penalties(start: Vec2, via: Vec2, goal: Vec2, obstacles: List[Obstacle]) -> float:
    """
    Cost = path length start->via + via->goal
           + big penalties if line-of-sight passes too close to obstacles.

    THIS IS PURE GEOMETRY (not unicycle dynamics).
    It is only used to pick "doorway vs direct goal" as subgoal.
    """
    segments = [(start, via), (via, goal)]
    cost = 0.0

    for s, t in segments:
        seg_len = v_len(v_sub(t, s))
        cost += seg_len

        for obs in obstacles:
            if segment_intersects_circle(s, t, obs.center, obs.radius, margin=0.05):
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

    Returns either:
        - doorway position
        - final goal position
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
# 4. Unicycle MPC planner (batched, GPU if available)
# ================================================================

class UnicycleMPCPlanner:
    """
    Simple batched MPC for a unicycle robot:

        state  s = (x, y, theta)
        action a = (v, omega)

    We:
        - sample many candidate (v, omega) sequences, biased toward the subgoal
        - roll out each candidate in parallel
        - compute cost for each trajectory
        - pick the best candidate
        - apply only the FIRST action (MPC style)

    Cost terms:
        - final distance to subgoal
        - collision penalty if too close to obstacles
        - action penalty on |v| and |omega|
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        num_candidates: int = 256,
        horizon: int = 15,
        v_scale: float = 1.5,       # typical forward speed scale
        omega_scale: float = 1.5,   # typical turn rate scale
        noise_v: float = 0.4,       # noise level on v
        noise_omega: float = 0.4,   # noise level on omega
        collision_margin: float = 0.05,
        collision_penalty: float = 1000.0,
        action_penalty_v: float = 0.05,
        action_penalty_omega: float = 0.01,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.num_candidates = num_candidates
        self.horizon = horizon
        self.v_scale = v_scale
        self.omega_scale = omega_scale
        self.noise_v = noise_v
        self.noise_omega = noise_omega
        self.collision_margin = collision_margin
        self.collision_penalty = collision_penalty
        self.action_penalty_v = action_penalty_v
        self.action_penalty_omega = action_penalty_omega

        print(
            f"[D40] UnicycleMPCPlanner device={self.device}, "
            f"candidates={self.num_candidates}, horizon={self.horizon}"
        )

    # ------------------------------------------------------------
    # Helper: obstacles as tensors
    # ------------------------------------------------------------
    def _obstacle_tensors(self, obstacles: List[Obstacle]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(obstacles) == 0:
            centers = torch.zeros(0, 2, device=self.device)
            radii = torch.zeros(0, device=self.device)
            return centers, radii

        centers_list = [list(obs.center) for obs in obstacles]
        radii_list = [obs.radius for obs in obstacles]

        centers = torch.tensor(centers_list, dtype=torch.float32, device=self.device)  # (O, 2)
        radii = torch.tensor(radii_list, dtype=torch.float32, device=self.device)       # (O,)
        return centers, radii

    # ------------------------------------------------------------
    # Main MPC step
    # ------------------------------------------------------------
    def plan(
        self,
        robot_pos: Vec2,
        robot_theta: float,
        current_subgoal: Vec2,
        obstacles: List[Obstacle],
        dt: float,
        max_speed: float,
        max_turn_rate: float,
    ) -> Tuple[float, float]:
        """
        One MPC step:

            - robot_pos: (x, y)
            - robot_theta: heading (rad)
            - current_subgoal: (x_sg, y_sg)
            - obstacles: list of Obstacle
            - dt: integration time step (world.dt)
            - max_speed: clip |v|
            - max_turn_rate: clip |omega|

        Returns:
            (v, omega) control to apply for this step.
        """
        # --------------------------------------------------------
        # 1) Initial state tensor
        # --------------------------------------------------------
        x0, y0 = robot_pos
        theta0 = robot_theta

        s0 = torch.tensor([x0, y0, theta0], dtype=torch.float32, device=self.device)  # (3,)
        s0_batch = s0.unsqueeze(0).expand(self.num_candidates, 3)  # (N, 3)

        subgoal = torch.tensor(current_subgoal, dtype=torch.float32, device=self.device)  # (2,)

        # --------------------------------------------------------
        # 2) Base control pointing to subgoal
        # --------------------------------------------------------
        dx = subgoal[0] - x0
        dy = subgoal[1] - y0
        goal_angle = math.atan2(dy.item(), dx.item() if isinstance(dx, torch.Tensor) else dx)
        heading_error = wrap_angle(goal_angle - theta0)

        base_v = self.v_scale * max(0.0, math.cos(heading_error))  # don't go backwards much
        base_omega = self.omega_scale * heading_error

        # Clip base controls
        base_v = max(min(base_v, max_speed), -max_speed)
        base_omega = max(min(base_omega, max_turn_rate), -max_turn_rate)

        base_control = torch.tensor([base_v, base_omega], dtype=torch.float32, device=self.device)  # (2,)

        # --------------------------------------------------------
        # 3) Sample candidate control sequences (biased)
        # --------------------------------------------------------
        # base_seq: (N, H, 2)
        base_seq = base_control.unsqueeze(0).unsqueeze(0).expand(self.num_candidates, self.horizon, 2)

        noise = torch.randn(self.num_candidates, self.horizon, 2, device=self.device)
        noise[:, :, 0] *= self.noise_v
        noise[:, :, 1] *= self.noise_omega

        a_seq = base_seq + noise  # (N, H, 2) where a = (v, omega)

        # Clip to robot limits
        # v in [-max_speed, max_speed], omega in [-max_turn_rate, max_turn_rate]
        a_seq[:, :, 0] = torch.clamp(a_seq[:, :, 0], -max_speed, max_speed)
        a_seq[:, :, 1] = torch.clamp(a_seq[:, :, 1], -max_turn_rate, max_turn_rate)

        # --------------------------------------------------------
        # 4) Roll out unicycle dynamics in batch
        # --------------------------------------------------------
        states = torch.zeros(self.num_candidates, self.horizon + 1, 3, device=self.device)
        states[:, 0, :] = s0_batch

        for t in range(self.horizon):
            v = a_seq[:, t, 0]      # (N,)
            omega = a_seq[:, t, 1]  # (N,)
            x = states[:, t, 0]
            y = states[:, t, 1]
            theta = states[:, t, 2]

            x_next = x + v * torch.cos(theta) * dt
            y_next = y + v * torch.sin(theta) * dt
            theta_next = theta + omega * dt
            theta_next = wrap_angle_tensor(theta_next)

            states[:, t + 1, 0] = x_next
            states[:, t + 1, 1] = y_next
            states[:, t + 1, 2] = theta_next

        # --------------------------------------------------------
        # 5) Cost: distance to subgoal
        # --------------------------------------------------------
        final_states = states[:, -1, :]         # (N, 3)
        final_xy = final_states[:, 0:2]         # (N, 2)
        diff_goal = final_xy - subgoal.unsqueeze(0)  # (N, 2)
        dist_goal = torch.linalg.norm(diff_goal, dim=-1)  # (N,)

        # --------------------------------------------------------
        # 6) Collision penalty
        # --------------------------------------------------------
        centers, radii = self._obstacle_tensors(obstacles)
        if centers.shape[0] > 0:
            # states_xy: (N, H+1, 2)
            states_xy = states[:, :, 0:2]

            # diff: (N, H+1, O, 2)
            diff = states_xy.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)
            dists = torch.linalg.norm(diff, dim=-1)  # (N, H+1, O)

            min_dist, _ = dists.min(dim=-1)   # (N, H+1)
            min_dist, _ = min_dist.min(dim=-1)  # (N,)

            collision_cost = torch.zeros_like(dist_goal)
            if radii.numel() > 0:
                threshold = radii.max() + self.collision_margin
                collision_mask = min_dist < threshold
                collision_cost[collision_mask] += self.collision_penalty
        else:
            collision_cost = torch.zeros_like(dist_goal)

        # --------------------------------------------------------
        # 7) Action penalty (energy / smoothness)
        # --------------------------------------------------------
        v_seq = a_seq[:, :, 0]      # (N, H)
        omega_seq = a_seq[:, :, 1]  # (N, H)

        action_cost = (
            self.action_penalty_v * torch.sum(torch.abs(v_seq), dim=-1)
            + self.action_penalty_omega * torch.sum(torch.abs(omega_seq), dim=-1)
        )

        # --------------------------------------------------------
        # 8) Total cost and best candidate
        # --------------------------------------------------------
        total_cost = dist_goal + collision_cost + action_cost

        best_idx = torch.argmin(total_cost)
        best_first_action = a_seq[best_idx, 0, :]  # (2,)

        v_best = float(best_first_action[0].detach().cpu().item())
        omega_best = float(best_first_action[1].detach().cpu().item())

        return v_best, omega_best


# ================================================================
# 5. Simulation loop
# ================================================================

def run_sim(
    world: World,
    max_steps: int = 500,
    subgoal_switch_epsilon: float = 0.6,
    planner: Optional[UnicycleMPCPlanner] = None,
):
    robot = world.robot
    goal_final = world.goal_final
    obstacles = world.obstacles
    candidates = world.waypoint_candidates

    if planner is None:
        planner = UnicycleMPCPlanner()

    # Choose initial subgoal via geometry (doorway vs direct)
    current_subgoal = infer_best_subgoal(robot.pos, goal_final, obstacles, candidates)
    subgoal_sequence = [current_subgoal]
    robot_path: List[Vec2] = [robot.pos]

    print("[D40] Starting simulation")
    print(f"  robot_start = {robot.pos}, theta={robot.theta:.3f} rad")
    print(f"  goal_final  = {goal_final}")
    print(f"  candidates  = {candidates}")
    print(f"  chosen_subgoal_0 = {current_subgoal}")
    print()

    last_t = 0

    for t in range(max_steps):
        last_t = t

        # Check subgoal reached
        d_to_subgoal = v_len(v_sub(current_subgoal, robot.pos))
        d_to_final = v_len(v_sub(goal_final, robot.pos))

        if d_to_subgoal < subgoal_switch_epsilon:
            if current_subgoal != goal_final:
                current_subgoal = goal_final
                subgoal_sequence.append(current_subgoal)
                print(f"[t={t}] Reached intermediate subgoal, switching to final goal {goal_final}")
            else:
                print(f"[t={t}] Reached final goal (within epsilon).")
                break

        # MPC step: choose (v, omega)
        v_cmd, omega_cmd = planner.plan(
            robot_pos=robot.pos,
            robot_theta=robot.theta,
            current_subgoal=current_subgoal,
            obstacles=obstacles,
            dt=world.dt,
            max_speed=robot.max_speed,
            max_turn_rate=robot.max_turn_rate,
        )

        # Clip again for safety
        v_mag = abs(v_cmd)
        if v_mag > robot.max_speed:
            v_cmd = robot.max_speed * (v_cmd / max(v_mag, 1e-8))

        if abs(omega_cmd) > robot.max_turn_rate:
            omega_cmd = robot.max_turn_rate * (omega_cmd / max(abs(omega_cmd), 1e-8))

        # Unicycle dynamics update
        x, y = robot.pos
        theta = robot.theta

        x_next = x + v_cmd * math.cos(theta) * world.dt
        y_next = y + v_cmd * math.sin(theta) * world.dt
        theta_next = theta + omega_cmd * world.dt
        theta_next = wrap_angle(theta_next)

        robot.pos = (x_next, y_next)
        robot.theta = theta_next

        robot_path.append(robot.pos)

        if t % 10 == 0:
            print(
                f"[t={t:03d}] pos=({robot.pos[0]:+.3f}, {robot.pos[1]:+.3f}) "
                f"theta={robot.theta:+.3f} "
                f"d_sub={d_to_subgoal:.3f} d_goal={d_to_final:.3f} "
                f"subgoal={current_subgoal}"
            )

        # Stop if very close to final goal
        if current_subgoal == goal_final and d_to_final < subgoal_switch_epsilon:
            print(f"[t={t}] Final goal reached (epsilon).")
            break

    print()
    print(f"[D40] Simulation ended at t={last_t}, final_pos={robot.pos}, theta={robot.theta:+.3f}")
    print(f"[D40] Distance to final goal = {v_len(v_sub(goal_final, robot.pos)):.3f}")

    return robot.pos, robot_path, subgoal_sequence


# ================================================================
# 6. Visualization
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

    # Subgoals (doorway + maybe goal)
    for i, sg in enumerate(subgoal_seq):
        ax.scatter([sg[0]], [sg[1]], marker="s", s=60, label=f"subgoal_{i}" if i == 0 else None)

    ax.set_title("D40 — Unicycle MPC (GPU/CPU) with doorway subgoal")
    ax.legend()
    ax.grid(True)
    plt.show()


# ================================================================
# 7. Main
# ================================================================

def main():
    """
    Scenario (same geometry as D36):
        - robot starts at left (x=-8, y=0), facing toward +x (theta=0)
        - final goal at right (x=+8, y=0)
        - two obstacles in the middle forming a "wall"
        - doorway waypoint above the obstacles (0, 5)
        - subgoal inference picks doorway vs direct
        - unicycle MPC steers the robot through the doorway
    """
    robot_start: Vec2 = (-8.0, 0.0)
    goal_final: Vec2 = (8.0, 0.0)

    obstacles = [
        Obstacle(center=(-1.5, 0.0), radius=2.0),
        Obstacle(center=(1.5, 0.0), radius=2.0),
    ]

    doorway: Vec2 = (0.0, 5.0)
    waypoint_candidates = [doorway, goal_final]

    world = World(
        robot=Robot(pos=robot_start, theta=0.0, max_speed=2.0, max_turn_rate=2.0),
        goal_final=goal_final,
        obstacles=obstacles,
        waypoint_candidates=waypoint_candidates,
        dt=0.05,
    )

    planner = UnicycleMPCPlanner()
    final_pos, robot_path, subgoal_seq = run_sim(world, planner=planner)
    plot_trajectory(world, robot_path, subgoal_seq)


if __name__ == "__main__":
    main()
