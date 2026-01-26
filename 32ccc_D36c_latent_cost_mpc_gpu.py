"""
32ccc_D36c_latent_cost_mpc_gpu.py

D36c — Latent MPC on GPU + JEPA perception, with JEPA latent cost

This version is a direct evolution of:
    32bbb_D36c_subgoal_latent_mpc_gpu.py

Key differences vs 32bbb:
    - JEPA is no longer just a "print latent and ignore" component.
    - We approximate how the JEPA latent z changes with robot position (x, y)
      via a simple finite-difference Jacobian around the current position.
    - We precompute scene latents for:
          * robot at doorway (subgoal)
          * robot at final goal
      and use these as JEPA "target latents".
    - The MPC cost now includes a JEPA latent distance term:
          cost_total = dist_goal_geom
                       + collision_cost
                       + action_smoothness_cost
                       + lambda_je = || z_pred_final - z_target ||_2

    - All JEPA work is still on GPU; MPC remains batched and parallelized.

This is still a toy demo:
    - It uses a first-order linear approximation of z(x,y)
      based on finite differences in JEPA space.
    - It is meant as an educational bridge toward
      "latent-space world model + planning" (D70),
      not as a production-quality controller.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # plotting is optional

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor

Vec2 = Tuple[float, float]


# ================================================================
# 1. Geometry helpers (same as 32bbb)
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
# 2. World objects (same as 32bbb)
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
# 3. JEPA-based perception (extended vs 32bbb)
# ================================================================

class JEPAPerception:
    """
    Minimal wrapper around facebook/ijepa_vith14_1k.

    Responsibilities:
        - render a top-down image of the World (2D toy world → pixels)
        - encode it to a JEPA latent z: shape (1, D)
        - provide helpers for:
            * scene latent at arbitrary robot position
            * finite-difference Jacobian of z wrt (x, y) near current pos

    The JEPA encoder is still frozen; we only use it as a feature extractor.
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

        print(f"[D36c-latent] JEPA device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    # ------------------------------------------------------------
    # 3.1 Base embedding
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 3.2 Render world → image
    # ------------------------------------------------------------
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
        gx, gy = world.world_to_px(world.goal_final) if False else world_to_px(world.goal_final)
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

    # ------------------------------------------------------------
    # 3.3 Helpers: scene latent at an arbitrary robot position
    # ------------------------------------------------------------
    def embed_world_at_pos(self, world: "World", pos: Vec2) -> torch.Tensor:
        """
        Convenience function:
            - temporarily move robot to 'pos'
            - render world
            - encode with JEPA
            - restore original robot position
        Returns:
            z: (1, D) on JEPA device
        """
        orig_pos = world.robot.pos
        world.robot.pos = pos
        img = self.render_world_to_image(world)
        z = self.embed_image(img)
        world.robot.pos = orig_pos
        return z

    # ------------------------------------------------------------
    # 3.4 Finite-difference Jacobian wrt (x, y) at current position
    # ------------------------------------------------------------
    def finite_difference_jacobian(
        self,
        world: "World",
        eps: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Approximate ∂z/∂x and ∂z/∂y at the current robot position.

        We do:
            z0 = z(x,       y)
            zx = z(x + eps, y)
            zy = z(x,       y + eps)

        Then:
            Jx ≈ (zx - z0) / eps
            Jy ≈ (zy - z0) / eps

        Returns:
            {
                "pos0":      torch.tensor([x, y]), on device
                "z0":        (D,) vector
                "Jx":        (D,) vector
                "Jy":        (D,) vector
            }
        """
        x, y = world.robot.pos
        pos0 = torch.tensor([x, y], dtype=torch.float32, device=self.device)

        # Center
        z0 = self.embed_world_at_pos(world, (x, y))   # (1, D)
        # x-perturbed
        zx = self.embed_world_at_pos(world, (x + eps, y))
        # y-perturbed
        zy = self.embed_world_at_pos(world, (x, y + eps))

        # Convert to (D,) vectors
        z0_vec = z0[0]
        zx_vec = zx[0]
        zy_vec = zy[0]

        Jx = (zx_vec - z0_vec) / eps
        Jy = (zy_vec - z0_vec) / eps

        return {
            "pos0": pos0,
            "z0": z0_vec,
            "Jx": Jx,
            "Jy": Jy,
        }


# ================================================================
# 4. CPU subgoal inference (same as 32bbb)
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
# 5. GPU latent MPC planner (extended vs 32bbb)
# ================================================================

class LatentMPCPlanner:
    """
    Simple batched MPC in a low-dimensional latent space.

    For D36c-latent-cost:
        - latent state = (x, y) robot position (treated as a 2D vector)
        - we sample many candidate action sequences (velocities) on GPU
        - samples are BIASED toward the subgoal direction, with small noise
        - we roll out each candidate in parallel
        - we compute a cost for each trajectory:
            * final distance to current subgoal (geometric)
            * penalty if any state comes too close to an obstacle
            * small penalty on action magnitude (smoothness / energy)
            * JEPA latent distance between predicted final z and target z
        - we choose the best candidate and return its first action.

    The JEPA latent cost uses a local linear approximation:
        z(s) ≈ z0 + (x - x0) * Jx + (y - y0) * Jy
    where z0, Jx, Jy are obtained from finite-difference around the current pose.
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
        latent_cost_weight: float = 0.3,   # weight for JEPA latent distance term
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
        self.latent_cost_weight = latent_cost_weight

        print(
            f"[D36c-latent] LatentMPCPlanner device={self.device}, "
            f"candidates={self.num_candidates}, horizon={self.horizon}, "
            f"latent_cost_weight={self.latent_cost_weight}"
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
        latent_params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Vec2:
        """
        Run a single MPC step in latent state space.

        Inputs:
            - robot_pos: current robot position (x, y)
            - current_subgoal: where we want to end up this horizon
            - obstacles: list of Obstacle
            - dt: simulation time step
            - latent_params: optional dict with JEPA data:
                {
                    "pos0":    (2,)  tensor with expansion point (x0, y0)
                    "z0":      (D,)  JEPA latent at (x0, y0)
                    "Jx":      (D,)  ∂z/∂x at (x0, y0)
                    "Jy":      (D,)  ∂z/∂y at (x0, y0)
                    "z_target":(D,)  JEPA latent for target scene (doorway or goal)
                }

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
        # base_seq: (N, H, 2)
        base_seq = base_action.unsqueeze(0).unsqueeze(0).expand(self.num_candidates, self.horizon, 2)

        noise = torch.randn(self.num_candidates, self.horizon, 2, device=self.device)
        a_seq = base_seq + self.noise_scale * noise  # (N, H, 2)

        # ---------------------------------------------
        # 3. Roll out candidate trajectories
        # ---------------------------------------------
        states = torch.zeros(self.num_candidates, self.horizon + 1, 2, device=self.device)
        states[:, 0, :] = s0

        for t in range(self.horizon):
            states[:, t + 1, :] = states[:, t, :] + a_seq[:, t, :] * dt

        # ---------------------------------------------
        # 4. Geometric + collision + smoothness costs
        # ---------------------------------------------
        # 4.1 final distance to subgoal (geometric)
        final_states = states[:, -1, :]  # (N, 2)
        diff_goal = final_states - subgoal_state.unsqueeze(0)
        dist_goal = torch.linalg.norm(diff_goal, dim=-1)  # (N,)

        # 4.2 obstacle collision penalty
        centers, radii = self._obstacle_tensors(obstacles)
        if centers.shape[0] > 0:
            # states: (N, H+1, 2)
            # centers: (O, 2)
            diff = states.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)  # (N, H+1, O, 2)
            dists = torch.linalg.norm(diff, dim=-1)  # (N, H+1, O)

            # min distance to any obstacle along trajectory
            min_dist, _ = dists.min(dim=-1)         # (N, H+1)
            min_dist, _ = min_dist.min(dim=-1)      # (N,)

            collision_cost = torch.zeros_like(dist_goal)
            if radii.numel() > 0:
                threshold = radii.max() + self.collision_margin
                collision_mask = min_dist < threshold
                collision_cost[collision_mask] += self.collision_penalty
        else:
            collision_cost = torch.zeros_like(dist_goal)

        # 4.3 action magnitude penalty (smoothness / energy)
        action_mag = torch.linalg.norm(a_seq, dim=-1)  # (N, H)
        action_cost = self.action_penalty * action_mag.sum(dim=-1)  # (N,)

        # ---------------------------------------------
        # 5. JEPA latent cost (NEW vs 32bbb)
        # ---------------------------------------------
        if latent_params is not None and self.latent_cost_weight > 0.0:
            # Unpack latent_params; all are (D,) or (2,) on planner.device
            pos0 = latent_params["pos0"]       # (2,)
            z0 = latent_params["z0"]           # (D,)
            Jx = latent_params["Jx"]           # (D,)
            Jy = latent_params["Jy"]           # (D,)
            z_target = latent_params["z_target"]  # (D,)

            # Linear approximation: z(s) ≈ z0 + (x - x0) * Jx + (y - y0) * Jy
            delta = final_states - pos0.unsqueeze(0)  # (N, 2)
            dx = delta[:, 0:1]  # (N, 1)
            dy = delta[:, 1:2]  # (N, 1)

            # v0 = z0 - z_target: (D,)
            v0 = z0 - z_target  # (D,)

            # Broadcast to (N, D)
            v = (
                v0.unsqueeze(0)
                + dx * Jx.unsqueeze(0)
                + dy * Jy.unsqueeze(0)
            )  # (N, D)

            # L2 norm in JEPA latent space
            latent_dist = torch.linalg.norm(v, dim=-1)  # (N,)

            latent_cost = self.latent_cost_weight * latent_dist
        else:
            latent_cost = torch.zeros_like(dist_goal)

        # ---------------------------------------------
        # 6. Total cost and best candidate
        # ---------------------------------------------
        total_cost = dist_goal + collision_cost + action_cost + latent_cost

        best_idx = torch.argmin(total_cost)
        best_first_action = a_seq[best_idx, 0, :]  # (2,)

        dx, dy = best_first_action.detach().cpu().tolist()
        return (dx, dy)


# ================================================================
# 6. Simulation loop using latent MPC + JEPA latent cost
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

    Differences vs 32bbb:
        - We still infer subgoal (doorway vs goal) on CPU.
        - Robot motion is controlled by latent GPU MPC, as before.
        - At each step, we:
            * compute a finite-difference Jacobian of JEPA latent wrt (x, y)
              at the CURRENT robot pose.
            * choose a JEPA target latent:
                  - doorway latent if current_subgoal == doorway
                  - goal latent if current_subgoal == final goal
            * pass these as 'latent_params' into the planner.
        - The planner includes a JEPA latent distance term in its cost.
    """
    robot = world.robot
    goal_final = world.goal_final
    obstacles = world.obstacles
    candidates = world.waypoint_candidates

    if perception is None:
        raise ValueError("JEPAPerception is required for D36c-latent-cost demo.")

    # 1) JEPA scene latents: doorway & final goal (targets)
    #    We precompute these once for efficiency.
    doorway = candidates[0]
    z_doorway = perception.embed_world_at_pos(world, doorway)[0]  # (D,)
    z_goal = perception.embed_world_at_pos(world, goal_final)[0]  # (D,)

    with torch.no_grad():
        print(f"[D36c-latent] z_doorway dim={z_doorway.shape}, ||z_doorway||={z_doorway.norm().item():.3f}")
        print(f"[D36c-latent] z_goal    dim={z_goal.shape}, ||z_goal||={z_goal.norm().item():.3f}")

    # 2) Latent MPC planner
    if planner is None:
        planner = LatentMPCPlanner(device=perception.device)

    # 3) Initial subgoal from explicit geometry
    current_subgoal = infer_best_subgoal(robot.pos, goal_final, obstacles, candidates)
    subgoal_sequence = [current_subgoal]
    robot_path = [robot.pos]

    print("[D36c-latent] Starting simulation")
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

        # 3.1 Compute JEPA Jacobian around CURRENT state
        jac = perception.finite_difference_jacobian(world, eps=0.2)
        # decide JEPA target latent based on which subgoal we are pursuing
        if current_subgoal == doorway:
            z_target = z_doorway
        else:
            z_target = z_goal

        # latent_params dict passed to planner
        latent_params = {
            "pos0": jac["pos0"].to(planner.device),
            "z0": jac["z0"].to(planner.device),
            "Jx": jac["Jx"].to(planner.device),
            "Jy": jac["Jy"].to(planner.device),
            "z_target": z_target.to(planner.device),
        }

        # MPC step: compute action from robot.pos → current_subgoal
        dx, dy = planner.plan(
            robot_pos=robot.pos,
            current_subgoal=current_subgoal,
            obstacles=obstacles,
            dt=world.dt,
            latent_params=latent_params,   # NEW vs 32bbb
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
    print(f"[D36c-latent] Simulation ended at t={last_t}, final_pos={robot.pos}")
    print(f"[D36c-latent] Distance to final goal = {v_len(v_sub(goal_final, robot.pos)):.3f}")

    return robot.pos, robot_path, subgoal_sequence


# ================================================================
# 7. Visualization (same as 32bbb)
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

    ax.set_title("D36c-latent — MPC (GPU) + JEPA latent cost")
    ax.legend()
    ax.grid(True)
    plt.show()


# ================================================================
# 8. Main
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
        - JEPA latent cost gently pulls trajectories toward scenes that
          "look like" being at the doorway / goal.
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



