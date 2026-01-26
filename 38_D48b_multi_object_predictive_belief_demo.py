#!/usr/bin/env python3
"""
38_D48b_multi_object_predictive_belief_demo.py
D48b: Multi-object predictive belief demo
-----------------------------------------
Toy 2D world with:
- 1 robot at the origin with limited FOV (field of view)
- 3 moving objects (small constant velocities)
- 1 occluder (rectangle)
- Per-object belief with:
    - mean position
    - uncertainty sigma
    - estimated velocity
- Simple predict → correct cycle:
    - Beliefs are predicted forward in time using velocity
    - If object is visible, belief is corrected to true position
      and velocity is updated
    - If not visible, belief drifts forward and sigma grows

Goal:
Show that the robot maintains a predictive belief over multiple moving objects,
even when they are temporarily occluded or outside FOV.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle

# -----------------------------
# Config
# -----------------------------

WORLD_X = (-1.0, 7.0)
WORLD_Y = (-3.0, 3.0)

NUM_STEPS = 180
DT = 0.1

FOV_RANGE = 5.5
FOV_HALF_ANGLE_DEG = 45.0  # half-angle of cone
FOV_HALF_ANGLE_RAD = math.radians(FOV_HALF_ANGLE_DEG)

# Robot: fixed position, rotating heading
ROBOT_POS = np.array([0.0, 0.0], dtype=float)
ROBOT_ANG_VEL = math.radians(3.5)  # rad/step, slow scanning

# Occluder: a small rectangle roughly in front of the robot
OCCLUDER_X = 2.5
OCCLUDER_Y = -0.2
OCCLUDER_W = 0.6
OCCLUDER_H = 0.8

# Belief dynamics
SIGMA_MIN = 0.05
SIGMA_MAX = 1.2
SIGMA_DECAY = 0.5   # when seen
SIGMA_GROW  = 1.1   # when not seen

VEL_MOMENTUM = 0.7  # smoothing for velocity updates


# -----------------------------
# Objects and Beliefs
# -----------------------------
# Three objects at different heights, each with a small velocity.
objects = [
    {
        "id": 0,
        "pos": np.array([2.0,  0.5], dtype=float),
        "vel": np.array([0.02, 0.00], dtype=float),
        "color": "tab:red",
    },
    {
        "id": 1,
        "pos": np.array([3.0,  1.2], dtype=float),
        "vel": np.array([0.01, -0.01], dtype=float),
        "color": "tab:green",
    },
    {
        "id": 2,
        "pos": np.array([3.5, -1.0], dtype=float),
        "vel": np.array([0.015, 0.005], dtype=float),
        "color": "tab:blue",
    },
]


class ObjectBelief:
    def __init__(self, pos: np.ndarray, sigma: float = 0.2):
        self.mean = pos.astype(float).copy()
        self.sigma = float(sigma)
        self.visible = True
        self.vel = np.zeros(2, dtype=float)

    def predict(self, dt: float):
        """Predict belief forward using current velocity estimate."""
        self.mean = self.mean + self.vel * dt

    def on_observed(self, true_pos: np.ndarray, dt: float):
        """
        Correct step when object is visible:
        - update velocity estimate based on position delta
        - pull mean to true position
        - shrink sigma
        """
        true_pos = true_pos.astype(float)
        delta = true_pos - self.mean
        if dt > 0.0:
            est_vel = delta / dt
            # Exponential moving average for velocity
            self.vel = VEL_MOMENTUM * self.vel + (1.0 - VEL_MOMENTUM) * est_vel

        self.mean = true_pos
        self.sigma = max(self.sigma * SIGMA_DECAY, SIGMA_MIN)
        self.visible = True

    def on_not_observed(self):
        """
        No correction, only growing uncertainty.
        Mean was already predicted forward; we just inflate sigma a bit.
        """
        self.sigma = min(self.sigma * SIGMA_GROW, SIGMA_MAX)
        self.visible = False


beliefs = {
    obj["id"]: ObjectBelief(obj["pos"], sigma=0.3)
    for obj in objects
}


# -----------------------------
# Geometry helpers
# -----------------------------

def angle_of(vec: np.ndarray) -> float:
    return math.atan2(vec[1], vec[0])


def wrap_to_pi(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def is_in_fov(robot_pos: np.ndarray,
              robot_theta: float,
              target_pos: np.ndarray,
              fov_range: float,
              fov_half_angle: float) -> bool:
    rel = target_pos - robot_pos
    dist = np.linalg.norm(rel)
    if dist > fov_range:
        return False
    ang_rel = angle_of(rel)
    dtheta = wrap_to_pi(ang_rel - robot_theta)
    return abs(dtheta) <= fov_half_angle


def is_point_in_rect(point: np.ndarray,
                     rx: float, ry: float,
                     rw: float, rh: float) -> bool:
    x, y = point
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def segment_intersects_rect(p0: np.ndarray,
                            p1: np.ndarray,
                            rx: float, ry: float,
                            rw: float, rh: float,
                            num_samples: int = 40) -> bool:
    """
    Simple intersection check: sample points along segment and see if
    any fall inside the rectangle. Good enough for this toy demo.
    """
    for alpha in np.linspace(0.0, 1.0, num_samples):
        p = (1.0 - alpha) * p0 + alpha * p1
        if is_point_in_rect(p, rx, ry, rw, rh):
            return True
    return False


def object_is_visible(robot_pos: np.ndarray,
                      robot_theta: float,
                      obj_pos: np.ndarray) -> bool:
    # FOV check
    if not is_in_fov(robot_pos, robot_theta, obj_pos,
                     FOV_RANGE, FOV_HALF_ANGLE_RAD):
        return False

    # Occlusion check
    if segment_intersects_rect(
        robot_pos, obj_pos,
        OCCLUDER_X, OCCLUDER_Y,
        OCCLUDER_W, OCCLUDER_H,
        num_samples=40,
    ):
        return False

    return True


# -----------------------------
# Visualization
# -----------------------------

def plot_world(ax,
               t: int,
               robot_pos: np.ndarray,
               robot_theta: float,
               objects,
               beliefs):
    ax.clear()
    ax.set_xlim(WORLD_X)
    ax.set_ylim(WORLD_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"D48b Multi-object predictive belief demo  |  t = {t}")

    # Draw occluder
    occl = Rectangle(
        (OCCLUDER_X, OCCLUDER_Y),
        OCCLUDER_W, OCCLUDER_H,
        alpha=0.4,
    )
    ax.add_patch(occl)

    # Draw robot
    ax.scatter([robot_pos[0]], [robot_pos[1]], marker="o", s=80, label="robot")
    # Heading arrow
    arrow_len = 0.6
    ax.arrow(
        robot_pos[0], robot_pos[1],
        arrow_len * math.cos(robot_theta),
        arrow_len * math.sin(robot_theta),
        head_width=0.15, length_includes_head=True
    )

    # Draw FOV cone
    fov_start = math.degrees(robot_theta - FOV_HALF_ANGLE_RAD)
    fov_end = math.degrees(robot_theta + FOV_HALF_ANGLE_RAD)
    wedge = Wedge(
        (robot_pos[0], robot_pos[1]),
        FOV_RANGE,
        fov_start, fov_end,
        alpha=0.08,
    )
    ax.add_patch(wedge)

    # Draw true objects and beliefs
    for obj in objects:
        oid = obj["id"]
        true_pos = obj["pos"]
        col = obj["color"]

        # True object
        ax.scatter([true_pos[0]], [true_pos[1]],
                   c=col, marker="o", s=60,
                   label=f"obj{oid} true" if t == 0 else None)

        # Belief
        b = beliefs[oid]
        bx, by = b.mean
        ax.scatter([bx], [by],
                   marker="x", s=80,
                   c=col,
                   label=f"obj{oid} belief" if t == 0 else None)

        # Uncertainty circle
        circ = Circle(
            (bx, by), radius=b.sigma,
            fill=False, linestyle="--", linewidth=0.8
        )
        ax.add_patch(circ)

        # Optional: tiny arrow for believed velocity
        vel_scale = 0.5
        vx, vy = b.vel * vel_scale
        ax.arrow(
            bx, by, vx, vy,
            head_width=0.05, length_includes_head=True,
            linewidth=0.8,
        )

    # Legend only on first frame to avoid clutter
    if t == 0:
        ax.legend(loc="upper right", fontsize=8)


# -----------------------------
# Main loop
# -----------------------------

def main():
    print("[D48b] Multi-object predictive belief demo starting...")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    robot_theta = 0.0

    for t in range(NUM_STEPS):
        # 1) Robot scanning
        robot_theta = wrap_to_pi(robot_theta + ROBOT_ANG_VEL)

        # 2) Move objects (ground truth)
        for obj in objects:
            obj["pos"] += obj["vel"] * DT

        # 3) Predict beliefs forward in time
        for b in beliefs.values():
            b.predict(DT)

        # 4) Visibility + correction
        visible_any = False
        for obj in objects:
            oid = obj["id"]
            pos = obj["pos"]

            visible = object_is_visible(ROBOT_POS, robot_theta, pos)
            b = beliefs[oid]

            if visible:
                visible_any = True
                b.on_observed(pos, DT)
            else:
                b.on_not_observed()

        # 5) Logging
        print(f"\nt={t:03d}  theta={math.degrees(robot_theta):6.1f} deg")
        for obj in objects:
            oid = obj["id"]
            pos = obj["pos"]
            b = beliefs[oid]
            print(
                f"  obj{oid}: "
                f"true=({pos[0]:+4.2f},{pos[1]:+4.2f})  "
                f"B=({b.mean[0]:+4.2f},{b.mean[1]:+4.2f})  "
                f"sigma={b.sigma:4.2f}  "
                f"visible={b.visible}  "
                f"vel=({b.vel[0]:+4.3f},{b.vel[1]:+4.3f})"
            )

        if not visible_any:
            print("  (no objects visible this step – pure prediction)")

        # 6) Draw
        plot_world(ax, t, ROBOT_POS, robot_theta, objects, beliefs)
        plt.pause(0.05)

    print("[D48b] Done. Close the figure window to exit.")
    plt.show()


if __name__ == "__main__":
    main()

