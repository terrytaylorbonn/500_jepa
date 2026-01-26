#!/usr/bin/env python3
"""
D48a: Multi-object belief demo
--------------------------------
Toy 2D world with:
- 1 robot at the origin with limited FOV (field of view)
- 3 static objects
- 1 occluder (rectangle)
- Per-object belief (mean position + uncertainty sigma)

Goal:
Show that the robot maintains a belief over multiple objects that
persists through occlusion and limited FOV.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle

# -----------------------------
# Config
# -----------------------------

WORLD_X = (-1.0, 6.0)
WORLD_Y = (-3.0, 3.0)

NUM_STEPS = 120
DT = 0.1

FOV_RANGE = 5.0
FOV_HALF_ANGLE_DEG = 40.0  # half-angle of cone
FOV_HALF_ANGLE_RAD = math.radians(FOV_HALF_ANGLE_DEG)

# Robot: fixed position, rotating heading
ROBOT_POS = np.array([0.0, 0.0], dtype=float)
ROBOT_ANG_VEL = math.radians(4.0)  # rad/step

# Occluder: simple vertical rectangle
OCCLUDER_X = 2.5
OCCLUDER_Y = -1.0
OCCLUDER_W = 0.5
OCCLUDER_H = 2.0

# Belief dynamics
SIGMA_MIN = 0.05
SIGMA_MAX = 1.0
SIGMA_DECAY = 0.5   # when seen
SIGMA_GROW  = 1.1   # when not seen

# -----------------------------
# Objects and Beliefs
# -----------------------------

objects = [
    {"id": 0, "pos": np.array([2.0,  0.5]), "color": "tab:red"},
    {"id": 1, "pos": np.array([3.2, -0.5]), "color": "tab:green"},
    {"id": 2, "pos": np.array([4.0,  0.0]), "color": "tab:blue"},
]


class ObjectBelief:
    def __init__(self, pos: np.ndarray, sigma: float = 0.1):
        self.mean = pos.astype(float).copy()
        self.sigma = float(sigma)
        self.visible = True

    def on_observed(self, true_pos: np.ndarray):
        # Pull belief to true position, reduce uncertainty
        self.mean = true_pos.astype(float).copy()
        self.sigma = max(self.sigma * SIGMA_DECAY, SIGMA_MIN)
        self.visible = True

    def on_not_observed(self):
        # Keep mean (for static objects) but grow uncertainty
        self.sigma = min(self.sigma * SIGMA_GROW, SIGMA_MAX)
        self.visible = False


beliefs = {
    obj["id"]: ObjectBelief(obj["pos"], sigma=0.2)
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
    Very simple intersection check: sample points along segment and see if
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
    ax.set_title(f"D48a Multi-object belief demo  |  t = {t}")

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
        # Belief position
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

    # Legend only on first frame to avoid clutter
    if t == 0:
        ax.legend(loc="upper right", fontsize=8)


# -----------------------------
# Main loop
# -----------------------------

def main():
    print("[D48a] Multi-object belief demo starting...")

    fig, ax = plt.subplots(figsize=(7, 4))

    # Initial heading
    robot_theta = 0.0

    for t in range(NUM_STEPS):
        # Robot does a slow scan
        robot_theta = wrap_to_pi(robot_theta + ROBOT_ANG_VEL)

        # Update beliefs based on visibility
        visible_any = False
        for obj in objects:
            oid = obj["id"]
            pos = obj["pos"]

            visible = object_is_visible(ROBOT_POS, robot_theta, pos)
            b = beliefs[oid]

            if visible:
                visible_any = True
                b.on_observed(pos)
            else:
                b.on_not_observed()

        # Logging
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
                f"visible={b.visible}"
            )

        if not visible_any:
            print("  (no objects visible this step)")

        # Draw
        plot_world(ax, t, ROBOT_POS, robot_theta, objects, beliefs)
        plt.pause(0.05)

    print("[D48a] Done. Close the figure window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
