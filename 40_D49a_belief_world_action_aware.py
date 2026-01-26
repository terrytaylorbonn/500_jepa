#!/usr/bin/env python
"""
40_D49a_belief_world_action_aware.py
[D49a] 2D belief world with action-aware uncertainty

Focus is on the updated belief update function and the use of a_t.

- Robot moves in 2D with heading theta.
- 3 objects with true positions + velocities.
- For each object we maintain:
    B (belief position), sigma (uncertainty), visible flag, last_seen.
- Belief update depends on:
    - visibility
    - robot action a_t = (dx, dy, dtheta)
"""

import math
import numpy as np
import matplotlib.pyplot as plt

DT = 0.1
T_STEPS = 250

# Field-of-view parameters
FOV_HALF_ANGLE = math.radians(60.0)  # +-60 deg
FOV_RANGE = 4.0

# Belief update hyperparams
SIGMA_BASE = 0.01
K_MOVE = 0.1
K_TURN = 0.2
ALPHA_MEAS = 0.5
BETA_SHRINK = 0.5
SIGMA_MIN = 0.05

rng = np.random.default_rng(0)


class Object2D:
    def __init__(self, x, y, vx, vy):
        self.true_pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)

        self.B = self.true_pos.copy()
        self.sigma = 0.1
        self.visible = False
        self.last_seen = 0

    def step_true(self):
        self.true_pos = self.true_pos + DT * self.vel


def angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def in_fov(robot_pos, robot_theta, obj_pos):
    # vector from robot to object
    rel = obj_pos - robot_pos
    dist = np.linalg.norm(rel)
    if dist > FOV_RANGE:
        return False

    angle_to_obj = math.atan2(rel[1], rel[0])
    dtheta = angle_wrap(angle_to_obj - robot_theta)
    return abs(dtheta) <= FOV_HALF_ANGLE


def simulate():
    # Robot state
    robot_pos = np.array([0.0, 0.0], dtype=float)
    robot_theta = math.radians(0.0)

    # Define 3 objects
    objs = [
        Object2D(2.0, 0.0, 0.01, 0.0),
        Object2D(3.0, 1.0, 0.005, -0.005),
        Object2D(3.5, -1.0, 0.008, 0.002),
    ]

    fig, ax = plt.subplots(figsize=(6, 6))

    prev_robot_pos = robot_pos.copy()
    prev_theta = robot_theta

    for t in range(T_STEPS):
        time = t * DT

        # ------------------------------
        # 1) Define robot action a_t
        # ------------------------------
        # Example: simple patrol / sweep pattern
        v_forward = 0.02
        omega = math.radians(0.0)

        # You can make omega non-zero in segments to see sigma changes
        if 5.0 < time < 8.0:
            omega = math.radians(20.0)
        elif 8.0 <= time < 11.0:
            omega = math.radians(-20.0)

        # Integrate robot motion
        robot_theta = angle_wrap(robot_theta + omega * DT)
        robot_pos = robot_pos + DT * v_forward * np.array(
            [math.cos(robot_theta), math.sin(robot_theta)]
        )

        # Action deltas
        dx, dy = robot_pos - prev_robot_pos
        dtheta = angle_wrap(robot_theta - prev_theta)
        action = np.array([dx, dy, dtheta], dtype=float)

        prev_robot_pos = robot_pos.copy()
        prev_theta = robot_theta

        # ------------------------------
        # 2) Step true objects
        # ------------------------------
        for obj in objs:
            obj.step_true()

        # ------------------------------
        # 3) Compute visibility (simple FOV + no occlusion)
        #    (You can extend to occlusion later if desired.)
        # ------------------------------
        for obj in objs:
            obj.visible = in_fov(robot_pos, robot_theta, obj.true_pos)

        # ------------------------------
        # 4) Belief + sigma update per object
        # ------------------------------
        for i, obj in enumerate(objs):
            # Predict step
            B_pred = obj.B + DT * obj.vel
            sigma_pred = obj.sigma + SIGMA_BASE

            if obj.visible:
                # Measurement update
                obj.B = (1.0 - ALPHA_MEAS) * B_pred + ALPHA_MEAS * obj.true_pos
                obj.sigma = max(SIGMA_MIN, BETA_SHRINK * sigma_pred)
                obj.last_seen = t
            else:
                # No measurement – action-aware uncertainty growth
                move_mag = math.sqrt(action[0] ** 2 + action[1] ** 2)
                turn_mag = abs(action[2])

                delta_move = K_MOVE * move_mag
                delta_turn = K_TURN * turn_mag

                obj.B = B_pred
                obj.sigma = sigma_pred + delta_move + delta_turn

        # ------------------------------
        # 5) Logging (partial, like your snippet)
        # ------------------------------
        if t % 10 == 0:
            print(f"t={t}")
            print(
                f"  robot_pos=(+{robot_pos[0]:.2f},+{robot_pos[1]:.2f})  "
                f"theta={math.degrees(robot_theta):6.1f} deg"
            )
            for i, obj in enumerate(objs):
                vis = "True" if obj.visible else "False"
                print(
                    f"  obj{i}: true=(+{obj.true_pos[0]:.2f},+{obj.true_pos[1]:.2f})  "
                    f"B=(+{obj.B[0]:.2f},+{obj.B[1]:.2f})  "
                    f"sigma={obj.sigma:.2f}  visible={vis}"
                )
            print()

        # ------------------------------
        # 6) Simple visualization
        # ------------------------------
        ax.clear()
        ax.set_title(f"[49a] t={t}, theta={math.degrees(robot_theta):.1f} deg")
        ax.set_xlim(-1, 5)
        ax.set_ylim(-2, 3)
        ax.set_aspect("equal")

        # Robot
        ax.plot(robot_pos[0], robot_pos[1], "ko")
        ax.arrow(
            robot_pos[0],
            robot_pos[1],
            0.5 * math.cos(robot_theta),
            0.5 * math.sin(robot_theta),
            head_width=0.1,
            length_includes_head=True,
        )

        # Draw FOV lines
        for sign in (-1, +1):
            angle = robot_theta + sign * FOV_HALF_ANGLE
            ax.plot(
                [robot_pos[0], robot_pos[0] + FOV_RANGE * math.cos(angle)],
                [robot_pos[1], robot_pos[1] + FOV_RANGE * math.sin(angle)],
                "k--",
                linewidth=0.5,
            )

        # Objects (true + belief)
        colors = ["r", "g", "b"]
        for i, obj in enumerate(objs):
            c = colors[i]
            # True position
            ax.plot(obj.true_pos[0], obj.true_pos[1], c + "o", label=f"obj{i}_true")
            # Belief
            ax.plot(obj.B[0], obj.B[1], c + "x", label=f"obj{i}_B")

            # Draw sigma as circle
            circle = plt.Circle(
                obj.B, obj.sigma, fill=False, linestyle="--", color=c, alpha=0.7
            )
            ax.add_patch(circle)

        ax.legend(loc="upper right", fontsize=8)
        plt.pause(0.01)

    print("[49a] Done. Close the figure window to exit.")
    plt.show()


if __name__ == "__main__":
    simulate()


