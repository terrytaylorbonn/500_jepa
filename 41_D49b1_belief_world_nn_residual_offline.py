#!/usr/bin/env python3
"""
41_D49b1_belief_world_nn_residual_offline.py
[41_D49b1] 2D belief world + offline NN residual (ΔB) training

- Same world as D49a: robot + 3 objects, belief B, sigma, visibility.
- Phase 1: run a simulation with classical belief update and record a dataset.
- Phase 2: train a tiny MLP to predict residual ΔB = B_true_next - B_pred.
- Phase 3: run a second simulation where belief update uses:
      B_next = B_pred + NN([B_t, action_features])

This is the first "learned world-model transition" rung in the ladder.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Utility
# -----------------------------

def angle_wrap(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# -----------------------------
# World / object definitions
# -----------------------------

@dataclass
class ObjectState:
    true_pos: np.ndarray      # shape (2,)
    vel: np.ndarray           # shape (2,)
    B: np.ndarray             # belief position (2,)
    sigma: float
    visible: bool
    last_seen: int


@dataclass
class RobotState:
    pos: np.ndarray  # shape (2,)
    theta: float     # heading (rad)


# -----------------------------
# Config parameters
# -----------------------------

DT = 0.1
NUM_STEPS_ROLLOUT = 300
NUM_STEPS_EVAL = 300

ALPHA_MEAS = 0.5     # measurement blending (same spirit as D49a)
BETA_GROW = 1.01     # sigma growth when unseen
BETA_SHRINK = 0.7    # sigma shrink when seen
SIGMA_MIN = 0.05
SIGMA_MAX = 1.5

FOV_RANGE = 5.0
FOV_HALF_ANGLE_DEG = 70.0
FOV_HALF_ANGLE = math.radians(FOV_HALF_ANGLE_DEG)

# NN / training config
INPUT_DIM = 5          # [B.x, B.y, dx_robot, dy_robot, dtheta]
HIDDEN_DIM = 64
OUTPUT_DIM = 2         # ΔB = [dx, dy]
NUM_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


# -----------------------------
# Visibility check
# -----------------------------

def is_visible(robot: RobotState, obj_pos: np.ndarray) -> bool:
    """Check if object is within FOV range + angle."""
    rel = obj_pos - robot.pos
    dist = np.linalg.norm(rel)
    if dist > FOV_RANGE:
        return False
    bearing = math.atan2(rel[1], rel[0])
    dtheta = angle_wrap(bearing - robot.theta)
    return abs(dtheta) <= FOV_HALF_ANGLE


# -----------------------------
# Robot control: simple pattern
# -----------------------------

def robot_control_policy(t: int) -> Tuple[float, float]:
    """
    Simple v, omega policy similar in spirit to D49a:
    - small forward motion
    - occasional head turns to create visibility changes
    """
    v = 0.2

    # Make the robot do a small scan left/right in a time window
    time = t * DT
    omega = 0.0
    if 5.0 < time < 8.0:
        omega = math.radians(20.0)
    elif 8.0 <= time < 11.0:
        omega = math.radians(-20.0)
    else:
        omega = 0.0

    return v, omega


# -----------------------------
# World initialization
# -----------------------------

def init_world(seed: int = 0) -> Tuple[RobotState, List[ObjectState]]:
    random.seed(seed)
    np.random.seed(seed)

    robot = RobotState(pos=np.array([0.0, 0.0], dtype=np.float32),
                       theta=0.0)

    objects: List[ObjectState] = []

    # Three objects, similar to D49a
    # obj0: ahead
    objects.append(ObjectState(
        true_pos=np.array([2.0, 0.0], dtype=np.float32),
        vel=np.array([0.01, 0.0], dtype=np.float32),
        B=np.array([2.0, 0.0], dtype=np.float32),
        sigma=0.06,
        visible=True,
        last_seen=0,
    ))

    # obj1: slightly up
    objects.append(ObjectState(
        true_pos=np.array([3.0, 1.0], dtype=np.float32),
        vel=np.array([0.01, -0.01], dtype=np.float32),
        B=np.array([3.0, 1.0], dtype=np.float32),
        sigma=0.06,
        visible=True,
        last_seen=0,
    ))

    # obj2: slightly down
    objects.append(ObjectState(
        true_pos=np.array([3.5, -1.0], dtype=np.float32),
        vel=np.array([0.01, 0.0], dtype=np.float32),
        B=np.array([3.5, -1.0], dtype=np.float32),
        sigma=0.06,
        visible=True,
        last_seen=0,
    ))

    return robot, objects


# -----------------------------
# Belief / dynamics step (classical)
# -----------------------------

def step_world_classical(robot: RobotState,
                         objects: List[ObjectState],
                         t: int,
                         noisy_measurement: bool = True
                         ) -> Tuple[RobotState, List[ObjectState]]:
    """
    One step of world with classical belief update (no NN).
    Returns updated robot + objects.
    """
    # Robot control
    v, omega = robot_control_policy(t)
    dx = v * math.cos(robot.theta) * DT
    dy = v * math.sin(robot.theta) * DT
    dtheta = omega * DT

    robot.pos = robot.pos + np.array([dx, dy], dtype=np.float32)
    robot.theta = angle_wrap(robot.theta + dtheta)

    # Update each object
    for obj in objects:
        # True motion
        obj.true_pos = obj.true_pos + obj.vel * DT

        # Predict belief position (simple "copy true velocity" model)
        B_pred = obj.B + obj.vel * DT
        sigma_pred = min(SIGMA_MAX, obj.sigma * BETA_GROW)

        # Visibility
        obj.visible = is_visible(robot, obj.true_pos)

        if obj.visible:
            # Measurement = true_pos + small noise
            if noisy_measurement:
                noise = 0.02 * np.random.randn(2).astype(np.float32)
            else:
                noise = np.zeros(2, dtype=np.float32)
            meas = obj.true_pos + noise

            # Blend with prediction
            obj.B = (1.0 - ALPHA_MEAS) * B_pred + ALPHA_MEAS * meas
            obj.sigma = max(SIGMA_MIN, BETA_SHRINK * sigma_pred)
            obj.last_seen = t
        else:
            # No measurement; belief just drifts with sigma growing
            obj.B = B_pred
            obj.sigma = sigma_pred

    return robot, objects


# -----------------------------
# Rollout & dataset collection
# -----------------------------

def collect_rollout_dataset(num_steps: int = NUM_STEPS_ROLLOUT,
                            seed: int = 0):
    """
    Run world with classical belief update and collect:
      X = [B_t.x, B_t.y, dx_robot, dy_robot, dtheta]
      Y = ΔB = true_pos_{t+1} - B_pred

    Returns:
      X_data: (N, 5)
      Y_data: (N, 2)
    """
    robot, objects = init_world(seed=seed)

    X_list = []
    Y_list = []

    for t in range(num_steps):
        # Record robot state before step
        v, omega = robot_control_policy(t)
        dx = v * math.cos(robot.theta) * DT
        dy = v * math.sin(robot.theta) * DT
        dtheta = omega * DT

        # Save copies of current belief & true before we mutate them
        B_t_list = [obj.B.copy() for obj in objects]
        true_t_list = [obj.true_pos.copy() for obj in objects]

        # Predict B_pred (using same rule as step_world_classical)
        B_pred_list = [B_t + obj.vel * DT for B_t, obj in zip(B_t_list, objects)]

        # Step world (this will compute new true_pos & B)
        robot, objects = step_world_classical(robot, objects, t)

        # After stepping, we have true_pos_{t+1}
        true_next_list = [obj.true_pos.copy() for obj in objects]

        # For each object, form training pair
        for B_t, B_pred, true_next in zip(B_t_list, B_pred_list, true_next_list):
            # Input features (5-dim)
            x_feat = np.array([B_t[0], B_t[1], dx, dy, dtheta], dtype=np.float32)

            # Target residual: ΔB = true_next - B_pred
            delta_B = (true_next - B_pred).astype(np.float32)

            X_list.append(x_feat)
            Y_list.append(delta_B)

        if t % 20 == 0:
            print(f"[rollout] t={t:3d}  robot=({robot.pos[0]:+.2f},{robot.pos[1]:+.2f}) theta={math.degrees(robot.theta):6.1f} deg")

    X_data = np.stack(X_list, axis=0)
    Y_data = np.stack(Y_list, axis=0)
    print(f"[rollout] Collected dataset: X {X_data.shape}, Y {Y_data.shape}")
    return X_data, Y_data


# -----------------------------
# Tiny residual MLP
# -----------------------------

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_residual_mlp(X_data: np.ndarray,
                       Y_data: np.ndarray,
                       num_epochs: int = NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    X_tensor = torch.from_numpy(X_data).to(device)
    Y_tensor = torch.from_numpy(Y_data).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ResidualMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"[train] epoch {epoch:2d}/{num_epochs}  loss={avg_loss:.6f}")

    return model


# -----------------------------
# Simulation using NN residual
# -----------------------------

def step_world_with_residual(robot: RobotState,
                             objects: List[ObjectState],
                             model: ResidualMLP,
                             t: int) -> Tuple[RobotState, List[ObjectState]]:
    """
    One step of world where belief update uses:
      B_pred = classical_motion(B_t)
      ΔB_hat = NN([B_t, action_features])
      B_next = B_pred + ΔB_hat
    """
    device = next(model.parameters()).device

    # Robot control
    v, omega = robot_control_policy(t)
    dx = v * math.cos(robot.theta) * DT
    dy = v * math.sin(robot.theta) * DT
    dtheta = omega * DT

    robot.pos = robot.pos + np.array([dx, dy], dtype=np.float32)
    robot.theta = angle_wrap(robot.theta + dtheta)

    # Update each object
    for obj in objects:
        # True motion (same as before)
        obj.true_pos = obj.true_pos + obj.vel * DT

        # Predict belief using classical model
        B_pred = obj.B + obj.vel * DT
        sigma_pred = min(SIGMA_MAX, obj.sigma * BETA_GROW)

        # NN residual
        x_feat = np.array([obj.B[0], obj.B[1], dx, dy, dtheta], dtype=np.float32)
        x_tensor = torch.from_numpy(x_feat[None, :]).to(device)
        with torch.no_grad():
            delta_hat = model(x_tensor).cpu().numpy()[0]

        B_nn = B_pred + delta_hat.astype(np.float32)

        # Visibility
        obj.visible = is_visible(robot, obj.true_pos)

        if obj.visible:
            # measurement (optional small noise)
            noise = 0.02 * np.random.randn(2).astype(np.float32)
            meas = obj.true_pos + noise
            # Blend NN belief with measurement
            obj.B = (1.0 - ALPHA_MEAS) * B_nn + ALPHA_MEAS * meas
            obj.sigma = max(SIGMA_MIN, BETA_SHRINK * sigma_pred)
            obj.last_seen = t
        else:
            # No measurement; purely NN + classical
            obj.B = B_nn
            obj.sigma = sigma_pred

    return robot, objects


def run_eval_with_and_without_nn(model: ResidualMLP,
                                 num_steps: int = NUM_STEPS_EVAL,
                                 seed: int = 1):
    """
    Run two simulations from the same initial world:
      - one with classical belief update
      - one with NN residual
    Plot trajectories + belief errors.
    """
    # Classical world
    robot_cl, objs_cl = init_world(seed=seed)
    # NN world (deep copy)
    robot_nn, objs_nn = init_world(seed=seed)

    # For plotting
    true_trajs = [[] for _ in objs_cl]
    B_cl_trajs = [[] for _ in objs_cl]
    B_nn_trajs = [[] for _ in objs_nn]

    for t in range(num_steps):
        # Classical step
        robot_cl, objs_cl = step_world_classical(robot_cl, objs_cl, t)
        # NN step
        robot_nn, objs_nn = step_world_with_residual(robot_nn, objs_nn, model, t)

        # Record
        for i, (oc, on) in enumerate(zip(objs_cl, objs_nn)):
            true_trajs[i].append(oc.true_pos.copy())
            B_cl_trajs[i].append(oc.B.copy())
            B_nn_trajs[i].append(on.B.copy())

        if t % 20 == 0:
            print(f"[eval] t={t:3d}  robot_nn=({robot_nn.pos[0]:+.2f},{robot_nn.pos[1]:+.2f})")

    # Convert to arrays
    true_trajs = [np.stack(tr, axis=0) for tr in true_trajs]
    B_cl_trajs = [np.stack(tr, axis=0) for tr in B_cl_trajs]
    B_nn_trajs = [np.stack(tr, axis=0) for tr in B_nn_trajs]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax0, ax1 = axes

    colors = ["tab:blue", "tab:orange", "tab:green"]

    # Trajectories in xy
    for i, (tru, bcl, bnn, c) in enumerate(zip(true_trajs, B_cl_trajs, B_nn_trajs, colors)):
        ax0.plot(tru[:, 0], tru[:, 1], color=c, linestyle="-", label=f"obj{i} true")
        ax0.plot(bcl[:, 0], bcl[:, 1], color=c, linestyle="--", alpha=0.6, label=f"obj{i} B_classic")
        ax0.plot(bnn[:, 0], bnn[:, 1], color=c, linestyle=":", alpha=0.9, label=f"obj{i} B_nn")

    ax0.set_title("Trajectories (true vs classical vs NN)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.axis("equal")
    ax0.grid(True)
    ax0.legend(fontsize=7)

    # Belief error over time (L2)
    for i, (tru, bcl, bnn, c) in enumerate(zip(true_trajs, B_cl_trajs, B_nn_trajs, colors)):
        err_cl = np.linalg.norm(tru - bcl, axis=1)
        err_nn = np.linalg.norm(tru - bnn, axis=1)
        ax1.plot(err_cl, color=c, linestyle="--", alpha=0.7, label=f"obj{i} err_classic")
        ax1.plot(err_nn, color=c, linestyle="-", alpha=0.9, label=f"obj{i} err_nn")

    ax1.set_title("Belief error ||true - B||")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("error")
    ax1.grid(True)
    ax1.legend(fontsize=7)

    fig.suptitle("[D49b1] Offline NN residual on belief dynamics")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------

def main():
    print("[D49b1] Collecting rollout dataset...")
    X_data, Y_data = collect_rollout_dataset(NUM_STEPS_ROLLOUT, seed=0)

    print("[D49b1] Training residual MLP...")
    model = train_residual_mlp(X_data, Y_data, num_epochs=NUM_EPOCHS)

    print("[D49b1] Evaluating with and without NN residual...")
    run_eval_with_and_without_nn(model, NUM_STEPS_EVAL, seed=1)

    print("[D49b1] Done. Close the figure window to exit.")


if __name__ == "__main__":
    main()

