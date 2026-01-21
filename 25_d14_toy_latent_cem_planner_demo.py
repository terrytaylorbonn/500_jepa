# 27_d30_minimal_2d.py
import numpy as np

# ==========================
# 1. 2D dynamics (toy world)
# ==========================

def step_dynamics_2d(state, action, dt=0.1):
    """
    state: [x, y, vx, vy]
    action: [ax, ay]
    """
    x, y, vx, vy = state
    ax, ay = action
    ax = np.clip(ax, -1.0, 1.0)
    ay = np.clip(ay, -1.0, 1.0)

    vx_next = vx + ax * dt
    vy_next = vy + ay * dt
    x_next = x + vx_next * dt
    y_next = y + vy_next * dt

    return np.array([x_next, y_next, vx_next, vy_next], dtype=np.float32)


def rollout_trajectory_2d(state0, actions, dt=0.1):
    """
    state0: (4,) [x, y, vx, vy]
    actions: (H, 2)
    returns:
        traj_states: (H+1, 4)
    """
    H = len(actions)
    traj = np.zeros((H + 1, 4), dtype=np.float32)
    traj[0] = state0
    state = state0.copy()
    for t in range(H):
        state = step_dynamics_2d(state, actions[t], dt=dt)
        traj[t + 1] = state
    return traj


# ===================================
# 2. Cost: terminal, path, effort
# ===================================

def score_trajectory_2d(traj, actions, goal_xy,
                        w_terminal=1.0, w_path=0.1, w_action=0.01):
    """
    traj: (H+1, 4)
    actions: (H, 2)
    goal_xy: (2,)
    """
    xy_traj = traj[:, :2]  # (H+1, 2)
    xy_final = xy_traj[-1]

    # 2D Euclidean distances
    terminal_dist = np.linalg.norm(xy_final - goal_xy)
    path_dist = np.mean(np.linalg.norm(xy_traj - goal_xy, axis=1))
    action_cost = np.mean(np.sum(actions ** 2, axis=1))

    total_cost = (
        w_terminal * terminal_dist +
        w_path * path_dist +
        w_action * action_cost
    )
    return -total_cost  # maximize score


# ===================================
# 3. CEM planner (fixed horizon)
# ===================================

def cem_plan_2d(
    state0,
    goal_xy,
    H=10,
    K=128,
    elite_frac=0.2,
    iters=5,
    init_std=1.0,
    dt=0.1,
):
    """
    D30-minimal planner:
    - still CEM/MPC
    - now in 2D
    """
    mean = np.zeros((H, 2), dtype=np.float32)
    std = np.ones((H, 2), dtype=np.float32) * init_std

    best_actions = None
    best_score = -np.inf

    for it in range(iters):
        # 1) sample candidates: (K, H, 2)
        actions_batch = (
            np.random.randn(K, H, 2).astype(np.float32) * std + mean
        )
        actions_batch = np.clip(actions_batch, -1.0, 1.0)

        scores = np.zeros(K, dtype=np.float32)

        # 2) evaluate each candidate
        for k in range(K):
            a_seq = actions_batch[k]
            traj = rollout_trajectory_2d(state0, a_seq, dt=dt)
            scores[k] = score_trajectory_2d(traj, a_seq, goal_xy)

        # 3) keep elites
        elite_count = max(1, int(elite_frac * K))
        elite_idxs = np.argsort(scores)[-elite_count:]
        elite_actions = actions_batch[elite_idxs]

        # store best of this iteration
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_actions = actions_batch[max_idx].copy()

        # 4) refit Gaussian to elites
        mean = elite_actions.mean(axis=0)
        std = elite_actions.std(axis=0) + 1e-3  # avoid zero std

    best_first_action = best_actions[0]
    debug_info = {
        "best_score": best_score,
        "best_actions": best_actions,
    }
    return best_first_action, debug_info


# ===================================
# 4. "Perception" layer (minimal)
# ===================================

def render_world_to_grid(state, goal_xy, grid_size=32):
    """
    Very simple "image":
      0 = empty
      1 = robot
      2 = goal

    state: (4,) [x, y, vx, vy]
    goal_xy: (2,)
    """
    img = np.zeros((grid_size, grid_size), dtype=np.int32)

    def to_pix(p):
        # map world coords [-5, 5] x [-5, 5] → grid [0..grid_size-1]
        x, y = p
        gx = int(np.clip((x + 5.0) / 10.0 * (grid_size - 1), 0, grid_size - 1))
        gy = int(np.clip((y + 5.0) / 10.0 * (grid_size - 1), 0, grid_size - 1))
        return gx, gy

    robot_xy = state[:2]
    rx, ry = to_pix(robot_xy)
    gx, gy = to_pix(goal_xy)

    img[ry, rx] = 1  # robot
    img[gy, gx] = 2  # goal

    return img


def perceive_from_grid(img):
    """
    "Vision" → latent.
    Find robot and goal positions in pixel space.

    Returns:
        robot_pos: (2,) [px, py]
        goal_pos:  (2,) [px, py]
    """
    robot_pixels = np.argwhere(img == 1)
    goal_pixels = np.argwhere(img == 2)

    if len(robot_pixels) == 0 or len(goal_pixels) == 0:
        raise ValueError("Robot or goal missing from image!")

    # Just take the first if multiple
    ry, rx = robot_pixels[0]
    gy, gx = goal_pixels[0]

    robot_pix = np.array([rx, ry], dtype=np.float32)
    goal_pix = np.array([gx, gy], dtype=np.float32)
    return robot_pix, goal_pix


# ===================================
# 5. Main loop (D30 minimal)
# ===================================

if __name__ == "__main__":
    # World: coordinates in [-5, 5] x [-5, 5]
    state = np.array([-4.0, -4.0, 0.0, 0.0], dtype=np.float32)  # [x, y, vx, vy]
    goal_xy = np.array([3.0, 2.0], dtype=np.float32)

    for t in range(20):
        # --- "Perception": world → image → latent ---
        img = render_world_to_grid(state, goal_xy, grid_size=32)
        robot_pix, goal_pix = perceive_from_grid(img)

        # You can log these to see image-space behavior
        # print(f"vision(t={t}): robot_pix={robot_pix}, goal_pix={goal_pix}")

        # Planner still works in world coordinates, but conceptually:
        # vision → (x,y) latent → planner → action
        a, info = cem_plan_2d(state, goal_xy, H=10, K=128, iters=5)

        # --- Apply greedy first action (MPC) ---
        state = step_dynamics_2d(state, a, dt=0.1)

        print(
            f"t={t:02d}  "
            f"a=({a[0]:+0.3f}, {a[1]:+0.3f})  "
            f"x=({state[0]:+0.3f}, {state[1]:+0.3f})  "
            f"v=({state[2]:+0.3f}, {state[3]:+0.3f})  "
            f"score={info['best_score']:+0.3f}"
        )

