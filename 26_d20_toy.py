#26_d20_toy.py

import numpy as np

# -----------------------------
# 1. Toy 1D car dynamics
# -----------------------------
def step_dynamics(state, action, dt=0.1):
    """
    state: [x, v]
    action: acceleration (scalar)
    """
    x, v = state
    a = np.clip(action, -1.0, 1.0)
    v_next = v + a * dt
    x_next = x + v_next * dt
    return np.array([x_next, v_next], dtype=np.float32)


# -----------------------------
# 2. Rollout trajectory
# -----------------------------
def rollout_trajectory(state0, actions, dt=0.1):
    H = len(actions)
    traj = np.zeros((H + 1, 2), dtype=np.float32)
    traj[0] = state0
    state = state0.copy()
    for t in range(H):
        state = step_dynamics(state, actions[t], dt=dt)
        traj[t + 1] = state
    return traj


# -----------------------------
# 3. Multi-part score (D20 style)
# -----------------------------
def score_trajectory(traj, actions, x_goal,
                     w_terminal=1.0, w_path=0.1, w_action=0.01):
    x_traj = traj[:, 0]
    x_final = x_traj[-1]

    terminal_dist = np.abs(x_final - x_goal)
    path_dist = np.mean(np.abs(x_traj - x_goal))
    action_cost = np.mean(actions ** 2)

    total_cost = (
        w_terminal * terminal_dist +
        w_path * path_dist +
        w_action * action_cost
    )
    return -total_cost  # maximize score


# -----------------------------
# 4. D20 = multi-horizon CEM
# -----------------------------
def cem_plan_multi_horizon(
    state0,
    x_goal,
    horizons=(5, 10, 15),
    K=128,
    elite_frac=0.2,
    iters=5,
    init_std=1.0,
    dt=0.1,
):
    best_overall_score = -np.inf
    best_overall_actions = None
    best_overall_H = None

    for H in horizons:
        mean = np.zeros(H, dtype=np.float32)
        std = np.ones(H, dtype=np.float32) * init_std

        for it in range(iters):
            actions_batch = np.random.randn(K, H).astype(np.float32) * std + mean
            actions_batch = np.clip(actions_batch, -1.0, 1.0)

            scores = np.zeros(K, dtype=np.float32)
            for k in range(K):
                a_seq = actions_batch[k]
                traj = rollout_trajectory(state0, a_seq, dt=dt)
                scores[k] = score_trajectory(traj, a_seq, x_goal)

            elite_count = max(1, int(elite_frac * K))
            elite_idxs = np.argsort(scores)[-elite_count:]
            elite_actions = actions_batch[elite_idxs]

            mean = elite_actions.mean(axis=0)
            std = elite_actions.std(axis=0) + 1e-3

        best_idx = np.argmax(scores)
        best_actions = actions_batch[best_idx]
        best_score = scores[best_idx]

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_actions = best_actions
            best_overall_H = H

    return best_overall_actions[0], {
        "best_score": best_overall_score,
        "best_horizon": best_overall_H,
        "best_actions": best_overall_actions,
    }


# -----------------------------
# 5. Run the control loop
# -----------------------------
if __name__ == "__main__":
    state = np.array([-3.0, 0.0], dtype=np.float32)  # x=-3, v=0
    x_goal = 2.0

    for t in range(20):
        a, info = cem_plan_multi_horizon(state, x_goal)
        state = step_dynamics(state, a, dt=0.1)

        print(f"t={t:02d}  a={a:+.3f}  "
              f"x={state[0]:+.3f}  v={state[1]:+.3f}  "
              f"H*={info['best_horizon']}  score={info['best_score']:.3f}")
