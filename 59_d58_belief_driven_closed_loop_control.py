#59_d58_belief_driven_closed_loop_control.py

import argparse, math, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter

# -----------------------------
# Geometry helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def norm2(v):
    return float(np.sqrt(v[0]*v[0] + v[1]*v[1]))

def angle_wrap(a):
    # wrap to [-pi, pi]
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def in_fov(robot_xy, robot_theta, target_xy, fov_range, fov_half_angle_rad):
    dx = target_xy[0] - robot_xy[0]
    dy = target_xy[1] - robot_xy[1]
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > fov_range:
        return False
    ang = math.atan2(dy, dx)
    dth = angle_wrap(ang - robot_theta)
    return abs(dth) <= fov_half_angle_rad

def segment_intersects_rect(p0, p1, rect):
    """
    p0,p1: (x,y) endpoints
    rect: (x1,y1,x2,y2) axis-aligned, x1<x2,y1<y2
    Uses Liang–Barsky style clipping to test intersection.
    """
    x1,y1,x2,y2 = rect
    x0,y0 = p0
    x3,y3 = p1
    dx = x3 - x0
    dy = y3 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - x1, x2 - x0, y0 - y1, y2 - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False
    # Intersection exists if clipped segment has any overlap
    return True

# -----------------------------
# Belief model (scalar sigma)
# -----------------------------
def predict_belief(mu, sigma, sigma_process):
    # simple random-walk growth; mu unchanged
    return mu.copy(), math.sqrt(sigma*sigma + sigma_process*sigma_process)

def fuse_belief(mu_pred, sigma_pred, meas, sigma_meas, gate_k):
    """
    Innovation-gated scalar fusion in 2D:
      innovation = meas - mu_pred
      trust w = exp(-(d^2)/(2*(gate_k*sigma_pred)^2))  in [0,1]
      mu = mu_pred + w * K * innovation
      sigma shrinks toward sigma_meas when w high
    """
    innov = meas - mu_pred
    d = norm2(innov)
    denom = max(1e-6, (gate_k * sigma_pred))
    w = math.exp(-(d*d) / (2.0 * denom*denom))
    # Kalman-like gain (scalar)
    K = (sigma_pred*sigma_pred) / (sigma_pred*sigma_pred + sigma_meas*sigma_meas + 1e-9)
    mu = mu_pred + (w * K) * innov
    # sigma update (shrink when we trust measurement)
    sigma_post = math.sqrt(max(1e-9, (1.0 - w*K) * (sigma_pred*sigma_pred)))
    return mu, sigma_post, innov, d, w, K

# -----------------------------
# Planner (belief-driven pursuit)
# -----------------------------
def plan_action(robot_xy, robot_theta, mu, sigma, cfg):
    """
    Returns (v, w) where
      v = forward speed
      w = angular rate
    Uses belief mean as aimpoint; uncertainty modulates speed.
    """
    dx = mu[0] - robot_xy[0]
    dy = mu[1] - robot_xy[1]
    dist = math.sqrt(dx*dx + dy*dy)

    desired = math.atan2(dy, dx)
    heading_err = angle_wrap(desired - robot_theta)

    # turn rate: proportional, clipped
    w = clamp(cfg.k_turn * heading_err, -cfg.w_max, cfg.w_max)

    # uncertainty-aware speed: confidence from sigma
    # small sigma => confidence ~1, large sigma => confidence ~0
    conf = 1.0 / (1.0 + (sigma / cfg.sigma_conf_scale)**2)

    # speed increases with distance but reduced by uncertainty and heading error
    v_nom = clamp(cfg.k_v * dist, 0.0, cfg.v_max)
    v = v_nom * conf * clamp(1.0 - abs(heading_err)/cfg.slow_turn_angle, 0.0, 1.0)

    return v, w, conf, dist, heading_err

def step_robot(robot_xy, robot_theta, v, w, dt, world_size):
    # unicycle model
    theta = robot_theta + w * dt
    x = robot_xy[0] + v * math.cos(theta) * dt
    y = robot_xy[1] + v * math.sin(theta) * dt
    # clamp to world
    x = clamp(x, 0.0, world_size)
    y = clamp(y, 0.0, world_size)
    return np.array([x,y], np.float32), theta

# -----------------------------
# Target motion (unknown to robot)
# -----------------------------
def step_target(t, target_xy, dt, world_size):
    # smooth wandering target: sinusoidal + drift
    x = target_xy[0] + 0.8*math.cos(0.07*t) * dt + 0.15*dt
    y = target_xy[1] + 0.8*math.sin(0.05*t) * dt + 0.05*dt
    # bounce off walls
    x = clamp(x, 0.0, world_size)
    y = clamp(y, 0.0, world_size)
    return np.array([x,y], np.float32)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--dt", type=float, default=0.15)
    ap.add_argument("--world_size", type=float, default=30.0)

    ap.add_argument("--fov_range", type=float, default=14.0)
    ap.add_argument("--fov_half_angle_deg", type=float, default=70.0)

    ap.add_argument("--meas_noise_std", type=float, default=0.8)
    ap.add_argument("--sigma_meas", type=float, default=0.9)
    ap.add_argument("--sigma_process", type=float, default=0.35)
    ap.add_argument("--gate_k", type=float, default=2.5)

    ap.add_argument("--k_turn", type=float, default=2.2)
    ap.add_argument("--k_v", type=float, default=0.35)
    ap.add_argument("--v_max", type=float, default=3.0)
    ap.add_argument("--w_max", type=float, default=2.8)
    ap.add_argument("--sigma_conf_scale", type=float, default=2.0)
    ap.add_argument("--slow_turn_angle", type=float, default=1.0)  # radians

    ap.add_argument("--obstacle", type=str, default="14,17,24,22")
    ap.add_argument("--out_gif", type=str, default="d58_closed_loop.gif")
    ap.add_argument("--seed", type=int, default=0)
    cfg = ap.parse_args()

    np.random.seed(cfg.seed)

    # obstacle rect
    ox1, oy1, ox2, oy2 = [float(x) for x in cfg.obstacle.split(",")]
    obstacle = (ox1, oy1, ox2, oy2)

    # init world state
    robot_xy = np.array([4.0, 4.0], np.float32)
    robot_theta = math.radians(25.0)

    target_xy = np.array([24.0, 22.0], np.float32)

    # belief about target
    mu = np.array([8.0, 8.0], np.float32)
    sigma = 6.0

    # logging
    hist = {
        "robot": [],
        "target": [],
        "mu": [],
        "sigma": [],
        "visible": [],
        "meas": [],
        "v": [],
        "w": [],
        "conf": [],
        "innov_d": [],
        "gate_w": [],
        "pos_err": [],
    }

    fov_half = math.radians(cfg.fov_half_angle_deg)

    for t in range(cfg.steps):
        # --- world advances (target moves on its own) ---
        target_xy = step_target(t, target_xy, cfg.dt, cfg.world_size)

        # --- prediction (belief drift) ---
        mu_pred, sigma_pred = predict_belief(mu, sigma, cfg.sigma_process)

        # --- sensing: visibility (FOV + occlusion) ---
        vis = in_fov(robot_xy, robot_theta, target_xy, cfg.fov_range, fov_half)
        if vis:
            # occlusion by obstacle: if segment robot->target intersects obstacle, then occluded
            if segment_intersects_rect(robot_xy, target_xy, obstacle):
                vis = False

        meas = None
        innov_d = 0.0
        gate_w = 0.0

        # --- measurement / fusion ---
        if vis:
            noise = cfg.meas_noise_std * np.random.randn(2).astype(np.float32)
            meas = target_xy + noise
            mu, sigma, innov, innov_d, gate_w, K = fuse_belief(
                mu_pred, sigma_pred, meas, cfg.sigma_meas, cfg.gate_k
            )
        else:
            mu, sigma = mu_pred, sigma_pred

        # --- planner uses belief to choose action ---
        v, w, conf, dist, heading_err = plan_action(robot_xy, robot_theta, mu, sigma, cfg)

        # --- execute action (robot moves) ---
        robot_xy, robot_theta = step_robot(robot_xy, robot_theta, v, w, cfg.dt, cfg.world_size)

        # --- logs ---
        hist["robot"].append(robot_xy.copy())
        hist["target"].append(target_xy.copy())
        hist["mu"].append(mu.copy())
        hist["sigma"].append(float(sigma))
        hist["visible"].append(bool(vis))
        hist["meas"].append(None if meas is None else meas.copy())
        hist["v"].append(float(v))
        hist["w"].append(float(w))
        hist["conf"].append(float(conf))
        hist["innov_d"].append(float(innov_d))
        hist["gate_w"].append(float(gate_w))
        hist["pos_err"].append(float(norm2(mu - target_xy)))

        if (t % 20) == 0:
            print(f"[D58][t={t:03d}] vis={int(vis)}  sigma={sigma:5.2f}  conf={conf:4.2f}  "
                  f"v={v:4.2f}  w={w:4.2f}  |mu-true|={hist['pos_err'][-1]:5.2f}  gate_w={gate_w:4.2f}")

    # -----------------------------
    # Visualization -> GIF
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_title("D58 — Belief-Driven Closed Loop Control")

    # obstacle
    rect = Rectangle((ox1, oy1), ox2-ox1, oy2-oy1, fill=False, linewidth=2)
    ax.add_patch(rect)

    # artists
    robot_dot = Circle((0,0), radius=0.6, fill=False, linewidth=2)
    ax.add_patch(robot_dot)
    robot_heading, = ax.plot([], [], linewidth=2)

    target_dot = Circle((0,0), radius=0.5, fill=False, linewidth=2)
    ax.add_patch(target_dot)

    mu_dot = Circle((0,0), radius=0.5, fill=False, linewidth=2)
    ax.add_patch(mu_dot)

    sigma_circle = Circle((0,0), radius=1.0, fill=False, linewidth=1, linestyle="--")
    ax.add_patch(sigma_circle)

    meas_dot, = ax.plot([], [], marker="x", linestyle="")

    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    def update(frame):
        r = hist["robot"][frame]
        tgt = hist["target"][frame]
        mu_ = hist["mu"][frame]
        sig = hist["sigma"][frame]
        vis = hist["visible"][frame]
        meas = hist["meas"][frame]
        conf = hist["conf"][frame]
        v = hist["v"][frame]
        w = hist["w"][frame]
        err = hist["pos_err"][frame]

        # robot
        robot_dot.center = (r[0], r[1])
        # heading line
        th = math.atan2(
            (hist["robot"][min(frame+1, len(hist["robot"])-1)][1] - r[1]),
            (hist["robot"][min(frame+1, len(hist["robot"])-1)][0] - r[0])
        )
        hx = r[0] + 1.2*math.cos(th)
        hy = r[1] + 1.2*math.sin(th)
        robot_heading.set_data([r[0], hx], [r[1], hy])

        # target
        target_dot.center = (tgt[0], tgt[1])

        # belief mean
        mu_dot.center = (mu_[0], mu_[1])

        # sigma radius
        sigma_circle.center = (mu_[0], mu_[1])
        sigma_circle.set_radius(max(0.3, min(8.0, sig)))

        # measurement
        if meas is not None:
            meas_dot.set_data([meas[0]], [meas[1]])
        else:
            meas_dot.set_data([], [])

        # text
        text.set_text(
            f"t={frame:03d}  vis={int(vis)}  sigma={sig:4.2f}  conf={conf:4.2f}\n"
            f"v={v:4.2f}  w={w:4.2f}  |mu-true|={err:4.2f}"
        )
        return robot_dot, robot_heading, target_dot, mu_dot, sigma_circle, meas_dot, text

    anim = FuncAnimation(fig, update, frames=cfg.steps, interval=30, blit=True)
    print(f"[D58] Saving GIF -> {cfg.out_gif}")
    anim.save(cfg.out_gif, writer=PillowWriter(fps=25))
    print("[D58] Done.")

if __name__ == "__main__":
    main()
