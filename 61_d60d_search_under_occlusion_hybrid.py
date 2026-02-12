# 61_d60d_search_under_occlusion_hybrid.py
#
# D60d — Search behavior under occlusion (HYBRID)
#   PURSUIT when visible
#   SEEK last-belief when recently lost / low uncertainty
#   ORBIT around belief when uncertainty grows
#   SPIRAL search when uncertainty is large / prolonged occlusion
#
# Based on D59 (S1–S7 visible, 2 Hz) with an explicit search-mode planner.
#
# Run:
#   python d60d_search_under_occlusion_hybrid.py --steps 300 --out_gif d60d.gif
#
# Examples:
#   python d60d_search_under_occlusion_hybrid.py --steps 500 --obstacle 14,12,24,16 --fov_range 25 --out_gif d60d_occlusion.gif
#   python d60d_search_under_occlusion_hybrid.py --no_occlusion --steps 200 --out_gif d60d_no_occ.gif

import argparse, math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation, PillowWriter


# -----------------------------
# Helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def norm2(v):
    return float(np.sqrt(v[0]*v[0] + v[1]*v[1]))

def angle_wrap(a):
    # wrap to [-pi, pi]
    while a > math.pi:
        a -= 2*math.pi
    while a < -math.pi:
        a += 2*math.pi
    return a

def fmt2(v):
    return f"({v[0]:+5.1f},{v[1]:+5.1f})"

def in_fov(robot_xy, robot_theta, target_xy, fov_range, fov_half_angle_rad):
    dx = float(target_xy[0] - robot_xy[0])
    dy = float(target_xy[1] - robot_xy[1])
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > fov_range:
        return False
    ang = math.atan2(dy, dx)
    dth = angle_wrap(ang - robot_theta)
    return abs(dth) <= fov_half_angle_rad

def segment_intersects_rect(p0, p1, rect):
    """
    p0,p1: (x,y) endpoints
    rect: (x1,y1,x2,y2) axis-aligned
    Liang–Barsky clipping intersection test.
    """
    x1,y1,x2,y2 = rect
    x0,y0 = float(p0[0]), float(p0[1])
    x3,y3 = float(p1[0]), float(p1[1])
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
      innov = meas - mu_pred
      w = exp(-(d^2)/(2*(gate_k*sigma_pred)^2))  in [0,1]
      K = P / (P + R), where P=sigma_pred^2, R=sigma_meas^2
      mu = mu_pred + w*K*innov
      sigma_post^2 = (1 - w*K) * sigma_pred^2
    """
    innov = meas - mu_pred
    d = norm2(innov)

    denom = max(1e-6, (gate_k * sigma_pred))
    w = math.exp(-(d*d) / (2.0 * denom*denom))

    P = sigma_pred*sigma_pred
    R = sigma_meas*sigma_meas
    K = P / (P + R + 1e-9)

    mu = mu_pred + (w * K) * innov
    sigma_post = math.sqrt(max(1e-9, (1.0 - w*K) * P))
    return mu, sigma_post, innov, d, w, K


# -----------------------------
# Robot dynamics
# -----------------------------
def step_robot(robot_xy, robot_theta, v, w, dt, world_size):
    # unicycle model
    theta = robot_theta + w * dt
    x = float(robot_xy[0] + v * math.cos(theta) * dt)
    y = float(robot_xy[1] + v * math.sin(theta) * dt)
    x = clamp(x, 0.0, world_size)
    y = clamp(y, 0.0, world_size)
    return np.array([x, y], np.float32), theta


# -----------------------------
# Target motion (unknown to robot)
# -----------------------------
def step_target(t, target_xy, dt, world_size):
    # smooth wandering target: sinusoidal + drift
    x = float(target_xy[0] + 0.8*math.cos(0.07*t) * dt + 0.15*dt)
    y = float(target_xy[1] + 0.8*math.sin(0.05*t) * dt + 0.05*dt)
    x = clamp(x, 0.0, world_size)
    y = clamp(y, 0.0, world_size)
    return np.array([x, y], np.float32)


# -----------------------------
# D60d Hybrid Search Planner
# -----------------------------
def _heading_to(robot_xy, robot_theta, goal_xy):
    dx = float(goal_xy[0] - robot_xy[0])
    dy = float(goal_xy[1] - robot_xy[1])
    desired = math.atan2(dy, dx)
    heading_err = angle_wrap(desired - robot_theta)
    dist = math.sqrt(dx*dx + dy*dy)
    return dist, heading_err

def _pursuit_action(robot_xy, robot_theta, mu, sigma, cfg):
    # D59-style pursuit: mean mu is aimpoint; sigma modulates speed via confidence
    dist, heading_err = _heading_to(robot_xy, robot_theta, mu)
    w = clamp(cfg.k_turn * heading_err, -cfg.w_max, cfg.w_max)
    conf = 1.0 / (1.0 + (sigma / cfg.sigma_conf_scale)**2)
    v_nom = clamp(cfg.k_v * dist, 0.0, cfg.v_max)
    turn_slow = clamp(1.0 - abs(heading_err)/cfg.slow_turn_angle, 0.0, 1.0)
    v = v_nom * max(conf, cfg.conf_floor) * turn_slow
    return v, w, conf, dist, heading_err

def _seek_action(robot_xy, robot_theta, mu, sigma, cfg):
    # seek-last-belief: go toward mu, with stronger minimum speed
    dist, heading_err = _heading_to(robot_xy, robot_theta, mu)
    w = clamp(cfg.k_turn * heading_err, -cfg.w_max, cfg.w_max)
    conf = 1.0 / (1.0 + (sigma / cfg.sigma_conf_scale)**2)
    v_nom = clamp(cfg.k_v * dist, 0.0, cfg.v_max)
    turn_slow = clamp(1.0 - abs(heading_err)/cfg.slow_turn_angle, 0.0, 1.0)
    v = v_nom * max(conf, cfg.seek_conf_floor) * turn_slow
    return v, w, conf, dist, heading_err

def _orbit_action(robot_xy, robot_theta, mu, sigma, cfg, orbit_phase):
    """
    Orbit around belief center mu at radius r_orbit.
    We create a moving subgoal on the orbit and drive toward it.
    """
    r = cfg.orbit_radius
    # orbit speed increases a bit with uncertainty (more aggressive reacquisition)
    omega = cfg.orbit_omega_base + cfg.orbit_omega_gain * clamp((sigma - cfg.sigma_orbit) / max(1e-6, cfg.sigma_spiral - cfg.sigma_orbit), 0.0, 1.0)
    ang = orbit_phase
    subgoal = np.array([mu[0] + r * math.cos(ang),
                        mu[1] + r * math.sin(ang)], np.float32)

    dist, heading_err = _heading_to(robot_xy, robot_theta, subgoal)
    w = clamp(cfg.k_turn * heading_err, -cfg.w_max, cfg.w_max)

    # keep motion even if conf is low
    conf = 1.0 / (1.0 + (sigma / cfg.sigma_conf_scale)**2)
    v_nom = clamp(cfg.k_v * dist, 0.0, cfg.v_max)
    v = v_nom * max(conf, cfg.search_conf_floor) * clamp(1.0 - abs(heading_err)/cfg.slow_turn_angle, 0.0, 1.0)

    # advance phase (caller keeps it)
    orbit_phase_next = orbit_phase + omega * cfg.dt
    return v, w, conf, dist, heading_err, subgoal, orbit_phase_next

def _spiral_action(robot_xy, robot_theta, mu, sigma, cfg, spiral_phase, spiral_radius):
    """
    Expanding spiral around belief center:
      radius grows slowly; subgoal moves along spiral.
    """
    # spiral parameters
    omega = cfg.spiral_omega
    dr = cfg.spiral_dr_per_sec * cfg.dt
    spiral_radius = clamp(spiral_radius + dr, cfg.spiral_r_min, cfg.spiral_r_max)

    ang = spiral_phase
    subgoal = np.array([mu[0] + spiral_radius * math.cos(ang),
                        mu[1] + spiral_radius * math.sin(ang)], np.float32)

    dist, heading_err = _heading_to(robot_xy, robot_theta, subgoal)
    w = clamp(cfg.k_turn * heading_err, -cfg.w_max, cfg.w_max)

    conf = 1.0 / (1.0 + (sigma / cfg.sigma_conf_scale)**2)
    v_nom = clamp(cfg.k_v * dist, 0.0, cfg.v_max)
    v = v_nom * max(conf, cfg.search_conf_floor) * clamp(1.0 - abs(heading_err)/cfg.slow_turn_angle, 0.0, 1.0)

    spiral_phase_next = spiral_phase + omega * cfg.dt
    return v, w, conf, dist, heading_err, subgoal, spiral_phase_next, spiral_radius

def plan_action_hybrid(robot_xy, robot_theta, mu_pred, sigma_pred, visible, lost_count, cfg,
                       orbit_phase, spiral_phase, spiral_radius):
    """
    Returns:
      (v,w, conf, dist, heading_err, mode, subgoal, orbit_phase, spiral_phase, spiral_radius)
    """
    # Mode selection:
    #   - visible: PURSUIT
    #   - not visible:
    #       * if sigma small or recently lost: SEEK
    #       * if sigma medium: ORBIT
    #       * if sigma large or lost long: SPIRAL
    if visible:
        v, w, conf, dist, heading_err = _pursuit_action(robot_xy, robot_theta, mu_pred, sigma_pred, cfg)
        return v, w, conf, dist, heading_err, "PURSUIT", mu_pred.copy(), orbit_phase, spiral_phase, spiral_radius

    # not visible
    if (sigma_pred < cfg.sigma_orbit) or (lost_count < cfg.lost_seek_steps):
        v, w, conf, dist, heading_err = _seek_action(robot_xy, robot_theta, mu_pred, sigma_pred, cfg)
        return v, w, conf, dist, heading_err, "SEEK", mu_pred.copy(), orbit_phase, spiral_phase, spiral_radius

    if (sigma_pred < cfg.sigma_spiral) and (lost_count < cfg.lost_spiral_steps):
        v, w, conf, dist, heading_err, subgoal, orbit_phase_next = _orbit_action(
            robot_xy, robot_theta, mu_pred, sigma_pred, cfg, orbit_phase
        )
        return v, w, conf, dist, heading_err, "ORBIT", subgoal, orbit_phase_next, spiral_phase, spiral_radius

    v, w, conf, dist, heading_err, subgoal, spiral_phase_next, spiral_radius_next = _spiral_action(
        robot_xy, robot_theta, mu_pred, sigma_pred, cfg, spiral_phase, spiral_radius
    )
    return v, w, conf, dist, heading_err, "SPIRAL", subgoal, orbit_phase, spiral_phase_next, spiral_radius_next


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--dt", type=float, default=0.5)  # 2 Hz
    ap.add_argument("--world_size", type=float, default=30.0)

    ap.add_argument("--fov_range", type=float, default=25.0)
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

    # D60d: speed floors
    ap.add_argument("--conf_floor", type=float, default=0.35)        # pursuit minimum confidence
    ap.add_argument("--seek_conf_floor", type=float, default=0.55)   # more aggressive seek
    ap.add_argument("--search_conf_floor", type=float, default=0.75) # keep moving during search

    # D60d: mode thresholds
    ap.add_argument("--sigma_orbit", type=float, default=2.0)        # below => SEEK (if not visible)
    ap.add_argument("--sigma_spiral", type=float, default=5.0)       # above => SPIRAL
    ap.add_argument("--lost_seek_steps", type=int, default=3)        # recently lost => SEEK even if sigma slightly higher
    ap.add_argument("--lost_spiral_steps", type=int, default=18)     # long lost => SPIRAL even if sigma moderate

    # ORBIT params
    ap.add_argument("--orbit_radius", type=float, default=3.0)
    ap.add_argument("--orbit_omega_base", type=float, default=1.3)   # rad/s
    ap.add_argument("--orbit_omega_gain", type=float, default=1.0)   # add when sigma grows

    # SPIRAL params
    ap.add_argument("--spiral_r_min", type=float, default=2.0)
    ap.add_argument("--spiral_r_max", type=float, default=10.0)
    ap.add_argument("--spiral_dr_per_sec", type=float, default=0.55) # radius growth per second
    ap.add_argument("--spiral_omega", type=float, default=1.5)       # rad/s

    ap.add_argument("--obstacle", type=str, default="14,17,24,22")
    ap.add_argument("--no_occlusion", action="store_true")
    ap.add_argument("--out_gif", type=str, default="d60d_search.gif")
    ap.add_argument("--seed", type=int, default=0)

    cfg = ap.parse_args()
    np.random.seed(cfg.seed)

    # obstacle
    ox1, oy1, ox2, oy2 = [float(x) for x in cfg.obstacle.split(",")]
    obstacle = (ox1, oy1, ox2, oy2)

    # init world state
    robot_xy = np.array([4.0, 4.0], np.float32)
    robot_theta = math.radians(25.0)
    target_xy = np.array([24.0, 22.0], np.float32)

    # belief about target
    mu = np.array([8.0, 8.0], np.float32)
    sigma = 6.0

    # search state
    lost_count = 0
    orbit_phase = 0.0
    spiral_phase = 0.0
    spiral_radius = cfg.spiral_r_min

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
        "K": [],
        "pos_err": [],
        "mode": [],
        "subgoal": [],
        "lost": [],
    }

    fov_half = math.radians(cfg.fov_half_angle_deg)

    for t in range(cfg.steps):
        # --- world advances ---
        target_xy = step_target(t, target_xy, cfg.dt, cfg.world_size)

        # -------------------------
        # S3) Predict belief (prior)
        # -------------------------
        mu_pred, sigma_pred = predict_belief(mu, sigma, cfg.sigma_process)

        # -------------------------
        # S1) Observe (sensor) + visibility
        # -------------------------
        vis = in_fov(robot_xy, robot_theta, target_xy, cfg.fov_range, fov_half)
        if vis and (not cfg.no_occlusion):
            if segment_intersects_rect(robot_xy, target_xy, obstacle):
                vis = False

        meas = None
        if vis:
            noise = cfg.meas_noise_std * np.random.randn(2).astype(np.float32)
            meas = target_xy + noise

        x_t = meas  # observation (None if not visible)

        # -------------------------
        # S2) Encode (latent)
        # -------------------------
        z_t = x_t  # placeholder (later: JEPA encoder output)

        # lost-count bookkeeping
        if vis:
            lost_count = 0
            # reset spiral radius when reacquired (keeps demo readable)
            spiral_radius = cfg.spiral_r_min
        else:
            lost_count += 1

        # -------------------------
        # S4) Plan action from predicted belief (D60d hybrid)
        # -------------------------
        v, w, conf, dist, heading_err, mode, subgoal, orbit_phase, spiral_phase, spiral_radius = plan_action_hybrid(
            robot_xy, robot_theta,
            mu_pred, sigma_pred,
            visible=vis,
            lost_count=lost_count,
            cfg=cfg,
            orbit_phase=orbit_phase,
            spiral_phase=spiral_phase,
            spiral_radius=spiral_radius
        )
        a_t = (v, w)

        # -------------------------
        # S5) Predict next belief (for visibility / teaching)
        # -------------------------
        mu_hat_next, sigma_hat_next = predict_belief(mu_pred, sigma_pred, cfg.sigma_process)

        # -------------------------
        # S6) Act (execute)
        # -------------------------
        robot_xy, robot_theta = step_robot(robot_xy, robot_theta, v, w, cfg.dt, cfg.world_size)

        # -------------------------
        # S7) Correct (fuse measurement)
        # -------------------------
        innov_d = 0.0
        gate_w = 0.0
        K = 0.0

        if z_t is not None:
            mu, sigma, innov, innov_d, gate_w, K = fuse_belief(
                mu_pred, sigma_pred, z_t, cfg.sigma_meas, cfg.gate_k
            )
        else:
            mu, sigma = mu_pred, sigma_pred

        # logs
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
        hist["K"].append(float(K))
        hist["pos_err"].append(float(norm2(mu - target_xy)))
        hist["mode"].append(mode)
        hist["subgoal"].append(subgoal.copy())
        hist["lost"].append(int(lost_count))

        # S1–S7 visible log (every step) + mode
        print(
            f"[D60d][t={t:03d}] "
            f"S1 x={('None' if x_t is None else fmt2(x_t))} vis={int(vis)} lost={lost_count:02d} | "
            f"S2 z={('None' if z_t is None else fmt2(z_t))} | "
            f"S3 Bpred=(mu={fmt2(mu_pred)}, s={sigma_pred:4.2f}) | "
            f"S4 mode={mode:7s} a=(v={v:4.2f}, w={w:4.2f}) conf={conf:4.2f} subg={fmt2(subgoal)} | "
            f"S5 Bhat=(mu={fmt2(mu_hat_next)}, s={sigma_hat_next:4.2f}) | "
            f"S7 upd: d={innov_d:4.2f} gate={gate_w:4.2f} K={K:4.2f} -> "
            f"B=(mu={fmt2(mu)}, s={sigma:4.2f})"
        )

    # -----------------------------
    # Visualization -> GIF
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, cfg.world_size)
    ax.set_ylim(0, cfg.world_size)
    ax.set_aspect("equal")
    ax.set_title("D60d — Hybrid Search under Occlusion (2 Hz)")

    rect = Rectangle((ox1, oy1), ox2 - ox1, oy2 - oy1, fill=False, linewidth=2)
    ax.add_patch(rect)

    robot_dot = Circle((0, 0), radius=0.6, fill=False, linewidth=2)
    ax.add_patch(robot_dot)
    robot_heading, = ax.plot([], [], linewidth=2)

    target_dot = Circle((0, 0), radius=0.5, fill=False, linewidth=2)
    ax.add_patch(target_dot)

    mu_dot = Circle((0, 0), radius=0.5, fill=False, linewidth=2)
    ax.add_patch(mu_dot)

    sigma_circle = Circle((0, 0), radius=1.0, fill=False, linewidth=1, linestyle="--")
    ax.add_patch(sigma_circle)

    meas_dot, = ax.plot([], [], marker="x", linestyle="")
    subgoal_dot, = ax.plot([], [], marker="o", linestyle="")

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
        mode = hist["mode"][frame]
        lost = hist["lost"][frame]
        subg = hist["subgoal"][frame]

        robot_dot.center = (r[0], r[1])

        # approximate heading from finite diff (robust)
        if frame < len(hist["robot"]) - 1:
            r2 = hist["robot"][frame + 1]
        else:
            r2 = r
        th = math.atan2(float(r2[1] - r[1]), float(r2[0] - r[0]))
        hx = float(r[0] + 1.2 * math.cos(th))
        hy = float(r[1] + 1.2 * math.sin(th))
        robot_heading.set_data([r[0], hx], [r[1], hy])

        target_dot.center = (tgt[0], tgt[1])
        mu_dot.center = (mu_[0], mu_[1])

        sigma_circle.center = (mu_[0], mu_[1])
        sigma_circle.set_radius(max(0.3, min(10.0, sig)))

        if meas is not None:
            meas_dot.set_data([meas[0]], [meas[1]])
        else:
            meas_dot.set_data([], [])

        # show planned subgoal (especially useful during ORBIT/SPIRAL)
        subgoal_dot.set_data([subg[0]], [subg[1]])

        text.set_text(
            f"t={frame:03d}  mode={mode}  vis={int(vis)}  lost={lost}\n"
            f"sigma={sig:4.2f}  conf={conf:4.2f}\n"
            f"v={v:4.2f}  w={w:4.2f}  |mu-true|={err:4.2f}"
        )
        return (robot_dot, robot_heading, target_dot, mu_dot, sigma_circle,
                meas_dot, subgoal_dot, text)

    anim = FuncAnimation(fig, update, frames=cfg.steps, interval=500, blit=True)
    print(f"[D60d] Saving GIF -> {cfg.out_gif}")
    anim.save(cfg.out_gif, writer=PillowWriter(fps=2))
    print("[D60d] Done.")


if __name__ == "__main__":
    main()