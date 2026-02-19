# 69_D68_clean_integration_min.py 

# D68 — Clean integration (MIN, no GIF)
# One "best" integrated deterministic demo with clean prints (NO GIF yet).
# Combines: occlusion + reacquire sweep + belief search + belief+velocity + mode arbitration + robustness.

##1 imports
import argparse
import numpy as np
import math

##2 deterministic seed
def set_seed(seed: int):
    np.random.seed(seed)

##3 args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_every", type=int, default=20)

    # time / dynamics
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--friction", type=float, default=0.985)
    p.add_argument("--accel_std", type=float, default=0.9)

    # measurement
    p.add_argument("--meas_noise_std", type=float, default=0.25)
    p.add_argument("--meas_gain", type=float, default=0.55)

    # robot / FOV
    p.add_argument("--robot_x", type=float, default=0.0)
    p.add_argument("--robot_y", type=float, default=0.0)
    p.add_argument("--robot_theta_deg", type=float, default=15.0)
    p.add_argument("--fov_range", type=float, default=14.0)
    p.add_argument("--fov_half_angle_deg", type=float, default=50.0)

    # obstacle
    # p.add_argument("--obstacle", type=float, nargs=4, default=[-1.0, -1.0, 5.5, 4.0])
    #########################################
    p.add_argument("--obstacle", type=float, nargs=4, default=[2.0, -3.0, 7.0, 1.0])

    # mode arbitration
    p.add_argument("--gate_good", type=float, default=1.2)
    p.add_argument("--gate_bad", type=float, default=6.0)
    p.add_argument("--reacquire_boost", type=float, default=2.0)

    # reacquire sweep (D62-style)
    p.add_argument("--sweep_amp_deg", type=float, default=120.0)
    p.add_argument("--sweep_period", type=int, default=100)

    # belief search (D63-style): when occluded too long, jump belief
    p.add_argument("--search_after", type=int, default=35)      # steps occluded before search triggers
    p.add_argument("--search_radius", type=float, default=2.2)  # how far to jump belief
    p.add_argument("--search_jitter", type=float, default=0.3)  # small deterministic jitter scale

    # stability
    p.add_argument("--innov_clip", type=float, default=5.0)

    return p.parse_args()

##4 helpers
def l2(xy):
    return float(np.sqrt(np.sum(xy * xy)))

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def deg2rad(d):
    return d * math.pi / 180.0

##4b clip vector norm
def clip_norm(v, max_norm):
    n = l2(v)
    if n <= max_norm:
        return v
    return (v * (max_norm / max(n, 1e-8))).astype(np.float32)

##5 geometry: FOV + occlusion
def in_fov(robot_xy, robot_theta, target_xy, fov_range, fov_half_angle):
    rel = target_xy - robot_xy
    dist = l2(rel)
    if dist > fov_range:
        return False
    ang = math.atan2(rel[1], rel[0])
    d = (ang - robot_theta + math.pi) % (2 * math.pi) - math.pi
    return abs(d) <= fov_half_angle

def seg_intersects_aabb(p0, p1, aabb):
    xmin, ymin, xmax, ymax = aabb
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    t0, t1 = 0.0, 1.0

    def clip(p, q, t0, t1):
        if p == 0.0:
            if q < 0.0:
                return None
            return (t0, t1)
        r = q / p
        if p < 0.0:
            if r > t1: return None
            if r > t0: t0 = r
        else:
            if r < t0: return None
            if r < t1: t1 = r
        return (t0, t1)

    out = clip(-dx, p0[0] - xmin, t0, t1)
    if out is None: return False
    t0, t1 = out
    out = clip(dx, xmax - p0[0], t0, t1)
    if out is None: return False
    t0, t1 = out
    out = clip(-dy, p0[1] - ymin, t0, t1)
    if out is None: return False
    t0, t1 = out
    out = clip(dy, ymax - p0[1], t0, t1)
    if out is None: return False
    t0, t1 = out
    return t1 >= t0

def visible(robot_xy, robot_theta, target_xy, fov_range, fov_half_angle, obstacle_aabb):
    if not in_fov(robot_xy, robot_theta, target_xy, fov_range, fov_half_angle):
        return False
    if seg_intersects_aabb(robot_xy, target_xy, obstacle_aabb):
        return False
    return True

##5b sweep heading on occlusion (D62-style)
def sweep_theta(theta0, t, occ_run, amp_deg=120.0, period=100):
    amp = deg2rad(amp_deg)
    phase = 2.0 * math.pi * (t % period) / float(period)
    return theta0 + amp * math.sin(phase)

##6 env (target dynamics)
def env_step(x, v, dt, friction, accel_std):
    a = np.random.randn(2).astype(np.float32) * accel_std
    v_next = friction * v + a * dt
    x_next = x + v_next * dt
    return x_next, v_next

##7 measurement
def measure_pos(x_true, meas_noise_std):
    n = np.random.randn(2).astype(np.float32) * meas_noise_std
    return x_true + n

##8 belief predict/update
def belief_predict(mu_x, mu_v, dt, friction):
    mu_vp = friction * mu_v
    mu_xp = mu_x + mu_vp * dt
    return mu_xp, mu_vp

def choose_mode(innov_norm, gate_good, gate_bad, reacquire_boost):
    # 0=MEAS  1=BLND  2=PRED  3=RCQ
    if innov_norm < gate_good:
        return 0
    if innov_norm < gate_bad:
        return 1
    if innov_norm > gate_bad * reacquire_boost:
        return 3
    return 2

##8b apply correction with innovation clip
def apply_correction(mu_xp, mu_vp, z, meas_gain, dt, mode, innov_clip):
    innov = (z - mu_xp).astype(np.float32)
    innov = clip_norm(innov, innov_clip)

    if mode == 2:
        return mu_xp, mu_vp, innov, 0.0

    if mode == 3:
        g = clamp(2.0 * meas_gain, 0.0, 1.0)
    elif mode == 1:
        g = 0.5 * meas_gain
    else:
        g = meas_gain

    mu_x = mu_xp + g * innov
    mu_v = mu_vp + (g * innov) / max(dt, 1e-6)
    return mu_x, mu_v, innov, g

##9 belief search (D63-style)
def belief_search(mu_x, mu_v, radius, jitter):
    vnorm = l2(mu_v)
    if vnorm < 1e-6:
        d = np.array([1.0, 0.0], np.float32)
    else:
        d = (mu_v / vnorm).astype(np.float32)
    j = (np.random.randn(2).astype(np.float32) * jitter)
    mu_x_new = mu_x + radius * d + j
    return mu_x_new, mu_v

##10 main
def main():
    args = parse_args()
    set_seed(args.seed)

    dt = args.dt
    robot_xy = np.array([args.robot_x, args.robot_y], np.float32)
    robot_theta0 = deg2rad(args.robot_theta_deg)
    fov_half_angle = deg2rad(args.fov_half_angle_deg)
    obstacle = [float(x) for x in args.obstacle]

    # true target init
    x = np.array([10.0, 2.0], np.float32)
    v = np.array([-0.8, 0.3], np.float32)

    # belief init
    mu_x = np.array([0.0, 0.0], np.float32)
    mu_v = np.array([0.0, 0.0], np.float32)

    mode_names = {0: "MEAS", 1: "BLND", 2: "PRED", 3: "RCQ "}
    occ_run = 0
    seen = 0
    occ = 0
    search_count = 0

    print("[D68] Clean integration (MIN, no GIF)")
    print(f"  steps={args.steps} seed={args.seed} dt={args.dt} friction={args.friction}")
    print(f"  accel_std={args.accel_std} meas_noise_std={args.meas_noise_std} meas_gain={args.meas_gain} innov_clip={args.innov_clip}")
    print(f"  fov_range={args.fov_range} fov_half_angle_deg={args.fov_half_angle_deg} robot_theta_deg={args.robot_theta_deg}")
    print(f"  obstacle={obstacle}")
    print(f"  gate_good/bad={args.gate_good}/{args.gate_bad} reacquire_boost={args.reacquire_boost}")
    print(f"  sweep_amp_deg={args.sweep_amp_deg} sweep_period={args.sweep_period}")
    print(f"  search_after={args.search_after} search_radius={args.search_radius} search_jitter={args.search_jitter}")

    for t in range(args.steps):
        ##48 true step
        x, v = env_step(x, v, dt, args.friction, args.accel_std)

        ##49 predict
        mu_xp, mu_vp = belief_predict(mu_x, mu_v, dt, args.friction)

        ##50 visibility (+ sweep while occluded)
        theta_use = robot_theta0 if occ_run == 0 else sweep_theta(robot_theta0, t, occ_run, args.sweep_amp_deg, args.sweep_period)
        vis = visible(robot_xy, theta_use, x, args.fov_range, fov_half_angle, obstacle)

        if not vis:
            ##51a occluded: predict only + maybe search
            occ += 1
            occ_run += 1
            mu_x, mu_v = mu_xp, mu_vp
            mode = 2
            innov_norm = float("nan")
            g = 0.0
            meas_err = float("nan")

            if occ_run == args.search_after:
                ##51a1 search trigger
                mu_x, mu_v = belief_search(mu_x, mu_v, args.search_radius, args.search_jitter)
                search_count += 1
        else:
            ##51b visible: reset occlusion run, measure, correct
            occ_run = 0
            seen += 1

            z = measure_pos(x, args.meas_noise_std)
            innov_raw = z - mu_xp
            innov_norm = l2(innov_raw)
            mode = choose_mode(innov_norm, args.gate_good, args.gate_bad, args.reacquire_boost)

            ##53 correct
            mu_x, mu_v, _, g = apply_correction(mu_xp, mu_vp, z, args.meas_gain, dt, mode, args.innov_clip)
            meas_err = l2(z - x)

        ##54 log
        if (t % args.print_every) == 0:
            pos_err = l2(mu_x - x)
            sweep = 1 if (occ_run > 0) else 0
            print(
                f"[t={t:04d}] vis={1 if vis else 0} sweep={sweep} mode={mode_names[mode]} "
                f"|pos_err|={pos_err:.3f} "
                f"|innov|={(innov_norm if vis else -1):.3f} "
                f"g={g:.2f} "
                f"|meas_err|={(meas_err if vis else -1):.3f} "
                f"seen={seen} occ={occ} search={search_count}"
            )

if __name__ == "__main__":
    main()
