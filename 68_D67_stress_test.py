# 68_D67_stress_test.py

# D67 — Stress test
# Harder settings: tighter FOV, faster target, larger obstacle, more noise.
# Minimal: no GIF, no embellishments, deterministic.

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

    # target motion (stress: faster / jerkier)
    p.add_argument("--accel_std", type=float, default=1.2)

    # measurement (stress: noisier)
    p.add_argument("--meas_noise_std", type=float, default=0.35)

    # FOV (stress: tighter)
    p.add_argument("--fov_range", type=float, default=14.0)          # tighter than before
    p.add_argument("--fov_half_angle_deg", type=float, default=35.0) # tighter than before

    # obstacle (stress: larger)
    # rectangle in world coords: [xmin, ymin, xmax, ymax]
    p.add_argument("--obstacle", type=float, nargs=4, default=[1.0, -4.0, 7.0, -0.5])

    # sensor occlusion: if line segment robot->target intersects obstacle => occluded
    p.add_argument("--robot_x", type=float, default=0.0)
    p.add_argument("--robot_y", type=float, default=0.0)
    p.add_argument("--robot_theta_deg", type=float, default=20.0)  # fixed heading (stress)

    # mode arbitration thresholds (same *style* as D65/D66)
    p.add_argument("--gate_good", type=float, default=1.2)
    p.add_argument("--gate_bad", type=float, default=6.0)
    p.add_argument("--reacquire_boost", type=float, default=2.0)

    # correction gain baseline (keep simple & deterministic)
    p.add_argument("--meas_gain", type=float, default=0.55)  # baseline trust when "MEAS"

    ##3b D67 stability: clip innovation before applying correction
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
    # smallest signed angle diff
    d = (ang - robot_theta + math.pi) % (2 * math.pi) - math.pi
    return abs(d) <= fov_half_angle

def seg_intersects_aabb(p0, p1, aabb):
    # Liang–Barsky style param clip (minimal)
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

##5b sweep heading on occlusion
def sweep_theta(theta0, t, occ_run, amp_deg=170.0, period=120):
    """
    Deterministic sweep around theta0, only used when occluded.
    Wide sweep so reacquire is possible under tight FOV.
    """
    amp = deg2rad(amp_deg)
    phase = 2.0 * math.pi * (t % period) / float(period)
    return theta0 + amp * math.sin(phase)

##6 env (target dynamics)
def env_step(x, v, dt, friction, accel_std):
    a = np.random.randn(2).astype(np.float32) * accel_std
    v_next = friction * v + a * dt
    x_next = x + v_next * dt
    return x_next, v_next

##7 measure
def measure_pos(x_true, meas_noise_std):
    n = np.random.randn(2).astype(np.float32) * meas_noise_std
    return x_true + n

##8 belief predict/update (same style)
def belief_predict(mu_x, mu_v, dt, friction):
    mu_vp = friction * mu_v
    mu_xp = mu_x + mu_vp * dt
    return mu_xp, mu_vp

def choose_mode(innov_norm, gate_good, gate_bad, reacquire_boost):
    # 0=MEAS  1=BLEND  2=PRED  3=REACQ
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

    # D67 stability: clip innovation to prevent velocity blow-up on RCQ
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

##9 main
def main():
    args = parse_args()
    set_seed(args.seed)

    dt = args.dt
    robot_xy = np.array([args.robot_x, args.robot_y], np.float32)
    robot_theta = deg2rad(args.robot_theta_deg)
    fov_half_angle = deg2rad(args.fov_half_angle_deg)
    obstacle = [float(x) for x in args.obstacle]

    # true target
    x = np.array([8.0, 4.0], np.float32)
    v = np.array([-1.0, 0.4], np.float32)

    # belief
    mu_x = np.array([0.0, 0.0], np.float32)
    mu_v = np.array([0.0, 0.0], np.float32)

    mode_names = {0: "MEAS", 1: "BLND", 2: "PRED", 3: "RCQ "}

    print("[D67] Stress test: tight FOV + more noise + larger obstacle + faster target")
    print(f"  steps={args.steps} seed={args.seed} dt={args.dt} friction={args.friction}")
    print(f"  accel_std={args.accel_std} meas_noise_std={args.meas_noise_std} meas_gain={args.meas_gain}")
    print(f"  fov_range={args.fov_range} fov_half_angle_deg={args.fov_half_angle_deg}")
    print(f"  obstacle={obstacle}")
    print(f"  gate_good/bad={args.gate_good}/{args.gate_bad} reacquire_boost={args.reacquire_boost}")
    print(f"  innov_clip={args.innov_clip}")

    seen = 0
    occluded = 0

    ##9b occlusion streak (for sweep)
    occ_run = 0

    for t in range(args.steps):
        ##48 true step
        x, v = env_step(x, v, dt, args.friction, args.accel_std)

        ##49 predict
        mu_xp, mu_vp = belief_predict(mu_x, mu_v, dt, args.friction)

        ##50 visibility
        ##50b1 sweep heading only while occluded to attempt reacquire
        theta_use = robot_theta if occ_run == 0 else sweep_theta(robot_theta, t, occ_run)
        vis = visible(robot_xy, theta_use, x, args.fov_range, fov_half_angle, obstacle)

        if not vis:
            ##51a no measurement: predict only
            mu_x, mu_v = mu_xp, mu_vp
            occluded += 1
            occ_run += 1
            mode = 2
            innov_norm = float("nan")
            g = 0.0
            meas_err = float("nan")
        else:
            ##51b measure
            z = measure_pos(x, args.meas_noise_std)
            seen += 1
            occ_run = 0

            ##52 innovation + mode
            innov_raw = z - mu_xp
            innov_norm = l2(innov_raw)
            mode = choose_mode(innov_norm, args.gate_good, args.gate_bad, args.reacquire_boost)

            ##53 correct
            ##53b1 correction uses innovation clip internally
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
                f"seen={seen} occ={occluded}"
            )

if __name__ == "__main__":
    main()