# 71_D69_capstone_min_clean.py
# D69 — Capstone (MIN, no GIF)
# Full showcase scenario (occlusion + search + decision + stable loop) with crisp narrative log.

import argparse
import numpy as np
import math


# -----------------------------
# Determinism
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # CLEAN
    p.add_argument("--innov_ema_beta", type=float, default=0.90)  # 0=no smoothing
    p.add_argument("--mode_hold", type=int, default=6)            # steps to hold mode before allowing changes
    p.add_argument("--vis_cooldown", type=int, default=10)        # cooldown to prevent LOST/REACQUIRED chatter
    p.add_argument("--vel_clip", type=float, default=10.0)        # clip belief velocity to prevent runaway

    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_every", type=int, default=20)

    # time / dynamics
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--friction", type=float, default=0.985)
    p.add_argument("--accel_std", type=float, default=0.85)

    # measurement
    p.add_argument("--meas_noise_std", type=float, default=0.25)
    p.add_argument("--meas_gain", type=float, default=0.55)

    # robot / FOV
    p.add_argument("--robot_x", type=float, default=0.0)
    p.add_argument("--robot_y", type=float, default=0.0)
    p.add_argument("--robot_theta_deg", type=float, default=10.0)
    p.add_argument("--fov_range", type=float, default=14.0)
    p.add_argument("--fov_half_angle_deg", type=float, default=55.0)

    # obstacle (blocks sometimes, not always)
    p.add_argument("--obstacle", type=float, nargs=4, default=[2.0, -3.0, 7.0, 1.0])

    # mode arbitration
    p.add_argument("--gate_good", type=float, default=1.2)
    p.add_argument("--gate_bad", type=float, default=6.0)
    p.add_argument("--reacquire_boost", type=float, default=2.0)

    # reacquire sweep
    p.add_argument("--sweep_amp_deg", type=float, default=150.0)
    p.add_argument("--sweep_period", type=int, default=120)

    # belief search
    p.add_argument("--search_after", type=int, default=35)
    p.add_argument("--search_every", type=int, default=0)   # 0 disables periodic search
    p.add_argument("--search_radius", type=float, default=2.2)
    p.add_argument("--search_jitter", type=float, default=0.3)

    # stability
    p.add_argument("--innov_clip", type=float, default=5.0)

    # vis debounce + occlusion damping
    p.add_argument("--vis_on_n", type=int, default=2)          # require N consecutive raw_vis to turn ON
    p.add_argument("--vis_off_n", type=int, default=2)         # require N consecutive raw_no_vis to turn OFF
    p.add_argument("--occ_vel_damp", type=float, default=0.90) # extra damping on belief velocity while occluded

    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
def l2(xy):
    return float(np.sqrt(np.sum(xy * xy)))

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def deg2rad(d):
    return d * math.pi / 180.0

def clip_vec(v, max_norm):
    n = l2(v)
    if n <= max_norm:
        return v
    return (v * (max_norm / max(n, 1e-8))).astype(np.float32)

def clip_innov(v, max_norm):
    # identical to clip_vec but semantically clearer for innovation
    return clip_vec(v, max_norm)


# -----------------------------
# Geometry: FOV + occlusion
# -----------------------------
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

def sweep_theta(theta0, t, amp_deg=150.0, period=120):
    amp = deg2rad(amp_deg)
    phase = 2.0 * math.pi * (t % period) / float(period)
    return theta0 + amp * math.sin(phase)


# -----------------------------
# Env / Measurement
# -----------------------------
def env_step(x, v, dt, friction, accel_std):
    a = np.random.randn(2).astype(np.float32) * accel_std
    v_next = friction * v + a * dt
    x_next = x + v_next * dt
    return x_next, v_next

def measure_pos(x_true, meas_noise_std):
    n = np.random.randn(2).astype(np.float32) * meas_noise_std
    return x_true + n


# -----------------------------
# Belief predict / mode / correction
# -----------------------------
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

def apply_correction(mu_xp, mu_vp, z, meas_gain, dt, mode, innov_clip):
    innov = (z - mu_xp).astype(np.float32)
    innov = clip_innov(innov, innov_clip)

    # PRED: no correction
    if mode == 2:
        return mu_xp, mu_vp, innov, 0.0

    # choose gain
    if mode == 3:
        g = clamp(2.0 * meas_gain, 0.0, 1.0)   # RCQ = aggressive position snap
    elif mode == 1:
        g = 0.5 * meas_gain                    # BLND = gentle correction
    else:
        g = meas_gain                          # MEAS

    mu_x = mu_xp + g * innov

    if mode == 3:
        # RCQ: don't convert big innov into huge velocity
        mu_v = np.array([0.0, 0.0], np.float32)
    else:
        mu_v = mu_vp + (g * innov) / max(dt, 1e-6)

    return mu_x, mu_v, innov, g


# -----------------------------
# Search
# -----------------------------
def belief_search(mu_x, mu_v, radius, jitter):
    vnorm = l2(mu_v)
    if vnorm < 1e-6:
        d = np.array([1.0, 0.0], np.float32)
    else:
        d = (mu_v / vnorm).astype(np.float32)
    j = (np.random.randn(2).astype(np.float32) * jitter)
    mu_x_new = mu_x + radius * d + j
    return mu_x_new, mu_v


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    dt = args.dt
    robot_xy = np.array([args.robot_x, args.robot_y], np.float32)
    theta0 = deg2rad(args.robot_theta_deg)
    fov_half_angle = deg2rad(args.fov_half_angle_deg)
    obstacle = [float(x) for x in args.obstacle]

    # true target init (capstone: start visible-ish)
    x = np.array([10.0, 3.5], np.float32)
    v = np.array([-0.7, 0.25], np.float32)

    # belief init
    mu_x = np.array([0.0, 0.0], np.float32)
    mu_v = np.array([0.0, 0.0], np.float32)

    mode_names = {0: "MEAS", 1: "BLND", 2: "PRED", 3: "RCQ "}
    occ_run = 0
    seen = 0
    occ = 0
    search_count = 0

    # visibility debounce state
    vis_state = False
    vis_on_run = 0
    vis_off_run = 0
    vis_lock_until = -1  # cooldown lockout

    # mode smoothing/holding
    prev_mode = None
    innov_ema = None
    mode_hold_left = 0

    print("[D69] Capstone (MIN, no GIF): occlusion + sweep + search + arbitration + stability")
    print(f"  steps={args.steps} seed={args.seed} dt={args.dt} friction={args.friction}")
    print(f"  accel_std={args.accel_std} meas_noise_std={args.meas_noise_std} meas_gain={args.meas_gain} innov_clip={args.innov_clip}")
    print(f"  fov_range={args.fov_range} fov_half_angle_deg={args.fov_half_angle_deg} robot_theta_deg={args.robot_theta_deg}")
    print(f"  obstacle={obstacle}")
    print(f"  gate_good/bad={args.gate_good}/{args.gate_bad} reacquire_boost={args.reacquire_boost}")
    print(f"  sweep_amp_deg={args.sweep_amp_deg} sweep_period={args.sweep_period}")
    print(f"  search_after={args.search_after} search_every={args.search_every} search_radius={args.search_radius} search_jitter={args.search_jitter}")
    print(f"  vis_on_n/off_n={args.vis_on_n}/{args.vis_off_n} vis_cooldown={args.vis_cooldown} occ_vel_damp={args.occ_vel_damp}")
    print(f"  innov_ema_beta={args.innov_ema_beta} mode_hold={args.mode_hold} vel_clip={args.vel_clip}")

    for t in range(args.steps):
        # true step
        x, v = env_step(x, v, dt, args.friction, args.accel_std)

        # predict
        mu_xp, mu_vp = belief_predict(mu_x, mu_v, dt, args.friction)

        # visibility (+ sweep while occluded)
        theta_use = theta0 if occ_run == 0 else sweep_theta(theta0, t, args.sweep_amp_deg, args.sweep_period)
        raw_vis = visible(robot_xy, theta_use, x, args.fov_range, fov_half_angle, obstacle)

        # ---- Correct debounce+cooldown (single update per step) ----
        can_flip = (t >= vis_lock_until)

        if can_flip:
            if raw_vis:
                vis_on_run += 1
                vis_off_run = 0
            else:
                vis_off_run += 1
                vis_on_run = 0

            if (not vis_state) and (vis_on_run >= args.vis_on_n):
                vis_state = True
                vis_lock_until = t + args.vis_cooldown
                vis_on_run = 0
                vis_off_run = 0
                print(f"[t={t:04d}] EVENT: REACQUIRED (vis=1)")

            elif vis_state and (vis_off_run >= args.vis_off_n):
                vis_state = False
                vis_lock_until = t + args.vis_cooldown
                vis_on_run = 0
                vis_off_run = 0
                print(f"[t={t:04d}] EVENT: LOST (vis=0)")
        else:
            # during cooldown, don't let counters build up
            vis_on_run = 0
            vis_off_run = 0

        vis = vis_state

        if not vis:
            # occluded: predict only + maybe search
            occ += 1
            occ_run += 1

            mu_x = mu_xp
            mu_v = (args.occ_vel_damp * mu_vp).astype(np.float32)
            mu_v = clip_vec(mu_v, args.vel_clip)

            mode = 2
            innov_norm = float("nan")
            g = 0.0
            meas_err = float("nan")

            # one-shot search at search_after, plus optional periodic search thereafter
            do_search = (occ_run == args.search_after)
            if args.search_every > 0 and (occ_run > args.search_after) and (occ_run % args.search_every == 0):
                do_search = True

            if do_search:
                mu_x, mu_v = belief_search(mu_x, mu_v, args.search_radius, args.search_jitter)
                mu_v = np.array([0.0, 0.0], np.float32)
                search_count += 1
                print(f"[t={t:04d}] EVENT: SEARCH jump (count={search_count})")

        else:
            # visible: reset occlusion run, measure, correct
            occ_run = 0
            seen += 1

            z = measure_pos(x, args.meas_noise_std)

            innov_raw = z - mu_xp
            innov_norm = l2(innov_raw)

            # smooth innovation norm
            if innov_ema is None:
                innov_ema = innov_norm
            else:
                b = clamp(args.innov_ema_beta, 0.0, 0.999)
                innov_ema = b * innov_ema + (1.0 - b) * innov_norm

            # propose a mode based on smoothed innov
            mode_prop = choose_mode(innov_ema, args.gate_good, args.gate_bad, args.reacquire_boost)

            # hold mode for stability
            if prev_mode is None:
                prev_mode = mode_prop
                mode_hold_left = args.mode_hold

            if mode_hold_left > 0:
                mode = prev_mode
                mode_hold_left -= 1
            else:
                mode = mode_prop
                if mode != prev_mode:
                    print(f"[t={t:04d}] EVENT: mode -> {mode_names[mode]} (innov_ema={innov_ema:.2f})")
                    prev_mode = mode
                mode_hold_left = args.mode_hold

            mu_x, mu_v, _, g = apply_correction(mu_xp, mu_vp, z, args.meas_gain, dt, mode, args.innov_clip)
            mu_v = clip_vec(mu_v, args.vel_clip)
            meas_err = l2(z - x)

        # periodic log
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