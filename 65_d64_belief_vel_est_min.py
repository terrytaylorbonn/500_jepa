# 65_d64_belief_vel_est_min.py
import math, random, argparse
import numpy as np

DT = 0.25

def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def in_fov(rx, ry, rth, tx, ty, rng=25.0, half_deg=70.0):
    dx, dy = tx - rx, ty - ry
    d = math.hypot(dx, dy)
    if d > rng:
        return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - rth)
    return abs(dth) <= math.radians(half_deg)

def main(
    steps=160,
    fov_range=25.0,
    fov_half_deg=70.0,
    seed=0,
    log_every=2,
):
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------
    # Robot state
    # -------------------------
    rx, ry, rth = 5.0, 5.0, 0.0

    # -------------------------
    # True target motion (ground truth)
    # -------------------------
    # tx, ty = 20.0, 14.0
    # tvx_true, tvy_true = 0.2, 0.0
    tx, ty = 12.0, 14.0          # <--- was 20.0, 14.0 ################################
    tvx_true, tvy_true = 0.35, 0.0  # <--- was 0.2, 0.0

    # -------------------------
    # Occluder: vertical wall segment blocks line of sight
    # -------------------------
    wall_x = 14.0
    wall_y0, wall_y1 = 10.0, 20.0

    def occluded(rx, ry, tx, ty):
        if (rx - wall_x) * (tx - wall_x) >= 0:
            return False
        u = (wall_x - rx) / (tx - rx + 1e-9)
        y_cross = ry + u * (ty - ry)
        return (wall_y0 <= y_cross <= wall_y1)

    # -------------------------
    # Belief state (D64 core)
    # -------------------------
    Bx, By = tx, ty          # belief mean
    Bvx, Bvy = 0.0, 0.0      # estimated target velocity (NO CHEAT)
    s = 0.6                  # uncertainty (std-ish)

    # Measurement + filter knobs
    meas_noise = 0.15
    alpha_pos = 0.55         # trust measurement for position
    alpha_vel = 0.35         # trust measurement-derived velocity
    s_grow = 1.08
    s_max = 12.0
    s_shrink = 0.85          #############################
    s_min = 0.30             #########################

    # For velocity-from-measurements
    have_prev_meas = False
    zpx, zpy = 0.0, 0.0

    lost = 0

    for t in range(steps):
        # --- true target motion ---
        tx += tvx_true * DT
        ty += tvy_true * DT

        vis = in_fov(rx, ry, rth, tx, ty, fov_range, fov_half_deg) and (not occluded(rx, ry, tx, ty))

        if vis:
            lost = 0
            mode = "TRACK"

            # ----- measurement -----
            zx = tx + np.random.randn() * meas_noise
            zy = ty + np.random.randn() * meas_noise

            # ----- belief position update -----
            Bx = (1 - alpha_pos) * Bx + alpha_pos * zx
            By = (1 - alpha_pos) * By + alpha_pos * zy

            # ----- belief velocity update (from consecutive measurements) -----
            if have_prev_meas:
                mvx = (zx - zpx) / DT
                mvy = (zy - zpy) / DT
                Bvx = (1 - alpha_vel) * Bvx + alpha_vel * mvx
                Bvy = (1 - alpha_vel) * Bvy + alpha_vel * mvy
            have_prev_meas = True
            zpx, zpy = zx, zy

            s = clamp(s * s_shrink, s_min, s_max)

            # Robot control: turn toward current belief (after update)
            ang = math.atan2(By - ry, Bx - rx)
            dth = wrap_pi(ang - rth)
            w = clamp(1.5 * dth, -1.0, 1.0)
            v = 1.2

        else:
            lost += 1
            mode = "SWEEP"

            # When lost, we *propagate belief* using estimated velocity (not true velocity)
            Bx += Bvx * DT
            By += Bvy * DT
            s = clamp(s * s_grow, s_min, s_max)

            # Robot control: same simple reacquire sweep
            v = 0.9
            w = 0.5 if (lost // 20) % 2 == 0 else -0.5

            # damp + clamp velocity estimate (simple stabilizer)
            Bvx *= 0.98
            Bvy *= 0.98
            Bvx = clamp(Bvx, -1.0, 1.0)
            Bvy = clamp(Bvy, -1.0, 1.0)

        # --- robot motion ---
        rth = wrap_pi(rth + w * DT)
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT

        # --- compact output ---
        if t % log_every == 0:
            if vis:
                print(
                    f"t={t:03d} VIS  {mode:5s}  "
                    f"robot=({rx:5.2f},{ry:5.2f})  "
                    f"x=({tx:5.2f},{ty:5.2f})  "
                    f"B=({Bx:5.2f},{By:5.2f})  "
                    f"vB=({Bvx:+5.2f},{Bvy:+5.2f})  s={s:4.2f}  lost={lost:03d}"
                )
            else:
                print(
                    f"t={t:03d} LOST {mode:5s}  "
                    f"robot=({rx:5.2f},{ry:5.2f})  "
                    f"x=( None, None)  "
                    f"B=({Bx:5.2f},{By:5.2f})  "
                    f"vB=({Bvx:+5.2f},{Bvy:+5.2f})  s={s:4.2f}  lost={lost:03d}"
                )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--fov_range", type=float, default=25.0)
    ap.add_argument("--fov_half_deg", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=2)
    args = ap.parse_args()

    main(
        steps=args.steps,
        fov_range=args.fov_range,
        fov_half_deg=args.fov_half_deg,
        seed=args.seed,
        log_every=args.log_every,
    )