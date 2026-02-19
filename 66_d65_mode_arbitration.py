# 66_d65_mode_arbitration.py
import math, random, argparse
import numpy as np

DT = 0.25  ##1

def wrap_pi(a):  ##2
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def clamp(x, lo, hi):  ##3
    return lo if x < lo else (hi if x > hi else x)

def in_fov(rx, ry, rth, tx, ty, rng=25.0, half_deg=70.0):  ##4
    dx, dy = tx - rx, ty - ry
    d = math.hypot(dx, dy)
    if d > rng:
        return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - rth)
    return abs(dth) <= math.radians(half_deg)

def main(
    steps=180,
    fov_range=25.0,
    fov_half_deg=70.0,
    seed=0,
    log_every=5,
    # arbitration knobs
    lost_to_sweep=12,      # after this many lost steps -> SWEEP no matter what
    s_conf=1.6,            # if uncertainty <= s_conf, we trust belief enough to GO2B
    # belief knobs
    meas_noise=0.15,
    alpha_pos=0.55,
    alpha_vel=0.35,
    s_shrink=0.85,
    s_grow=1.08,
    s_min=0.30,
    s_max=12.0,
):
    random.seed(seed)
    np.random.seed(seed)

    # -------------------------
    # World / robot init
    # -------------------------  ##5
    rx, ry, rth = 5.0, 5.0, 0.0

    # target: drifts right, can be occluded by a wall
    tx, ty = 12.0, 14.0
    tvx_true, tvy_true = 0.35, 0.0

    # wall occluder (vertical segment)
    wall_x = 14.0
    wall_y0, wall_y1 = 10.0, 20.0

    def occluded(rx, ry, tx, ty):  ##6
        if (rx - wall_x) * (tx - wall_x) >= 0:
            return False
        u = (wall_x - rx) / (tx - rx + 1e-9)
        y_cross = ry + u * (ty - ry)
        return (wall_y0 <= y_cross <= wall_y1)

    # -------------------------
    # Belief state
    # -------------------------  ##7
    Bx, By = tx, ty
    Bvx, Bvy = 0.0, 0.0
    s = 0.6

    have_prev_meas = False
    zpx, zpy = 0.0, 0.0
    lost = 0

    for t in range(steps):
        # true target motion  ##8
        tx += tvx_true * DT
        ty += tvy_true * DT

        vis = in_fov(rx, ry, rth, tx, ty, fov_range, fov_half_deg) and (not occluded(rx, ry, tx, ty))

        if vis:
            # =========================
            # MODE: TRACK
            # =========================  ##9
            lost = 0
            mode = "TRACK"

            # measurement  ##10
            zx = tx + np.random.randn() * meas_noise
            zy = ty + np.random.randn() * meas_noise

            # belief position update  ##11
            Bx = (1 - alpha_pos) * Bx + alpha_pos * zx
            By = (1 - alpha_pos) * By + alpha_pos * zy

            # belief velocity update (from consecutive measurements)  ##12
            if have_prev_meas:
                mvx = (zx - zpx) / DT
                mvy = (zy - zpy) / DT
                Bvx = (1 - alpha_vel) * Bvx + alpha_vel * mvx
                Bvy = (1 - alpha_vel) * Bvy + alpha_vel * mvy
            have_prev_meas = True
            zpx, zpy = zx, zy

            # uncertainty shrinks  ##13
            s = clamp(s * s_shrink, s_min, s_max)

            # control: steer to belief (after update)  ##14
            ang = math.atan2(By - ry, Bx - rx)
            dth = wrap_pi(ang - rth)
            w = clamp(1.5 * dth, -1.0, 1.0)
            v = 1.2

        else:
            # =========================
            # LOST: propagate belief
            # =========================  ##15
            lost += 1

            # propagate belief using estimated velocity  ##16
            Bx += Bvx * DT
            By += Bvy * DT
            s = clamp(s * s_grow, s_min, s_max)

            # damp + clamp velocity estimate (simple stabilizer)  ##17
            Bvx *= 0.98
            Bvy *= 0.98
            Bvx = clamp(Bvx, -1.0, 1.0)
            Bvy = clamp(Bvy, -1.0, 1.0)

            # =========================
            # MODE ARBITRATION
            # =========================  ##18
            # If belief still confident AND we haven't been lost too long -> GO2B.
            # Else -> SWEEP (search).
            if (lost <= lost_to_sweep) and (s <= s_conf):
                mode = "GO2B"
                ang = math.atan2(By - ry, Bx - rx)
                dth = wrap_pi(ang - rth)
                w = clamp(1.5 * dth, -1.0, 1.0)
                v = 1.0
            else:
                mode = "SWEEP"
                v = 0.9
                w = 0.5 if (lost // 20) % 2 == 0 else -0.5

        # robot motion  ##19
        rth = wrap_pi(rth + w * DT)
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT

        # log  ##20
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

if __name__ == "__main__":  ##21
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--fov_range", type=float, default=25.0)
    ap.add_argument("--fov_half_deg", type=float, default=70.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--lost_to_sweep", type=int, default=12)
    ap.add_argument("--s_conf", type=float, default=1.6)
    args = ap.parse_args()

    main(
        steps=args.steps,
        fov_range=args.fov_range,
        fov_half_deg=args.fov_half_deg,
        seed=args.seed,
        log_every=args.log_every,
        lost_to_sweep=args.lost_to_sweep,
        s_conf=args.s_conf,
    )
