# 67_d66_robustness_sweep.py
import math, random, argparse  ##1
import numpy as np             ##2

DT = 0.25                      ##3

# -------------------------
# Helpers
# -------------------------
def wrap_pi(a):                ##4
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def clamp(x, lo, hi):          ##5
    return lo if x < lo else (hi if x > hi else x)

def in_fov(rx, ry, rth, tx, ty, rng=25.0, half_deg=70.0):  ##6
    dx, dy = tx - rx, ty - ry
    d = math.hypot(dx, dy)
    if d > rng:
        return False
    ang = math.atan2(dy, dx)
    dth = wrap_pi(ang - rth)
    return abs(dth) <= math.radians(half_deg)

# -------------------------
# Main
# -------------------------
def main(  ##7
    steps=220,
    seed=0,
    log_every=10,
    # sweep over these (robustness)
    fov_half_deg=70.0,
    meas_noise=0.15,
    tvx_true=0.35,
    wall_x=14.0,
    wall_y0=10.0,
    wall_y1=20.0,
    print_summary=True,          ##7a
):
    random.seed(seed)          ##8
    np.random.seed(seed)       ##9

    # -------------------------
    # Robot state
    # -------------------------
    rx, ry, rth = 5.0, 5.0, 0.0            ##10

    # -------------------------
    # True target motion
    # -------------------------
    tx, ty = 12.0, 14.0                    ##11
    tvy_true = 0.0                         ##12

    # -------------------------
    # Occlusion model
    # -------------------------
    def occluded(rx, ry, tx, ty):          ##13
        if (rx - wall_x) * (tx - wall_x) >= 0:
            return False
        u = (wall_x - rx) / (tx - rx + 1e-9)
        y_cross = ry + u * (ty - ry)
        return (wall_y0 <= y_cross <= wall_y1)

    # -------------------------
    # Belief state
    # -------------------------
    Bx, By = tx, ty                         ##14
    Bvx, Bvy = 0.0, 0.0                     ##15
    s = 0.6                                 ##16

    # Filter knobs (kept constant in this demo)
    alpha_pos = 0.55                        ##17
    alpha_vel = 0.35                        ##18
    s_shrink = 0.85                         ##19
    s_grow   = 1.08                         ##20
    s_min, s_max = 0.30, 12.0               ##21

    # Velocity stabilizer
    vel_damp = 0.98                         ##22
    vel_clip = 1.0                          ##23

    # Arbitration
    lost_to_sweep = 12                      ##24
    s_conf = 2.0                            ##25

    have_prev_meas = False                  ##26
    zpx, zpy = 0.0, 0.0

    lost = 0                                ##27

    # -------------------------
    # Quick robustness counters
    # -------------------------
    reacq_count = 0                          ##28
    max_lost = 0                             ##29

    for t in range(steps):                  ##30
        # --- truth motion ---
        tx += tvx_true * DT                 ##31
        ty += tvy_true * DT                 ##32

        vis = in_fov(rx, ry, rth, tx, ty, rng=25.0, half_deg=fov_half_deg) and (not occluded(rx, ry, tx, ty))  ##33

        if vis:                             ##34
            if lost > 0:
                reacq_count += 1            ##35
            lost = 0
            mode = "TRACK"

            # measurement
            zx = tx + np.random.randn() * meas_noise  ##36
            zy = ty + np.random.randn() * meas_noise

            # pos update
            Bx = (1 - alpha_pos) * Bx + alpha_pos * zx  ##37
            By = (1 - alpha_pos) * By + alpha_pos * zy

            # vel update (from meas diffs)
            if have_prev_meas:
                mvx = (zx - zpx) / DT
                mvy = (zy - zpy) / DT
                Bvx = (1 - alpha_vel) * Bvx + alpha_vel * mvx
                Bvy = (1 - alpha_vel) * Bvy + alpha_vel * mvy
            have_prev_meas = True
            zpx, zpy = zx, zy

            # shrink uncertainty
            s = clamp(s * s_shrink, s_min, s_max)      ##38

            # control: go toward belief
            ang = math.atan2(By - ry, Bx - rx)         ##39
            dth = wrap_pi(ang - rth)
            w = clamp(1.5 * dth, -1.0, 1.0)
            v = 1.2

        else:                                           ##40
            lost += 1
            max_lost = max(max_lost, lost)              ##41
            have_prev_meas = False

            # propagate belief
            Bx += Bvx * DT                               ##42
            By += Bvy * DT
            s  = clamp(s * s_grow, s_min, s_max)

            # damp + clamp vel estimate
            Bvx *= vel_damp                              ##43
            Bvy *= vel_damp
            Bvx = clamp(Bvx, -vel_clip, vel_clip)
            Bvy = clamp(Bvy, -vel_clip, vel_clip)

            # arbitration: GO2B while confident and recently lost; else SWEEP
            if (lost < lost_to_sweep) and (s <= s_conf):  ##44
                mode = "GO2B"
                ang = math.atan2(By - ry, Bx - rx)
                dth = wrap_pi(ang - rth)
                w = clamp(1.5 * dth, -1.0, 1.0)
                v = 1.0
            else:
                mode = "SWEEP"
                v = 0.9
                w = 0.5 if (lost // 20) % 2 == 0 else -0.5

        # robot motion
        rth = wrap_pi(rth + w * DT)                      ##45
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT

        # logs
        if (log_every > 0) and (t % log_every == 0):     ##46
            if vis:
                print(f"t={t:03d} VIS  {mode:5s}  robot=({rx:5.2f},{ry:5.2f})  x=({tx:5.2f},{ty:5.2f})  B=({Bx:5.2f},{By:5.2f})  s={s:4.2f}")
            else:
                print(f"t={t:03d} LOST {mode:5s}  robot=({rx:5.2f},{ry:5.2f})  B=({Bx:5.2f},{By:5.2f})  s={s:4.2f}  lost={lost:03d}")

    # summary
    if print_summary:                                    ##47a
        print("\n[SUMMARY]")                             ##47
        print(f"  fov_half_deg = {fov_half_deg}")
        print(f"  meas_noise   = {meas_noise}")
        print(f"  tvx_true     = {tvx_true}")
        print(f"  wall_x       = {wall_x}  wall_y=[{wall_y0},{wall_y1}]")
        print(f"  reacq_count  = {reacq_count}")
        print(f"  max_lost     = {max_lost}")

    return reacq_count, max_lost                          ##47b

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":                                ##48
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)

    ap.add_argument("--fov_half_deg", type=float, default=70.0)
    ap.add_argument("--meas_noise", type=float, default=0.15)
    ap.add_argument("--tvx_true", type=float, default=0.35)

    ap.add_argument("--wall_x", type=float, default=14.0)
    ap.add_argument("--wall_y0", type=float, default=10.0)
    ap.add_argument("--wall_y1", type=float, default=20.0)

    ap.add_argument("--sweep", action="store_true")       ##48a

    args = ap.parse_args()

    if args.sweep:                                        ##48b
        seeds = [0, 1, 2]                                 ##48c
        fovs = [55.0, 70.0, 85.0]                         ##48d
        noises = [0.10, 0.15, 0.30]                        ##48e

        print("[SWEEP] seed x fov_half_deg x meas_noise")  ##48f
        for sd in seeds:
            for fov in fovs:
                for nz in noises:
                    reacq, maxlost = main(
                        steps=args.steps,
                        seed=sd,
                        log_every=0,            # no per-step logs in sweep
                        fov_half_deg=fov,
                        meas_noise=nz,
                        tvx_true=args.tvx_true,
                        wall_x=args.wall_x,
                        wall_y0=args.wall_y0,
                        wall_y1=args.wall_y1,
                        print_summary=False,    # suppress the block summary
                    )
                    print(f"seed={sd}  fov={fov:5.1f}  noise={nz:4.2f}  reacq={reacq:2d}  max_lost={maxlost:3d}")
    else:
        main(
            steps=args.steps,
            seed=args.seed,
            log_every=args.log_every,
            fov_half_deg=args.fov_half_deg,
            meas_noise=args.meas_noise,
            tvx_true=args.tvx_true,
            wall_x=args.wall_x,
            wall_y0=args.wall_y0,
            wall_y1=args.wall_y1,
            print_summary=True,                 ##48g
        )


