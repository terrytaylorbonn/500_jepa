#56_d56C_multi_object_belief_uncertainty_viz.py
import argparse
import math
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np

##56cG0 — add imports (near the top, after numpy)
# ##56cG0 (GIF) optional dependency
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

# ##56cG2b (GIF text overlay)
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

# ##56cG2b (GIF text overlay)
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None



# =========================
# Helpers
# =========================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def segments_intersect(A, B, C, D):
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

def segment_intersects_rect(p0, p1, rect):
    xmin, ymin, xmax, ymax = rect

    # quick reject: both points on same outside side
    if (p0[0] < xmin and p1[0] < xmin) or (p0[0] > xmax and p1[0] > xmax) or \
       (p0[1] < ymin and p1[1] < ymin) or (p0[1] > ymax and p1[1] > ymax):
        return False

    # endpoints inside rect => intersect
    if xmin <= p0[0] <= xmax and ymin <= p0[1] <= ymax:
        return True
    if xmin <= p1[0] <= xmax and ymin <= p1[1] <= ymax:
        return True

    edges = [
        ((xmin, ymin), (xmax, ymin)),
        ((xmax, ymin), (xmax, ymax)),
        ((xmax, ymax), (xmin, ymax)),
        ((xmin, ymax), (xmin, ymin)),
    ]
    for a, b in edges:
        if segments_intersect(p0, p1, a, b):
            return True
    return False

def in_fov(robot_xy, robot_theta, obj_xy, fov_range, fov_half_angle_deg):
    dx = obj_xy[0] - robot_xy[0]
    dy = obj_xy[1] - robot_xy[1]
    r = math.hypot(dx, dy)
    if r > fov_range:
        return False
    ang = math.atan2(dy, dx)
    da = wrap_angle(ang - robot_theta)
    return abs(da) <= math.radians(fov_half_angle_deg)

def visible(robot_xy, robot_theta, obj_xy, obstacles, fov_range, fov_half_angle_deg):
    if not in_fov(robot_xy, robot_theta, obj_xy, fov_range, fov_half_angle_deg):
        return False
    for rect in obstacles:
        if segment_intersects_rect(robot_xy, obj_xy, rect):
            return False
    return True


# =========================
# World
# =========================

@dataclass
class Obj:
    oid: int
    xy: np.ndarray  # (2,)
    v: np.ndarray   # (2,)

def step_world(objects: List[Obj], dt: float, world_size: float, accel_std: float, friction: float):
    for o in objects:
        if accel_std > 0.0:
            a = np.random.randn(2).astype(np.float32) * accel_std
            o.v = friction * o.v + a * dt
        else:
            o.v = friction * o.v

        o.xy = o.xy + o.v * dt

        # bounce off walls
        for k in range(2):
            if o.xy[k] < 0:
                o.xy[k] = 0
                o.v[k] *= -1.0
            if o.xy[k] > world_size:
                o.xy[k] = world_size
                o.v[k] *= -1.0


# =========================
# Kalman Track
# =========================

@dataclass
class Track:
    tid: int
    mu: np.ndarray          # (4,) = [px,py,vx,vy]
    Sigma: np.ndarray       # (4,4)
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False

    def predict(self, F, Q):
        self.mu = F @ self.mu
        self.Sigma = F @ self.Sigma @ F.T + Q
        self.age += 1

    def innovation(self, z, H, R):
        zhat = H @ self.mu
        S = H @ self.Sigma @ H.T + R
        y = z - zhat
        return y, S

    def update(self, z, H, R):
        y, S = self.innovation(z, H, R)
        K = self.Sigma @ H.T @ np.linalg.inv(S)
        self.mu = self.mu + K @ y
        I = np.eye(4, dtype=np.float32)
        self.Sigma = (I - K @ H) @ self.Sigma

        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True


# =========================
# Hungarian assignment (unchanged from D56-B)
# =========================

def hungarian_min_cost(cost: np.ndarray) -> List[int]:
    n = cost.shape[0]
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int32)   # matching for columns: p[j] = i
    way = np.zeros(n + 1, dtype=np.int32)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=np.bool_)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * n
    for j in range(1, n + 1):
        i = p[j]
        if i != 0:
            assignment[i - 1] = j - 1
    return assignment


# =========================
# Multi-Object Tracker (same logic as D56-B)
# =========================

@dataclass
class MOT:
    next_id: int = 1
    tracks: List[Track] = field(default_factory=list)

    def spawn(self, z_xy, init_vel_std=0.8, init_pos_std=1.0):
        mu = np.array([z_xy[0], z_xy[1],
                       np.random.randn() * init_vel_std,
                       np.random.randn() * init_vel_std], dtype=np.float32)
        Sigma = np.diag([init_pos_std**2, init_pos_std**2,
                         (2.0 * init_vel_std)**2, (2.0 * init_vel_std)**2]).astype(np.float32)
        t = Track(tid=self.next_id, mu=mu, Sigma=Sigma)
        self.next_id += 1
        self.tracks.append(t)

    def prune(self, max_misses):
        self.tracks = [t for t in self.tracks if t.misses <= max_misses]

    def step(self,
             Z: List[np.ndarray],
             F, Q, H, R,
             gate_maha: float,
             max_misses: int,
             spawn_unmatched: bool = True,
             use_hungarian: bool = True,
             track_no_match_cost: float = 12.0,
             meas_no_match_cost: float = 12.0):

        for t in self.tracks:
            t.predict(F, Q)

        if len(self.tracks) == 0:
            if spawn_unmatched:
                for z in Z:
                    self.spawn(z)
            return

        T = len(self.tracks)
        M = len(Z)

        if M == 0:
            for tr in self.tracks:
                tr.misses += 1
            self.prune(max_misses)
            return

        C = np.full((T, M), 1e9, dtype=np.float64)
        for i, tr in enumerate(self.tracks):
            for j, z in enumerate(Z):
                y, S = tr.innovation(z, H, R)
                try:
                    d2 = float(y.T @ np.linalg.inv(S) @ y)
                except np.linalg.LinAlgError:
                    d2 = 1e9
                if d2 <= gate_maha:
                    C[i, j] = d2
                else:
                    C[i, j] = 1e9

        if not use_hungarian:
            assigned_tracks = set()
            assigned_meas = set()
            flat = [(C[i, j], i, j) for i in range(T) for j in range(M)]
            flat.sort(key=lambda x: x[0])
            for d2, i, j in flat:
                if d2 > gate_maha:
                    break
                if i in assigned_tracks or j in assigned_meas:
                    continue
                assigned_tracks.add(i)
                assigned_meas.add(j)
                self.tracks[i].update(Z[j], H, R)
            for i, tr in enumerate(self.tracks):
                if i not in assigned_tracks:
                    tr.misses += 1
            if spawn_unmatched:
                for j, z in enumerate(Z):
                    if j not in assigned_meas:
                        self.spawn(z)
            self.prune(max_misses)
            return

        n = T + M
        BIG = 1e9
        A = np.full((n, n), BIG, dtype=np.float64)

        A[:T, :M] = C

        for i in range(T):
            A[i, M + i] = float(track_no_match_cost)

        for j in range(M):
            A[T + j, j] = float(meas_no_match_cost)

        A[T:, M:] = 0.0

        assign = hungarian_min_cost(A)

        track_assigned_meas = [-1] * T
        meas_is_matched = [False] * M

        for i in range(T):
            j = assign[i]
            if j < M and A[i, j] < BIG / 2:
                track_assigned_meas[i] = j
                meas_is_matched[j] = True
            else:
                track_assigned_meas[i] = -1

        for i, tr in enumerate(self.tracks):
            j = track_assigned_meas[i]
            if j >= 0:
                tr.update(Z[j], H, R)
            else:
                tr.misses += 1

        if spawn_unmatched:
            for j in range(M):
                if not meas_is_matched[j]:
                    self.spawn(Z[j])

        self.prune(max_misses)

def render_ascii(world_size, robot_xy, robot_theta, objects, obstacles, mot: MOT,
                 grid_w=30, grid_h=20,
                 unc_low=3.0, unc_mid=6.0):
    def to_cell(xy):
        gx = int(clamp((xy[0] / world_size) * (grid_w - 1), 0, grid_w - 1))
        gy = int(clamp((xy[1] / world_size) * (grid_h - 1), 0, grid_h - 1))
        return gx, gy

    canvas = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    def set_if_empty(cx, cy, ch):
        """cy is world-y cell (0 bottom), we map to canvas row with flip."""
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            rr = grid_h - 1 - cy
            if canvas[rr][cx] == ".":
                canvas[rr][cx] = ch

    def set_force(cx, cy, ch):
        """Overwrite (used for track center and robot), respecting bounds."""
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            canvas[grid_h - 1 - cy][cx] = ch

    # obstacle
    for (xmin, ymin, xmax, ymax) in obstacles:
        x0, y0 = to_cell((xmin, ymin))
        x1, y1 = to_cell((xmax, ymax))
        for yy in range(min(y0, y1), max(y0, y1) + 1):
            for xx in range(min(x0, x1), max(x0, x1) + 1):
                set_force(xx, yy, "#")

    # truth objects
    for o in objects:
        x, y = to_cell(o.xy)
        set_force(x, y, str(o.oid % 10))

    # --- D56-C CHANGE ---
    # draw halos first (only fill '.' cells), then draw track letters
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])

        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc < unc_low:
            unc_level = "LOW"
        elif pos_unc < unc_mid:
            unc_level = "MID"
        else:
            unc_level = "HIGH"

        # infer mode: misses==0 means it updated this step (most of the time)
        mode = "UPDATE" if tr.misses == 0 else "COAST"
        # --- D56-C FIX ---
        # ':' can look like it adds vertical noise in some monospace fonts.
        halo_char = "+" if mode == "UPDATE" else ","
        # halo_char = "+" if mode == "UPDATE" else ":"

        # choose halo shape by uncertainty
        if unc_level == "LOW":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif unc_level == "MID":
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        for dx, dy in offsets:
            set_if_empty(tx + dx, ty + dy, halo_char)

    # draw track centers (letters) after halos
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])
        idx = (tr.tid - 1) % 26
        letter = chr(ord("A") + idx)
        if not tr.confirmed:
            letter = letter.lower()

        # if very uncertain, force lowercase for the letter too (same behavior as earlier)
        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc >= unc_mid:
            letter = letter.lower()

        set_force(tx, ty, letter)

    # robot
    rx, ry = to_cell(robot_xy)
    set_force(rx, ry, "R")

    # heading marker
    hx = robot_xy[0] + 0.8 * math.cos(robot_theta)
    hy = robot_xy[1] + 0.8 * math.sin(robot_theta)
    hx, hy = to_cell((hx, hy))
    if 0 <= hx < grid_w and 0 <= hy < grid_h:
        rr = grid_h - 1 - hy
        if canvas[rr][hx] == ".":
            canvas[rr][hx] = "^"

    return "\n".join("".join(row) for row in canvas)


##56cG2 — add a “render to RGB” helper (place near render_ascii, below it)

#This recreates your canvas but returns an RGB image instead of text.

# ##56cG2 (GIF) render grid to RGB
def render_rgb(world_size, robot_xy, robot_theta, objects, obstacles, mot: MOT,
               grid_w=30, grid_h=20,
               unc_low=3.0, unc_mid=6.0,
               scale=16):
    def to_cell(xy):
        gx = int(clamp((xy[0] / world_size) * (grid_w - 1), 0, grid_w - 1))
        gy = int(clamp((xy[1] / world_size) * (grid_h - 1), 0, grid_h - 1))
        return gx, gy

    canvas = [["." for _ in range(grid_w)] for _ in range(grid_h)]

    def set_if_empty(cx, cy, ch):
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            rr = grid_h - 1 - cy
            if canvas[rr][cx] == ".":
                canvas[rr][cx] = ch

    def set_force(cx, cy, ch):
        if 0 <= cx < grid_w and 0 <= cy < grid_h:
            canvas[grid_h - 1 - cy][cx] = ch

    # obstacle
    for (xmin, ymin, xmax, ymax) in obstacles:
        x0, y0 = to_cell((xmin, ymin))
        x1, y1 = to_cell((xmax, ymax))
        for yy in range(min(y0, y1), max(y0, y1) + 1):
            for xx in range(min(x0, x1), max(x0, x1) + 1):
                set_force(xx, yy, "#")

    # truth objects (digits)
    for o in objects:
        x, y = to_cell(o.xy)
        set_force(x, y, str(o.oid % 10))

    # halos
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])
        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc < unc_low:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif pos_unc < unc_mid:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        mode = "UPDATE" if tr.misses == 0 else "COAST"
        halo_char = "+" if mode == "UPDATE" else ","
        for dx, dy in offsets:
            set_if_empty(tx + dx, ty + dy, halo_char)

    # track letters
    for tr in mot.tracks:
        tx, ty = to_cell(tr.mu[:2])
        idx = (tr.tid - 1) % 26
        letter = chr(ord("A") + idx)
        if (not tr.confirmed):
            letter = letter.lower()
        pos_unc = float(np.trace(tr.Sigma[:2, :2]))
        if pos_unc >= unc_mid:
            letter = letter.lower()
        set_force(tx, ty, letter)

    # robot + heading caret
    rx, ry = to_cell(robot_xy)
    set_force(rx, ry, "R")

    hx = robot_xy[0] + 0.8 * math.cos(robot_theta)
    hy = robot_xy[1] + 0.8 * math.sin(robot_theta)
    hx, hy = to_cell((hx, hy))
    if 0 <= hx < grid_w and 0 <= hy < grid_h:
        rr = grid_h - 1 - hy
        if canvas[rr][hx] == ".":
            canvas[rr][hx] = "^"

    # map chars -> RGB
    palette = {
        ".": (245,245,245),
        "#": ( 40, 40, 40),
        "R": (220, 30, 30),
        "^": (255,120,120),
        "+": ( 40,180, 40),
        ",": ( 40,120,200),
    }
    # digits 0-9 (truth objects)
    for d in "0123456789":
        palette[d] = (255,165,  0)

    # letters = tracks
    # uppercase (confident) vs lowercase (uncertain)
    import string
    for ch in string.ascii_uppercase:
        palette[ch] = (160,  0, 200)
    for ch in string.ascii_lowercase:
        palette[ch] = (200,140,220)

    img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    for r in range(grid_h):
        for c in range(grid_w):
            img[r, c] = palette.get(canvas[r][c], (0,0,0))

    if scale > 1:
        img = np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))

    # ##56cG2c (GIF) overlay characters so GIF matches ASCII
    if Image is not None and scale >= 10:
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)

        # Try monospace font; fall back safely
        font = None
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", int(scale * 0.9))
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", int(scale * 0.9))
            except Exception:
                font = ImageFont.load_default()

        for rr in range(grid_h):
            for cc in range(grid_w):
                ch = canvas[rr][cc]
                if ch == ".":
                    continue

                # simple placement inside cell
                x = cc * scale + int(scale * 0.18)
                y = rr * scale + int(scale * 0.02)

                # draw black text
                dr.text((x, y), ch, fill=(0, 0, 0), font=font)

        img = np.array(im, dtype=np.uint8)

    return img


        
# ##56cG2c (GIF) overlay characters so GIF matches ASCII
    if Image is not None and scale >= 10:
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)

        # Try a monospace-ish font; fall back to default
        font = None
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", int(scale * 0.9))
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", int(scale * 0.9))
            except Exception:
                font = ImageFont.load_default()

        for rr in range(grid_h):
            for cc in range(grid_w):
                ch = canvas[rr][cc]
                if ch == ".":
                    continue  # don't label empty
                # center-ish placement inside the cell
                x = cc * scale + int(scale * 0.18)
                y = rr * scale + int(scale * 0.02)
                dr.text((x, y), ch, fill=(0, 0, 0), font=font)

        img = np.array(im, dtype=np.uint8)
     
        
        
        
        
    return img



def main():
    ap = argparse.ArgumentParser()

# Robot motion (your preference: up/down)
# Use the updated main-loop block I gave you earlier (the one that moves robot_xy[1] up/down and flips robot_theta).
# If you want it configurable, add:

    ap.add_argument("--robot_speed", type=float, default=1.0)
    ap.add_argument("--robot_move", type=int, default=1)

# and keep the patrol gated by if args.robot_move:.


    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--print_every", type=int, default=2)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--world_size", type=float, default=30.0)

    # vision
    ap.add_argument("--fov_range", type=float, default=100.0)
    ap.add_argument("--fov_half_angle", type=float, default=180.0)
    ap.add_argument("--meas_noise_std", type=float, default=1.5)

    # objects
    ap.add_argument("--num_objects", type=int, default=2)
    ap.add_argument("--accel_std", type=float, default=0.0)
    ap.add_argument("--friction", type=float, default=1.0)

    # tracker model
    ap.add_argument("--process_pos_std", type=float, default=0.10)
    ap.add_argument("--process_vel_std", type=float, default=0.30)
    ap.add_argument("--gate_maha", type=float, default=9.0)
    ap.add_argument("--max_misses", type=int, default=200)

    # association
    ap.add_argument("--track_no_match_cost", type=float, default=12.0)
    ap.add_argument("--meas_no_match_cost", type=float, default=12.0)
    ap.add_argument("--use_hungarian", type=int, default=1)

    # rendering
    ap.add_argument("--grid_w", type=int, default=30)
    ap.add_argument("--grid_h", type=int, default=20)
    
    ##56cG1 — add CLI args (in main(), under your rendering args)

    ## Put this right after your existing “# rendering” args block:

    # ##56cG1 (GIF)
    ap.add_argument("--gif", type=int, default=0)
    ap.add_argument("--gif_out", type=str, default="d56c.gif")
    ap.add_argument("--gif_every", type=int, default=2)   # capture every N steps
    ap.add_argument("--gif_fps", type=float, default=1.0) # 1–2 fps = slow playback
    ap.add_argument("--gif_scale", type=int, default=16)  # pixel scaling per cell

    # --- D56-C CHANGE ---
    # uncertainty thresholds for halo behavior
    ap.add_argument("--unc_low", type=float, default=3.0)
    ap.add_argument("--unc_mid", type=float, default=6.0)

    # obstacle
    ap.add_argument("--obstacle", type=str, default="14,17,24,22")

    # scenario controls
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--force_initial_spawn", type=int, default=1)
    
    ###########################################
    # ap.add_argument("--robot_speed", type=float, default=1.0)

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    oxmin, oymin, oxmax, oymax = [float(x) for x in args.obstacle.split(",")]
    obstacles = [(oxmin, oymin, oxmax, oymax)]

    # Robot fixed for comparability
    # robot_xy = np.array([4.0, 4.0], dtype=np.float32)
    robot_theta = 0.70
    # robot starts just below obstacle band (y < 17), same x as before
    # robot_xy = np.array([4.0, 16.0], dtype=np.float32)
    # ##56cR0 (robot start near occlusion)
    robot_xy = np.array([4.0, max(0.0, oymin - 1.0)], dtype=np.float32)  # y just below obstacle

    # Objects: same deterministic story as before
    objects: List[Obj] = []
    if args.deterministic:
        objects.append(Obj(
            oid=1,
            xy=np.array([20.0, 24.0], dtype=np.float32),
            v=np.array([-0.2, 0.0], dtype=np.float32),
        ))
        objects.append(Obj(
            oid=2,
            xy=np.array([10.0, 19.5], dtype=np.float32),
            v=np.array([1.2, 0.0], dtype=np.float32),
        ))
    else:
        for i in range(args.num_objects):
            xy = np.array([np.random.uniform(6, args.world_size - 2),
                           np.random.uniform(6, args.world_size - 2)], dtype=np.float32)
            v = np.random.randn(2).astype(np.float32) * 1.0
            objects.append(Obj(oid=i + 1, xy=xy, v=v))

    # Kalman model
    dt = args.dt
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]], dtype=np.float32)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=np.float32)

    Q = np.diag([args.process_pos_std**2, args.process_pos_std**2,
                 args.process_vel_std**2, args.process_vel_std**2]).astype(np.float32)
    R = np.diag([args.meas_noise_std**2, args.meas_noise_std**2]).astype(np.float32)

    mot = MOT()

    if args.force_initial_spawn:
        for o in objects:
            z0 = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
            mot.spawn(z0)

    # --- D56-C CHANGE ---
    # Print legend once so you can read halos at a glance.
    print("Legend:")
    print("  R,^ = robot + heading")
    print("  1,2 = true objects")
    print("  A,B = confirmed tracks (lowercase = unconfirmed or very uncertain)")
    print("  '+' halo = UPDATE step")
    print("  ',' halo = COAST step")
    print("  halo size indicates uncertainty: LOW(4-neigh), MID(8-neigh), HIGH(dist-2 cross)")
    print("-" * 60)

##56cR2 + ##56cG3/4/5 (UPDATED main-loop block)
####Paste this after the legend prints, replacing your current frames / loop / write section:

    # ##56cG3 (GIF) init V2222222222222222222222
    frames = []
    if args.gif:
        if imageio is None:
            raise RuntimeError("imageio not available. pip install imageio")

    # ##56cR2 (robot motion) deterministic up/down patrol
    # start moving upward; bounce at world bounds
    robot_vy = +1.0   # world units/sec (set via args if you want)
    if hasattr(args, "robot_speed"):
        robot_vy = float(args.robot_speed)

    ###########################
    print(f"Robot start: {robot_xy}  obstacle_y=[{oymin},{oymax}]")

    for t in range(args.steps):
        # --- world step (objects) ---
        step_world(objects, dt, args.world_size, args.accel_std, args.friction)

        # --- robot step (up/down only) ---
        # keep x fixed; move y; bounce at 0/world_size
        robot_xy[1] = float(robot_xy[1] + robot_vy * dt)
        if robot_xy[1] < 0.0:
            robot_xy[1] = 0.0
            robot_vy *= -1.0
        elif robot_xy[1] > args.world_size:
            robot_xy[1] = float(args.world_size)
            robot_vy *= -1.0

        # heading should match direction (so '^' makes sense)
        robot_theta = (math.pi / 2) if robot_vy >= 0 else (-math.pi / 2)

        # --- measurements ---
        Z: List[np.ndarray] = []
        vis = 0
        for o in objects:
            if visible(tuple(robot_xy), robot_theta, tuple(o.xy),
                       obstacles, args.fov_range, args.fov_half_angle):
                vis += 1
                z = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
                Z.append(z)

        # --- tracker step ---
        mot.step(
            Z=Z,
            F=F, Q=Q,
            H=H, R=R,
            gate_maha=args.gate_maha,
            max_misses=args.max_misses,
            spawn_unmatched=True,
            use_hungarian=bool(args.use_hungarian),
            track_no_match_cost=args.track_no_match_cost,
            meas_no_match_cost=args.meas_no_match_cost,
        )

        # ##56cG4 (GIF) capture (AFTER tracker update)
        if args.gif and (t % args.gif_every == 0):
            frame = render_rgb(args.world_size, robot_xy, robot_theta,
                               objects, obstacles, mot,
                               grid_w=args.grid_w, grid_h=args.grid_h,
                               unc_low=args.unc_low, unc_mid=args.unc_mid,
                               scale=args.gif_scale)
            frames.append(frame)

        # --- your existing logging ---
        if (t % args.print_every == 0) or (t == args.steps - 1):
            confirmed = sum(1 for tr in mot.tracks if tr.confirmed)
            print(f"[D56-C][t={t:03d}] vis_meas={vis}  tracks={len(mot.tracks)}  confirmed={confirmed}")
            print(render_ascii(args.world_size, robot_xy, robot_theta,
                               objects, obstacles, mot,
                               grid_w=args.grid_w, grid_h=args.grid_h,
                               unc_low=args.unc_low, unc_mid=args.unc_mid))
            print()

            for tr in mot.tracks:
                pos_unc = float(np.trace(tr.Sigma[:2, :2]))
                idx = (tr.tid - 1) % 26
                name = chr(ord("A") + idx)
                if not tr.confirmed:
                    name = name.lower()
                mode = "UPDATE" if tr.misses == 0 and vis > 0 else "COAST"
                unc_level = "LOW" if pos_unc < args.unc_low else ("MID" if pos_unc < args.unc_mid else "HIGH")
                print(f"  track {name} (tid={tr.tid:02d}) conf={int(tr.confirmed)} "
                      f"mode={mode:6s} miss={tr.misses:02d} "
                      f"mu=({tr.mu[0]:5.2f},{tr.mu[1]:5.2f}) pos_unc={pos_unc:6.2f} unc={unc_level}")
            print("-" * 60)

    # ##56cG5 (GIF) write ONCE, AFTER the loop
    if args.gif:
        imageio.mimsave(args.gif_out, frames, fps=float(args.gif_fps))
        print(f"Wrote GIF: {args.gif_out}  frames={len(frames)}  fps={args.gif_fps}  gif_every={args.gif_every}")


# ##56cG3 — collect frames + write GIF (inside main())

#    #Add this after your legend prints (before the for t in range(args.steps): loop):

#     # ##56cG3 (GIF) init
#     frames = []
#     if args.gif:
#         if imageio is None:
#             raise RuntimeError("imageio not available. pip install imageio")

#     for t in range(args.steps):
#         step_world(objects, dt, args.world_size, args.accel_std, args.friction)

#         Z: List[np.ndarray] = []
#         vis = 0
#         for o in objects:
#             if visible(tuple(robot_xy), robot_theta, tuple(o.xy),
#                        obstacles, args.fov_range, args.fov_half_angle):
#                 vis += 1
#                 z = o.xy + np.random.randn(2).astype(np.float32) * args.meas_noise_std
#                 Z.append(z)

#         mot.step(
#             Z=Z,
#             F=F, Q=Q,
#             H=H, R=R,
#             gate_maha=args.gate_maha,
#             max_misses=args.max_misses,
#             spawn_unmatched=True,
#             use_hungarian=bool(args.use_hungarian),
#             track_no_match_cost=args.track_no_match_cost,
#             meas_no_match_cost=args.meas_no_match_cost,
#         )
        
#         # ##56cG4 (GIF) capture
#         if args.gif and (t % args.gif_every == 0):
#             frame = render_rgb(args.world_size, robot_xy, robot_theta,
#                                objects, obstacles, mot,
#                                grid_w=args.grid_w, grid_h=args.grid_h,
#                                unc_low=args.unc_low, unc_mid=args.unc_mid,
#                                scale=args.gif_scale)
#             frames.append(frame)

#         if (t % args.print_every == 0) or (t == args.steps - 1):
#             confirmed = sum(1 for tr in mot.tracks if tr.confirmed)
#             print(f"[D56-C][t={t:03d}] vis_meas={vis}  tracks={len(mot.tracks)}  confirmed={confirmed}")
#             print(render_ascii(args.world_size, robot_xy, robot_theta,
#                                objects, obstacles, mot,
#                                grid_w=args.grid_w, grid_h=args.grid_h,
#                                unc_low=args.unc_low, unc_mid=args.unc_mid))
#             print()

#             for tr in mot.tracks:
#                 pos_unc = float(np.trace(tr.Sigma[:2, :2]))
#                 idx = (tr.tid - 1) % 26
#                 name = chr(ord("A") + idx)
#                 if not tr.confirmed:
#                     name = name.lower()
#                 mode = "UPDATE" if tr.misses == 0 and vis > 0 else "COAST"
#                 unc_level = "LOW" if pos_unc < args.unc_low else ("MID" if pos_unc < args.unc_mid else "HIGH")
#                 print(f"  track {name} (tid={tr.tid:02d}) conf={int(tr.confirmed)} "
#                       f"mode={mode:6s} miss={tr.misses:02d} "
#                       f"mu=({tr.mu[0]:5.2f},{tr.mu[1]:5.2f}) pos_unc={pos_unc:6.2f} unc={unc_level}")
#             print("-" * 60)

# # ##56cG5 (GIF) write
#         if args.gif:
#                 imageio.mimsave(args.gif_out, frames, fps=float(args.gif_fps))
#                 print(f"Wrote GIF: {args.gif_out}  frames={len(frames)}  fps={args.gif_fps}  gif_every={args.gif_every}")


if __name__ == "__main__":
    main()
