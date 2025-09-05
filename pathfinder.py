#!/usr/bin/env python3
import argparse, math, sys, os
from heapq import heappush, heappop
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# -------------------- robust grid loading --------------------
def try_numeric_loader(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    use_delim = "," if any("," in ln for ln in lines[:50]) else None
    arr = np.genfromtxt(lines, delimiter=use_delim, dtype=float)
    if arr.ndim != 2:
        raise ValueError("numeric load did not yield 2D array")
    return arr

def try_char_loader(path: str):
    rows = []
    with open(path, "r") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            if any(ch in s for ch in (" ", "\t", ",")):
                toks = [t for t in s.replace(",", " ").split()]
                row = []
                for t in toks:
                    if t in ("0","1"):
                        row.append(int(t))
                    elif t.replace(".","",1).isdigit():
                        row.append(1 if float(t) > 0.5 else 0)
                if row:
                    rows.append(row)
            else:
                rows.append([1 if c=="1" else 0 for c in s if c in ("0","1")])
    if not rows:
        raise ValueError("char loader parsed no rows")
    maxw = max(len(r) for r in rows)
    rows = [r + [1]*(maxw-len(r)) for r in rows]   # pad short lines w/ obstacles
    return np.array(rows, dtype=float)

def load_grid(path: str, invert: bool, thresh: float) -> np.ndarray:
    """Return uint8 grid: 1 = obstacle, 0 = free."""
    try:
        arr = try_numeric_loader(path)
        obs = (arr > thresh).astype(np.uint8)
    except Exception:
        arr = try_char_loader(path)
        obs = (arr > thresh).astype(np.uint8)
    if invert:
        obs = 1 - obs
    return np.asarray(obs, dtype=np.uint8)

# -------------------- helpers --------------------
def grid_stats(grid: np.ndarray, name="grid"):
    h, w = grid.shape
    occ = int(grid.sum())
    frac = occ / (h*w)
    print(f"{name}: shape={h}x{w}, obstacles={occ} ({frac:.1%})")

def disk_offsets(radius: int):
    r2 = radius*radius
    return [(dx,dy) for dy in range(-radius,radius+1)
                   for dx in range(-radius,radius+1)
                   if dx*dx+dy*dy <= r2]

def inflate_obstacles(grid: np.ndarray, diam_cells: int) -> np.ndarray:
    """Dilate obstacles by a disk radius ceil(d/2). Treat border as obstacle."""
    if diam_cells <= 0:
        return grid.copy()
    r = int(math.ceil(diam_cells / 2.0))
    g = (grid.astype(bool))
    pad = r
    gp = np.pad(g, pad_width=pad, mode='constant', constant_values=True)
    try:
        from scipy import ndimage as ndi
        yy, xx = np.ogrid[-r:r+1, -r:r+1]
        se = (xx*xx + yy*yy) <= r*r
        gp_dil = ndi.binary_dilation(gp, structure=se)
    except Exception:
        gp_dil = gp.copy()
        offs = disk_offsets(r)
        ys, xs = np.where(gp)
        H, W = gp.shape
        for y, x in zip(ys, xs):
            for dx, dy in offs:
                xx = x + dx; yy = y + dy
                if 0 <= xx < W and 0 <= yy < H:
                    gp_dil[yy, xx] = True
    inflated = gp_dil[pad:-pad, pad:-pad]
    return inflated.astype(np.uint8)

def neighbors(x, y, w, h, allow_diag=True, grid=None):
    """Yield (xx, yy, step_cost). Prevent diagonal corner-cutting when used."""
    steps4 = [(1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0)]
    for dx, dy, cost in steps4:
        xx, yy = x+dx, y+dy
        if 0 <= xx < w and 0 <= yy < h:
            yield xx, yy, cost
    if not allow_diag:
        return
    for dx, dy in ((1,1), (1,-1), (-1,1), (-1,-1)):
        xx, yy = x+dx, y+dy
        if 0 <= xx < w and 0 <= yy < h:
            if grid is not None and (grid[y, x+dx] == 1 or grid[y+dy, x] == 1):
                continue  # block corner cutting
            yield xx, yy, 1.4142135623730951

def reconstruct_path(parent, goal):
    path=[]; cur=goal
    while cur is not None:
        path.append(cur); cur=parent[cur]
    path.reverse()
    return path

def astar(grid: np.ndarray, start, goal, allow_diag=True):
    """A* over grid. Returns (path, length, errstr|None)."""
    h, w = grid.shape
    sx, sy = start; gx, gy = goal
    if not (0 <= sx < w and 0 <= sy < h and 0 <= gx < w and 0 <= gy < h):
        return None, math.inf, "out_of_bounds"
    if grid[sy, sx] == 1:
        return None, math.inf, "start_blocked"
    if grid[gy, gx] == 1:
        return None, math.inf, "goal_blocked"

    def heur(x, y):
        dx = abs(x-gx); dy = abs(y-gy)
        if allow_diag:
            # octile
            return (dx+dy) + (1.4142135623730951 - 2.0) * min(dx, dy)
        else:
            return dx + dy

    openh=[]; heappush(openh,(heur(sx,sy),0.0,(sx,sy)))
    g = {(sx,sy):0.0}; parent={(sx,sy):None}
    closed=set()
    while openh:
        f, gc, (x,y)=heappop(openh)
        if (x,y) in closed:
            continue
        closed.add((x,y))
        if (x,y)==(gx,gy):
            return reconstruct_path(parent, (gx,gy)), g[(gx,gy)], None
        for xx,yy,step in neighbors(x,y,w,h, allow_diag=allow_diag, grid=grid):
            if grid[yy,xx]==1 or (xx,yy) in closed:
                continue
            ng = g[(x,y)] + step
            if (xx,yy) not in g or ng < g[(xx,yy)]:
                g[(xx,yy)] = ng
                parent[(xx,yy)] = (x,y)
                heappush(openh,(ng+heur(xx,yy),ng,(xx,yy)))
    return None, math.inf, "disconnected"

def snap_to_free(grid: np.ndarray, p, rmax=20):
    """Return nearest free cell to p within rmax (BFS ring search)."""
    x0, y0 = p
    h, w = grid.shape
    if 0 <= x0 < w and 0 <= y0 < h and grid[y0,x0]==0:
        return p
    for r in range(1, rmax+1):
        for dy in range(-r, r+1):
            for dx in (-r, r):
                x, y = x0+dx, y0+dy
                if 0 <= x < w and 0 <= y < h and grid[y,x]==0:
                    return (x,y)
        for dx in range(-r+1, r):
            for dy in (-r, r):
                x, y = x0+dx, y0+dy
                if 0 <= x < w and 0 <= y < h and grid[y,x]==0:
                    return (x,y)
    return None

def save_plot(grid: np.ndarray, path_points=None, out_path="path.png",
              title="Path", zoom=False, margin=10, dpi=200):
    h, w = grid.shape
    img = np.where(grid==1, 0.7, 1.0)
    x0=y0=0; x1=w; y1=h
    if zoom and path_points:
        xs = np.array([p[0] for p in path_points]); ys = np.array([p[1] for p in path_points])
        x0 = max(int(xs.min())-margin, 0); x1 = min(int(xs.max())+margin+1, w)
        y0 = max(int(ys.min())-margin, 0); y1 = min(int(ys.max())+margin+1, h)
        img = img[y0:y1, x0:x1]
    lw = max(2, int(0.01*max(h,w)))
    plt.figure(figsize=(8,8), dpi=dpi)
    plt.imshow(img, cmap="gray", interpolation="nearest", origin="upper")
    if path_points:
        xs = [p[0]-x0 for p in path_points]; ys = [p[1]-y0 for p in path_points]
        plt.plot(xs, ys, linewidth=lw, color="red", alpha=0.9)
        plt.scatter([xs[0]],[ys[0]], s=80*lw, c="lime", edgecolors="black", linewidths=1.5, zorder=3)
        plt.scatter([xs[-1]],[ys[-1]], s=80*lw, c="dodgerblue", edgecolors="black", linewidths=1.5, zorder=3)
    plt.title(title); plt.axis("equal"); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=dpi); plt.close()

# -------- Live / streaming A* helpers --------
def astar_stream(grid: np.ndarray, start, goal, allow_diag=True, yield_every=300):
    """
    A* that yields progress every `yield_every` pops.
    Yields dicts: {"closed": closed_bool, "current": (x,y), "steps": n}
    and finally returns (path, cost, None) like astar().
    """
    h, w = grid.shape
    sx, sy = start; gx, gy = goal

    def heur(x, y):
        dx = abs(x-gx); dy = abs(y-gy)
        return ((dx+dy) + (1.4142135623730951 - 2.0) * min(dx, dy)) if allow_diag else (dx+dy)

    openh = []
    heappush(openh, (heur(sx,sy), 0.0, (sx,sy)))
    g = {(sx,sy): 0.0}
    parent = {(sx,sy): None}
    closed = np.zeros((h, w), dtype=bool)

    step = 0
    while openh:
        f, gc, (x,y) = heappop(openh)
        if closed[y, x]:
            continue
        closed[y, x] = True
        step += 1
        if step % yield_every == 0:
            yield {"closed": closed.copy(), "current": (x,y), "steps": step}

        if (x,y) == (gx,gy):
            path=[]; cur=(x,y)
            while cur is not None:
                path.append(cur); cur=parent[cur]
            path.reverse()
            return path, g[(gx,gy)], None

        for xx, yy, step_cost in neighbors(x, y, w, h, allow_diag=allow_diag, grid=grid):
            if grid[yy, xx] == 1 or closed[yy, xx]:
                continue
            ng = g[(x,y)] + step_cost
            if (xx,yy) not in g or ng < g[(xx,yy)]:
                g[(xx,yy)] = ng
                parent[(xx,yy)] = (x,y)
                heappush(openh, (ng + heur(xx,yy), ng, (xx,yy)))

    return None, math.inf, "disconnected"

def live_plot(grid: np.ndarray, start, goal, allow_diag=True, yield_every=300,
              title="A* (live)", out_anim=None, dpi=120,
              show_stats=True, auto_close_secs=None):
    """
    Live animation using matplotlib. If out_anim ends with '.gif' or '.mp4',
    the animation is saved; else it's shown interactively.
    """
    h, w = grid.shape
    base = np.where(grid==1, 0.7, 1.0)  # light gray obstacles, white free

    fig, ax = plt.subplots(figsize=(8,8), dpi=dpi)
    ax.set_title(title); ax.axis("off"); ax.set_aspect("equal")
    img = ax.imshow(base, cmap="gray", interpolation="nearest", origin="upper")
    start_pt = ax.scatter([start[0]],[start[1]], s=60, c="lime", edgecolors="black", zorder=3)
    goal_pt  = ax.scatter([goal[0] ],[goal[1] ],  s=60, c="dodgerblue", edgecolors="black", zorder=3)
    path_plot, = ax.plot([], [], linewidth=2.5, color="red", alpha=0.9, zorder=4)

    stream = astar_stream(grid, start, goal, allow_diag=allow_diag, yield_every=yield_every)
    final_result = {"path": None, "cost": math.inf, "why": None}
    finished = {"done": False}  # mutable so inner fn can modify
    last_steps = {"n": 0}       # expansions seen

    def draw_final_path():
        if final_result["path"]:
            xs = [p[0] for p in final_result["path"]]
            ys = [p[1] for p in final_result["path"]]
            path_plot.set_data(xs, ys)

    def update(_frame):
        if finished["done"]:
            return (img, path_plot, start_pt, goal_pt)

        try:
            progress = next(stream)
            last_steps["n"] = progress.get("steps", last_steps["n"])
            closed = progress["closed"]
            overlay = base.copy()
            overlay[closed] = 0.55  # explored cells darker
            img.set_data(overlay)
            return (img, path_plot, start_pt, goal_pt)

        except StopIteration as e:
            res = e.value  # may be None on subsequent StopIterations
            if isinstance(res, tuple) and len(res) == 3:
                final_result["path"], final_result["cost"], final_result["why"] = res
            else:
                final_result["path"], final_result["cost"], final_result["why"] = (None, math.inf, "done")

            draw_final_path()

            if show_stats:
                cost_txt = (f"{final_result['cost']:.2f} cells"
                            if np.isfinite(final_result["cost"]) else "no path")
                ax.set_title(f"{title}\nDone: cost={cost_txt}, expansions={last_steps['n']}")
                fig.canvas.draw_idle()

            if auto_close_secs and not out_anim:
                t = fig.canvas.new_timer(interval=int(1000*auto_close_secs))
                t.add_callback(plt.close, fig)
                t.start()

            finished["done"] = True
            return (img, path_plot, start_pt, goal_pt)

    ani = animation.FuncAnimation(fig, update, interval=60, blit=False, repeat=False)

    if out_anim:
        ext = os.path.splitext(out_anim)[1].lower()
        if ext == ".gif":
            try:
                from matplotlib.animation import PillowWriter
                ani.save(out_anim, writer=PillowWriter(fps=int(1000/60)))
                print(f"Saved animation to {out_anim}")
                plt.close(fig); return final_result
            except Exception as ex:
                print(f"GIF save failed ({ex}); showing live window instead.")
        elif ext == ".mp4":
            try:
                ani.save(out_anim, writer="ffmpeg")
                print(f"Saved animation to {out_anim}")
                plt.close(fig); return final_result
            except Exception as ex:
                print(f"MP4 save failed ({ex}); showing live window instead.")
    plt.show()
    return final_result

# -------- Downsampling (speed-up) --------
def downsample_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample by block max (if any obstacle in k×k block -> obstacle).
    Pads edges with obstacles so shape becomes divisible by factor.
    """
    if factor <= 1:
        return grid
    h, w = grid.shape
    H = ((h + factor - 1) // factor) * factor
    W = ((w + factor - 1) // factor) * factor
    pad_h = H - h
    pad_w = W - w
    if pad_h or pad_w:
        grid = np.pad(grid, ((0,pad_h),(0,pad_w)), mode="constant", constant_values=1)
    grid = grid.astype(np.uint8)
    g = grid.reshape(H//factor, factor, W//factor, factor).max(axis=(1,3))
    return g.astype(np.uint8)

# -------------------- Randomize-mode helpers --------------------
def parse_range(spec: str):
    """Parse A:B[:C] -> (start, stop, step). Inclusive stop."""
    parts = [int(p) for p in spec.split(":")]
    if len(parts) == 2:
        a, b = parts; c = 1
    elif len(parts) == 3:
        a, b, c = parts
    else:
        raise ValueError("diam_list must be A:B or A:B:C")
    if c == 0:
        raise ValueError("step cannot be 0")
    return a, b, c

def parse_diam_list(s: str):
    """Accept '0,2,4,6' or '0:10:2'."""
    if "," in s:
        return sorted({int(x) for x in s.split(",")})
    a,b,c = parse_range(s)
    if c > 0:
        return list(range(a, b+1, c))
    else:
        return list(range(a, b-1, c))

def random_free_pairs(grid0: np.ndarray, n_pairs: int, rng: np.random.Generator, max_tries=20_000):
    """Sample n_pairs start/goal pairs from FREE cells of the baseline grid."""
    free = np.argwhere(grid0 == 0)
    if free.size == 0:
        raise RuntimeError("No free cells found in grid.")
    pairs = []
    tries = 0
    while len(pairs) < n_pairs and tries < max_tries:
        i, j = rng.integers(0, len(free), size=2)
        if i == j:
            tries += 1; continue
        sx, sy = int(free[i][1]), int(free[i][0])
        gx, gy = int(free[j][1]), int(free[j][0])
        pairs.append(((sx,sy),(gx,gy)))
        tries += 1
    if len(pairs) < n_pairs:
        print(f"Warning: only sampled {len(pairs)} pairs (requested {n_pairs}).")
    return pairs

def randomize(grid, diameters, n_pairs, allow_diag, seed, target_cov, target_ratio, csv_path=None):
    """
    Runs randomized trials:
      - Baseline = diam=0 path
      - For each diameter d: coverage = fraction of pairs with a path,
        efficiency_ok = fraction with (len_d / len_0) <= target_ratio.
    Recommends the largest d meeting coverage >= target_cov and efficiency_ok >= target_cov.
    """
    rng = np.random.default_rng(seed)
    infl = {d: inflate_obstacles(grid, d) for d in sorted(set(diameters))}
    base = infl.get(0, inflate_obstacles(grid, 0))
    pairs = random_free_pairs(base, n_pairs, rng)
    if 0 not in infl:
        infl[0] = base

    valid_pairs = []
    base_costs = []
    for (s,g) in pairs:
        p0, c0, why0 = astar(infl[0], s, g, allow_diag=allow_diag)
        if p0 is None:
            continue
        valid_pairs.append((s,g))
        base_costs.append(c0)
    m = len(valid_pairs)
    if m == 0:
        raise RuntimeError("No random pair had a reachable baseline (diam=0) path.")
    print(f"Randomize mode: using {m} valid pairs out of requested {n_pairs} (baseline reachable).")

    import math as _math
    rows = []
    summary = []

    for d in sorted(diameters):
        if d == 0:
            cover = 1.0
            eff_ok = 1.0
            med_ratio = 1.0
            mean_ratio = 1.0
            summary.append((d, cover, eff_ok, med_ratio, mean_ratio))
            continue

        g_d = infl[d]
        ok_count = 0
        ok_and_eff_count = 0
        ratios_d = []
        for k, (s,g) in enumerate(valid_pairs):
            if g_d[s[1], s[0]] == 1 or g_d[g[1], g[0]] == 1:
                ratios_d.append(_math.inf)
                if csv_path is not None:
                    rows.append([k, d, "blocked", _math.inf])
                continue
            p, c, why = astar(g_d, s, g, allow_diag=allow_diag)
            if p is None:
                ratios_d.append(_math.inf)
                if csv_path is not None:
                    rows.append([k, d, "no_path", _math.inf])
            else:
                ok_count += 1
                r = c / base_costs[k]
                ratios_d.append(r)
                if r <= target_ratio:
                    ok_and_eff_count += 1
                if csv_path is not None:
                    rows.append([k, d, "ok", c])

        cover = ok_count / m
        eff_ok = ok_and_eff_count / m
        finite_ratios = [r for r in ratios_d if np.isfinite(r)]
        med_ratio = float(np.median(finite_ratios)) if finite_ratios else _math.inf
        mean_ratio = float(np.mean(finite_ratios)) if finite_ratios else _math.inf
        summary.append((d, cover, eff_ok, med_ratio, mean_ratio))
        print(f"[d={d:>2}] coverage={cover*100:5.1f}%,"
              f" efficiency_ok@{target_ratio:.2f}x={eff_ok*100:5.1f}%"
              f"  median_ratio={med_ratio:.3f}, mean_ratio={mean_ratio:.3f}")

    candidates = [t for t in summary if (t[1] >= target_cov and t[2] >= target_cov)]
    if candidates:
        best = max(candidates, key=lambda t: t[0])
        print(f"\nRecommended diameter = {best[0]} (meets coverage≥{target_cov:.0%},"
              f" and efficiency_ok≥{target_cov:.0%} @ ratio≤{target_ratio:.2f}x)")
    else:
        print("\nNo diameter met both targets. Consider relaxing thresholds or the map is too tight.")

    if csv_path is not None:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_index", "diam", "status", "path_len_cells_or_inf"])
            for row in rows:
                w.writerow(row)
        print(f"Wrote detailed per-pair results to {csv_path}")

    return summary

# -------------------- CLI --------------------
def parse_xy(s: str):
    x,y = s.split(","); return (int(x), int(y))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Path to occupancy grid text")
    ap.add_argument("--start", type=parse_xy, help="start x,y")
    ap.add_argument("--goal",  type=parse_xy, help="goal x,y")
    ap.add_argument("--diam", type=int, default=0, help="robot diameter (cells)")
    ap.add_argument("--out", default="path.png", help="output PNG")
    ap.add_argument("--invert", action="store_true", help="flip semantics (1=free -> 1=obstacle)")
    ap.add_argument("--thresh", type=float, default=0.5, help="threshold for numeric grids")
    ap.add_argument("--zoom", action="store_true", help="zoom view to path bbox")
    ap.add_argument("--autosnap", action="store_true", help="snap start/goal to nearest free cell")
    ap.add_argument("--cell_size", type=float, default=None, help="meters per cell (prints metric length if set)")
    ap.add_argument("--no_diag", action="store_true", help="disallow diagonal motion (4-connected)")

    # Live animation
    ap.add_argument("--live", action="store_true", help="live animate Phase 1 (A* exploration)")
    ap.add_argument("--live_every", type=int, default=300, help="frames: yield every N expansions")
    ap.add_argument("--animate_out", default=None, help="optional output animation (.gif or .mp4)")
    ap.add_argument("--auto_close", type=float, default=None,
                    help="seconds to keep the live window open after finishing")

    # Downsampling
    ap.add_argument("--downsample", type=int, default=1,
                    help="speed hack: coarsen grid by this factor (>=1). Approximate lengths.")

    # ---- Randomize-mode options ----
    ap.add_argument("--randomize", action="store_true",
                    help="run randomized trials to choose optimal diameter (Phase 2)")
    ap.add_argument("--pairs", type=int, default=200, help="number of random start/goal pairs")
    ap.add_argument("--diam_list", default="0:20:2",
                    help="diameters to test (e.g., '0,2,4,6,8' or '0:20:2'); MUST include 0")
    ap.add_argument("--target_coverage", type=float, default=0.80,
                    help="required fraction of pairs with a path (e.g., 0.80)")
    ap.add_argument("--target_ratio", type=float, default=1.50,
                    help="required max path length factor vs diam=0 for 'efficient' (e.g., 1.50)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--csv", help="write per-pair randomize-mode results to CSV")

    # Path export
    ap.add_argument("--save_path_csv", help="write the resulting path (x,y per line) to CSV")

    args = ap.parse_args()

    grid = load_grid(args.grid, invert=args.invert, thresh=args.thresh)

    # Downsample (applies to both modes). For Phase 1, we also scale inputs.
    if args.downsample and args.downsample > 1:
        ds = args.downsample
        print(f"Downsampling grid by factor {ds} (approximate).")
        grid = downsample_grid(grid, ds)
    grid_stats(grid)

    allow_diag = not args.no_diag

    # -------- Randomize mode --------
    if args.randomize:
        if args.downsample and args.downsample > 1:
            print("Note: diameters in --diam_list are interpreted in downsampled cells.")
        diameters = parse_diam_list(args.diam_list)
        if 0 not in diameters:
            diameters = [0] + diameters
        print(f"Testing diameters: {diameters}")
        summary = randomize(grid, diameters, args.pairs, allow_diag,
                            args.seed, args.target_coverage, args.target_ratio,
                            csv_path=args.csv)
        print("\nSummary (diam, coverage, efficiency_ok, median_ratio, mean_ratio):")
        for d, cov, eff, medr, meanr in summary:
            print(f"  {d:>3}   {cov:5.2f}   {eff:5.2f}   {medr:7.3f}   {meanr:7.3f}")
        return

    # -------- Phase 1 mode (original spec) --------
    # If downsampled, scale user-provided start/goal/diam into coarse grid.
    if args.start is None:
        args.start = parse_xy(input("Enter start x,y: ").strip())
    if args.goal is None:
        args.goal  = parse_xy(input("Enter goal x,y: ").strip())

    if args.downsample and args.downsample > 1:
        ds = args.downsample
        args.start = (args.start[0]//ds, args.start[1]//ds)
        args.goal  = (args.goal[0]//ds,  args.goal[1]//ds)
        args.diam  = int(math.ceil(args.diam / ds))

    inflated = inflate_obstacles(grid, args.diam)

    s, g = args.start, args.goal
    if args.autosnap:
        s2 = snap_to_free(inflated, s)
        g2 = snap_to_free(inflated, g)
        if s2 is None or g2 is None:
            print("Could not find free cells near start/goal within radius. Try smaller diameter.")
            save_plot(inflated, None, args.out, title=f"No path (diam={args.diam})")
            sys.exit(1)
        if s2 != s or g2 != g:
            print(f"Snapped start {s} -> {s2}, goal {g} -> {g2}")
        s, g = s2, g2

    if args.live:
        res = live_plot(inflated, s, g, allow_diag=allow_diag, yield_every=args.live_every,
                        title=f"A* live (diam={args.diam}, {'8' if allow_diag else '4'}-conn)",
                        out_anim=args.animate_out, auto_close_secs=args.auto_close)
        path, cost, why = res["path"], res["cost"], res["why"]
    else:
        path, cost, why = astar(inflated, s, g, allow_diag=allow_diag)

    if path is None:
        print(f"No path (reason: {why}); diam={args.diam}")
        save_plot(inflated, None, args.out, title=f"No path (diam={args.diam})")
        print(f"Saved {args.out}")
        sys.exit(1)

    print(f"Path length (cells) = {cost:.3f}  with diameter={args.diam}  "
          f"({'8-connected' if allow_diag else '4-connected'})")
    if args.cell_size:
        # scale by downsample factor to estimate metric length in original resolution
        ds = args.downsample if (args.downsample and args.downsample > 1) else 1
        meters = cost * args.cell_size * ds
        if ds > 1:
            print(f"Path length (meters, approx with downsample={ds}) = {meters:.3f}")
        else:
            print(f"Path length (meters) = {meters:.3f}")

    if args.save_path_csv and path:
        import csv
        with open(args.save_path_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x","y"])
            for x,y in path:
                w.writerow([x,y])
        print(f"Wrote path to {args.save_path_csv}")

    save_plot(inflated, path, args.out,
              title=f"Shortest path (diam={args.diam})", zoom=args.zoom)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
