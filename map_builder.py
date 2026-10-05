"""
map_builder.py
==============
Generates random race-track maps and saves them in the same three-file
structure used by the 25 real-circuit maps shipped with OptiLine::

    maps/<name>/
        <name>_centerline.csv   # x_m, y_m, w_tr_right_m, w_tr_left_m
        <name>_map.png          # 2000×2000 greyscale occupancy-grid image
        <name>_map.yaml         # ROS-style map metadata

Typical usage (class API)
-------------------------
>>> from OptiLine.map_builder import MapBuilder
>>> mb = MapBuilder(seed=42, maps_dir="maps/Zoo_Maps")
>>> mb.generate(n=10)                   # writes example_1 … example_10

>>> mb2 = MapBuilder(seed=0, variable_width=True)
>>> mb2.generate(n=5, start_index=26)   # extends an existing dataset

Single-track generation without file I/O
-----------------------------------------
>>> track = mb.generate_track(style="circuit")
>>> track.keys()   # {'xy', 'w_right', 'w_left', 'style'}

Functional API (unchanged for backward compatibility)
------------------------------------------------------
>>> from OptiLine.map_builder import generate_random_track, build_examples
"""

import os
import argparse
from collections import defaultdict

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
DEFAULT_HALF_WIDTH  = 1.10        # m  – uniform half-width for tracks
POINT_SPACING       = 0.40        # m  – target arc-length step between CSV rows
IMAGE_SIZE          = 2000        # px – output PNG resolution (square)
IMAGE_MARGIN_FRAC   = 0.12        # fraction of image used as padding on each side
OCCUPIED_THRESH     = 0.45
FREE_THRESH         = 0.196

# Occupancy values (ROS convention, greyscale 0-255)
PIXEL_FREE          = 255         # track surface
PIXEL_UNKNOWN       = 205         # off-track / unknown area
PIXEL_OCCUPIED      = 0           # track boundary walls

#: Named track archetypes available to the generator.
TRACK_STYLES = [
    "oval_complex",
    "circuit",
    "circuit_complex",
    "street_circuit",
    "tilodrome",
]

# ---------------------------------------------------------------------------
# Per-style configuration (internal)
# ---------------------------------------------------------------------------
_STYLE_CFG = {
    # n      – (lo,hi) polygon vertex count
    # asp    – (lo,hi) bounding-box aspect ratio (ellipse axes width/height)
    # hp/ch  – probability of hairpin / chicane on each polygon edge
    # plo/hi – push fraction for hairpin apex (fraction of mid-radius pushed in)
    # bmp    – chicane lateral bump fraction
    # mhp    – max hairpin count per track
    # stiff  – corner-stiffening distance from vertex (metres, pre-scale)
    # conc   – per-vertex probability of inward concavity
    "oval_complex":    dict(n=(4, 6),  asp=(2.2, 3.5), hp=0.35, ch=0.18,
                            plo=0.50, phi=0.75, bmp=0.12, mhp=2,
                            stiff=5.5, conc=0.15),
    "circuit":         dict(n=(5, 8),  asp=(1.4, 2.2), hp=0.30, ch=0.30,
                            plo=0.40, phi=0.68, bmp=0.14, mhp=4,
                            stiff=5.0, conc=0.25),
    "circuit_complex": dict(n=(6, 9),  asp=(1.1, 1.8), hp=0.35, ch=0.28,
                            plo=0.38, phi=0.62, bmp=0.12, mhp=5,
                            stiff=4.5, conc=0.30),
    "street_circuit":  dict(n=(6, 10), asp=(1.0, 1.5), hp=0.08, ch=0.50,
                            plo=0.25, phi=0.48, bmp=0.20, mhp=1,
                            stiff=3.5, conc=0.38),
    "tilodrome":       dict(n=(5, 7),  asp=(1.5, 2.6), hp=0.30, ch=0.20,
                            plo=0.44, phi=0.72, bmp=0.10, mhp=3,
                            stiff=6.5, conc=0.20),
}


# ===========================================================================
#  Low-level geometry helpers
# ===========================================================================

def _arc_lengths(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length vector for an (N, 2) array of 2-D points."""
    diffs = np.diff(pts, axis=0)
    seg   = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_at_spacing(pts: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a closed 2-D polyline so consecutive points are ~*spacing* m
    apart (arc-length parametrisation).  Returns an open loop (first ≠ last).
    """
    pts_c  = np.vstack([pts, pts[0]])
    s      = _arc_lengths(pts_c)
    total  = s[-1]
    n_pts  = max(4, int(round(total / spacing)))
    s_uni  = np.linspace(0, total, n_pts + 1)[:-1]
    x_rs   = np.interp(s_uni, s, pts_c[:, 0])
    y_rs   = np.interp(s_uni, s, pts_c[:, 1])
    return np.column_stack([x_rs, y_rs])


def _smooth_widths(n: int, w_min: float, w_max: float,
                   rng: np.random.Generator,
                   n_harmonics: int = 4) -> np.ndarray:
    """
    Generate a smooth, periodic width profile with values in [w_min, w_max].
    Built from a sum of low-frequency Fourier components.
    """
    t       = np.linspace(0, 2 * np.pi, n, endpoint=False)
    profile = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        amp   = rng.uniform(0, 1.0 / k)
        phase = rng.uniform(0, 2 * np.pi)
        profile += amp * np.cos(k * t + phase)
    profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-12)
    return w_min + (w_max - w_min) * profile


# ===========================================================================
#  Track generator: polygon backbone → corner stiffening → spline → check
# ===========================================================================

def _make_polygon(rng: np.random.Generator,
                  n: int, a: float, b: float,
                  concavity_prob: float = 0.25) -> np.ndarray:
    """
    Return *n* vertices of an irregular polygon inscribed in an a×b ellipse.

    * Angular spacing between vertices is jittered ±35 % of even spacing.
    * Each vertex radius varies ±40 % of the base ellipse radius.
    * With probability *concavity_prob*, a vertex is pushed further inward
      (35–65 % of base radius), creating concavities that mimic real circuits.
    """
    phi  = rng.uniform(0.0, 2.0 * np.pi)
    angs = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phi
    angs += rng.uniform(-0.35, 0.35, n) * (2.0 * np.pi / n)
    angs  = np.sort(angs)

    r = 1.0 + rng.uniform(-0.38, 0.38, n)
    for i in range(n):
        if rng.random() < concavity_prob:
            r[i] *= rng.uniform(0.35, 0.65)

    return np.column_stack([a * r * np.cos(angs), b * r * np.sin(angs)])


def _hairpin_pts(pt1: np.ndarray, pt2: np.ndarray,
                 rng: np.random.Generator,
                 push_lo: float, push_hi: float):
    """Two apex control points for a hairpin between *pt1* and *pt2*."""
    mid   = 0.5 * (pt1 + pt2)
    r_mid = np.linalg.norm(mid)
    if r_mid < 1.0:
        return None
    inward = -mid / r_mid
    apex   = mid + inward * r_mid * rng.uniform(push_lo, push_hi)
    elen   = np.linalg.norm(pt2 - pt1)
    hw     = min(elen * 0.07, 3.5)
    ehat   = (pt2 - pt1) / (elen + 1e-9)
    return np.array([apex - ehat * hw, apex + ehat * hw])


def _chicane_pts(pt1: np.ndarray, pt2: np.ndarray,
                 rng: np.random.Generator,
                 bump_frac: float):
    """Two laterally offset control points (S-chicane) between *pt1* and *pt2*."""
    edge  = pt2 - pt1
    elen  = np.linalg.norm(edge)
    if elen < 8.0:
        return None
    ehat  = edge / elen
    perp  = np.array([-ehat[1], ehat[0]])
    r_mid = np.linalg.norm(0.5 * (pt1 + pt2))
    bump  = r_mid * rng.uniform(0.05, bump_frac) * rng.choice([-1, 1])
    c1    = pt1 + ehat * elen * 0.33 + perp *  bump
    c2    = pt1 + ehat * elen * 0.67 + perp * -bump
    return np.array([c1, c2])


def _fit_periodic_spline(ctrl: np.ndarray):
    """
    Fit a periodic CubicSpline through *ctrl* (N, 2) using cumulative chord
    length.  Returns (cs_x, cs_y, total_t).
    """
    cp   = np.vstack([ctrl, ctrl[0]])
    d    = np.diff(cp, axis=0)
    chrd = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
    t    = np.concatenate([[0.0], np.cumsum(chrd)])
    cs_x = CubicSpline(t, cp[:, 0], bc_type="periodic")
    cs_y = CubicSpline(t, cp[:, 1], bc_type="periodic")
    return cs_x, cs_y, float(t[-1])


def _has_self_intersection(pts: np.ndarray, clearance: float = 2.8) -> bool:
    """
    Return True if any two non-adjacent track points are closer than
    *clearance* metres (grid-hash check, O(N) average).
    """
    n    = len(pts)
    skip = max(6, int(clearance / POINT_SPACING) + 3)
    xmin, ymin = pts.min(axis=0)
    c    = clearance
    grid = defaultdict(list)
    for i in range(n):
        gx = int((pts[i, 0] - xmin) / c)
        gy = int((pts[i, 1] - ymin) / c)
        grid[(gx, gy)].append(i)

    for i in range(n):
        gx = int((pts[i, 0] - xmin) / c)
        gy = int((pts[i, 1] - ymin) / c)
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                for j in grid[(gx + dgx, gy + dgy)]:
                    gap = abs(i - j)
                    if skip < gap < n - skip:
                        if np.hypot(pts[i, 0] - pts[j, 0],
                                    pts[i, 1] - pts[j, 1]) < clearance:
                            return True
    return False


def _try_build_track(rng: np.random.Generator,
                     cfg: dict,
                     target_length: float) -> np.ndarray:
    """
    Single attempt to build a track from *cfg*.
    Returns an (N, 2) array or None if validation fails.

    Pipeline
    --------
    1. Irregular polygon backbone with radial jitter and concavities.
    2. 3-point corner stiffening (enter → vertex → leave) at every vertex.
    3. Hairpin / chicane feature injection in straight sections.
    4. Periodic CubicSpline guarantees exact closure.
    5. Light Gaussian smoothing (σ ≈ 0.5 m) to suppress numerical oscillations.
    6. Self-intersection rejection via grid-hash check.
    """
    n   = int(rng.integers(cfg["n"][0], cfg["n"][1] + 1))
    asp = float(rng.uniform(*cfg["asp"]))
    R   = 50.0
    a, b = R * np.sqrt(asp), R / np.sqrt(asp)

    polygon = _make_polygon(rng, n, a, b, concavity_prob=cfg.get("conc", 0.25))
    stiff   = cfg.get("stiff", 5.0)

    ctrl = []
    n_hp = 0
    for i in range(n):
        p_prev = polygon[(i - 1) % n]
        p1     = polygon[i]
        p2     = polygon[(i + 1) % n]

        d_in  = p1 - p_prev;  l_in  = np.linalg.norm(d_in)
        d_out = p2 - p1;      l_out = np.linalg.norm(d_out)

        s_in  = min(stiff, l_in  * 0.42) if l_in  > 1e-6 else 0.0
        s_out = min(stiff, l_out * 0.42) if l_out > 1e-6 else 0.0

        if l_in  > 1e-6: ctrl.append(p1 - (d_in  / l_in)  * s_in)
        ctrl.append(p1)
        if l_out > 1e-6: ctrl.append(p1 + (d_out / l_out) * s_out)

        elen = float(l_out)
        rv   = float(rng.random())
        if rv < cfg["hp"] and n_hp < cfg["mhp"] and elen > 12.0:
            hp = _hairpin_pts(p1, p2, rng, cfg["plo"], cfg["phi"])
            if hp is not None:
                ctrl.extend(hp); n_hp += 1
        elif rv < cfg["hp"] + cfg["ch"] and elen > 12.0:
            ch = _chicane_pts(p1, p2, rng, cfg["bmp"])
            if ch is not None:
                ctrl.extend(ch)

    ctrl = np.array(ctrl)
    if len(ctrl) < 5:
        return None

    try:
        cs_x, cs_y, total_t = _fit_periodic_spline(ctrl)
    except Exception:
        return None

    t_d   = np.linspace(0.0, total_t, 4000, endpoint=False)
    dense = np.column_stack([cs_x(t_d), cs_y(t_d)])

    approx = _arc_lengths(np.vstack([dense, dense[0]]))[-1]
    if approx < 50.0:
        return None
    ctrl = ctrl * (target_length / approx)

    try:
        cs_x, cs_y, total_t = _fit_periodic_spline(ctrl)
    except Exception:
        return None

    t_d   = np.linspace(0.0, total_t, 4000, endpoint=False)
    dense = np.column_stack([cs_x(t_d), cs_y(t_d)])

    sig = max(1, int(0.5 / POINT_SPACING))
    dense[:, 0] = gaussian_filter1d(dense[:, 0], sigma=sig, mode="wrap")
    dense[:, 1] = gaussian_filter1d(dense[:, 1], sigma=sig, mode="wrap")

    pts = _resample_at_spacing(dense, POINT_SPACING)
    pts = pts - pts[0]

    return None if _has_self_intersection(pts) else pts


# ===========================================================================
#  Rendering helpers
# ===========================================================================

def _world_to_pixel(xy_world: np.ndarray,
                    origin: np.ndarray,
                    resolution: float,
                    img_height: int) -> np.ndarray:
    """World metres → image pixel coordinates (ROS convention)."""
    px = (xy_world[:, 0] - origin[0]) / resolution
    py = img_height - (xy_world[:, 1] - origin[1]) / resolution
    return np.column_stack([px, py]).astype(int)


def render_map_png(xy: np.ndarray,
                   w_right: np.ndarray,
                   w_left: np.ndarray,
                   resolution: float,
                   origin: np.ndarray,
                   img_size: int = IMAGE_SIZE) -> Image.Image:
    """
    Render an occupancy-grid PNG for a closed track.

    Parameters
    ----------
    xy         : (N, 2) centrelinepoints in world metres
    w_right    : (N,) right half-widths in metres
    w_left     : (N,) left half-widths in metres
    resolution : metres per pixel
    origin     : [x_world, y_world] of the bottom-left pixel
    img_size   : output image side length in pixels

    Returns
    -------
    PIL.Image.Image
        Greyscale occupancy-grid image (255 = free, 0 = wall, 205 = unknown).
    """
    dp      = np.vstack([np.diff(xy, axis=0), xy[0] - xy[-1]])
    seg_len = np.hypot(dp[:, 0], dp[:, 1]) + 1e-12
    nx      = -dp[:, 1] / seg_len
    ny      =  dp[:, 0] / seg_len

    right_edge = xy - np.column_stack([nx * w_right, ny * w_right])
    left_edge  = xy + np.column_stack([nx * w_left,  ny * w_left])

    poly_world = np.vstack([left_edge, right_edge[::-1]])
    poly_px    = _world_to_pixel(poly_world, origin, resolution, img_size)

    img  = Image.new("L", (img_size, img_size), PIXEL_UNKNOWN)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(p) for p in poly_px], fill=PIXEL_FREE)

    left_px  = _world_to_pixel(np.vstack([left_edge,  left_edge[0:1]]),
                                origin, resolution, img_size)
    right_px = _world_to_pixel(np.vstack([right_edge, right_edge[0:1]]),
                                origin, resolution, img_size)
    draw.line([tuple(p) for p in left_px],  fill=PIXEL_OCCUPIED, width=2)
    draw.line([tuple(p) for p in right_px], fill=PIXEL_OCCUPIED, width=2)
    return img


# ===========================================================================
#  File writers
# ===========================================================================

def write_centerline_csv(path: str,
                         xy: np.ndarray,
                         w_right: np.ndarray,
                         w_left: np.ndarray) -> None:
    """Write the centrelineCSV in the standard OptiLine format."""
    rows = np.column_stack([xy, w_right, w_left])
    rows[0, :2] = 0.0          # force exact (0, 0) origin
    with open(path, "w") as fh:
        fh.write("# x_m, y_m, w_tr_right_m, w_tr_left_m\n")
        for row in rows:
            fh.write(f"{row[0]}, {row[1]}, {row[2]}, {row[3]}\n")


def write_map_yaml(path: str,
                   png_name: str,
                   resolution: float,
                   origin: np.ndarray) -> None:
    """Write the ROS-style map YAML file."""
    with open(path, "w") as fh:
        fh.write(f"image: {png_name}\n")
        fh.write(f"resolution: {resolution:.5f}\n")
        fh.write(f"origin: [{origin[0]:.15f},{origin[1]:.15f}, 0.000000]\n")
        fh.write("negate: 0\n")
        fh.write(f"occupied_thresh: {OCCUPIED_THRESH}\n")
        fh.write(f"free_thresh: {FREE_THRESH}\n")


# ===========================================================================
#  Functional public API
# ===========================================================================

def generate_random_track(
    rng: np.random.Generator,
    target_length: float = 450.0,
    style: str = None,
    variable_width: bool = False,
    half_width_uniform: float = DEFAULT_HALF_WIDTH,
    w_min: float = 0.8,
    w_max: float = 3.0,
) -> dict:
    """
    Generate a single random closed race-track (no file I/O).

    Parameters
    ----------
    rng                : NumPy ``Generator`` (e.g. ``np.random.default_rng(42)``)
    target_length      : desired total track length in metres
    style              : one of :data:`TRACK_STYLES`, or *None* to pick randomly
    variable_width     : if True, half-widths vary smoothly along the track
    half_width_uniform : uniform half-width (m) when *variable_width* is False
    w_min / w_max      : half-width bounds for variable-width mode

    Returns
    -------
    dict
        ``{'xy': (N,2), 'w_right': (N,), 'w_left': (N,), 'style': str}``
    """
    if style is None:
        style = str(rng.choice(TRACK_STYLES))
    cfg = _STYLE_CFG[style]

    pts = None
    for _ in range(30):
        pts = _try_build_track(rng, cfg, target_length)
        if pts is not None:
            break

    if pts is None:          # absolute fallback: simple oval
        t_fb = np.linspace(0.0, 2.0 * np.pi,
                           int(target_length / POINT_SPACING), endpoint=False)
        R_fb = target_length / (2.0 * np.pi)
        pts  = np.column_stack([R_fb * np.cos(t_fb), R_fb * np.sin(t_fb)])
        pts  = pts - pts[0]

    n_pts = len(pts)
    if variable_width:
        w_right = _smooth_widths(n_pts, w_min, w_max, rng)
        w_left  = _smooth_widths(n_pts, w_min, w_max, rng)
    else:
        w_right = np.full(n_pts, half_width_uniform)
        w_left  = np.full(n_pts, half_width_uniform)

    return {"xy": pts, "w_right": w_right, "w_left": w_left, "style": style}


def build_example(
    example_id: int,
    maps_dir: str,
    rng: np.random.Generator,
    variable_width: bool = False,
    verbose: bool = True,
) -> str:
    """
    Generate one random track and write its three files to *maps_dir*.

    Returns the path to the created folder.
    """
    name   = f"example_{example_id}"
    folder = os.path.join(maps_dir, name)
    os.makedirs(folder, exist_ok=True)

    track = generate_random_track(rng=rng, variable_width=variable_width)
    xy, w_right, w_left = track["xy"], track["w_right"], track["w_left"]

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    max_hw        = max(w_right.max(), w_left.max())
    x_min -= max_hw;  x_max += max_hw
    y_min -= max_hw;  y_max += max_hw

    span        = max(x_max - x_min, y_max - y_min)
    margin      = span * IMAGE_MARGIN_FRAC
    world_span  = span + 2 * margin
    origin      = np.array([x_min - margin, y_min - margin])
    resolution  = world_span / IMAGE_SIZE

    csv_path  = os.path.join(folder, f"{name}_centerline.csv")
    png_name  = f"{name}_map.png"
    png_path  = os.path.join(folder, png_name)
    yaml_path = os.path.join(folder, f"{name}_map.yaml")

    write_centerline_csv(csv_path, xy, w_right, w_left)
    render_map_png(xy, w_right, w_left, resolution, origin).save(png_path)
    write_map_yaml(yaml_path, png_name, resolution, origin)

    if verbose:
        total_len = _arc_lengths(np.vstack([xy, xy[0]]))[-1]
        print(
            f"  [{name}]  pts={len(xy):4d}  "
            f"length={total_len:5.0f}m  "
            f"extent=({x_max-x_min:.0f}×{y_max-y_min:.0f})m  "
            f"resolution={resolution:.5f}m/px  "
            f"widths={'variable' if variable_width else f'{w_right[0]:.2f}m uniform'}"
        )
    return folder


def build_examples(
    n: int = 5,
    maps_dir: str = None,
    seed: int = 42,
    variable_width: bool = False,
    start_index: int = 1,
) -> list:
    """
    Generate *n* random tracks and save them under *maps_dir*.

    Parameters
    ----------
    n              : number of examples to generate
    maps_dir       : target directory (auto-detected from project layout if None)
    seed           : random seed for reproducibility
    variable_width : smoothly varying half-widths if True
    start_index    : first example index (1 → ``example_1``, ``example_2``, …)

    Returns
    -------
    list of str
        Paths to the created example folders.
    """
    if maps_dir is None:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(script_dir))  # src/OptiLine → project root
        candidate   = os.path.join(project_dir, "maps", "Zoo_Maps")
        if os.path.isdir(candidate):
            maps_dir = candidate
        elif os.path.isdir("maps"):
            maps_dir = os.path.abspath("maps")
        else:
            raise FileNotFoundError(
                "Cannot locate 'maps/' directory. "
                "Pass maps_dir explicitly or run from the project root."
            )

    rng = np.random.default_rng(seed)
    print(f"\nGenerating {n} map(s) → {maps_dir}")
    print(f"  seed={seed}  variable_width={variable_width}\n")

    folders = []
    for i in range(n):
        folder = build_example(
            example_id=start_index + i,
            maps_dir=maps_dir,
            rng=rng,
            variable_width=variable_width,
        )
        folders.append(folder)

    print(f"\nDone. {n} example(s) written to {maps_dir}.")
    return folders


# ===========================================================================
#  MapBuilder class — stateful, object-oriented interface
# ===========================================================================

class MapBuilder:
    """
    Stateful interface to the OptiLine map-generation pipeline.

    Captures default settings (seed, output directory, width mode) so that
    repeated calls to :meth:`generate` or :meth:`generate_track` do not
    require re-specifying them.

    Parameters
    ----------
    seed           : master random seed (int); passed to
                     ``numpy.random.default_rng``.
    maps_dir       : path to the output directory where map folders are
                     created.  Auto-detected from the project layout when
                     *None* (looks for ``maps/Zoo_Maps/`` relative to the
                     package root).
    variable_width : if *True*, track half-widths vary smoothly along the
                     track (Fourier-based profile).
    target_length  : desired track length in metres (default 450 m).
    half_width     : uniform half-width in metres used when
                     *variable_width* is *False* (default 1.10 m).
    w_min / w_max  : half-width bounds for variable-width mode (metres).

    Examples
    --------
    >>> mb = MapBuilder(seed=42, maps_dir="maps/Zoo_Maps")
    >>> folders = mb.generate(n=10)           # writes example_1 … example_10
    >>> folders = mb.generate(n=5, start_index=11)  # extends the dataset

    >>> track = mb.generate_track(style="circuit")
    >>> xy, w_r, w_l = track["xy"], track["w_right"], track["w_left"]
    """

    def __init__(
        self,
        seed: int = 42,
        maps_dir: str = None,
        variable_width: bool = False,
        target_length: float = 450.0,
        half_width: float = DEFAULT_HALF_WIDTH,
        w_min: float = 0.8,
        w_max: float = 3.0,
    ):
        self.seed           = seed
        self.maps_dir       = maps_dir
        self.variable_width = variable_width
        self.target_length  = target_length
        self.half_width     = half_width
        self.w_min          = w_min
        self.w_max          = w_max
        self._rng           = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def reset(self, seed: int = None) -> None:
        """
        Reset the internal RNG.

        Parameters
        ----------
        seed : new seed; if *None*, the original seed passed to ``__init__``
               is reused, making the sequence fully reproducible.
        """
        self._rng = np.random.default_rng(
            seed if seed is not None else self.seed
        )

    # ------------------------------------------------------------------
    def generate_track(self, style: str = None) -> dict:
        """
        Generate a single random track without writing any files.

        Parameters
        ----------
        style : one of :data:`TRACK_STYLES` or *None* (chosen randomly).

        Returns
        -------
        dict
            ``{'xy': ndarray (N,2), 'w_right': ndarray (N,),
               'w_left': ndarray (N,), 'style': str}``
        """
        return generate_random_track(
            rng=self._rng,
            target_length=self.target_length,
            style=style,
            variable_width=self.variable_width,
            half_width_uniform=self.half_width,
            w_min=self.w_min,
            w_max=self.w_max,
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        n: int = 1,
        start_index: int = 1,
        maps_dir: str = None,
        verbose: bool = True,
    ) -> list:
        """
        Generate *n* random tracks and write their files to disk.

        Parameters
        ----------
        n           : number of maps to generate.
        start_index : index of the first map (``example_<start_index>``).
        maps_dir    : override the instance-level output directory.
        verbose     : print a one-line summary for each generated map.

        Returns
        -------
        list of str
            Absolute paths to the created example folders.
        """
        out_dir = maps_dir or self.maps_dir
        if out_dir is None:
            # Auto-detect project root: src/OptiLine/map_builder.py → project root
            here        = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(here))
            candidate   = os.path.join(project_dir, "maps", "Zoo_Maps")
            if os.path.isdir(candidate):
                out_dir = candidate
            elif os.path.isdir("maps"):
                out_dir = os.path.abspath("maps")
            else:
                raise FileNotFoundError(
                    "Cannot locate 'maps/' directory. "
                    "Set maps_dir in the constructor or pass it explicitly."
                )

        if verbose:
            print(f"\nGenerating {n} map(s) → {out_dir}")
            print(f"  seed={self.seed}  variable_width={self.variable_width}\n")

        folders = []
        for i in range(n):
            folder = build_example(
                example_id=start_index + i,
                maps_dir=out_dir,
                rng=self._rng,
                variable_width=self.variable_width,
                verbose=verbose,
            )
            folders.append(folder)

        if verbose:
            print(f"\nDone. {n} example(s) written to {out_dir}.")
        return folders

    # ------------------------------------------------------------------
    @staticmethod
    def render(xy: np.ndarray,
               w_right: np.ndarray,
               w_left: np.ndarray,
               resolution: float,
               origin: np.ndarray,
               img_size: int = IMAGE_SIZE) -> Image.Image:
        """
        Render an occupancy-grid PNG for a closed track.

        Thin static wrapper around the module-level :func:`render_map_png`.
        """
        return render_map_png(xy, w_right, w_left, resolution, origin, img_size)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"MapBuilder(seed={self.seed}, variable_width={self.variable_width}, "
            f"target_length={self.target_length}m, maps_dir={self.maps_dir!r})"
        )


# ===========================================================================
#  CLI entry-point (kept for direct script invocation)
# ===========================================================================

def _cli():
    parser = argparse.ArgumentParser(
        description="Generate random race-track maps for OptiLine."
    )
    parser.add_argument("--n",            type=int,  default=25,
                        help="Number of example maps to generate.")
    parser.add_argument("--seed",         type=int,  default=7,
                        help="Random seed for reproducibility.")
    parser.add_argument("--maps-dir",     type=str,  default=None,
                        help="Path to the output directory (auto-detected if omitted).")
    parser.add_argument("--variable-width", action="store_true",
                        help="Generate smoothly varying track widths.")
    parser.add_argument("--start-index",  type=int,  default=1,
                        help="Starting example index (default: 1).")
    args = parser.parse_args()

    mb = MapBuilder(
        seed=args.seed,
        maps_dir=args.maps_dir,
        variable_width=args.variable_width,
    )
    mb.generate(n=args.n, start_index=args.start_index)


if __name__ == "__main__":
    _cli()
