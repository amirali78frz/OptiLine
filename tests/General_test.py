"""
General_test.py
===============
Complete test and demonstration script for the OptiLine package.
Covers: utils, KinematicProfs, and solvers modules (opt_mintime excluded).

Stages
------
  Stage 1 : Track import and preprocessing
  Stage 2 : Spline computations  (utils)
  Stage 3 : Minimum-curvature raceline  (solvers.opt_min_curv)
  Stage 4 : Shortest-path raceline  (solvers.ShortestPath)
  Stage 5 : Optimal-shortest-path raceline  (solvers.OSP)
  Stage 6 : Kinematic profiles  (KinematicProfs)
  Stage 7 : Full optimization comparison – ZO vs CMA-ES  (Opt_min_CurvTime.Comparison)
  Stage 8 : Re-optimization on a refined reference track

Run from the tests/ directory:
    cd tests && python General_test.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Package imports
# ---------------------------------------------------------------------------
from OptiLine import utils
from OptiLine import solvers
from OptiLine.KinematicProfs import (
    calc_vel_profile,
    calc_ax_profile,
    calc_t_profile,
    curvature_profile,
    curvature_profile2,
    cumulative_distances,
)
from OptiLine.utils import (
    calc_splines,
    calc_spline_lengths,
    interp_splines,
    create_raceline,
    calc_head_curv_an,
    calc_head_curv_num,
    import_veh_dyn_info,
)

# ===========================================================================
# Helper utilities
# ===========================================================================

def print_stage(number, title):
    """Print a clearly visible stage header."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Stage {number}: {title}")
    print(sep)


def subsample_track(reftrack_full, step=4):
    """
    Return every `step`-th row of reftrack_full, always including the last row,
    to reduce the problem size while keeping the overall track shape.
    """
    nn = reftrack_full.shape[0]
    indices = np.arange(0, nn, step)
    indices = np.unique(np.concatenate([indices, [nn - 1]]))
    return reftrack_full[indices]


# ===========================================================================
# Stage 1 – Track import and preprocessing
# ===========================================================================
print_stage(1, "Track import and preprocessing")


# Change the below paths if your map files are located elsewhere. The script expects:
# - A centerline CSV with columns [x, y, w_right, w_left]
# - A GGV table CSV for vehicle dynamics
# - An ax_max_machines CSV for maximum acceleration limits
MAP_PATH = "maps/Melbourne/Melbourne_centerline.csv"
GGV_PATH = "maps/ggv.csv"
AX_MAX_PATH = "maps/ax_max_machines.csv"

csv_data = np.loadtxt(MAP_PATH, comments='#', delimiter=',')
reftrack_full = csv_data[:, 0:4]    # columns: [x, y, w_right, w_left]

# Sub-sample to reduce computation time while preserving shape
reftrack = subsample_track(reftrack_full, step=4)
print(f"Full track points : {reftrack_full.shape[0]}")
print(f"Subsampled points : {reftrack.shape[0]}")

# Compute initial segment lengths (Euclidean distances between consecutive points)
lengths = np.sqrt(np.sum(np.diff(reftrack[:, 0:2], axis=0) ** 2, axis=1))
lengths = np.append(lengths, lengths[0])   # close the loop
ds_0 = lengths

print(f"Segment length range : [{ds_0.min():.3f}, {ds_0.max():.3f}] m")

# Import vehicle dynamics tables (GGV and ax_max_machines)
ggv, ax_max_machines = import_veh_dyn_info(
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
)
print(f"GGV table shape       : {ggv.shape}")
print(f"ax_max_machines shape : {ax_max_machines.shape}")


# ===========================================================================
# Stage 2 – Spline computations (utils)
# ===========================================================================
print_stage(2, "Spline computations (utils)")

# --- 2a. calc_splines: fit cubic splines through the reference track points ---
coeffs_x, coeffs_y, M, normvec_norm = calc_splines(
    path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])),  # closed path
    el_lengths=ds_0,
    use_dist_scaling=True,
)
print(f"Spline coeff matrices : coeffs_x {coeffs_x.shape}, coeffs_y {coeffs_y.shape}")
print(f"System matrix M       : {M.shape}")
print(f"Normal vectors        : {normvec_norm.shape}")

# --- 2b. calc_spline_lengths: verify recovered segment lengths are consistent ---
spline_lengths = calc_spline_lengths(coeffs_x, coeffs_y, no_interp_points=20)
length_error = np.abs(spline_lengths - ds_0).max()
print(f"Max spline-length error vs ds_0 : {length_error:.4f} m")

# --- 2c. interp_splines: interpolate along the fitted splines ---
path_interp, spline_inds, t_vals, s_pts = interp_splines(
    coeffs_x=coeffs_x,
    coeffs_y=coeffs_y,
    spline_lengths=spline_lengths,
    stepsize_approx=2.0,
    incl_last_point=False,
)
print(f"Interpolated points   : {path_interp.shape[0]}  (stepsize ≈ 2.0 m)")

# --- 2d. calc_head_curv_an: analytical heading and curvature ---
psi_an, kappa_an = calc_head_curv_an(
    coeffs_x=coeffs_x,
    coeffs_y=coeffs_y,
    ind_spls=spline_inds,
    t_spls=t_vals,
)
print(f"Curvature range (analytical): [{kappa_an.min():.4f}, {kappa_an.max():.4f}] rad/m")

# --- 2e. calc_head_curv_num: numerical heading and curvature (cross-check) ---
el_path = np.hypot(
    np.diff(np.r_[path_interp[:, 0], path_interp[0, 0]]),
    np.diff(np.r_[path_interp[:, 1], path_interp[0, 1]]),
)
psi_num, kappa_num = calc_head_curv_num(
    path=path_interp,
    el_lengths=el_path,
    is_closed=True,
)
print(f"Curvature range (numerical) : [{kappa_num.min():.4f}, {kappa_num.max():.4f}] rad/m")


# --- 2g. cumulative_distances ---
el_lengths_an = calc_spline_lengths(coeffs_x, coeffs_y)
s_an = cumulative_distances(el_lengths_an)
print(f"Track length (spline sum) : {s_an[-1]:.1f} m")


# ===========================================================================
# Stage 3 – Minimum-curvature raceline (solvers.opt_min_curv)
# ===========================================================================
print_stage(3, "Minimum-curvature raceline (opt_min_curv)")

t_start = time.time()

alpha_mincurv, curv_error = solvers.opt_min_curv(
    reftrack=reftrack,
    normvectors=normvec_norm,
    A=M,
    kappa_bound=0.5,
    w_veh=1.0,
    print_debug=True,
    plot_debug=False,
    closed=True,
)

t_elapsed = time.time() - t_start
print(f"opt_min_curv solved in {t_elapsed:.3f} s")
print(f"alpha range     : [{alpha_mincurv.min():.4f}, {alpha_mincurv.max():.4f}] m")
print(f"Max curv. error : {curv_error:.6f} rad/m")

# Build the min-curvature raceline
raceline_mc, a_mc, cx_mc, cy_mc, si_mc, tv_mc, sp_mc, sl_mc, el_mc = create_raceline(
    refline=reftrack[:, :2],
    normvectors=normvec_norm,
    alpha=alpha_mincurv,
    stepsize_interp=2.0,
)
print(f"Min-curv raceline points : {raceline_mc.shape[0]}")

# Heading and curvature for min-curv raceline
psi_mc, kappa_mc = calc_head_curv_an(coeffs_x=cx_mc, coeffs_y=cy_mc,
                                      ind_spls=si_mc, t_spls=tv_mc)
print(f"Min-curv curvature range : [{kappa_mc.min():.4f}, {kappa_mc.max():.4f}] rad/m")

# Velocity and time profile for min-curv raceline
vx_mc = calc_vel_profile(
    ggv=ggv, ax_max_machines=ax_max_machines, v_max=22.88,
    kappa=kappa_mc, el_lengths=el_mc, closed=True,
    filt_window=3, dyn_model_exp=1.0, drag_coeff=0.75, m_veh=1000,
)
ax_mc = calc_ax_profile(
    vx_profile=np.append(vx_mc, vx_mc[0]),
    el_lengths=el_mc,
    eq_length_output=False,
)
t_mc = calc_t_profile(vx_profile=vx_mc, ax_profile=ax_mc, el_lengths=el_mc)
print(f"Min-curv estimated laptime : {t_mc[-1]:.2f} s")


# ===========================================================================
# Stage 4 – Shortest-path raceline (solvers.ShortestPath)
# ===========================================================================
print_stage(4, "Shortest-path raceline (ShortestPath)")

t_start = time.time()

(raceline_sp, alpha_sp, s_sp,
 vx_sp, ax_sp, kappa_sp, t_sp) = solvers.ShortestPath(
    reftrack=reftrack,
    w_veh=1.0,
    stepsize=2.0,
    plot=False,
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
)

t_elapsed = time.time() - t_start
print(f"ShortestPath solved in {t_elapsed:.3f} s")
print(f"Shortest-path raceline points  : {raceline_sp.shape[0]}")
print(f"Shortest-path estimated laptime: {t_sp[-1]:.2f} s")



# ===========================================================================
# Stage 5 – Kinematic profiles (KinematicProfs)
# ===========================================================================
print_stage(5, "Kinematic profiles (KinematicProfs)")

# Use the min-curvature results from Stage 3 for profile demonstration
s_splines_mc = cumulative_distances(el_mc)

print(f"Track length (min-curv) : {s_splines_mc[-1]:.1f} m")
print(f"Velocity profile range  : [{vx_mc.min():.2f}, {vx_mc.max():.2f}] m/s")
print(f"Accel. profile range    : [{ax_mc.min():.2f}, {ax_mc.max():.2f}] m/s²")
print(f"Lap time profile length : {t_mc.shape[0]} samples, final = {t_mc[-1]:.2f} s")

plt.figure(figsize=(14, 5))
plt.suptitle("Stage 6 – Kinematic profiles (min-curvature raceline)")
plt.subplot(1, 3, 1)
plt.plot(s_splines_mc, vx_mc)
plt.xlabel("s [m]"); plt.ylabel("v [m/s]"); plt.title("Velocity profile"); plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(s_splines_mc, ax_mc)
plt.xlabel("s [m]"); plt.ylabel("ax [m/s²]"); plt.title("Acceleration profile"); plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(s_splines_mc, kappa_mc)
plt.xlabel("s [m]"); plt.ylabel("κ [rad/m]"); plt.title("Curvature profile"); plt.grid(True)
plt.tight_layout()
plt.show()


# ===========================================================================
# Stage 6 – Full optimization comparison: ZO vs CMA-ES
#           (solvers.Opt_min_CurvTime.Comparison)
# ===========================================================================
print_stage(6, "Full optimization comparison – ZO vs CMA-ES (Opt_min_CurvTime)")

# Instantiate the combined curvature+time optimizer.
# iterations_ZO / iterations_CMA are kept small here so the script runs in
# a reasonable time; increase them for production-quality results.
t_start = time.time()

optct = solvers.Opt_min_CurvTime(
    reftrack=reftrack,
    center=reftrack,         # reference centerline (same as reftrack here)
    mu=0.01,
    h=0.001,
    kapb=0.5,                # curvature bound [rad/m]
    sfty=1.0,                # half vehicle width [m]
    si=2,                    # interpolation stepsize [m]
    vm=22.88,                # maximum velocity [m/s]
    m_veh=1000,              # vehicle mass [kg]
    drag_coeff=0.75,         # aerodynamic drag coefficient
    MC=1,                    # Monte Carlo repetitions
    min_s=0.5,               # minimum spline segment length [m]
    max_s=2.0,               # maximum spline segment length [m]
    sigma=0.005,             # initial CMA-ES covariance
    iterations_ZO=100,       # ZO optimizer iterations (increase for better results)
    iterations_CMA=10,       # CMA-ES optimizer iterations
    popsize=16,              # CMA-ES population size
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
    fw=3,                    # velocity filter window length
)

# Run objective function once with the initial segment lengths to sanity-check
laptime_initial = optct.f_t(ds_0)
print(f"Initial lap time (f_t with ds_0) : {laptime_initial:.2f} s")

# --- 7a: generate_raceline – geometry only, using ZO-optimized lengths ---
t0 = time.time()
raceline_zo, ds_zo = optct.generate_raceline(solver='ZO')
print(f"generate_raceline (ZO) done in {time.time()-t0:.2f} s, "
      f"points: {raceline_zo.shape[0]}")

# --- 7b: generate_kinProfs – full profiles using ZO-optimized lengths ---
t0 = time.time()
s_zo, vx_zo, ax_zo, kappa_zo, t_zo, rl_zo = optct.generate_kinProfs(ds=ds_zo)
print(f"generate_kinProfs (ZO) done in {time.time()-t0:.2f} s, "
      f"laptime: {t_zo[-1]:.2f} s")

# --- 7c: full Comparison (ZO + CMA-ES + initial + centerline) ---
t0 = time.time()
s_splines, vx_opt, ax_opt, kappa_opt, t_opt, rl_opt = optct.Comparison(
    plot='Y',
    output='ZO',
)
print(f"Comparison done in {time.time()-t0:.2f} s  |  ZO laptime: {t_opt[-1]:.2f} s")

t_elapsed = time.time() - t_start
print(f"Stage 7 total time : {t_elapsed:.1f} s")

# Save ZO results for potential re-use in Stage 7
np.save("v_opt.npy", vx_opt)
np.save("kappa_opt.npy", kappa_opt)
np.save("s_splines.npy", s_splines)


# ===========================================================================
# Stage 7 – Re-optimization on a refined reference track
#
# Workflow:
#   1. Use the ZO-optimized raceline from Stage 7 as the new reference track.
#   2. Sub-sample it to produce a denser control-point set than the original.
#   3. Re-run Opt_min_CurvTime on the refined track.
#   4. Compare the re-optimized lap time with the first-pass ZO result.
#
# This mirrors the real-world use-case of iterating optimization passes:
# the first pass finds a coarse raceline; subsequent passes refine it on a
# finer track representation, which can uncover additional lap-time savings.
# ===========================================================================
print_stage(7, "Re-optimization on refined reference track")

# --- Build refined reference track from the ZO raceline ---
# Interpolate original track widths onto the denser ZO raceline arc-length grid
f_wr = interp1d(np.linspace(0, 1, reftrack.shape[0]),
                reftrack[:, 2], kind='linear', fill_value='extrapolate')
f_wl = interp1d(np.linspace(0, 1, reftrack.shape[0]),
                reftrack[:, 3], kind='linear', fill_value='extrapolate')
s_norm_zo = np.linspace(0, 1, rl_opt.shape[0])
wr_zo = f_wr(s_norm_zo)
wl_zo = f_wl(s_norm_zo)

# Compose the new reference track: [x, y, w_right, w_left]
reftrack_new_full = np.column_stack([rl_opt, wr_zo, wl_zo])

# Sub-sample to keep computation manageable while providing finer resolution
reftrack_new = subsample_track(reftrack_new_full, step=2)
print(f"Refined reftrack points : {reftrack_new.shape[0]}  "
      f"(was {reftrack.shape[0]} in first pass)")

# New initial segment lengths for the refined track
lengths_new = np.hypot(np.diff(reftrack_new[:, 0]), np.diff(reftrack_new[:, 1]))
lengths_new = np.append(lengths_new, lengths_new[0])
print(f"Refined segment length range : [{lengths_new.min():.3f}, {lengths_new.max():.3f}] m")

# --- Instantiate the re-optimizer on the refined track ---
t_start = time.time()

optct_reopt = solvers.Opt_min_CurvTime(
    reftrack=reftrack_new,
    center=reftrack_new,
    mu=0.01,
    h=0.001,
    kapb=0.5,
    sfty=1.0,
    si=2,
    vm=22.88,
    m_veh=1000,
    drag_coeff=0.75,
    MC=1,
    min_s=0.5,
    max_s=2.0,
    sigma=0.005,
    iterations_ZO=100,       # increase for a thorough re-optimization run
    iterations_CMA=10,
    popsize=16,
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
    fw=3,
)

# Run ZO on the refined track
ds_reopt = optct_reopt.CurveLenOpt(solver='ZO')
s_ro, vx_ro, ax_ro, kappa_ro, t_ro, rl_ro = optct_reopt.generate_kinProfs(ds=ds_reopt)

t_elapsed = time.time() - t_start
print(f"Re-optimization done in {t_elapsed:.1f} s")
print(f"First-pass ZO laptime  : {t_opt[-1]:.2f} s")
print(f"Re-optimized laptime   : {t_ro[-1]:.2f} s")
improvement = t_opt[-1] - t_ro[-1]
print(f"Lap-time improvement   : {improvement:+.2f} s "
      f"({'better' if improvement > 0 else 'worse or equal'})")

# Comparison plot: first-pass vs re-optimized raceline and velocity
plt.figure(figsize=(14, 5))
plt.suptitle("Stage 8 – Re-optimization comparison")

plt.subplot(1, 2, 1)
plt.plot(rl_opt[:, 0], rl_opt[:, 1], 'r-',  label=f'ZO pass 1  ({t_opt[-1]:.1f} s)')
plt.plot(rl_ro[:, 0],  rl_ro[:, 1],  'b--', label=f'ZO pass 2  ({t_ro[-1]:.1f} s)')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title('Racelines'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(s_zo, vx_zo, 'r-',  label='ZO pass 1')
plt.plot(s_ro, vx_ro, 'b--', label='ZO pass 2')
plt.xlabel('s [m]'); plt.ylabel('v [m/s]')
plt.title('Velocity profiles'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("  Test summary")
print("=" * 60)
print(f"  Shortest-path laptime  : {t_sp[-1]:.2f} s")
print(f"  Min-curv laptime       : {t_mc[-1]:.2f} s")
print(f"  ZO (pass 1) laptime    : {t_opt[-1]:.2f} s")
print(f"  ZO (pass 2) laptime    : {t_ro[-1]:.2f} s")
print("=" * 60)
print("All stages completed successfully.")
