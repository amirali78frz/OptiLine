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
  Stage 6 : Full optimization comparison – ZO vs CMA-ES  (Opt_min_CurvTime.Comparison)
  Stage 7 : Re-optimization on a refined reference track
  Stage 8 : Blackbox_raceline – direct alpha optimisation (ZO + CMA-ES)

Run from the tests/ directory:
    cd tests && python General_test.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt

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
MAP_PATH = "maps/Catalunya/Catalunya_centerline.csv"
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
    t=1,                     # number of sampled directions for gradient estimation
    si=2,                    # interpolation stepsize [m]
    vm=22.88,                # maximum velocity [m/s]
    m_veh=1000,              # vehicle mass [kg]
    drag_coeff=0.75,         # aerodynamic drag coefficient
    MC=1,                    # Monte Carlo repetitions
    # min_s=0.5,               # minimum spline segment length [m]
    # max_s=2.0,               # maximum spline segment length [m]
    sigma=0.005,             # initial CMA-ES covariance
    iterations_ZO=30,       # ZO optimizer iterations (increase for better results)
    iterations_CMA=5,       # CMA-ES optimizer iterations
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
print(f"Stage 6 total time : {t_elapsed:.1f} s")

# Save ZO results for potential re-use in Stage 7
np.save("v_opt.npy", vx_opt)
np.save("kappa_opt.npy", kappa_opt)
np.save("s_splines.npy", s_splines)


# ===========================================================================
# Stage 7 – In-loop path refinement  (new Opt_min_CurvTime feature)
#
# The refine_every parameter added to Opt_min_CurvTime causes CurveLenOpt
# to automatically rebuild the reference track from the current best raceline
# every q iterations.  Each subsequent block of iterations therefore runs on
# a geometrically tighter track.  Crucially, refinement is CHAINED: each
# pass uses the previous refined track as its base (not the original), so
# the improvements accumulate.
#
# Three variants are run head-to-head with the same total iteration budget:
#   A. ZO  – no refinement                      (baseline)
#   B. ZO  – refine every REFINE_Q_ZO  iters    (new feature)
#   C. CMA – refine every REFINE_Q_CMA iters    (new feature)
#
# After each variant the object is reset to the original track so the next
# variant starts from the same initial conditions.
# ===========================================================================
print_stage(7, "In-loop path refinement – new feature demonstration")

REFINE_Q_ZO    = 33   # ZO  cadence: 100 total iters → 3 blocks (2 refinements)
REFINE_Q_CMA   = 3    # CMA cadence:  10 total iters → 3 blocks (2 refinements)
REFINE_SUBSAMP = 1    # sub-sampling stride; keep =1 to avoid spline-bowing violations

# A single optimizer instance is created with refine_every set to the ZO cadence.
# The CMA cadence is overridden at call time via the refine_every argument.
t_start = time.time()

optct_ref = solvers.Opt_min_CurvTime(
    reftrack=reftrack,
    center=reftrack,
    mu=0.01,
    h=0.001,
    kapb=0.5,
    sfty=1.0,
    t=1,
    si=2,
    vm=22.88,
    m_veh=1000,
    drag_coeff=0.75,
    MC=1,
    sigma=0.005,
    iterations_ZO=100,
    iterations_CMA=10,
    popsize=16,
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
    fw=3,
    refine_every=REFINE_Q_ZO,       # instance default (used by generate_kinProfs / ZO)
    refine_subsample=REFINE_SUBSAMP,
)

# ---- A : ZO without refinement ----
# Override refine_every=0 to disable refinement for this call only.
print("\n  A. ZO – no refinement")
t0 = time.time()
s_a, vx_a, ax_a, kappa_a, t_a, rl_a = optct_ref.generate_kinProfs(
    solver='ZO', refine_every=0)
print(f"     Done in {time.time()-t0:.1f} s  |  laptime = {t_a[-1]:.2f} s")
optct_ref.reset()   # restore original track for the next variant

# ---- B : ZO with in-loop refinement ----
# No override needed: uses self.refine_every = REFINE_Q_ZO.
print(f"\n  B. ZO – refine every {REFINE_Q_ZO} iters  (sub-sample stride {REFINE_SUBSAMP})")
t0 = time.time()
s_b, vx_b, ax_b, kappa_b, t_b, rl_b = optct_ref.generate_kinProfs(solver='ZO')
n_ctrl_b = optct_ref.reftrack.shape[0]
print(f"     Done in {time.time()-t0:.1f} s  |  laptime = {t_b[-1]:.2f} s  "
      f"|  final ctrl pts: {n_ctrl_b}  (was {reftrack.shape[0]})")
optct_ref.reset()

# ---- C : CMA-ES with in-loop refinement ----
# Override cadence per-call to use the CMA-specific value.
print(f"\n  C. CMA-ES – refine every {REFINE_Q_CMA} iters  (sub-sample stride {REFINE_SUBSAMP})")
t0 = time.time()
s_c, vx_c, ax_c, kappa_c, t_c, rl_c = optct_ref.generate_kinProfs(
    solver='CMA', refine_every=REFINE_Q_CMA)
n_ctrl_c = optct_ref.reftrack.shape[0]
print(f"     Done in {time.time()-t0:.1f} s  |  laptime = {t_c[-1]:.2f} s  "
      f"|  final ctrl pts: {n_ctrl_c}")
optct_ref.reset()

t_elapsed = time.time() - t_start

# ---- Results table ----
col = 44
print(f"\n  {'Variant':<{col}} {'Laptime (s)':>11}")
print(f"  {'-' * (col + 12)}")
print(f"  {'A.  ZO  – no refinement (150 iters)':<{col}} {t_a[-1]:>11.2f}")
print(f"  {'B.  ZO  – refine every ' + str(REFINE_Q_ZO)  + ' iters (150 iters)':<{col}} {t_b[-1]:>11.2f}")
print(f"  {'C.  CMA – refine every ' + str(REFINE_Q_CMA) + ' iters ( 15 iters)':<{col}} {t_c[-1]:>11.2f}")
print(f"\n  Stage 7 total time : {t_elapsed:.1f} s")

# ---- Plots ----
plt.figure(figsize=(14, 5))
plt.suptitle("Stage 7 – In-loop path refinement comparison")

plt.subplot(1, 2, 1)
plt.plot(optct_ref.bound1[:, 0], optct_ref.bound1[:, 1], 'k')
plt.plot(optct_ref.bound2[:, 0], optct_ref.bound2[:, 1], 'k', label='Track')
plt.plot(reftrack[:, 0], reftrack[:, 1], 'b--', alpha=0.5, label='Ref line')
plt.plot(rl_a[:, 0], rl_a[:, 1], 'g-',
         label=f'A: ZO no refine ({t_a[-1]:.1f} s)')
plt.plot(rl_b[:, 0], rl_b[:, 1], 'r.-',
         label=f'B: ZO refine/{REFINE_Q_ZO} ({t_b[-1]:.1f} s)')
plt.plot(rl_c[:, 0], rl_c[:, 1], 'm--',
         label=f'C: CMA refine/{REFINE_Q_CMA} ({t_c[-1]:.1f} s)')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.title('Racelines'); plt.legend(fontsize=8); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(s_a, vx_a, 'g-',  label=f'A: ZO no refine ({t_a[-1]:.1f} s)')
plt.plot(s_b, vx_b, 'r-',  label=f'B: ZO refine/{REFINE_Q_ZO} ({t_b[-1]:.1f} s)')
plt.plot(s_c, vx_c, 'm--', label=f'C: CMA refine/{REFINE_Q_CMA} ({t_c[-1]:.1f} s)')
plt.xlabel('s [m]'); plt.ylabel('v [m/s]')
plt.title('Velocity profiles'); plt.legend(fontsize=8); plt.grid(True)
plt.tight_layout()
plt.show()


# ===========================================================================
# Stage 8 – Blackbox_raceline – direct alpha optimisation (ZO + CMA-ES)
#
# Unlike Opt_min_CurvTime (which optimises spline segment lengths ds and uses
# a QP min-curvature proxy for the inner solve), Blackbox_raceline optimises
# the lateral shift vector alpha DIRECTLY for minimum lap time.
#
# Sub-stages:
#   8a  – compare initial costs across all three init modes
#   8b  – ZO optimizer on every init mode (convergence comparison)
#   8c  – CMA-ES optimizer on mincurv and shortest init
#   8d  – custom user-supplied cost function demo
#   8e  – extract full kinematic profiles for the best solutions
#   8f  – convergence-history and raceline / velocity comparison plots
# ===========================================================================
print_stage(8, "Blackbox_raceline – direct alpha optimisation (ZO + CMA-ES)")

t9_start = time.time()

# ---------------------------------------------------------------------------
# Shared hyper-parameters
# Keep iteration counts small so the script finishes in a reasonable time;
# increase BB_ITERS_ZO / BB_ITERS_CMA for production-quality results.
# ---------------------------------------------------------------------------
BB_SFTY      = 0.2     # safety half-width [m] (same as Opt_min_CurvTime)
BB_VM        = 22.88   # maximum velocity [m/s]
BB_SI        = 2.0     # raceline interpolation stepsize [m]
BB_FW        = 3       # velocity-profile filter window
BB_M         = 1000    # vehicle mass [kg]
BB_DRAG      = 0.75    # aerodynamic drag coefficient
BB_MU        = 0.001    # ZO perturbation magnitude μ
BB_H         = 0.0001   # ZO gradient-step size h
BB_T         = 5       # ZO directions averaged per estimate
BB_ITERS_ZO  = 360      # ZO gradient steps
BB_ITERS_CMA = 15      # CMA-ES generations
BB_SIGMA     = 0.01    # CMA-ES initial step size σ
BB_POPSIZE   = 20      # CMA-ES population size

# ---------------------------------------------------------------------------
# 8a: Initial cost at alpha_0 for all three init modes
# ---------------------------------------------------------------------------
print("\n--- 8a: Initial cost comparison ---")

init_modes  = ['random', 'zero', 'mincurv', 'shortest']
init_costs  = {}
init_alphas = {}

for mode in init_modes:
    _bb = solvers.Blackbox_raceline(
        reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
        sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
        m_veh=BB_M, drag_coeff=BB_DRAG,
        init=mode, oracle='gaussian',
        mu=BB_MU, h=BB_H, t=BB_T,
        iterations=BB_ITERS_ZO, seed=None)
    c = _bb._eval_cost(_bb.alpha_0)
    init_costs[mode]  = c
    init_alphas[mode] = _bb.alpha_0.copy()
    print(f"  init='{mode:8s}'  laptime0 = {c:.4f} s  "
          f"alpha in [{_bb.alpha_0.min():.4f}, {_bb.alpha_0.max():.4f}]")

# ---------------------------------------------------------------------------
# 8b: ZO optimizer on all three init modes
# ---------------------------------------------------------------------------
print(f"\n--- 8b: ZO optimizer ({BB_ITERS_ZO} iters, oracle='gaussian') ---")

zo_results = {}   # mode -> (best_alpha, history, bb_object)

for mode in init_modes:
    bb_zo = solvers.Blackbox_raceline(
        reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
        sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
        m_veh=BB_M, drag_coeff=BB_DRAG,
        init=mode, oracle='gaussian',
        mu=BB_MU, h=BB_H, t=BB_T,
        iterations=BB_ITERS_ZO, seed=None)
    t0 = time.time()
    best_a, hist = bb_zo.find_alpha(
        solver='ZO', n_iter=BB_ITERS_ZO,
        verbose=True, print_every=max(1, BB_ITERS_ZO // 5))
    zo_results[mode] = (best_a, hist, bb_zo)
    valid = hist[~np.isnan(hist)]
    print(f"  init='{mode}'  ZO done in {time.time()-t0:.1f} s  "
          f"best = {valid.min():.4f} s  (delta = {init_costs[mode]-valid.min():+.4f} s)")

# ---------------------------------------------------------------------------
# 8c: CMA-ES optimizer on mincurv and shortest init
# ---------------------------------------------------------------------------
print(f"\n--- 8c: CMA-ES optimizer ({BB_ITERS_CMA} gens, popsize={BB_POPSIZE}) ---")

cma_results = {}   # mode -> (best_alpha, history, bb_object)

for mode in ['mincurv', 'shortest']:
    bb_cma = solvers.Blackbox_raceline(
        reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
        sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
        m_veh=BB_M, drag_coeff=BB_DRAG,
        init=mode, oracle='gaussian',
        mu=BB_MU, h=BB_H, t=BB_T,
        iterations=BB_ITERS_CMA, seed=None)
    t0 = time.time()
    best_a, hist = bb_cma.find_alpha(
        solver='CMA', n_iter=BB_ITERS_CMA,
        sigma=BB_SIGMA, popsize=BB_POPSIZE,
        verbose=True, print_every=max(1, BB_ITERS_CMA // 5))
    cma_results[mode] = (best_a, hist, bb_cma)
    valid = hist[~np.isnan(hist)]
    print(f"  init='{mode}'  CMA done in {time.time()-t0:.1f} s  "
          f"best = {valid.min():.4f} s  (delta = {init_costs[mode]-valid.min():+.4f} s)")

# ---------------------------------------------------------------------------
# 8d: Custom user-supplied cost function
#
# The user passes any callable  cost_fn(alpha) -> float  to Blackbox_raceline.
# Here we demonstrate a WEIGHTED objective:
#
#     cost = lap_time + w * ay_peak
#
# where ay_peak = max(v^2 * |kappa|) is the peak lateral acceleration [m/s^2].
# Minimising this pushes the optimizer toward lines that slow down less in
# corners, trading a small amount of raw lap time for reduced tyre loading.
#
# The closure captures normvectors and track geometry from a reference
# Blackbox_raceline instance so that create_raceline can be called inside.
# ---------------------------------------------------------------------------
print("\n--- 8d: Custom cost function  (lap time + lateral-accel penalty) ---")

_bb_ref = solvers.Blackbox_raceline(
    reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
    sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
    m_veh=BB_M, drag_coeff=BB_DRAG,
    init='shortest', seed=None)   # reference object for normvec / geometry

W_COMFORT = 0.5   # weight on lateral-accel penalty [s/(m/s^2)]

def custom_cost(alpha):
    """
    Weighted objective: lap time + W_COMFORT * peak lateral acceleration.

    This is a simple example of a user-supplied cost function.
    Any callable(alpha) -> float is accepted by Blackbox_raceline.
    """
    _, _, cx, cy, si_idx, tv, _, _, el = create_raceline(
        refline=_bb_ref.reftrack[:, :2],
        normvectors=_bb_ref._normvec,
        alpha=alpha,
        stepsize_interp=_bb_ref.si)
    _, kappa = calc_head_curv_an(coeffs_x=cx, coeffs_y=cy,
                                  ind_spls=si_idx, t_spls=tv)
    n = kappa.size
    if n not in _bb_ref._p_ggv_cache:
        _bb_ref._p_ggv_cache[n] = np.repeat(
            np.expand_dims(_bb_ref.ggv, axis=0), n, axis=0)
    vx = calc_vel_profile(
        ggv=_bb_ref.ggv, ax_max_machines=_bb_ref.ax_max_machines,
        v_max=_bb_ref.v_max, kappa=kappa, el_lengths=el,
        closed=True, filt_window=_bb_ref.fw,
        dyn_model_exp=_bb_ref.dyn_model_exp,
        drag_coeff=_bb_ref.drag_coeff, m_veh=_bb_ref.m_veh,
        p_ggv=_bb_ref._p_ggv_cache[n])
    vx_cl  = np.append(vx, vx[0])
    ax_p   = calc_ax_profile(vx_profile=vx_cl, el_lengths=el,
                              eq_length_output=False)
    t_prof = calc_t_profile(vx_profile=vx, ax_profile=ax_p, el_lengths=el)
    laptime = float(t_prof[-1])
    ay_peak = float(np.max(vx**2 * np.abs(kappa)))   # [m/s^2]
    return laptime + W_COMFORT * ay_peak

custom_cost_0 = custom_cost(init_alphas['shortest'])
print(f"  Custom objective at shortest-path init : {custom_cost_0:.4f}")

bb_custom = solvers.Blackbox_raceline(
    reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
    sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
    m_veh=BB_M, drag_coeff=BB_DRAG,
    init='shortest', oracle='gaussian',
    mu=BB_MU, h=BB_H, t=BB_T,
    iterations=BB_ITERS_ZO,
    cost_fn=custom_cost,
    seed=None)
t0 = time.time()
best_custom, hist_custom = bb_custom.find_alpha(
    solver='ZO', n_iter=BB_ITERS_ZO,
    verbose=True, print_every=max(1, BB_ITERS_ZO // 5))
print(f"  Custom cost ZO done in {time.time()-t0:.1f} s  "
      f"best custom objective = {hist_custom[~np.isnan(hist_custom)].min():.4f}")

# Evaluate the DEFAULT laptime at the custom-optimized alpha so we can plot it
s_cust, vx_cust, ax_cust, _, t_cust, rl_cust = bb_custom.generate_raceline()
print(f"  Default laptime at custom-optimized alpha = {t_cust[-1]:.4f} s")

# ---------------------------------------------------------------------------
# 8e: Kinematic profiles for all best solutions + unoptimized references
# ---------------------------------------------------------------------------
print("\n--- 8e: Kinematic profiles for best solutions ---")

_bb_gen = solvers.Blackbox_raceline(
    reftrack=reftrack, ggv=ggv, ax_max_machines=ax_max_machines,
    sfty=BB_SFTY, v_max=BB_VM, si=BB_SI, fw=BB_FW,
    m_veh=BB_M, drag_coeff=BB_DRAG, init='random', seed=None)

def _kp(alpha):
    """Shorthand: generate kinematic profiles from any alpha."""
    return _bb_gen.generate_raceline(alpha=alpha)

# Unoptimized QP warm-starts
s_mc_bb, vx_mc_bb, _, _, t_mc_bb, rl_mc_bb = _kp(init_alphas['mincurv'])
s_sp_bb, vx_sp_bb, _, _, t_sp_bb, rl_sp_bb = _kp(init_alphas['shortest'])
print(f"  Mincurv  (unoptimized) laptime = {t_mc_bb[-1]:.4f} s")
print(f"  Shortest (unoptimized) laptime = {t_sp_bb[-1]:.4f} s")

# ZO best: pick the init with the lowest converged cost
best_zo_mode = min(zo_results,
                   key=lambda m: zo_results[m][1][~np.isnan(zo_results[m][1])].min())
s_zo_bb, vx_zo_bb, _, _, t_zo_bb, rl_zo_bb = _kp(zo_results[best_zo_mode][0])
print(f"  ZO best  (init='{best_zo_mode}') laptime = {t_zo_bb[-1]:.4f} s")

# CMA best
best_cma_mode = min(cma_results,
                    key=lambda m: cma_results[m][1][~np.isnan(cma_results[m][1])].min())
s_cma_bb, vx_cma_bb, _, _, t_cma_bb, rl_cma_bb = _kp(cma_results[best_cma_mode][0])
print(f"  CMA best (init='{best_cma_mode}') laptime = {t_cma_bb[-1]:.4f} s")

# ---------------------------------------------------------------------------
# 8f: Plots
# ---------------------------------------------------------------------------
print("\n--- 8f: Plotting ---")

# Track boundary lines for the map
_nv = _bb_gen._normvec
bnd_right = np.vstack([reftrack[:, :2] + _nv * reftrack[:, 2:3],
                        reftrack[0, :2] + _nv[0] * reftrack[0, 2:3]])
bnd_left  = np.vstack([reftrack[:, :2] - _nv * reftrack[:, 3:4],
                        reftrack[0, :2] - _nv[0] * reftrack[0, 3:4]])

# --- Figure 1: Convergence histories ---
fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
fig1.suptitle("Stage 8 - Blackbox_raceline: Convergence histories")

colors = {'random': 'tab:orange', 'zero': 'tab:red', 'mincurv': 'tab:blue', 'shortest': 'tab:green'}

# Left panel: ZO on all three inits
for mode in init_modes:
    _, hist, _ = zo_results[mode]
    valid = hist[~np.isnan(hist)]
    ax_left.plot(np.arange(len(hist)), hist,
                 color=colors[mode],
                 label=f"ZO / {mode}  -> {valid.min():.3f} s")
ax_left.set_xlabel("ZO iteration")
ax_left.set_ylabel("Best lap time [s]")
ax_left.set_title("ZO: effect of initialisation")
ax_left.legend(fontsize=9)
ax_left.grid(True)

# Right panel: ZO vs CMA-ES on mincurv and shortest; plus custom cost
for mode in ['mincurv', 'shortest']:
    _, hist_z, _ = zo_results[mode]
    _, hist_c, _ = cma_results[mode]
    vz = hist_z[~np.isnan(hist_z)].min()
    vc = hist_c[~np.isnan(hist_c)].min()
    ax_right.plot(np.arange(len(hist_z)), hist_z,
                  color=colors[mode], ls='-',
                  label=f"ZO  / {mode} -> {vz:.3f} s")
    ax_right.plot(np.arange(len(hist_c)), hist_c,
                  color=colors[mode], ls='--',
                  label=f"CMA / {mode} -> {vc:.3f} s")
vhc = hist_custom[~np.isnan(hist_custom)].min()
ax_right.plot(np.arange(len(hist_custom)), hist_custom,
              color='tab:purple', ls=':',
              label=f"ZO  / custom-cost -> {vhc:.3f}")
ax_right.set_xlabel("Iteration / Generation")
ax_right.set_ylabel("Best objective value")
ax_right.set_title("ZO vs CMA-ES (solid=ZO, dashed=CMA)")
ax_right.legend(fontsize=8)
ax_right.grid(True)
plt.tight_layout()
plt.show()

# --- Figure 2: Raceline map + velocity profiles ---
fig2, (ax_map, ax_vel) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Stage 8 - Blackbox_raceline: Raceline & velocity comparison")

ax_map.plot(bnd_right[:, 0], bnd_right[:, 1], 'k-',  lw=1.5, label='Boundary')
ax_map.plot(bnd_left[:, 0],  bnd_left[:, 1],  'k-',  lw=1.5)
ax_map.plot(rl_mc_bb[:, 0],  rl_mc_bb[:, 1],  'b--', lw=1.0,
            label=f"Mincurv  init ({t_mc_bb[-1]:.2f} s)")
ax_map.plot(rl_sp_bb[:, 0],  rl_sp_bb[:, 1],  'g--', lw=1.0,
            label=f"Shortest init ({t_sp_bb[-1]:.2f} s)")
ax_map.plot(rl_zo_bb[:, 0],  rl_zo_bb[:, 1],  'r-',  lw=2.0,
            label=f"ZO best ({t_zo_bb[-1]:.2f} s)")
ax_map.plot(rl_cma_bb[:, 0], rl_cma_bb[:, 1], 'm-',  lw=2.0,
            label=f"CMA best ({t_cma_bb[-1]:.2f} s)")
ax_map.plot(rl_cust[:, 0],   rl_cust[:, 1],   'c:',  lw=2.0,
            label=f"Custom cost ({t_cust[-1]:.2f} s)")
ax_map.set_xlabel("X [m]"); ax_map.set_ylabel("Y [m]")
ax_map.set_title("Racelines")
ax_map.legend(fontsize=8); ax_map.grid(True); ax_map.set_aspect('equal')

ax_vel.plot(s_mc_bb,  vx_mc_bb,  'b--', lw=1.0,
            label=f"Mincurv  init ({t_mc_bb[-1]:.2f} s)")
ax_vel.plot(s_sp_bb,  vx_sp_bb,  'g--', lw=1.0,
            label=f"Shortest init ({t_sp_bb[-1]:.2f} s)")
ax_vel.plot(s_zo_bb,  vx_zo_bb,  'r-',  lw=2.0,
            label=f"ZO best ({t_zo_bb[-1]:.2f} s)")
ax_vel.plot(s_cma_bb, vx_cma_bb, 'm-',  lw=2.0,
            label=f"CMA best ({t_cma_bb[-1]:.2f} s)")
ax_vel.plot(s_cust,   vx_cust,   'c:',  lw=2.0,
            label=f"Custom cost ({t_cust[-1]:.2f} s)")
ax_vel.set_xlabel("s [m]"); ax_vel.set_ylabel("v [m/s]")
ax_vel.set_title("Velocity profiles")
ax_vel.legend(fontsize=8); ax_vel.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 8f (extended): per-init  init vs optimised  comparison
# ---------------------------------------------------------------------------

# --- Generate kinematic profiles for EVERY init mode (initial AND ZO best) ---
# This fills in per-mode data that 8e only computed for the overall best.
init_profiles = {}   # mode -> (s, vx, ax, kappa, t, rl)  at alpha_0
zo_profiles   = {}   # mode -> (s, vx, ax, kappa, t, rl)  at ZO best alpha
cma_profiles  = {}   # mode -> (s, vx, ax, kappa, t, rl)  at CMA best alpha (subset)

for mode in init_modes:
    init_profiles[mode] = _kp(init_alphas[mode])
    zo_profiles[mode]   = _kp(zo_results[mode][0])

for mode in cma_results:
    cma_profiles[mode] = _kp(cma_results[mode][0])

# --- Detailed improvement table ---
print("\n--- Per-init improvement summary ---")
hdr = (f"  {'Init mode':<10} {'Init (s)':>9} {'ZO best (s)':>11}"
       f" {'ZO gain (s)':>11} {'ZO gain %':>10}"
       f" {'CMA best (s)':>12} {'CMA gain %':>10}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for mode in init_modes:
    t_i = init_profiles[mode][4][-1]
    t_z = zo_profiles[mode][4][-1]
    dz  = t_i - t_z
    pz  = dz / t_i * 100
    if mode in cma_profiles:
        t_c = cma_profiles[mode][4][-1]
        dc  = t_i - t_c
        pc  = dc / t_i * 100
        cma_col = f"{t_c:>12.3f} {pc:>+9.1f}%"
    else:
        cma_col = f"{'—':>12} {'—':>10}"
    print(f"  {mode:<10} {t_i:>9.3f} {t_z:>11.3f} {dz:>+11.3f} {pz:>+9.1f}%  {cma_col}")

# --- Velocity statistics per mode ---
print("\n--- Velocity statistics (initial vs ZO-optimized) ---")
stat_hdr = f"  {'Mode / state':<22} {'min vx':>8} {'mean vx':>9} {'max vx':>8} {'std vx':>8}"
print(stat_hdr)
print("  " + "-" * (len(stat_hdr) - 2))
for mode in init_modes:
    vx_i = init_profiles[mode][1]
    vx_z = zo_profiles[mode][1]
    print(f"  {mode + '  init':<22} {vx_i.min():>8.2f} {vx_i.mean():>9.2f}"
          f" {vx_i.max():>8.2f} {vx_i.std():>8.2f}")
    print(f"  {mode + '  ZO-opt':<22} {vx_z.min():>8.2f} {vx_z.mean():>9.2f}"
          f" {vx_z.max():>8.2f} {vx_z.std():>8.2f}")
    print()

# --- Alpha statistics ---
print("--- Alpha statistics (how much did the optimizer move the line?) ---")
alpha_hdr = f"  {'Mode':<10} {'init rms (m)':>13} {'ZO rms (m)':>11} {'delta rms (m)':>14} {'max |Δα| (m)':>13}"
print(alpha_hdr)
print("  " + "-" * (len(alpha_hdr) - 2))
for mode in init_modes:
    a_i = init_alphas[mode]
    a_z = zo_results[mode][0]
    print(f"  {mode:<10}"
          f" {np.sqrt(np.mean(a_i**2)):>13.4f}"
          f" {np.sqrt(np.mean(a_z**2)):>11.4f}"
          f" {np.sqrt(np.mean((a_z-a_i)**2)):>14.4f}"
          f" {np.max(np.abs(a_z-a_i)):>13.4f}")

# --- Figure 3: per-init grid  (raceline map + velocity, init vs ZO-opt) ---
n_modes = len(init_modes)
fig3, axes3 = plt.subplots(n_modes, 2, figsize=(14, 3.8 * n_modes))
fig3.suptitle("Stage 8 – per-init: initial (dashed) vs ZO-optimized (solid)",
              fontsize=12, fontweight='bold')

for row, mode in enumerate(init_modes):
    s_i, vx_i, _, _, t_i, rl_i = init_profiles[mode]
    s_z, vx_z, _, _, t_z, rl_z = zo_profiles[mode]
    clr = colors[mode]
    delta_t = t_i[-1] - t_z[-1]

    # Left column: raceline map
    ax_m = axes3[row, 0]
    ax_m.plot(bnd_right[:, 0], bnd_right[:, 1], 'k-', lw=1.0)
    ax_m.plot(bnd_left[:, 0],  bnd_left[:, 1],  'k-', lw=1.0)
    ax_m.plot(rl_i[:, 0], rl_i[:, 1], color=clr, ls='--', lw=1.2, alpha=0.7,
              label=f"Init   {t_i[-1]:.3f} s")
    ax_m.plot(rl_z[:, 0], rl_z[:, 1], color=clr, ls='-',  lw=2.0,
              label=f"ZO opt {t_z[-1]:.3f} s  ({delta_t:+.3f} s)")
    if mode in cma_profiles:
        _, _, _, _, t_c, rl_c = cma_profiles[mode]
        dc = t_i[-1] - t_c[-1]
        ax_m.plot(rl_c[:, 0], rl_c[:, 1], color=clr, ls=':', lw=1.5, alpha=0.8,
                  label=f"CMA    {t_c[-1]:.3f} s  ({dc:+.3f} s)")
    ax_m.set_xlabel("X [m]"); ax_m.set_ylabel("Y [m]")
    ax_m.set_title(f"init='{mode}':  raceline")
    ax_m.legend(fontsize=8); ax_m.grid(True); ax_m.set_aspect('equal')

    # Right column: velocity profiles + shaded improvement
    ax_v = axes3[row, 1]
    s_max = min(s_i[-1], s_z[-1])
    s_fb  = np.linspace(0, s_max, min(len(s_i), len(s_z)))
    vx_i_fb = np.interp(s_fb, s_i, vx_i)
    vx_z_fb = np.interp(s_fb, s_z, vx_z)
    ax_v.fill_between(s_fb, vx_i_fb, vx_z_fb,
                      where=(vx_z_fb >= vx_i_fb),
                      alpha=0.20, color='green',  label='ZO faster')
    ax_v.fill_between(s_fb, vx_i_fb, vx_z_fb,
                      where=(vx_z_fb < vx_i_fb),
                      alpha=0.20, color='red', label='ZO slower')
    ax_v.plot(s_i, vx_i, color=clr, ls='--', lw=1.2, alpha=0.7,
              label=f"Init   {t_i[-1]:.3f} s")
    ax_v.plot(s_z, vx_z, color=clr, ls='-',  lw=2.0,
              label=f"ZO opt {t_z[-1]:.3f} s")
    if mode in cma_profiles:
        s_c, vx_c = cma_profiles[mode][0], cma_profiles[mode][1]
        ax_v.plot(s_c, vx_c, color=clr, ls=':', lw=1.5, alpha=0.8,
                  label=f"CMA    {cma_profiles[mode][4][-1]:.3f} s")
    ax_v.set_xlabel("s [m]"); ax_v.set_ylabel("v [m/s]")
    ax_v.set_title(f"init='{mode}':  velocity profile")
    ax_v.legend(fontsize=8); ax_v.grid(True)

plt.tight_layout()
plt.show()

# --- Figure 4: Δvx profiles (ZO-opt minus init) on a common s-grid ---
s_all_max = min(init_profiles[m][0][-1] for m in init_modes)
s_common  = np.linspace(0, s_all_max, 500)

fig4, (ax4t, ax4b) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
fig4.suptitle("Stage 8 – velocity gain from ZO optimisation  (Δvx = ZO-opt − init)",
              fontsize=12, fontweight='bold')

for mode in init_modes:
    s_i, vx_i = init_profiles[mode][0], init_profiles[mode][1]
    s_z, vx_z = zo_profiles[mode][0],   zo_profiles[mode][1]
    vx_i_c = np.interp(s_common, s_i, vx_i)
    vx_z_c = np.interp(s_common, s_z, vx_z)
    delta   = vx_z_c - vx_i_c
    mean_d  = delta.mean()
    ax4t.plot(s_common, delta, color=colors[mode], lw=1.5,
              label=f"{mode:<10} mean Δ = {mean_d:+.3f} m/s")
    ax4b.plot(s_i, vx_i, color=colors[mode], ls='--', lw=1.0, alpha=0.5)
    ax4b.plot(s_z, vx_z, color=colors[mode], ls='-',  lw=1.5,
              label=f"{mode} ZO ({zo_profiles[mode][4][-1]:.3f} s)")

ax4t.axhline(0, color='k', lw=0.8, ls=':')
ax4t.set_ylabel("Δvx [m/s]")
ax4t.set_title("Δvx per track position  (positive = ZO faster, negative = ZO slower)")
ax4t.legend(fontsize=8); ax4t.grid(True)

ax4b.set_xlabel("s [m]"); ax4b.set_ylabel("v [m/s]")
ax4b.set_title("All initial (dashed, faded) vs ZO-optimized (solid) velocity profiles")
ax4b.legend(fontsize=8); ax4b.grid(True)

plt.tight_layout()
plt.show()

# --- Figure 5: alpha profiles (init vs ZO-opt, per control point) ---
ctrl_idx = np.arange(reftrack.shape[0])

fig5, (ax5l, ax5r) = plt.subplots(1, 2, figsize=(14, 5))
fig5.suptitle("Stage 8 – alpha profiles  (lateral shift at each control point)",
              fontsize=12, fontweight='bold')

for mode in init_modes:
    a_i = init_alphas[mode]
    a_z = zo_results[mode][0]
    ax5l.plot(ctrl_idx, a_i, color=colors[mode], ls='--', lw=1.0, alpha=0.6,
              label=f"{mode} init")
    ax5l.plot(ctrl_idx, a_z, color=colors[mode], ls='-',  lw=1.5,
              label=f"{mode} ZO opt")
    ax5r.plot(ctrl_idx, a_z - a_i, color=colors[mode], lw=1.5,
              label=f"{mode}  rms={np.sqrt(np.mean((a_z-a_i)**2)):.4f} m")

ax5l.axhline(0, color='k', lw=0.6, ls=':')
ax5l.set_xlabel("Control point index"); ax5l.set_ylabel("alpha [m]")
ax5l.set_title("Alpha at each ctrl point  (dashed=init, solid=ZO-opt)")
ax5l.legend(fontsize=7, ncol=2); ax5l.grid(True)

ax5r.axhline(0, color='k', lw=0.8, ls=':')
ax5r.set_xlabel("Control point index"); ax5r.set_ylabel("Δalpha [m]")
ax5r.set_title("Alpha change: ZO-opt minus init  (positive = moved right)")
ax5r.legend(fontsize=8); ax5r.grid(True)

plt.tight_layout()
plt.show()

print(f"\nStage 8 summary:")
print(f"  Mincurv  (unopt.)      : {t_mc_bb[-1]:.3f} s")
print(f"  Shortest (unopt.)      : {t_sp_bb[-1]:.3f} s")
print(f"  ZO best  ({best_zo_mode:9s}) : {t_zo_bb[-1]:.3f} s")
print(f"  CMA best ({best_cma_mode:9s}) : {t_cma_bb[-1]:.3f} s")
print(f"  Custom cost ZO best    : {t_cust[-1]:.3f} s  (weighted time + comfort)")
print(f"Stage 8 total time : {time.time()-t9_start:.1f} s")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("  Test summary")
print("=" * 60)
print(f"  Shortest-path laptime                      : {t_sp[-1]:.2f} s")
print(f"  Min-curvature laptime                      : {t_mc[-1]:.2f} s")
print(f"  ZO stage 6 (50 iters, no refine)           : {t_opt[-1]:.2f} s")
print(f"  ZO stage 7A (150 iters, no refine)         : {t_a[-1]:.2f} s")
print(f"  ZO stage 7B (150 iters, refine/{REFINE_Q_ZO})      : {t_b[-1]:.2f} s")
print(f"  CMA stage 7C ( 15 iters, refine/{REFINE_Q_CMA})       : {t_c[-1]:.2f} s")
print(f"  BB / ZO best                               : {t_zo_bb[-1]:.2f} s  (direct alpha opt.)")
print(f"  BB / CMA best                              : {t_cma_bb[-1]:.2f} s  (direct alpha opt.)")
print(f"  BB / custom cost                           : {t_cust[-1]:.2f} s  (time + comfort)")
print("=" * 60)
print("All stages completed successfully.")
