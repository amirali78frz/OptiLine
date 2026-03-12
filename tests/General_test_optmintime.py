"""
General_test_optmintime.py
==========================
Complete test and demonstration script for opt_mintime.py.

This script solves the minimum-lap-time optimal control problem on a closed
race track using direct orthogonal collocation (Gauss-Legendre) via CasADi /
IPOPT, as implemented in opt_mintime.opt_mintime().

Stages
------
  Stage 1 : Track import and sub-sampling
  Stage 2 : Configuration loading (racecar.ini + GGV tables)
  Stage 3 : Track preprocessing  (utils.prep_track)
  Stage 4 : Minimum lap-time optimization  (opt_mintime.opt_mintime)
  Stage 5 : Re-optimization pass
              – widen the search corridor around the first-pass solution
              – solve again to check for further lap-time savings
  Stage 6 : Results inspection and plots

Run from the tests/ directory:
    cd tests && python General_test_optmintime.py
"""

import copy
import json
import os
import configparser
import time

import numpy as np
import matplotlib.pyplot as plt

from OptiLine import utils
from OptiLine import opt_mintime

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
    Return every `step`-th row of reftrack_full, always including the last row.
    Reduces the problem size while preserving overall track shape.
    """
    nn = reftrack_full.shape[0]
    indices = np.arange(0, nn, step)
    indices = np.unique(np.concatenate([indices, [nn - 1]]))
    return reftrack_full[indices]


# ===========================================================================
# Stage 1 – Track import and sub-sampling
# ===========================================================================
print_stage(1, "Track import and sub-sampling")

# Path to the centerline CSV: [x_m, y_m, w_tr_right_m, w_tr_left_m]
MAP_PATH    = "maps/Melbourne/Melbourne_centerline.csv"
GGV_PATH    = "maps/ggv.csv"
AX_MAX_PATH = "maps/ax_max_machines.csv"

csv_data      = np.loadtxt(MAP_PATH, comments='#', delimiter=',')
reftrack_full = csv_data[:, 0:4]

# Sub-sample to lower the number of collocation intervals for faster solving.
# Every 4th point is kept; increase `step` further to speed up, or set step=1
# to use the full resolution.
reftrack = subsample_track(reftrack_full, step=4)

# Compute Euclidean segment lengths between consecutive points
lengths  = np.sqrt(np.sum(np.diff(reftrack[:, 0:2], axis=0) ** 2, axis=1))
lengths  = np.append(lengths, lengths[0])   # close the loop
ds_0     = lengths

print(f"Full track points     : {reftrack_full.shape[0]}")
print(f"Subsampled points     : {reftrack.shape[0]}")
print(f"Segment length range  : [{ds_0.min():.3f}, {ds_0.max():.3f}] m")


# ===========================================================================
# Stage 2 – Configuration loading (racecar.ini + GGV tables)
# ===========================================================================
print_stage(2, "Configuration loading (racecar.ini + GGV tables)")

# ---- 2a: locate files relative to this script ----------------------------
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

file_paths = {
    "veh_params_file"  : "racecar.ini",
    "module"           : MODULE_DIR,
}


# ---- 2b: mintime-specific solver options ---------------------------------
# warm_start:              reuse a previously saved IPOPT warm-start solution
# var_friction:            set to "linear" or "gauss" for spatially varying mu
# reopt_mintime_solution:  when True (Stage 5), widen the corridor and re-run
# recalc_vel_profile_by_tph: recalculate velocity profile externally after opt.
mintime_opts = {
    "tpadata"                 : None,
    "warm_start"              : False,
    "var_friction"            : None,
    "reopt_mintime_solution"  : False,   # changed to True in Stage 5
    "recalc_vel_profile_by_tph": False,
}

# ---- 2c: parse vehicle parameters from INI file --------------------------
parser = configparser.ConfigParser()

ini_path = os.path.join(file_paths["module"], "params", file_paths["veh_params_file"])
if not parser.read(ini_path):
    raise FileNotFoundError(
        f"Vehicle parameter file not found or empty: {ini_path}"
    )

pars = {}
pars["ggv_file"]               = json.loads(parser.get('GENERAL_OPTIONS', 'ggv_file'))
pars["ax_max_machines_file"]   = json.loads(parser.get('GENERAL_OPTIONS', 'ax_max_machines_file'))
pars["stepsize_opts"]          = json.loads(parser.get('GENERAL_OPTIONS', 'stepsize_opts'))
pars["reg_smooth_opts"]        = json.loads(parser.get('GENERAL_OPTIONS', 'reg_smooth_opts'))
pars["veh_params"]             = json.loads(parser.get('GENERAL_OPTIONS', 'veh_params'))
pars["vel_calc_opts"]          = json.loads(parser.get('GENERAL_OPTIONS', 'vel_calc_opts'))
pars["curv_calc_opts"]         = json.loads(parser.get('GENERAL_OPTIONS', 'curv_calc_opts'))
pars["optim_opts"]             = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'optim_opts_mintime'))
pars["vehicle_params_mintime"] = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'vehicle_params_mintime'))
pars["tire_params_mintime"]    = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'tire_params_mintime'))
pars["pwr_params_mintime"]     = json.loads(parser.get('OPTIMIZATION_OPTIONS', 'pwr_params_mintime'))

# Apply runtime overrides from mintime_opts
pars["optim_opts"]["var_friction"] = mintime_opts["var_friction"]
pars["optim_opts"]["warm_start"]   = mintime_opts["warm_start"]

# Derived wheelbase = front + rear
pars["vehicle_params_mintime"]["wheelbase"] = (
    pars["vehicle_params_mintime"]["wheelbase_front"]
    + pars["vehicle_params_mintime"]["wheelbase_rear"]
)

print(f"Vehicle parameter file : {ini_path}")
print(f"Optimization width_opt : {pars['optim_opts']['width_opt']} m")
print(f"Friction coefficient   : {pars['optim_opts']['mue']}")
print(f"Wheelbase (total)      : {pars['vehicle_params_mintime']['wheelbase']:.2f} m")

# ---- 2d: load GGV table and ax_max_machines ------------------------------
ggv, ax_max_machines = utils.import_veh_dyn_info(
    ggv_import_path=GGV_PATH,
    ax_max_machines_import_path=AX_MAX_PATH,
)
print(f"GGV table shape        : {ggv.shape}")
print(f"ax_max_machines shape  : {ax_max_machines.shape}")

# Fill in "safe trajectory" limits from the GGV table when left as None
if pars["optim_opts"]["ax_pos_safe"] is None:
    pars["optim_opts"]["ax_pos_safe"] = np.amin(ggv[:, 1])
if pars["optim_opts"]["ax_neg_safe"] is None:
    pars["optim_opts"]["ax_neg_safe"] = -np.amin(ggv[:, 1])
if pars["optim_opts"]["ay_safe"] is None:
    pars["optim_opts"]["ay_safe"] = np.amin(ggv[:, 2])

print(f"ax_pos_safe            : {pars['optim_opts']['ax_pos_safe']:.2f} m/s²")
print(f"ax_neg_safe            : {pars['optim_opts']['ax_neg_safe']:.2f} m/s²")
print(f"ay_safe                : {pars['optim_opts']['ay_safe']:.2f} m/s²")


# ===========================================================================
# Stage 3 – Track preprocessing (utils.prep_track)
# ===========================================================================
print_stage(3, "Track preprocessing (utils.prep_track)")

# prep_track:
#   • applies B-spline approximation to smooth the imported centerline
#   • re-samples at a uniform stepsize suitable for collocation
#   • returns interpolated track, normal vectors, spline coefficients
t_start = time.time()

(reftrack_interp,
 normvec_normalized_interp,
 a_interp,
 coeffs_x_interp,
 coeffs_y_interp) = utils.prep_track(
    reftrack_imp=reftrack,
    reg_smooth_opts=pars["reg_smooth_opts"],
    stepsize_opts=pars["stepsize_opts"],
    debug=True,
    min_width=None,   # set to a float [m] to enforce a minimum track width
)

print(f"prep_track done in {time.time()-t_start:.2f} s")
print(f"Raw track points       : {reftrack.shape[0]}")
print(f"Interpolated points    : {reftrack_interp.shape[0]}")
print(f"Normal vector shape    : {normvec_normalized_interp.shape}")
print(f"Spline coeffs x/y      : {coeffs_x_interp.shape} / {coeffs_y_interp.shape}")
print(f"System matrix a_interp : {a_interp.shape}")


# ===========================================================================
# Stage 4 – Minimum lap-time optimization (opt_mintime.opt_mintime)
# ===========================================================================
print_stage(4, "Minimum lap-time optimization (opt_mintime.opt_mintime)")

# For the first optimization pass the vehicle corridor equals width_opt.
# No warm-start or re-optimization adjustment is applied yet.
pars_pass1 = pars   # use parameters as loaded from INI

t_start = time.time()

(alpha_opt,
 v_opt,
 raceline_opt,
 reftrack_interp_out,
 a_interp_tmp,
 normvec_out,
 s_opt, laptime) = opt_mintime.opt_mintime(
    reftrack    = reftrack_interp,
    coeffs_x    = coeffs_x_interp,
    coeffs_y    = coeffs_y_interp,
    normvectors = normvec_normalized_interp,
    pars        = pars_pass1,
    print_debug = True,
    plot_       = True,     # set to False to suppress per-step plots
)

t_elapsed = time.time() - t_start
print(f"\nopt_mintime (pass 1) solved in {t_elapsed:.1f} s")
print(f"alpha_opt range  : [{alpha_opt.min():.4f}, {alpha_opt.max():.4f}] m")
print(f"Velocity range   : [{v_opt.min():.2f}, {v_opt.max():.2f}] m/s")
print(f"Arc-length range : [{s_opt.min():.2f}, {s_opt.max():.2f}] m")

# If non-regular sampling modified the spline matrices, update for later use
if a_interp_tmp is not None:
    a_interp = a_interp_tmp

# Save first-pass velocity and arc-length profiles
np.save("vt.npy", v_opt)
np.save("st.npy", s_opt)
print("Saved v_opt -> vt.npy,  s_opt -> st.npy")


# ===========================================================================
# Stage 5 – Re-optimization pass
#
# Workflow:
#   1. Widen the optimization corridor by w_tr_reopt - w_veh_reopt
#      (plus w_add_spl_regr to compensate second spline regression).
#   2. Use the output of Stage 4 (reftrack_interp_out, normvec_out, a_interp)
#      as the new reference track – these may differ from Stage 3 outputs when
#      non-regular sampling was active.
#   3. Solve the NLP again on the widened corridor.
#   4. Compare first-pass vs re-optimized lap time.
#
# This simulates the standard two-pass approach:
#   Pass 1 – find a good solution on a conservative corridor.
#   Pass 2 – exploit the extra track width available around that solution.
# ===========================================================================
print_stage(5, "Re-optimization pass (widened corridor)")

# Build the widened parameter set:
#   width_opt_reopt = width_opt + (w_tr_reopt - w_veh_reopt) + w_add_spl_regr
w_veh_tmp = (
    pars["optim_opts"]["width_opt"]
    + (pars["optim_opts"]["w_tr_reopt"] - pars["optim_opts"]["w_veh_reopt"])
    + pars["optim_opts"]["w_add_spl_regr"]
)
pars_reopt = copy.deepcopy(pars)
pars_reopt["optim_opts"]["width_opt"] = w_veh_tmp

print(f"Pass 1 width_opt  : {pars['optim_opts']['width_opt']:.3f} m")
print(f"Pass 2 width_opt  : {w_veh_tmp:.3f} m  "
      f"(+{w_veh_tmp - pars['optim_opts']['width_opt']:.3f} m)")

t_start = time.time()

(alpha_reopt,
 v_reopt,
 raceline_reopt,
 reftrack_reopt_out,
 a_reopt_tmp,
 normvec_reopt_out,
 s_reopt, laptime_reopt) = opt_mintime.opt_mintime(
    reftrack    = reftrack_interp_out,   # output from pass 1
    coeffs_x    = coeffs_x_interp,
    coeffs_y    = coeffs_y_interp,
    normvectors = normvec_out,           # output from pass 1
    pars        = pars_reopt,
    print_debug = True,
    plot_       = False,    # suppress intermediate plots for re-opt pass
)

t_elapsed = time.time() - t_start
print(f"\nopt_mintime (pass 2 / re-opt) solved in {t_elapsed:.1f} s")
print(f"Pass 1 laptime  : {laptime:.2f} s")
print(f"Pass 2 laptime  : {laptime_reopt:.2f} s")
print(f"v_max pass 1    : {v_opt.max():.2f} m/s")
print(f"v_max pass 2    : {v_reopt.max():.2f} m/s")


# ===========================================================================
# Stage 6 – Results inspection and plots
# ===========================================================================
print_stage(6, "Results inspection and plots")

# ---- 6a: racelines on track -----------------------------------------------
plt.figure(figsize=(10, 8))
plt.title("Stage 6a – Racelines on track (pass 1 vs re-opt)")
plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1],
         'k--', linewidth=1.0, label='Centerline')
plt.plot(raceline_opt[:, 0],   raceline_opt[:, 1],
         'r-',  linewidth=1.5, label='Pass 1')
plt.plot(raceline_reopt[:, 0], raceline_reopt[:, 1],
         'b--', linewidth=1.5, label='Pass 2 (re-opt)')
plt.xlabel('X [m]'); plt.ylabel('Y [m]')
plt.legend(); plt.grid(True); plt.axis('equal')
plt.tight_layout()
plt.show()

# ---- 6b: velocity profiles ------------------------------------------------
plt.figure(figsize=(12, 4))
plt.title("Stage 6b – Velocity profiles")
plt.plot(s_opt,    v_opt,    'r-',  label='Pass 1')
plt.plot(s_reopt,  v_reopt,  'b--', label='Pass 2 (re-opt)')
plt.xlabel('Arc length s [m]'); plt.ylabel('v [m/s]')
plt.ylim(bottom=0); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 6c: lateral shift (alpha) profiles -----------------------------------
# alpha > 0  : raceline shifted toward the left boundary
# alpha < 0  : raceline shifted toward the right boundary
s_alpha = np.linspace(0, s_opt[-1], len(alpha_opt))
s_alpha_r = np.linspace(0, s_reopt[-1], len(alpha_reopt))

plt.figure(figsize=(12, 4))
plt.title("Stage 6c – Lateral shift (alpha) profiles")
plt.plot(s_alpha,   alpha_opt,    'r-',  label='Pass 1')
plt.plot(s_alpha_r, alpha_reopt,  'b--', label='Pass 2 (re-opt)')
plt.axhline(0, color='k', linewidth=0.8, linestyle=':')
plt.xlabel('Arc length s [m]'); plt.ylabel('α [m]')
plt.legend(); plt.grid(True)
plt.tight_layout()
plt.show()

# ---- 6d: print summary table ----------------------------------------------
print("\n" + "=" * 60)
print("  Optimization summary")
print("=" * 60)
print(f"  Track             : {MAP_PATH}")
print(f"  Collocation pts   : {reftrack_interp.shape[0]}")
print(f"  Pass 1 width_opt  : {pars['optim_opts']['width_opt']:.3f} m")
print(f"  Pass 2 width_opt  : {w_veh_tmp:.3f} m")
print(f"  v_max  pass 1     : {v_opt.max():.2f} m/s")
print(f"  v_max  pass 2     : {v_reopt.max():.2f} m/s")
print(f"  |alpha| max p1    : {np.abs(alpha_opt).max():.4f} m")
print(f"  |alpha| max p2    : {np.abs(alpha_reopt).max():.4f} m")
print("=" * 60)
print("All stages completed successfully.")
