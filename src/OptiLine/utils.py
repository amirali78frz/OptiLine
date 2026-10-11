import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import quadprog
from scipy import interpolate
from scipy import optimize
from scipy import spatial
from typing import Union

def calc_spline_lengths(coeffs_x: np.ndarray,
                        coeffs_y: np.ndarray,
                        quickndirty: bool = False,
                        no_interp_points: int = 15) -> np.ndarray:
    """
    Calculate spline lengths for third order splines defining x- and y-coordinates using intermediate steps.

    Parameters
    ----------
    coeffs_x : np.ndarray
        Coefficient matrix of the x splines with shape (n_splines, 4).
    coeffs_y : np.ndarray
        Coefficient matrix of the y splines with shape (n_splines, 4).
    quickndirty : bool, optional
        If True, returns approximate lengths using Euclidean distance between start and end points.
    no_interp_points : int, optional
        Number of interpolation steps for length calculation (default is 15).

    Returns
    -------
    np.ndarray
        Array of spline segment lengths.

    Notes
    -----
    Assumes cubic splines with coefficients in order: [a0, a1, a2, a3].
    """
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise ValueError("Coefficient matrices must have the same number of splines!")

    if coeffs_x.ndim == 1:
        coeffs_x = np.expand_dims(coeffs_x, 0)
        coeffs_y = np.expand_dims(coeffs_y, 0)

    # Calculating the lengths
    if quickndirty:
        end_x = np.sum(coeffs_x, axis=1)
        end_y = np.sum(coeffs_y, axis=1)
        start_x = coeffs_x[:, 0]
        start_y = coeffs_y[:, 0]
        return np.hypot(end_x - start_x, end_y - start_y)

    t_steps = np.linspace(0.0, 1.0, no_interp_points)
    vander_t = np.vander(t_steps, N=4, increasing=True)  

    x_vals_all = vander_t @ coeffs_x.T  # shape: (no_interp_points, no_splines)
    y_vals_all = vander_t @ coeffs_y.T
    dx = np.diff(x_vals_all, axis=0)
    dy = np.diff(y_vals_all, axis=0)
    spline_lengths = np.sum(np.sqrt(dx**2 + dy**2), axis=0)

    return spline_lengths

import numpy as np
import math

def interp_splines(coeffs_x, coeffs_y, no_interp_points=None, stepnum_fixed=None, 
                   spline_lengths=None, stepsize_approx=None, incl_last_point=False):
    """
    Interpolates 2D spline segments with given polynomial coefficients.

    Parameters
    ----------
    coeffs_x : ndarray
        Shape (n, 4), where each row is [a0, a1, a2, a3] for a cubic x spline segment.
    coeffs_y : ndarray
        Same as coeffs_x, but for y coordinates.
    no_interp_points : int, optional
        Number of interpolated points. Only needed if `stepnum_fixed` is None.
    stepnum_fixed : list or ndarray, optional
        Number of interpolation points per segment (can vary per segment).
    spline_lengths : list or ndarray, optional
        Lengths of spline segments (used with stepsize_approx).
    stepsize_approx : float, optional
        Approximate step size for interpolation.
    incl_last_point : bool, optional
        If True, include final point at t = 1 of last segment.

    Returns
    -------
    path_interp : ndarray
        Interpolated 2D path of shape (no_interp_points, 2).
    spline_inds : ndarray
        Indices of spline segment for each interpolated point.
    t_values : ndarray
        Normalized t value [0, 1] of point within its segment.
    dists_interp : ndarray or None
        Distance along the path for each point (only if stepsize_approx is used).
    """
    # check sizes
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise RuntimeError("Coefficient matrices must have the same length!")

    if spline_lengths is not None and coeffs_x.shape[0] != spline_lengths.size:
        raise RuntimeError("coeffs_x/y and spline_lengths must have the same length!")

    # check if coeffs_x and coeffs_y have exactly two dimensions and raise error otherwise
    if not (coeffs_x.ndim == 2 and coeffs_y.ndim == 2):
        raise RuntimeError("Coefficient matrices do not have two dimensions!")

    # check if step size specification is valid
    if (stepsize_approx is None and stepnum_fixed is None) \
            or (stepsize_approx is not None and stepnum_fixed is not None):
        raise RuntimeError("Provide one of 'stepsize_approx' and 'stepnum_fixed' and set the other to 'None'!")

    if stepnum_fixed is not None and len(stepnum_fixed) != coeffs_x.shape[0]:
        raise RuntimeError("The provided list 'stepnum_fixed' must hold an entry for every spline!")
    

    coeffs_x = np.array(coeffs_x)
    coeffs_y = np.array(coeffs_y)
    dists_interp = None

    if stepsize_approx is not None:
        spline_lengths = np.asarray(spline_lengths).flatten()
        dists_cum = np.cumsum(spline_lengths)

        no_interp_points = math.ceil(dists_cum[-1] / stepsize_approx) + 1
        dists_interp = np.linspace(0.0, dists_cum[-1], no_interp_points)

        path_interp = np.zeros((no_interp_points, 2))
        spline_inds = np.zeros(no_interp_points, dtype=int)
        t_values = np.zeros(no_interp_points)

        js = np.searchsorted(dists_cum, dists_interp[:-1])
        spline_inds[:-1] = js
        d0s = np.where(js > 0, dists_cum[js - 1], 0.0)
        t_vals = (dists_interp[:-1] - d0s) / spline_lengths[js]
        t_values[:-1] = t_vals
        t2 = t_vals ** 2
        t3 = t_vals ** 3
        path_interp[:-1, 0] = coeffs_x[js, 0] + coeffs_x[js, 1] * t_vals + coeffs_x[js, 2] * t2 + coeffs_x[js, 3] * t3
        path_interp[:-1, 1] = coeffs_y[js, 0] + coeffs_y[js, 1] * t_vals + coeffs_y[js, 2] * t2 + coeffs_y[js, 3] * t3

    elif stepnum_fixed is not None:
        no_interp_points = np.sum(stepnum_fixed)
        path_interp = np.zeros((no_interp_points, 2))
        spline_inds = np.zeros(no_interp_points, dtype=int)
        t_values = np.zeros(no_interp_points)

        j = 0
        for i, steps in enumerate(stepnum_fixed):
            if i < len(stepnum_fixed) - 1:
                t_values[j:j+steps-1] = np.linspace(0, 1, steps)[:-1]
                spline_inds[j:j+steps-1] = i
                j += steps - 1
            else:
                t_values[j:j+steps] = np.linspace(0, 1, steps)
                spline_inds[j:j+steps] = i
                j += steps

        t_set = np.column_stack((np.ones(no_interp_points), t_values, t_values**2, t_values**3))

        # Repeat coefficients according to stepnum_fixed (excluding last point per segment)
        coeffs_x_rep = np.vstack([
            np.tile(row, (n - 1 if i < len(stepnum_fixed) - 1 else n, 1))
            for i, (row, n) in enumerate(zip(coeffs_x, stepnum_fixed))
        ])
        coeffs_y_rep = np.vstack([
            np.tile(row, (n - 1 if i < len(stepnum_fixed) - 1 else n, 1))
            for i, (row, n) in enumerate(zip(coeffs_y, stepnum_fixed))
        ])

        path_interp[:, 0] = np.sum(coeffs_x_rep * t_set, axis=1)
        path_interp[:, 1] = np.sum(coeffs_y_rep * t_set, axis=1)

    else:
        raise ValueError("Either stepsize_approx or stepnum_fixed must be provided.")

    # Include the final point at t=1 of the last segment if requested
    if incl_last_point:
        path_interp[-1, 0] = np.dot(coeffs_x[-1], [1, 1, 1, 1])
        path_interp[-1, 1] = np.dot(coeffs_y[-1], [1, 1, 1, 1])
        spline_inds[-1] = coeffs_x.shape[0] - 1
        t_values[-1] = 1.0
    else:
        path_interp = path_interp[:-1]
        spline_inds = spline_inds[:-1]
        t_values = t_values[:-1]
        if dists_interp is not None:
            dists_interp = dists_interp[:-1]

    return path_interp, spline_inds, t_values, dists_interp


def calc_splines(path: np.ndarray,
                 el_lengths: np.ndarray = None,
                 psi_s: float = None,
                 psi_e: float = None,
                 use_dist_scaling: bool = True) -> tuple:
    """

    .. description::
    Solve for curvature continuous cubic splines (spline parameter t) between given points i (splines evaluated at
    t = 0 and t = 1). The splines must be set up separately for x- and y-coordinate.

    Spline equations:
    P_{x,y}(t)   =  a_3 * t³ +  a_2 * t² + a_1 * t + a_0
    P_{x,y}'(t)  = 3a_3 * t² + 2a_2 * t  + a_1
    P_{x,y}''(t) = 6a_3 * t  + 2a_2

    a * {x; y} = {b_x; b_y}

    .. inputs::
    :param path:                x and y coordinates as the basis for the spline construction (closed or unclosed). If
                                path is provided unclosed, headings psi_s and psi_e are required!
    :type path:                 np.ndarray
    :param el_lengths:          distances between path points (closed or unclosed). The input is optional. The distances
                                are required for the scaling of heading and curvature values. They are calculated using
                                euclidian distances if required but not supplied.
    :type el_lengths:           np.ndarray
    :param psi_s:               orientation of the {start, end} point.
    :type psi_s:                float
    :param psi_e:               orientation of the {start, end} point.
    :type psi_e:                float
    :param use_dist_scaling:    bool flag to indicate if heading and curvature scaling should be performed. This should
                                be done if the distances between the points in the path are not equal.
    :type use_dist_scaling:     bool

    .. outputs::
    :return x_coeff:            spline coefficients of the x-component.
    :rtype x_coeff:             np.ndarray
    :return y_coeff:            spline coefficients of the y-component.
    :rtype y_coeff:             np.ndarray
    :return M:                  LES coefficients.
    :rtype M:                   np.ndarray
    :return normvec_normalized: normalized normal vectors [x, y].
    :rtype normvec_normalized:  np.ndarray

    .. notes::
    Outputs are always unclosed!

    path and el_lengths inputs can either be closed or unclosed, but must be consistent! The function detects
    automatically if the path was inserted closed.

    Coefficient matrices have the form a_0i, a_1i * t, a_2i * t^2, a_3i * t^3.
    """

    # check if path is closed
    if np.all(np.isclose(path[0], path[-1])) and psi_s is None:
        closed = True
    else:
        closed = False

    # check inputs
    if not closed and (psi_s is None or psi_e is None):
        raise RuntimeError("Headings must be provided for unclosed spline calculation!")

    if el_lengths is not None and path.shape[0] != el_lengths.size + 1:
        raise RuntimeError("el_lengths input must be one element smaller than path input!")

    # if distances between path coordinates are not provided but required, calculate euclidean distances as el_lengths
    if use_dist_scaling and el_lengths is None:
        el_lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    elif el_lengths is not None:
        el_lengths = np.copy(el_lengths)

    # if closed and use_dist_scaling active append element length in order to obtain overlapping elements for proper
    # scaling of the last element afterwards
    if use_dist_scaling and closed:
        el_lengths = np.append(el_lengths, el_lengths[0])

    # get number of splines
    no_splines = path.shape[0] - 1

    # calculate scaling factors between every pair of splines
    if use_dist_scaling:
        scaling = el_lengths[:-1] / el_lengths[1:]
    else:
        scaling = np.ones(no_splines - 1)

    # ------------------------------------------------------------------------------------------------------------------
    # DEFINE LINEAR EQUATION SYSTEM ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M_{x,y} * a_{x,y} = b_{x,y}) with a_{x,y} being the desired spline param
    # *4 because of 4 parameters in cubic spline
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))

    # create template for M array entries
    # row 1: beginning of current spline should be placed on current point (t = 0)
    # row 2: end of current spline should be placed on next point (t = 1)
    # row 3: heading at end of current spline should be equal to heading at beginning of next spline (t = 1 and t = 0)
    # row 4: curvature at end of current spline should be equal to curvature at beginning of next spline (t = 1 and
    #        t = 0)
    template_M = np.array(                          # current point               | next point              | bounds
                [[1,  0,  0,  0,  0,  0,  0,  0],   # a_0i                                                  = {x,y}_i
                 [1,  1,  1,  1,  0,  0,  0,  0],   # a_0i + a_1i +  a_2i +  a_3i                           = {x,y}_i+1
                 [0,  1,  2,  3,  0, -1,  0,  0],   # _      a_1i + 2a_2i + 3a_3i      - a_1i+1             = 0
                 [0,  0,  2,  6,  0,  0, -2,  0]])  # _             2a_2i + 6a_3i               - 2a_2i+1   = 0

    for i in range(no_splines):
        j = i * 4

        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M

            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)

        else:
            # no curvature and heading bounds on last element (handled afterwards)
            M[j: j + 2, j: j + 4] = [[1,  0,  0,  0],
                                     [1,  1,  1,  1]]

        b_x[j: j + 2] = [[path[i,     0]],
                         [path[i + 1, 0]]]
        b_y[j: j + 2] = [[path[i,     1]],
                         [path[i + 1, 1]]]

    # ------------------------------------------------------------------------------------------------------------------
    # SET BOUNDARY CONDITIONS FOR LAST AND FIRST POINT -----------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not closed:
        # if the path is unclosed we want to fix heading at the start and end point of the path (curvature cannot be
        # determined in this case) -> set heading boundary conditions

        # heading start point
        M[-2, 1] = 1  # heading start point (evaluated at t = 0)

        if el_lengths is None:
            el_length_s = 1.0
        else:
            el_length_s = el_lengths[0]

        b_x[-2] = math.cos(psi_s + math.pi / 2) * el_length_s
        b_y[-2] = math.sin(psi_s + math.pi / 2) * el_length_s

        # heading end point
        M[-1, -4:] = [0, 1, 2, 3]  # heading end point (evaluated at t = 1)

        if el_lengths is None:
            el_length_e = 1.0
        else:
            el_length_e = el_lengths[-1]

        b_x[-1] = math.cos(psi_e + math.pi / 2) * el_length_e
        b_y[-1] = math.sin(psi_e + math.pi / 2) * el_length_e

    else:
        # heading boundary condition (for a closed spline)
        M[-2, 1] = scaling[-1]
        M[-2, -3:] = [-1, -2, -3]
        # b_x[-2] = 0
        # b_y[-2] = 0

        # curvature boundary condition (for a closed spline)
        M[-1, 2] = 2 * math.pow(scaling[-1], 2)
        M[-1, -2:] = [-2, -6]
        # b_x[-1] = 0
        # b_y[-1] = 0

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # M is block-banded -- each cubic couples only to its neighbour -- plus a handful
    # of corner entries from the closed-loop boundary conditions. A dense solve is
    # therefore O(n^3) on a matrix that is ~99.8 % zeros, and it was being run TWICE
    # on the same matrix (once per coordinate). Factorise once, sparsely, and reuse
    # it for both right-hand sides. Mathematically identical; agrees with the dense
    # path to ~1e-12. The dense fallback keeps this safe if SciPy's sparse solver is
    # unavailable or the factorisation fails.
    #
    # The triplets are gathered from the two places this function writes to M rather
    # than by scanning it: csc_matrix(M) has to walk all (4n)^2 entries to find the
    # ~12n nonzeros, which at 466 nodes costs 13 ms against 0.7 ms for the
    # factorisation it feeds. Every write above lands either in the 4x8 block band
    # (rows 4i..4i+3, columns 4i..4i+7) or in the final two boundary rows, which are
    # scanned in full, so the gather is complete by construction.
    try:
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.linalg import splu as _splu

        _m = no_splines * 4
        _rows_band = np.arange(_m - 2)
        _cols_band = ((_rows_band // 4) * 4)[:, None] + np.arange(8)[None, :]
        _in_band = _cols_band < _m
        _ri = np.broadcast_to(_rows_band[:, None], _cols_band.shape)[_in_band]
        _ci = _cols_band[_in_band]
        _vi = M[_ri, _ci]

        _rt, _ct = np.nonzero(M[_m - 2:])                        # the two boundary rows
        _ri = np.concatenate((_ri, _rt + (_m - 2)))
        _ci = np.concatenate((_ci, _ct))
        _vi = np.concatenate((_vi, M[_m - 2:][_rt, _ct]))

        _keep = _vi != 0.0
        _lu = _splu(_coo((_vi[_keep], (_ri[_keep], _ci[_keep])),
                         shape=(_m, _m)).tocsc())
        x_les = np.squeeze(_lu.solve(np.asarray(b_x, dtype=float).ravel()))
        y_les = np.squeeze(_lu.solve(np.asarray(b_y, dtype=float).ravel()))
    except Exception:                                            # pragma: no cover
        x_les = np.squeeze(np.linalg.solve(M, b_x))
        y_les = np.squeeze(np.linalg.solve(M, b_y))

    # get coefficients of every piece into one row -> reshape
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))

    # get normal vector (behind used here instead of ahead for consistency with other functions) (second coefficient of
    # cubic splines is relevant for the heading)
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)

    # normalize normal vectors
    norm_factors = 1.0 / np.sqrt(np.sum(np.power(normvec, 2), axis=1))
    normvec_normalized = np.expand_dims(norm_factors, axis=1) * normvec

    return coeffs_x, coeffs_y, M, normvec_normalized


_warned_nonuniform = False       # see create_raceline: warn once, not once per call


def create_raceline(refline: np.ndarray,
                    normvectors: np.ndarray,
                    alpha: np.ndarray,
                    stepsize_interp: float,
                    el_lengths: np.ndarray = None) -> tuple:
    """

    .. description::
    This function includes the algorithm part connected to the interpolation of the raceline after the optimization.

    .. inputs::
    :param refline:         array containing the track reference line [x, y] (unit is meter, must be unclosed!)
    :type refline:          np.ndarray
    :param normvectors:     normalized normal vectors for every point of the reference line [x_component, y_component]
                            (unit is meter, must be unclosed!)
    :type normvectors:      np.ndarray
    :param alpha:           solution vector of the optimization problem containing the lateral shift in m for every point.
    :type alpha:            np.ndarray
    :param stepsize_interp: stepsize in meters which is used for the interpolation after the raceline creation.
    :type stepsize_interp:  float

    .. outputs::
    :return raceline_interp:                interpolated raceline [x, y] in m.
    :rtype raceline_interp:                 np.ndarray
    :return A_raceline:                     linear equation system matrix of the splines on the raceline.
    :rtype A_raceline:                      np.ndarray
    :return coeffs_x_raceline:              spline coefficients of the x-component.
    :rtype coeffs_x_raceline:               np.ndarray
    :return coeffs_y_raceline:              spline coefficients of the y-component.
    :rtype coeffs_y_raceline:               np.ndarray
    :return spline_inds_raceline_interp:    contains the indices of the splines that hold the interpolated points.
    :rtype spline_inds_raceline_interp:     np.ndarray
    :return t_values_raceline_interp:       containts the relative spline coordinate values (t) of every point on the
                                            splines.
    :rtype t_values_raceline_interp:        np.ndarray
    :return s_raceline_interp:              total distance in m (i.e. s coordinate) up to every interpolation point.
    :rtype s_raceline_interp:               np.ndarray
    :return spline_lengths_raceline:        lengths of the splines on the raceline in m.
    :rtype spline_lengths_raceline:         np.ndarray
    :return el_lengths_raceline_interp_cl:  distance between every two points on interpolated raceline in m (closed!).
    :rtype el_lengths_raceline_interp_cl:   np.ndarray
    """

    # calculate raceline on the basis of the optimized alpha values
    raceline = refline + np.expand_dims(alpha, 1) * normvectors

    # calculate new splines on the basis of the raceline
    raceline_cl = np.vstack((raceline, raceline[0]))
    if el_lengths is None:
       coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline =calc_splines(path=raceline_cl,use_dist_scaling=False)
    else:
       coeffs_x_raceline, coeffs_y_raceline, A_raceline, normvectors_raceline =calc_splines(path=raceline_cl,el_lengths=el_lengths)

    # calculate new spline lengths
    spline_lengths_raceline =calc_spline_lengths(coeffs_x=coeffs_x_raceline,
                                                coeffs_y=coeffs_y_raceline)

    # interpolate splines for evenly spaced raceline points
    raceline_interp, spline_inds_raceline_interp, t_values_raceline_interp, s_raceline_interp = interp_splines(spline_lengths=spline_lengths_raceline,
                                      coeffs_x=coeffs_x_raceline,
                                      coeffs_y=coeffs_y_raceline,
                                      incl_last_point=False,
                                      stepsize_approx=stepsize_interp)

    # --- discretisation sanity check -------------------------------------------
    # A reference track with strongly non-uniform node spacing yields a short spline
    # segment whose cubic has to curl sharply to match its neighbours' tangents. The
    # resulting curvature spike is a genuine property of that spline, not an artefact
    # of the interpolation, but it silently corrupts any velocity profile computed
    # from it. A coarse stepsize_interp may never sample the offending segment, so the
    # problem can stay hidden until the step size is reduced.
    global _warned_nonuniform
    _sl_min = float(np.min(spline_lengths_raceline))
    _sl_max = float(np.max(spline_lengths_raceline))
    if _sl_min > 0.0 and _sl_max / _sl_min > 5.0 and not _warned_nonuniform:
        # Warn ONCE per session. This fires from inside the cost oracle of the
        # zeroth-order solvers, i.e. once per candidate evaluation -- thousands of
        # identical messages per run, which drowns the log and costs real time in
        # the warnings machinery. The condition is a property of the reference
        # track, so saying it once is enough.
        _warned_nonuniform = True
        warnings.warn(
            "create_raceline: reference-track node spacing is strongly non-uniform "
            "(longest/shortest spline segment = %.1f; shortest = %.3f m at index %d). "
            "Short segments make the spline curl sharply, which can corrupt the "
            "curvature and the lap time, especially at small stepsize_interp. "
            "Re-sample the reference track with (near-)uniform spacing."
            % (_sl_max / _sl_min, _sl_min, int(np.argmin(spline_lengths_raceline))),
            RuntimeWarning, stacklevel=2)

    # calculate element lengths
    s_tot_raceline = float(np.sum(spline_lengths_raceline))
    el_lengths_raceline_interp = np.diff(s_raceline_interp)
    el_lengths_raceline_interp_cl = np.append(el_lengths_raceline_interp, s_tot_raceline - s_raceline_interp[-1])

    return raceline_interp, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, \
           t_values_raceline_interp, s_raceline_interp, spline_lengths_raceline, el_lengths_raceline_interp_cl

def conv_filt(signal: np.ndarray,
              filt_window: int,
              closed: bool) -> np.ndarray:
    """

    .. description::
    Filter a given temporal signal using a convolution (moving average) filter.

    .. inputs::
    :param signal:          temporal signal that should be filtered (always unclosed).
    :type signal:           np.ndarray
    :param filt_window:     filter window size for moving average filter (must be odd).
    :type filt_window:      int
    :param closed:          flag showing if the signal can be considered as closable, e.g. for velocity profiles.
    :type closed:           bool

    .. outputs::
    :return signal_filt:    filtered input signal (always unclosed).
    :rtype signal_filt:     np.ndarray

    .. notes::
    signal input is always unclosed!

    len(signal) = len(signal_filt)
    """

    # check if window width is odd
    if not filt_window % 2 == 1:
        raise RuntimeError("Window width of moving average filter must be odd!")

    # calculate half window width - 1
    w_window_half = int((filt_window - 1) / 2)

    # apply filter
    if closed:
        # temporarily add points in front of and behind signal
        signal_tmp = np.concatenate((signal[-w_window_half:], signal, signal[:w_window_half]), axis=0)

        # apply convolution filter used as a moving average filter and remove temporary points
        signal_filt = np.convolve(signal_tmp,
                                  np.ones(filt_window) / float(filt_window),
                                  mode="same")[w_window_half:-w_window_half]

    else:
        # implementation 1: include boundaries during filtering
        # no_points = signal.size
        # signal_filt = np.zeros(no_points)
        #
        # for i in range(no_points):
        #     if i < w_window_half:
        #         signal_filt[i] = np.average(signal[:i + w_window_half + 1])
        #
        #     elif i < no_points - w_window_half:
        #         signal_filt[i] = np.average(signal[i - w_window_half:i + w_window_half + 1])
        #
        #     else:
        #         signal_filt[i] = np.average(signal[i - w_window_half:])

        # implementation 2: start filtering at w_window_half and stop at -w_window_half
        signal_filt = np.copy(signal)
        signal_filt[w_window_half:-w_window_half] = np.convolve(signal,
                                                                np.ones(filt_window) / float(filt_window),
                                                                mode="same")[w_window_half:-w_window_half]

    return signal_filt

def import_veh_dyn_info(ggv_import_path: str = None,
                        ax_max_machines_import_path: str = None) -> tuple:
    """
    .. description::
    This function imports the required vehicle dynamics information from several files: The vehicle ggv diagram
    ([vx, ax_max, ay_max], velocity in m/s, accelerations in m/s2) and the ax_max_machines array containing the
    longitudinal acceleration limits by the electrical motors ([vx, ax_max_machines], velocity in m/s, acceleration in
    m/s2).

    .. inputs::
    :param ggv_import_path:             Path to the ggv csv file.
    :type ggv_import_path:              str
    :param ax_max_machines_import_path: Path to the ax_max_machines csv file.
    :type ax_max_machines_import_path:  str

    .. outputs::
    :return ggv:                        ggv diagram
    :rtype ggv:                         np.ndarray
    :return ax_max_machines:            ax_max_machines array
    :rtype ax_max_machines:             np.ndarray
    """

    # GGV --------------------------------------------------------------------------------------------------------------
    if ggv_import_path is not None:

        # load csv
        with open(ggv_import_path, "rb") as fh:
            ggv = np.loadtxt(fh, comments='#', delimiter=",")

        # expand dimension in case of a single row
        if ggv.ndim == 1:
            ggv = np.expand_dims(ggv, 0)

        # check columns
        if ggv.shape[1] != 3:
            raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

        # check values
        invalid_1 = ggv[:, 0] < 0.0     # assure velocities > 0.0
        invalid_2 = ggv[:, 1:] > 50.0   # assure valid maximum accelerations
        invalid_3 = ggv[:, 1] < 0.0     # assure positive accelerations
        invalid_4 = ggv[:, 2] < 0.0     # assure positive accelerations

        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3) or np.any(invalid_4):
            raise RuntimeError("ggv seems unreasonable!")

    else:
        ggv = None

    # AX_MAX_MACHINES --------------------------------------------------------------------------------------------------
    if ax_max_machines_import_path is not None:

        # load csv
        with open(ax_max_machines_import_path, "rb") as fh:
            ax_max_machines = np.loadtxt(fh, comments='#',  delimiter=",")

        # expand dimension in case of a single row
        if ax_max_machines.ndim == 1:
            ax_max_machines = np.expand_dims(ax_max_machines, 0)

        # check columns
        if ax_max_machines.shape[1] != 2:
            raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

        # check values
        invalid_1 = ax_max_machines[:, 0] < 0.0     # assure velocities > 0.0
        invalid_2 = ax_max_machines[:, 1] > 20.0    # assure valid maximum accelerations
        invalid_3 = ax_max_machines[:, 1] < 0.0     # assure positive accelerations

        if np.any(invalid_1) or np.any(invalid_2) or np.any(invalid_3):
            raise RuntimeError("ax_max_machines seems unreasonable!")

    else:
        ax_max_machines = None

    return ggv, ax_max_machines


def normalize_psi(psi: Union[np.ndarray, float]) -> np.ndarray:
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)

    # restrict psi to [-pi,pi[
    if type(psi_out) is np.ndarray:
        psi_out[psi_out >= math.pi] -= 2 * math.pi
        psi_out[psi_out < -math.pi] += 2 * math.pi

    else:
        if psi_out >= math.pi:
            psi_out -= 2 * math.pi
        elif psi_out < -math.pi:
            psi_out += 2 * math.pi

    return psi_out

def calc_head_curv_an(coeffs_x: np.ndarray,
                      coeffs_y: np.ndarray,
                      ind_spls: np.ndarray,
                      t_spls: np.ndarray,
                      calc_curv: bool = True,
                      calc_dcurv: bool = False) -> tuple:
    """

    .. description::
    Analytical calculation of heading psi, curvature kappa, and first derivative of the curvature dkappa
    on the basis of third order splines for x- and y-coordinate.

    .. inputs::
    :param coeffs_x:    coefficient matrix of the x splines with size (no_splines x 4).
    :type coeffs_x:     np.ndarray
    :param coeffs_y:    coefficient matrix of the y splines with size (no_splines x 4).
    :type coeffs_y:     np.ndarray
    :param ind_spls:    contains the indices of the splines that hold the points for which we want to calculate heading/curv.
    :type ind_spls:     np.ndarray
    :param t_spls:      containts the relative spline coordinate values (t) of every point on the splines.
    :type t_spls:       np.ndarray
    :param calc_curv:   bool flag to show if curvature should be calculated as well (kappa is set 0.0 otherwise).
    :type calc_curv:    bool
    :param calc_dcurv:  bool flag to show if first derivative of curvature should be calculated as well.
    :type calc_dcurv:   bool

    .. outputs::
    :return psi:        heading at every point.
    :rtype psi:         float
    :return kappa:      curvature at every point.
    :rtype kappa:       float
    :return dkappa:     first derivative of curvature at every point (if calc_dcurv bool flag is True).
    :rtype dkappa:      float

    .. notes::
    len(ind_spls) = len(t_spls) = len(psi) = len(kappa) = len(dkappa)
    """

    # check inputs
    if coeffs_x.shape[0] != coeffs_y.shape[0]:
        raise ValueError("Coefficient matrices must have the same length!")

    if ind_spls.size != t_spls.size:
        raise ValueError("ind_spls and t_spls must have the same length!")

    if not calc_curv and calc_dcurv:
        raise ValueError("dkappa cannot be calculated without kappa!")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE HEADING ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate required derivatives
    x_d = coeffs_x[ind_spls, 1] \
          + 2 * coeffs_x[ind_spls, 2] * t_spls \
          + 3 * coeffs_x[ind_spls, 3] * np.power(t_spls, 2)

    y_d = coeffs_y[ind_spls, 1] \
          + 2 * coeffs_y[ind_spls, 2] * t_spls \
          + 3 * coeffs_y[ind_spls, 3] * np.power(t_spls, 2)

    # calculate heading psi (pi/2 must be substracted due to our convention that psi = 0 is north)
    psi = np.arctan2(y_d, x_d) - math.pi / 2
    psi = normalize_psi(psi)

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CURVATURE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if calc_curv:
        # calculate required derivatives
        x_dd = 2 * coeffs_x[ind_spls, 2] \
               + 6 * coeffs_x[ind_spls, 3] * t_spls

        y_dd = 2 * coeffs_y[ind_spls, 2] \
               + 6 * coeffs_y[ind_spls, 3] * t_spls

        # calculate curvature kappa
        kappa = (x_d * y_dd - y_d * x_dd) / np.power(np.power(x_d, 2) + np.power(y_d, 2), 1.5)

    else:
        kappa = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE FIRST DERIVATIVE OF CURVATURE --------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if calc_dcurv:
        # calculate required derivatives
        x_ddd = 6 * coeffs_x[ind_spls, 3]

        y_ddd = 6 * coeffs_y[ind_spls, 3]

        # calculate first derivative of curvature dkappa
        dkappa = ((np.power(x_d, 2) + np.power(y_d, 2)) * (x_d * y_ddd - y_d * x_ddd) -
                  3 * (x_d * y_dd - y_d * x_dd) * (x_d * x_dd + y_d * y_dd)) / \
                 np.power(np.power(x_d, 2) + np.power(y_d, 2), 3)

        return psi, kappa, dkappa

    else:

        return psi, kappa



def spline_extraction_ops(A: np.ndarray,
                          no_points: int,
                          closed: bool = True) -> tuple:
    """
    .. description::
    Return ``(A_inv, T_b, T_c)`` for a spline system matrix ``A``, where

        T_b = A_ex_b @ A_inv        picks the b_i coefficients (first derivative)
        T_c = A_ex_c @ A_inv        picks 2 * c_i   (second derivative / curvature)

    ``A_ex_b`` and ``A_ex_c`` are the extraction matrices used by the minimum
    curvature formulation. For a **closed** track each of their rows holds exactly
    one nonzero -- 1 at column 4i+1 and 2 at column 4i+2 -- so both products are a
    strided row slice of ``A_inv``, scaled by 1 or by 2. Both factors are powers
    of two, and the omitted terms of the dot product are exact zeros, so the
    result is *bitwise* identical to the ``(n x 4n) @ (4n x 4n)`` matmul it
    replaces while costing O(n^2) instead of O(n^3). At 466 nodes that is ~1.6e9
    flops and two 7 MB temporaries saved per call.

    The **unclosed** case carries an extra end-of-spline row ([0,1,2,3] and
    [0,0,2,6]) that mixes four coefficients, so that branch builds the extraction
    matrices and falls back to the original matmul rather than risk a different
    summation order.

    Callers that need several of these products should fetch them once here: the
    inverse is by far the most expensive part, and the minimum curvature path used
    to form it twice per solve.

    .. inputs::
    :param A:           spline system matrix from :func:`calc_splines`.
    :param no_points:   number of reference track points.
    :param closed:      whether the track is closed.

    .. outputs::
    :return A_inv:      inverse of A.
    :return T_b:        A_ex_b @ A_inv.
    :return T_c:        A_ex_c @ A_inv.
    """
    A_inv = np.linalg.inv(A)

    if closed:
        return A_inv, A_inv[1::4], 2.0 * A_inv[2::4]

    no_splines = no_points - 1

    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)
    A_ex_b[np.arange(no_splines), np.arange(no_splines) * 4 + 1] = 1
    A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)
    A_ex_c[np.arange(no_splines), np.arange(no_splines) * 4 + 2] = 2
    A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    return A_inv, np.matmul(A_ex_b, A_inv), np.matmul(A_ex_c, A_inv)


def interp_corridor_rows(A: np.ndarray,
                         reftrack: np.ndarray,
                         normvectors: np.ndarray,
                         w_veh: float,
                         closed: bool = True,
                         n_interp_con: int = 2,
                         A_inv: np.ndarray = None) -> tuple:
    """
    .. description::
    Build linear inequality rows that keep the *interpolated* raceline inside the
    track corridor, not merely the raceline at the optimisation nodes.

    The QPs of this module constrain alpha only at the reference-track nodes. The
    raceline actually driven is the cubic spline through those nodes, and between
    two nodes that spline bulges outside the corridor -- by up to ~0.27 m on an
    18 m node spacing. Reducing the node spacing fixes this but scales the number
    of optimisation variables, which is expensive.

    Instead we exploit the fact that the spline coefficients solve A @ coeffs =
    q + M @ alpha, so that the position at any FIXED (segment, t) is affine in
    alpha. Measuring the deviation from the reference spline at the same (segment,
    t) cancels the constant part, leaving a purely linear map alpha -> deviation.
    Bounding that map at a few interior t values adds constraint ROWS while
    leaving the number of variables unchanged.

    .. inputs::
    :param A:               spline system matrix from calc_splines (same as opt_min_curv).
    :param reftrack:        reference track [x, y, w_tr_right, w_tr_left].
    :param normvectors:     normalised normal vectors of the reference track.
    :param w_veh:           vehicle width in m (per-side clearance is w_veh / 2).
    :param closed:          whether the track is closed.
    :param n_interp_con:    number of interior points constrained per spline segment.
                            0 disables the extra rows and restores the node-only
                            behaviour; 2 constrains t = 1/3 and t = 2/3.
    :param A_inv:           inverse of A, if the caller already has it. Inverting A
                            is the dominant cost here, so a caller that needs it
                            anyway (the minimum curvature QP does) should pass it
                            in rather than have it formed a second time.

    .. outputs::
    :return G_add:          additional inequality rows (2 * n_interp_con * no_splines, no_points).
    :return h_add:          corresponding right-hand sides.
    """
    no_points = reftrack.shape[0]
    no_splines = no_points if closed else no_points - 1

    if n_interp_con < 1:
        return np.zeros((0, no_points)), np.zeros(0)

    # Only the columns 0::4 and 1::4 of A_inv are ever reached below, because M_x and
    # M_y are zero in every other row. Restricting the product to them, or solving A
    # for just those 2n identity columns instead of inverting it, is ~2x and ~1.6x
    # respectively -- but neither was bitwise exact when measured: OpenBLAS's packed
    # dgemm kernels order their k-blocks differently for K = 2n than for K = 4n
    # (1e-16 on 15 % of entries), and LAPACK's triangular solve likewise blocks
    # differently for 2n right-hand sides. These numbers go into the constraint
    # matrix of an active-set QP, where a last-bit nudge can change which
    # constraints are active, so both shortcuts were dropped. Passing A_inv in from
    # a caller that already has it (H_f does) is the part that is exact.
    if A_inv is None:
        A_inv = np.linalg.inv(A)

    # M_x / M_y map alpha onto the spline right-hand side (see opt_min_curv)
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))
    rows_0 = np.arange(no_splines) * 4
    rows_1 = rows_0 + 1
    cols_0 = np.arange(no_splines)
    cols_1 = np.arange(1, no_splines + 1) % no_points
    M_x[rows_0, cols_0] = normvectors[cols_0, 0]
    M_x[rows_1, cols_1] = normvectors[cols_1, 0]
    M_y[rows_0, cols_0] = normvectors[cols_0, 1]
    M_y[rows_1, cols_1] = normvectors[cols_1, 1]

    G_add, h_add = [], []
    for t in (np.arange(1, n_interp_con + 1) / (n_interp_con + 1.0)):
        # (E_t @ A_inv), where E_t evaluates [1, t, t^2, t^3] on each spline segment
        EA = (A_inv[0::4] + t * A_inv[1::4]
              + t ** 2 * A_inv[2::4] + t ** 3 * A_inv[3::4])
        D_x = np.matmul(EA, M_x)
        D_y = np.matmul(EA, M_y)

        # normal direction at the interior point (interpolated between nodes)
        n_x = (1.0 - t) * normvectors[cols_0, 0] + t * normvectors[cols_1, 0]
        n_y = (1.0 - t) * normvectors[cols_0, 1] + t * normvectors[cols_1, 1]
        n_norm = np.hypot(n_x, n_y)
        n_norm[n_norm < 1e-12] = 1.0
        n_x, n_y = n_x / n_norm, n_y / n_norm

        # deviation of the interpolated point along that normal (linear in alpha)
        B = D_x * n_x[:, None] + D_y * n_y[:, None]

        # corridor half-widths at the interior point, linearly interpolated
        dev_pos = (1.0 - t) * reftrack[cols_0, 2] + t * reftrack[cols_1, 2] - w_veh / 2
        dev_neg = (1.0 - t) * reftrack[cols_0, 3] + t * reftrack[cols_1, 3] - w_veh / 2
        dev_pos = np.maximum(dev_pos, 0.001)
        dev_neg = np.maximum(dev_neg, 0.001)

        G_add.append(B);  h_add.append(dev_pos)
        G_add.append(-B); h_add.append(dev_neg)

    return np.vstack(G_add), np.concatenate(h_add)

def H_f(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float,
                 print_debug: bool = False,
                 plot_debug: bool = False,
                 closed: bool = True,
                 psi_s: float = None,
                 psi_e: float = None,
                 fix_s: bool = False,
                 fix_e: bool = False,
                 n_interp_con: int = 2) -> tuple:

    """
    .. description::
    This function uses outputs the data neede to solve the min curvature problem

    Please refer to the paper for further information:
    Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann
    Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar
    DOI: 10.1080/00423114.2019.1631455

    .. inputs::
    :param reftrack:    array containing the reference track, i.e. a reference line and the according track widths to
                        the right and to the left [x, y, w_tr_right, w_tr_left] (unit is meter, must be unclosed!)
    :type reftrack:     np.ndarray
    :param normvectors: normalized normal vectors for every point of the reference track [x_component, y_component]
                        (unit is meter, must be unclosed!)
    :type normvectors:  np.ndarray
    :param A:           linear equation system matrix for splines (applicable for both, x and y direction)
                        -> System matrices have the form a_i, b_i * t, c_i * t^2, d_i * t^3
                        -> see calc_splines.py for further information or to obtain this matrix
    :type A:            np.ndarray
    :param kappa_bound: curvature boundary to consider during optimization.
    :type kappa_bound:  float
    :param w_veh:       vehicle width in m. It is considered during the calculation of the allowed deviations from the
                        reference line.
    :type w_veh:        float
    :param print_debug: bool flag to print debug messages.
    :type print_debug:  bool
    :param plot_debug:  bool flag to plot the curvatures that are calculated based on the original linearization and on
                        a linearization around the solution.
    :type plot_debug:   bool
    :param closed:      bool flag specifying whether a closed or unclosed track should be assumed
    :type closed:       bool
    :param psi_s:       heading to be enforced at the first point for unclosed tracks
    :type psi_s:        float
    :param psi_e:       heading to be enforced at the last point for unclosed tracks
    :type psi_e:        float
    :param fix_s:       determines if start point is fixed to reference line for unclosed tracks
    :type fix_s:        bool
    :param fix_e:       determines if last point is fixed to reference line for unclosed tracks
    :type fix_e:        bool

    .. outputs::
    :return H:  Matrix H defined in the mentioned paper
    :rtype alpha_mincurv:   np.ndarray
    :return f: Matrix f defined in the mentioned paper
    :rtype curv_error_max:  np.ndarray
    :return G:  Constrait matrix 
    :rtype alpha_mincurv:   np.ndarray
    :return h:  Constraint on norm of alpha defined in thementioned paper
    :rtype curv_error_max:  np.ndarray
    """   

    no_points = reftrack.shape[0]

    no_splines = no_points
    if not closed:
        no_splines -= 1

    # check inputs
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")

    if (no_points * 4 != A.shape[0] and closed) or (no_splines * 4 != A.shape[0] and not closed)\
            or A.shape[0] != A.shape[1]:
        raise RuntimeError("Spline equation system matrix A has wrong dimensions!")

    # invert A once and extract the b_i / 2*c_i coefficient rows. See
    # spline_extraction_ops: for a closed track those two products are exact row
    # slices of A_inv, not matmuls, and A_inv is reused by the corridor rows below
    # instead of being formed a second time.
    A_inv, T_b, T_c = spline_extraction_ops(A, no_points, closed=closed)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    rows_0 = np.arange(no_splines) * 4
    rows_1 = rows_0 + 1
    cols_0 = np.arange(no_splines)
    cols_1 = np.arange(1, no_splines + 1) % no_points   # wraps to 0 on a closed track

    M_x[rows_0, cols_0] = normvectors[cols_0, 0]
    M_x[rows_1, cols_1] = normvectors[cols_1, 0]
    M_y[rows_0, cols_0] = normvectors[cols_0, 1]
    M_y[rows_1, cols_1] = normvectors[cols_1, 1]

    # set up q_x and q_y matrices including the point coordinate information
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))

    q_x[rows_0, 0] = reftrack[cols_0, 0]
    q_x[rows_1, 0] = reftrack[cols_1, 0]
    q_y[rows_0, 0] = reftrack[cols_0, 1]
    q_y[rows_1, 0] = reftrack[cols_1, 1]

    # for unclosed tracks, specify start- and end-heading constraints
    if not closed:
        q_x[-2, 0] = math.cos(psi_s + math.pi / 2)
        q_y[-2, 0] = math.sin(psi_s + math.pi / 2)

        q_x[-1, 0] = math.cos(psi_e + math.pi / 2)
        q_y[-1, 0] = math.sin(psi_e + math.pi / 2)

    # set up P_xx, P_xy, P_yy matrices
    #
    # x_prime / y_prime are built as np.eye(n) * (n, 1), i.e. they are DIAGONAL --
    # the code below used to carry them, and everything derived from them (P_xx,
    # P_yy, P_xy, Q_x, Q_y), as full n x n matrices and multiply them with
    # np.matmul. Every entry of such a product has a single nonzero term, so
    # holding the diagonal as a vector and scaling rows gives bitwise identical
    # numbers while turning ~8 O(n^3) matmuls into O(n) / O(n^2) work.
    x_prime = np.squeeze(np.matmul(T_b, q_x), axis=1)       # the diagonal of x_prime
    y_prime = np.squeeze(np.matmul(T_b, q_y), axis=1)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * (x_prime * y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # divide where not zero (diag elements)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = curv_part_sq * y_prime_sq
    P_yy = curv_part_sq * x_prime_sq
    P_xy = curv_part_sq * x_prime_y_prime

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    # P_* are diagonal, so these are row scalings; each is also needed twice (once
    # for H, once for f), so they are formed once here.
    P_xx_T_nx = P_xx[:, None] * T_nx
    P_yy_T_ny = P_yy[:, None] * T_ny
    P_xy_T_nx = P_xy[:, None] * T_nx
    P_xy_T_ny = P_xy[:, None] * T_ny

    H_x = np.matmul(T_nx.T, P_xx_T_nx)
    H_xy = np.matmul(T_ny.T, P_xy_T_nx)
    H_y = np.matmul(T_ny.T, P_yy_T_ny)
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # make H symmetric

    q_x_T_c = np.matmul(q_x.T, T_c.T)
    q_y_T_c = np.matmul(q_y.T, T_c.T)

    f_x = 2 * np.matmul(q_x_T_c, P_xx_T_nx)
    f_xy = np.matmul(q_x_T_c, P_xy_T_ny) \
           + np.matmul(q_y_T_c, P_xy_T_nx)
    f_y = 2 * np.matmul(q_y_T_c, P_yy_T_ny)
    f = f_x + f_xy + f_y
    f = np.squeeze(f)   # remove non-singleton dimensions

    Q_x = curv_part * y_prime                               # diagonal, as above
    Q_y = curv_part * x_prime

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = Q_y[:, None] * T_ny - Q_x[:, None] * T_nx

    # original curvature part (static part)
    k_kappa_ref = Q_y[:, None] * np.matmul(T_c, q_y) - Q_x[:, None] * np.matmul(T_c, q_x)

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # check that there is space remaining between left and right maximum deviation (both can be negative as well!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")

    # consider value boundaries (dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # keep the INTERPOLATED spline inside the corridor as well, not only the nodes
    G_ic, h_ic = interp_corridor_rows(A=A, reftrack=reftrack, normvectors=normvectors,
                                      w_veh=w_veh, closed=closed,
                                      n_interp_con=n_interp_con, A_inv=A_inv)
    if G_ic.shape[0] > 0:
        G = np.vstack((G, G_ic))
        h = np.append(h, h_ic)

    return H , f, G ,h

def interp_track_widths(w_track: np.ndarray,
                        spline_inds: np.ndarray,
                        t_values: np.ndarray,
                        incl_last_point: bool = False) -> np.ndarray:
    """
    .. description::
    The function (linearly) interpolates the track widths in the same steps as the splines were interpolated before.

    Keep attention that the (multiple) interpolation of track widths can lead to unwanted effects, e.g. that peaks
    in the track widths can disappear if the stepsize is too large (kind of an aliasing effect).

    .. inputs::
    :param w_track:         array containing the track widths in meters [w_track_right, w_track_left] to interpolate,
                            optionally with banking angle in rad: [w_track_right, w_track_left, banking]
    :type w_track:          np.ndarray
    :param spline_inds:     indices that show which spline (and here w_track element) shall be interpolated.
    :type spline_inds:      np.ndarray
    :param t_values:        relative spline coordinate values (t) of every point on the splines specified by spline_inds
    :type t_values:         np.ndarray
    :param incl_last_point: bool flag to show if last point should be included or not.
    :type incl_last_point:  bool

    .. outputs::
    :return w_track_interp: array with interpolated track widths (and optionally banking angle).
    :rtype w_track_interp:  np.ndarray

    .. notes::
    All inputs are unclosed.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE INTERMEDIATE STEPS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    w_track_cl = np.vstack((w_track, w_track[0]))
    no_interp_points = t_values.size  # unclosed

    if incl_last_point:
        w_track_interp = np.zeros((no_interp_points + 1, w_track.shape[1]))
        w_track_interp[-1] = w_track_cl[-1]
    else:
        w_track_interp = np.zeros((no_interp_points, w_track.shape[1]))

    # vectorized linear interpolation: w0 + t*(w1-w0)
    w0 = w_track_cl[spline_inds]
    w1 = w_track_cl[spline_inds + 1]
    w_track_interp[:no_interp_points] = w0 + t_values[:, np.newaxis] * (w1 - w0)

    return w_track_interp

import quadprog
def New_reftrack(reftrack: np.ndarray,
              ds: np.ndarray,
              interp_step: float,
              kappa_bound:float,
              wveh: float) -> np.ndarray:
    """

    .. description::
    Modify the reftrack for reoptimisation

    .. inputs::
    :param signal:          temporal signal that should be filtered (always unclosed).
    :type signal:           np.ndarray


    .. outputs::
    :return signal_filt:    filtered input signal (always unclosed).
    :rtype signal_filt:     np.ndarray

    """
    kapb = kappa_bound
    sfty = wveh
    si = interp_step
    reftrack_tmp = reftrack
    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((reftrack[:, 0:2],reftrack[0, 0:2])),el_lengths=ds)
    H, f, G , h = H_f(reftrack=reftrack,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=kapb,
                                                 w_veh=sfty,
                                                 closed=True)
                                                 

    alpha = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]
    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=reftrack[:, :2],
                    normvectors=normvec_norm,
                    alpha=alpha,
                    stepsize_interp=si)
    
    reftrack_tmp[:, 2] -= alpha
    reftrack_tmp[:, 3] += alpha

    ws_track_tmp = interp_track_widths(w_track=reftrack_tmp[:, 2:],
                                        spline_inds=spline_inds_opt_interp,
                                        t_values=t_vals_opt_interp,
                                        incl_last_point=False)

        # create new reftrack
    reftrack_tmp = np.column_stack((raceline_interp, ws_track_tmp))


    return reftrack_tmp

def nonreg_sampling(track: np.ndarray,
                    eps_kappa: float = 1e-3,
                    step_non_reg: int = 0) -> tuple:
    """
    .. description::
    The non-regular sampling function runs through the curvature profile and determines straight and corner sections.
    During straight sections it reduces the amount of points by skipping them depending on the step_non_reg parameter.

    .. inputs::
    :param track:           [x, y, w_tr_right, w_tr_left] (always unclosed).
    :type track:            np.ndarray
    :param eps_kappa:       identify straights using this threshold in curvature in rad/m, i.e. straight if
                            kappa < eps_kappa
    :type eps_kappa:        float
    :param step_non_reg:    determines how many points are skipped in straight sections, e.g. step_non_reg = 3 means
                            every fourth point is used while three points are skipped
    :type step_non_reg:     int

    .. outputs::
    :return track_sampled:  [x, y, w_tr_right, w_tr_left] sampled track (always unclosed).
    :rtype track_sampled:   np.ndarray
    :return sample_idxs:    indices of points that are kept
    :rtype sample_idxs:     np.ndarray
    """

    # if stepsize is equal to zero simply return the input
    if step_non_reg == 0:
        return track, np.arange(0, track.shape[0])

    # calculate curvature (required to be able to differentiate straight and corner sections)
    path_cl = np.vstack((track[:, :2], track[0, :2]))
    coeffs_x, coeffs_y = calc_splines(path=path_cl)[:2]
    kappa_path = calc_head_curv_an(coeffs_x=coeffs_x,
                                                         coeffs_y=coeffs_y,
                                                         ind_spls=np.arange(0, coeffs_x.shape[0]),
                                                         t_spls=np.zeros(coeffs_x.shape[0]))[1]

    # run through the profile to determine the indices of the points that are kept
    idx_latest = step_non_reg + 1
    sample_idxs = [0]

    for idx in range(1, len(kappa_path)):
        if np.abs(kappa_path[idx]) >= eps_kappa or idx >= idx_latest:
            # keep this point
            sample_idxs.append(idx)
            idx_latest = idx + step_non_reg + 1

    return track[sample_idxs], np.array(sample_idxs)


def calc_head_curv_num(path: np.ndarray,
                       el_lengths: np.ndarray,
                       is_closed: bool,
                       stepsize_psi_preview: float = 1.0,
                       stepsize_psi_review: float = 1.0,
                       stepsize_curv_preview: float = 2.0,
                       stepsize_curv_review: float = 2.0,
                       calc_curv: bool = True) -> tuple:
    """
    .. description::
    Numerical calculation of heading psi and curvature kappa on the basis of a given path.

    .. inputs::
    :param path:                    array of points [x, y] (always unclosed).
    :type path:                     np.ndarray
    :param el_lengths:              array containing the element lengths.
    :type el_lengths:               np.ndarray
    :param is_closed:               close path for heading and curvature calculation.
    :type is_closed:                bool
    :param stepsize_psi_preview:    preview/review distances used for numerical heading/curvature calculation.
    :type stepsize_psi_preview:     float
    :param stepsize_psi_review:     preview/review distances used for numerical heading/curvature calculation.
    :type stepsize_psi_review:      float
    :param stepsize_curv_preview:   preview/review distances used for numerical heading/curvature calculation.
    :type stepsize_curv_preview:    float
    :param stepsize_curv_review:    preview/review distances used for numerical heading/curvature calculation.
    :type stepsize_curv_review:     float
    :param calc_curv:               bool flag to show if curvature should be calculated (kappa is set 0.0 otherwise).
    :type calc_curv:                bool

    .. outputs::
    :return psi:                    heading at every point (always unclosed).
    :rtype psi:                     float
    :return kappa:                  curvature at every point (always unclosed).
    :rtype kappa:                   float

    .. notes::
    path must be inserted unclosed, i.e. path[-1] != path[0], even if is_closed is set True! (el_lengths is kind
    of closed if is_closed is True of course!)

    case is_closed is True:
    len(path) = len(el_lengths) = len(psi) = len(kappa)

    case is_closed is False:
    len(path) = len(el_lengths) + 1 = len(psi) = len(kappa)
    """

    # check inputs
    if is_closed and path.shape[0] != el_lengths.size:
        raise RuntimeError("path and el_lenghts must have the same length!")

    elif not is_closed and path.shape[0] != el_lengths.size + 1:
        raise RuntimeError("path must have the length of el_lengths + 1!")

    # get number if points
    no_points = path.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    # CASE: CLOSED PATH ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if is_closed:

        # --------------------------------------------------------------------------------------------------------------
        # PREVIEW/REVIEW DISTANCES -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate how many points we look to the front and rear of the current position for the head/curv calculations
        ind_step_preview_psi = round(stepsize_psi_preview / float(np.average(el_lengths)))
        ind_step_review_psi = round(stepsize_psi_review / float(np.average(el_lengths)))
        ind_step_preview_curv = round(stepsize_curv_preview / float(np.average(el_lengths)))
        ind_step_review_curv = round(stepsize_curv_review / float(np.average(el_lengths)))

        ind_step_preview_psi = max(ind_step_preview_psi, 1)
        ind_step_review_psi = max(ind_step_review_psi, 1)
        ind_step_preview_curv = max(ind_step_preview_curv, 1)
        ind_step_review_curv = max(ind_step_review_curv, 1)

        steps_tot_psi = ind_step_preview_psi + ind_step_review_psi
        steps_tot_curv = ind_step_preview_curv + ind_step_review_curv

        # --------------------------------------------------------------------------------------------------------------
        # HEADING ------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate tangent vectors for every point
        path_temp = np.vstack((path[-ind_step_review_psi:], path, path[:ind_step_preview_psi]))
        tangvecs = np.stack((path_temp[steps_tot_psi:, 0] - path_temp[:-steps_tot_psi, 0],
                             path_temp[steps_tot_psi:, 1] - path_temp[:-steps_tot_psi, 1]), axis=1)

        # calculate psi of tangent vectors (pi/2 must be substracted due to our convention that psi = 0 is north)
        psi = np.arctan2(tangvecs[:, 1], tangvecs[:, 0]) - math.pi / 2
        psi = normalize_psi(psi)

        # --------------------------------------------------------------------------------------------------------------
        # CURVATURE ----------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if calc_curv:
            psi_temp = np.insert(psi, 0, psi[-ind_step_review_curv:])
            psi_temp = np.append(psi_temp, psi[:ind_step_preview_curv])

            # calculate delta psi
            delta_psi = normalize_psi(psi_temp[steps_tot_curv:steps_tot_curv + no_points] - psi_temp[:no_points])

            # calculate kappa
            s_points_cl = np.cumsum(el_lengths)
            s_points_cl = np.insert(s_points_cl, 0, 0.0)
            s_points = s_points_cl[:-1]
            s_points_cl_reverse = np.flipud(-np.cumsum(np.flipud(el_lengths)))  # should not include 0.0 as last value

            s_points_temp = np.insert(s_points, 0, s_points_cl_reverse[-ind_step_review_curv:])
            s_points_temp = np.append(s_points_temp, s_points_cl[-1] + s_points[:ind_step_preview_curv])

            kappa = delta_psi / (s_points_temp[steps_tot_curv:] - s_points_temp[:-steps_tot_curv])

        else:
            kappa = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # CASE: UNCLOSED PATH ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    else:

        # --------------------------------------------------------------------------------------------------------------
        # HEADING ------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # calculate tangent vectors for every point
        tangvecs = np.zeros((no_points, 2))

        tangvecs[0, 0] = path[1, 0] - path[0, 0]  # i == 0
        tangvecs[0, 1] = path[1, 1] - path[0, 1]

        tangvecs[1:-1, 0] = path[2:, 0] - path[:-2, 0]  # 0 < i < no_points - 1
        tangvecs[1:-1, 1] = path[2:, 1] - path[:-2, 1]

        tangvecs[-1, 0] = path[-1, 0] - path[-2, 0]  # i == -1
        tangvecs[-1, 1] = path[-1, 1] - path[-2, 1]

        # calculate psi of tangent vectors (pi/2 must be substracted due to our convention that psi = 0 is north)
        psi = np.arctan2(tangvecs[:, 1], tangvecs[:, 0]) - math.pi / 2
        psi = normalize_psi(psi)

        # --------------------------------------------------------------------------------------------------------------
        # CURVATURE ----------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if calc_curv:
            # calculate delta psi
            delta_psi = np.zeros(no_points)

            delta_psi[0] = psi[1] - psi[0]  # i == 0
            delta_psi[1:-1] = psi[2:] - psi[:-2]  # 0 < i < no_points - 1
            delta_psi[-1] = psi[-1] - psi[-2]  # i == -1

            # normalize delta_psi
            delta_psi = normalize_psi(delta_psi)

            # calculate kappa
            kappa = np.zeros(no_points)

            kappa[0] = delta_psi[0] / el_lengths[0]  # i == 0
            kappa[1:-1] = delta_psi[1:-1] / (el_lengths[1:] + el_lengths[:-1])  # 0 < i < no_points - 1
            kappa[-1] = delta_psi[-1] / el_lengths[-1]  # i == -1

        else:
            kappa = 0.0

    return psi, kappa


def check_normals_crossing(track: np.ndarray,
                           normvec_normalized: np.ndarray,
                           horizon: int = 3) -> bool:
    """
    .. description::
    This function checks spline normals for crossings. Returns True if a crossing was found, otherwise False.

    .. inputs::
    :param track:               array containing the track [x, y, w_tr_right, w_tr_left] to check
    :type track:                np.ndarray
    :param normvec_normalized:  array containing normalized normal vectors for every track point
                                [x_component, y_component]
    :type normvec_normalized:   np.ndarray
    :param horizon:             determines the number of normals in forward and backward direction that are checked
                                against each normal on the line
    :type horizon:              int

    .. outputs::
    :return found_crossing:     bool value indicating if a crossing was found or not
    :rtype found_crossing:      bool

    .. notes::
    The checks can take a while if full check is performed. Inputs are unclosed.
    """

    # check input
    no_points = track.shape[0]

    if horizon >= no_points:
        raise RuntimeError("Horizon of %i points is too large for a track with %i points, reduce horizon!"
                           % (horizon, no_points))

    elif horizon >= no_points / 2:
        print("WARNING: Horizon of %i points makes no sense for a track with %i points, reduce horizon!"
              % (horizon, no_points))

    # initialization
    les_mat = np.zeros((2, 2))
    idx_list = list(range(0, no_points))
    idx_list = idx_list[-horizon:] + idx_list + idx_list[:horizon]

    # loop through all points of the track to check for crossings in their neighbourhoods
    for idx in range(no_points):

        # determine indices of points in the neighbourhood of the current index
        idx_neighbours = idx_list[idx:idx + 2 * horizon + 1]
        del idx_neighbours[horizon]
        idx_neighbours = np.array(idx_neighbours)

        # remove indices of normal vectors that are collinear to the current index
        is_collinear_b = np.isclose(np.cross(normvec_normalized[idx], normvec_normalized[idx_neighbours]), 0.0)
        idx_neighbours_rel = idx_neighbours[np.nonzero(np.invert(is_collinear_b))[0]]

        # check crossings solving an LES
        for idx_comp in list(idx_neighbours_rel):

            # LES: x_1 + lambda_1 * nx_1 = x_2 + lambda_2 * nx_2; y_1 + lambda_1 * ny_1 = y_2 + lambda_2 * ny_2;
            const = track[idx_comp, :2] - track[idx, :2]
            les_mat[:, 0] = normvec_normalized[idx]
            les_mat[:, 1] = -normvec_normalized[idx_comp]

            # solve LES
            lambdas = np.linalg.solve(les_mat, const)

            # we have a crossing within the relevant part if both lambdas lie between -w_tr_left and w_tr_right
            if -track[idx, 3] <= lambdas[0] <= track[idx, 2] \
                    and -track[idx_comp, 3] <= lambdas[1] <= track[idx_comp, 2]:
                return True  # found crossing

    return False

def interp_track(track: np.ndarray,
                 stepsize: float) -> np.ndarray:
    """

    .. description::
    Interpolate track points linearly to a new stepsize.

    .. inputs::
    :param track:           track in the format [x, y, w_tr_right, w_tr_left, (banking)].
    :type track:            np.ndarray
    :param stepsize:        desired stepsize after interpolation in m.
    :type stepsize:         float

    .. outputs::
    :return track_interp:   interpolated track [x, y, w_tr_right, w_tr_left, (banking)].
    :rtype track_interp:    np.ndarray

    .. notes::
    Track input and output are unclosed! track input must however be closable in the current form!
    The banking angle is optional and must not be provided!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LINEAR INTERPOLATION OF TRACK ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create closed track
    track_cl = np.vstack((track, track[0]))

    # calculate element lengths (euclidian distance)
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))

    # sum up total distance (from start) to every element
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)

    # calculate desired lenghts depending on specified stepsize (+1 because last element is included)
    no_points_interp_cl = math.ceil(dists_cum_cl[-1] / stepsize) + 1
    dists_interp_cl = np.linspace(0.0, dists_cum_cl[-1], no_points_interp_cl)

    # interpolate closed track points
    track_interp_cl = np.zeros((no_points_interp_cl, track_cl.shape[1]))

    track_interp_cl[:, 0] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 0])
    track_interp_cl[:, 1] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 1])
    track_interp_cl[:, 2] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 2])
    track_interp_cl[:, 3] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 3])

    if track_cl.shape[1] == 5:
        track_interp_cl[:, 4] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, 4])

    return track_interp_cl[:-1]


def side_of_line(a: Union[tuple, np.ndarray],
                 b: Union[tuple, np.ndarray],
                 z: Union[tuple, np.ndarray]) -> float:
    """
    .. description::
    Function determines if a point z is on the left or right side of a line from a to b. It is based on the z component
    orientation of the cross product, see question on
    https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is-to-the-right-or-left-side-of-a-line

    .. inputs::
    :param a:       point coordinates [x, y]
    :type a:        Union[tuple, np.ndarray]
    :param b:       point coordinates [x, y]
    :type b:        Union[tuple, np.ndarray]
    :param z:       point coordinates [x, y]
    :type z:        Union[tuple, np.ndarray]

    .. outputs::
    :return side:   0.0 = on line, 1.0 = left side, -1.0 = right side.
    :rtype side:    float
    """

    # calculate side
    side = np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0]))

    return side




# ----------------------------------------------------------------------------------------------------------------------
# DISTANCE CALCULATION FOR OPTIMIZATION --------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# return distance from point p to a point on the spline at spline parameter t_glob
def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):
    s_vals = interpolate.splev(t_glob, path)
    s = np.array([s_vals[0].item(), s_vals[1].item()])
    assert p.ndim == 1, f"Expected 1D p, got shape {p.shape}"
    assert s.ndim == 1, f"Expected 1D s, got shape {s.shape}"
    return spatial.distance.euclidean(p, s)


def spline_approximation(track: np.ndarray,
                         k_reg: int = 3,
                         s_reg: int = 10,
                         stepsize_prep: float = 1.0,
                         stepsize_reg: float = 3.0,
                         debug: bool = False) -> np.ndarray:
    """

    .. description::
    Smooth spline approximation for a track (e.g. centerline, reference line).

    .. inputs::
    :param track:           [x, y, w_tr_right, w_tr_left, (banking)] (always unclosed).
    :type track:            np.ndarray
    :param k_reg:           order of B splines.
    :type k_reg:            int
    :param s_reg:           smoothing factor (usually between 5 and 100).
    :type s_reg:            int
    :param stepsize_prep:   stepsize used for linear track interpolation before spline approximation.
    :type stepsize_prep:    float
    :param stepsize_reg:    stepsize after smoothing.
    :type stepsize_reg:     float
    :param debug:           flag for printing debug messages
    :type debug:            bool

    .. outputs::
    :return track_reg:      [x, y, w_tr_right, w_tr_left, (banking)] (always unclosed).
    :rtype track_reg:       np.ndarray

    .. notes::
    The function can only be used for closable tracks, i.e. track is closed at the beginning!
    The banking angle is optional and must not be provided!
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LINEAR INTERPOLATION BEFORE SMOOTHING ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    track_interp =interp_track(track=track,
                                                 stepsize=stepsize_prep)
    track_interp_cl = np.vstack((track_interp, track_interp[0]))

    # ------------------------------------------------------------------------------------------------------------------
    # SPLINE APPROXIMATION / PATH SMOOTHING ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create closed track (original track)
    track_cl = np.vstack((track, track[0]))
    no_points_track_cl = track_cl.shape[0]
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)

    # find B spline representation of the inserted path and smooth it in this process
    # (tck_cl: tuple (vector of knots, the B-spline coefficients, and the degree of the spline))
    tck_cl, t_glob_cl = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]],
                                            k=k_reg,
                                            s=s_reg,
                                            per=1)[:2]

    # calculate total length of smooth approximating spline based on euclidian distance with points at every 0.25m
    no_points_lencalc_cl = math.ceil(dists_cum_cl[-1]) * 4
    path_smoothed_tmp = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
    len_path_smoothed_tmp = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_tmp, axis=0), 2), axis=1)))

    # get smoothed path
    no_points_reg_cl = math.ceil(len_path_smoothed_tmp / stepsize_reg) + 1
    path_smoothed = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_reg_cl), tck_cl)).T[:-1]

    # ------------------------------------------------------------------------------------------------------------------
    # PROCESS TRACK WIDTHS (AND BANKING ANGLE IF GIVEN) ----------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # find the closest points on the B spline to input points
    dists_cl = np.zeros(no_points_track_cl)                 # contains (min) distances between input points and spline
    closest_point_cl = np.zeros((no_points_track_cl, 2))    # contains the closest points on the spline
    closest_t_glob_cl = np.zeros(no_points_track_cl)        # containts the t_glob values for closest points
    t_glob_guess_cl = dists_cum_cl / dists_cum_cl[-1]       # start guess for the minimization

    for i in range(no_points_track_cl):
        # get t_glob value for the point on the B spline with a minimum distance to the input points
        closest_t_glob_cl[i] = optimize.fmin(dist_to_p,
                                             x0=t_glob_guess_cl[i],
                                             args=(tck_cl, track_cl[i, :2]),
                                             disp=False)

        # evaluate B spline on the basis of t_glob to obtain the closest point
        closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)

        # save distance from closest point to input point
        dists_cl[i] = math.sqrt(math.pow(closest_point_cl[i, 0] - track_cl[i, 0], 2)
                                + math.pow(closest_point_cl[i, 1] - track_cl[i, 1], 2))

    if debug:
        print("Spline approximation: mean deviation %.2fm, maximum deviation %.2fm"
              % (float(np.mean(dists_cl)), float(np.amax(np.abs(dists_cl)))))

    # get side of smoothed track compared to the inserted track
    sides = np.zeros(no_points_track_cl - 1)

    for i in range(no_points_track_cl - 1):
        sides[i] = side_of_line(a=track_cl[i, :2], b=track_cl[i+1, :2],z=closest_point_cl[i])

    sides_cl = np.hstack((sides, sides[0]))

    # calculate new track widths on the basis of the new reference line, but not interpolated to new stepsize yet
    w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
    w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl

    # interpolate track widths after smoothing (linear)
    w_tr_right_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_right_new_cl)
    w_tr_left_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, w_tr_left_new_cl)

    track_reg = np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))

    # interpolate banking if given (linear)
    if track_cl.shape[1] == 5:
        banking_smoothed_cl = np.interp(np.linspace(0.0, 1.0, no_points_reg_cl), closest_t_glob_cl, track_cl[:, 4])
        track_reg = np.column_stack((track_reg, banking_smoothed_cl[:-1]))

    return track_reg

import sys
def prep_track(reftrack_imp: np.ndarray,
               reg_smooth_opts: dict,
               stepsize_opts: dict,
               debug: bool = False,
               min_width: float = None) -> tuple:
    """
    Documentation:
    This function prepares the inserted reference track for optimization.

    Inputs:
    reftrack_imp:               imported track [x_m, y_m, w_tr_right_m, w_tr_left_m]
    reg_smooth_opts:            parameters for the spline approximation
    stepsize_opts:              dict containing the stepsizes before spline approximation and after spline interpolation
    debug:                      boolean showing if debug messages should be printed
    min_width:                  [m] minimum enforced track width (None to deactivate)

    Outputs:
    reftrack_interp:            track after smoothing and interpolation [x_m, y_m, w_tr_right_m, w_tr_left_m]
    normvec_normalized_interp:  normalized normal vectors on the reference line [x_m, y_m]
    a_interp:                   LES coefficients when calculating the splines
    coeffs_x_interp:            spline coefficients of the x-component
    coeffs_y_interp:            spline coefficients of the y-component
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE REFTRACK AND CALCULATE INITIAL SPLINES ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # smoothing and interpolating reference track
    reftrack_interp = spline_approximation(track=reftrack_imp,
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"],
                             debug=debug)

    # calculate splines
    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = calc_splines(path=refpath_interp_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    normals_crossing = check_normals_crossing(track=reftrack_interp,normvec_normalized=normvec_normalized_interp,horizon=3)

    if normals_crossing:
        bound_1_tmp = reftrack_interp[:, :2] + normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 2], axis=1)
        bound_2_tmp = reftrack_interp[:, :2] - normvec_normalized_interp * np.expand_dims(reftrack_interp[:, 3], axis=1)

        plt.figure()

        plt.plot(reftrack_interp[:, 0], reftrack_interp[:, 1], 'k-')
        for i in range(bound_1_tmp.shape[0]):
            temp = np.vstack((bound_1_tmp[i], bound_2_tmp[i]))
            plt.plot(temp[:, 0], temp[:, 1], "r-", linewidth=0.7)

        plt.grid()
        ax = plt.gca()
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.title("Error: at least one pair of normals is crossed!")

        plt.show()

        raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

    # ------------------------------------------------------------------------------------------------------------------
    # ENFORCE MINIMUM TRACK WIDTH (INFLATE TIGHTER SECTIONS UNTIL REACHED) ---------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    manipulated_track_width = False

    if min_width is not None:
        for i in range(reftrack_interp.shape[0]):
            cur_width = reftrack_interp[i, 2] + reftrack_interp[i, 3]

            if cur_width < min_width:
                manipulated_track_width = True

                # inflate to both sides equally
                reftrack_interp[i, 2] += (min_width - cur_width) / 2
                reftrack_interp[i, 3] += (min_width - cur_width) / 2

    if manipulated_track_width:
        print("WARNING: Track region was smaller than requested minimum track width -> Applied artificial inflation in"
              " order to match the requirements!", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


def reftrack_to_mlts_track(reftrack: np.ndarray,
                           elevation: np.ndarray = None,
                           slope: np.ndarray = None,
                           banking: np.ndarray = None):
    """

    .. description::
    Convert an OptiLine reference track into the track table expected by the external
    point-mass minimum-lap-time solver MLTS-point-mass (see solvers.PointMassOCP).

    The reference line is fitted with the same closed cubic splines used everywhere
    else in OptiLine, so the arc length ("abscissa"), the heading and the curvature
    all describe one and the same curve. This matters: a track table whose abscissa
    disagrees with its own tabulated coordinates makes the curvilinear lap time
    differ from the lap time implied by the exported (x, y) path, by exactly the
    ratio between the two length scales.

    Conventions. OptiLine stores the heading as psi = atan2(dy, dx) - pi/2 (psi = 0
    is north) and its normal vectors point to the RIGHT of the direction of travel,
    whereas MLTS-point-mass expects dir_mid_line = atan2(dy, dx) and a lateral
    coordinate n that is positive to the LEFT. Both are handled here, so the lateral
    offset n returned by the solver relates to an OptiLine lateral shift by
    alpha = -n.

    If the last point of `reftrack` duplicates the first, it is dropped before the
    splines are fitted: keeping it would leave a near-zero-length closing segment,
    which produces a spurious curvature spike at the seam.

    .. inputs::
    :param reftrack:    reference track [x, y, w_tr_right, w_tr_left] in m (unclosed).
    :type reftrack:     np.ndarray
    :param elevation:   optional elevation of every reference point in m (default 0).
    :type elevation:    np.ndarray
    :param slope:       optional slope angle of every reference point in rad (default 0).
    :type slope:        np.ndarray
    :param banking:     optional banking angle of every reference point in rad (default 0).
    :type banking:      np.ndarray

    .. outputs::
    :return track:      pandas DataFrame with the columns abscissa, curvature,
                        dir_mid_line, x_mid_line, y_mid_line, width_no_kerbs_L,
                        width_no_kerbs_R, elevation, slope and banking. The last row
                        repeats the first one, so that the table describes a closed lap.
    :rtype track:       pandas.DataFrame
    """

    try:
        import pandas as pd
    except ImportError as exc:                                  # pragma: no cover
        raise ImportError(
            "reftrack_to_mlts_track requires pandas. It is installed together with "
            "the MLTS-point-mass solver:\n"
            "    pip install MLTS-point-mass"
        ) from exc

    reftrack = np.asarray(reftrack, dtype=float)
    if reftrack.ndim != 2 or reftrack.shape[1] < 4:
        raise ValueError("reftrack must be of shape (n, 4): [x, y, w_tr_right, w_tr_left]")

    # drop a duplicated closing point (it would create a degenerate spline segment)
    spacing = np.median(np.hypot(*np.diff(reftrack[:, :2], axis=0).T))
    if np.hypot(*(reftrack[-1, :2] - reftrack[0, :2])) < 0.5 * spacing:
        reftrack = reftrack[:-1]

    n_pts = reftrack.shape[0]

    def _closed(col, default):
        if col is None:
            return np.full(n_pts + 1, float(default))
        col = np.asarray(col, dtype=float)[:n_pts]
        return np.append(col, col[0])

    # closed cubic splines through the reference line
    path = np.vstack((reftrack[:, :2], reftrack[0, :2]))
    el_lengths = np.hypot(*np.diff(path, axis=0).T)
    coeffs_x, coeffs_y, _, _ = calc_splines(path=path, el_lengths=el_lengths)

    # arc length of the reference line (NOT the chord polyline)
    spline_lengths = calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    abscissa = np.concatenate(([0.0], np.cumsum(spline_lengths)))

    # heading and curvature at the spline start points, i.e. at the track nodes
    no_splines = coeffs_x.shape[0]
    psi, kappa = calc_head_curv_an(coeffs_x=coeffs_x,
                                   coeffs_y=coeffs_y,
                                   ind_spls=np.arange(no_splines),
                                   t_spls=np.zeros(no_splines))

    # close the lap: node N repeats node 0
    psi = np.append(psi, psi[0])
    kappa = np.append(kappa, kappa[0])

    # OptiLine psi is north-zero, MLTS-point-mass wants atan2(dy, dx); unwrap so that
    # the solver can spline the heading across the +/-pi branch cut
    dir_mid_line = np.unwrap(psi + math.pi / 2.0)

    return pd.DataFrame({
        "abscissa":         abscissa,
        "curvature":        kappa,
        "dir_mid_line":     dir_mid_line,
        "x_mid_line":       path[:, 0],
        "y_mid_line":       path[:, 1],
        "width_no_kerbs_L": _closed(reftrack[:, 3], 0.0),       # left  half-width
        "width_no_kerbs_R": _closed(reftrack[:, 2], 0.0),       # right half-width
        "elevation":        _closed(elevation, 0.0),
        "slope":            _closed(slope, 0.0),
        "banking":          _closed(banking, 0.0),
    })


# ===========================================================================
# G-G-V envelope loading
#
# A full g-g-v envelope (the JSON form shared by the Trento group) gives the
# longitudinal limits on a 2-D (v, ay) grid plus a 1-D ay_max(v). FBGA consumes it
# through three callables, gg_upper(ay, v), gg_lower(ay, v) and gg_range(v), which it
# evaluates once per segment from C++ -- so they must be fast and picklable (the
# zeroth-order solvers run them in worker processes).
#
# Before v0.2.8 every user had to write that plumbing themselves, which is how two
# notebooks ended up reconstructing the same envelope differently and disagreeing
# about whether a solution respected it. load_ggv_envelope() is the single
# implementation.
# ===========================================================================

class Bilinear:
    """f(ay, v) on a uniform (v, ay) grid, with linear extrapolation. Picklable."""

    def __init__(self, v_grid, ay_grid, Z):
        self.v0 = float(v_grid[0]);  self.dv = float(v_grid[1] - v_grid[0]);  self.nv = len(v_grid)
        self.a0 = float(ay_grid[0]); self.da = float(ay_grid[1] - ay_grid[0]); self.na = len(ay_grid)
        self.Z = np.asarray(Z, float).tolist()

    def __call__(self, ay, v):
        fi = (v - self.v0) / self.dv
        i = int(math.floor(fi)); i = 0 if i < 0 else (self.nv - 2 if i > self.nv - 2 else i)
        ti = fi - i
        fj = (ay - self.a0) / self.da
        j = int(math.floor(fj)); j = 0 if j < 0 else (self.na - 2 if j > self.na - 2 else j)
        tj = fj - j
        r0, r1 = self.Z[i], self.Z[i + 1]
        return (r0[j] * (1 - ti) * (1 - tj) + r0[j + 1] * (1 - ti) * tj +
                r1[j] * ti * (1 - tj)       + r1[j + 1] * ti * tj)

    def as_callable(self):
        """
        Return a plain closure computing exactly what ``__call__`` computes.

        Fb2d calls these bounds back ~10^5 times per velocity profile, where the
        bound-method frame and the eight ``self.*`` loads cost more than the
        arithmetic. Binding them into cell variables is ~1.4x faster per call and
        bitwise identical -- the expression below is character-for-character the
        one above. The instance stays the picklable thing that crosses process
        boundaries to the zeroth-order workers; this is unwrapped on arrival.
        """
        v0, dv, a0, da, Z = self.v0, self.dv, self.a0, self.da, self.Z
        i_hi, j_hi, floor = self.nv - 2, self.na - 2, math.floor

        def f(ay, v):
            fi = (v - v0) / dv
            i = int(floor(fi)); i = 0 if i < 0 else (i_hi if i > i_hi else i)
            ti = fi - i
            fj = (ay - a0) / da
            j = int(floor(fj)); j = 0 if j < 0 else (j_hi if j > j_hi else j)
            tj = fj - j
            r0 = Z[i]; r1 = Z[i + 1]
            return (r0[j] * (1 - ti) * (1 - tj) + r0[j + 1] * (1 - ti) * tj +
                    r1[j] * ti * (1 - tj)       + r1[j + 1] * ti * tj)
        return f


class Linear1D:
    """f(v) on a uniform v grid, with linear extrapolation. Picklable."""

    def __init__(self, v_grid, y):
        self.v0 = float(v_grid[0]); self.dv = float(v_grid[1] - v_grid[0]); self.nv = len(v_grid)
        self.Y = np.asarray(y, float).tolist()

    def __call__(self, v):
        fi = (v - self.v0) / self.dv
        i = int(math.floor(fi)); i = 0 if i < 0 else (self.nv - 2 if i > self.nv - 2 else i)
        t = fi - i
        return self.Y[i] * (1 - t) + self.Y[i + 1] * t

    def as_callable(self):
        """Plain closure equivalent of ``__call__``; see :meth:`Bilinear.as_callable`."""
        v0, dv, Y = self.v0, self.dv, self.Y
        i_hi, floor = self.nv - 2, math.floor

        def f(v):
            fi = (v - v0) / dv
            i = int(floor(fi)); i = 0 if i < 0 else (i_hi if i > i_hi else i)
            t = fi - i
            return Y[i] * (1 - t) + Y[i + 1] * t
        return f


def load_ggv_envelope(source,
                      interp: str = "mlts",
                      n_v: int = 200,
                      n_ay: int = 200,
                      ggv_scales=None) -> dict:
    """

    .. description::
    Load a full g-g-v envelope and return everything FBGA and the diagnostics need.

    Two reconstructions of the same data are available, and they are NOT equivalent:

      'mlts' (default)
          Build the envelope with mlts_point_mass.GGVConstr, then sample its boundary
          onto a uniform (v, ay) grid for fast lookup. GGVConstr normalises each row
          onto s = ay / ay_max(v), enforces |ax| non-increasing with ay (which strips
          solver glitches) and discards speeds where the envelope has collapsed. This
          is the construction the point-mass OCP is itself constrained with, so a
          solution produced by that OCP is consistent with these bounds.

      'linear'
          Bilinear interpolation straight on the tabulated (v, ay) grid. Faithful to
          the file, but it also reproduces its artefacts: a g-g-v fitted to close at
          top speed stores ax_min = 0 there, and interpolating into that row removes
          most of the braking capability over the last few m/s. On the Catalunya
          envelope the two agree to ~0.01 m/s^2 below 70 m/s and diverge above ~74,
          which costs a minimum-curvature lap about 0.5 s.

    Returned 'g', available only for 'mlts', is the exact constraint g(ax, ay, V) <= 1.
    Use it for diagnostics: the gridded callables are an interpolation of the same
    boundary, so a solution that rides the boundary reads as marginally outside them
    even when it is exactly feasible.

    .. inputs::
    :param source:      path to the envelope JSON, or the already-loaded dict.
    :type source:       str or dict
    :param interp:      'mlts' (default) or 'linear'.
    :type interp:       str
    :param n_v:         speed samples of the lookup grid ('mlts' only).
    :type n_v:          int
    :param n_ay:        lateral samples of the lookup grid ('mlts' only).
    :type n_ay:         int
    :param ggv_scales:  optional grip scales passed to GGVConstr ('mlts' only).
    :type ggv_scales:   list or np.ndarray

    .. outputs::
    :return env:    dict with gg_upper, gg_lower, gg_range, g (None for 'linear'),
                    v_max, and the raw grids ay_vec, vel_vec, ax_max_grid,
                    ax_min_grid, vel_vec_lat, ay_max_vec.
    :rtype env:     dict
    """

    if isinstance(source, dict):
        data = source
    else:
        import json as _json
        with open(source, "r") as fh:
            data = _json.load(fh)

    lon = data["longitudinal_acceleration"]
    ay_vec = np.array(lon["lateral_acceleration"], float)
    vel_vec = np.array(lon["longitudinal_velocity"], float)
    ax_max_grid = np.array(lon["positive_longitudinal_acceleration"], float
                           ).reshape(len(vel_vec), len(ay_vec))
    ax_min_grid = np.array(lon["negative_longitudinal_acceleration"], float
                           ).reshape(len(vel_vec), len(ay_vec))
    lat = data["lateral_acceleration"]
    vel_vec_lat = np.array(lat["longitudinal_velocity"], float)
    ay_max_vec = np.array(lat["positive_lateral_acceleration"], float)

    # v_max always comes from the tabulated speed axis, never from a trimmed grid:
    # ax_max -> 0 at the top row is the physical definition of the car's top speed.
    v_max = float(vel_vec.max())
    gg_range = Linear1D(vel_vec_lat, ay_max_vec)
    g_exact = None

    if interp == "linear":
        gg_upper = Bilinear(vel_vec, ay_vec, ax_max_grid)
        gg_lower = Bilinear(vel_vec, ay_vec, ax_min_grid)

    elif interp == "mlts":
        try:
            from mlts_point_mass import GGVConstr
        except ImportError as exc:                              # pragma: no cover
            raise ImportError(
                "interp='mlts' needs the MLTS-point-mass package, which is a "
                "dependency of OptiLine but is not importable from this interpreter. "
                "Install it here, or use interp='linear'."
            ) from exc

        gc = GGVConstr(data, scales=ggv_scales)
        g_exact = gc.g

        v_grid = np.linspace(max(float(vel_vec.min()), gc.v_min),
                             min(v_max, gc.v_max), int(n_v))
        ay_grid = np.linspace(-float(ay_vec.max()), float(ay_vec.max()), int(n_ay))
        up = np.zeros((len(v_grid), len(ay_grid)))
        lo = np.zeros_like(up)
        half = float(ay_grid[1] - ay_grid[0])
        for i, vv in enumerate(v_grid):
            b_ay, b_ax = gc.boundary(float(vv), n=1441)         # closed boundary (ay, ax)
            keep = b_ay >= 0.0
            A, X = b_ay[keep], b_ax[keep]
            for j, a in enumerate(np.abs(ay_grid)):
                m = np.abs(A - a) <= half
                if m.any():
                    up[i, j] = X[m].max()
                    lo[i, j] = X[m].min()
        gg_upper = Bilinear(v_grid, ay_grid, up)
        gg_lower = Bilinear(v_grid, ay_grid, lo)

    else:
        raise ValueError(f"interp must be 'mlts' or 'linear', got {interp!r}")

    return {"gg_upper": gg_upper, "gg_lower": gg_lower, "gg_range": gg_range,
            "g": g_exact, "v_max": v_max, "interp": interp,
            "ay_vec": ay_vec, "vel_vec": vel_vec,
            "ax_max_grid": ax_max_grid, "ax_min_grid": ax_min_grid,
            "vel_vec_lat": vel_vec_lat, "ay_max_vec": ay_max_vec}


# ===========================================================================
# Reference-track and raceline utilities
#
# These were duplicated across the experiment notebooks (seg_bounds and the
# sub-sampling helper appeared in three of them, in two different variants).
# ===========================================================================

def seg_bounds(reftrack: np.ndarray, lo: float = 0.8, hi: float = 1.3) -> tuple:
    """

    .. description::
    Segment lengths of a CLOSED reference track, plus the spline-length bounds the
    bilevel optimiser uses. The closing segment (last point -> first point) is
    included, which is what the spline fit and the solvers assume.

    .. inputs::
    :param reftrack:    reference track [x, y, w_tr_right, w_tr_left] (unclosed).
    :param lo:          lower bound as a fraction of the shortest segment.
    :param hi:          upper bound as a fraction of the longest segment.

    .. outputs::
    :return min_s:      lower spline-length bound.
    :return max_s:      upper spline-length bound.
    :return lengths:    segment lengths including the closing one.
    """
    lengths = np.hypot(*np.diff(reftrack[:, :2], axis=0).T)
    lengths = np.append(lengths, np.hypot(*(reftrack[0, :2] - reftrack[-1, :2])))
    return lo * lengths.min(), hi * lengths.max(), lengths


def subsample_reftrack(reftrack: np.ndarray,
                       step: int = 1,
                       mode: str = "arc",
                       abscissa: np.ndarray = None) -> np.ndarray:
    """

    .. description::
    Thin a reference track down to roughly every `step`-th point.

    `mode` decides HOW the nodes are picked, and it matters:

      'arc'    nodes at uniform arc length (default). Use this whenever the track
               file is sampled non-uniformly -- a track stored at 1 m through the
               corners and 4 m down the straights yields segments between 7 and 32 m
               under plain index sub-sampling, and a cubic spline through such a node
               set curls sharply at the short segments, corrupting the curvature and
               hence the velocity profile.
      'index'  every step-th row, the historical behaviour.

    Either way the node COUNT is round(n / step), so `step` keeps its meaning.

    A duplicated closing point is dropped first: keeping it leaves a zero-length
    closing segment, which is the degenerate case that produces |kappa| spikes.

    .. inputs::
    :param reftrack:    reference track [x, y, w_tr_right, w_tr_left].
    :param step:        thinning factor.
    :param mode:        'arc' or 'index'.
    :param abscissa:    optional arc length of each row; computed from the chord
                        polyline when not given ('arc' mode only).

    .. outputs::
    :return reftrack:   the thinned reference track.
    """
    reftrack = np.asarray(reftrack, dtype=float)
    xy = reftrack[:, :2]
    spacing = np.median(np.hypot(*np.diff(xy, axis=0).T))
    if np.hypot(*(xy[-1] - xy[0])) < 0.5 * spacing:
        reftrack = reftrack[:-1]
        if abscissa is not None:
            abscissa = np.asarray(abscissa, float)[:-1]
        xy = reftrack[:, :2]

    n = max(3, int(round(reftrack.shape[0] / step)))

    if mode == "index":
        idx = np.unique(np.linspace(0, reftrack.shape[0], n, endpoint=False).astype(int))
        return reftrack[idx]

    if mode != "arc":
        raise ValueError(f"mode must be 'arc' or 'index', got {mode!r}")

    if abscissa is None:
        seg = np.hypot(*np.diff(np.vstack((xy, xy[0])), axis=0).T)
        abscissa = np.concatenate(([0.0], np.cumsum(seg)[:-1]))
        total = float(seg.sum())
    else:
        abscissa = np.asarray(abscissa, float)
        total = float(abscissa[-1] + np.hypot(*(xy[0] - xy[-1])))

    s_nodes = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack([np.interp(s_nodes, abscissa, reftrack[:, c])
                            for c in range(reftrack.shape[1])])


def raceline_from_alpha(reftrack: np.ndarray,
                        normvectors: np.ndarray,
                        alpha: np.ndarray,
                        stepsize_interp: float = 1.0) -> tuple:
    """

    .. description::
    Interpolate the raceline for a vector of lateral shifts and return everything a
    velocity profiler needs. Wraps create_raceline + calc_head_curv_an, which is the
    pairing every solver and notebook repeats.

    .. inputs::
    :param reftrack:        reference track [x, y, w_tr_right, w_tr_left].
    :param normvectors:     unit normals from calc_splines.
    :param alpha:           lateral shift per reference point in m.
    :param stepsize_interp: raceline interpolation step in m.

    .. outputs::
    :return raceline:   interpolated raceline [x, y].
    :return kappa:      curvature at every interpolated point in 1/m.
    :return el_lengths: element lengths between interpolated points in m.
    """
    raceline, _, coeffs_x, coeffs_y, spline_inds, t_vals, _, _, el_lengths = \
        create_raceline(refline=reftrack[:, :2], normvectors=normvectors,
                        alpha=alpha, stepsize_interp=stepsize_interp)
    _, kappa = calc_head_curv_an(coeffs_x=coeffs_x, coeffs_y=coeffs_y,
                                 ind_spls=spline_inds, t_spls=t_vals)
    return raceline, kappa, el_lengths


def track_borders(x_mid, y_mid, dir_mid, width_left, width_right) -> tuple:
    """

    .. description::
    Left and right track edges from the mid-line, its heading and the half-widths.
    The normal is taken positive to the LEFT of the direction of travel, matching the
    convention of the shared track files (and the opposite of OptiLine's own
    normvectors, which point right).

    .. outputs::
    :return left:   (x, y) of the left edge.
    :return right:  (x, y) of the right edge.
    """
    th = np.unwrap(np.asarray(dir_mid, float))
    nx, ny = -np.sin(th), np.cos(th)
    x_mid, y_mid = np.asarray(x_mid, float), np.asarray(y_mid, float)
    wl, wr = np.asarray(width_left, float), np.asarray(width_right, float)
    return ((x_mid + wl * nx, y_mid + wl * ny), (x_mid - wr * nx, y_mid - wr * ny))


# ===========================================================================
# Projection onto the centreline curvilinear abscissa
#
# Comparing racelines needs a common independent coordinate, and each raceline's own
# travelled arc length is not one: different paths have different lengths, so the
# curves slide relative to each other by tens of metres over a lap. The centreline
# abscissa is an objective spatial position, so a given value is the same place on
# the track for every solution.
# ===========================================================================

def unwrap_period(v: np.ndarray, period: float) -> np.ndarray:
    """Make a periodic coordinate increase monotonically over one lap."""
    v = np.asarray(v, float)
    d = (np.diff(v) + 0.5 * period) % period - 0.5 * period
    return np.concatenate([[v[0]], v[0] + np.cumsum(d)])


def centreline_frame(x_c, y_c) -> dict:
    """

    .. description::
    Pre-process a centreline polyline for repeated point projection. Build this once
    and reuse it; the KD-tree is what makes project_to_centreline cheap.

    .. inputs::
    :param x_c:     x of the centreline.
    :param y_c:     y of the centreline. The loop is closed automatically.

    .. outputs::
    :return frame:  dict with P, d, seg2, s0, L (lap length) and tree.
    """
    P = np.column_stack([np.asarray(x_c, float), np.asarray(y_c, float)])
    if np.hypot(*(P[0] - P[-1])) > 1e-9:
        P = np.vstack([P, P[0]])
    d = np.diff(P, axis=0)
    seg2 = np.einsum('ij,ij->i', d, d)
    s0 = np.concatenate([[0.0], np.cumsum(np.sqrt(seg2))])
    return dict(P=P, d=d, seg2=seg2, s0=s0, L=float(s0[-1]),
                tree=spatial.cKDTree(P[:-1]))


def project_to_centreline(frame: dict, xy: np.ndarray, n_nb: int = 4) -> tuple:
    """

    .. description::
    Centreline abscissa and lateral distance of every point of `xy`.

    Each query point is projected onto the nearest of the 2*n_nb+1 centreline
    segments around its nearest centreline vertex, which is exact for a path that
    stays inside the corridor.

    .. outputs::
    :return s_c:    centreline abscissa, unwrapped to increase over the lap.
    :return dist:   unsigned lateral distance to the centreline in m.
    """
    Q = np.asarray(xy, float)
    _, iv = frame['tree'].query(Q, k=1)
    ns = len(frame['seg2'])
    cand = (iv[:, None] + np.arange(-n_nb, n_nb + 1)[None, :]) % ns
    A = frame['P'][cand]
    D = frame['d'][cand]
    W = Q[:, None, :] - A
    t = np.clip(np.einsum('nkj,nkj->nk', W, D) / frame['seg2'][cand], 0.0, 1.0)
    E = W - t[..., None] * D
    dist2 = np.einsum('nkj,nkj->nk', E, E)
    k = np.argmin(dist2, axis=1)
    n = np.arange(len(Q))
    s_c = frame['s0'][cand[n, k]] + t[n, k] * np.sqrt(frame['seg2'][cand[n, k]])
    return unwrap_period(s_c, frame['L']), np.sqrt(dist2[n, k])


def project_signed(frame: dict, xy: np.ndarray, n_nb: int = 4) -> tuple:
    """
    As project_to_centreline, but the lateral offset carries a sign: positive to the
    LEFT of the direction of travel (the ISO convention used by the shared files).
    """
    Q = np.asarray(xy, float)
    s_c, off = project_to_centreline(frame, Q, n_nb=n_nb)
    _, iv = frame['tree'].query(Q, k=1)
    t = frame['d'][iv]
    w = Q - frame['P'][iv]
    cross = t[:, 0] * w[:, 1] - t[:, 1] * w[:, 0]
    return s_c, np.where(cross >= 0.0, 1.0, -1.0) * off


def smooth_over(y: np.ndarray, x: np.ndarray, win_m: float = 60.0) -> np.ndarray:
    """Periodic moving average of y over a window of win_m metres of x."""
    k = max(3, int(round(win_m / max(np.median(np.diff(x)), 1e-9))) | 1)
    pad = np.r_[y[-(k // 2):], y, y[:k // 2]]
    return np.convolve(pad, np.ones(k) / k, mode='valid')[:len(y)]


def ggv_tables_from_envelope(env: dict) -> tuple:
    """

    .. description::
    Minimal point-mass tables taken off a full g-g-v envelope at ay = 0.

    calc_vel_profile_fb2d only consults `ggv` and `ax_max_machines` when gg_upper /
    gg_lower / gg_range are NOT all supplied (and for a v_max fallback), so with a
    full envelope in use they are never read -- but several solver signatures still
    require them. This provides them without pretending to be a faithful projection
    of the envelope onto a point-mass model.

    For the vanilla profiler (vel_solver='fb') these tables ARE used, together with
    dyn_model_exp; the result is only as good as an e-superellipse fit of the real
    envelope, so prefer the fb2d path when a full envelope is available.

    .. inputs::
    :param env:         envelope dict from load_ggv_envelope.

    .. outputs::
    :return ggv:        [v, ax_max, ay_max], shape (n, 3).
    :return ax_max_machines: [v, ax_max_machines], shape (n, 2).
    """
    v = env['vel_vec']
    j0 = int(np.argmin(np.abs(env['ay_vec'])))
    brake = np.maximum(-env['ax_min_grid'][:, j0], 1e-3)     # tyre longitudinal capacity
    drive = np.maximum(env['ax_max_grid'][:, j0], 1e-3)      # motor drive limit
    ay_max = np.maximum(np.interp(v, env['vel_vec_lat'], env['ay_max_vec']), 1e-3)
    return np.column_stack([v, brake, ay_max]), np.column_stack([v, drive])
