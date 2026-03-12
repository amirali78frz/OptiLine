import math

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

    x_les = np.squeeze(np.linalg.solve(M, b_x))  # squeeze removes single-dimensional entries
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
                 fix_e: bool = False) -> tuple:

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

    # create extraction matrix -> only b_i coefficients of the solved linear equation system are needed for gradient
    # information
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    # create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
    # information
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)

    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    # invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]  # close spline

            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    # set up q_x and q_y matrices including the point coordinate information
    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))

    for i in range(no_splines):
        j = i * 4

        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]

            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    # for unclosed tracks, specify start- and end-heading constraints
    if not closed:
        q_x[-2, 0] = math.cos(psi_s + math.pi / 2)
        q_y[-2, 0] = math.sin(psi_s + math.pi / 2)

        q_x[-1, 0] = math.cos(psi_e + math.pi / 2)
        q_y[-1, 0] = math.sin(psi_e + math.pi / 2)

    # set up P_xx, P_xy, P_yy matrices
    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)                   # calculate curvature denominator
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den),
                          where=curv_den != 0)                          # divide where not zero (diag elements)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2   # make H symmetric

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = f_x + f_xy + f_y
    f = np.squeeze(f)   # remove non-singleton dimensions

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # original curvature part (static part)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

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

    # G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    # h = np.append(dev_max_right, dev_max_left)

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
