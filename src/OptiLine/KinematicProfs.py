import numpy as np
import math
from OptiLine.utils import calc_splines, conv_filt
from scipy.interpolate import splprep, splev

try:
    from numba import njit as _njit
except ImportError:
    def _njit(fn=None, **_):             # transparent no-op fallback
        return fn if fn is not None else lambda f: f


# mode_int encoding: 0 = accel_forw, 1 = decel_forw, 2 = decel_backw
@_njit(cache=True)
def _calc_ax_poss_nb(vx_start, radius,
                     ggv_vx, ggv_ax, ggv_ay,
                     ax_mach_vx, ax_mach_ax,
                     mu, dyn_model_exp, drag_coeff, m_veh,
                     mode_int):
    ax_max_tires = mu * np.interp(vx_start, ggv_vx, ggv_ax)
    ay_max_tires = mu * np.interp(vx_start, ggv_vx, ggv_ay)
    ay_used = vx_start * vx_start / radius

    if mode_int != 1 and ax_max_tires < 0.0:   # accel_forw or decel_backw
        ax_max_tires = -ax_max_tires
    elif mode_int == 1 and ax_max_tires > 0.0:  # decel_forw
        ax_max_tires = -ax_max_tires

    radicand = 1.0 - (ay_used / ay_max_tires) ** dyn_model_exp
    ax_avail_tires = ax_max_tires * radicand ** (1.0 / dyn_model_exp) if radicand > 0.0 else 0.0

    if mode_int == 0:   # accel_forw — apply motor limits
        ax_mach_tmp = np.interp(vx_start, ax_mach_vx, ax_mach_ax)
        ax_avail = ax_avail_tires if ax_avail_tires < ax_mach_tmp else ax_mach_tmp
    else:
        ax_avail = ax_avail_tires

    ax_drag = -(vx_start * vx_start) * drag_coeff / m_veh

    if mode_int != 2:   # accel_forw or decel_forw
        return ax_avail + ax_drag
    else:               # decel_backw
        return ax_avail - ax_drag

def calc_vel_profile(ax_max_machines: np.ndarray,
                     kappa: np.ndarray,
                     el_lengths: np.ndarray,
                     closed: bool,
                     drag_coeff: float,
                     m_veh: float,
                     ggv: np.ndarray = None,
                     loc_gg: np.ndarray = None,
                     v_max: float = None,
                     dyn_model_exp: float = 1.0,
                     mu: np.ndarray = None,
                     v_start: float = None,
                     v_end: float = None,
                     filt_window: int = None,
                     p_ggv: np.ndarray = None) -> np.ndarray:
    """

    .. description::
    Calculates a velocity profile using the tire and motor limits as good as possible.

    .. inputs::
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance,
                            i.e. simply by calculating F_x_drivetrain / m_veh
    :type ax_max_machines:  np.ndarray
    :param kappa:           curvature profile of given trajectory in rad/m (always unclosed).
    :type kappa:            np.ndarray
    :param el_lengths:      element lengths (distances between coordinates) of given trajectory.
    :type el_lengths:       np.ndarray
    :param closed:          flag to set if the velocity profile must be calculated for a closed or unclosed trajectory.
    :type closed:           bool
    :param drag_coeff:      drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           vehicle mass in kg.
    :type m_veh:            float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
                            ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type ggv:              np.ndarray
    :param loc_gg:          local gg diagrams along the path points: [[ax_max_0, ay_max_0], [ax_max_1, ay_max_1], ...],
                            accelerations in m/s2. ATTENTION: Insert either ggv + mu (optional) or loc_gg!
    :type loc_gg:           np.ndarray
    :param v_max:           Maximum longitudinal speed in m/s (optional if ggv is supplied, taking the minimum of the
                            fastest velocities covered by the ggv and ax_max_machines arrays then).
    :type v_max:            float
    :param dyn_model_exp:   exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param mu:              friction coefficients (always unclosed).
    :type mu:               np.ndarray
    :param v_start:         start velocity in m/s (used in unclosed case only).
    :type v_start:          float
    :param v_end:           end velocity in m/s (used in unclosed case only).
    :type v_end:            float
    :param filt_window:     filter window size for moving average filter (must be odd).
    :type filt_window:      int

    .. outputs::
    :return vx_profile:     calculated velocity profile (always unclosed).
    :rtype vx_profile:      np.ndarray

    .. notes::
    All inputs must be inserted unclosed, i.e. kappa[-1] != kappa[0], even if closed is set True! (el_lengths is kind of
    closed if closed is True of course!)

    case closed is True:
    len(kappa) = len(el_lengths) = len(mu) = len(vx_profile)

    case closed is False:
    len(kappa) = len(el_lengths) + 1 = len(mu) = len(vx_profile)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check if either ggv (and optionally mu) or loc_gg are handed in
    if (ggv is not None or mu is not None) and loc_gg is not None:
        raise RuntimeError("Either ggv and optionally mu OR loc_gg must be supplied, not both (or all) of them!")

    if ggv is None and loc_gg is None:
        raise RuntimeError("Either ggv or loc_gg must be supplied!")

    # check shape of loc_gg
    if loc_gg is not None:
        if loc_gg.ndim != 2:
            raise RuntimeError("loc_gg must have two dimensions!")

        if loc_gg.shape[0] != kappa.size:
            raise RuntimeError("Length of loc_gg and kappa must be equal!")

        if loc_gg.shape[1] != 2:
            raise RuntimeError("loc_gg must consist of two columns: [ax_max, ay_max]!")

    # check shape of ggv
    if ggv is not None and ggv.shape[1] != 3:
        raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")

    # check size of mu
    if mu is not None and kappa.size != mu.size:
        raise RuntimeError("kappa and mu must have the same length!")

    # check size of kappa and element lengths
    if closed and kappa.size != el_lengths.size:
        raise RuntimeError("kappa and el_lengths must have the same length if closed!")

    elif not closed and kappa.size != el_lengths.size + 1:
        raise RuntimeError("kappa must have the length of el_lengths + 1 if unclosed!")

    # check start and end velocities
    if not closed and v_start is None:
        raise RuntimeError("v_start must be provided for the unclosed case!")

    if v_start is not None and v_start < 0.0:
        v_start = 0.0
        print('WARNING: Input v_start was < 0.0. Using v_start = 0.0 instead!')

    if v_end is not None and v_end < 0.0:
        v_end = 0.0
        print('WARNING: Input v_end was < 0.0. Using v_end = 0.0 instead!')

    # check dyn_model_exp
    if not 1.0 <= dyn_model_exp <= 2.0:
        print('WARNING: Exponent for the vehicle dynamics model should be in the range [1.0, 2.0]!')

    # check shape of ax_max_machines
    if ax_max_machines.shape[1] != 2:
        raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

    # check v_max
    if v_max is None:
        if ggv is None:
            raise RuntimeError("v_max must be supplied if ggv is None!")
        else:
            v_max = min(ggv[-1, 0], ax_max_machines[-1, 0])

    else:
        # check if ggv covers velocity until v_max
        if ggv is not None and ggv[-1, 0] < v_max:
            raise RuntimeError("ggv has to cover the entire velocity range of the car (i.e. >= v_max)!")

        # check if ax_max_machines covers velocity until v_max
        if ax_max_machines[-1, 0] < v_max:
            raise RuntimeError("ax_max_machines has to cover the entire velocity range of the car (i.e. >= v_max)!")

    # ------------------------------------------------------------------------------------------------------------------
    # BRINGING GGV OR LOC_GG INTO SHAPE FOR EQUAL HANDLING AFTERWARDS --------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """For an equal/easier handling of every case afterwards we bring all cases into a form where the local ggv is made
    available for every waypoint, i.e. [ggv_0, ggv_1, ggv_2, ...] -> we have a three dimensional array p_ggv (path_ggv)
    where the first dimension is the waypoint, the second is the velocity and the third is the two acceleration columns
    -> DIM = NO_WAYPOINTS_CLOSED x NO_VELOCITY ENTRIES x 3"""

    # CASE 1: ggv supplied -> copy it for every waypoint (skip if pre-built p_ggv provided)
    if ggv is not None:
        if p_ggv is None or p_ggv.shape[0] != kappa.size:
            p_ggv = np.repeat(np.expand_dims(ggv, axis=0), kappa.size, axis=0)
        op_mode = 'ggv'

    # CASE 2: local gg diagram supplied -> add velocity dimension (artificial velocity of 10.0 m/s)
    else:
        p_ggv = np.expand_dims(np.column_stack((np.ones(loc_gg.shape[0]) * 10.0, loc_gg)), axis=1)
        op_mode = 'loc_gg'

    # ------------------------------------------------------------------------------------------------------------------
    # SPEED PROFILE CALCULATION (FB) -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # transform curvature kappa into corresponding radii (abs because curvature has a sign in our convention)
    radii = np.abs(np.divide(1.0, kappa, out=np.full(kappa.size, np.inf), where=kappa != 0.0))

    # call solver
    if not closed:
        vx_profile = __solver_fb_unclosed(p_ggv=p_ggv,
                                          ax_max_machines=ax_max_machines,
                                          v_max=v_max,
                                          radii=radii,
                                          el_lengths=el_lengths,
                                          mu=mu,
                                          v_start=v_start,
                                          v_end=v_end,
                                          dyn_model_exp=dyn_model_exp,
                                          drag_coeff=drag_coeff,
                                          m_veh=m_veh,
                                          op_mode=op_mode)

    else:
        vx_profile = __solver_fb_closed(p_ggv=p_ggv,
                                        ax_max_machines=ax_max_machines,
                                        v_max=v_max,
                                        radii=radii,
                                        el_lengths=el_lengths,
                                        mu=mu,
                                        dyn_model_exp=dyn_model_exp,
                                        drag_coeff=drag_coeff,
                                        m_veh=m_veh,
                                        op_mode=op_mode)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = conv_filt(signal=vx_profile,
                                filt_window=filt_window,
                                closed=closed)

    return vx_profile


def calc_vel_profile_fb2d(ax_max_machines: np.ndarray,
                          kappa: np.ndarray,
                          el_lengths: np.ndarray,
                          closed: bool,
                          drag_coeff: float,
                          m_veh: float,
                          ggv: np.ndarray = None,
                          loc_gg: np.ndarray = None,
                          v_max: float = None,
                          dyn_model_exp: float = 1.0,
                          mu: np.ndarray = None,
                          v_start: float = None,
                          v_end: float = None,
                          filt_window: int = None,
                          p_ggv: np.ndarray = None,
                          gg_upper=None,
                          gg_lower=None,
                          gg_range=None,
                          n_laps: int = 3,
                          return_accel: bool = False):
    """

    .. description::
    Alternative velocity-profile calculator based on the FBGA ``Fb2d`` solver
    (github.com/DRIVEWISE/FBGA, pip package ``fbga-py``). It is a drop-in
    alternative to :func:`calc_vel_profile`: it solves the same minimum-time,
    fixed-path velocity-profile problem, but integrates the longitudinal
    dynamics semi-analytically (see Frego et al., "Semi-Analytical Minimum Time
    Solutions with Velocity Constraints for Trajectory Following of Vehicles").

    The g-g bounds handed to ``Fb2d`` are built so that they reproduce exactly
    the acceleration model used inside :func:`calc_ax_poss`, i.e.::

        ax_max_tire(v) = mu * interp(v, ggv[:,0], ggv[:,1])
        ay_max_tire(v) = mu * interp(v, ggv[:,0], ggv[:,2])
        ax_avail(ay,v) = ax_max_tire * (1 - (|ay|/ay_max_tire)^exp)^(1/exp)
        drive : min(ax_avail, ax_machine(v)) - drag_coeff*v^2/m_veh
        brake : -ax_avail                    - drag_coeff*v^2/m_veh
        + hard top-speed cap at v_max

    so that, for the same inputs, both calculators refer to the same vehicle.

    .. inputs::
    Signature mirrors :func:`calc_vel_profile` (see there for the shared
    parameters). Additional / changed parameters:

    :param gg_upper:        optional custom drive bound ``ax = f(ay, v)`` handed
                            directly to ``Fb2d`` (the "generic acceleration"
                            feature). If None it is built from ``ggv`` as above.
    :param gg_lower:        optional custom brake bound ``ax = f(ay, v)``. If
                            None it is built from ``ggv`` as above.
    :param gg_range:        optional lateral bound. Either an ``fbga_py``
                            ``GgRangeMaxMin`` object, or a callable
                            ``ay_max = f(v)`` (a symmetric range +/-f(v) is
                            built from it). If None it is built from ``ggv``.
    :param n_laps:          number of laps the (closed) track is tiled over to
                            obtain a periodic profile (the middle lap is kept,
                            same trick as the closed forward-backward solver).
                            Must be odd so that a well-defined middle lap exists.
    :type n_laps:           int
    :param return_accel:    if True, also return the longitudinal acceleration
                            profile provided by the solver (same length as the
                            returned velocity profile).
    :type return_accel:     bool

    .. outputs::
    :return vx_profile:     calculated velocity profile (always unclosed, same
                            shape convention as :func:`calc_vel_profile`).
    :return ax_profile:     (only if return_accel is True) longitudinal
                            acceleration profile from the solver.

    .. notes::
    * Requires the optional dependency ``fbga-py`` (imported lazily so that the
      rest of the package works without it).
    * Only the ``ggv`` operating mode is supported. Position dependent inputs
      (``loc_gg`` or a spatially varying ``mu`` array) are not representable by
      the single velocity/lateral dependent g-g functions ``Fb2d`` expects; a
      ``mu`` array is collapsed to its mean and ``loc_gg`` is rejected.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # INPUT CHECKS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    try:
        import fbga_py as _fb
    except ImportError as exc:                                    # pragma: no cover
        raise ImportError(
            "calc_vel_profile_fb2d requires the optional dependency 'fbga-py' "
            "(pip install fbga-py). Install it or use vel_solver='fb'."
        ) from exc

    if loc_gg is not None and ggv is None:
        raise RuntimeError("calc_vel_profile_fb2d supports the ggv mode only (loc_gg is not supported)!")

    # which pieces still have to be derived from ggv (i.e. were not supplied explicitly)
    need_ggv = (gg_upper is None) or (gg_lower is None) or (gg_range is None)

    if need_ggv:
        if ggv is None:
            raise RuntimeError("ggv must be supplied unless gg_upper, gg_lower and gg_range are all provided!")
        if ggv.ndim != 2 or ggv.shape[1] != 3:
            raise RuntimeError("ggv diagram must consist of the three columns [vx, ax_max, ay_max]!")
        if ax_max_machines.shape[1] != 2:
            raise RuntimeError("ax_max_machines must consist of the two columns [vx, ax_max_machines]!")

    if closed and kappa.size != el_lengths.size:
        raise RuntimeError("kappa and el_lengths must have the same length if closed!")

    if not closed and kappa.size != el_lengths.size + 1:
        raise RuntimeError("kappa must have the length of el_lengths + 1 if unclosed!")

    if not closed and v_start is None:
        raise RuntimeError("v_start must be provided for the unclosed case!")

    if closed and n_laps % 2 == 0:
        raise RuntimeError("n_laps must be odd so that a well-defined middle lap exists!")

    if v_max is None:
        if ggv is None:
            raise RuntimeError("v_max must be supplied when ggv is not given!")
        v_max = float(min(ggv[-1, 0], ax_max_machines[-1, 0]))

    # collapse friction to a scalar (Fb2d g-g functions cannot depend on position)
    if mu is None:
        mu_s = 1.0
    else:
        mu_s = float(np.mean(mu))

    # ------------------------------------------------------------------------------------------------------------------
    # BUILD THE G-G BOUNDS (identical accel model to calc_ax_poss) ------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # default g-g bounds derived from ggv (used for whichever of upper/lower/range was not supplied)
    def _ay_max_tire(v):
        return mu_s * np.interp(v, ggv[:, 0], ggv[:, 2])

    if need_ggv:
        exp = dyn_model_exp

        def _ax_max_tire(v):
            return mu_s * np.interp(v, ggv[:, 0], ggv[:, 1])

        def _ax_machine(v):
            return np.interp(v, ax_max_machines[:, 0], ax_max_machines[:, 1])

        def _ax_avail_tire(ay, v):
            ay_max = _ay_max_tire(v)
            if ay_max <= 0.0:
                return 0.0
            radicand = 1.0 - (min(abs(ay) / ay_max, 1.0)) ** exp
            return _ax_max_tire(v) * radicand ** (1.0 / exp) if radicand > 0.0 else 0.0

        def _default_gg_upper(ay, v):                            # max longitudinal accel (drive)
            ax = min(_ax_avail_tire(ay, v), _ax_machine(v)) - v * v * drag_coeff / m_veh
            return min(ax, 0.0) if v >= v_max else ax            # hard top-speed cap

        def _default_gg_lower(ay, v):                            # min longitudinal accel (brake)
            return -_ax_avail_tire(ay, v) - v * v * drag_coeff / m_veh

    # longitudinal bounds: use the user-supplied callables where given, otherwise the ggv defaults
    upper_fn = gg_upper if gg_upper is not None else _default_gg_upper
    lower_fn = gg_lower if gg_lower is not None else _default_gg_lower

    # lateral range: object -> use directly; callable -> symmetric +/-f(v); None -> build from ggv
    if gg_range is None:
        range_obj = _fb.GgRangeMaxMin()
        range_obj.max = lambda v: _ay_max_tire(v)
        range_obj.min = lambda v: -_ay_max_tire(v)
    elif callable(gg_range):
        range_obj = _fb.GgRangeMaxMin()
        range_obj.max = lambda v: gg_range(v)
        range_obj.min = lambda v: -gg_range(v)
    else:
        range_obj = gg_range                                     # assumed to be a GgRangeMaxMin-like object

    solver = _fb.Fb2d(upper_fn, lower_fn, range_obj)

    # ------------------------------------------------------------------------------------------------------------------
    # SOLVE ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if closed:
        # el_lengths is closed (len == kappa). Tile over n_laps and keep the middle lap so that the resulting
        # profile is periodic (vx[0] matches vx[-1]) -- same rationale as the closed forward-backward solver.
        no_points = kappa.size
        lap_length = float(np.sum(el_lengths))
        s_lap = np.concatenate(([0.0], np.cumsum(el_lengths)[:-1]))     # arc length of each point, 0 .. L-el[-1]
        SS = np.concatenate([s_lap + i * lap_length for i in range(n_laps)])
        KK = np.tile(kappa, n_laps)
        v0 = v_max                                                # forward-backward settles the middle lap
        solver.compute(SS.tolist(), KK.tolist(), float(v0), float(v_max))
        AX, _, VX = solver.evaluate(SS.tolist())
        mid = n_laps // 2
        sel = slice(mid * no_points, (mid + 1) * no_points)
        vx_profile = np.asarray(VX, dtype=float)[sel]
        ax_profile = np.asarray(AX, dtype=float)[sel]
    else:
        # unclosed: kappa has len(el_lengths) + 1 points.
        SS = np.concatenate(([0.0], np.cumsum(el_lengths)))
        KK = np.asarray(kappa, dtype=float)
        v0 = v_start
        solver.compute(SS.tolist(), KK.tolist(), float(v0), float(v_max))
        AX, _, VX = solver.evaluate(SS.tolist())
        vx_profile = np.asarray(VX, dtype=float)
        ax_profile = np.asarray(AX, dtype=float)

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if filt_window is not None:
        vx_profile = conv_filt(signal=vx_profile,
                               filt_window=filt_window,
                               closed=closed)

    if return_accel:
        return vx_profile, ax_profile

    return vx_profile


def calc_vel_profile_solver(vel_solver: str = 'fb', **kwargs) -> np.ndarray:
    """

    .. description::
    Thin dispatcher that returns a velocity profile from either the vanilla
    forward-backward solver (:func:`calc_vel_profile`) or the FBGA-based
    alternative (:func:`calc_vel_profile_fb2d`), so that call sites can offer
    the choice with a single keyword. Both branches return the velocity profile
    with the same shape convention.

    .. inputs::
    :param vel_solver:  'fb' (default, vanilla forward-backward) or 'fb2d'
                        (FBGA Fb2d). Aliases: 'vanilla'/'forward_backward' -> fb,
                        'fbga'/'FBGA' -> fb2d.
    :type vel_solver:   str
    :param kwargs:      forwarded to the selected calculator. Besides the shared
                        keyword set of :func:`calc_vel_profile`, the fb2d-only
                        keywords ``gg_upper``, ``gg_lower``, ``gg_range``,
                        ``n_laps`` and ``return_accel`` may be supplied; they are
                        used by the 'fb2d' branch and silently dropped for 'fb'
                        (the vanilla solver has no custom-g-g concept). Likewise
                        ``p_ggv`` is only used by the 'fb' branch.

    .. outputs::
    :return vx_profile: calculated velocity profile.
    :rtype vx_profile:  np.ndarray
    """

    key = (vel_solver or 'fb').lower()

    if key in ('fb', 'vanilla', 'forward_backward', 'fb1d'):
        # drop fb2d-only keywords so they never reach the vanilla solver
        for _k in ('gg_upper', 'gg_lower', 'gg_range', 'n_laps', 'return_accel'):
            kwargs.pop(_k, None)
        return calc_vel_profile(**kwargs)

    if key in ('fb2d', 'fbga'):
        kwargs.pop('p_ggv', None)                # performance cache only used by the vanilla solver
        return calc_vel_profile_fb2d(**kwargs)

    raise ValueError(f"Unknown vel_solver '{vel_solver}'. Use 'fb' or 'fb2d'.")


def __solver_fb_unclosed(p_ggv: np.ndarray,
                         ax_max_machines: np.ndarray,
                         v_max: float,
                         radii: np.ndarray,
                         el_lengths: np.ndarray,
                         v_start: float,
                         drag_coeff: float,
                         m_veh: float,
                         op_mode: str,
                         mu: np.ndarray = None,
                         v_end: float = None,
                         dyn_model_exp: float = 1.0) -> np.ndarray:
    """

    .. description::
    Forward-backward solver for an unclosed (open) trajectory. Computes the velocity profile by first applying
    a forward acceleration pass from v_start, then a backward deceleration pass toward v_end.

    .. inputs::
    :param p_ggv:               per-waypoint ggv diagram: [vx, ax_max, ay_max] repeated for each point.
    :type p_ggv:                np.ndarray
    :param ax_max_machines:     longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines].
    :type ax_max_machines:      np.ndarray
    :param v_max:               maximum longitudinal speed in m/s.
    :type v_max:                float
    :param radii:               radius profile of the trajectory in m.
    :type radii:                np.ndarray
    :param el_lengths:          element lengths between consecutive trajectory points in m.
    :type el_lengths:           np.ndarray
    :param v_start:             start velocity in m/s.
    :type v_start:              float
    :param drag_coeff:          drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air.
    :type drag_coeff:           float
    :param m_veh:               vehicle mass in kg.
    :type m_veh:                float
    :param op_mode:             operation mode, either 'ggv' or 'loc_gg'.
    :type op_mode:              str
    :param mu:                  friction coefficients along the trajectory (optional, defaults to 1.0).
    :type mu:                   np.ndarray
    :param v_end:               end velocity in m/s (optional).
    :type v_end:                float
    :param dyn_model_exp:       exponent used in the vehicle dynamics model (usual range [1.0, 2.0]).
    :type dyn_model_exp:        float

    .. outputs::
    :return vx_profile:         calculated velocity profile (unclosed).
    :rtype vx_profile:          np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # handle mu
    if mu is None:
        mu = np.ones(radii.size)
        mu_mean = 1.0
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity profile estimate

        ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
        vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity profile estimate

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    # consider v_start
    if vx_profile[0] > v_start:
        vx_profile[0] = v_start

    # calculate acceleration profile
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=False,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)

    # consider v_end
    if v_end is not None and vx_profile[-1] > v_end:
        vx_profile[-1] = v_end

    # calculate deceleration profile
    vx_profile = __solver_fb_acc_profile(p_ggv=p_ggv,
                                         ax_max_machines=ax_max_machines,
                                         v_max=v_max,
                                         radii=radii,
                                         el_lengths=el_lengths,
                                         mu=mu,
                                         vx_profile=vx_profile,
                                         backwards=True,
                                         dyn_model_exp=dyn_model_exp,
                                         drag_coeff=drag_coeff,
                                         m_veh=m_veh)

    return vx_profile


def __solver_fb_closed(p_ggv: np.ndarray,
                       ax_max_machines: np.ndarray,
                       v_max: float,
                       radii: np.ndarray,
                       el_lengths: np.ndarray,
                       drag_coeff: float,
                       m_veh: float,
                       op_mode: str,
                       mu: np.ndarray = None,
                       dyn_model_exp: float = 1.0) -> np.ndarray:
    """

    .. description::
    Forward-backward solver for a closed (circuit) trajectory. Processes two laps to determine a consistent
    velocity profile, applying an acceleration pass followed by a deceleration pass on the doubled arrays.

    .. inputs::
    :param p_ggv:               per-waypoint ggv diagram: [vx, ax_max, ay_max] repeated for each point.
    :type p_ggv:                np.ndarray
    :param ax_max_machines:     longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines].
    :type ax_max_machines:      np.ndarray
    :param v_max:               maximum longitudinal speed in m/s.
    :type v_max:                float
    :param radii:               radius profile of the trajectory in m.
    :type radii:                np.ndarray
    :param el_lengths:          element lengths between consecutive trajectory points in m.
    :type el_lengths:           np.ndarray
    :param drag_coeff:          drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air.
    :type drag_coeff:           float
    :param m_veh:               vehicle mass in kg.
    :type m_veh:                float
    :param op_mode:             operation mode, either 'ggv' or 'loc_gg'.
    :type op_mode:              str
    :param mu:                  friction coefficients along the trajectory (optional, defaults to 1.0).
    :type mu:                   np.ndarray
    :param dyn_model_exp:       exponent used in the vehicle dynamics model (usual range [1.0, 2.0]).
    :type dyn_model_exp:        float

    .. outputs::
    :return vx_profile:         calculated velocity profile (unclosed, second lap extracted).
    :rtype vx_profile:          np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # FORWARD BACKWARD SOLVER ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = radii.size

    # handle mu
    if mu is None:
        mu = np.ones(no_points)*1
        mu_mean = 1.0*1
    else:
        mu_mean = np.mean(mu)

    # run through all the points and check for possible lateral acceleration
    if op_mode == 'ggv':
        # in ggv mode all ggvs are equal -> we can use the first one
        ay_max_global = mu_mean * np.amin(p_ggv[0, :, 2])   # get first lateral acceleration estimate
        vx_profile = np.sqrt(ay_max_global * radii)         # get first velocity estimate (radii must be positive!)

        # iterate until the initial velocity profile converges (break after max. 100 iterations)
        converged = False

        for i in range(10):
            vx_profile_prev_iteration = vx_profile

            ay_max_curr = mu * np.interp(vx_profile, p_ggv[0, :, 0], p_ggv[0, :, 2])
            vx_profile = np.sqrt(np.multiply(ay_max_curr, radii))

            # break the loop if the maximum change of the velocity profile was below 0.5%
            if np.max(np.abs(vx_profile / vx_profile_prev_iteration - 1.0)) < 0.005:
                converged = True
                break

        if not converged:
            print("The initial vx profile did not converge after 100 iterations, please check radii and ggv!")

    else:
        # in loc_gg mode all ggvs consist of a single line due to the missing velocity dependency, mu is None in this
        # case
        vx_profile = np.sqrt(p_ggv[:, 0, 2] * radii)        # get first velocity estimate (radii must be positive!)

    # cut vx_profile to car's top speed
    vx_profile[vx_profile > v_max] = v_max

    """We need to calculate the speed profile for three laps and keep the middle lap.

    Why 3 laps, not 2:
    The backward pass is implemented by *flipping* the array and running the same forward
    algorithm.  The forward algorithm only ever writes vx_profile[i+1] — it never touches
    vx_profile[0] (index 0 of the flipped array = vx[-1] of the original).

    With 2-lap doubling the flipped array has size 2*N.  vx[-1] of the second copy sits at
    flipped[0] and is therefore *never* constrained by the backward pass, regardless of
    how many iterations you run.  The car's speed at the end of the lap is never forced to
    account for the braking needed to enter the next lap's corner — so vx[-1] stays too
    high and the appended wrap-around acceleration ax = (vx[0]²-vx[-1]²)/(2*ds) comes out
    at -36 m/s².

    With 3 laps the flipped array has size 3*N.  In the flipped representation the corner
    at the start of the third copy sits at flipped[N-1] (= vx[0] of the third copy =
    the slow-corner speed, e.g. 15 m/s).  That creates an acceleration phase at index
    N-1 → N, so the forward-on-flipped algorithm constrains flipped[N] = vx[-1] of the
    middle (second) copy to the physically correct braking distance from the corner.
    After flipping back, result[2*N-1] = vx[-1] of the middle lap is correctly limited. ✓
    """

    # Triple the track arrays (fixed for both passes)
    radii_triple      = np.concatenate((radii,      radii,      radii),      axis=0)
    el_lengths_triple = np.concatenate((el_lengths, el_lengths, el_lengths), axis=0)
    mu_triple         = np.concatenate((mu,         mu,         mu),         axis=0)
    p_ggv_triple      = np.concatenate((p_ggv,      p_ggv,      p_ggv),      axis=0)

    # --- forward (acceleration) pass on 3 laps; keep middle lap ---
    vx_profile_triple = np.concatenate((vx_profile, vx_profile, vx_profile), axis=0)
    vx_profile_triple = __solver_fb_acc_profile(p_ggv=p_ggv_triple,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_triple,
                                                el_lengths=el_lengths_triple,
                                                mu=mu_triple,
                                                vx_profile=vx_profile_triple,
                                                backwards=False,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)
    vx_profile = vx_profile_triple[no_points:2 * no_points]   # middle lap

    # --- backward (deceleration) pass on 3 laps; keep middle lap ---
    # vx[-1] of the middle lap is now at flipped[N] (not flipped[0]) and IS
    # constrained by the acceleration phase that starts at flipped[N-1] = vx[0] corner.
    vx_profile_triple = np.concatenate((vx_profile, vx_profile, vx_profile), axis=0)
    vx_profile_triple = __solver_fb_acc_profile(p_ggv=p_ggv_triple,
                                                ax_max_machines=ax_max_machines,
                                                v_max=v_max,
                                                radii=radii_triple,
                                                el_lengths=el_lengths_triple,
                                                mu=mu_triple,
                                                vx_profile=vx_profile_triple,
                                                backwards=True,
                                                dyn_model_exp=dyn_model_exp,
                                                drag_coeff=drag_coeff,
                                                m_veh=m_veh)
    vx_profile = vx_profile_triple[no_points:2 * no_points]   # middle lap

    return vx_profile


def __solver_fb_acc_profile(p_ggv: np.ndarray,
                            ax_max_machines: np.ndarray,
                            v_max: float,
                            radii: np.ndarray,
                            el_lengths: np.ndarray,
                            mu: np.ndarray,
                            vx_profile: np.ndarray,
                            drag_coeff: float,
                            m_veh: float,
                            dyn_model_exp: float = 1.0,
                            backwards: bool = False) -> np.ndarray:
    """

    .. description::
    Applies a single forward or backward acceleration pass over the velocity profile. Iterates through
    acceleration phases and enforces physically possible velocities point by point using calc_ax_poss.

    .. inputs::
    :param p_ggv:               per-waypoint ggv diagram: [vx, ax_max, ay_max].
    :type p_ggv:                np.ndarray
    :param ax_max_machines:     longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines].
    :type ax_max_machines:      np.ndarray
    :param v_max:               maximum longitudinal speed in m/s.
    :type v_max:                float
    :param radii:               radius profile of the trajectory in m.
    :type radii:                np.ndarray
    :param el_lengths:          element lengths between consecutive trajectory points in m.
    :type el_lengths:           np.ndarray
    :param mu:                  friction coefficients along the trajectory.
    :type mu:                   np.ndarray
    :param vx_profile:          current velocity profile to be updated in-place.
    :type vx_profile:           np.ndarray
    :param drag_coeff:          drag coefficient including all constants: drag_coeff = 0.5 * c_w * A_front * rho_air.
    :type drag_coeff:           float
    :param m_veh:               vehicle mass in kg.
    :type m_veh:                float
    :param dyn_model_exp:       exponent used in the vehicle dynamics model (usual range [1.0, 2.0]).
    :type dyn_model_exp:        float
    :param backwards:           if True, performs a backward deceleration pass; otherwise forward acceleration.
    :type backwards:            bool

    .. outputs::
    :return vx_profile:         updated velocity profile after the acceleration/deceleration pass.
    :rtype vx_profile:          np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    no_points = vx_profile.size

    # check for reversed direction
    if backwards:
        radii_mod = np.flipud(radii)
        el_lengths_mod = np.flipud(el_lengths)
        mu_mod = np.flipud(mu)
        vx_profile = np.flipud(vx_profile)
    else:
        radii_mod = radii
        el_lengths_mod = el_lengths
        mu_mod = mu

    # Pre-extract constant arrays and mode integer for the numba kernel
    mode_int = 2 if backwards else 0    # 0 = accel_forw, 2 = decel_backw
    ax_mach_vx = ax_max_machines[:, 0]
    ax_mach_ax = ax_max_machines[:, 1]

    # ------------------------------------------------------------------------------------------------------------------
    # SEARCH START POINTS FOR ACCELERATION PHASES ----------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    vx_diffs = np.diff(vx_profile)
    acc_inds = np.where(vx_diffs > 0.0)[0]                  # indices of points with positive acceleration
    if acc_inds.size != 0:
        # check index diffs -> we only need the first point of every acceleration phase
        acc_inds_diffs = np.diff(acc_inds)
        acc_inds_diffs = np.insert(acc_inds_diffs, 0, 2)    # first point is always a starting point
        acc_inds_rel = acc_inds[acc_inds_diffs > 1]         # starting point indices for acceleration phases
    else:
        acc_inds_rel = []                                   # if vmax is low and can be driven all the time

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE VELOCITY PROFILE ---------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # cast np.array as a list
    acc_inds_rel = list(acc_inds_rel)

    # while we have indices remaining in the list
    while acc_inds_rel:
        # set index to first list element
        i = acc_inds_rel.pop(0)

        # start from current index and run until either the end of the lap or a termination criterion are reached
        while i < no_points - 1:

            ax_possible_cur = _calc_ax_poss_nb(
                float(vx_profile[i]), float(radii_mod[i]),
                p_ggv[i, :, 0], p_ggv[i, :, 1], p_ggv[i, :, 2],
                ax_mach_vx, ax_mach_ax,
                float(mu_mod[i]), dyn_model_exp, drag_coeff, m_veh, mode_int)

            vx_possible_next = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_cur * el_lengths_mod[i])

            if backwards:
                """
                We have to loop the calculation if we are in the backwards iteration (currently just once). This is 
                because we calculate the possible ax at a point i which does not necessarily fit for point i + 1 
                (which is i - 1 in the real direction). At point i + 1 (or i - 1 in real direction) we have a different 
                start velocity (vx_possible_next), radius and mu value while the absolute value of ax remains the same 
                in both directions.
                """

                # looping just once at the moment
                for j in range(1):
                    ax_possible_next = _calc_ax_poss_nb(
                        float(vx_possible_next), float(radii_mod[i + 1]),
                        p_ggv[i + 1, :, 0], p_ggv[i + 1, :, 1], p_ggv[i + 1, :, 2],
                        ax_mach_vx, ax_mach_ax,
                        float(mu_mod[i + 1]), dyn_model_exp, drag_coeff, m_veh, mode_int)

                    vx_tmp = math.sqrt(math.pow(vx_profile[i], 2) + 2 * ax_possible_next * el_lengths_mod[i])

                    if vx_tmp < vx_possible_next:
                        vx_possible_next = vx_tmp
                    else:
                        break

            # save possible next velocity if it is smaller than the current value
            if vx_possible_next < vx_profile[i + 1]:
                vx_profile[i + 1] = vx_possible_next

            i += 1

            # break current acceleration phase if next speed would be higher than the maximum vehicle velocity or if we
            # are at the next acceleration phase start index
            if vx_possible_next > v_max or (acc_inds_rel and i >= acc_inds_rel[0]):
                break

    # ------------------------------------------------------------------------------------------------------------------
    # POSTPROCESSING ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # flip output vel_profile if necessary
    if backwards:
        vx_profile = np.flipud(vx_profile)

    return vx_profile


def calc_ax_poss(vx_start: float,
                 radius: float,
                 ggv: np.ndarray,
                 mu: float,
                 dyn_model_exp: float,
                 drag_coeff: float,
                 m_veh: float,
                 ax_max_machines: np.ndarray = None,
                 mode: str = 'accel_forw') -> float:
    """
    This function returns the possible longitudinal acceleration in the current step/point.

    .. inputs::
    :param vx_start:        [m/s] velocity at current point
    :type vx_start:         float
    :param radius:          [m] radius on which the car is currently driving
    :type radius:           float
    :param ggv:             ggv-diagram to be applied: [vx, ax_max, ay_max]. Velocity in m/s, accelerations in m/s2.
    :type ggv:              np.ndarray
    :param mu:              [-] current friction value
    :type mu:               float
    :param dyn_model_exp:   [-] exponent used in the vehicle dynamics model (usual range [1.0,2.0]).
    :type dyn_model_exp:    float
    :param drag_coeff:      [m2*kg/m3] drag coefficient incl. all constants: drag_coeff = 0.5 * c_w * A_front * rho_air
    :type drag_coeff:       float
    :param m_veh:           [kg] vehicle mass
    :type m_veh:            float
    :param ax_max_machines: longitudinal acceleration limits by the electrical motors: [vx, ax_max_machines]. Velocity
                            in m/s, accelerations in m/s2. They should be handed in without considering drag resistance.
                            Can be set None if using one of the decel modes.
    :type ax_max_machines:  np.ndarray
    :param mode:            [-] operation mode, can be 'accel_forw', 'decel_forw', 'decel_backw'
                            -> determines if machine limitations are considered and if ax should be considered negative
                            or positive during deceleration (for possible backwards iteration)
    :type mode:             str

    .. outputs::
    :return ax_final:       [m/s2] final acceleration from current point to next one
    :rtype ax_final:        float
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARATIONS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # check inputs
    if mode not in ['accel_forw', 'decel_forw', 'decel_backw']:
        raise RuntimeError("Unknown operation mode for calc_ax_poss!")

    if mode == 'accel_forw' and ax_max_machines is None:
        raise RuntimeError("ax_max_machines is required if operation mode is accel_forw!")

    if ggv.ndim != 2 or ggv.shape[1] != 3:
        raise RuntimeError("ggv must have two dimensions and three columns [vx, ax_max, ay_max]!")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER TIRE POTENTIAL ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate possible and used accelerations (considering tires)
    ax_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 1])
    ay_max_tires = mu * np.interp(vx_start, ggv[:, 0], ggv[:, 2])
    ay_used = math.pow(vx_start, 2) / radius

    # during forward acceleration and backward deceleration ax_max_tires must be considered positive, during forward
    # deceleration it must be considered negative
    if mode in ['accel_forw', 'decel_backw'] and ax_max_tires < 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be positive but was negative!")
        ax_max_tires *= -1.0
    elif mode == 'decel_forw' and ax_max_tires > 0.0:
        print("WARNING: Inverting sign of ax_max_tires because it should be negative but was positve!")
        ax_max_tires *= -1.0

    radicand = 1.0 - math.pow(ay_used / ay_max_tires, dyn_model_exp)

    if radicand > 0.0:
        ax_avail_tires = ax_max_tires * math.pow(radicand, 1.0 / dyn_model_exp)
    else:
        ax_avail_tires = 0.0

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER MACHINE LIMITATIONS -------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # consider limitations imposed by electrical machines during forward acceleration
    if mode == 'accel_forw':
        # interpolate machine acceleration to be able to consider varying gear ratios, efficiencies etc.
        ax_max_machines_tmp = np.interp(vx_start, ax_max_machines[:, 0], ax_max_machines[:, 1])
        ax_avail_vehicle = min(ax_avail_tires, ax_max_machines_tmp)
    else:
        ax_avail_vehicle = ax_avail_tires

    # ------------------------------------------------------------------------------------------------------------------
    # CONSIDER DRAG ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate equivalent longitudinal acceleration of drag force at the current speed

    ax_drag = -math.pow(vx_start, 2) * drag_coeff / m_veh
    # ax_drag = 0


    # drag reduces the possible acceleration in the forward case and increases it in the backward case
    if mode in ['accel_forw', 'decel_forw']:
        ax_final = ax_avail_vehicle + ax_drag
        # attention: this value will now be negative in forward direction if tire is entirely used for cornering
    else:
        ax_final = ax_avail_vehicle - ax_drag

    return ax_final



def calc_ax_profile(vx_profile: np.ndarray,
                    el_lengths: np.ndarray,
                    eq_length_output: bool = False) -> np.ndarray:
    """
    .. description::
    The function calculates the acceleration profile for a given velocity profile.

    .. inputs::
    :param vx_profile:          array containing the velocity profile used as a basis for the acceleration calculations.
    :type vx_profile:           np.ndarray
    :param el_lengths:          array containing the element lengths between every point of the velocity profile.
    :type el_lengths:           np.ndarray
    :param eq_length_output:    assumes zero acceleration for the last point of the acceleration profile and therefore
                                returns ax_profile with equal length to vx_profile.
    :type eq_length_output:     bool

    .. outputs::
    :return ax_profile:         acceleration profile calculated for the inserted vx_profile.
    :rtype ax_profile:          np.ndarray

    .. notes::
    case eq_length_output is True:
    len(vx_profile) = len(el_lengths) + 1 = len(ax_profile)

    case eq_length_output is False:
    len(vx_profile) = len(el_lengths) + 1 = len(ax_profile) + 1
    """

    # check inputs
    if vx_profile.size != el_lengths.size + 1:
        raise RuntimeError("Array size of vx_profile should be 1 element bigger than el_lengths!")

    # calculate longitudinal acceleration profile array numerically: (v_end^2 - v_beg^2) / 2*s
    if eq_length_output:
        ax_profile = np.zeros(vx_profile.size)
        ax_profile[:-1] = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)
    else:
        ax_profile = (np.power(vx_profile[1:], 2) - np.power(vx_profile[:-1], 2)) / (2 * el_lengths)

    return ax_profile


def calc_t_profile(vx_profile: np.ndarray,
                   el_lengths: np.ndarray,
                   t_start: float = 0.0,
                   ax_profile: np.ndarray = None) -> np.ndarray:
    """

    .. description::
    Calculate a temporal duration profile for a given trajectory.

    .. inputs::
    :param vx_profile:  array containing the velocity profile.
    :type vx_profile:   np.ndarray
    :param el_lengths:  array containing the element lengths between every point of the velocity profile.
    :type el_lengths:   np.ndarray
    :param t_start:     start time in seconds added to first array element.
    :type t_start:      float
    :param ax_profile:  acceleration profile fitting to the velocity profile.
    :type ax_profile:   np.ndarray

    .. outputs::
    :return t_profile:  time profile for the given velocity profile.
    :rtype t_profile:   np.ndarray

    .. notes::
    len(el_lengths) + 1 = len(t_profile)

    len(vx_profile) and len(ax_profile) must be >= len(el_lengths) as the temporal duration from one point to the next
    is only calculated based on the previous point.
    """

    # check inputs
    if vx_profile.size < el_lengths.size:
        raise RuntimeError("vx_profile and el_lenghts must have at least the same length!")

    if ax_profile is not None and ax_profile.size < el_lengths.size:
        raise RuntimeError("ax_profile and el_lenghts must have at least the same length!")

    # calculate acceleration profile if required
    if ax_profile is None:
        ax_profile = calc_ax_profile(vx_profile=vx_profile,
                                    el_lengths=el_lengths,
                                    eq_length_output=False)

    # calculate temporal duration of every step between two points
    no_points = el_lengths.size
    ax = ax_profile[:no_points]
    vx = vx_profile[:no_points]
    ax_nonzero = ~np.isclose(ax, 0.0)
    discriminant = np.maximum(vx**2 + 2 * ax * el_lengths, 0.0)
    t_steps = np.where(
        ax_nonzero,
        (-vx + np.sqrt(discriminant)) / np.where(ax_nonzero, ax, 1.0),
        el_lengths / vx
    )

    # calculate temporal duration profile out of steps
    t_profile = np.insert(np.cumsum(t_steps), 0, 0.0) + t_start

    return t_profile


def curvature_profile(points, s=3):
    """

    .. description::
    Fits a parametric spline to a set of 2D points and computes the curvature profile at each point.

    .. inputs::
    :param points:      array of 2D coordinates [[x0, y0], [x1, y1], ...].
    :type points:       np.ndarray
    :param s:           smoothing factor for the spline fit (passed to splprep).
    :type s:            float

    .. outputs::
    :return tck:        spline representation (knots, coefficients, degree) as returned by splprep.
    :rtype tck:         tuple
    :return curvatures: curvature values at each input point in rad/m.
    :rtype curvatures:  np.ndarray
    """

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Fit the spline to the points
    tck, u = splprep([x, y], s=s)

    # Evaluate first and second derivatives of the spline
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    # Compute curvature
    curvatures = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    return tck, curvatures

def curvature_profile2(reftrack):
    """

    .. description::
    Computes the curvature profile of a closed reference track using cubic spline coefficients. Splines are
    calculated via calc_splines and curvature is derived analytically from the first and second derivatives.

    .. inputs::
    :param reftrack:    reference track array where columns 0 and 1 are the x and y coordinates respectively.
    :type reftrack:     np.ndarray

    .. outputs::
    :return curv:       curvature profile of the reference track in rad/m.
    :rtype curv:        np.ndarray
    :return coeffs_x:   cubic spline coefficients for the x coordinate.
    :rtype coeffs_x:    np.ndarray
    :return coeffs_y:   cubic spline coefficients for the y coordinate.
    :rtype coeffs_y:    np.ndarray
    """

    coeffs_x, coeffs_y, M, normvec_norm2 = calc_splines(path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])))
    # dx[i] = (1/ds_f[i])*coeffs_x[i,1]
    # dy[i] = (1/ds_f[i])*coeffs_y[i,1]
    # ddx[i] = (1/ds_f[i])**2*2*coeffs_x[i,2]
    # ddy[i] = (1/ds_f[i])**2*2*coeffs_y[i,2]
    dx = coeffs_x[:, 1]
    dy = coeffs_y[:, 1]
    ddx = 2 * coeffs_x[:, 2]
    ddy = 2 * coeffs_y[:, 2]
    curv = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    return  curv,coeffs_x,coeffs_y

def cumulative_distances(distances):
    """

    .. description::
    Computes cumulative distances along a path, starting from zero, given segment lengths.

    .. inputs::
    :param distances:           array of segment lengths between consecutive path points.
    :type distances:            np.ndarray

    .. outputs::
    :return cumulative_d:       cumulative distance array starting at 0.0, length equal to len(distances) - 1 + 1.
    :rtype cumulative_d:        np.ndarray
    """

    cumulative_d = np.cumsum(distances[:-1])
    cumulative_d = np.insert(cumulative_d, 0, 0)   
    return cumulative_d
