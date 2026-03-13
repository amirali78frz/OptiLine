import numpy as np
import math
import matplotlib.pyplot as plt
import quadprog
import time


def opt_min_curv(reftrack: np.ndarray,
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
    This function uses a QP solver to minimize the summed curvature of a path by moving the path points along their
    normal vectors within the track width. The function can be used for closed and unclosed tracks. For unclosed tracks
    the heading psi_s and psi_e is enforced on the first and last point of the reftrack. Furthermore, in case of an
    unclosed track, the first and last point of the reftrack are not subject to optimization and stay same.

    Please refer to our paper for further information:
    Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann
    Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar
    DOI: 10.1080/00423114.2019.1631455

    Hint: CVXOPT can be used as a solver instead of quadprog by uncommenting the import and corresponding code section.

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
    :return alpha_mincurv:  solution vector of the opt. problem containing the lateral shift in m for every point.
    :rtype alpha_mincurv:   np.ndarray
    :return curv_error_max: maximum curvature error when comparing the curvature calculated on the basis of the
                            linearization around the original refererence track and around the solution.
    :rtype curv_error_max:  float
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

    A_ex_b[np.arange(no_splines), np.arange(no_splines) * 4 + 1] = 1    # 1 * b_ix = E_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_b[-1, -4:] = np.array([0, 1, 2, 3])

    # create extraction matrix -> only c_i coefficients of the solved linear equation system are needed for curvature
    # information
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)

    A_ex_c[np.arange(no_splines), np.arange(no_splines) * 4 + 2] = 2    # 2 * c_ix = D_x * x

    # coefficients for end of spline (t = 1)
    if not closed:
        A_ex_c[-1, -4:] = np.array([0, 0, 2, 6])

    # invert matrix A resulting from the spline setup linear equation system and apply extraction matrix
    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    # set up M_x and M_y matrices including the gradient information, i.e. bring normal vectors into matrix form
    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))

    rows_0 = np.arange(no_splines) * 4
    rows_1 = rows_0 + 1
    cols_0 = np.arange(no_splines)
    cols_1 = np.arange(1, no_splines + 1) % no_points  # wraps last index to 0 for closed track

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

    # ------------------------------------------------------------------------------------------------------------------
    # KAPPA CONSTRAINTS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)

    # this part is multiplied by alpha within the optimization (variable part)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)

    # original curvature part (static part)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)  # multiplied by -1 as only LE conditions are poss.
    con_stack = np.append(con_ge, con_le)

    # ------------------------------------------------------------------------------------------------------------------
    # CALL QUADRATIC PROGRAMMING ALGORITHM -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """
    quadprog interface description taken from 
    https://github.com/stephane-caron/qpsolvers/blob/master/qpsolvers/quadprog_.py

    Solve a Quadratic Program defined as:

        minimize
            (1/2) * alpha.T * H * alpha + f.T * alpha

        subject to
            G * alpha <= h
            A * alpha == b

    using quadprog <https://pypi.python.org/pypi/quadprog/>.

    Parameters
    ----------
    H : numpy.array
        Symmetric quadratic-cost matrix.
    f : numpy.array
        Quadratic-cost vector.
    G : numpy.array
        Linear inequality constraint matrix.
    h : numpy.array
        Linear inequality constraint vector.
    A : numpy.array, optional
        Linear equality constraint matrix.
    b : numpy.array, optional
        Linear equality constraint vector.
    initvals : numpy.array, optional
        Warm-start guess vector (not used).

    Returns
    -------
    alpha : numpy.array
            Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    The quadprog solver only considers the lower entries of `H`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """

    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # constrain resulting path to reference line at start- and end-point for open tracks
    if not closed and fix_s:
        dev_max_left[0] = 0.05
        dev_max_right[0] = 0.05

    if not closed and fix_e:
        dev_max_left[-1] = 0.05
        dev_max_right[-1] = 0.05

    # check that there is space remaining between left and right maximum deviation (both can be negative as well!)
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")

    # consider value boundaries (-dev_max_left <= alpha <= dev_max_right)
    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # save start time
    t_start = time.perf_counter()

    # solve problem (CVXOPT) -------------------------------------------------------------------------------------------
    # args = [cvxopt.matrix(H), cvxopt.matrix(f), cvxopt.matrix(G), cvxopt.matrix(h)]
    # sol = cvxopt.solvers.qp(*args)
    #
    # if 'optimal' not in sol['status']:
    #     print("WARNING: Optimal solution not found!")
    #
    # alpha_mincurv = np.array(sol['x']).reshape((H.shape[1],))

    # solve problem (quadprog) -----------------------------------------------------------------------------------------
    alpha_mincurv = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]

    # print runtime into console window
    if print_debug:
        print("Solver runtime opt_min_curv: " + "{:.3f}".format(time.perf_counter() - t_start) + "s")

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE CURVATURE ERROR ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate curvature once based on original linearization and once based on a new linearization around the solution
    q_x_tmp = q_x + np.matmul(M_x, np.expand_dims(alpha_mincurv, 1))
    q_y_tmp = q_y + np.matmul(M_y, np.expand_dims(alpha_mincurv, 1))

    x_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x_tmp)
    y_prime_tmp = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y_tmp)

    x_prime_prime = np.squeeze(np.matmul(T_c, q_x) + np.matmul(T_nx, np.expand_dims(alpha_mincurv, 1)))
    y_prime_prime = np.squeeze(np.matmul(T_c, q_y) + np.matmul(T_ny, np.expand_dims(alpha_mincurv, 1)))

    xp_d = np.diag(x_prime)
    yp_d = np.diag(y_prime)
    xp_tmp_d = np.diag(x_prime_tmp)
    yp_tmp_d = np.diag(y_prime_tmp)

    curv_orig_lin = (xp_d * y_prime_prime - yp_d * x_prime_prime) / np.power(xp_d**2 + yp_d**2, 1.5)
    curv_sol_lin = (xp_tmp_d * y_prime_prime - yp_tmp_d * x_prime_prime) / np.power(xp_tmp_d**2 + yp_tmp_d**2, 1.5)

    if plot_debug:
        plt.plot(curv_orig_lin)
        plt.plot(curv_sol_lin)
        plt.legend(("original linearization", "solution based linearization"))
        plt.show()

    # calculate maximum curvature error
    curv_error_max = np.amax(np.abs(curv_sol_lin - curv_orig_lin))

    return alpha_mincurv, curv_error_max


class ConstrainedCMAES_t:
    """
        CMAES optimizer.

        Parameters:
        - f_t: The objective function to minimize.
        - mean: initial mean.
        - sigma: Initial step size.
        - popsize: Population size.
        - bounds1: Upper bounds.
        - bounds2: Lower bounds.
    """
    def __init__(self, f_t,mean, sigma, popsize, bounds1=None, bounds2=None):
        self.mean = np.array(mean)
        self.sigma = sigma
        self.popsize = popsize
        self.dim = len(mean)
        self.cov_matrix = np.eye(self.dim)
        self.bounds1 = bounds1
        self.bounds2 = bounds2
        self.ft = f_t
        # CMA-ES parameters
        self.weights = np.log(self.popsize/2 + 1) - np.log(np.arange(1, self.popsize + 1))
        self.weights[self.weights < 0] = 0
        self.weights /= np.sum(self.weights)
        self.mu = int(self.popsize / 2)
        self.mu_w = 1 / np.sum(self.weights[:self.mu]**2)
        
        # Learning rates and constants
        self.c_sigma = (self.mu_w + 2) / (self.dim + self.mu_w + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_w - 1) / (self.dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_w/self.dim) / (self.dim + 4 + 2*self.mu_w/self.dim)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_w)
        self.cmu = min(1 - self.c1, 2 * (self.mu_w - 2 + 1/self.mu_w) / ((self.dim + 2)**2 + self.mu_w))

        # Evolution paths
        self.p_c = np.zeros(self.dim)
        self.p_sigma = np.zeros(self.dim)

        # Expected length of N(0,I)
        self.chi_n = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))

    def sample_population(self):
        """

        .. description::
        Draws a new population of candidate solutions from the current multivariate Gaussian distribution and
        clips them to the specified bounds if provided.

        .. outputs::
        :return samples:    array of candidate solutions with shape (popsize, dim), clipped to [bounds2, bounds1].
        :rtype samples:     np.ndarray
        """

        samples = np.random.multivariate_normal(self.mean, self.sigma**2 * self.cov_matrix, self.popsize)
        if self.bounds1 is not None:
            samples = np.clip(samples, self.bounds2, self.bounds1)
        return samples

    def objective_function(self, ds):
        return self.ft(ds)

    def update(self, solutions, fitness):
        """

        .. description::
        Updates the CMA-ES internal state (mean, evolution paths, covariance matrix, step size) using the
        current population and their fitness values. Enforces positive semi-definiteness of the covariance matrix.

        .. inputs::
        :param solutions:   array of candidate solutions evaluated in the current generation, shape (popsize, dim).
        :type solutions:    np.ndarray
        :param fitness:     array of objective function values corresponding to each solution, shape (popsize,).
        :type fitness:      np.ndarray
        """

        # Sort solutions
        sorted_indices = np.argsort(fitness)
        sorted_solutions = solutions[sorted_indices]

        # Calculate weighted mean of the new population
        old_mean = self.mean.copy()
        self.mean = np.sum((self.weights[:self.mu, np.newaxis] * sorted_solutions[:self.mu]), axis=0)

        # Update evolution paths
        y = self.mean - old_mean
        try:
            C_2 = np.linalg.cholesky(self.cov_matrix)
        except np.linalg.LinAlgError:
            print("Warning: Cholesky decomposition failed. Using diagonal matrix instead.")
            C_2 = np.diag(np.sqrt(np.abs(np.diag(self.cov_matrix))))
        
        z = np.linalg.solve(C_2, y) / self.sigma

        h_sigma = (np.linalg.norm(self.p_sigma) / 
                   np.sqrt(1 - (1-self.c_sigma)**(2*(self.iterations+1))) / self.chi_n
                   < (1.4 + 2/(self.dim+1)))

        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_w) * z
        self.p_c = (1 - self.c_c) * self.p_c + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_w) * y / self.sigma

        # Adapt covariance matrix
        c1a = self.c1 * (1 - (1-h_sigma**2) * self.c_c * (2-self.c_c))
        self.cov_matrix = ((1 - c1a - self.cmu) * self.cov_matrix +
                           c1a * np.outer(self.p_c, self.p_c) +
                           self.cmu * np.sum(self.weights[:self.mu, np.newaxis, np.newaxis] * 
                                             (sorted_solutions[:self.mu] - old_mean)[:, :, np.newaxis] * 
                                             (sorted_solutions[:self.mu] - old_mean)[:, np.newaxis, :], axis=0) / 
                           self.sigma**2)

        # Adapt step size
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.p_sigma) / self.chi_n - 1))

        # Ensure positive semidefinite covariance matrix
        epsilon = 1e-8
        self.cov_matrix += epsilon * np.eye(self.dim)
        eigvals, eigvecs = np.linalg.eigh(self.cov_matrix)
        eigvals = np.maximum(eigvals, epsilon)
        self.cov_matrix = np.dot(eigvecs, np.dot(np.diag(eigvals), eigvecs.T))

    def optimize(self, iterations):
        """

        .. description::
        Runs the CMA-ES optimization loop for a fixed number of generations. Each generation samples a new
        population, evaluates the objective, and updates the distribution parameters.

        .. inputs::
        :param iterations:  number of generations to run the optimization.
        :type iterations:   int

        .. outputs::
        :return mean:       optimized mean vector representing the best solution estimate.
        :rtype mean:        np.ndarray
        """

        self.iterations = 0
        for _ in range(iterations):
            samples = self.sample_population()
            fitness = np.array([self.objective_function(s) for s in samples])
            self.update(samples, fitness)
            self.iterations += 1
        return self.mean



class ZORM:
    def __init__(self, func, x0, num_iterations, mu=0.05, h=0.001, grad_type='noth', constraint_type='notc'):
        """
        ZORM optimizer.

        Parameters:
        - func: The objective function to minimize.
        - mu: Smoothing parameter for gradient approximation.
        - h: Step size.
        - x0: initial guess
        - num_iterations: number of iterations
        - grad_type: 'noth' for forward gradient, 'h' for symmetric (central difference).
        - constraint_type: 'notc' for unconstrained, 'c' for constrained optimization.
        """
        self.func = func
        self.mu = mu
        self.h = h
        self.x0=x0
        self.T = num_iterations
        self.grad_type = grad_type
        self.constraint_type = constraint_type

    def _sample_gaussian(self, d):
        return np.random.normal(0, 1, size=d)

    def _feasible_projection(self, x, min, max):
        y = np.clip(x,min,max)
        return y

    def _grad(self, x):
        u = self._sample_gaussian(len(x))
        perturbation = self.mu * u
        if self.grad_type == 'noth':
            g = (self.func(x + perturbation) - self.func(x)) / self.mu
        elif self.grad_type == 'h':
            g = (self.func(x + perturbation) - self.func(x - perturbation)) / (2 * self.mu)
        else:
            raise ValueError("Invalid gradient type. Use 'noth' or 'h'.")
        return g * u

    def _step(self, x):
        grad_est = self._grad(x)
        return x - self.h * grad_est

    def optimize(self, lower_bounds=None, upper_bounds=None):
        """
        Runs the ZORM optimization.

        Parameters:
        - x0: Initial guess (1D numpy array).
        - T: Number of iterations.
        - lower_bounds: Lower bounds matrix G for constraints (only needed if constraint_type='c').
        - upper_bounds: Upper bounds vector h for constraints (only needed if constraint_type='c').

        Returns:
        - x: A (dim, T+1) array of iterates.
        """
        dim = len(self.x0)
        x = np.zeros((dim, self.T + 1))
        x[:, 0] = self.x0
        if lower_bounds is None or upper_bounds is None:
                    raise ValueError("Bounds must be provided when constraint_type is 'c'.")
        for k in range(self.T):
            new_x = self._step(x[:, k])
            if self.constraint_type == 'c':
                new_x = self._feasible_projection(new_x, lower_bounds, upper_bounds)
            x[:, k + 1] = new_x

        return x
    
from OptiLine.utils import calc_splines,create_raceline, calc_head_curv_an, H_f, import_veh_dyn_info
from OptiLine.KinematicProfs import calc_vel_profile, calc_ax_profile, calc_t_profile , cumulative_distances

class Opt_min_CurvTime:
    def __init__(self, reftrack,center, mu=0.05, h=0.001, kapb = 0.7, sfty = 1, si = 0.8, vm = 22.88, m_veh = 3, drag_coeff =0.0045,  MC = 1, min_s =0.02 , max_s=0.4, sigma = 0.001,\
                  iterations_ZO= 300, iterations_CMA=30, popsize = 16  ,ggv_import_path="maps/ggv.csv",ax_max_machines_import_path="maps/ax_max_machines.csv",fw=3):
        """
        Min Curv and Time optimizer.

        Parameters:
        - reftrack: reference track array containing the reference line and track widths [x, y, w_tr_right, w_tr_left].
        - center: array containing the center line and track widths [x, y, w_tr_right, w_tr_left].
        - mu: Smoothing parameter for gradient approximation for ZO solver
        - h: Step size for ZO solver
        - iterations_ZO: number of iterations for ZO solver
        - sfty: half of the vehicle width
        - kapb = bound on the maximum allowed curvature
        - si: interpolation step size of the race line
        - vm: maximum available velocity
        - MC: Monte Carlo number of repetion for ZO solvers
        - min_s: minimum curv length
        - max_s: maximum curv length
        - sigma: Initial covariance for CMA-es solver
        - popsize: population size for CMA-es solver
        - iterations_CMA: number of iterations for ZORM solver
        - iterations_CMA: number of iterations for CMA-es solver
        - ggv_import_path: Path for importing ggv file
        - ax_max_machines_import_path: path for importing ax_max_machines file.
        - fw: filter window lengths for convolution (moving average) filtering of velocity profile
        - m_veh: vehicle mass
        - drag_coeff: drag coefficient
        """
        self.reftrack = reftrack
        self.mu = mu
        self.h = h
        self.ggv_import_path = ggv_import_path
        self.ax_max_machines_import_path = ax_max_machines_import_path
        self.sfty = sfty
        self.kapb = kapb
        self.si = si
        self.vm = vm
        self.MC = MC
        self.min_s=min_s
        self.max_s=max_s
        self.iterations_ZO=iterations_ZO
        self.iterations_CMA=iterations_CMA
        self.sigma = sigma
        self.popsize = popsize
        self.fw = fw
        self.m_veh=m_veh
        self.drag_coeff=drag_coeff
        self.center=center
        lengths = np.sqrt(np.sum(np.power(np.diff(self.reftrack[:,0:2], axis=0), 2), axis=1))
        lengths=np.append(lengths, lengths[0])
        self.lengths=lengths
        self.ggv,self.ax_max_machines =import_veh_dyn_info(ggv_import_path=self.ggv_import_path,ax_max_machines_import_path=self.ax_max_machines_import_path)

        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((center[:, 0:2], center[0, 0:2])),use_dist_scaling=True)
        
        self.bound1 = center[:, 0:2] - normvec_norm * np.expand_dims(center[:, 2], axis=1)
        self.bound2 = center[:, 0:2] + normvec_norm * np.expand_dims(center[:, 3], axis=1)


    def f_t(self,ds):
        """

        .. description::
        Objective function for the curve-length optimization. Given a set of spline segment lengths ds, computes
        the minimum-curvature raceline, interpolates the path, calculates the velocity and acceleration profiles,
        and returns the estimated lap time.

        .. inputs::
        :param ds:          array of spline segment lengths used to parameterize the reference track.
        :type ds:           np.ndarray

        .. outputs::
        :return laptime:    estimated lap time in seconds for the raceline generated from the given segment lengths.
        :rtype laptime:     float
        """

        sfty = self.sfty
        kapb = self.kapb

        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),el_lengths=ds)
        H, f, G , h = H_f(reftrack=self.reftrack,
                                                    normvectors=normvec_norm,
                                                    A=M,
                                                    kappa_bound=kapb,
                                                    w_veh=sfty,
                                                    closed=True)
        
        alpha_m = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]
        si=self.si
        raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
        spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=self.reftrack[:, :2],
                        normvectors=normvec_norm,
                        alpha=alpha_m,
                        stepsize_interp=si,)
        

        psi_vel_opt, kappa_opt =calc_head_curv_an(coeffs_x=coeffs_x_opt,
                            coeffs_y=coeffs_y_opt,
                            ind_spls=spline_inds_opt_interp,
                            t_spls=t_vals_opt_interp)

        # s_splines = cumulative_distances(el_lengths_opt_interp)

        vm =self.vm
        fw = 3
        vx_profile_opt = calc_vel_profile(ggv=self.ggv,
                                ax_max_machines=self.ax_max_machines,
                                v_max=vm,
                                kappa=kappa_opt,
                                el_lengths=el_lengths_opt_interp,
                                closed=True,
                                filt_window=fw,
                                dyn_model_exp=1.0,
                                drag_coeff=self.drag_coeff,
                                m_veh=self.m_veh,
                                v_start = 0.0)

        # calculate longitudinal acceleration profile
        vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
        ax_profile_opt = calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                        el_lengths=el_lengths_opt_interp,
                                        eq_length_output=False)

        # calculate laptime
        t_profile_cl = calc_t_profile(vx_profile=vx_profile_opt,
                                    ax_profile=ax_profile_opt,
                                    el_lengths=el_lengths_opt_interp)
        
        return t_profile_cl[-1]
    
    def CurveLenOpt(self,solver='ZO'):
        """

        .. description::
        Optimizes the spline segment lengths of the reference track to minimize lap time. Supports two solvers:
        zeroth-order random method (ZO) and CMA-ES (CMA). Monte Carlo averaging is applied over self.MC runs.

        .. inputs::
        :param solver:      optimization solver to use. 'ZO' for zeroth-order random method, 'CMA' for CMA-ES.
        :type solver:       str

        .. outputs::
        :return ds_ff:      optimized array of spline segment lengths averaged over Monte Carlo runs.
        :rtype ds_ff:       np.ndarray
        """

        if solver == 'ZO':
            ds_0 = self.lengths
            ds_ff = np.zeros_like(ds_0)
            for i in range(self.MC):
                ZOs = ZORM(self.f_t, ds_0, self.iterations_ZO, mu=self.mu, h=self.h, constraint_type='c')
                ds = ZOs.optimize(lower_bounds=self.min_s,upper_bounds=self.max_s)
                ds_ff += ds[:,-1]
            ds_ff=ds_ff/self.MC
            return ds_ff

        if solver == 'CMA':
            mean = self.lengths
            sigma = 0.001
            popsize = 16
            s_cmaa = np.zeros_like(mean)
            mc=1
            for i in range(self.MC):
                cma_es_s = ConstrainedCMAES_t(self.f_t,mean, sigma, popsize, bounds1=np.ones_like(self.lengths)*self.max_s, bounds2=np.ones_like(self.lengths)*self.min_s)
                s_cmai= cma_es_s.optimize(iterations=self.iterations_CMA)
                s_cmaa += s_cmai
            s_cmaa = s_cmaa/self.MC
            return s_cmaa

    def generate_raceline(self,ds=None,solver='ZO'):
        """

        .. description::
        Generates the optimized raceline geometry for a given set of spline segment lengths. If no segment
        lengths are provided, runs CurveLenOpt first to obtain them.

        .. inputs::
        :param ds:          array of spline segment lengths. If None, computed via CurveLenOpt.
        :type ds:           np.ndarray
        :param solver:      solver to use if ds is None. 'ZO' or 'CMA'.
        :type solver:       str

        .. outputs::
        :return raceline_interp:    interpolated raceline coordinates [x, y].
        :rtype raceline_interp:     np.ndarray
        :return ds:                 spline segment lengths used (either provided or optimized).
        :rtype ds:                  np.ndarray
        """

        if ds is None:
            ds = self.CurveLenOpt(solver)

        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),el_lengths=ds)
        H, f, G , h = H_f(reftrack=self.reftrack,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=self.sfty,
                                                 closed=True)
                                                 

        alpha_m_s = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]
        raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
        spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=self.reftrack[:, :2],
                    normvectors=normvec_norm,
                    alpha=alpha_m_s,
                    stepsize_interp=self.si)
        return raceline_interp,ds
    
    def generate_kinProfs(self,ds=None,solver='ZO'):
        """

        .. description::
        Generates the full kinematic profiles (velocity, acceleration, curvature, time, raceline) for a given
        set of spline segment lengths. If no segment lengths are provided, runs CurveLenOpt first.

        .. inputs::
        :param ds:          array of spline segment lengths. If None, computed via CurveLenOpt.
        :type ds:           np.ndarray
        :param solver:      solver to use if ds is None. 'ZO' or 'CMA'.
        :type solver:       str

        .. outputs::
        :return s_splines:          cumulative distance profile along the raceline in m.
        :rtype s_splines:           np.ndarray
        :return vx_profile_opt:     optimized velocity profile in m/s.
        :rtype vx_profile_opt:      np.ndarray
        :return ax_profile_opt:     longitudinal acceleration profile in m/s2.
        :rtype ax_profile_opt:      np.ndarray
        :return kappa_opt:          curvature profile of the raceline in rad/m.
        :rtype kappa_opt:           np.ndarray
        :return t_profile_cl:       lap time profile in seconds (cumulative).
        :rtype t_profile_cl:        np.ndarray
        :return raceline_interp:    interpolated raceline coordinates [x, y].
        :rtype raceline_interp:     np.ndarray
        """

        if ds is None:
            ds = self.CurveLenOpt(solver)
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),el_lengths=ds)
        H, f, G , h = H_f(reftrack=self.reftrack,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=self.sfty,
                                                 closed=True)
                                                 

        alpha_m_s = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]
        
        raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
        spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=self.reftrack[:, :2],
                    normvectors=normvec_norm,
                    alpha=alpha_m_s,
                    stepsize_interp=self.si)

        psi_vel_opt, kappa_opt =calc_head_curv_an(coeffs_x=coeffs_x_opt,
                      coeffs_y=coeffs_y_opt,
                      ind_spls=spline_inds_opt_interp,
                      t_spls=t_vals_opt_interp)

        s_splines = cumulative_distances(el_lengths_opt_interp)
        vx_profile_opt = calc_vel_profile(ggv=self.ggv,
                         ax_max_machines=self.ax_max_machines,
                         v_max=self.vm,
                         kappa=kappa_opt,
                         el_lengths=el_lengths_opt_interp,
                         closed=True,
                         filt_window=self.fw,
                         dyn_model_exp=1.0,
                         drag_coeff=self.drag_coeff,
                         m_veh=self.m_veh,
                         v_start = 0.0)

        # calculate longitudinal acceleration profile
        vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
        ax_profile_opt = calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                        el_lengths=el_lengths_opt_interp,
                                        eq_length_output=False)

        # calculate laptime
        t_profile_cl = calc_t_profile(vx_profile=vx_profile_opt,
                                    ax_profile=ax_profile_opt,
                                    el_lengths=el_lengths_opt_interp)
        
        return s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp
    
    def Comparison(self,ds_ZO=None,ds_CMA=None, plot='N', output = 'N'):
        """

        .. description::
        Compares raceline kinematic profiles across four cases: ZO-optimized, CMA-ES-optimized, initial
        segment lengths, and the centerline. Prints lap times and optionally plots and returns all profiles.

        .. inputs::
        :param ds_ZO:       optimized segment lengths from the ZO solver. If None, computed internally.
        :type ds_ZO:        np.ndarray
        :param ds_CMA:      optimized segment lengths from the CMA-ES solver. If None, computed internally.
        :type ds_CMA:       np.ndarray
        :param plot:        'Y' to display comparison plots, 'N' to skip.
        :type plot:         str
        :param output:      controls which profiles are returned. 'Y' returns all four, 'ZO'/'CMA'/'initial'/'center'
                            returns the corresponding case only. 'N' returns nothing.
        :type output:       str

        .. outputs::
        :return profiles:   kinematic profiles (s_splines, vx, ax, kappa, t_profile, raceline) for the selected
                            output case(s). None if output is 'N'.
        :rtype profiles:    tuple or None
        """

        if ds_ZO is None:
            ds_ZO = self.CurveLenOpt(solver='ZO')
        if ds_CMA is None:
            ds_CMA = self.CurveLenOpt(solver='CMA')
        ds0 = self.lengths

        s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp = self.generate_kinProfs(ds=ds_ZO)
        s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1 = self.generate_kinProfs(ds=ds_CMA)
        s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2 = self.generate_kinProfs(ds=ds0)


        ##Calculate the profiles for centerline
        lengths1 = np.sqrt(np.sum(np.power(np.diff(self.center[:,0:2], axis=0), 2), axis=1))
        lengths1=np.append(lengths1, lengths1[0])
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.center[:, 0:2], self.center[0, 0:2])),el_lengths=lengths1)
        H, f, G , h = H_f(reftrack=self.center,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=self.sfty,
                                                 closed=True)
        alpha_m_0 = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]

        raceline_interp4, a_opt4, coeffs_x_opt4, coeffs_y_opt4, spline_inds_opt_interp4, t_vals_opt_interp4, s_points_opt_interp4,\
        spline_lengths_opt4, el_lengths_opt_interp4 = create_raceline(refline=self.center[:, :2],
                    normvectors=normvec_norm,
                    alpha=np.zeros_like(alpha_m_0),
                    stepsize_interp=self.si)

        psi_vel_opt4, kappa_opt4 =calc_head_curv_an(coeffs_x=coeffs_x_opt4,
                      coeffs_y=coeffs_y_opt4,
                      ind_spls=spline_inds_opt_interp4,
                      t_spls=t_vals_opt_interp4)

        s_splines4 = cumulative_distances(el_lengths_opt_interp4)
        vx_profile_opt4 = calc_vel_profile(ggv=self.ggv,
                         ax_max_machines=self.ax_max_machines,
                         v_max=self.vm,
                         kappa=kappa_opt4,
                         el_lengths=el_lengths_opt_interp4,
                         closed=True,
                         filt_window=self.fw,
                         dyn_model_exp=1.0,
                         drag_coeff=self.drag_coeff,
                         m_veh=self.m_veh,
                         v_start = 0.0)

        # calculate longitudinal acceleration profile
        vx_profile_opt_cl4 = np.append(vx_profile_opt4, vx_profile_opt4[0])
        ax_profile_opt4 = calc_ax_profile(vx_profile=vx_profile_opt_cl4,
                                        el_lengths=el_lengths_opt_interp4,
                                        eq_length_output=False)

        # calculate laptime
        t_profile_cl4 = calc_t_profile(vx_profile=vx_profile_opt4,
                                    ax_profile=ax_profile_opt4,
                                    el_lengths=el_lengths_opt_interp4)
        
        print("INFO: Estimated laptime for ZO: %.2fs" % t_profile_cl[-1])
        print("INFO: Estimated laptime for CMA-ES: %.2fs" % t_profile_cl1[-1])
        print("INFO: Estimated laptime for initial: %.2fs" % t_profile_cl2[-1])
        print("INFO: Estimated laptime for centerline: %.2fs" % t_profile_cl4[-1])

        if plot=='Y':
            plt.figure(figsize=(12, 6))
            plt.subplot(1,2,1)
            plt.plot(self.reftrack[:,0],self.reftrack[:,1],'b--',label='ref_line')
            plt.plot(self.bound1[:, 0], self.bound1[:, 1], 'k', label=' Track')
            plt.plot(self.bound2[:, 0], self.bound2[:, 1], 'k')
            plt.plot(raceline_interp[:,0], raceline_interp[:,1],'r.-' , label='Opt_ZO')
            plt.plot(raceline_interp2[:,0], raceline_interp2[:,1],'g-' , label='initial')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.subplot(1,2,2)
            plt.plot(self.reftrack[:,0],self.reftrack[:,1],'b--',label='ref_line')
            plt.plot(self.bound1[:, 0], self.bound1[:, 1], 'k', label=' Track')
            plt.plot(self.bound2[:, 0], self.bound2[:, 1], 'k')
            plt.plot(raceline_interp1[:,0], raceline_interp1[:,1],'r.-' , label='Opt_CMA')
            plt.plot(raceline_interp2[:,0], raceline_interp2[:,1],'g-' , label='initial')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(15, 15))
            plt.subplot(3,4,1)
            plt.plot(s_splines, vx_profile_opt,'b' , label='v_ZO')
            plt.plot(s_splines4, vx_profile_opt4,'r' , label='v_center')
            plt.plot(s_splines2, vx_profile_opt2,'g' , label='v_initial')
            plt.ylabel('v_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,2)
            plt.plot(s_splines, ax_profile_opt,'b' , label='a_ZO')
            plt.plot(s_splines4, ax_profile_opt4,'r' , label='a_center')
            plt.plot(s_splines2, ax_profile_opt2,'g' , label='a_initial')
            plt.ylabel('a_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,3)
            plt.plot(s_splines, kappa_opt,'b' , label='k_ZO')
            plt.plot(s_splines4, kappa_opt4,'r' , label='k_center')
            plt.plot(s_splines2, kappa_opt2,'g' , label='k_initial')
            plt.ylabel('curv_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,4)
            plt.plot(s_splines, t_profile_cl[:-1],'b' , label='t_ZO')
            plt.plot(s_splines4, t_profile_cl4[:-1],'r' , label='t_center')
            plt.plot(s_splines2, t_profile_cl2[:-1],'g' , label='t_initial')
            plt.ylabel('time')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,5)
            plt.plot(s_splines1, vx_profile_opt1,'b' , label='v_cma')
            plt.plot(s_splines4, vx_profile_opt4,'r' , label='v_center')
            plt.plot(s_splines2, vx_profile_opt2,'g' , label='v_initial')
            plt.ylabel('v_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,6)
            plt.plot(s_splines1, ax_profile_opt1,'b' , label='a_cma')
            plt.plot(s_splines4, ax_profile_opt4,'r' , label='a_center')
            plt.plot(s_splines2, ax_profile_opt2,'g' , label='a_initial')
            plt.ylabel('a_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,7)
            plt.plot(s_splines1, kappa_opt1,'b' , label='k_cma')
            plt.plot(s_splines4, kappa_opt4,'r' , label='k_center')
            plt.plot(s_splines2, kappa_opt2,'g' , label='k_initial')
            plt.ylabel('curv_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,8)
            plt.plot(s_splines1, t_profile_cl1[:-1],'b' , label='t_cma')
            plt.plot(s_splines4, t_profile_cl4[:-1],'r' , label='t_center')
            plt.plot(s_splines2, t_profile_cl2[:-1],'g' , label='t_initial')
            plt.ylabel('time')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,9)
            plt.plot(s_splines2, vx_profile_opt2,'b' , label='v_initial')
            plt.plot(s_splines4, vx_profile_opt4,'r' , label='v_center')
            plt.ylabel('v_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,10)
            plt.plot(s_splines2, ax_profile_opt2,'b' , label='a_initial')
            plt.plot(s_splines4, ax_profile_opt4,'r' , label='a_center')
            plt.ylabel('a_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,11)
            plt.plot(s_splines2, kappa_opt2,'b' , label='k_initial')
            plt.plot(s_splines4, kappa_opt4,'r' , label='k_center')
            plt.ylabel('curv_profile')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.subplot(3,4,12)
            plt.plot(s_splines2, t_profile_cl2[:-1],'b' , label='t_initial')
            plt.plot(s_splines4, t_profile_cl4[:-1],'r' , label='t_center')
            plt.ylabel('time')
            plt.xlabel('distance')
            plt.legend()
            plt.grid(True)
            plt.show()

        if output =='Y':
            return  s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp,\
                    s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1,\
                    s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2,\
                    s_splines4, vx_profile_opt4, ax_profile_opt4, kappa_opt4, t_profile_cl4, raceline_interp4
        if output == 'ZO':
            return  s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp
        if output == 'CMA':
            return  s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1
        if output =='initial':
            return s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2
        if output =='center':
            return s_splines4, vx_profile_opt4, ax_profile_opt4, kappa_opt4, t_profile_cl4, raceline_interp4
        


from scipy.integrate import quad
from scipy.optimize import fsolve

class Clothoid_raceline:
    def __init__(self, k_0, s, x0, y0, th0, nump=10,a_max=5.3,a_min=12,ay_max=12,c0=0.00002,c1=0.0015,v_max=22.88,v0=0,vf=22.88):
        self.k_0 = k_0
        self.s = s
        self.x0 = x0
        self.y0 = y0
        self.th0 = th0
        self.nump = nump
        self.a_max = a_max
        self.a_min = a_min 
        self.ay_max = ay_max 
        self.c0 = c0
        self.c1 = c1
        self.v_max = v_max
        self.v0=v0
        self.vf=vf


    def X_0(self,a, b, c):
        """

        .. description::
        Computes the normalized x-displacement integral of a clothoid segment by numerically integrating
        cos(a/2 * tau^2 + b * tau + c) over [0, 1].

        .. inputs::
        :param a:   curvature rate coefficient (k1 * s^2).
        :type a:    float
        :param b:   initial curvature coefficient (k0 * s).
        :type b:    float
        :param c:   initial heading angle theta_0 in radians.
        :type c:    float

        .. outputs::
        :return result: normalized x-displacement of the clothoid segment.
        :rtype result:  float
        """

        integrand = lambda tau: np.cos(a/2 * tau**2 + b * tau + c)
        result, _ = quad(integrand, 0, 1)
        return result

    def Y_0(self,a, b, c):
        """

        .. description::
        Computes the normalized y-displacement integral of a clothoid segment by numerically integrating
        sin(a/2 * tau^2 + b * tau + c) over [0, 1].

        .. inputs::
        :param a:   curvature rate coefficient (k1 * s^2).
        :type a:    float
        :param b:   initial curvature coefficient (k0 * s).
        :type b:    float
        :param c:   initial heading angle theta_0 in radians.
        :type c:    float

        .. outputs::
        :return result: normalized y-displacement of the clothoid segment.
        :rtype result:  float
        """

        integrand = lambda tau: np.sin(a/2 * tau**2 + b * tau + c)
        result, _ = quad(integrand, 0, 1)
        return result
    
    def compute_clothoid_path(self):
        """

        .. description::
        Computes the full x-y path, arc-length stations, and curvature profile of a piecewise clothoid curve
        defined by the curvature array k_0 and station array s. Each segment uses a linearly varying curvature
        (clothoid), and the path is reconstructed by numerically integrating X_0 and Y_0.

        .. outputs::
        :return x_full:         x-coordinates of all sampled path points in m.
        :rtype x_full:          np.ndarray
        :return y_full:         y-coordinates of all sampled path points in m.
        :rtype y_full:          np.ndarray
        :return s_full:         arc-length position of each sampled point along the path in m.
        :rtype s_full:          np.ndarray
        :return curvature_full: curvature value at each sampled point in rad/m.
        :rtype curvature_full:  np.ndarray
        """

        x = [self.x0]  
        y = [self.y0]
        
        theta_0 = self.th0
        
        x_full = []
        y_full = []
        s_full = []
        curvature_full = []
        
        for i in range(1, len(self.s)):

            L = self.s[i] - self.s[i - 1]
            
            if L == 0:
                s_full.append(np.nan)
                curvature_full.append(np.nan)
                continue  
            
            k0 = self.k_0[i - 1]
            k1 = (self.k_0[i] - self.k_0[i - 1]) / L
            
            s_values = np.linspace(0, L, self.nump)
            
            for s_val in s_values:

                X = s_val * self.X_0(k1 * s_val**2, k0 * s_val, theta_0)
                Y = s_val * self.Y_0(k1 * s_val**2, k0 * s_val, theta_0)
                
                x_new = x[-1] + X
                y_new = y[-1] + Y   

                x_full.append(x_new)
                y_full.append(y_new)
                
                s_full.append(self.s[i - 1] + s_val)
                curvature_full.append(k0 + k1 * s_val)
            
            theta_0 = theta_0 + k0 * L + (k1 * L**2) / 2

            x.append(x_full[-1])
            y.append(y_full[-1])
        
        x_full = np.array(x_full)
        y_full = np.array(y_full)
        s_full = np.array(s_full)
        curvature_full = np.array(curvature_full)
        
        return x_full, y_full, s_full, curvature_full
    
    
                       



def OSP(reftrack: np.ndarray,
                      normvectors: np.ndarray,
                      w_veh: float,
                      print_debug: bool = False) -> np.ndarray:
    """

    .. description::
    Builds the quadratic programming matrices H, f, G, h for the Optimal Shortest Path (OSP) problem.
    The objective minimizes the total squared lateral deviation between consecutive raceline points,
    subject to track-width bounds on the lateral shift alpha.

    .. inputs::
    :param reftrack:    reference track array [x, y, w_tr_right, w_tr_left] in m (unclosed).
    :type reftrack:     np.ndarray
    :param normvectors: normalized normal vectors for every track point [x_component, y_component].
    :type normvectors:  np.ndarray
    :param w_veh:       vehicle width in m used to reduce the allowed lateral deviation.
    :type w_veh:        float
    :param print_debug: flag to print debug information (currently unused).
    :type print_debug:  bool

    .. outputs::
    :return H:          symmetric quadratic cost matrix for the QP.
    :rtype H:           np.ndarray
    :return f:          linear cost vector for the QP.
    :rtype f:           np.ndarray
    :return G:          inequality constraint matrix (stacked identity matrices).
    :rtype G:           np.ndarray
    :return h:          inequality constraint vector encoding the allowed lateral deviations.
    :rtype h:           np.ndarray
    """

    no_points = reftrack.shape[0]

    # check inputs
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")

    # ------------------------------------------------------------------------------------------------------------------
    # SET UP FINAL MATRICES FOR SOLVER ---------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    H = np.zeros((no_points, no_points))
    f = np.zeros(no_points)

    i_idx = np.arange(no_points)
    next_idx = (i_idx + 1) % no_points
    prev_idx = (i_idx - 1) % no_points

    np.fill_diagonal(H, 4 * (normvectors[:, 0]**2 + normvectors[:, 1]**2))
    off_diag = -2 * (normvectors[i_idx, 0] * normvectors[next_idx, 0]
                     + normvectors[i_idx, 1] * normvectors[next_idx, 1])
    H[i_idx, next_idx] = off_diag
    H[next_idx, i_idx] = off_diag

    f = (2 * normvectors[:, 0] * (2 * reftrack[:, 0] - reftrack[prev_idx, 0] - reftrack[next_idx, 0])
         + 2 * normvectors[:, 1] * (2 * reftrack[:, 1] - reftrack[prev_idx, 1] - reftrack[next_idx, 1]))



    # calculate allowed deviation from refline
    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2

    # set minimum deviation to zero
    dev_max_right[dev_max_right < 0.001] = 0.001
    dev_max_left[dev_max_left < 0.001] = 0.001

    # consider value boundaries (-dev_max <= alpha <= dev_max)
    G = np.vstack((np.eye(no_points), -np.eye(no_points)))
    h = np.ones(2 * no_points) * np.append(dev_max_right, dev_max_left)

    return H,f,G,h

def ShortestPath(reftrack: np.ndarray,
        w_veh: float,
        stepsize: float,
        plot: bool,
        ggv_import_path="maps/ggv.csv",ax_max_machines_import_path="maps/ax_max_machines.csv") -> np.ndarray:
    """

    .. description::
    Computes the shortest feasible path through a closed reference track by solving a QP via the OSP formulation.
    After obtaining the optimal lateral shift, the function interpolates the raceline, computes kinematic profiles
    (velocity, acceleration, lap time), and optionally plots the results.

    .. inputs::
    :param reftrack:                        reference track array [x, y, w_tr_right, w_tr_left] in m (unclosed).
    :type reftrack:                         np.ndarray
    :param w_veh:                           vehicle width in m used to reduce the allowed lateral deviation.
    :type w_veh:                            float
    :param stepsize:                        interpolation step size for the raceline in m (currently unused internally).
    :type stepsize:                         float
    :param plot:                            if True, displays plots of the raceline and kinematic profiles.
    :type plot:                             bool
    :param ggv_import_path:                 file path to the ggv diagram CSV.
    :type ggv_import_path:                  str
    :param ax_max_machines_import_path:     file path to the maximum machine acceleration CSV.
    :type ax_max_machines_import_path:      str

    .. outputs::
    :return raceline_interp:    interpolated shortest-path raceline coordinates [x, y].
    :rtype raceline_interp:     np.ndarray
    :return alpha_shpath:       optimal lateral shift in m for every reference track point.
    :rtype alpha_shpath:        np.ndarray
    :return s_splines:          cumulative arc-length stations along the raceline in m.
    :rtype s_splines:           np.ndarray
    :return vx_profile_opt_cl:  closed velocity profile (last point appended) in m/s.
    :rtype vx_profile_opt_cl:   np.ndarray
    :return ax_profile_opt:     longitudinal acceleration profile in m/s2.
    :rtype ax_profile_opt:      np.ndarray
    :return kappa_opt:          curvature profile of the raceline in rad/m.
    :rtype kappa_opt:           np.ndarray
    :return t_profile_cl:       cumulative lap time profile in seconds.
    :rtype t_profile_cl:        np.ndarray
    """

    lengths = np.sqrt(np.sum(np.power(np.diff(reftrack[:,0:2], axis=0), 2), axis=1))
    lengths=np.append(lengths, lengths[0])
    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])),el_lengths=lengths)
    H, f, G , h = OSP(reftrack=reftrack,
                    normvectors=normvec_norm,
                    w_veh=w_veh,)
    alpha_shpath = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]
    # sampled_pointss=np.zeros_like(reftrack[:,:2])
    # for i in range(len(alpha_shpath)):
    #     sampled_pointss[i,:] = reftrack[i,:2]+alpha_shpath[i]*normvec_norm[i,:]

    
    

    raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
    spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=reftrack[:, :2],
                normvectors=normvec_norm,
                alpha=alpha_shpath,
                stepsize_interp=2.0,)
    

    psi_vel_opt, kappa_opt =calc_head_curv_an(coeffs_x=coeffs_x_opt,
                    coeffs_y=coeffs_y_opt,
                    ind_spls=spline_inds_opt_interp,
                    t_spls=t_vals_opt_interp)

    # s_splines = cumulative_distances(el_lengths_opt_interp)

    vm = 70
    fw = 3

    ggv,ax_max_machines =import_veh_dyn_info(ggv_import_path=ggv_import_path,ax_max_machines_import_path=ax_max_machines_import_path)
    vx_profile_opt = calc_vel_profile(ggv=ggv,
                            ax_max_machines=ax_max_machines,
                            v_max=vm,
                            kappa=kappa_opt,
                            el_lengths=el_lengths_opt_interp,
                            closed=True,
                            filt_window=fw,
                            dyn_model_exp=1.0,
                            drag_coeff=0.75,
                            m_veh=1000.0,
                            v_start = 0.0)

    # calculate longitudinal acceleration profile
    s_splines = cumulative_distances(el_lengths_opt_interp)
    vx_profile_opt_cl = np.append(vx_profile_opt, vx_profile_opt[0])
    ax_profile_opt = calc_ax_profile(vx_profile=vx_profile_opt_cl,
                                    el_lengths=el_lengths_opt_interp,
                                    eq_length_output=False)

    # calculate laptime
    t_profile_cl = calc_t_profile(vx_profile=vx_profile_opt,
                                ax_profile=ax_profile_opt,
                                el_lengths=el_lengths_opt_interp)
    
    if plot == True:
        bound1 = reftrack[:, 0:2] - normvec_norm * np.expand_dims(reftrack[:, 2], axis=1)
        bound2 = reftrack[:, 0:2] + normvec_norm * np.expand_dims(reftrack[:, 3], axis=1)

        plt.figure(figsize=(10, 10))
        plt.plot(reftrack[:,0],reftrack[:,1],'b--',label='ref_line')
        plt.plot(bound1[:, 0], bound1[:, 1], 'k', label=' Track')
        plt.plot(bound2[:, 0], bound2[:, 1], 'k')
        plt.plot(raceline_interp[:,0], raceline_interp[:,1],'r.-' , label='Shortest Path')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()


        plt.figure(figsize=(15, 15))
        plt.subplot(2,2,1)
        plt.plot(s_splines, vx_profile_opt,'b' , label='v')
        plt.ylabel('v_profile')
        plt.xlabel('distance')
        plt.legend()
        plt.grid(True)
        plt.subplot(2,2,2)
        plt.plot(s_splines, ax_profile_opt,'b' , label='a')
        plt.ylabel('a_profile')
        plt.xlabel('distance')
        plt.legend()
        plt.grid(True)
        plt.subplot(2,2,3)
        plt.plot(s_splines, kappa_opt,'b' , label='curvature')
        plt.ylabel('curv_profile')
        plt.xlabel('distance')
        plt.legend()
        plt.grid(True)
        plt.subplot(2,2,4)
        plt.plot(s_splines, t_profile_cl[:-1],'b' , label='time')
        plt.ylabel('time')
        plt.xlabel('distance')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(t_profile_cl[-1])

    
    return raceline_interp,alpha_shpath,s_splines,vx_profile_opt_cl,ax_profile_opt,kappa_opt,t_profile_cl