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

    Please refer to the paper for further information:
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
    def __init__(self, func, x0, num_iterations, mu=0.05, h=0.001, t=1, grad_type='noth', constraint_type='notc'):
        """
        ZORM optimizer.

        Parameters:
        - func: The objective function to minimize.
        - mu: Smoothing parameter for gradient approximation.
        - h: Step size.
        - x0: initial guess
        - num_iterations: number of iterations
        - t: number of sampled directions for gradient estimation (averaged).
        - grad_type: 'noth' for forward gradient, 'h' for symmetric (central difference).
        - constraint_type: 'notc' for unconstrained, 'c' for constrained optimization.
        """
        self.func = func
        self.mu = mu
        self.h = h
        self.x0=x0
        self.T = num_iterations
        self.t = t
        self.grad_type = grad_type
        self.constraint_type = constraint_type

    def _sample_gaussian(self, d):
        return np.random.normal(0, 1, size=d)

    def _feasible_projection(self, x, min, max):
        y = np.clip(x,min,max)
        return y

    def _grad(self, x):
        grad_sum = np.zeros(len(x))
        fx = self.func(x)
        for _ in range(self.t):
            u = self._sample_gaussian(len(x))
            perturbation = self.mu * u
            if self.grad_type == 'noth':
                g = (self.func(x + perturbation) - fx) / self.mu
            elif self.grad_type == 'h':
                g = (self.func(x + perturbation) - self.func(x - perturbation)) / (2 * self.mu)
            else:
                raise ValueError("Invalid gradient type. Use 'noth' or 'h'.")
            grad_sum += g * u
        return grad_sum / self.t

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
    def __init__(self, reftrack, center, mu=0.05, h=0.001, kapb=0.7, sfty=1, t=1, si=0.8,
                 vm=22.88, m_veh=3, drag_coeff=0.0045, MC=1, min_s=None, max_s=None, sigma=0.001,
                 iterations_ZO=300, iterations_CMA=30, popsize=16,
                 ggv_import_path="maps/ggv.csv", ax_max_machines_import_path="maps/ax_max_machines.csv",
                 fw=3, refine_every=None, refine_subsample=1):
        """
        Min Curv and Time optimizer.

        Parameters:
        - reftrack: reference track array containing the reference line and track widths [x, y, w_tr_right, w_tr_left].
        - center: array containing the center line and track widths [x, y, w_tr_right, w_tr_left].
        - mu: Smoothing parameter for gradient approximation for ZO solver
        - h: Step size for ZO solver
        - iterations_ZO: number of iterations for ZO solver
        - sfty: half of the vehicle width
        - t = number of sampled directions for gradient estimation (averaged) for ZO solver
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
        - refine_every: int or None.  When set to a positive integer q, CurveLenOpt rebuilds the
                        reference track from the current optimised raceline every q ZO iterations.
                        None (default) disables refinement and preserves the original behaviour.
                        Call reset() to restore self.reftrack to the original track at any time.
        - refine_subsample: int.  Sub-sampling stride applied to the new raceline before it
                        becomes the refined reference track.  1 = no sub-sampling (default).
        """

        if min_s is None and max_s is None:
            el_lengths = np.sqrt(np.sum(np.power(np.diff(np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])), axis=0), 2), axis=1))
            min_s = np.min(el_lengths) *1
            max_s = np.max(el_lengths) *1.7

        self.reftrack = reftrack
        self.mu = mu
        self.h = h
        self.t = t
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
        # Pre-built p_ggv cache keyed by kappa array size (size is approx constant across f_t calls)
        self._p_ggv_cache = {}

        # Refinement settings
        self.refine_every = refine_every
        self.refine_subsample = int(refine_subsample)
        # Immutable copy of the original track; used by reset() and _build_refined_reftrack()
        self._base_reftrack = reftrack.copy()

        # Populated by CurveLenOpt — mirrors alpha_opt / history in Blackbox_raceline
        self.ds_opt     = None
        self.ds_history = None

        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((center[:, 0:2], center[0, 0:2])),use_dist_scaling=True)
        
        self.bound1 = center[:, 0:2] - normvec_norm * np.expand_dims(center[:, 2], axis=1)
        self.bound2 = center[:, 0:2] + normvec_norm * np.expand_dims(center[:, 3], axis=1)


    # ------------------------------------------------------------------
    # Public helper
    # ------------------------------------------------------------------

    def reset(self):
        """
        .. description::
        Restore self.reftrack to the original reference track supplied at construction
        time, reverting any in-place refinements made during CurveLenOpt.
        """
        self._update_reftrack(self._base_reftrack.copy())

    # ------------------------------------------------------------------
    # Private helpers for path refinement
    # ------------------------------------------------------------------

    def _update_reftrack(self, reftrack_new: np.ndarray) -> None:
        """
        .. description::
        Replace self.reftrack with reftrack_new and recompute all derived quantities:
        segment lengths, optimisation bounds (min_s / max_s), and the p_ggv cache.

        .. inputs::
        :param reftrack_new:    new reference track [x, y, w_right, w_left].
        :type reftrack_new:     np.ndarray
        """
        self.reftrack = reftrack_new
        lengths = np.sqrt(np.sum(
            np.power(np.diff(reftrack_new[:, 0:2], axis=0), 2), axis=1))
        lengths = np.append(lengths, lengths[0])
        self.lengths = lengths
        self.min_s = float(np.min(lengths))
        self.max_s = float(np.max(lengths)) * 1.7
        self._p_ggv_cache = {}   # kappa size may have changed — invalidate

    def _build_refined_reftrack(self,
                                alpha_m: np.ndarray,
                                normvec_norm: np.ndarray) -> np.ndarray:
        """
        .. description::
        Build a geometrically correct refined reference track from the QP solution on
        the CURRENT self.reftrack.

        The new centreline is the set of control points shifted laterally by alpha_m
        along their unit normal vectors.  The track half-widths are adjusted so that
        the two physical boundary lines stay EXACTLY in place:

            new_w_col2[i] = w_col2[i] + alpha_m[i]
            new_w_col3[i] = w_col3[i] − alpha_m[i]

        Derivation (QP constraint boundaries preserved):
            The QP constraint is  −(w_col3 − sfty) ≤ α ≤ (w_col2 − sfty),
            so positive α moves the car toward  B_A = center + n · w_col2
            and negative α moves it toward      B_B = center − n · w_col3.

            After the shift, the remaining distances to each boundary are:
                room toward B_A: w_col2 − alpha   (moved closer → less room)
                room toward B_B: w_col3 + alpha   (moved away  → more room)

            From the new center the next-pass optimizer can reach at most:
                new_center + (new_w_col2 − sfty)·n
                    = (center + alpha·n) + (w_col2 − alpha − sfty)·n
                    = center + (w_col2 − sfty)·n                          ✓
                new_center − (new_w_col3 − sfty)·n
                    = (center + alpha·n) − (w_col3 + alpha − sfty)·n
                    = center − (w_col3 − sfty)·n                          ✓
            i.e. the optimizer is confined to exactly the same physical region
            as in the original pass, regardless of how many refinement steps
            have been taken.

        A minimum clearance of self.sfty is enforced on both sides so widths can
        never fall below the vehicle half-width.  The optional sub-sampling stride
        self.refine_subsample is applied before returning.

        .. inputs::
        :param alpha_m:         lateral shifts in m at every control point, shape (N,).
        :type alpha_m:          np.ndarray
        :param normvec_norm:    unit normal vectors at every control point, shape (N, 2).
        :type normvec_norm:     np.ndarray

        .. outputs::
        :return reftrack_new:   refined reference track [x, y, w_col2, w_col3].
        :rtype reftrack_new:    np.ndarray
        """
        new_xy  = self.reftrack[:, :2] + alpha_m[:, np.newaxis] * normvec_norm
        new_w2  = self.reftrack[:, 2] - alpha_m
        new_w3  = self.reftrack[:, 3] + alpha_m

        reftrack_new = np.column_stack([new_xy, new_w2, new_w3])

        # step = int(self.refine_subsample)
        # if step > 1:
        #     nn  = reftrack_new.shape[0]
        #     idx = np.arange(0, nn, step)
        #     idx = np.unique(np.concatenate([idx, [nn - 1]]))
        #     reftrack_new = reftrack_new[idx]

        return reftrack_new

    def _solve_alpha(self, ds: np.ndarray):
        """
        .. description::
        Solve the minimum-curvature QP on self.reftrack for the given segment lengths
        and return the lateral shift vector alpha_m together with the normal vectors.
        Both arrays are at CONTROL-POINT resolution (one entry per track point), which
        is what _build_refined_reftrack needs to compute physically correct widths.

        .. inputs::
        :param ds:  spline segment lengths for the current reference track, shape (N,).
        :type ds:   np.ndarray

        .. outputs::
        :return alpha_m:        lateral shift at every control point in m, shape (N,).
        :rtype alpha_m:         np.ndarray
        :return normvec_norm:   unit normal vectors at every control point, shape (N, 2).
        :rtype normvec_norm:    np.ndarray
        """
        coeffs_x, coeffs_y, M_mat, normvec_norm = calc_splines(
            path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),
            el_lengths=ds)
        H, f, G, h = H_f(
            reftrack=self.reftrack,
            normvectors=normvec_norm,
            A=M_mat,
            kappa_bound=self.kapb,
            w_veh=2*self.sfty,
            closed=True)
        alpha_m = quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]
        return alpha_m, normvec_norm

    def _extract_raceline(self, ds: np.ndarray) -> np.ndarray:
        """
        .. description::
        Convenience method: solve the QP and return the DENSE interpolated raceline
        (one point every self.si metres).  Useful for external inspection or plotting.
        Refinement internally uses _solve_alpha + _build_refined_reftrack instead.

        .. inputs::
        :param ds:  spline segment lengths for the current reference track, shape (N,).
        :type ds:   np.ndarray

        .. outputs::
        :return raceline_interp:    interpolated raceline [x, y], shape (M, 2).
        :rtype raceline_interp:     np.ndarray
        """
        alpha_m, normvec_norm = self._solve_alpha(ds)
        raceline_interp, *_ = create_raceline(
            refline=self.reftrack[:, :2],
            normvectors=normvec_norm,
            alpha=alpha_m,
            stepsize_interp=self.si)
        return raceline_interp

    # ------------------------------------------------------------------

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
                                                    w_veh=2*sfty,
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
        n = kappa_opt.size
        if n not in self._p_ggv_cache:
            self._p_ggv_cache[n] = np.repeat(np.expand_dims(self.ggv, axis=0), n, axis=0)
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
                                v_start = 0.0,
                                p_ggv=self._p_ggv_cache[n])

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
    
    def CurveLenOpt(self, solver='ZO', refine_every=None):
        """
        Optimise spline segment lengths to minimise lap time, tracking and
        returning the **best** iterate seen rather than the last one.


        Parameters
        ----------
        solver : str
            ``'ZO'`` for projected zeroth-order gradient descent, ``'CMA'``
            for CMA-ES.
        refine_every : int or None
            Per-call override for ``self.refine_every``.  Positive integer
            ``q`` enables in-loop path refinement every ``q`` iterations /
            generations; ``0`` or ``None`` disables it.

        Returns
        -------
        best_ds : np.ndarray
            Segment lengths that achieved the lowest ``f_t`` value observed.
        history : np.ndarray
            ``history[0]`` = cost at the initial ``ds_0``.
            ``history[k]`` = best-so-far cost after step / generation ``k``
            (accumulated across all MC runs; concatenated across refinement
            blocks with the post-refinement initial cost inserted between
            blocks).
        """

        q = refine_every if refine_every is not None else self.refine_every
        use_refinement = (q is not None and int(q) > 0)

        # ====================================================================
        # ZO solver
        # ====================================================================
        if solver == 'ZO':
            ds_0 = self.lengths.copy()

            if not use_refinement:
                # ----------------------------------------------------------------
                # Standard mode: MC independent runs, best tracked globally
                # ----------------------------------------------------------------
                f_start   = self.f_t(ds_0)
                best_ds   = ds_0.copy()
                best_cost = f_start
                history   = np.full(self.MC * self.iterations_ZO + 1, np.nan)
                history[0] = f_start

                hist_idx = 0
                for _ in range(self.MC):
                    ds     = ds_0.copy()
                    f_curr = f_start
                    for k in range(self.iterations_ZO):
                        # gradient estimate: t forward-difference directions
                        grad_sum = np.zeros_like(ds)
                        for _ in range(self.t):
                            u      = np.random.randn(len(ds))
                            ds_fwd = np.clip(ds + self.mu * u, self.min_s, self.max_s)
                            f_fwd  = self.f_t(ds_fwd)
                            grad_sum += (f_fwd - f_curr) / self.mu * u
                        # projected gradient step; evaluate new iterate
                        ds     = np.clip(ds - self.h * (grad_sum / self.t),
                                         self.min_s, self.max_s)
                        f_curr = self.f_t(ds)
                        hist_idx += 1
                        if f_curr < best_cost:
                            best_cost = f_curr
                            best_ds   = ds.copy()
                        history[hist_idx] = best_cost

                self.ds_opt     = best_ds
                self.ds_history = history
                return best_ds, history

            else:
                # ----------------------------------------------------------------
                # Refinement-aware loop
                # ----------------------------------------------------------------
                if self.MC > 1:
                    print("WARNING [CurveLenOpt]: MC > 1 is not supported with path "
                          "refinement; running a single pass (MC=1).")

                q = int(q)
                if q > self.iterations_ZO:
                    print(f"WARNING [CurveLenOpt]: refine_every ({q}) > iterations_ZO "
                          f"({self.iterations_ZO}). No refinement will occur.")

                # Partition total iterations into blocks of q (last block may be shorter).
                # Refinement happens BETWEEN blocks, NOT after the final block.
                n_full     = self.iterations_ZO // q
                rem        = self.iterations_ZO % q
                all_blocks = [q] * n_full
                if rem > 0:
                    all_blocks.append(rem)
                n_blocks      = len(all_blocks)
                n_refinements = n_blocks - 1

                ds_current = self.lengths.copy()
                f_curr     = self.f_t(ds_current)
                best_ds    = ds_current.copy()
                best_cost  = f_curr
                all_hist   = [np.array([f_curr])]  # history[0] = initial cost

                for blk_idx, blk_iters in enumerate(all_blocks):
                    blk_hist = []
                    for k in range(blk_iters):
                        grad_sum = np.zeros_like(ds_current)
                        for _ in range(self.t):
                            u         = np.random.randn(len(ds_current))
                            ds_fwd    = np.clip(ds_current + self.mu * u,
                                                self.min_s, self.max_s)
                            f_fwd     = self.f_t(ds_fwd)
                            grad_sum += (f_fwd - f_curr) / self.mu * u
                        ds_current = np.clip(ds_current - self.h * (grad_sum / self.t),
                                             self.min_s, self.max_s)
                        f_curr = self.f_t(ds_current)
                        if f_curr < best_cost:
                            best_cost = f_curr
                            best_ds   = ds_current.copy()
                        blk_hist.append(best_cost)
                    all_hist.append(np.array(blk_hist))

                    is_last = (blk_idx == n_blocks - 1)
                    if not is_last:
                        # Use best_ds (not last iterate) to build the refined track
                        alpha_m, normvec = self._solve_alpha(best_ds)
                        reftrack_refined = self._build_refined_reftrack(alpha_m, normvec)
                        self._update_reftrack(reftrack_refined)
                        ds_current = self.lengths.copy()
                        f_curr     = self.f_t(ds_current)
                        # Reset best for the new track
                        best_ds    = ds_current.copy()
                        best_cost  = f_curr
                        all_hist.append(np.array([f_curr]))  # cost at start of new block
                        print(f"  [path refine {blk_idx + 1}/{n_refinements}] "
                              f"{reftrack_refined.shape[0]} ctrl pts  |  "
                              f"laptime ≈ {f_curr:.3f} s")

                history = np.concatenate(all_hist)
                self.ds_opt     = best_ds
                self.ds_history = history
                return best_ds, history

        # ====================================================================
        # CMA-ES solver
        # ====================================================================
        if solver == 'CMA':
            if not use_refinement:
                # ----------------------------------------------------------------
                # Standard mode: MC independent runs, best tracked globally
                # ----------------------------------------------------------------
                mean    = self.lengths.copy()
                f_start = self.f_t(mean)
                best_ds   = mean.copy()
                best_cost = f_start
                history   = np.full(self.MC * self.iterations_CMA + 1, np.nan)
                history[0] = f_start

                hist_idx = 0
                for _ in range(self.MC):
                    _best = {'ds': best_ds.copy(), 'cost': best_cost}

                    def _tracked(ds):
                        c = self.f_t(ds)
                        if c < _best['cost']:
                            _best['cost'] = c
                            _best['ds']   = ds.copy()
                        return c

                    cma = ConstrainedCMAES_t(
                        _tracked, mean.copy(), self.sigma, self.popsize,
                        bounds1=np.ones_like(mean) * self.max_s,
                        bounds2=np.ones_like(mean) * self.min_s)
                    cma.iterations = 0

                    for gen in range(self.iterations_CMA):
                        samples = cma.sample_population()
                        fitness = np.array([cma.objective_function(s) for s in samples])
                        cma.update(samples, fitness)
                        cma.iterations += 1
                        hist_idx += 1
                        history[hist_idx] = _best['cost']

                    if _best['cost'] < best_cost:
                        best_cost = _best['cost']
                        best_ds   = _best['ds'].copy()

                self.ds_opt     = best_ds
                self.ds_history = history
                return best_ds, history

            else:
                # ----------------------------------------------------------------
                # Refinement-aware CMA loop
                # ----------------------------------------------------------------
                if self.MC > 1:
                    print("WARNING [CurveLenOpt]: MC > 1 is not supported with path "
                          "refinement; running a single pass (MC=1).")

                q = int(q)
                if q > self.iterations_CMA:
                    print(f"WARNING [CurveLenOpt]: refine_every ({q}) > iterations_CMA "
                          f"({self.iterations_CMA}). No refinement will occur.")

                # Partition total CMA iterations into blocks of q.
                # Refinement happens BETWEEN blocks, not after the final block.
                n_full     = self.iterations_CMA // q
                rem        = self.iterations_CMA % q
                all_blocks = [q] * n_full
                if rem > 0:
                    all_blocks.append(rem)
                n_blocks      = len(all_blocks)
                n_refinements = n_blocks - 1

                ds_current = self.lengths.copy()
                f_curr     = self.f_t(ds_current)
                best_ds    = ds_current.copy()
                best_cost  = f_curr
                all_hist   = [np.array([f_curr])]  # history[0] = initial cost

                for blk_idx, blk_iters in enumerate(all_blocks):
                    _best = {'ds': best_ds.copy(), 'cost': best_cost}

                    def _tracked(ds):
                        c = self.f_t(ds)
                        if c < _best['cost']:
                            _best['cost'] = c
                            _best['ds']   = ds.copy()
                        return c

                    cma = ConstrainedCMAES_t(
                        _tracked, ds_current.copy(), self.sigma, self.popsize,
                        bounds1=np.ones_like(ds_current) * self.max_s,
                        bounds2=np.ones_like(ds_current) * self.min_s)
                    cma.iterations = 0

                    blk_hist = []
                    for gen in range(blk_iters):
                        samples = cma.sample_population()
                        fitness = np.array([cma.objective_function(s) for s in samples])
                        cma.update(samples, fitness)
                        cma.iterations += 1
                        blk_hist.append(_best['cost'])
                    all_hist.append(np.array(blk_hist))

                    if _best['cost'] < best_cost:
                        best_cost = _best['cost']
                        best_ds   = _best['ds'].copy()
                    ds_current = cma.mean.copy()

                    is_last = (blk_idx == n_blocks - 1)
                    if not is_last:
                        # Use best_ds (not CMA mean) to build the refined track
                        alpha_m, normvec = self._solve_alpha(best_ds)
                        reftrack_refined = self._build_refined_reftrack(alpha_m, normvec)
                        self._update_reftrack(reftrack_refined)
                        ds_current = self.lengths.copy()
                        f_curr     = self.f_t(ds_current)
                        # Reset best for the new track
                        best_ds    = ds_current.copy()
                        best_cost  = f_curr
                        all_hist.append(np.array([f_curr]))  # cost at start of new block
                        print(f"  [path refine {blk_idx + 1}/{n_refinements}] "
                              f"{reftrack_refined.shape[0]} ctrl pts  |  "
                              f"laptime ≈ {f_curr:.3f} s")

                history = np.concatenate(all_hist)
                self.ds_opt     = best_ds
                self.ds_history = history
                return best_ds, history

    def generate_raceline(self, ds=None, solver='ZO', refine_every=None):
        """

        .. description::
        Generates the optimized raceline geometry for a given set of spline segment lengths.
        If no segment lengths are provided, runs CurveLenOpt first to obtain them.

        When refine_every is set, CurveLenOpt is called with that cadence and self.reftrack
        may be updated as a side-effect (see CurveLenOpt docs).  The returned raceline is
        computed on whatever self.reftrack is current after that call.

        .. inputs::
        :param ds:              array of spline segment lengths. If None, computed via
                                CurveLenOpt(solver, refine_every).
        :type ds:               np.ndarray
        :param solver:          solver to use if ds is None. 'ZO' or 'CMA'.
        :type solver:           str
        :param refine_every:    per-call refinement cadence passed to CurveLenOpt when
                                ds is None.  None uses self.refine_every (default: no refinement).
        :type refine_every:     int or None

        .. outputs::
        :return raceline_interp:    interpolated raceline coordinates [x, y].
        :rtype raceline_interp:     np.ndarray
        :return ds:                 spline segment lengths used (either provided or optimized).
        :rtype ds:                  np.ndarray
        """

        if ds is None:
            ds, _ = self.CurveLenOpt(solver=solver, refine_every=refine_every)

        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),el_lengths=ds)
        H, f, G , h = H_f(reftrack=self.reftrack,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=2*self.sfty,
                                                 closed=True)


        alpha_m_s = quadprog.solve_qp(H, -f, -G.T,-h,0)[0]
        raceline_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp, s_points_opt_interp,\
        spline_lengths_opt, el_lengths_opt_interp = create_raceline(refline=self.reftrack[:, :2],
                    normvectors=normvec_norm,
                    alpha=alpha_m_s,
                    stepsize_interp=self.si)
        return raceline_interp,ds
    
    def generate_kinProfs(self, ds=None, solver='ZO', refine_every=None):
        """

        .. description::
        Generates the full kinematic profiles (velocity, acceleration, curvature, time,
        raceline) for a given set of spline segment lengths.  If no segment lengths are
        provided, runs CurveLenOpt first.

        When refine_every is set, CurveLenOpt is called with that cadence and self.reftrack
        may be updated as a side-effect (see CurveLenOpt docs).  All profiles are then
        computed on whatever self.reftrack is current after that call.

        .. inputs::
        :param ds:              array of spline segment lengths. If None, computed via
                                CurveLenOpt(solver, refine_every).
        :type ds:               np.ndarray
        :param solver:          solver to use if ds is None. 'ZO' or 'CMA'.
        :type solver:           str
        :param refine_every:    per-call refinement cadence passed to CurveLenOpt when
                                ds is None.  None uses self.refine_every (default: no refinement).
        :type refine_every:     int or None

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
            ds, _ = self.CurveLenOpt(solver=solver, refine_every=refine_every)
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.reftrack[:, 0:2], self.reftrack[0, 0:2])),el_lengths=ds)
        H, f, G , h = H_f(reftrack=self.reftrack,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=2*self.sfty,
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
    
    def Comparison(self, ds_ZO=None, ds_CMA=None, plot='N', output='N', refine_every=None):
        """

        .. description::
        Compares raceline kinematic profiles across four cases: ZO-optimized, CMA-ES-optimized,
        initial segment lengths, and the centerline.  Prints lap times and optionally plots and
        returns all profiles.

        Each solver is always run from the original base reference track so the comparison is
        fair regardless of whether path refinement is used for ZO.  self.reftrack is reset to
        the base track when Comparison returns.

        .. inputs::
        :param ds_ZO:           optimized segment lengths from the ZO solver. If None, computed
                                internally via CurveLenOpt('ZO', refine_every).
        :type ds_ZO:            np.ndarray
        :param ds_CMA:          optimized segment lengths from the CMA-ES solver. If None,
                                computed internally via CurveLenOpt('CMA').
        :type ds_CMA:           np.ndarray
        :param plot:            'Y' to display comparison plots, 'N' to skip.
        :type plot:             str
        :param output:          controls which profiles are returned. 'Y' returns all four,
                                'ZO'/'CMA'/'initial'/'center' returns the corresponding case
                                only. 'N' returns nothing.
        :type output:           str
        :param refine_every:    path-refinement cadence forwarded to CurveLenOpt for the ZO
                                solver only.  None (default) uses self.refine_every.
        :type refine_every:     int or None

        .. outputs::
        :return profiles:   kinematic profiles (s_splines, vx, ax, kappa, t_profile, raceline)
                            for the selected output case(s). None if output is 'N'.
        :rtype profiles:    tuple or None
        """

        # ---- ZO (run from original base track; refinement optional) ----
        self._update_reftrack(self._base_reftrack.copy())
        if ds_ZO is None:
            ds_ZO, _ = self.CurveLenOpt(solver='ZO', refine_every=refine_every)
        s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp = \
            self.generate_kinProfs(ds=ds_ZO)

        # ---- CMA (always from original base track, no refinement) ----
        self._update_reftrack(self._base_reftrack.copy())
        if ds_CMA is None:
            ds_CMA, _ = self.CurveLenOpt(solver='CMA')
        s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1 = \
            self.generate_kinProfs(ds=ds_CMA)

        # ---- initial segment lengths (base track, no optimisation) ----
        self._update_reftrack(self._base_reftrack.copy())
        ds0 = self.lengths.copy()
        s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2 = \
            self.generate_kinProfs(ds=ds0)


        ##Calculate the profiles for centerline
        lengths1 = np.sqrt(np.sum(np.power(np.diff(self.center[:,0:2], axis=0), 2), axis=1))
        lengths1=np.append(lengths1, lengths1[0])
        coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((self.center[:, 0:2], self.center[0, 0:2])),el_lengths=lengths1)
        H, f, G , h = H_f(reftrack=self.center,
                                                 normvectors=normvec_norm,
                                                 A=M,
                                                 kappa_bound=self.kapb,
                                                 w_veh=2*self.sfty,
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

        # Always restore original track when Comparison exits
        self._update_reftrack(self._base_reftrack.copy())

        if output == 'Y':
            return  s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp,\
                    s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1,\
                    s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2,\
                    s_splines4, vx_profile_opt4, ax_profile_opt4, kappa_opt4, t_profile_cl4, raceline_interp4
        if output == 'ZO':
            return  s_splines, vx_profile_opt, ax_profile_opt, kappa_opt, t_profile_cl, raceline_interp
        if output == 'CMA':
            return  s_splines1, vx_profile_opt1, ax_profile_opt1, kappa_opt1, t_profile_cl1, raceline_interp1
        if output == 'initial':
            return s_splines2, vx_profile_opt2, ax_profile_opt2, kappa_opt2, t_profile_cl2, raceline_interp2
        if output == 'center':
            return s_splines4, vx_profile_opt4, ax_profile_opt4, kappa_opt4, t_profile_cl4, raceline_interp4


# ===========================================================================
# Blackbox_raceline
# ===========================================================================

class Blackbox_raceline:
    """
    Direct zeroth-order (ZO) optimization of the lateral shift vector alpha
    for minimum lap time (or a user-supplied blackbox cost function).

    Motivation
    ----------
    The existing ``Opt_min_CurvTime`` class uses a two-stage pipeline:
    ZO/CMA-ES optimizes spline segment lengths ds → QP solves for alpha that
    minimizes *curvature* (a proxy for lap time).  The curvature proxy is
    inexact and couples geometry to the inner QP.

    ``Blackbox_raceline`` bypasses both stages.  Alpha is the optimization
    variable directly and the true lap time (from ``KinematicProfs``) is the
    cost function, so there is no proxy and no inner QP.

    Formulation
    -----------
    * Variable   : ``alpha`` ∈ R^N, one lateral shift per control point
    * Feasible set: box  ``alpha[i] ∈ [-(w_left[i] - sfty), (w_right[i] - sfty)]``
    * Objective  : ``minimize cost(alpha)``  [default: lap time]
    * Method     : projected ZO stochastic gradient descent

    The box constraint is the same as the QP constraint used inside
    ``Opt_min_CurvTime``, so the feasible region is identical.

    Fixed normal vectors
    --------------------
    Spline normal vectors are computed once from the Euclidean segment lengths
    of the initial ``reftrack`` and held fixed throughout the run.  Alpha is
    interpreted as displacement along these fixed normals, which keeps the
    box-projection exact and avoids recomputing splines at every iteration.

    Gradient estimation
    -------------------
    Two oracles are supported for sampling the random direction u:

    * ``'gaussian'`` : u ~ N(0, I_N)  — standard Gaussian ZO estimator
    * ``'sphere'``   : u ~ Uniform(S^{N-1})  — normalized Gaussian; trades
      higher variance for uniform coverage of the search space

    Two finite-difference schemes are supported:

    * ``'noth'`` (forward) : ``g = (f(alpha + μu) - f(alpha)) / μ * u``
      — 1 extra function evaluation per direction
    * ``'h'`` (central)    : ``g = (f(alpha + μu) - f(alpha - μu)) / 2μ * u``
      — 2 extra evaluations per direction, lower bias

    Parameters
    ----------
    reftrack : np.ndarray, shape (N, 4)
        Reference track [x, y, w_right, w_left].  Unclosed.
    ggv : np.ndarray
        GGV table already loaded, shape (K, 3) → [vx, ax_max, ay_max].
    ax_max_machines : np.ndarray
        Machine longitudinal limits, shape (L, 2) → [vx, ax_max].
    sfty : float
        Vehicle safety half-width in m (box constraint clearance).
    v_max : float
        Maximum velocity [m/s].
    si : float
        Raceline interpolation stepsize [m].
    fw : int or None
        Velocity profile convolution filter window length.  None = no filter.
    m_veh : float
        Vehicle mass [kg].
    drag_coeff : float
        Drag coefficient: 0.5 * c_w * A_front * rho_air.
    dyn_model_exp : float
        Dynamics model exponent in [1.0, 2.0].
    cost_fn : callable or None
        User-supplied cost function ``cost_fn(alpha) -> float``.
        If None the default lap-time oracle is used.
    init : str
        Initial alpha strategy.

        ``'random'``  — alpha_0 sampled uniformly inside the feasible box.
        ``'mincurv'`` — alpha_0 = QP minimum-curvature solution on reftrack
                        (warm start from the best geometric proxy).
    oracle : str
        ``'gaussian'`` or ``'sphere'`` (see above).
    mu : float
        ZO smoothing / perturbation magnitude μ.
    h : float
        Gradient-descent step size.
    t : int
        Number of random directions averaged per gradient estimate.
    grad_type : str
        ``'noth'`` (forward difference) or ``'h'`` (central difference).
    iterations : int
        Total number of gradient steps.
    kappa_bound : float
        Curvature bound used for the QP when ``init='mincurv'``.
    seed : int or None
        Global random seed set at construction time.  Reapply with
        ``find_alpha(seed=...)`` for per-run reproducibility.
    """

    def __init__(
        self,
        reftrack: np.ndarray,
        ggv: np.ndarray,
        ax_max_machines: np.ndarray,
        sfty: float = 1.0,
        v_max: float = 22.88,
        si: float = 0.8,
        fw: int = 3,
        m_veh: float = 1000.0,
        drag_coeff: float = 0.0,
        dyn_model_exp: float = 1.0,
        cost_fn=None,
        init: str = 'random',
        oracle: str = 'gaussian',
        mu: float = 0.05,
        h: float = 0.001,
        t: int = 1,
        grad_type: str = 'noth',
        iterations: int = 300,
        kappa_bound: float = 0.7,
        seed: int = None,
    ):
        # ---- store hyper-parameters -------------------------------------------
        self.reftrack      = reftrack.copy()
        self.N             = reftrack.shape[0]
        self.ggv           = ggv
        self.ax_max_machines = ax_max_machines
        self.sfty          = sfty
        self.v_max         = v_max
        self.si            = si
        self.fw            = fw
        self.m_veh         = m_veh
        self.drag_coeff    = drag_coeff
        self.dyn_model_exp = dyn_model_exp
        self.cost_fn       = cost_fn
        self.init          = init
        self.oracle        = oracle
        self.mu            = mu
        self.h             = h
        self.t             = t
        self.grad_type     = grad_type
        self.iterations    = iterations
        self.kappa_bound   = kappa_bound

        if seed is not None:
            np.random.seed(seed)

        # ---- box constraints  (same convention as QP inside Opt_min_CurvTime) --
        #   positive alpha → shift toward the right boundary (w_right side)
        #   negative alpha → shift toward the left  boundary (w_left  side)
        self.lo = -(reftrack[:, 3] - sfty)   # lower bound (max left shift)
        self.hi =   reftrack[:, 2] - sfty    # upper bound (max right shift)

        bad = np.where(self.lo > self.hi)[0]
        if bad.size > 0:
            raise ValueError(
                f"Track is narrower than the safety margin at {bad.size} point(s) "
                f"(indices: {bad[:5]}{'...' if bad.size > 5 else ''}).  "
                "Reduce sfty or check track widths.")

        # ---- fixed normal vectors (computed once from initial Euclidean ds) ----
        # Convention must match ShortestPath / Opt_min_CurvTime / run_mc:
        # compute N-1 inter-point distances then append ds_init[0] as the closing
        # segment proxy, NOT the actual wrap-around distance (reftrack[-1]→reftrack[0]).
        ds_init = np.sqrt(np.sum(np.power(np.diff(reftrack[:, :2], axis=0), 2), axis=1))
        ds_init = np.append(ds_init, ds_init[0])
        _, _, self._M, self._normvec = calc_splines(
            path=np.vstack((reftrack[:, :2], reftrack[0, :2])),
            el_lengths=ds_init)

        # ---- p_ggv cache (keyed by interpolated raceline length) ---------------
        self._p_ggv_cache: dict = {}

        # ---- initial alpha + result placeholders --------------------------------
        self.alpha_0   = self._init_alpha()
        self.alpha_opt = self.alpha_0.copy()
        self.history   = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_alpha(self) -> np.ndarray:
        """Return the initial alpha vector according to self.init.

        Supported modes
        ---------------
        'zero'
            Start with zero shift (alpha = 0), i.e. the original reference track.
        'random'
            Uniform sample inside the feasible box [lo, hi].
        'mincurv'
            Solve the minimum-curvature QP (H_f) and use the result as the
            warm-start.  This is the best geometric proxy for lap-time and
            typically the fastest starting point for further ZO/CMA refinement.
        'shortest' (aliases: 'shortest_path', 'sp')
            Solve the Optimal Shortest Path (OSP) QP and use that alpha.
            The shortest path has maximum corner radii (less lateral movement
            overall) which sometimes beats mincurv as a starting point on
            tracks where sustained high speed on long straights matters more
            than corner apex curvature.
        """
        if self.init == 'random':
            return np.random.uniform(self.lo, self.hi)
        elif self.init == 'zero':
            return np.zeros(self.N)

        elif self.init == 'mincurv':
            # Warm-start from the QP min-curvature solution on the original reftrack.
            # This is the best geometric proxy available and typically a better starting
            # point than a random interior point.
            H, f_vec, G, h_vec = H_f(
                reftrack=self.reftrack,
                normvectors=self._normvec,
                A=self._M,
                kappa_bound=self.kappa_bound,
                w_veh=2*self.sfty,
                closed=True)
            alpha_mc = quadprog.solve_qp(H, -f_vec, -G.T, -h_vec, 0)[0]
            # Clip in case QP and Blackbox box constraints differ by a rounding epsilon
            return alpha_mc

        elif self.init in ('shortest', 'shortest_path', 'sp'):
            # Warm-start from the OSP (Optimal Shortest Path) QP solution.
            # OSP internally uses w_veh/2 as the per-side clearance; we pass
            # 2*sfty so that w_veh/2 = sfty, giving the same feasible box as
            # Blackbox_raceline uses for the gradient descent.
            H_sp, f_sp, G_sp, h_sp = OSP(
                reftrack=self.reftrack,
                normvectors=self._normvec,
                w_veh=2.0 * self.sfty)
            alpha_sp = quadprog.solve_qp(H_sp, -f_sp, -G_sp.T, -h_sp, 0)[0]
            # Clip to Blackbox box (OSP and H_f constraints can differ slightly
            # due to numerical tolerances or rounding in dev_max).
            return alpha_sp

        else:
            raise ValueError(
                f"Unknown init mode '{self.init}'.  "
                "Use 'random', 'mincurv', or 'shortest'.")

    def _sample_direction(self) -> np.ndarray:
        """Sample a unit-scaled random direction u according to self.oracle."""
        u = np.random.normal(0.0, 1.0, size=self.N)
        if self.oracle == 'sphere':
            norm = np.linalg.norm(u)
            if norm > 1e-12:
                u /= norm          # project onto N-sphere
        elif self.oracle != 'gaussian':
            raise ValueError(
                f"Unknown oracle '{self.oracle}'.  Use 'gaussian' or 'sphere'.")
        return u

    def _eval_cost(self, alpha: np.ndarray) -> float:
        """Evaluate cost at alpha (dispatch to user cost_fn or lap-time oracle)."""
        if self.cost_fn is not None:
            return float(self.cost_fn(alpha))
        return self._laptime_cost(alpha)

    def _laptime_cost(self, alpha: np.ndarray) -> float:
        """
        Default cost oracle: build raceline from alpha and return the lap time
        computed by the KinematicProfs forward-backward velocity solver.
        """
        _, _, cx, cy, si_idx, tv, _, _, el = create_raceline(
            refline=self.reftrack[:, :2],
            normvectors=self._normvec,
            alpha=alpha,
            stepsize_interp=self.si)
        _, kappa = calc_head_curv_an(
            coeffs_x=cx, coeffs_y=cy, ind_spls=si_idx, t_spls=tv)

        n = kappa.size
        if n not in self._p_ggv_cache:
            self._p_ggv_cache[n] = np.repeat(
                np.expand_dims(self.ggv, axis=0), n, axis=0)

        vx = calc_vel_profile(
            ggv=self.ggv,
            ax_max_machines=self.ax_max_machines,
            v_max=self.v_max,
            kappa=kappa,
            el_lengths=el,
            closed=True,
            filt_window=self.fw,
            dyn_model_exp=self.dyn_model_exp,
            drag_coeff=self.drag_coeff,
            m_veh=self.m_veh,
            p_ggv=self._p_ggv_cache[n])

        vx_cl  = np.append(vx, vx[0])
        ax_prof = calc_ax_profile(
            vx_profile=vx_cl, el_lengths=el, eq_length_output=False)
        t_prof  = calc_t_profile(
            vx_profile=vx, ax_profile=ax_prof, el_lengths=el)
        return float(t_prof[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_alpha(
        self,
        solver: str = 'ZO',
        n_iter: int = None,
        # ZO parameters
        step_size: float = None,
        mu: float = None,
        t: int = None,
        grad_type: str = None,
        # CMA-ES parameters
        sigma: float = None,
        popsize: int = None,
        # Common
        seed: int = None,
        verbose: bool = True,
        print_every: int = 50,
    ):
        """
        Minimize the cost over alpha using either ZO gradient descent or CMA-ES.

        Both solvers respect the box constraint ``alpha ∈ [lo, hi]`` via
        projection (ZO) or clipping of the sampled population (CMA-ES).
        The best alpha seen across all evaluations is tracked and stored in
        ``self.alpha_opt``; ``self.history`` records the best-so-far cost at
        the end of each iteration/generation.

        Parameters
        ----------
        solver : str
            ``'ZO'``  — projected zeroth-order stochastic gradient descent.
            ``'CMA'`` — CMA-ES evolutionary strategy.
        n_iter : int or None
            For ZO  : number of gradient steps (default: ``self.iterations``).
            For CMA : number of generations    (default: ``self.iterations``).
            One CMA generation = ``popsize`` function evaluations.
        step_size : float or None   [ZO only]
            Gradient step size h.  Defaults to ``self.h``.
        mu : float or None          [ZO only]
            Perturbation magnitude μ.  Defaults to ``self.mu``.
        t : int or None             [ZO only]
            Directions averaged per gradient estimate.  Defaults to ``self.t``.
        grad_type : str or None     [ZO only]
            ``'noth'`` (forward difference) or ``'h'`` (central difference).
            Defaults to ``self.grad_type``.
        sigma : float or None       [CMA only]
            Initial step-size (standard deviation) for CMA-ES.
            A good default is ~(1/3) × (hi - lo) range, but it is
            problem-specific.  Defaults to ``self.h`` (used as a scale proxy).
        popsize : int or None       [CMA only]
            Population size (number of candidate solutions per generation).
            Defaults to ``4 + int(3 * np.log(self.N))`` (CMA-ES heuristic).
        seed : int or None
            Random seed reset before the loop.
        verbose : bool
            Print progress every ``print_every`` iterations/generations.
        print_every : int
            Verbosity interval.

        Returns
        -------
        best_alpha : np.ndarray, shape (N,)
            Alpha that achieved the lowest observed cost.
        history : np.ndarray, shape (n_iter + 1,)
            ``history[0]`` = cost of the initial alpha_0.
            ``history[k]`` = best-so-far cost after iteration/generation k.
        """
        if seed is not None:
            np.random.seed(seed)

        n_iter = n_iter if n_iter is not None else self.iterations

        # ---- evaluate starting point (common to both solvers) ------------------
        alpha_start = self.alpha_0.copy()
        f_start     = self._eval_cost(alpha_start)
        best_alpha  = alpha_start.copy()
        best_cost   = f_start
        history     = np.full(n_iter + 1, np.nan)
        history[0]  = f_start

        if verbose:
            print(f"[Blackbox_raceline/{solver}] init  laptime = {f_start:.4f} s"
                  f"  (init='{self.init}', N={self.N})")

        # ====================================================================
        # ZO — projected stochastic gradient descent
        # ====================================================================
        if solver == 'ZO':
            step_size = step_size if step_size is not None else self.h
            mu        = mu        if mu        is not None else self.mu
            t_dirs    = t         if t         is not None else self.t
            grad_type = grad_type if grad_type is not None else self.grad_type

            alpha  = alpha_start.copy()
            f_curr = f_start

            for k in range(n_iter):

                # gradient estimate averaged over t_dirs random directions
                grad_sum = np.zeros(self.N)
                for _ in range(t_dirs):
                    u = self._sample_direction()

                    alpha_fwd = np.clip(alpha + mu * u, self.lo, self.hi)
                    f_fwd     = self._eval_cost(alpha_fwd)

                    if grad_type == 'noth':
                        g = (f_fwd - f_curr) / mu * u
                    else:                              # 'h' = central difference
                        alpha_bwd = np.clip(alpha - mu * u, self.lo, self.hi)
                        f_bwd     = self._eval_cost(alpha_bwd)
                        g = (f_fwd - f_bwd) / (2.0 * mu) * u

                    grad_sum += g

                grad_est = grad_sum / t_dirs

                # projected gradient step
                alpha  = np.clip(alpha - step_size * grad_est, self.lo, self.hi)
                f_curr = self._eval_cost(alpha)

                if f_curr < best_cost:
                    best_cost  = f_curr
                    best_alpha = alpha.copy()
                history[k + 1] = best_cost

                if verbose and (k + 1) % print_every == 0:
                    print(f"  iter {k + 1:4d}/{n_iter}:  laptime = {f_curr:.4f} s"
                          f"   best = {best_cost:.4f} s")

        # ====================================================================
        # CMA-ES — covariance matrix adaptation evolution strategy
        # ====================================================================
        elif solver == 'CMA':
            # Default sigma: 1/6 of the mean box half-width (explores ~1σ
            # within the feasible range with high probability)
            if sigma is None:
                sigma = float(np.mean(self.hi - self.lo)) / 6.0
            if popsize is None:
                popsize = 4 + int(3 * np.log(self.N))   # CMA-ES standard heuristic

            # Wrap _eval_cost to track best-ever across all individual evaluations
            _best = {'alpha': best_alpha.copy(), 'cost': best_cost}

            def _tracked_cost(a: np.ndarray) -> float:
                c = self._eval_cost(a)
                if c < _best['cost']:
                    _best['cost']  = c
                    _best['alpha'] = a.copy()
                return c

            cma = ConstrainedCMAES_t(
                _tracked_cost,
                mean=alpha_start.copy(),
                sigma=sigma,
                popsize=popsize,
                bounds1=self.hi,
                bounds2=self.lo)
            # ConstrainedCMAES_t.update() references self.iterations (normally
            # set inside optimize()).  We manage the loop ourselves for
            # best-tracking, so we must initialise it here.
            cma.iterations = 0

            for gen in range(n_iter):
                samples = cma.sample_population()
                fitness = np.array([cma.objective_function(s) for s in samples])
                cma.update(samples, fitness)
                cma.iterations += 1
                history[gen + 1] = _best['cost']  # best-so-far after this generation

                if verbose and (gen + 1) % print_every == 0:
                    print(f"  gen  {gen + 1:4d}/{n_iter}:  gen_best = {min(fitness):.4f} s"
                          f"   best = {_best['cost']:.4f} s")

            best_alpha = _best['alpha']
            best_cost  = _best['cost']

        else:
            raise ValueError(f"Unknown solver '{solver}'.  Use 'ZO' or 'CMA'.")

        # ---- finalise ----------------------------------------------------------
        if verbose:
            print(f"[Blackbox_raceline/{solver}] done  best laptime = {best_cost:.4f} s"
                  f"  (improvement = {f_start - best_cost:.4f} s)")

        self.alpha_opt = best_alpha
        self.history   = history
        return best_alpha, history

    def generate_raceline(self, alpha: np.ndarray = None):
        """
        Build the full raceline and kinematic profiles for a given alpha.

        Parameters
        ----------
        alpha : np.ndarray or None
            Lateral shift vector, shape (N,).  If None, uses self.alpha_opt
            (the best alpha from the most recent find_alpha() call).

        Returns
        -------
        s : np.ndarray
            Cumulative distance along the raceline [m], shape (M,).
        vx : np.ndarray
            Velocity profile [m/s], shape (M,)  (unclosed).
        ax : np.ndarray
            Longitudinal acceleration [m/s²], shape (M,)  (closed wrap-around).
        kappa : np.ndarray
            Curvature [rad/m], shape (M,).
        t_prof : np.ndarray
            Cumulative lap time [s], shape (M + 1,).  t_prof[-1] = total lap time.
        raceline_xy : np.ndarray
            Interpolated raceline coordinates [x, y], shape (M, 2).
        """
        if alpha is None:
            if self.alpha_opt is None:
                raise RuntimeError(
                    "No alpha available.  Run find_alpha() first or pass alpha explicitly.")
            alpha = self.alpha_opt

        raceline_xy, _, cx, cy, si_idx, tv, _, _, el = create_raceline(
            refline=self.reftrack[:, :2],
            normvectors=self._normvec,
            alpha=alpha,
            stepsize_interp=self.si)
        _, kappa = calc_head_curv_an(
            coeffs_x=cx, coeffs_y=cy, ind_spls=si_idx, t_spls=tv)

        s = cumulative_distances(el)

        n = kappa.size
        if n not in self._p_ggv_cache:
            self._p_ggv_cache[n] = np.repeat(
                np.expand_dims(self.ggv, axis=0), n, axis=0)

        vx = calc_vel_profile(
            ggv=self.ggv,
            ax_max_machines=self.ax_max_machines,
            v_max=self.v_max,
            kappa=kappa,
            el_lengths=el,
            closed=True,
            filt_window=self.fw,
            dyn_model_exp=self.dyn_model_exp,
            drag_coeff=self.drag_coeff,
            m_veh=self.m_veh,
            p_ggv=self._p_ggv_cache[n])

        vx_cl  = np.append(vx, vx[0])
        ax     = calc_ax_profile(
            vx_profile=vx_cl, el_lengths=el, eq_length_output=False)
        t_prof = calc_t_profile(
            vx_profile=vx, ax_profile=ax, el_lengths=el)

        return s, vx, ax, kappa, t_prof, raceline_xy

    def reset_alpha(self, new_init: str = None):
        """
        Re-initialize alpha_0 (and reset alpha_opt / history).

        Useful for multi-start experiments: call reset_alpha() then find_alpha()
        to run a fresh trial from a new starting point.

        Parameters
        ----------
        new_init : str or None
            Override init strategy ('random' or 'mincurv').  None keeps current.
        """
        if new_init is not None:
            self.init = new_init
        self.alpha_0   = self._init_alpha()
        self.alpha_opt = self.alpha_0.copy()
        self.history   = None


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
        v_max = 22.88,
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
    :param v_max:                           maximum velocity in m/s for the velocity profile calculation.
    :type v_max:                            float
    :param plot:                            flag to enable plotting of the track and raceline.
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

    vm = v_max
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