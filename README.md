# OptiLine-Py

**Raceline Optimisation Toolkit**

OptiLine-Py is a Python package for computing optimal racing lines around closed circuits. It offers multiple levels of fidelity — from purely geometric methods (minimum-curvature and shortest-path via QP) to lap-time proxy optimisation (CMA-ES / ZORM over kinematic velocity profiles) to full minimum-time optimal control via direct Gauss–Legendre collocation with CasADi and IPOPT. The package includes a
procedural map generator, enabling the creation of arbitrarily large synthetic track datasets for machine-learning research.

Requires Python 3.10–3.12, along with NumPy, SciPy, Matplotlib, quadprog, and CasADi.

## Documentation

For full documentation, API reference, usage examples, and track/vehicle data formats, see the [GitHub repository](https://github.com/amirali78frz/OptiLine).
