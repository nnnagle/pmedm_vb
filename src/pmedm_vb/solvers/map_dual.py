"""MAP / Laplace solver: Gauss-Newton on the dual objective.

This is the port of the existing R/Rcpp implementation, kept so the paper can
compare against it. The Hessian, from the derivation, is

.. math::

    n^{-1} \\frac{d^2 M}{d\\lambda d\\lambda'}
        = X' \\mathrm{diag}(p) X + \\frac{n}{N^2}\\Sigma - (X'p)(p'X)

-- a sparse matrix plus a diagonal, downdated by a rank-one term. The rank-one
part is dense and is what made a custom Hessian evaluation necessary in the
original code. Two routes are open here and the choice is open:

* Form the sparse part and apply the rank-one downdate through the
  Sherman-Morrison identity.
* Skip the explicit Hessian and use matrix-free conjugate gradients, since
  :class:`~pmedm_vb.solvers.base.ConstraintOperator` already supplies
  Hessian-vector products.

The Laplace approximation reuses the converged Hessian as the posterior
precision, which is the comparison point for the VB posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmedm_vb.assemble.inputs import PMEDMInputs


@dataclass
class MAPResult:
    """Converged MAP fit.

    Attributes
    ----------
    lam:
        Multipliers at the optimum.
    W:
        Population weights, ``(n_zones, n_individuals)``.
    objective:
        Dual objective at the optimum.
    n_iter:
        Iterations taken.
    converged:
        Whether the gradient tolerance was met before the iteration cap.
    """

    lam: np.ndarray
    W: np.ndarray
    objective: float
    n_iter: int
    converged: bool


def solve_map(
    inputs: PMEDMInputs,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    device: str = "cpu",
) -> MAPResult:
    """Fit the MAP / penalised MaxEnt solution."""
    raise NotImplementedError


def laplace_precision(inputs: PMEDMInputs, result: MAPResult) -> np.ndarray:
    """Posterior precision at the MAP estimate -- the converged dual Hessian."""
    raise NotImplementedError
