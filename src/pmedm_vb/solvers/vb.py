"""Variational Bayes solver -- the method the paper is about.

Unlike :mod:`~pmedm_vb.solvers.map_dual`, which reports a point estimate plus a
curvature-based approximation to the posterior, this fits an approximating
distribution by maximising the evidence lower bound.

The variational family is **not yet chosen**, and the choice is the substantive
modelling decision of the paper rather than an implementation detail: it sets
what correlation structure the approximate posterior can express, and so what
the comparison against the Laplace approximation is actually measuring. Fix it
before filling this in.

The README notes VB may be seeded from a MAP solution, or from a default that
does not require one; :func:`solve_vb` takes ``init`` for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.solvers.map_dual import MAPResult


@dataclass
class VBResult:
    """Converged variational fit.

    Attributes
    ----------
    params:
        Variational parameters, keyed by name. Shapes depend on the family.
    elbo:
        Evidence lower bound at convergence.
    elbo_trace:
        ELBO by iteration -- the practical convergence diagnostic, since the
        bound must increase monotonically under a correct implementation.
    n_iter, converged:
        As for :class:`~pmedm_vb.solvers.map_dual.MAPResult`.
    """

    params: dict[str, np.ndarray]
    elbo: float
    elbo_trace: np.ndarray
    n_iter: int
    converged: bool


def solve_vb(
    inputs: PMEDMInputs,
    *,
    init: MAPResult | None = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    device: str = "cpu",
) -> VBResult:
    """Fit the variational posterior.

    Parameters
    ----------
    init:
        Optional MAP fit to seed from. ``None`` uses a family-specific default.
    """
    raise NotImplementedError


def posterior_weights(result: VBResult, *, n_draws: int = 1) -> np.ndarray:
    """Draw population weight matrices from the fitted variational posterior."""
    raise NotImplementedError
