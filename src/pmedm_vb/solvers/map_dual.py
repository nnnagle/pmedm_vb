"""MAP / Laplace solver: damped Newton on the dual objective.

The Hessian, from the derivation, is

.. math::

    n^{-1} \\frac{d^2 M}{d\\lambda d\\lambda'}
        = X' \\mathrm{diag}(p) X + \\frac{n}{N^2}\\Sigma - (X'p)(p'X)

**Structure.** Zones are block groups and block groups nest in tracts, so
``S = X' diag(p) X`` is block diagonal by tract: an entry pairing two stacked
rows sums over zones lying in both rows' areas, and a block group lies in one
tract. Each tract's block holds its own cells and the cells of its block
groups, with

* tract x tract: ``X_T' diag(sum_b p_b) X_T``
* tract x block group ``b``: ``X_T' diag(p_b) X_B``
* block group ``b`` x itself: ``X_B' diag(p_b) X_B``
* distinct block groups: zero

for ``p_b`` block group ``b``'s row of ``p``. The tract-tapered ``Sigma`` is
block diagonal by the same tracts (see :mod:`pmedm_vb.assemble.sigma`), so
``K = S + c * Sigma_blocks`` is a set of small dense blocks, and

.. math::

    H = K + W J W', \\qquad W = [\\sqrt{c} B_{global} \\;\\; u], \\qquad
    J = \\mathrm{diag}(I, -1)

with ``u = X'p``. The ``B_global`` columns are present only for the untapered
``Sigma``; tapered, the correction is the rank-1 downdate alone. A Newton step
is one Cholesky per tract and a Woodbury solve through the ``J + W'K^{-1}W``
capacitance -- indefinite because of the downdate, so it goes through LU.

**Method, and how it differs from PMEDMrcpp.** The original solver was a
trust-region Newton-CG, preconditioned by a factorisation of the Hessian that
was refreshed every 20 iterations. That amortised one expensive sparse
factorisation; here a factorisation is a few dozen small dense Choleskys, so it
is simply redone every iteration and the step is an exact Newton step.

A trust region is not needed either. For ``alpha`` in ``(0, 1]`` ``Sigma`` is
positive definite, and ``X'diag(p)X - (X'p)(p'X)`` is the covariance of ``X``
under ``p``, so ``H >= c * Sigma > 0`` everywhere: the dual is strongly convex.
Newton with a backtracking line search converges globally on such problems
(Boyd & Vandenberghe, *Convex Optimization*, section 9.5). Strong convexity
also means the minimiser is unique, so the choice of algorithm changes the cost
of the answer and not the answer.

The stopping rule is the Newton decrement, ``0.5 * g' H^{-1} g``, rather than a
gradient norm (Boyd & Vandenberghe, section 9.5.1). It is affine invariant --
the gradient norm depends on how the stacked constraints are scaled -- and it
is free, since ``H^{-1} g`` is the step already computed.

The Laplace approximation reuses the converged Hessian; see
:func:`laplace_precision`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.progress import logger
from pmedm_vb.solvers.base import (
    ConstraintOperator,
    DualState,
    dual_state,
    penalty_scale,
)

#: Armijo sufficient-decrease fraction and backtracking factor. Both sit inside
#: the ranges Boyd & Vandenberghe (section 9.2) give: 0.01-0.3 and 0.1-0.8.
ARMIJO = 0.25
BACKTRACK = 0.5
MAX_HALVINGS = 50


@dataclass
class MAPResult:
    """Converged MAP fit.

    Attributes
    ----------
    lam:
        Multipliers at the optimum.
    W:
        Population weights, ``(n_zones, n_individuals)``, summing to ``N``.
    objective:
        Dual objective at the optimum.
    n_iter:
        Newton steps taken.
    converged:
        Whether the decrement tolerance was met before the iteration cap.
    alpha, taper:
        The ``Sigma`` this was fitted under, as passed to
        :meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.sigma`.
    newton_decrement:
        ``0.5 * g' H^{-1} g`` at ``lam`` -- an estimate of how far the
        objective is above its minimum.
    trace:
        Per iteration: ``objective``, ``decrement``, ``step`` (the line-search
        length taken, 0 on the final row) and ``max_abs_z`` (the largest
        ``|N X'p - Y| / sqrt(v)``, residuals in published standard errors).
        ``max_abs_z`` is a diagnostic, not a target: the penalised fit leaves
        residuals by design. ``mahalanobis`` is ``e' Sigma^{-1} e / m`` for
        ``e = N X'p - Y`` and ``m`` constraints: the fit measured against this
        run's own ``Sigma``, correlations included. At ``alpha = 1`` it is the
        mean of ``z^2``. Where ``max_abs_z`` reads each cell against its own
        standard error, this lets a residual be explained by correlated
        neighbours, which is what the penalty itself does -- so it is the one to
        compare across ``alpha`` and ``taper``. It is not a chi-squared
        statistic: the fitted residuals are shrunk by the penalty. Linearising
        the fit about the optimum, its expectation when the model holds is
        ``tr(c Sigma H^{-1}) / m``, which is below 1 because
        ``H = Cov_p(X) + c Sigma``.
    """

    lam: np.ndarray
    W: np.ndarray
    objective: float
    n_iter: int
    converged: bool
    alpha: float
    taper: str | None
    newton_decrement: float
    trace: dict[str, np.ndarray] = field(default_factory=dict)


class DualHessian:
    """The dual Hessian at one ``lambda``, factored and never materialised.

    Offers :meth:`matvec`, :meth:`solve` and :meth:`logdet`, the same interface
    as :class:`~pmedm_vb.assemble.sigma.Sigma`.

    Parameters
    ----------
    inputs:
        The problem.
    state:
        The dual evaluated at the ``lambda`` to take the Hessian at.
    sigma:
        The covariance the dual was evaluated under.
    """

    def __init__(self, inputs: PMEDMInputs, state: DualState, sigma: Sigma) -> None:
        c = penalty_scale(inputs)
        self.size = inputs.n_constraints
        self.rows: list[np.ndarray] = []
        self.blocks: list[np.ndarray] = []
        self.factors = []
        for rows, s_block in _tract_blocks(inputs, state.p):
            block = s_block + c * sigma.block_dense(rows)
            try:
                factor = cho_factor(block, lower=True)
            except np.linalg.LinAlgError as error:
                raise ValueError(
                    "a tract block of S + c*Sigma is not positive definite, "
                    "which D > 0 should rule out; check sigma_v and sigma_l"
                ) from error
            self.rows.append(rows)
            self.blocks.append(block)
            self.factors.append(factor)

        b = sigma.global_factor
        columns = [state.u[:, None]] if b is None else [np.sqrt(c) * b, state.u[:, None]]
        self.w = np.hstack(columns)
        self.signs = np.ones(self.w.shape[1])
        self.signs[-1] = -1.0
        self.k_inv_w = self._k_solve(self.w)
        capacitance = np.diag(self.signs) + self.w.T @ self.k_inv_w
        self.capacitance = lu_factor(capacitance)

    def _k_solve(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for rows, factor in zip(self.rows, self.factors):
            out[rows] = cho_solve(factor, x[rows])
        return out

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """``H @ x``. For checking; the solver needs only :meth:`solve`."""
        out = np.empty_like(x, dtype=float)
        for rows, block in zip(self.rows, self.blocks):
            out[rows] = block @ x[rows]
        wx = self.w.T @ x
        return out + self.w @ (self.signs[:, None] * wx if x.ndim == 2 else self.signs * wx)

    def solve(self, x: np.ndarray) -> np.ndarray:
        """``H^-1 @ x`` by Woodbury against the per-tract factors."""
        y = self._k_solve(x)
        return y - self.k_inv_w @ lu_solve(self.capacitance, self.w.T @ y)

    def logdet(self) -> float:
        """``log det H`` by the matrix determinant lemma.

        ``det(K + WJW') = det(K) det(J) det(J + W'K^{-1}W)``, using
        ``J^{-1} = J``. ``H`` is positive definite, so the last two factors
        must have the same sign, which is checked.
        """
        k_part = sum(2.0 * float(np.log(np.diag(f[0])).sum()) for f in self.factors)
        lu, _ = self.capacitance
        diagonal = np.diag(lu)
        sign_c = np.prod(np.sign(diagonal)) * _permutation_sign(self.capacitance[1])
        sign_j = np.prod(self.signs)
        if sign_c * sign_j <= 0:
            raise ValueError("Hessian determinant is not positive; H is not definite")
        return k_part + float(np.log(np.abs(diagonal)).sum())


def solve_map(
    inputs: PMEDMInputs,
    *,
    alpha: float,
    taper: str | None = "tract",
    init: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    device: str = "cpu",
) -> MAPResult:
    """Fit the MAP / penalised MaxEnt solution.

    Parameters
    ----------
    alpha, taper:
        Select ``Sigma``; see
        :meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.sigma`. ``alpha`` has no
        default because the sweep over it is the comparison.
    init:
        Starting multipliers. ``None`` starts at zero, where ``p = q``.
    tol:
        Stop once the Newton decrement ``0.5 * g' H^{-1} g`` is at most this.
    device:
        Only ``"cpu"``: this solver is numpy and scipy. The argument exists for
        symmetry with :func:`~pmedm_vb.solvers.vb.solve_vb`.
    """
    if device != "cpu":
        raise ValueError(f"solve_map runs on numpy/scipy only; got device={device!r}")
    sigma = inputs.sigma(alpha, taper)
    op = ConstraintOperator(inputs)
    lam = np.zeros(inputs.n_constraints) if init is None else np.asarray(init, float)
    state = dual_state(inputs, lam, sigma, op)
    scale = np.sqrt(inputs.sigma_v)
    targets = inputs.targets()

    trace: dict[str, list[float]] = {
        "objective": [], "decrement": [], "step": [], "max_abs_z": [], "mahalanobis": [],
    }
    converged = False
    decrement = np.inf
    n_iter = 0
    started = time.perf_counter()
    logger.info(
        "solve_map PUMA %s: %s constraints, alpha=%s, taper=%s",
        inputs.puma, f"{inputs.n_constraints:,}", alpha, taper,
    )
    for n_iter in range(max_iter + 1):
        tick = time.perf_counter()
        hessian = DualHessian(inputs, state, sigma)
        direction = -hessian.solve(state.gradient)
        slope = float(state.gradient @ direction)
        if slope >= 0:
            raise ValueError(
                "Newton direction is not a descent direction; the Hessian solve "
                "has lost precision"
            )
        decrement = -0.5 * slope
        trace["objective"].append(state.objective)
        trace["decrement"].append(decrement)
        residual = inputs.N * state.u - targets
        trace["max_abs_z"].append(float(np.abs(residual / scale).max()))
        trace["mahalanobis"].append(float(residual @ sigma.solve(residual)) / residual.size)
        if decrement <= tol:
            converged = True
            trace["step"].append(0.0)
            break
        if n_iter == max_iter:
            trace["step"].append(0.0)
            break

        step = 1.0
        for _ in range(MAX_HALVINGS):
            candidate = dual_state(inputs, state.lam + step * direction, sigma, op)
            if candidate.objective <= state.objective + ARMIJO * step * slope:
                break
            step *= BACKTRACK
        else:
            raise ValueError(
                f"line search failed after {MAX_HALVINGS} halvings at iteration "
                f"{n_iter} (decrement {decrement:.3e}); tol may be below what "
                f"the objective can resolve in floating point"
            )
        trace["step"].append(step)
        logger.info(
            "  iter %3d  objective %.12g  decrement %.2e  step %.3g  max|z| %.2f  "
            "e'S^-1e/m %.4f  %.1fs",
            n_iter, state.objective, decrement, step, trace["max_abs_z"][-1],
            trace["mahalanobis"][-1],
            time.perf_counter() - tick,
        )
        state = candidate

    logger.info(
        "solve_map PUMA %s: %s after %d iterations, decrement %.2e, "
        "e'S^-1e/m %.4f, %.1fs",
        inputs.puma, "converged" if converged else "NOT converged",
        n_iter, decrement, trace["mahalanobis"][-1], time.perf_counter() - started,
    )
    return MAPResult(
        lam=state.lam,
        W=inputs.N * state.p,
        objective=state.objective,
        n_iter=n_iter,
        converged=converged,
        alpha=alpha,
        taper=taper,
        newton_decrement=decrement,
        trace={key: np.asarray(value) for key, value in trace.items()},
    )


def laplace_precision(inputs: PMEDMInputs, result: MAPResult) -> DualHessian:
    """The converged dual Hessian, factored.

    This is the Hessian of ``n^{-1} M``, the objective as
    :func:`~pmedm_vb.solvers.base.dual_objective` defines it, at
    ``result.lam``. It is returned as an operator rather than an
    ``(n_constraints, n_constraints)`` array, which would reintroduce the
    materialisation :class:`~pmedm_vb.assemble.sigma.Sigma` exists to avoid.

    Still open, and deliberately not decided here: how this is scaled (the
    ``n^{-1}``) and mapped into the posterior precision the VB comparison uses.
    """
    sigma = inputs.sigma(result.alpha, result.taper)
    state = dual_state(inputs, result.lam, sigma)
    return DualHessian(inputs, state, sigma)


def _tract_blocks(inputs: PMEDMInputs, p: np.ndarray):
    """Yield ``(rows, S_t)`` per tract: the stacked-row indices of tract ``t``'s
    block of ``S = X' diag(p) X``, and the block itself, dense, in that order.

    Rows are the tract's own cells followed by each of its block groups' cells;
    the sub-blocks are those in the module docstring.
    """
    inputs.constraint_tracts()  # raises unless A_B is the identity, assumed below
    x_t = sp.csr_matrix(inputs.X_T)
    x_b = sp.csr_matrix(inputs.X_B)
    n_tracts, n_zones = inputs.A_T.shape
    c_t, c_b = x_t.shape[1], x_b.shape[1]
    split = inputs.Y_T.size
    zone_tract = inputs.zone_tracts()
    tract_cells = np.arange(c_t) * n_tracts
    bg_cells = split + np.arange(c_b) * n_zones

    for tract in range(n_tracts):
        zones = np.flatnonzero(zone_tract == tract)
        size = c_t + zones.size * c_b
        block = np.zeros((size, size))
        rows = np.concatenate([tract_cells + tract] + [bg_cells + z for z in zones])

        weight = p[zones].sum(axis=0)
        block[:c_t, :c_t] = (x_t.T @ x_t.multiply(weight[:, None])).toarray()
        for i, zone in enumerate(zones):
            weighted = x_b.multiply(p[zone][:, None]).tocsr()
            at = slice(c_t + i * c_b, c_t + (i + 1) * c_b)
            cross = (x_t.T @ weighted).toarray()
            block[:c_t, at] = cross
            block[at, :c_t] = cross.T
            block[at, at] = (x_b.T @ weighted).toarray()
        yield rows, block


def _permutation_sign(pivots: np.ndarray) -> float:
    """Sign of the row permutation LAPACK's ``getrf`` pivots describe."""
    return -1.0 if np.count_nonzero(pivots != np.arange(pivots.size)) % 2 else 1.0
