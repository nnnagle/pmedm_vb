"""Pieces shared by the MAP and VB solvers.

The centrepiece is :class:`ConstraintOperator`, which applies the stacked
Kronecker product from ``pmedm_derivation.md`` without ever forming it.

Given ``W`` of shape ``(n_zones, n_individuals)``, the constraints are

.. math::

    Y_T = A_T W X_T, \\qquad Y_B = A_B W X_B

and ``vec(A W X) = (X' kron A) vec(W)``. So the product against the Kronecker
matrix is just the two matrix multiplications on the left, and its adjoint,
which the gradient needs, is

.. math::

    V \\mapsto A' V X'

by the same identity. The Kronecker matrix has
``(tract cells + block group cells)`` rows and ``(zones x individuals)``
columns and is dense in the rank-one term; the factors are small and sparse.

All ``vec`` operations are column-major (``order="F"``), as the identity
requires.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.sigma import Sigma


class ConstraintOperator:
    """Applies ``X_tilde'`` and its adjoint via the stored factors.

    Parameters
    ----------
    inputs:
        Supplies ``A_T``, ``X_T``, ``A_B``, ``X_B`` and the shapes.
    """

    def __init__(self, inputs: PMEDMInputs) -> None:
        self.A_T = sp.csr_matrix(inputs.A_T)
        self.A_B = sp.csr_matrix(inputs.A_B)
        self.X_T = sp.csr_matrix(inputs.X_T)
        self.X_B = sp.csr_matrix(inputs.X_B)
        self.shape_T = inputs.Y_T.shape
        self.shape_B = inputs.Y_B.shape
        self.split = inputs.Y_T.size

    def forward(self, W: np.ndarray) -> np.ndarray:
        """Map weights to the stacked constraint vector.

        ``W`` is ``(n_zones, n_individuals)``; the result is the concatenation
        of ``vec(A_T W X_T)`` and ``vec(A_B W X_B)``, of length
        :attr:`~pmedm_vb.assemble.inputs.PMEDMInputs.n_constraints`.
        """
        tract = self.A_T @ (self.X_T.T @ W.T).T
        block_group = self.A_B @ (self.X_B.T @ W.T).T
        return np.concatenate([tract.ravel(order="F"), block_group.ravel(order="F")])

    def adjoint(self, v: np.ndarray) -> np.ndarray:
        """Map a stacked constraint vector back to ``(n_zones, n_individuals)``.

        Splits ``v`` into its tract and block group blocks, reshapes each
        column-major, and accumulates ``A' V X'`` over the two. Applied to
        ``lambda`` this is ``X lambda`` in the derivation's notation, laid out
        as a weight matrix.
        """
        v_t = v[: self.split].reshape(self.shape_T, order="F")
        v_b = v[self.split :].reshape(self.shape_B, order="F")
        return (self.A_T.T @ (self.X_T @ v_t.T).T) + (self.A_B.T @ (self.X_B @ v_b.T).T)


def penalty_scale(inputs: PMEDMInputs) -> float:
    """``c = n / N^2``, the factor on ``Sigma`` throughout the dual."""
    return inputs.n / inputs.N**2


def _log_weights(
    inputs: PMEDMInputs, lam: np.ndarray, op: ConstraintOperator
) -> tuple[np.ndarray, float]:
    """``log q - X lam`` and its log-sum-exp, the two pieces ``p`` and the
    objective share. ``q = 0`` gives ``-inf``, which exponentiates to 0."""
    with np.errstate(divide="ignore"):
        logits = np.log(inputs.q) - op.adjoint(lam)
    return logits, float(logsumexp(logits))


def weights_from_lambda(
    inputs: PMEDMInputs, lam: np.ndarray, op: ConstraintOperator | None = None
) -> np.ndarray:
    """Recover ``p`` from the multipliers.

    From the derivation, ``p = q * exp(-X lam) / (q' exp(-X lam))`` -- the
    softmax-like form that makes the primal solution a closed function of the
    dual variable. Computed in log space, as a softmax over every
    (zone, unit) pair; the exponent is unbounded and overflows readily at
    realistic constraint counts.

    Returns
    -------
    numpy.ndarray
        ``(n_zones, n_units)``, summing to one. Multiply by ``N`` for
        population weights.
    """
    op = op or ConstraintOperator(inputs)
    logits, total = _log_weights(inputs, lam, op)
    return np.exp(logits - total)


class DualState(NamedTuple):
    """Everything one evaluation of the dual produces, so nothing is recomputed.

    ``u`` is ``X'p``, which the gradient needs and the Hessian reuses. With a
    hierarchy, ``lam`` is the solver's ``xi`` and ``lam_data`` the multipliers
    the data see (:mod:`pmedm_vb.assemble.hierarchy`); without one they are the
    same array.
    """

    lam: np.ndarray
    p: np.ndarray
    u: np.ndarray
    objective: float
    gradient: np.ndarray
    lam_data: np.ndarray | None = None


def dual_state(
    inputs: PMEDMInputs,
    lam: np.ndarray,
    sigma: Sigma,
    op: ConstraintOperator | None = None,
    hierarchy=None,
) -> DualState:
    """Evaluate ``p``, ``X'p``, the objective and the gradient at ``lam``.

    With a non-trivial ``hierarchy`` (a :class:`~pmedm_vb.assemble.hierarchy.Hierarchy`),
    ``lam`` is ``xi``, ``sigma`` must be ``hierarchy.sigma(...)`` and the
    objective is ``f~`` of that module.
    """
    op = op or ConstraintOperator(inputs)
    c = penalty_scale(inputs)
    if hierarchy is None or hierarchy.is_trivial:
        logits, total = _log_weights(inputs, lam, op)
        p = np.exp(logits - total)
        u = op.forward(p)
        y = inputs.targets() / inputs.N
        sigma_lam = sigma.matvec(lam)
        objective = float(y @ lam + total + 0.5 * c * (lam @ sigma_lam))
        gradient = y + c * sigma_lam - u
        return DualState(lam=lam, p=p, u=u, objective=objective, gradient=gradient, lam_data=lam)
    h = hierarchy
    lam_data = h.lambda_data(lam)
    logits, total = _log_weights(inputs, lam_data, op)
    p = np.exp(logits - total)
    u = op.forward(p)
    zeta = h.zeta(lam)
    sigma_zeta = sigma.matvec(zeta)
    null = h.null(lam[: h.m])
    objective = float(h.y_ext @ zeta + total + 0.5 * c * (zeta @ sigma_zeta)
                      + 0.5 * h.kappa * (null @ null))
    gradient = h.zeta_T(h.y_ext + c * sigma_zeta) - h.lambda_data_T(u) + h.ridge_gradient(lam)
    return DualState(lam=lam, p=p, u=u, objective=objective, gradient=gradient, lam_data=lam_data)


def dual_objective(inputs: PMEDMInputs, lam: np.ndarray, sigma: Sigma) -> float:
    """Evaluate the dual objective being minimised.

    From the derivation,

    .. math::

        n^{-1} M(\\lambda) = Y'\\lambda/N + \\log(q' \\exp(-X\\lambda))
                             + 0.5 (n/N^2) \\lambda' \\Sigma \\lambda
    """
    return dual_state(inputs, lam, sigma).objective


def dual_gradient(inputs: PMEDMInputs, lam: np.ndarray, sigma: Sigma) -> np.ndarray:
    """Gradient of :func:`dual_objective`.

    ``(Y/N + (n/N^2) Sigma lam) - X'p``, which is the cheap part of MaxEnt: it
    costs one :meth:`ConstraintOperator.forward` and nothing else.
    """
    return dual_state(inputs, lam, sigma).gradient


def category_pairs(inputs: PMEDMInputs) -> tuple[np.ndarray, np.ndarray]:
    """Stacked-row indices of every (tract, block group) multiplier pair.

    For each category constrained at both levels -- matched by name between
    ``tract_constraints`` and ``bg_constraints`` -- and each block group ``b``
    in tract ``t``: the tract row ``T`` and the block group row ``B``. A unit
    placed in ``b`` has its logit shifted by ``-x_ik (lambda_T + lambda_B)``,
    so the data see their sum. ``B`` rows are unique; a tract row repeats once
    per block group in its tract. Assumes ``A_B`` is the identity, as
    :meth:`~pmedm_vb.assemble.inputs.PMEDMInputs.zone_tracts` does.
    """
    n_tracts, split = inputs.Y_T.shape[0], inputs.Y_T.size
    zone_tract = inputs.zone_tracts()
    bg_index = {name: k for k, name in enumerate(inputs.bg_constraints)}
    tract_rows, bg_rows = [], []
    for kt, name in enumerate(inputs.tract_constraints):
        kb = bg_index.get(name)
        if kb is None:
            continue
        tract_rows.append(kt * n_tracts + zone_tract)
        bg_rows.append(split + kb * inputs.n_zones + np.arange(inputs.n_zones))
    if not tract_rows:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(tract_rows).astype(int), np.concatenate(bg_rows).astype(int)
