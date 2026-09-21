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

import numpy as np

from pmedm_vb.assemble.inputs import PMEDMInputs


class ConstraintOperator:
    """Applies ``X_tilde'`` and its adjoint via the stored factors.

    Parameters
    ----------
    inputs:
        Supplies ``A_T``, ``X_T``, ``A_B``, ``X_B`` and the shapes.
    """

    def __init__(self, inputs: PMEDMInputs) -> None:
        raise NotImplementedError

    def forward(self, W: np.ndarray) -> np.ndarray:
        """Map weights to the stacked constraint vector.

        ``W`` is ``(n_zones, n_individuals)``; the result is the concatenation
        of ``vec(A_T W X_T)`` and ``vec(A_B W X_B)``, of length
        :attr:`~pmedm_vb.assemble.inputs.PMEDMInputs.n_constraints`.
        """
        raise NotImplementedError

    def adjoint(self, v: np.ndarray) -> np.ndarray:
        """Map a stacked constraint vector back to ``(n_zones, n_individuals)``.

        Splits ``v`` into its tract and block group blocks, reshapes each
        column-major, and accumulates ``A' V X'`` over the two.
        """
        raise NotImplementedError


def weights_from_lambda(inputs: PMEDMInputs, lam: np.ndarray) -> np.ndarray:
    """Recover ``p`` from the multipliers.

    From the derivation, ``p = q * exp(-X lam) / (q' exp(-X lam))`` -- the
    softmax-like form that makes the primal solution a closed function of the
    dual variable. Must be computed in log space; the exponent is unbounded and
    overflows readily at realistic constraint counts.
    """
    raise NotImplementedError


def dual_objective(inputs: PMEDMInputs, lam: np.ndarray) -> float:
    """Evaluate the dual objective being minimised.

    From the derivation,

    .. math::

        n^{-1} M(\\lambda) = Y'\\lambda/N + \\log(q' \\exp(-X\\lambda))
                             + 0.5 (n/N^2) \\lambda' \\Sigma \\lambda
    """
    raise NotImplementedError


def dual_gradient(inputs: PMEDMInputs, lam: np.ndarray) -> np.ndarray:
    """Gradient of :func:`dual_objective`.

    ``(Y/N + (n/N^2) Sigma lam) - X'p``, which is the cheap part of MaxEnt: it
    costs one :meth:`ConstraintOperator.forward` and nothing else.
    """
    raise NotImplementedError
