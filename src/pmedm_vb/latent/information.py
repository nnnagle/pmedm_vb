"""The information diagnostic: do the tables identify ``u_b``? (step 2,
``PLAN_latent.md`` finding 1; ``PLAN_latent_step2.md`` version 2, §3, §4.3).

At ``u = 0`` every block group of the PUMA has ``p_b = q^P``. With
``C_b = Cov_{p_b}(t, h)`` (``J × K``), the Jacobian of ``m`` with respect to
``u_b`` is ``N_b D_b C_b``, and the mean part of the Fisher information is
(derived here, §3)

    I_bb' = N_b N_b' C_b' D_b' [S⁻¹]_{tt'} D_b' C_b'   (t, t' the tracts of b, b').

It omits the information from ``S``'s dependence on ``u``, so it is a lower
bound on the full Fisher information. With ``S = B + LL'``
(:mod:`pmedm_vb.latent.collapsed`) and ``J_t`` the Jacobian of tract ``t``'s
rows with respect to its block groups' ``u_b``,

    I = blockdiag_t(J_t' B_t⁻¹ J_t) - Z' K⁻¹ Z,     Z = [X_t' J_t]_t   (80 × BK),

so block groups in different tracts share information through the low-rank
term.

* :func:`puma_information`: the full ``I`` (``BK × BK``, block groups in the
  area's order); its tract blocks are ``I_t`` and its diagonal blocks the
  ``I_bb``.
* :func:`degrees_of_freedom`: ``df(τ) = tr[I (I + τ⁻² I)⁻¹] = Σ_k λ_k / (λ_k + τ⁻²)``.
* :func:`share_gradient` and :func:`variance_removed`: for a share
  ``f = E_{p_b}[g] / E_{p_b}[n]`` (``g`` a count per unit, ``n`` its
  denominator), ``a = ∂f/∂u_b = Cov_{p_b}(g - f n, h) / E_{p_b}[n]``; the
  share of its prior variance removed by the tables is
  ``1 - a'(I_bb + τ⁻² I)⁻¹ a / (τ² a'a)``, independent of ``a``'s scale.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve

from pmedm_vb.latent.collapsed import CollapsedLikelihood, Evaluation


def block_covariances(engine: CollapsedLikelihood, ev: Evaluation, theta: np.ndarray
                      ) -> list[np.ndarray]:
    """``C_b = Cov_{p_b}(t0, h)`` on the base cells (``J0 × K``) per block
    group, computed once per distinct ``θ_b`` (``theta``: ``B × K``)."""
    pairs, H = engine.pairs, engine.H
    _, first, which = np.unique(theta, axis=0, return_index=True, return_inverse=True)
    distinct = []
    for b in first:
        p = ev.p[:, b]
        hbar = H.T @ p
        C = pairs.TuT @ pairs.aggregate(p[:, None] * H) - np.outer(ev.mu[:, b], hbar)
        distinct.append(C)
    return [distinct[k] for k in np.asarray(which).ravel()]


def columns_of(block_groups: np.ndarray, K: int) -> np.ndarray:
    """Positions in ``I`` of these block groups' ``u_b``."""
    return (np.asarray(block_groups)[:, None] * K + np.arange(K)[None, :]).ravel()


def puma_information(engine: CollapsedLikelihood, ev: Evaluation,
                     C: list[np.ndarray]) -> np.ndarray:
    """The full ``I`` (``BK × BK``) for the PUMA (module notes)."""
    K, N, src = engine.K, engine.N, engine.src
    size = engine.area.B * K
    info = np.zeros((size, size))
    Z = None if ev.K_factor is None else np.zeros((ev.K_factor[0].shape[0], size))
    for tract, factor, Xt in zip(engine.tracts, ev.factors, ev.X):
        members = tract["members"]
        jac = np.zeros((tract["rows"].size, members.size * K))
        for j, b in enumerate(members):
            jac[tract["local"][j], j * K:(j + 1) * K] = N[b] * C[b][src]
        cols = columns_of(members, K)
        info[np.ix_(cols, cols)] = jac.T @ cho_solve(factor, jac, check_finite=False)
        if Z is not None:
            Z[:, cols] = Xt.T @ jac
    if Z is not None:
        info -= Z.T @ cho_solve(ev.K_factor, Z, check_finite=False)
    info += info.T
    info *= 0.5
    return info


def eigenvalues(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending, clipped at 0) and eigenvectors."""
    values, vectors = np.linalg.eigh(I)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


def joint_eigenvalues(I: np.ndarray) -> np.ndarray:
    """Eigenvalues only (descending, clipped at 0), for a large ``I``."""
    return np.maximum(np.linalg.eigvalsh(I)[::-1], 0.0)


def degrees_of_freedom(values: np.ndarray, tau: float) -> float:
    """``Σ_k λ_k / (λ_k + τ⁻²)``."""
    return float((values / (values + tau ** -2)).sum())


def share_gradient(g: np.ndarray, n: np.ndarray, p: np.ndarray, H: np.ndarray
                   ) -> tuple[np.ndarray, float]:
    """``a = ∂f/∂u`` of the share ``f = E_p[g] / E_p[n]`` at ``p``, and ``f``."""
    En = float(p @ n)
    f = float(p @ g) / En
    x = g - f * n
    a = (H.T @ (p * x) - (H.T @ p) * float(p @ x)) / En
    return a, f


def variance_removed(values: np.ndarray, vectors: np.ndarray, a: np.ndarray,
                     tau: float) -> float:
    """``1 - a'(I + τ⁻² I)⁻¹ a / (τ² a'a)`` from ``I``'s eigendecomposition."""
    aa = float(a @ a)
    if aa <= 0:
        return float("nan")
    proj = vectors.T @ a
    return float(1.0 - (proj ** 2 / (values + tau ** -2)).sum() / (tau ** 2 * aa))
