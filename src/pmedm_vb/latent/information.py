"""The information diagnostic: do the tables identify ``u_b``? (step 2,
``PLAN_latent.md`` finding 1; ``PLAN_latent_step2.md`` §3, §4.3).

At ``u = 0`` every block group of the PUMA has ``p_b = q^P``. With
``C_b = Cov_{p_b}(t, h)`` (``J × K``), the Jacobian of ``m`` with respect to
``u_b`` is ``N_b D_b C_b``, and the mean part of the Fisher information is
(derived here, §3)

    I_bb' = N_b N_b' C_b' D_b' S_t⁻¹ D_b' C_b'   (b, b' in tract t; zero otherwise).

It omits the information from ``S``'s dependence on ``u``, so it is a lower
bound on the full Fisher information.

* :func:`tract_information`: ``I_t`` jointly over each tract's block groups
  (its diagonal blocks are the ``I_bb``), from the Cholesky factors of
  :class:`~pmedm_vb.latent.collapsed.Evaluation`.
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


def tract_information(engine: CollapsedLikelihood, ev: Evaluation,
                      C: list[np.ndarray]) -> list[np.ndarray]:
    """``I_t`` (``n_t K × n_t K``) per tract, block groups in tract order."""
    K, N, src = engine.K, engine.N, engine.src
    out = []
    for tract, factor in zip(engine.tracts, ev.factors):
        members = tract["members"]
        jac = np.zeros((tract["rows"].size, members.size * K))
        for j, b in enumerate(members):
            jac[tract["local"][j], j * K:(j + 1) * K] = N[b] * C[b][src]
        info = jac.T @ cho_solve(factor, jac, check_finite=False)
        out.append(0.5 * (info + info.T))
    return out


def eigenvalues(I: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (descending, clipped at 0) and eigenvectors."""
    values, vectors = np.linalg.eigh(I)
    order = np.argsort(values)[::-1]
    return np.maximum(values[order], 0.0), vectors[:, order]


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
