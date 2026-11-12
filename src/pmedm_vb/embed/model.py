"""The embedding's decoder and loss, in PyTorch.

A unit's position is ``φ = [g_inst, g_noninst, z]``: two fixed 0/1
group-quarters indicators and ``d`` learned coordinates ``z``. The decoder is
linear with a softmax per target,

    p_hg = softmax(W_g z_h + b_g + O_g g_h),

so the slopes ``W`` are shared by every unit type and the indicators enter only
through the per-category intercept offsets ``O`` (one column per
group-quarters type): the agreed "shared slopes, type-specific intercepts".

The loss is the settled one,

    L = Σ_h w_h Σ_g Σ_k c_k · (−n_hk log p_hgk) / Σ_h w_h
        + λ (Σ_h w_h ‖z_h‖² / Σ_h w_h + ‖W‖²),

over the **training** (row, target) entries only: held-out entries and entries
outside a target's universe (zero counts) contribute nothing. ``c_k`` are the
rare-category weights of :func:`category_weights`. The intercepts ``b`` and
offsets ``O`` are not penalised.

Categories with no unit in them at all (a mode of transport nobody in the state
reports, say) are excluded from their softmax by a fixed large negative logit:
left in, the softmax would push their intercept towards minus infinity for
ever, and nothing could be learned about them anyway.
"""

from __future__ import annotations

import numpy as np
import torch

#: Logit given to categories with no unit in them; ``exp`` of it is zero in
#: float32, and multiplied by a zero count it adds nothing to the loss.
EMPTY_LOGIT = -1.0e4


def category_weights(
    counts: np.ndarray, weight: np.ndarray, groups: np.ndarray, gamma: float,
    train: np.ndarray | None = None,
) -> np.ndarray:
    """``c_k = π_k^(−γ)``, scaled within each target so that ``Σ_k π_k c_k = 1``.

    ``π_k`` is category ``k``'s weighted share of its target's universe,
    from the training entries when ``train`` (rows × targets, bool) is given.

    **Normalisation.** The agreed rule was "normalised to mean 1 within each
    table". It is implemented as the *share-weighted* mean, ``Σ_k π_k c_k = 1``,
    so that a target's total weight per member of its universe is the same at
    every γ: only the balance *within* the target moves. The plain mean over
    categories would instead shrink a whole target whenever one of its
    categories is very rare (a single tiny ``π_k`` makes the mean of ``1/π_k``
    huge), which is not what γ is for. At γ = 1 every category of a target then
    carries an equal share of its weight. Empty categories get ``c_k = 0``.
    """
    c = np.zeros(counts.shape[1])
    for g, (start, end) in enumerate(groups):
        block = counts[:, start:end].astype(np.float64)
        rows = weight if train is None else weight * train[:, g]
        totals = rows @ block
        if totals.sum() <= 0:
            continue
        share = totals / totals.sum()
        present = share > 0
        raw = np.zeros_like(share)
        raw[present] = share[present] ** (-gamma)
        c[start:end] = raw / (share * raw).sum()
    return c


class Decoder(torch.nn.Module):
    """Linear decoder with one softmax per target."""

    def __init__(self, n_categories: int, d: int, n_gq: int, groups: np.ndarray,
                 empty: np.ndarray):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(n_categories, d))
        self.b = torch.nn.Parameter(torch.zeros(n_categories))
        self.O = torch.nn.Parameter(torch.zeros(n_categories, n_gq))
        self.groups = [(int(s), int(e)) for s, e in groups]
        self.register_buffer("empty", torch.as_tensor(empty, dtype=torch.float32) * EMPTY_LOGIT)

    def log_probs(self, z: torch.Tensor, gq: torch.Tensor) -> torch.Tensor:
        """``(rows, K)`` log-probabilities, a log-softmax within each target."""
        logits = z @ self.W.T + self.b + gq @ self.O.T + self.empty
        return torch.cat([torch.log_softmax(logits[:, s:e], dim=1) for s, e in self.groups], dim=1)


def data_term(log_p: torch.Tensor, counts: torch.Tensor, row_weight: torch.Tensor,
              c: torch.Tensor, total_weight: float) -> torch.Tensor:
    """``Σ_h w_h Σ_k c_k (−n_hk log p_hk) / Σ_h w_h``; ``counts`` already
    has held-out and out-of-universe entries zeroed."""
    return -(row_weight[:, None] * counts * c[None, :] * log_p).sum() / total_weight


def ridge_term(z: torch.Tensor, W: torch.Tensor, row_weight: torch.Tensor,
               total_weight: float, ridge: float) -> torch.Tensor:
    return ridge * ((row_weight * (z * z).sum(1)).sum() / total_weight + (W * W).sum())


def whiten(z: np.ndarray, W: np.ndarray, b: np.ndarray, weight: np.ndarray,
           floor: float = 1e-10) -> dict:
    """Design-weighted centring and whitening of ``z``, with the decoder
    transformed so every logit is unchanged.

    ``z = z' S + μ`` with ``S = Σ^{1/2}`` (symmetric), so
    ``z W^T + b = z' (W S)^T + (b + W μ)``. Directions whose variance is below
    ``floor`` times the largest are left unscaled (dividing by a near-zero
    standard deviation would only amplify noise) and reported in
    ``eigenvalues``.
    """
    w = weight / weight.sum()
    mu = w @ z
    centred = z - mu
    cov = (centred * w[:, None]).T @ centred
    values, vectors = np.linalg.eigh(cov)
    keep = values > floor * values.max()
    root = np.where(keep, np.sqrt(np.where(keep, values, 1.0)), 1.0)
    S = (vectors * root) @ vectors.T
    S_inv = (vectors / root) @ vectors.T
    return {
        "z": centred @ S_inv,
        "W": W @ S,
        "b": b + W @ mu,
        "mean": mu,
        "sqrt_cov": S,
        "eigenvalues": values[::-1],
    }
