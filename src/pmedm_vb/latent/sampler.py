"""Exact MCMC for the realized populations given one parameter draw (step 5,
phase 1; ``PLAN_latent_step5.md`` version 1, §2–§3; model
``MODEL_latent_v5_draft.md`` §2 and §7).

**Target.** Given a parameter draw the zone totals ``N_b`` are fixed, and
(model version 5, free shares or the cut form)

    π(C | ·, Y) ∝ Π_b DirMult(C_b; N_b, κ s_b p_b) · exp{-½ e'Σ⁻¹e},
    e = Y - Σ_b D_b T0'C_b,

with ``D_b`` placing zone ``b``'s cells on its rows of the stacked layout
(row ``k*Z + b`` for cell ``k``, as :mod:`~pmedm_vb.latent.free_shares`),
``T0`` the atoms' contributions to the cells (``M × C``) and ``s_b`` the
zone's share ``N_b / Σ N``. ``κ = ∞`` gives ``Multinomial(N_b, p_b)``.

**State** (per chain): each zone's ``N_b`` individuals as atom labels, the
counts ``C_b``, the residual ``e`` and what the covariance backend needs of
``Σ⁻¹e`` (below).

**Move** (``PLAN_latent_step5.md`` §3). In zone ``b`` pick an individual
uniformly (its atom ``i``), propose ``j ~ Q(C)``, accept with

    α = min{1, (C_bj + κ s_b p_bj)/(C_bi - 1 + κ s_b p_bi) · Q_i(C')/Q_j(C)
               · exp(d'Σ⁻¹e - ½ d'Σ⁻¹d)},     d = D_b{t(x_j) - t(x_i)},

``C'`` the state after the move (``κ = ∞``: the first factor is
``p_bj / p_bi``). The state is the labels, whose target is ``π(C)`` divided
by the multinomial coefficient (individuals are exchangeable), so the count
terms ``C_bi/(C_bj + 1)`` of the Dirichlet-multinomial's ratio cancel against
the coefficient: derived here (the plan's derivation, checked here by
enumeration in ``experiments/latent/step5_gate.py``). ``j = i`` is a null
move (the state does not change).

**Proposals,** a mixture with weights ``(w_a, w_b, w_c)``:

    Q_j(C) = w_a C_bj/N_b + w_b r̂_bj + w_c p_bj,

* (a) occupancy: another individual of the zone, chosen uniformly (it may be
  the same one), and its atom. Alone it never puts an individual on an empty
  atom, and never empties an atom (the reverse move would have
  ``Q_i(C') = 0``), so alone it is *reducible*: it samples ``π`` restricted
  to the start's set of occupied atoms (derived here; the gate checks it
  against that restricted posterior). Use it inside a mixture.
* (b) the realized MAP ``r̂_b`` (fixed for the parameter draw);
* (c) the superpopulation ``p_b``.

The state-dependent part is evaluated at ``C`` and at ``C'``
(``C'_bi = C_bi - 1``). (b) and (c) are drawn by inverse CDF from cumulative
sums in float64, and the Hastings ratio uses the probabilities *as
represented by those cumulative sums* (their differences), so the ratio is
the one of the proposal actually sampled, rounding included. All cumulative
sums sit in one flat sorted array (slot ``s``'s on ``[s, s + 1]``), so the
probabilities' resolution is about ``(number of slots) × 2⁻⁵²``; an atom
whose probability is below it is never proposed by that component (the
chain stays exact, for the proposal as represented).

**Covariance backends.** ``d`` touches only zone ``b``'s rows and the
nonzero cells of ``t(x_j)`` and ``t(x_i)`` (``T0``'s rows are held padded
to their largest number of nonzeros, ``k``).

* :class:`BlockCovariance`: ``Σ`` block-diagonal by zone, each zone's
  ``P_b = Σ_b⁻¹`` held dense, with ``g_b = P_b e_b``: ``d'Σ⁻¹e = d'g_b``,
  ``d'Σ⁻¹d = t_j'P_b t_j + t_i'P_b t_i - 2 t_j'P_b t_i`` (the first two
  held for every zone and atom, the cross term a ``k × k`` gather), and on
  acceptance ``g_b ← g_b - P_b d`` (``C × 2k``). Zones are then exactly independent
  under the target (the quadratic form splits by zone), so every zone of
  every chain moves at once: the product of the zones' kernels leaves the
  product target invariant. One step moves every zone once; a *sweep* is
  ``max N_b`` steps (every zone at least ``N_b`` moves).
* :class:`LowRankCovariance`: ``Σ = D + LL'`` (rank ``R``, coupling zones).
  With ``z = L'D⁻¹e``, ``y = K⁻¹z``, ``K = I + L'D⁻¹L`` (Woodbury) and
  ``w = L'D⁻¹d`` (derived here):

      d'Σ⁻¹e = d'D⁻¹e - w'y,    d'Σ⁻¹d = d'D⁻¹d - w'K⁻¹w,
      on acceptance  e ← e - d,  z ← z - w,  y ← y - K⁻¹w,

  about ``R·nnz(d) + R²`` per move. Zones move one at a time within a
  chain: each step picks one individual uniformly from the chain's
  ``Σ_b N_b`` (so zone ``b`` with probability ``N_b / Σ N``, a random scan);
  a *sweep* is ``max_chains Σ_b N_b`` steps.

**Chains.** One chain per parameter draw, all advanced together as a batch
(torch, any ``device``; float64 throughout, integer counts and labels).
Inputs may differ by chain (``N``, ``p``, ``r̂``, ``s``); ``Y``, ``T0`` and
``Σ`` are shared. Start: a multinomial draw from ``r̂_b`` restricted to the
zone's support (atoms with ``κ s_b p_bj > 0``; ``r̂`` mass off it is dropped
for the start only), or given counts.

**Numerical care.** Log-space ratios; a proposal whose evaluated
``Q_j(C)`` is not positive, or a non-finite ``log α``, is rejected; zero
reverse probabilities give ``log α = -∞``. The running residual is
recomputed exactly from the counts after every sweep (``refresh``), and its
largest drift is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

DTYPE = torch.float64
#: The proposal components, in mixture order.
PROPOSALS = ("occupancy", "realized", "prior")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class SamplerInputs:
    """One batch of chains (module notes). ``n`` chains, ``Z`` zones, ``M``
    atoms, ``C`` cells per zone.

    ``T0``: ``M × C`` atom contributions to one zone's cells (``X["A"]``).
    ``Y``: ``(C Z,)`` targets in the stacked layout (row ``k*Z + b``).
    ``N``: ``(n, Z)`` integer zone totals. ``p``, ``r_hat``: ``(n, Z, M)`` or
    ``(Z, M)`` (shared), normalized per zone here (the engine's
    ``probabilities`` is ``M × Z``: pass its transpose; ``r̂`` from
    :func:`~pmedm_vb.latent.free_shares.solve_conditional_fixed_shares` is
    ``res.r``, ``Z × M``, rows summing to the shares). ``kappa``: ``κ``
    (``np.inf`` for the multinomial). ``shares``: ``(n, Z)``, default
    ``N / N.sum(1)``.
    """

    T0: sp.spmatrix
    Y: np.ndarray
    N: np.ndarray
    p: np.ndarray
    r_hat: np.ndarray
    kappa: float
    shares: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.T0 = sp.csr_matrix(self.T0, dtype=float)
        N = np.asarray(self.N)
        if N.ndim != 2:
            raise ValueError("N must be (chains, zones)")
        if not np.all(N == np.rint(N)):
            raise ValueError("N must be integer")
        self.N = np.rint(N).astype(np.int64)
        if np.any(self.N < 0):
            raise ValueError("N must be non-negative")
        n, Z = self.N.shape
        M, C = self.T0.shape
        self.Y = np.asarray(self.Y, float)
        if self.Y.shape != (C * Z,):
            raise ValueError(f"Y must be ({C * Z},) in the stacked layout, got {self.Y.shape}")
        self.p = _per_chain(self.p, n, Z, M, "p")
        self.r_hat = _per_chain(self.r_hat, n, Z, M, "r_hat")
        self.kappa = float(self.kappa)
        if not self.kappa > 0:
            raise ValueError("kappa must be positive (np.inf for the multinomial)")
        if self.shares is None:
            tot = self.N.sum(axis=1, keepdims=True).astype(float)
            self.shares = self.N / np.maximum(tot, 1.0)
        self.shares = np.asarray(self.shares, float)
        if self.shares.shape != (n, Z):
            raise ValueError("shares must be (chains, zones)")
        active = self.N > 0
        if np.any(active & (self.p.sum(axis=2) <= 0)):
            raise ValueError("a zone with N_b > 0 has no atom with p_bj > 0")
        if np.isfinite(self.kappa) and np.any(active & (self.shares <= 0)):
            raise ValueError("a zone with N_b > 0 has share 0, so κ s_b p_b = 0")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """``(n, Z, M, C)``."""
        n, Z = self.N.shape
        M, C = self.T0.shape
        return n, Z, M, C

    def alpha(self) -> np.ndarray:
        """``κ s_b p_b`` (``(n, Z, M)``; ``inf`` where ``p > 0`` if ``κ = ∞``)."""
        if not np.isfinite(self.kappa):
            return np.where(self.p > 0, np.inf, 0.0)
        return self.kappa * self.shares[:, :, None] * self.p


def _per_chain(x: np.ndarray, n: int, Z: int, M: int, name: str) -> np.ndarray:
    x = np.asarray(x, float)
    if x.shape == (Z, M):
        x = np.broadcast_to(x, (n, Z, M))
    if x.shape != (n, Z, M):
        raise ValueError(f"{name} must be ({n}, {Z}, {M}) or ({Z}, {M}), got {x.shape}")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    tot = x.sum(axis=2, keepdims=True)
    # C order: the sampler views these flat (torch .view(-1)), which fails on
    # e.g. the F-ordered result of p[:, keep]
    return np.ascontiguousarray(np.where(tot > 0, x / np.where(tot > 0, tot, 1.0), 0.0))


def restrict_support(p: np.ndarray, T0: sp.spmatrix, r_hat: np.ndarray | None = None,
                     G: sp.spmatrix | None = None) -> dict:
    """Drop atoms with ``p = 0`` in every chain and zone (they can never be
    occupied). ``p``, ``r_hat``: ``(..., M)``; ``T0``, ``G``: ``M × ·``.
    Returns ``{"keep", "p", "T0", "r_hat", "G"}`` restricted. Dropping
    ``r̂``'s mass off the support changes proposal (b) (it is renormalized),
    not the target."""
    p = np.asarray(p, float)
    keep = np.flatnonzero(p.reshape(-1, p.shape[-1]).max(axis=0) > 0)
    return {"keep": keep, "p": p[..., keep], "T0": sp.csr_matrix(T0)[keep],
            "r_hat": None if r_hat is None else np.asarray(r_hat, float)[..., keep],
            "G": None if G is None else sp.csr_matrix(G)[keep]}


# --------------------------------------------------------------------------
# Covariance backends
# --------------------------------------------------------------------------


class BlockCovariance:
    """``Σ`` block-diagonal by zone: ``blocks[b]`` (``C × C``) is zone
    ``b``'s block in cell order (module notes)."""

    kind = "block"

    def __init__(self, blocks: np.ndarray):
        self.blocks = np.asarray(blocks, float)
        if self.blocks.ndim != 3 or self.blocks.shape[1] != self.blocks.shape[2]:
            raise ValueError("blocks must be (Z, C, C)")
        self.Z, self.C = self.blocks.shape[:2]

    @classmethod
    def from_dense(cls, Sigma: np.ndarray, Z: int, tol: float = 0.0) -> BlockCovariance:
        """From a dense stacked ``Σ``; raises if a cross-zone entry exceeds
        ``tol`` times the largest entry."""
        Sigma = np.asarray(Sigma, float)
        C = Sigma.shape[0] // Z
        idx = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]          # (Z, C) rows
        blocks = Sigma[idx[:, :, None], idx[:, None, :]]
        rest = Sigma.copy()
        rest[idx[:, :, None], idx[:, None, :]] = 0.0
        if np.abs(rest).max() > tol * np.abs(Sigma).max():
            raise ValueError("Σ couples zones; use LowRankCovariance")
        return cls(blocks)

    @classmethod
    def from_factors(cls, sigma, Z: int) -> BlockCovariance:
        """From an object with stacked ``d`` and factor ``b`` (``Σ = diag(d) +
        bb'``), e.g. :class:`~pmedm_vb.latent.calibration.BlockSigma`; raises
        if a column of ``b`` reaches two zones."""
        d = np.asarray(sigma.d, float)
        C = d.size // Z
        b = getattr(sigma, "b", None)
        idx = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
        blocks = np.zeros((Z, C, C))
        if b is not None and np.shape(b)[1] == 0:
            b = None
        if b is not None:
            b = np.asarray(b, float)
            reach = np.stack([np.abs(b[idx[z]]).max(axis=0) > 0 for z in range(Z)])
            if np.any(reach.sum(axis=0) > 1):
                raise ValueError("a factor column couples zones; use LowRankCovariance")
        for z in range(Z):
            blocks[z] = np.diag(d[idx[z]])
            if b is not None:
                blocks[z] += b[idx[z]] @ b[idx[z]].T
        return cls(blocks)

    def dense(self) -> np.ndarray:
        Z, C = self.Z, self.C
        idx = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
        out = np.zeros((C * Z, C * Z))
        out[idx[:, :, None], idx[:, None, :]] = self.blocks
        return out


class LowRankCovariance:
    """``Σ = diag(d) + LL'`` in the stacked layout (module notes)."""

    kind = "lowrank"

    def __init__(self, d: np.ndarray, L: np.ndarray | None):
        self.d = np.asarray(d, float)
        if np.any(self.d <= 0):
            raise ValueError("d must be positive")
        self.L = np.zeros((self.d.size, 0)) if L is None else np.asarray(L, float)
        if self.L.shape[0] != self.d.size:
            raise ValueError("L must have one row per stacked row")

    @classmethod
    def from_factors(cls, sigma) -> LowRankCovariance:
        """From an object with stacked ``d`` and factor ``b``
        (:class:`~pmedm_vb.assemble.sigma.Sigma` untapered, as
        :func:`~pmedm_vb.latent.region_engine.region_sigma` gives, or
        :class:`~pmedm_vb.latent.calibration.BlockSigma`)."""
        if getattr(sigma, "is_tapered", False):
            raise ValueError("a tapered Σ is not D + LL'")
        return cls(sigma.d, getattr(sigma, "b", None))

    @property
    def rank(self) -> int:
        return self.L.shape[1]

    def dense(self) -> np.ndarray:
        return np.diag(self.d) + self.L @ self.L.T


def _zone_rows(C: int, Z: int) -> np.ndarray:
    """``(Z, C)``: stacked row of zone ``b``'s cell ``k`` (``k*Z + b``)."""
    return np.arange(C)[None, :] * Z + np.arange(Z)[:, None]


class _BlockBackend:
    """``g = P_b e_b`` per slot (module notes)."""

    def __init__(self, cov: BlockCovariance, n: int, device, T0: sp.csr_matrix, K: int,
                 chunk: int = 20000):
        P = np.empty_like(cov.blocks)
        for z, S in enumerate(cov.blocks):
            Lc = np.linalg.cholesky(0.5 * (S + S.T))
            Li = np.linalg.solve(Lc, np.eye(S.shape[0]))
            P[z] = Li.T @ Li
        self.P = torch.as_tensor(0.5 * (P + P.transpose(0, 2, 1)), dtype=DTYPE, device=device)
        self.Pflat = self.P.reshape(-1)
        self.Z, self.C, self.K = cov.Z, cov.C, K
        self.n = n
        self.g = None
        # t_a'P_b t_a for every zone and atom, so a move's d'P_b d needs only the
        # cross term t_j'P_b t_i (a K x K gather instead of 2K x 2K)
        P = 0.5 * (P + P.transpose(0, 2, 1))
        M = T0.shape[0]
        diag = np.zeros((cov.Z, M))
        for z in range(cov.Z):
            for lo in range(0, M, chunk):
                Tc = T0[lo:lo + chunk]
                diag[z, lo:lo + chunk] = np.asarray(Tc.multiply(Tc @ P[z]).sum(1)).ravel()
        self.diag = torch.as_tensor(diag.reshape(-1), dtype=DTYPE, device=device)
        self.M = M

    def refresh(self, e: torch.Tensor) -> float:
        """``g = P_b e_b`` from ``e`` (``(n Z, C)``); returns the running
        ``g``'s largest drift."""
        g = torch.einsum("nzc,zcd->nzd", e.view(self.n, self.Z, self.C), self.P)
        g = g.reshape(-1, self.C)
        drift = 0.0 if self.g is None else float((self.g - g).abs().max())
        self.g = g
        return drift

    def terms(self, pk, idx, val, e, ij):
        zone, K, C = pk["zone"], self.K, self.C
        lin = (val * self.g.gather(1, idx)).sum(1)                   # every slot moves
        # d'P d = t_j'P t_j + t_i'P t_i - 2 t_j'P t_i; val's second half is -t_i
        ix_j, ix_i = idx[:, :K], idx[:, K:]
        flat = (zone * (C * C))[:, None, None] + ix_j[:, :, None] * C + ix_i[:, None, :]
        sub = torch.take(self.Pflat, flat)                          # (S, K, K)
        cross = (torch.bmm(sub, val[:, K:, None]).squeeze(2) * val[:, :K]).sum(1)
        dg = self.diag.index_select(0, torch.cat([pk["offZM"], pk["offZM"]]) + ij)
        S = idx.shape[0]
        quad = dg[:S] + dg[S:] + 2.0 * cross
        return lin, quad, None

    def update(self, acc_slots, acc_chain, acc_zone, idx, val, cache_rows) -> None:
        A, K2 = idx.shape
        rows = self.P.view(-1, self.C).index_select(
            0, (acc_zone[:, None] * self.C + idx).reshape(-1)).view(A, K2, self.C)
        self.g[acc_slots] -= torch.bmm(val[:, None, :], rows).squeeze(1)


class _LowRankBackend:
    """``z = L'D⁻¹e``, ``y = K⁻¹z`` per chain (module notes)."""

    def __init__(self, cov: LowRankCovariance, Z: int, C: int, n: int, device):
        rows = _zone_rows(C, Z)                                     # (Z, C)
        dinv = 1.0 / cov.d[rows]                                    # (Z, C)
        L = cov.L[rows]                                             # (Z, C, R)
        R = cov.rank
        K = np.eye(R) + np.einsum("zcr,zc,zcs->rs", L, dinv, L)
        Kinv = np.linalg.inv(0.5 * (K + K.T))
        self.dinv = torch.as_tensor(dinv, dtype=DTYPE, device=device)
        self.LD = torch.as_tensor(L * dinv[:, :, None], dtype=DTYPE, device=device)
        self.Kinv = torch.as_tensor(0.5 * (Kinv + Kinv.T), dtype=DTYPE, device=device)
        self.Z, self.C, self.R, self.n = Z, C, R, n
        self.z = self.y = None

    def refresh(self, e: torch.Tensor) -> float:
        """``z``, ``y`` from ``e``; returns the running ``y``'s largest drift."""
        ez = e.reshape(self.n, self.Z, self.C)
        self.z = torch.einsum("nzc,zcr->nr", ez, self.LD)
        y = self.z @ self.Kinv
        drift = 0.0 if self.y is None or y.numel() == 0 else float((self.y - y).abs().max())
        self.y = y
        return drift

    def terms(self, pk, idx, val, e, ij):
        slots, chain, zone = pk["sl"], pk["chain"], pk["zone"]
        eg = e[slots].gather(1, idx)                                 # (S, 2K)
        S, K2 = idx.shape
        rows = (zone[:, None] * self.C + idx).reshape(-1)
        dg = self.dinv.view(-1).index_select(0, rows).view(S, K2)   # (S, 2K)
        LDg = self.LD.view(-1, self.R).index_select(0, rows).view(S, K2, self.R)
        w = torch.bmm(val[:, None, :], LDg).squeeze(1)
        lin = (val * dg * eg).sum(1) - (w * self.y[chain]).sum(1)
        dd = torch.zeros(idx.shape[0], self.C, dtype=DTYPE, device=idx.device)
        dd.scatter_add_(1, idx, val)                                 # merges repeated cells
        Kw = w @ self.Kinv
        quad = (dd * dd * self.dinv[zone]).sum(1) - (w * Kw).sum(1)
        return lin, quad, (w, Kw)

    def update(self, acc_slots, acc_chain, acc_zone, idx, val, cache_rows) -> None:
        w, Kw = cache_rows
        self.z[acc_chain] -= w
        self.y[acc_chain] -= Kw


# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------


@dataclass
class ProposalMix:
    """Mixture weights of proposals (a) occupancy, (b) realized MAP, (c)
    superpopulation; normalized. ``drop_correction`` leaves the Hastings
    factor ``Q_i(C')/Q_j(C)`` out: a deliberately *wrong* sampler, for the
    gate only."""

    occupancy: float = 1.0 / 3.0
    realized: float = 1.0 / 3.0
    prior: float = 1.0 / 3.0
    drop_correction: bool = False

    def weights(self) -> np.ndarray:
        w = np.array([self.occupancy, self.realized, self.prior], float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("proposal weights must be non-negative, not all 0")
        return w / w.sum()


@dataclass
class SamplerResult:
    """Draws and diagnostics of :meth:`RealizedSampler.run`.

    ``functionals``: ``(D, n, Z, F)`` draws of ``G'C_b`` (if ``G`` given);
    ``counts``: ``D`` draws, each ``(n, Z, M)`` int32 (``"dense"``) or a CSR
    ``(n Z, M)`` (``"sparse"``, row ``chain*Z + zone``); ``sweeps``: the sweep
    number of each draw (counted from the sampler's creation).
    ``acceptance``: per proposal, the number proposed, accepted (state
    changed), null (``j = i``) and the rates (``rate``: accepted / proposed;
    ``rate_non_null``: accepted / non-null). ``seconds_per_sweep``: wall time
    of each sweep (burn-in included, refresh included); ``steps_per_sweep``:
    block, ``max N_b`` (every zone moves at every step); low rank,
    ``max Σ_b N_b`` (one zone per chain per step, by size). ``drift``: the
    largest difference between the running and the recomputed ``e`` and
    ``g`` (or ``y``) at any refresh."""

    functionals: np.ndarray | None
    counts: list | None
    sweeps: np.ndarray
    acceptance: dict
    seconds_per_sweep: np.ndarray
    steps_per_sweep: int
    backend: str
    drift: float
    seconds: float

    def summary(self) -> dict:
        sps = self.seconds_per_sweep
        return {"backend": self.backend, "steps_per_sweep": self.steps_per_sweep,
                "sweeps": int(sps.size),
                "seconds_per_sweep": float(np.median(sps)) if sps.size else float("nan"),
                "drift": self.drift, "seconds": self.seconds,
                **{f"accept_{k}": v["rate"] for k, v in self.acceptance.items()}}


class RealizedSampler:
    """The sampler of the module notes, for a batch of chains. State
    persists across :meth:`run` calls (burn-in, then draws)."""

    def __init__(self, inputs: SamplerInputs, cov: BlockCovariance | LowRankCovariance,
                 mix: ProposalMix | None = None, *, device: str | torch.device = "cpu",
                 seed: int = 0, start_counts: np.ndarray | None = None):
        self.inputs, self.cov = inputs, cov
        self.mix = mix or ProposalMix()
        self.w = self.mix.weights()
        self.device = torch.device(device)
        dev = self.device
        n, Z, M, C = inputs.shape
        self.n, self.Z, self.M, self.C = n, Z, M, C
        S = n * Z
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed))

        # T0 rows, padded to K nonzeros
        T0 = inputs.T0
        nnz = np.diff(T0.indptr)
        K = max(int(nnz.max()) if nnz.size else 0, 1)
        idx = np.zeros((M, K), dtype=np.int64)
        val = np.zeros((M, K))
        for a in range(M):
            lo, hi = T0.indptr[a], T0.indptr[a + 1]
            idx[a, :hi - lo] = T0.indices[lo:hi]
            val[a, :hi - lo] = T0.data[lo:hi]
        self.K = K
        self.Tidx = torch.as_tensor(idx, device=dev)
        self.Tval = torch.as_tensor(val, dtype=DTYPE, device=dev)
        T0t = T0.T.tocoo()
        self.T0t = torch.sparse_coo_tensor(np.vstack([T0t.row, T0t.col]), T0t.data,
                                           (C, M), dtype=DTYPE, device=dev,
                                           check_invariants=True).coalesce()

        # per-slot (chain, zone) data, slot s = chain*Z + zone
        self.Nslot = torch.as_tensor(inputs.N.reshape(S), device=dev)
        self.Nf = self.Nslot.to(DTYPE)
        self.zone_of_slot = torch.arange(S, device=dev) % Z
        self.chain_of_slot = torch.arange(S, device=dev) // Z
        alpha = inputs.alpha().reshape(S, M)
        self.finite_kappa = bool(np.isfinite(inputs.kappa))
        if self.finite_kappa:
            self.alpha = torch.as_tensor(alpha, dtype=DTYPE, device=dev)
        else:
            with np.errstate(divide="ignore"):
                self.logp = torch.as_tensor(np.log(inputs.p.reshape(S, M)), dtype=DTYPE,
                                            device=dev)
        self.support = torch.as_tensor(alpha > 0, device=dev)
        self.cdf = {}
        if self.w[1] > 0:
            self.cdf["realized"] = self._flat_cdf(inputs.r_hat.reshape(S, M))
        if self.w[2] > 0:
            self.cdf["prior"] = self._flat_cdf(inputs.p.reshape(S, M))
        rows = _zone_rows(C, Z)
        Yz = inputs.Y[rows]                                        # (Z, C)
        self.Yslot = torch.as_tensor(np.tile(Yz, (n, 1)), dtype=DTYPE, device=dev)

        # backend
        if cov.kind == "block":
            if (cov.Z, cov.C) != (Z, C):
                raise ValueError("BlockCovariance does not match (Z, C)")
            self.backend = _BlockBackend(cov, n, dev, T0, K)
        elif cov.kind == "lowrank":
            if cov.d.size != Z * C:
                raise ValueError("LowRankCovariance does not match Z*C rows")
            self.backend = _LowRankBackend(cov, Z, C, n, dev)
        else:
            raise ValueError(f"unknown covariance kind {cov.kind!r}")

        # start
        self.Nmax = max(int(inputs.N.max()), 1)
        if start_counts is None:
            self.labels = self._start_from(inputs.r_hat.reshape(S, M), alpha > 0)
        else:
            self.labels = self._labels_from_counts(np.asarray(start_counts))
        self.counts = torch.zeros(S, M, dtype=torch.int64, device=dev)
        live = (torch.arange(self.Nmax, device=dev)[None, :] < self.Nslot[:, None])
        self.counts.scatter_add_(1, self.labels, live.to(torch.int64))
        bad = (self.counts > 0) & ~self.support
        if bool(bad.any()):
            raise ValueError("the start occupies an atom with κ s_b p_bj = 0")
        self.e = None
        self.refresh()
        self.sweep = 0
        self.stats = torch.zeros(3, 3, dtype=torch.int64, device=dev)  # comp x (rej, acc, null)
        self._sign = torch.tensor([1.0, -1.0], dtype=DTYPE, device=dev)[None, :, None]
        self._all = self._pack(torch.arange(S, device=dev))
        self.Ntot = self.Nslot.reshape(n, Z).sum(1)
        self.cumN = torch.cumsum(self.Nslot.reshape(n, Z), 1)

    # -- set-up helpers -----------------------------------------------------

    def _flat_cdf(self, q: np.ndarray) -> dict:
        """Slot ``s``'s cumulative sums on ``[s, s + 1]`` in one flat sorted
        array of ``S (M + 1)`` entries (module notes). Rows without mass get
        a uniform row (never used: such a zone has ``N_b = 0`` or proposals
        from it have ``Q = 0`` and are rejected)."""
        S, M = q.shape
        tot = q.sum(1, keepdims=True)
        q = np.where(tot > 0, q / np.where(tot > 0, tot, 1.0), 1.0 / M)
        c = np.cumsum(q, axis=1)
        c /= c[:, -1:]
        c[:, -1] = 1.0
        flat = np.concatenate([np.zeros((S, 1)), c], axis=1) + np.arange(S)[:, None]
        flat = np.maximum.accumulate(flat.ravel())                 # monotone after rounding
        last = (M - 1) - np.argmax((q > 0)[:, ::-1], axis=1)        # last atom with mass
        return {"flat": torch.as_tensor(flat, dtype=DTYPE, device=self.device),
                "last": torch.as_tensor(last, device=self.device)}

    def _start_from(self, rhat: np.ndarray, support: np.ndarray) -> torch.Tensor:
        q = np.where(support, rhat, 0.0)
        bad = q.sum(1) <= 0
        if np.any(bad):                                             # r̂ misses the support
            q[bad] = np.where(support[bad], 1.0, 0.0)
        q = q / np.maximum(q.sum(1, keepdims=True), 1e-300)
        qt = torch.as_tensor(q, dtype=DTYPE, device=self.device)
        empty = qt.sum(1) <= 0
        qt[empty, 0] = 1.0                                          # N_b = 0 zones only
        return torch.multinomial(qt, self.Nmax, replacement=True, generator=self.gen)

    def _labels_from_counts(self, counts: np.ndarray) -> torch.Tensor:
        n, Z, M = self.n, self.Z, self.M
        counts = np.asarray(counts).reshape(n * Z, M)
        if np.any(counts < 0) or not np.array_equal(counts.sum(1), self.inputs.N.reshape(-1)):
            raise ValueError("start_counts must be non-negative with zone sums N")
        labels = np.zeros((n * Z, self.Nmax), dtype=np.int64)
        for s in range(n * Z):
            lab = np.repeat(np.arange(M), counts[s])
            labels[s, :lab.size] = lab
        return torch.as_tensor(labels, device=self.device)

    # -- state ----------------------------------------------------------------

    def refresh(self) -> float:
        """Recompute ``e`` (and the backend's state: ``g``, or ``z`` and
        ``y``) exactly from the counts; returns the largest drift of the
        running values (``e``, and ``g`` or ``y``)."""
        fitted = torch.sparse.mm(self.T0t, self.counts.to(DTYPE).T).T     # (S, C)
        e = self.Yslot - fitted
        drift = 0.0 if self.e is None else float((self.e - e).abs().max())
        self.e = e
        return max(drift, self.backend.refresh(e))

    def counts_numpy(self) -> np.ndarray:
        """Current counts, ``(n, Z, M)`` int64."""
        return self.counts.reshape(self.n, self.Z, self.M).cpu().numpy()

    # -- one step -------------------------------------------------------------

    def _pack(self, sl: torch.Tensor) -> dict:
        """Per-slot quantities a step needs (cached for the block backend)."""
        Nb = self.Nslot[sl]
        return {"sl": sl, "chain": self.chain_of_slot[sl], "zone": self.zone_of_slot[sl],
                "Nf": Nb.to(DTYPE), "Nf1": Nb.clamp(min=1).to(DTYPE),
                "Nm1": (Nb - 1).clamp(min=0), "active": Nb > 0, "slf": sl.to(DTYPE),
                "offN": sl * self.Nmax, "offM": sl * self.M, "offQ": sl * (self.M + 1),
                "offZM": self.zone_of_slot[sl] * self.M}

    def _draw_flat(self, which: str, pk: dict, u: torch.Tensor) -> torch.Tensor:
        cd = self.cdf[which]
        pos = torch.searchsorted(cd["flat"], pk["slf"] + u, right=True)
        atom = pos - pk["offQ"] - 1
        # u rounding up to the slot's end (probability ~2^-53) lands past it
        return torch.minimum(atom.clamp(min=0), cd["last"].index_select(0, pk["sl"]))

    def _q_flat(self, which: str, base_i: torch.Tensor, base_j: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """The represented probabilities of atoms ``i`` and ``j`` (module notes)."""
        flat = self.cdf[which]["flat"]
        v = flat.index_select(0, torch.cat([base_i, base_i + 1, base_j, base_j + 1])).view(4, -1)
        return v[1] - v[0], v[3] - v[2]

    def _step(self, pk: dict) -> None:
        dev, w = self.device, self.w
        sl, chain, zone = pk["sl"], pk["chain"], pk["zone"]
        S = sl.shape[0]
        Nf = pk["Nf"]
        r = torch.rand(S, 6, dtype=DTYPE, device=dev, generator=self.gen)
        lab_flat = self.labels.view(-1)
        u = torch.minimum((r[:, 0] * Nf).long(), pk["Nm1"])
        i = lab_flat.index_select(0, pk["offN"] + u)
        comp = (r[:, 1] >= w[0]).long() + (r[:, 1] >= w[0] + w[1]).long()
        comp = comp.clamp(max=2)
        j = None
        for c, name in enumerate(PROPOSALS):
            if w[c] <= 0:
                continue
            if c == 0:
                cand = lab_flat.index_select(
                    0, pk["offN"] + torch.minimum((r[:, 2] * Nf).long(), pk["Nm1"]))
            else:
                cand = self._draw_flat(name, pk, r[:, 2 + c])
            j = cand if j is None else torch.where(comp == c, cand, j)

        cnt = self.counts.view(-1)
        ij = torch.cat([pk["offM"] + i, pk["offM"] + j])
        Cij = cnt.index_select(0, ij).to(DTYPE)
        Ci, Cj = Cij[:S], Cij[S:]
        # proposal probabilities: forward at C, reverse at C' (C'_i = C_i - 1)
        qi = torch.zeros(S, dtype=DTYPE, device=dev)
        qj = torch.zeros(S, dtype=DTYPE, device=dev)
        if w[0] > 0:
            qi = qi + (w[0] * (Ci - 1.0)) / pk["Nf1"]
            qj = qj + (w[0] * Cj) / pk["Nf1"]
        for c, name in ((1, "realized"), (2, "prior")):
            if w[c] > 0:
                q_i, q_j = self._q_flat(name, pk["offQ"] + i, pk["offQ"] + j)
                qi = qi + w[c] * q_i
                qj = qj + w[c] * q_j
        if self.mix.drop_correction:
            log_a = torch.zeros(S, dtype=DTYPE, device=dev)
        else:
            log_a = torch.log(qi.clamp(min=0.0)) - torch.log(qj.clamp(min=1e-300))
        # prior ratio
        if self.finite_kappa:
            a = self.alpha.view(-1).index_select(0, ij)
            log_a = log_a + torch.log(Cj + a[S:]) - torch.log(Ci - 1.0 + a[:S])
        else:
            lp = self.logp.view(-1).index_select(0, ij)
            log_a = log_a + lp[S:] - lp[:S]
        # likelihood ratio, d = t(x_j) - t(x_i) on the zone's rows
        ji = torch.stack([j, i], 1)
        ji = ji.reshape(-1)
        idx = self.Tidx.index_select(0, ji).view(S, -1)                    # (S, 2K)
        val = (self.Tval.index_select(0, ji).view(S, 2, -1) * self._sign).view(S, -1)
        lin, quad, cache = self.backend.terms(pk, idx, val, self.e, torch.cat([i, j]))
        log_a = log_a + lin - 0.5 * quad
        null = (j == i) | ~pk["active"]
        accept = (qj > 0) & ~null & (torch.log(r[:, 5]) < log_a)    # NaN, -inf: rejected
        outcome = torch.where(null, 2, accept.long())
        self.stats.view(-1).index_add_(0, comp * 3 + outcome, pk["active"].long())
        acc = accept.nonzero().squeeze(1)
        if acc.numel() == 0:
            return
        a_sl, a_i, a_j = sl[acc], i[acc], j[acc]
        a_idx, a_val = idx[acc], val[acc]
        lab_flat[pk["offN"][acc] + u[acc]] = a_j
        one = torch.ones_like(a_i)
        cnt.index_add_(0, torch.cat([ij[acc], ij[S + acc]]), torch.cat([-one, one]))
        self.e.view(-1).index_add_(0, (a_sl[:, None] * self.C + a_idx).reshape(-1),
                                   -a_val.reshape(-1))
        rows = None if cache is None else tuple(c[acc] for c in cache)
        self.backend.update(a_sl, chain[acc], zone[acc], a_idx, a_val, rows)

    def _lowrank_slots(self) -> torch.Tensor:
        """One individual uniformly from each chain's ``Σ_b N_b``: its slot."""
        r = torch.rand(self.n, dtype=DTYPE, device=self.device, generator=self.gen)
        g = torch.minimum((r * self.Ntot.to(DTYPE)).long(), (self.Ntot - 1).clamp(min=0))
        zone = torch.searchsorted(self.cumN, g[:, None], right=True).squeeze(1)
        zone = zone.clamp(max=self.Z - 1)
        return torch.arange(self.n, device=self.device) * self.Z + zone

    @property
    def steps_per_sweep(self) -> int:
        if self.cov.kind == "block":
            return int(self.inputs.N.max())
        return int(self.inputs.N.sum(1).max())

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_sweeps(self, sweeps: int) -> tuple[np.ndarray, float]:
        """Advance ``sweeps`` sweeps, refreshing after each; returns the
        seconds of each sweep and the largest drift."""
        steps = self.steps_per_sweep
        secs, drift = np.zeros(sweeps), 0.0
        block = self.cov.kind == "block"
        for k in range(sweeps):
            self._sync()
            t0 = time.perf_counter()
            for _ in range(steps):
                self._step(self._all if block else self._pack(self._lowrank_slots()))
            drift = max(drift, self.refresh())
            self._sync()
            secs[k] = time.perf_counter() - t0
            self.sweep += 1
        return secs, drift

    def time_steps(self, steps: int) -> float:
        """Seconds per step over ``steps`` steps (no refresh), for timing
        large problems without running whole sweeps."""
        block = self.cov.kind == "block"
        self._sync()
        t0 = time.perf_counter()
        for _ in range(steps):
            self._step(self._all if block else self._pack(self._lowrank_slots()))
        self._sync()
        return (time.perf_counter() - t0) / max(steps, 1)

    def acceptance(self) -> dict:
        st = self.stats.cpu().numpy()
        out = {}
        for c, name in enumerate(PROPOSALS):
            prop = int(st[c].sum())
            out[name] = {"proposed": prop, "accepted": int(st[c, 1]), "null": int(st[c, 2]),
                         "rate": st[c, 1] / prop if prop else float("nan"),
                         "rate_non_null": (st[c, 1] / (prop - st[c, 2])
                                           if prop > st[c, 2] else float("nan"))}
        return out

    def run(self, sweeps: int, *, burn: int = 0, thin: int = 1, G: sp.spmatrix | None = None,
            keep_counts: str | None = None) -> SamplerResult:
        """``burn`` sweeps, then ``sweeps`` sweeps keeping every ``thin``-th:
        functionals ``G'C_b`` (``G``: ``M × F``) and/or counts
        (``keep_counts``: ``"dense"`` or ``"sparse"``). Acceptance counts
        cover this call only."""
        t_start = time.perf_counter()
        if keep_counts not in (None, "dense", "sparse"):
            raise ValueError("keep_counts must be None, 'dense' or 'sparse'")
        Gt = None
        if G is not None:
            Gc = sp.coo_matrix(G, dtype=float)
            if Gc.shape[0] != self.M:
                raise ValueError("G must be M x F")
            Gt = torch.sparse_coo_tensor(np.vstack([Gc.col, Gc.row]), Gc.data,
                                         (Gc.shape[1], self.M), dtype=DTYPE,
                                         device=self.device, check_invariants=True).coalesce()
        secs, drift = [], 0.0
        if burn:
            s, d = self.run_sweeps(burn)
            secs.append(s)
            drift = max(drift, d)
        self.stats.zero_()
        fun, cnts, at = [], [], []
        for k in range(sweeps):
            s, d = self.run_sweeps(1)
            secs.append(s)
            drift = max(drift, d)
            if (k + 1) % thin:
                continue
            at.append(self.sweep)
            if Gt is not None:
                v = torch.sparse.mm(Gt, self.counts.to(DTYPE).T).T
                fun.append(v.reshape(self.n, self.Z, -1).cpu().numpy())
            if keep_counts == "dense":
                cnts.append(self.counts.reshape(self.n, self.Z, self.M).to(torch.int32)
                            .cpu().numpy())
            elif keep_counts == "sparse":
                cnts.append(sp.csr_matrix(self.counts.cpu().numpy().astype(np.int32)))
        return SamplerResult(
            functionals=np.stack(fun) if fun else None, counts=cnts or None,
            sweeps=np.asarray(at, dtype=int), acceptance=self.acceptance(),
            seconds_per_sweep=np.concatenate(secs) if secs else np.zeros(0),
            steps_per_sweep=self.steps_per_sweep, backend=self.cov.kind, drift=drift,
            seconds=time.perf_counter() - t_start)


def sample_counts(inputs: SamplerInputs, cov: BlockCovariance | LowRankCovariance,
                  mix: ProposalMix | None = None, *, sweeps: int, burn: int = 0,
                  thin: int = 1, G: sp.spmatrix | None = None, keep_counts: str | None = None,
                  device: str | torch.device = "cpu", seed: int = 0,
                  start_counts: np.ndarray | None = None) -> SamplerResult:
    """One call: build a :class:`RealizedSampler` and :meth:`~RealizedSampler.run` it."""
    sampler = RealizedSampler(inputs, cov, mix, device=device, seed=seed,
                              start_counts=start_counts)
    return sampler.run(sweeps, burn=burn, thin=thin, G=G, keep_counts=keep_counts)


__all__ = ["BlockCovariance", "LowRankCovariance", "PROPOSALS", "ProposalMix",
           "RealizedSampler", "SamplerInputs", "SamplerResult", "restrict_support",
           "sample_counts"]
