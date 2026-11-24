"""Exact MCMC for the realized populations given one parameter draw, compiled
with numba (step 5, phase 2; ``PLAN_latent_step5.md`` version 1, §3–§4).

The sampler of :mod:`pmedm_vb.latent.sampler` (its module notes state the
target, the move, the acceptance ratio, the proposals and both covariance
backends), with the move loop in ``numba.njit`` kernels on numpy arrays
instead of batched torch operations. Same inputs (:class:`SamplerInputs`,
:class:`BlockCovariance`, :class:`LowRankCovariance`, :class:`ProposalMix`,
reused from there), same target, same acceptance ratio

    α = min{1, (C_bj + κ s_b p_bj)/(C_bi - 1 + κ s_b p_bi) · Q_i(C')/Q_j(C)
               · exp(d'Σ⁻¹e - ½ d'Σ⁻¹d)},      d = D_b{t(x_j) - t(x_i)},

same mixture ``Q_j(C) = w_a C_bj/N_b + w_b r̂_bj + w_c p_bj`` evaluated
exactly at ``C`` and ``C'`` (``C'_bi = C_bi - 1``), same outputs as
:meth:`RealizedSampler.run <pmedm_vb.latent.sampler.RealizedSampler.run>`
(:class:`NumbaSamplerResult` extends its result with a per-draw trace of the
log target and a zone-totals check).

**State** (per chain): each zone's labels (``int32``, one per individual),
the counts (``int64``, ``(n Z, M)``, row ``chain*Z + zone``: a *slot*), the
residual ``e`` (``(n Z, C)``), and ``g_b = P_b e_b`` (block backend,
``P_b = Σ_b⁻¹``) or ``z = L'D⁻¹e``, ``y = K⁻¹z`` (low-rank backend).

**Randomness exactly as represented** (derived here). Every random choice is
made from 53-bit integers ``k`` uniform on ``{0, …, 2⁵³ - 1}`` (the top bits
of a 64-bit generator, below):

* the individual (and, for proposal (a), the other individual): a uniform
  integer below ``N_b`` by rejection (``k`` is redrawn while
  ``k ≥ ⌊2⁵³/N_b⌋ N_b``), so each individual has probability ``1/N_b``
  exactly;
* the component: thresholds ``T_a = round(w_a 2⁵³)``,
  ``T_b = round((w_a + w_b) 2⁵³)``; the weights used in ``Q`` are the
  thresholds' differences divided by ``2⁵³``, i.e. the probabilities with
  which the components are actually chosen;
* proposals (b) and (c) by inverse CDF on *integer* cumulative sums
  ``K_0 = 0 ≤ K_1 ≤ … ≤ K_M = 2⁵³`` (``K_j = round(2⁵³ Σ_{l<j} q_l)``,
  made monotone), atom ``j`` when ``K_j ≤ k < K_{j+1}`` (binary search).
  Atom ``j``'s probability is ``(K_{j+1} - K_j)/2⁵³``, exactly representable
  in float64, and that value is what enters ``Q_j``: the Hastings ratio is the
  one of the proposal actually sampled, with no rounding gap (the torch
  sampler's float cumulative sums are exact to about ``2⁻⁵²`` times the
  number of slots). An atom with ``q < 2⁻⁵⁴`` gets width 0 and is never
  proposed by that component, which leaves the chain exact.

**Generator and seeding.** Each *stream* has its own xoshiro256** generator
(Blackman and Vigna's, 256 bits of state, implemented here in numba).
Stream ``k``'s state is ``numpy.random.SeedSequence(seed, spawn_key=(1,
k)).generate_state(4, uint64)``; the start (multinomial draws) uses
``numpy.random.default_rng(SeedSequence(seed, spawn_key=(0,)))``. Streams
are bound to work items, not to threads: block backend, one stream per slot
(``chain*Z + zone``); low-rank backend, one per chain. A run is therefore
reproducible for a given ``seed`` whatever the number of threads and however
``prange`` schedules the work. (numba's own ``np.random`` inside ``prange``
is per *thread*, so its streams would depend on the scheduling; it is not
used.)

**Parallelism and sweeps.**

* Block backend: zones are exactly independent under the target, so every
  slot (every zone of every chain) runs in parallel (``prange`` over slots,
  dynamic scheduling with chunks of one slot). A *sweep* is ``N_b`` moves in
  zone ``b`` (each individual once on average; the torch sampler instead
  moves every zone ``max N_b`` times per sweep).
* Low-rank backend: zones are coupled within a chain, so a chain's moves are
  sequential (random scan: each move picks one individual uniformly from the
  chain's ``Σ_b N_b``, as the torch sampler); chains run in parallel. A
  *sweep* is the chain's ``Σ_b N_b`` moves.

``steps_per_sweep`` in the result is the moves per chain per sweep (the
largest over chains), for both backends. Threads: ``threads`` (or numba's
default, ``NUMBA_NUM_THREADS``), set with ``numba.set_num_threads``.

**Per move** (block): ``d'g_b`` (``2k`` reads; ``k`` the largest number of
nonzero cells of an atom), ``d'P_b d = t_j'P_b t_j + t_i'P_b t_i -
2 t_j'P_b t_i`` (the first two held for every zone and atom, the cross term
``k²``), and on acceptance ``g_b ← g_b - P_b d`` (``2k C``). Low rank:
``w = L'D⁻¹d`` (``2k R``), ``d'D⁻¹d`` by merging the two atoms' sorted cell
lists, ``K⁻¹w`` (``R²``); on acceptance ``e``, ``z``, ``y`` updated
(``PLAN_latent_step5.md`` §3).

**Refresh** after every sweep: ``e`` recomputed from the counts (only the
occupied atoms are visited), then ``g`` (or ``z``, ``y``); the largest
difference from the running values is the reported drift.

**Log target** (:meth:`NumbaSampler.log_target`): per chain,
``Σ_b log DirMult(C_b; N_b, κ s_b p_b) - ½ e'Σ⁻¹e`` (``κ = ∞``: the
multinomial), i.e. the log posterior of the *counts* up to the Gaussian's
normalizing constant; :func:`log_target_numpy` is a dense numpy version for
checks.

numba is imported when this module is imported if it is installed; without
it the module still imports (so no other module needs numba), and
constructing a :class:`NumbaSampler` raises ``ImportError``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

from pmedm_vb.latent.sampler import (
    PROPOSALS,
    BlockCovariance,
    LowRankCovariance,
    ProposalMix,
    SamplerInputs,
    SamplerResult,
    _zone_rows,
    restrict_support,
)

try:                                                     # optional dependency
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:                                      # pragma: no cover
    numba = None
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        """Stand-in decorator: the kernels exist but raise if called."""
        def wrap(fn):
            def missing(*a, **k):
                raise ImportError("numba is not installed: the compiled sampler needs it")
            missing.__name__ = fn.__name__
            missing.__doc__ = fn.__doc__
            return missing
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return wrap(args[0])
        return wrap

#: 2⁵³, the resolution of every random choice (module notes).
TWO53 = 1 << 53
_INV53 = 1.0 / float(TWO53)


# --------------------------------------------------------------------------
# Generator: xoshiro256** (Blackman & Vigna), one stream per row of `state`
# --------------------------------------------------------------------------


@njit(inline="always", cache=True)
def _rotl(x, k):
    return (x << k) | (x >> (np.uint64(64) - k))


@njit(inline="always", cache=True)
def _next_u64(state, r):
    s0 = state[r, 0]
    s1 = state[r, 1]
    s2 = state[r, 2]
    s3 = state[r, 3]
    result = _rotl(s1 * np.uint64(5), np.uint64(7)) * np.uint64(9)
    t = s1 << np.uint64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, np.uint64(45))
    state[r, 0] = s0
    state[r, 1] = s1
    state[r, 2] = s2
    state[r, 3] = s3
    return result


@njit(inline="always", cache=True)
def _k53(state, r):
    """A uniform integer on ``{0, …, 2⁵³ - 1}``."""
    return np.int64(_next_u64(state, r) >> np.uint64(11))


@njit(inline="always", cache=True)
def _below(state, r, n):
    """A uniform integer on ``{0, …, n - 1}`` (``1 ≤ n ≤ 2⁵³``), exactly, by
    rejection."""
    limit = (TWO53 // n) * n
    while True:
        k = _k53(state, r)
        if k < limit:
            return k % n


@njit(inline="always", cache=True)
def _uniform_open(state, r):
    """A uniform on ``(0, 1)`` (never 0, so its log is finite)."""
    return (np.float64(_k53(state, r)) + 0.5) * _INV53


@njit(inline="always", cache=True)
def _draw_cdf(Kc, s, k):
    """Atom ``j`` with ``Kc[s, j] ≤ k < Kc[s, j + 1]`` (``Kc[s, 0] = 0``,
    ``Kc[s, M] = 2⁵³``)."""
    lo = 0
    hi = Kc.shape[1] - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if Kc[s, mid] <= k:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _draw_raw(state, n_draws):
    """Raw 64-bit outputs of stream 0 (for the generator's tests)."""
    out = np.empty(n_draws, dtype=np.uint64)
    for t in range(n_draws):
        out[t] = _next_u64(state, 0)
    return out


@njit(cache=True)
def _draw_below(state, n_draws, n):
    """``_below(n)`` draws of stream 0 (for the tests)."""
    out = np.empty(n_draws, dtype=np.int64)
    for t in range(n_draws):
        out[t] = _below(state, 0, n)
    return out


@njit(cache=True)
def _draw_uniform(state, n_draws):
    """``_uniform_open`` draws of stream 0 (for the tests)."""
    out = np.empty(n_draws)
    for t in range(n_draws):
        out[t] = _uniform_open(state, 0)
    return out


def stream_states(seed: int, n_streams: int) -> np.ndarray:
    """``(n_streams, 4)`` uint64 xoshiro256** states (module notes)."""
    out = np.empty((n_streams, 4), dtype=np.uint64)
    for k in range(n_streams):
        st = np.random.SeedSequence(int(seed), spawn_key=(1, k)).generate_state(4, np.uint64)
        if not st.any():                                 # the one invalid state
            st[0] = np.uint64(1)
        out[k] = st
    return out


def xoshiro_reference(state: np.ndarray, n: int) -> list[int]:
    """Pure-Python xoshiro256** (for checking the kernel)."""
    mask = (1 << 64) - 1
    s = [int(v) for v in state]

    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & mask

    out = []
    for _ in range(n):
        out.append((rotl((s[1] * 5) & mask, 7) * 9) & mask)
        t = (s[1] << 17) & mask
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl(s[3], 45)
    return out


# --------------------------------------------------------------------------
# Move kernels
# --------------------------------------------------------------------------


@njit(inline="always", cache=True)
def _propose(s, N, labels, Kb, Kc, thr0, thr1, rng, r):
    """Component and proposed atom (RNG stream ``r``)."""
    k = _k53(rng, r)
    if k < thr0:
        return 0, labels[s, _below(rng, r, N)]
    if k < thr1:
        return 1, _draw_cdf(Kb, s, _k53(rng, r))
    return 2, _draw_cdf(Kc, s, _k53(rng, r))


@njit(inline="always", cache=True)
def _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop):
    """Log of the prior ratio times the Hastings factor; ``-inf`` for a move
    that must be rejected (``Q_j(C) ≤ 0``, an atom off the support, or a zero
    reverse probability)."""
    w0 = weff[0]
    w1 = weff[1]
    w2 = weff[2]
    qi = 0.0
    qj = 0.0
    if w0 > 0.0:
        qi += w0 * (Ci - 1.0) * invN
        qj += w0 * Cj * invN
    if w1 > 0.0:
        qi += w1 * (Kb[s, i + 1] - Kb[s, i]) * _INV53
        qj += w1 * (Kb[s, j + 1] - Kb[s, j]) * _INV53
    if w2 > 0.0:
        qi += w2 * (Kc[s, i + 1] - Kc[s, i]) * _INV53
        qj += w2 * (Kc[s, j + 1] - Kc[s, j]) * _INV53
    if not qj > 0.0:
        return -np.inf
    la = 0.0
    if not drop:
        if not qi > 0.0:
            return -np.inf
        la = math.log(qi) - math.log(qj)
    if finite:
        num = Cj + prior[s, j]
        if not num > 0.0:
            return -np.inf
        la += math.log(num) - math.log(Ci - 1.0 + prior[s, i])
    else:
        lpj = prior[s, j]
        if lpj == -np.inf:
            return -np.inf
        la += lpj - prior[s, i]
    return la


@njit(parallel=True, cache=True)
def _block_moves(moves, labels, counts, Nslot, zone_of, e, g, P, diagq, Tidx, Tval, Tnnz,
                 prior, finite, Kb, Kc, thr0, thr1, weff, drop, rng, stats):
    """``moves[s]`` moves in every slot ``s`` (block backend; stream ``s``)."""
    S = moves.shape[0]
    C = e.shape[1]
    for s in prange(S):
        N = Nslot[s]
        if N <= 0:
            continue
        z = zone_of[s]
        invN = 1.0 / N
        for _ in range(moves[s]):
            u = _below(rng, s, N)
            i = labels[s, u]
            comp, j = _propose(s, N, labels, Kb, Kc, thr0, thr1, rng, s)
            if j == i:
                stats[s, comp, 2] += 1
                continue
            Ci = np.float64(counts[s, i])
            Cj = np.float64(counts[s, j])
            la = _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop)
            if la == -np.inf:
                stats[s, comp, 0] += 1
                continue
            nj = Tnnz[j]
            ni = Tnnz[i]
            lin = 0.0
            for a in range(nj):
                lin += Tval[j, a] * g[s, Tidx[j, a]]
            for a in range(ni):
                lin -= Tval[i, a] * g[s, Tidx[i, a]]
            cross = 0.0
            for a in range(nj):
                ca = Tidx[j, a]
                acc = 0.0
                for b in range(ni):
                    acc += P[z, ca, Tidx[i, b]] * Tval[i, b]
                cross += Tval[j, a] * acc
            quad = diagq[z, j] + diagq[z, i] - 2.0 * cross
            la += lin - 0.5 * quad
            if not math.log(_uniform_open(rng, s)) < la:      # NaN: rejected
                stats[s, comp, 0] += 1
                continue
            stats[s, comp, 1] += 1
            labels[s, u] = j
            counts[s, i] -= 1
            counts[s, j] += 1
            for a in range(nj):
                c = Tidx[j, a]
                v = Tval[j, a]
                e[s, c] -= v
                for k in range(C):
                    g[s, k] -= v * P[z, c, k]
            for a in range(ni):
                c = Tidx[i, a]
                v = Tval[i, a]
                e[s, c] += v
                for k in range(C):
                    g[s, k] += v * P[z, c, k]


@njit(parallel=True, cache=True)
def _lowrank_moves(moves, labels, counts, Nslot, cumN, Ntot, Z, e, zr, yr, dinv, LD, Kinv,
                   Tidx, Tval, Tnnz, prior, finite, Kb, Kc, thr0, thr1, weff, drop, rng, stats):
    """``moves[c]`` random-scan moves in every chain ``c`` (low-rank backend;
    stream ``c``)."""
    n = moves.shape[0]
    R = zr.shape[1]
    for c in prange(n):
        if Ntot[c] <= 0:
            continue
        w = np.empty(R)
        Kw = np.empty(R)
        for _ in range(moves[c]):
            gidx = _below(rng, c, Ntot[c])
            b = 0
            while cumN[c, b] <= gidx:
                b += 1
            u = gidx - (cumN[c, b - 1] if b > 0 else 0)
            s = c * Z + b
            N = Nslot[s]
            invN = 1.0 / N
            i = labels[s, u]
            comp, j = _propose(s, N, labels, Kb, Kc, thr0, thr1, rng, c)
            if j == i:
                stats[s, comp, 2] += 1
                continue
            Ci = np.float64(counts[s, i])
            Cj = np.float64(counts[s, j])
            la = _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop)
            if la == -np.inf:
                stats[s, comp, 0] += 1
                continue
            nj = Tnnz[j]
            ni = Tnnz[i]
            for r in range(R):
                w[r] = 0.0
            lin = 0.0
            for a in range(nj):
                cc = Tidx[j, a]
                v = Tval[j, a]
                lin += v * dinv[b, cc] * e[s, cc]
                for r in range(R):
                    w[r] += v * LD[b, cc, r]
            for a in range(ni):
                cc = Tidx[i, a]
                v = Tval[i, a]
                lin -= v * dinv[b, cc] * e[s, cc]
                for r in range(R):
                    w[r] -= v * LD[b, cc, r]
            # d'D⁻¹d: merge the two sorted cell lists (cells shared by i and j)
            dd = 0.0
            pa = 0
            pb = 0
            while pa < nj or pb < ni:
                if pb >= ni or (pa < nj and Tidx[j, pa] < Tidx[i, pb]):
                    v = Tval[j, pa]
                    dd += v * v * dinv[b, Tidx[j, pa]]
                    pa += 1
                elif pa >= nj or Tidx[i, pb] < Tidx[j, pa]:
                    v = Tval[i, pb]
                    dd += v * v * dinv[b, Tidx[i, pb]]
                    pb += 1
                else:
                    v = Tval[j, pa] - Tval[i, pb]
                    dd += v * v * dinv[b, Tidx[j, pa]]
                    pa += 1
                    pb += 1
            wy = 0.0
            wKw = 0.0
            for r in range(R):
                acc = 0.0
                for q in range(R):
                    acc += Kinv[r, q] * w[q]
                Kw[r] = acc
                wy += w[r] * yr[c, r]
                wKw += w[r] * acc
            lin -= wy
            quad = dd - wKw
            la += lin - 0.5 * quad
            if not math.log(_uniform_open(rng, c)) < la:
                stats[s, comp, 0] += 1
                continue
            stats[s, comp, 1] += 1
            labels[s, u] = j
            counts[s, i] -= 1
            counts[s, j] += 1
            for a in range(nj):
                e[s, Tidx[j, a]] -= Tval[j, a]
            for a in range(ni):
                e[s, Tidx[i, a]] += Tval[i, a]
            for r in range(R):
                zr[c, r] -= w[r]
                yr[c, r] -= Kw[r]


# --------------------------------------------------------------------------
# Set-up, refresh and summary kernels
# --------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _counts_times_csr(counts, indptr, indices, data, F):
    """``(S, F)``: ``counts @ A`` for a CSR ``A`` (``M × F``), visiting only
    occupied atoms."""
    S, M = counts.shape
    out = np.zeros((S, F))
    for s in prange(S):
        for m in range(M):
            c = counts[s, m]
            if c != 0:
                for k in range(indptr[m], indptr[m + 1]):
                    out[s, indices[k]] += c * data[k]
    return out


@njit(parallel=True, cache=True)
def _quad_diag(P, Tidx, Tval, Tnnz):
    """``(Z, M)``: ``t_a'P_b t_a`` for every zone and atom."""
    Z = P.shape[0]
    M = Tidx.shape[0]
    out = np.zeros((Z, M))
    for a in prange(M):
        n = Tnnz[a]
        for z in range(Z):
            acc = 0.0
            for k in range(n):
                row = Tidx[a, k]
                inner = 0.0
                for l in range(n):
                    inner += P[z, row, Tidx[a, l]] * Tval[a, l]
                acc += Tval[a, k] * inner
            out[z, a] = acc
    return out


@njit(parallel=True, cache=True)
def _log_prior_slots(counts, prior, finite):
    """``(S,)``: ``Σ_j log Γ(C_j + α_j) - log Γ(α_j) - log Γ(C_j + 1)``
    (finite ``κ``) or ``Σ_j C_j log p_j - log Γ(C_j + 1)`` (``κ = ∞``), over
    occupied atoms."""
    S, M = counts.shape
    out = np.zeros(S)
    for s in prange(S):
        acc = 0.0
        for m in range(M):
            c = counts[s, m]
            if c > 0:
                cf = np.float64(c)
                if finite:
                    a = prior[s, m]
                    acc += math.lgamma(cf + a) - math.lgamma(a) - math.lgamma(cf + 1.0)
                else:
                    acc += cf * prior[s, m] - math.lgamma(cf + 1.0)
        out[s] = acc
    return out


def _integer_cdf(q: np.ndarray) -> np.ndarray:
    """``(S, M + 1)`` int64 cumulative sums on ``[0, 2⁵³]`` (module notes).
    Rows without mass get a uniform row (as the torch sampler)."""
    S, M = q.shape
    tot = q.sum(1, keepdims=True)
    q = np.where(tot > 0, q / np.where(tot > 0, tot, 1.0), 1.0 / M)
    c = np.cumsum(q, axis=1)
    c /= c[:, -1:]
    K = np.zeros((S, M + 1), dtype=np.int64)
    K[:, 1:] = np.rint(c * float(TWO53)).astype(np.int64)
    K[:, -1] = TWO53
    np.maximum.accumulate(K, axis=1, out=K)
    np.minimum(K, TWO53, out=K)
    return K


def _thresholds(w: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Integer component thresholds and the weights they realize."""
    t0 = 0 if w[0] <= 0 else int(round(w[0] * TWO53))
    t1 = TWO53 if w[2] <= 0 else (t0 if w[1] <= 0 else int(round((w[0] + w[1]) * TWO53)))
    t0, t1 = min(max(t0, 0), TWO53), min(max(t1, t0), TWO53)
    weff = np.array([t0, t1 - t0, TWO53 - t1], dtype=float) / float(TWO53)
    return t0, t1, weff


def _pad_rows(T0: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``T0``'s rows padded to their largest number of nonzeros, cells sorted."""
    T0 = sp.csr_matrix(T0, dtype=float, copy=True)
    T0.sum_duplicates()
    T0.sort_indices()
    M = T0.shape[0]
    nnz = np.diff(T0.indptr).astype(np.int64)
    K = max(int(nnz.max()) if nnz.size else 0, 1)
    idx = np.zeros((M, K), dtype=np.int64)
    val = np.zeros((M, K))
    pos = np.arange(T0.nnz) - np.repeat(T0.indptr[:-1], nnz)
    rows = np.repeat(np.arange(M), nnz)
    idx[rows, pos] = T0.indices
    val[rows, pos] = T0.data
    return idx, val, nnz


# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------


@dataclass
class NumbaSamplerResult(SamplerResult):
    """:class:`~pmedm_vb.latent.sampler.SamplerResult` plus, with
    ``trace=True``, per recorded draw: ``log_target`` ``(D, n)`` (module
    notes) and ``totals_error`` ``(D,)``, the largest ``|Σ_j C_bj - N_b|``
    over slots (0 when the zone totals hold) or ``-1`` if a count is
    negative. ``seconds_functionals``: time spent on ``G'C_b`` and the trace
    (outside ``seconds_per_sweep``); ``threads``: numba threads used."""

    log_target: np.ndarray | None = None
    totals_error: np.ndarray | None = None
    seconds_functionals: float = 0.0
    threads: int = 0


class NumbaSampler:
    """The sampler of :mod:`pmedm_vb.latent.sampler` in numba (module notes).
    Same constructor as :class:`~pmedm_vb.latent.sampler.RealizedSampler`
    (``device`` is not taken; ``threads`` sets numba's thread count). State
    persists across :meth:`run` calls."""

    def __init__(self, inputs: SamplerInputs, cov: BlockCovariance | LowRankCovariance,
                 mix: ProposalMix | None = None, *, seed: int = 0,
                 start_counts: np.ndarray | None = None, threads: int | None = None):
        if not HAVE_NUMBA:
            raise ImportError("numba is not installed: NumbaSampler needs it "
                              "(the torch RealizedSampler does not)")
        self.inputs, self.cov = inputs, cov
        self.mix = mix or ProposalMix()
        self.w = self.mix.weights()
        self.threads = threads
        self.seed = int(seed)
        n, Z, M, C = inputs.shape
        self.n, self.Z, self.M, self.C = n, Z, M, C
        S = n * Z
        self.S = S

        self.Tidx, self.Tval, self.Tnnz = _pad_rows(inputs.T0)
        self.K = self.Tidx.shape[1]
        T0 = sp.csr_matrix(inputs.T0, dtype=float)
        self._T = (T0.indptr.astype(np.int64), T0.indices.astype(np.int64), T0.data.copy())

        self.Nslot = inputs.N.reshape(S).astype(np.int64)
        self.zone_of = (np.arange(S) % Z).astype(np.int64)
        self.finite = bool(np.isfinite(inputs.kappa))
        alpha = inputs.alpha().reshape(S, M)
        self.support = alpha > 0
        if self.finite:
            self.prior = np.ascontiguousarray(alpha, dtype=float)
        else:
            with np.errstate(divide="ignore"):
                self.prior = np.ascontiguousarray(np.log(inputs.p.reshape(S, M)))
        dummy = np.zeros((1, 2), dtype=np.int64)
        self.Kb = _integer_cdf(inputs.r_hat.reshape(S, M)) if self.w[1] > 0 else dummy
        self.Kc = _integer_cdf(inputs.p.reshape(S, M)) if self.w[2] > 0 else dummy
        self.thr0, self.thr1, self.weff = _thresholds(self.w)
        rows = _zone_rows(C, Z)
        self.Yslot = np.ascontiguousarray(np.tile(inputs.Y[rows], (n, 1)))   # (S, C)

        self._set_threads()
        if cov.kind == "block":
            if (cov.Z, cov.C) != (Z, C):
                raise ValueError("BlockCovariance does not match (Z, C)")
            P = np.empty_like(cov.blocks)
            for z, Sb in enumerate(cov.blocks):
                Lc = np.linalg.cholesky(0.5 * (Sb + Sb.T))
                Li = np.linalg.solve(Lc, np.eye(C))
                P[z] = Li.T @ Li
            self.P = np.ascontiguousarray(0.5 * (P + P.transpose(0, 2, 1)))
            self.diagq = _quad_diag(self.P, self.Tidx, self.Tval, self.Tnnz)
            self.g = None
        elif cov.kind == "lowrank":
            if cov.d.size != Z * C:
                raise ValueError("LowRankCovariance does not match Z*C rows")
            dinv = 1.0 / cov.d[rows]                                         # (Z, C)
            L = cov.L[rows]                                                  # (Z, C, R)
            R = cov.rank
            Kmat = np.eye(R) + np.einsum("zcr,zc,zcs->rs", L, dinv, L)
            Kinv = np.linalg.inv(0.5 * (Kmat + Kmat.T))
            self.dinv = np.ascontiguousarray(dinv)
            self.LD = np.ascontiguousarray(L * dinv[:, :, None])
            self.Kinv = np.ascontiguousarray(0.5 * (Kinv + Kinv.T))
            self.R = R
            self.zr = self.yr = None
            Nz = inputs.N.astype(np.int64)
            self.cumN = np.ascontiguousarray(np.cumsum(Nz, axis=1))
            self.Ntot = Nz.sum(1)
        else:
            raise ValueError(f"unknown covariance kind {cov.kind!r}")

        # streams and start
        self.rng = stream_states(self.seed, S if cov.kind == "block" else n)
        self.Nmax = max(int(inputs.N.max()), 1)
        start_rng = np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=(0,)))
        if start_counts is None:
            q = self._start_weights(inputs.r_hat.reshape(S, M), self.support)
            counts = _multinomial_start(q, self.Nslot, start_rng)
        else:
            counts = np.asarray(start_counts).reshape(S, M)
            if np.any(counts < 0) or not np.array_equal(counts.sum(1), self.Nslot):
                raise ValueError("start_counts must be non-negative with zone sums N")
        if np.any((counts > 0) & ~self.support):
            raise ValueError("the start occupies an atom with κ s_b p_bj = 0")
        self.counts = np.ascontiguousarray(counts, dtype=np.int64)
        self.labels = self._labels_from_counts(self.counts)
        self.e = None
        self.refresh()
        self.sweep = 0
        self.stats = np.zeros((S, 3, 3), dtype=np.int64)     # slot x comp x (rej, acc, null)

    # -- set-up helpers -----------------------------------------------------

    def _set_threads(self) -> int:
        if self.threads is not None:
            numba.set_num_threads(int(self.threads))
        return numba.get_num_threads()

    @staticmethod
    def _start_weights(rhat: np.ndarray, support: np.ndarray) -> np.ndarray:
        """The default start's weights: ``r̂_b`` restricted to the support,
        or uniform on it where ``r̂_b`` misses it (the torch sampler's rule)."""
        q = np.where(support, rhat, 0.0)
        bad = q.sum(1) <= 0
        if np.any(bad):
            q[bad] = np.where(support[bad], 1.0, 0.0)
        return q

    def _labels_from_counts(self, counts: np.ndarray) -> np.ndarray:
        S, M = counts.shape
        labels = np.zeros((S, self.Nmax), dtype=np.int32)
        for s in range(S):
            lab = np.repeat(np.arange(M, dtype=np.int32), counts[s])
            labels[s, :lab.size] = lab
        return labels

    # -- state ----------------------------------------------------------------

    def refresh(self) -> float:
        """Recompute ``e`` and ``g`` (or ``z``, ``y``) from the counts;
        returns the largest drift of the running values."""
        indptr, indices, data = self._T
        fitted = _counts_times_csr(self.counts, indptr, indices, data, self.C)
        e = np.ascontiguousarray(self.Yslot - fitted)
        drift = 0.0 if self.e is None else float(np.abs(self.e - e).max(initial=0.0))
        self.e = e
        if self.cov.kind == "block":
            g = np.empty_like(e)
            for z in range(self.Z):
                sl = slice(z, self.S, self.Z)
                g[sl] = e[sl] @ self.P[z]                    # P symmetric
            if self.g is not None:
                drift = max(drift, float(np.abs(self.g - g).max(initial=0.0)))
            self.g = np.ascontiguousarray(g)
        else:
            zr = np.einsum("nzc,zcr->nr", e.reshape(self.n, self.Z, self.C), self.LD)
            yr = zr @ self.Kinv
            if self.yr is not None:
                drift = max(drift, float(np.abs(self.yr - yr).max(initial=0.0)))
            self.zr, self.yr = np.ascontiguousarray(zr), np.ascontiguousarray(yr)
        return drift

    def counts_numpy(self) -> np.ndarray:
        """Current counts, ``(n, Z, M)`` int64 (a copy)."""
        return self.counts.reshape(self.n, self.Z, self.M).copy()

    def log_target(self) -> np.ndarray:
        """``(n,)``: the log target of the current counts (module notes), from
        the refreshed ``e`` and ``g`` (or ``z``, ``y``)."""
        lp = _log_prior_slots(self.counts, self.prior, self.finite)
        N = self.Nslot.astype(float)
        lp = lp + gammaln(N + 1.0)
        if self.finite:
            A = self.prior.sum(1)
            lp = lp + np.where(N > 0, gammaln(A) - gammaln(N + A), 0.0)
        lp = lp.reshape(self.n, self.Z).sum(1)
        if self.cov.kind == "block":
            quad = (self.e * self.g).sum(1).reshape(self.n, self.Z).sum(1)
        else:
            ez = self.e.reshape(self.n, self.Z, self.C)
            quad = np.einsum("nzc,zc->n", ez * ez, self.dinv) - (self.zr * self.yr).sum(1)
        return lp - 0.5 * quad

    def totals_error(self) -> int:
        """Largest ``|Σ_j C_bj - N_b|`` over slots; ``-1`` if a count is
        negative."""
        if self.counts.min(initial=0) < 0:
            return -1
        return int(np.abs(self.counts.sum(1) - self.Nslot).max(initial=0))

    # -- moves ------------------------------------------------------------------

    @property
    def steps_per_sweep(self) -> int:
        """Moves per chain per sweep (the largest over chains)."""
        return int(self.inputs.N.sum(1).max())

    def _moves(self, moves_per_unit: np.ndarray) -> None:
        old = numba.get_parallel_chunksize()
        numba.set_parallel_chunksize(1)              # dynamic scheduling, one slot a chunk
        try:
            if self.cov.kind == "block":
                _block_moves(moves_per_unit, self.labels, self.counts, self.Nslot, self.zone_of,
                             self.e, self.g, self.P, self.diagq, self.Tidx, self.Tval, self.Tnnz,
                             self.prior, self.finite, self.Kb, self.Kc, self.thr0, self.thr1,
                             self.weff, bool(self.mix.drop_correction), self.rng, self.stats)
            else:
                _lowrank_moves(moves_per_unit, self.labels, self.counts, self.Nslot, self.cumN,
                               self.Ntot, self.Z, self.e, self.zr, self.yr, self.dinv, self.LD,
                               self.Kinv, self.Tidx, self.Tval, self.Tnnz, self.prior,
                               self.finite, self.Kb, self.Kc, self.thr0, self.thr1, self.weff,
                               bool(self.mix.drop_correction), self.rng, self.stats)
        finally:
            numba.set_parallel_chunksize(old)

    def _sweep_moves(self) -> np.ndarray:
        if self.cov.kind == "block":
            return self.Nslot.copy()
        return self.Ntot.astype(np.int64).copy()

    def run_sweeps(self, sweeps: int) -> tuple[np.ndarray, float]:
        """Advance ``sweeps`` sweeps, refreshing after each; returns the
        seconds of each sweep and the largest drift."""
        self._set_threads()
        moves = self._sweep_moves()
        secs, drift = np.zeros(sweeps), 0.0
        for k in range(sweeps):
            t0 = time.perf_counter()
            self._moves(moves)
            drift = max(drift, self.refresh())
            secs[k] = time.perf_counter() - t0
            self.sweep += 1
        return secs, drift

    def time_moves(self, moves: int) -> float:
        """Wall seconds for ``moves`` moves in every slot (block) or chain
        (low rank), without refresh; for timing."""
        self._set_threads()
        units = self.S if self.cov.kind == "block" else self.n
        m = np.full(units, int(moves), dtype=np.int64)
        t0 = time.perf_counter()
        self._moves(m)
        return time.perf_counter() - t0

    def acceptance(self) -> dict:
        st = self.stats.sum(0)
        out = {}
        for c, name in enumerate(PROPOSALS):
            prop = int(st[c].sum())
            out[name] = {"proposed": prop, "accepted": int(st[c, 1]), "null": int(st[c, 2]),
                         "rate": st[c, 1] / prop if prop else float("nan"),
                         "rate_non_null": (st[c, 1] / (prop - st[c, 2])
                                           if prop > st[c, 2] else float("nan"))}
        return out

    def functionals(self, G: sp.spmatrix) -> np.ndarray:
        """``(n, Z, F)``: ``G'C_b`` at the current counts (``G``: ``M × F``)."""
        Gr = sp.csr_matrix(G, dtype=float)
        if Gr.shape[0] != self.M:
            raise ValueError("G must be M x F")
        out = _counts_times_csr(self.counts, Gr.indptr.astype(np.int64),
                                Gr.indices.astype(np.int64), Gr.data, Gr.shape[1])
        return out.reshape(self.n, self.Z, -1)

    def run(self, sweeps: int, *, burn: int = 0, thin: int = 1, G: sp.spmatrix | None = None,
            keep_counts: str | None = None, trace: bool = False) -> NumbaSamplerResult:
        """As :meth:`RealizedSampler.run <pmedm_vb.latent.sampler.RealizedSampler.run>`
        (``burn`` sweeps, then ``sweeps`` keeping every ``thin``-th:
        functionals ``G'C_b`` and/or counts); ``trace=True`` also records the
        log target and the zone-totals check at each kept draw."""
        t_start = time.perf_counter()
        if keep_counts not in (None, "dense", "sparse"):
            raise ValueError("keep_counts must be None, 'dense' or 'sparse'")
        Gr = None
        if G is not None:
            Gr = sp.csr_matrix(G, dtype=float)
            if Gr.shape[0] != self.M:
                raise ValueError("G must be M x F")
            Gparts = (Gr.indptr.astype(np.int64), Gr.indices.astype(np.int64), Gr.data,
                      Gr.shape[1])
        threads = self._set_threads()
        secs, drift = [], 0.0
        if burn:
            s, d = self.run_sweeps(burn)
            secs.append(s)
            drift = max(drift, d)
        self.stats[:] = 0
        fun, cnts, at, lt, te = [], [], [], [], []
        t_fun = 0.0
        for k in range(sweeps):
            s, d = self.run_sweeps(1)
            secs.append(s)
            drift = max(drift, d)
            if (k + 1) % thin:
                continue
            t1 = time.perf_counter()
            at.append(self.sweep)
            if Gr is not None:
                v = _counts_times_csr(self.counts, *Gparts)
                fun.append(v.reshape(self.n, self.Z, -1))
            if trace:
                lt.append(self.log_target())
                te.append(self.totals_error())
            if keep_counts == "dense":
                cnts.append(self.counts.reshape(self.n, self.Z, self.M).astype(np.int32))
            elif keep_counts == "sparse":
                cnts.append(sp.csr_matrix(self.counts.astype(np.int32)))
            t_fun += time.perf_counter() - t1
        return NumbaSamplerResult(
            functionals=np.stack(fun) if fun else None, counts=cnts or None,
            sweeps=np.asarray(at, dtype=int), acceptance=self.acceptance(),
            seconds_per_sweep=np.concatenate(secs) if secs else np.zeros(0),
            steps_per_sweep=self.steps_per_sweep, backend=self.cov.kind, drift=drift,
            seconds=time.perf_counter() - t_start,
            log_target=np.stack(lt) if lt else None,
            totals_error=np.asarray(te, dtype=int) if te else None,
            seconds_functionals=t_fun, threads=threads)


def _multinomial_start(q: np.ndarray, N: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``(S, M)`` counts: ``Multinomial(N_s, q_s / Σ q_s)`` per row."""
    S, M = q.shape
    out = np.zeros((S, M), dtype=np.int64)
    for s in range(S):
        if N[s] <= 0:
            continue
        row = q[s] / q[s].sum()
        out[s] = rng.multinomial(int(N[s]), row)
    return out


def multinomial_counts(q: np.ndarray, N: np.ndarray, seed_or_rng) -> np.ndarray:
    """``(n, Z, M)`` counts drawn ``Multinomial(N_cb, q_cb)`` (``q``:
    ``(n, Z, M)`` or ``(Z, M)``, normalized per zone here; ``N``: ``(n, Z)``),
    e.g. for dispersed starts."""
    N = np.asarray(N, dtype=np.int64)
    n, Z = N.shape
    q = np.asarray(q, float)
    if q.ndim == 2:
        q = np.broadcast_to(q, (n, Z, q.shape[1]))
    rng = seed_or_rng if isinstance(seed_or_rng, np.random.Generator) \
        else np.random.default_rng(seed_or_rng)
    M = q.shape[2]
    return _multinomial_start(q.reshape(n * Z, M), N.reshape(-1), rng).reshape(n, Z, M)


def log_target_numpy(inputs: SamplerInputs, cov: BlockCovariance | LowRankCovariance,
                     counts: np.ndarray) -> np.ndarray:
    """``(n,)``: the log target of ``counts`` ``(n, Z, M)`` (module notes),
    densely in numpy (for checks, and for the torch sampler's traces)."""
    n, Z, M, C = inputs.shape
    counts = np.asarray(counts, float).reshape(n, Z, M)
    N = inputs.N.astype(float)
    lp = gammaln(N + 1.0) - gammaln(counts + 1.0).sum(2)
    if np.isfinite(inputs.kappa):
        a = inputs.alpha()
        pos = a > 0
        safe = np.where(pos, a, 1.0)
        terms = np.where(pos, gammaln(counts + safe) - gammaln(safe),
                         np.where(counts > 0, -np.inf, 0.0))
        A = a.sum(2)
        lp = lp + terms.sum(2) + np.where(N > 0, gammaln(A) - gammaln(N + A), 0.0)
    else:
        with np.errstate(divide="ignore"):
            lpm = np.log(inputs.p)
        lp = lp + np.where(inputs.p > 0, counts * np.where(inputs.p > 0, lpm, 0.0),
                           np.where(counts > 0, -np.inf, 0.0)).sum(2)
    rows = _zone_rows(C, Z)
    T0 = sp.csr_matrix(inputs.T0)
    e = np.zeros((n, C * Z))
    for b in range(Z):
        e[:, rows[b]] = inputs.Y[rows[b]][None, :] - np.asarray(counts[:, b, :] @ T0)
    Sigma = cov.dense()
    quad = np.einsum("ni,ni->n", e, np.linalg.solve(Sigma, e.T).T)
    return lp.sum(1) - 0.5 * quad


__all__ = [
    "HAVE_NUMBA",
    "TWO53",
    "BlockCovariance",
    "LowRankCovariance",
    "NumbaSampler",
    "NumbaSamplerResult",
    "ProposalMix",
    "SamplerInputs",
    "log_target_numpy",
    "multinomial_counts",
    "restrict_support",
    "stream_states",
    "xoshiro_reference",
]
