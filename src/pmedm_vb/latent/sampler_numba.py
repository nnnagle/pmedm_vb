"""Exact MCMC for the realized populations given one parameter draw, compiled
with numba (step 5, phase 2; ``PLAN_latent_step5.md`` version 1, §3–§4;
version 2's kernel proposal and pile moves, §3a, and version 3's
occupied-record pile move, §3b, below).

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
  ``T_b = round((w_a + w_b) 2⁵³)``, ``T_c = round((w_a + w_b + w_c) 2⁵³)``
  (a threshold after which every weight is 0 is ``2⁵³``; one of a zero
  weight equals the previous one); the weights used in ``Q`` are the
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

**Version 2 moves** (``PLAN_latent_step5.md`` version 2, §3a): two moves
the torch sampler does not have. Both are off by default (kernel weight 0,
no pile moves), and the sampler then makes exactly the old random choices
(the same draws for the same seed).

*(e) Kernel proposal* (:class:`KernelSpec`, weight ``kernel`` of
:class:`ProposalMix4`). Raw kernel weights ``K_ik = exp{-‖z_i - c_k‖²/2ℓ²}``
of a kernel basis, recomputed from its stored centres ``c_k`` and width
``ℓ`` and the atoms' positions ``z`` (the stored basis columns are
standardized, not these weights). For a unit at atom ``i`` of zone ``b``:
centre ``k`` with probability ``a_ik = K_ik/Σ_k' K_ik'``, then atom ``j``
with probability ``β_bkj = w_bj K_jk/S_bk``, ``S_bk = Σ_j' w_bj' K_j'k``
(``w_b = p_b`` by default, ``r̂_b`` on the support as an option). So

    Q_e(j | i) = Σ_k a_ik β_bkj,

a sum over the centres; the single-unit mixture becomes
``Q(j | i, C) = w_a C_bj/N_b + w_b r̂_bj + w_c p_bj + w_e Q_e(j | i)`` and
the Hastings factor uses ``Q(i | j, C')/Q(j | i, C)`` with ``(e)``'s part
evaluated in both directions (the reverse needs ``Q_e(i | j)``, a second sum
over the centres). Both draws are by integer inverse CDF, as (b) and (c):
``a_ik`` from an ``M × (K + 1)`` table (one row per atom), ``β_bk·`` from one
table per (weight row, centre), and the probabilities entering ``Q`` are the
widths of those tables over 2⁵³, so the ratio is the one of the proposal
actually sampled. A table with no mass (no atom with ``w_bj > 0`` kept for
centre ``k``) makes the proposal a null move (``j = i``); it then contributes
0 to ``Q_e`` in both directions. Computed in log space (``a_ik`` as a
softmax over ``k``, ``β`` from ``K_jk / max_j K_jk``), which changes no
probability and avoids underflow for atoms far from every centre.

*Memory* of (e), per distinct weight row (rows shared by chains, the same
``p_b``, are stored once): the cumulative tables for drawing, ``Σ_k (L_k +
1)`` int64 (``L_k`` the atoms kept for centre ``k``, ``M`` without
truncation), and the same widths again by atom, ``(M, K)`` float64 (exact
integers), so that ``Q_e(j | i)`` reads one contiguous row per direction
rather than ``K`` scattered entries: about ``16 K M`` bytes per zone and
parameter draw (30 centres, 155,000 atoms: 74 MB; 100 centres: 248 MB).
Plus ``a`` (``(M, K + 1)`` int64 and ``(M, K)`` float64). **Truncation**
(``KernelSpec.trunc = ε``): centre ``k``'s table keeps only atoms with
``K_jk ≥ ε max_j K_jk``; the ``β`` used in ``Q`` are then those of the
truncated tables (an atom not kept has ``β_bkj = 0`` in both directions), so
the chain stays exact; truncation can only make (e) alone reducible, not
wrong (the mixture's (b), (c) still reach every atom). It shortens the
drawing tables, not the by-atom widths.

*Pile moves* (exact two-atom Gibbs; ``pile_moves`` per zone per sweep). Pick
atom ``i`` of zone ``b`` from ``r̂_b`` restricted to the zone's support (a
fixed integer table, as the start uses), and ``j`` from ``i`` by the kernel
route of (e) (same tables): the pair's probability does not depend on the
counts. With ``n = C_bi + C_bj`` held, redraw ``m`` = the new ``C_bi`` from
its conditional (derived here; the plan's §3a)

    log Pr(m) = const + log Γ(m + α_bi) - log m! + log Γ(n - m + α_bj)
                - log (n - m)! + Δ d'Σ⁻¹e - ½ Δ² d'Σ⁻¹d,

``α = κ s_b p_b``, ``Δ = C_bi - m`` (units moved from ``i`` to ``j``),
``d = D_b{t(x_j) - t(x_i)}``, ``e`` the current residual (``κ = ∞``:
``m log p_bi - log m! + (n - m) log p_bj - log (n - m)!``). The
``n + 1`` log weights are accumulated from the ratios of successive terms
(one log each; :func:`_pile_split`), exponentiated after subtracting their
maximum, and ``m`` is drawn by inverse CDF with one uniform. A random-scan Gibbs step on ``(C_bi, C_bj)`` with a
state-independent choice of the pair leaves the target invariant. Null
cases (no change, but counted): ``j = i`` or an empty kernel table; ``n =
0``; an atom off the zone's support (``α = 0``: its count is 0 and must stay
0, so the only allowed ``m`` is the current one). On a change, the counts,
``e`` and ``g`` (or ``z``, ``y``) are updated as ``Δ`` single moves would
update them (``e ← e - Δ d`` and so on).

*Labels after pile moves.* The labels only serve to draw an atom with
probability ``C_bi/N_b`` (the moving unit, and proposal (a)): which
individuals carry which label does not enter the transition probabilities
of the counts. So after each sweep's pile moves the labels are rebuilt from
the counts (``O(M + N_b)`` per zone; atom order), rather than tracking which
individuals moved.

*A sweep* is then: the single-unit moves (``N_b`` per zone, or none with
``single_moves=False``), then the pile moves (``pile_moves`` per zone; block
backend: every slot in parallel; low rank: per chain, zone by zone), then
the labels' rebuild, then the refresh. Pile statistics
(:meth:`NumbaSampler.pile_stats`): pairs tried, null pairs, ``n = 0``,
off-support pairs, splits drawn and changed, mean ``|Δ|`` and ``n``, and a
histogram of ``n`` (bins by powers of 2).

**Version 3: pile pairs from occupied records** (``PLAN_latent_step5.md``
version 3, §3b; ``pile_first="occupied"``; off by default, and with the
default ``pile_first="r_hat"`` the sampler makes exactly version 2's random
choices). In zone ``b``: pick a unit uniformly, i.e. record ``i`` with
probability ``C_bi/N_b``; pick ``j`` from ``i`` by the kernel route of (e)
(``pile_second="kernel"``, the same tables) or from ``r̂_b`` restricted to
the zone's support (``pile_second="r_hat"``, the ablation; the integer table
of version 2's first record); draw ``m`` = the new ``C_bi`` from the pair's
exact conditional ``π(m | n)``, ``n = C_bi + C_bj ≥ 1`` (the version 2
conditional, :func:`_pile_split`); accept with probability

    min{1, s(C')/s(C)},   s(C) = (C_bi/N_b) Q(j | i) + (C_bj/N_b) Q(i | j),

``s(C)`` the probability of choosing the unordered pair ``{i, j}`` in state
``C`` (either record first) and ``Q`` the second-record probability actually
used: the kernel's ``Q_e`` from the integer tables (:func:`_kernel_q`, both
directions) or ``r̂``'s table widths over 2⁵³.

*Why it is exact* (derived here). The proposal ``C → C'`` (``C'`` differs
from ``C`` only at ``i``, ``j``, with the same ``n``) can arise only from the
pair ``{i, j}``, so its probability is ``s(C) π(C')/π_n``, ``π_n`` the sum of
``π`` over the splits of ``n`` with the rest of the state held. Then
``π(C) s(C) (π(C')/π_n) min{1, s(C')/s(C)} = π(C) π(C') min{s(C), s(C')}/π_n``
is symmetric in ``C``, ``C'``: detailed balance. The target and the
conditional's terms cancel, leaving the pair-choice ratio. A pair with
``Q(i | j) = 0`` (e.g. a truncated kernel table) can give ``s(C') = 0``
(``m = 0``), always rejected, which keeps the balance. ``j = i`` (or an empty
kernel table) leaves the state as it is and is counted as a null pair; a
drawn split equal to the current one (``m = C_bi``) is a no-op with ratio 1.
The user's own check of this move against enumeration (5 atoms, 7 units,
330 states, second record from a fixed random matrix): total variation 0.004
at noise level, acceptance 0.976; without the ratio, 0.107 (reported by the
user, not rerun here; the gate's phase 1c rows are this module's check).

*Drawing ``i``* from the current counts, not from the labels: the labels
are rebuilt only after a sweep's pile moves, so during them they lag the
counts. Each slot (block) or each zone of a chain (low rank) builds a
Fenwick tree of its counts at the start of its pile moves (``O(M)``), draws
``u`` uniform below ``N_b`` (exactly, as the single moves) and finds the atom
whose cumulative count interval holds ``u`` (``O(log M)``): probability
``C_bi/N_b`` exactly. An accepted change updates the tree at ``i`` and
``j``. The tree is a work array (``M + 1`` int64 per thread), not part of
the state.

*Costs* per move: ``O(log M)`` for ``i``, the second record as version 2
(kernel) or one binary search (``r̂``), ``Q`` in both directions (kernel:
two sums over the centres), the split ``O(n)`` as version 2 (but ``n ≥ 1``
always: no empty pairs), and one uniform for the acceptance when the ratio
is below 1. Statistics: as version 2, plus ``rejected`` (proposed changes
rejected by the ratio), the acceptance and the share of null pairs
(:meth:`NumbaSampler.pile_stats`).

**Gate-only variants** (deliberately wrong samplers): ``ProposalMix4(...,
kernel_reverse_wrong=True)`` uses ``Q_e(j | i)`` in place of ``Q_e(i | j)``
in the reverse probability; ``pile_drop_likelihood=True`` draws the pile
split from the prior part alone; ``pile_drop_ratio=True`` accepts the
occupied-record move's split without the pair-choice ratio.

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
def _draw_seg(A, row, lo, hi, k):
    """Position ``l`` in ``[lo, hi)`` with ``A[row, l] ≤ k < A[row, l + 1]``
    (``A[row, lo] = 0 ≤ k < A[row, hi]``; zero-width entries are never
    returned)."""
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if A[row, mid] <= k:
            lo = mid
        else:
            hi = mid
    return lo


@njit(inline="always", cache=True)
def _kernel_draw(i, t, kn_ccum, kn_atoms, kn_aoff, kn_cum, kn_coff, rng, r):
    """Proposal (e) from atom ``i`` with table row ``t``: a centre from row
    ``i`` of ``kn_ccum``, then an atom from that centre's table; ``-1`` if
    the table has no mass (a null move)."""
    nk = kn_ccum.shape[1] - 1
    k = _draw_seg(kn_ccum, i, 0, nk, _k53(rng, r))
    base = kn_coff[k]
    L = kn_aoff[k + 1] - kn_aoff[k]
    if L == 0 or kn_cum[t, base + L] == 0:
        return -1
    l = _draw_seg(kn_cum, t, base, base + L, _k53(rng, r)) - base
    return kn_atoms[kn_aoff[k] + l]


@njit(inline="always", cache=True)
def _kernel_q(i, j, t, kn_aw, kn_wid):
    """``Q_e(j | i) = Σ_k a_ik β_tkj`` from the integer tables (module
    notes): ``kn_aw[i, k]`` and ``kn_wid[t, j, k]`` are the tables' widths
    (integers below 2⁵³, held exactly as float64), each over 2⁵³."""
    nk = kn_aw.shape[1]
    q = 0.0
    for k in range(nk):
        a = kn_aw[i, k]
        if a != 0.0:
            q += a * kn_wid[t, j, k]
    return q * _INV53 * _INV53


@njit(inline="always", cache=True)
def _propose(s, i, N, labels, Kb, Kc, thr, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff,
             rng, r):
    """Component and proposed atom from atom ``i`` (RNG stream ``r``).
    ``thr``: the three integer thresholds (module notes); with the kernel
    weight 0, ``thr[2] = 2⁵³`` and the draws are those of version 1."""
    k = _k53(rng, r)
    if k < thr[0]:
        return 0, labels[s, _below(rng, r, N)]
    if k < thr[1]:
        return 1, _draw_cdf(Kb, s, _k53(rng, r))
    if k < thr[2]:
        return 2, _draw_cdf(Kc, s, _k53(rng, r))
    j = _kernel_draw(i, kn_tab[s], kn_ccum, kn_atoms, kn_aoff, kn_cum, kn_coff, rng, r)
    return 3, (i if j < 0 else j)


@njit(inline="always", cache=True)
def _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop,
                      kn_aw, kn_tab, kn_wid, kwrong):
    """Log of the prior ratio times the Hastings factor; ``-inf`` for a move
    that must be rejected (``Q_j(C) ≤ 0``, an atom off the support, or a zero
    reverse probability). ``kwrong``: the gate's broken variant, ``Q_e(j|i)``
    in place of ``Q_e(i|j)`` in the reverse probability."""
    w0 = weff[0]
    w1 = weff[1]
    w2 = weff[2]
    w3 = weff[3]
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
    if w3 > 0.0:
        t = kn_tab[s]
        fwd = _kernel_q(i, j, t, kn_aw, kn_wid)
        qj += w3 * fwd
        if kwrong:
            qi += w3 * fwd
        else:
            qi += w3 * _kernel_q(j, i, t, kn_aw, kn_wid)
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
                 prior, finite, Kb, Kc, thr, weff, drop, kn_ccum, kn_tab, kn_atoms, kn_aoff,
                 kn_cum, kn_coff, kn_aw, kn_wid, kwrong, rng, stats):
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
            comp, j = _propose(s, i, N, labels, Kb, Kc, thr, kn_ccum, kn_tab, kn_atoms, kn_aoff,
                               kn_cum, kn_coff, rng, s)
            if j == i:
                stats[s, comp, 2] += 1
                continue
            Ci = np.float64(counts[s, i])
            Cj = np.float64(counts[s, j])
            la = _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop,
                                   kn_aw, kn_tab, kn_wid, kwrong)
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
                   Tidx, Tval, Tnnz, prior, finite, Kb, Kc, thr, weff, drop, kn_ccum, kn_tab,
                   kn_atoms, kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid, kwrong, rng, stats):
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
            comp, j = _propose(s, i, N, labels, Kb, Kc, thr, kn_ccum, kn_tab, kn_atoms, kn_aoff,
                               kn_cum, kn_coff, rng, c)
            if j == i:
                stats[s, comp, 2] += 1
                continue
            Ci = np.float64(counts[s, i])
            Cj = np.float64(counts[s, j])
            la = _log_ratio_counts(s, i, j, Ci, Cj, invN, prior, finite, Kb, Kc, weff, drop,
                                   kn_aw, kn_tab, kn_wid, kwrong)
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
# Pile moves (version 2, module notes)
# --------------------------------------------------------------------------

#: Columns of the pile statistics (per slot), then the histogram of ``n``.
#: ``changed``: splits that changed the counts (for the occupied-record move:
#: proposed changes accepted); ``rejected``: proposed changes rejected by the
#: occupied-record move's pair-choice ratio (always 0 for the version 2 move).
PILE_STATS = ("tried", "null_pair", "n_zero", "off_support", "drawn", "changed",
              "sum_abs_delta", "sum_n", "rejected")
#: Histogram bins of ``n = C_bi + C_bj`` over non-null pairs (bit length).
PILE_N_BINS = ("0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128+")
_NP = len(PILE_STATS)


@njit(inline="always", cache=True)
def _on_support(prior, s, a, finite):
    if finite:
        return prior[s, a] > 0.0
    return prior[s, a] > -np.inf


@njit(inline="always", cache=True)
def _n_bin(n):
    """0 for ``n = 0``, else the bit length of ``n`` capped at 8."""
    b = 0
    while n > 0 and b < 8:
        n >>= 1
        b += 1
    return b


@njit(inline="always", cache=True)
def _pile_split(n, Ci, ai, aj, finite, lin, quad, buf, rng, r):
    """Draw the new ``C_bi = m ∈ {0..n}`` from its conditional (module notes):
    ``ai``, ``aj`` are ``α`` (finite ``κ``) or ``log p`` (``κ = ∞``);
    ``lin = d'Σ⁻¹e``, ``quad = d'Σ⁻¹d``. The log weights are accumulated from
    the ratios of successive terms (derived here from the formula):

        log Pr(m+1) - log Pr(m) = log{(m + α_i)(n - m)} - log{(m + 1)(n - m - 1 + α_j)}
                                  - d'Σ⁻¹e + d'Σ⁻¹d (Δ_m - ½),   Δ_m = C_bi - m

    (``κ = ∞``: the first two terms are ``log p_i - log p_j + log{(n - m)/(m + 1)}``),
    one log per ``m`` instead of four log-gammas; rounding grows like ``n``
    ulps of the log weights."""
    lw = 0.0
    buf[0] = 0.0
    mx = 0.0
    dp = ai - aj
    for m in range(n):
        mf = np.float64(m)
        rf = np.float64(n - m)
        if finite:
            step = math.log(((mf + ai) * rf) / ((mf + 1.0) * (rf - 1.0 + aj)))
        else:
            step = dp + math.log(rf / (mf + 1.0))
        dl = np.float64(Ci - m)
        lw += step - lin + quad * (dl - 0.5)
        buf[m + 1] = lw
        if lw > mx:
            mx = lw
    tot = 0.0
    for m in range(n + 1):
        w = math.exp(buf[m] - mx)
        buf[m] = w
        tot += w
    u = _uniform_open(rng, r) * tot
    acc = 0.0
    for m in range(n + 1):
        acc += buf[m]
        if u < acc:
            return m
    return n                                              # u at the top (rounding)


@njit(inline="always", cache=True)
def _block_lin_quad(s, z, i, j, g, P, diagq, Tidx, Tval, Tnnz):
    """``d'Σ⁻¹e = d'g_b`` and ``d'Σ⁻¹d`` for ``d = t_j - t_i`` (block)."""
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
    return lin, diagq[z, j] + diagq[z, i] - 2.0 * cross


@njit(inline="always", cache=True)
def _block_apply(s, z, i, j, delta, e, g, P, Tidx, Tval, Tnnz):
    """``Δ`` units from ``i`` to ``j``: ``e ← e - Δd``, ``g ← g - Δ P d``."""
    C = e.shape[1]
    dl = np.float64(delta)
    for a in range(Tnnz[j]):
        c = Tidx[j, a]
        v = dl * Tval[j, a]
        e[s, c] -= v
        for k in range(C):
            g[s, k] -= v * P[z, c, k]
    for a in range(Tnnz[i]):
        c = Tidx[i, a]
        v = dl * Tval[i, a]
        e[s, c] += v
        for k in range(C):
            g[s, k] += v * P[z, c, k]


@njit(inline="always", cache=True)
def _pile_pair(s, counts, prior, finite, Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum,
               kn_coff, rng, r, pstats):
    """The pair ``(i, j)`` of a pile move and ``n``; ``j < 0`` for a null
    case (counted in ``pstats``)."""
    pstats[s, 0] += 1
    i = _draw_cdf(Ki, s, _k53(rng, r))
    j = _kernel_draw(i, kn_tab[s], kn_ccum, kn_atoms, kn_aoff, kn_cum, kn_coff, rng, r)
    if j < 0 or j == i:
        pstats[s, 1] += 1
        return i, -1, 0
    n = counts[s, i] + counts[s, j]
    pstats[s, _NP + _n_bin(n)] += 1
    if n == 0:
        pstats[s, 2] += 1
        return i, -1, 0
    if not (_on_support(prior, s, i, finite) and _on_support(prior, s, j, finite)):
        pstats[s, 3] += 1
        return i, -1, 0
    return i, j, n


@njit(parallel=True, cache=True)
def _block_piles(piles, counts, Nslot, zone_of, e, g, P, diagq, Tidx, Tval, Tnnz, prior, finite,
                 Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff, droplik, rng, pstats):
    """``piles[s]`` pile moves in every slot ``s`` (block backend; stream
    ``s``, continuing after the slot's single moves)."""
    S = piles.shape[0]
    for s in prange(S):
        if Nslot[s] <= 0 or piles[s] <= 0:
            continue
        z = zone_of[s]
        buf = np.empty(Nslot[s] + 1)
        for _ in range(piles[s]):
            i, j, n = _pile_pair(s, counts, prior, finite, Ki, kn_ccum, kn_tab, kn_atoms,
                                 kn_aoff, kn_cum, kn_coff, rng, s, pstats)
            if j < 0:
                continue
            lin = 0.0
            quad = 0.0
            if not droplik:
                lin, quad = _block_lin_quad(s, z, i, j, g, P, diagq, Tidx, Tval, Tnnz)
            Ci = counts[s, i]
            m = _pile_split(n, Ci, prior[s, i], prior[s, j], finite, lin, quad, buf, rng, s)
            pstats[s, 4] += 1
            pstats[s, 7] += n
            delta = Ci - m
            if delta == 0:
                continue
            pstats[s, 5] += 1
            pstats[s, 6] += abs(delta)
            counts[s, i] = m
            counts[s, j] = n - m
            _block_apply(s, z, i, j, delta, e, g, P, Tidx, Tval, Tnnz)


@njit(parallel=True, cache=True)
def _lowrank_piles(piles, counts, Nslot, Z, e, zr, yr, dinv, LD, Kinv, Tidx, Tval, Tnnz, prior,
                   finite, Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff, droplik, rng,
                   pstats):
    """Pile moves per chain ``c`` (low-rank backend; stream ``c``): zone by
    zone, ``piles[c Z + b]`` in zone ``b``."""
    n_chains = zr.shape[0]
    R = zr.shape[1]
    Nmax = 1
    for s in range(Nslot.shape[0]):
        if Nslot[s] > Nmax:
            Nmax = Nslot[s]
    for c in prange(n_chains):
        w = np.empty(R)
        Kw = np.empty(R)
        buf = np.empty(Nmax + 1)
        for b in range(Z):
            s = c * Z + b
            if Nslot[s] <= 0:
                continue
            for _ in range(piles[s]):
                i, j, n = _pile_pair(s, counts, prior, finite, Ki, kn_ccum, kn_tab, kn_atoms,
                                     kn_aoff, kn_cum, kn_coff, rng, c, pstats)
                if j < 0:
                    continue
                # w = L'D⁻¹d, d'D⁻¹e (always: needed for the update too)
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
                if droplik:
                    lin = 0.0
                    quad = 0.0
                Ci = counts[s, i]
                m = _pile_split(n, Ci, prior[s, i], prior[s, j], finite, lin, quad, buf, rng, c)
                pstats[s, 4] += 1
                pstats[s, 7] += n
                delta = Ci - m
                if delta == 0:
                    continue
                pstats[s, 5] += 1
                pstats[s, 6] += abs(delta)
                counts[s, i] = m
                counts[s, j] = n - m
                dl = np.float64(delta)
                for a in range(nj):
                    e[s, Tidx[j, a]] -= dl * Tval[j, a]
                for a in range(ni):
                    e[s, Tidx[i, a]] += dl * Tval[i, a]
                for r in range(R):
                    zr[c, r] -= dl * w[r]
                    yr[c, r] -= dl * Kw[r]


# --------------------------------------------------------------------------
# Occupied-record pile moves (version 3, module notes)
# --------------------------------------------------------------------------


@njit(inline="always", cache=True)
def _fen_build(tree, counts, s, M):
    """Fenwick (binary indexed) tree of row ``s`` of the counts: ``tree[k]``
    (1-based) holds the sum of the counts of atoms ``k - lowbit(k), …, k - 1``
    (0-based atoms)."""
    tree[0] = 0
    for k in range(1, M + 1):
        tree[k] = counts[s, k - 1]
    for k in range(1, M + 1):
        p = k + (k & -k)
        if p <= M:
            tree[p] += tree[k]


@njit(inline="always", cache=True)
def _fen_add(tree, M, a, v):
    """Add ``v`` to atom ``a``'s count in the tree."""
    k = a + 1
    while k <= M:
        tree[k] += v
        k += k & -k


@njit(inline="always", cache=True)
def _fen_find(tree, M, top, u):
    """The atom ``a`` with ``Σ_{l<a} C_l ≤ u < Σ_{l≤a} C_l`` (``0 ≤ u < N``;
    ``top`` the largest power of 2 not above ``M``). Such an atom has
    ``C_a ≥ 1``; with ``u`` uniform below ``N`` it is atom ``a`` with
    probability ``C_a/N`` exactly."""
    pos = 0
    step = top
    while step > 0:
        nxt = pos + step
        if nxt <= M and tree[nxt] <= u:
            pos = nxt
            u -= tree[nxt]
        step >>= 1
    return pos


@njit(inline="always", cache=True)
def _occ_pair(s, counts, N, prior, finite, tree, M, top, second_kernel, Ki, kn_ccum, kn_tab,
              kn_atoms, kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid, rng, r, pstats):
    """The ordered pair ``(i, j)`` of an occupied-record pile move, ``n`` and
    the two second-record weights ``q_ji ∝ Q(j | i)``, ``q_ij ∝ Q(i | j)`` (a
    common positive factor left out; it cancels in the ratio); ``j < 0`` for a
    null case (counted)."""
    pstats[s, 0] += 1
    i = _fen_find(tree, M, top, _below(rng, r, N))
    if second_kernel:
        j = _kernel_draw(i, kn_tab[s], kn_ccum, kn_atoms, kn_aoff, kn_cum, kn_coff, rng, r)
    else:
        j = _draw_cdf(Ki, s, _k53(rng, r))
    if j < 0 or j == i:
        pstats[s, 1] += 1
        return i, -1, 0, 0.0, 0.0
    n = counts[s, i] + counts[s, j]
    pstats[s, _NP + _n_bin(n)] += 1
    if not (_on_support(prior, s, i, finite) and _on_support(prior, s, j, finite)):
        pstats[s, 3] += 1
        return i, -1, 0, 0.0, 0.0
    if second_kernel:
        t = kn_tab[s]
        qji = _kernel_q(i, j, t, kn_aw, kn_wid)
        qij = _kernel_q(j, i, t, kn_aw, kn_wid)
    else:
        qji = np.float64(Ki[s, j + 1] - Ki[s, j])
        qij = np.float64(Ki[s, i + 1] - Ki[s, i])
    return i, j, n, qji, qij


@njit(inline="always", cache=True)
def _occ_accept(m, n, Ci, qji, qij, rng, r):
    """Accept the split ``m`` with probability ``min{1, s(C')/s(C)}``,
    ``s(C) ∝ C_bi Q(j | i) + C_bj Q(i | j)`` (module notes; ``N_b`` and the
    common factor of the ``q`` cancel)."""
    den = Ci * qji + (n - Ci) * qij
    if not den > 0.0:                                     # cannot happen: j was drawn from i
        return False
    ratio = (m * qji + (n - m) * qij) / den
    if ratio >= 1.0:
        return True
    return _uniform_open(rng, r) < ratio


@njit(parallel=True, cache=True)
def _block_piles_occ(piles, counts, Nslot, zone_of, e, g, P, diagq, Tidx, Tval, Tnnz, prior,
                     finite, second_kernel, Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff,
                     kn_aw, kn_wid, droplik, dropratio, rng, pstats):
    """``piles[s]`` occupied-record pile moves in every slot ``s`` (block
    backend; stream ``s``, continuing after the slot's single moves)."""
    S, M = counts.shape
    top = 1
    while top * 2 <= M:
        top *= 2
    for s in prange(S):
        N = Nslot[s]
        if N <= 0 or piles[s] <= 0:
            continue
        z = zone_of[s]
        buf = np.empty(N + 1)
        tree = np.empty(M + 1, dtype=np.int64)
        _fen_build(tree, counts, s, M)
        for _ in range(piles[s]):
            i, j, n, qji, qij = _occ_pair(s, counts, N, prior, finite, tree, M, top,
                                          second_kernel, Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff,
                                          kn_cum, kn_coff, kn_aw, kn_wid, rng, s, pstats)
            if j < 0:
                continue
            lin = 0.0
            quad = 0.0
            if not droplik:
                lin, quad = _block_lin_quad(s, z, i, j, g, P, diagq, Tidx, Tval, Tnnz)
            Ci = counts[s, i]
            m = _pile_split(n, Ci, prior[s, i], prior[s, j], finite, lin, quad, buf, rng, s)
            pstats[s, 4] += 1
            pstats[s, 7] += n
            delta = Ci - m
            if delta == 0:
                continue
            if not dropratio and not _occ_accept(m, n, Ci, qji, qij, rng, s):
                pstats[s, 8] += 1
                continue
            pstats[s, 5] += 1
            pstats[s, 6] += abs(delta)
            counts[s, i] = m
            counts[s, j] = n - m
            _fen_add(tree, M, i, -delta)
            _fen_add(tree, M, j, delta)
            _block_apply(s, z, i, j, delta, e, g, P, Tidx, Tval, Tnnz)


@njit(inline="always", cache=True)
def _lowrank_lin_quad(s, b, c, i, j, e, yr, dinv, LD, Kinv, Tidx, Tval, Tnnz, w, Kw):
    """``d'Σ⁻¹e`` and ``d'Σ⁻¹d`` for ``d = t_j - t_i`` in zone ``b`` of chain
    ``c`` (low rank, Woodbury), filling ``w = L'D⁻¹d`` and ``Kw = K⁻¹w``
    (needed for the update); the arithmetic of :func:`_lowrank_piles`."""
    R = w.shape[0]
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
    return lin - wy, dd - wKw


@njit(parallel=True, cache=True)
def _lowrank_piles_occ(piles, counts, Nslot, Z, e, zr, yr, dinv, LD, Kinv, Tidx, Tval, Tnnz,
                       prior, finite, second_kernel, Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff,
                       kn_cum, kn_coff, kn_aw, kn_wid, droplik, dropratio, rng, pstats):
    """Occupied-record pile moves per chain ``c`` (low-rank backend; stream
    ``c``): zone by zone, ``piles[c Z + b]`` in zone ``b``."""
    n_chains = zr.shape[0]
    R = zr.shape[1]
    M = counts.shape[1]
    top = 1
    while top * 2 <= M:
        top *= 2
    Nmax = 1
    for s in range(Nslot.shape[0]):
        if Nslot[s] > Nmax:
            Nmax = Nslot[s]
    for c in prange(n_chains):
        w = np.empty(R)
        Kw = np.empty(R)
        buf = np.empty(Nmax + 1)
        tree = np.empty(M + 1, dtype=np.int64)
        for b in range(Z):
            s = c * Z + b
            N = Nslot[s]
            if N <= 0 or piles[s] <= 0:
                continue
            _fen_build(tree, counts, s, M)
            for _ in range(piles[s]):
                i, j, n, qji, qij = _occ_pair(s, counts, N, prior, finite, tree, M, top,
                                              second_kernel, Ki, kn_ccum, kn_tab, kn_atoms,
                                              kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid, rng, c,
                                              pstats)
                if j < 0:
                    continue
                lin, quad = _lowrank_lin_quad(s, b, c, i, j, e, yr, dinv, LD, Kinv, Tidx, Tval,
                                              Tnnz, w, Kw)
                if droplik:
                    lin = 0.0
                    quad = 0.0
                Ci = counts[s, i]
                m = _pile_split(n, Ci, prior[s, i], prior[s, j], finite, lin, quad, buf, rng, c)
                pstats[s, 4] += 1
                pstats[s, 7] += n
                delta = Ci - m
                if delta == 0:
                    continue
                if not dropratio and not _occ_accept(m, n, Ci, qji, qij, rng, c):
                    pstats[s, 8] += 1
                    continue
                pstats[s, 5] += 1
                pstats[s, 6] += abs(delta)
                counts[s, i] = m
                counts[s, j] = n - m
                _fen_add(tree, M, i, -delta)
                _fen_add(tree, M, j, delta)
                dl = np.float64(delta)
                for a in range(Tnnz[j]):
                    e[s, Tidx[j, a]] -= dl * Tval[j, a]
                for a in range(Tnnz[i]):
                    e[s, Tidx[i, a]] += dl * Tval[i, a]
                for r in range(R):
                    zr[c, r] -= dl * w[r]
                    yr[c, r] -= dl * Kw[r]


@njit(parallel=True, cache=True)
def _rebuild_labels(counts, labels):
    """Labels from the counts, in atom order (module notes)."""
    S, M = counts.shape
    for s in prange(S):
        pos = 0
        for m in range(M):
            lab = np.int32(m)
            for _ in range(counts[s, m]):
                labels[s, pos] = lab
                pos += 1


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


def _thresholds(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer component thresholds (``len(w) - 1`` of them) and the weights
    they realize (module notes). For three weights and ``(w_a, w_b, w_c, 0)``
    the thresholds are version 1's (``T_c = 2⁵³``)."""
    w = np.asarray(w, float)
    thr, prev = [], 0
    for k in range(w.size - 1):
        if w[k] <= 0:
            t = prev
        elif np.all(w[k + 1:] <= 0):
            t = TWO53
        else:
            t = int(round(w[:k + 1].sum() * TWO53))
        t = min(max(t, prev), TWO53)
        thr.append(t)
        prev = t
    edges = np.array([0, *thr, TWO53], dtype=np.int64)
    weff = np.diff(edges).astype(float) / float(TWO53)
    return np.array(thr, dtype=np.int64), weff


#: The numba sampler's proposal components, in mixture order: (a)–(c) as
#: :data:`~pmedm_vb.latent.sampler.PROPOSALS`, then (e) the kernel proposal.
PROPOSALS4 = (*PROPOSALS, "kernel")


@dataclass
class ProposalMix4:
    """Weights of the single-unit mixture with the kernel proposal (e):
    (a) occupancy, (b) realized MAP, (c) superpopulation, (e) kernel;
    normalized. ``drop_correction`` and ``kernel_reverse_wrong``: the gate's
    broken variants (module notes). :class:`NumbaSampler` also takes a
    :class:`~pmedm_vb.latent.sampler.ProposalMix` (kernel weight 0)."""

    occupancy: float = 0.3
    realized: float = 0.2
    prior: float = 0.2
    kernel: float = 0.3
    drop_correction: bool = False
    kernel_reverse_wrong: bool = False

    @classmethod
    def from_mix(cls, mix: ProposalMix) -> ProposalMix4:
        return cls(mix.occupancy, mix.realized, mix.prior, 0.0,
                   drop_correction=mix.drop_correction)

    def weights(self) -> np.ndarray:
        w = np.array([self.occupancy, self.realized, self.prior, self.kernel], float)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("proposal weights must be non-negative, not all 0")
        return w / w.sum()


@dataclass
class KernelSpec:
    """The kernel of proposal (e) and of the pile moves' second atom (module
    notes): atom positions ``z`` (``M × d``, the sampler's atoms, in its
    order), centres ``(K, d)``, width ``ell``. ``trunc``: relative
    truncation ``ε`` of each centre's table (0: none; ``0 ≤ ε < 1``).
    ``weights``: ``"p"`` (``w_b = p_b``, the default) or ``"r_hat"``
    (``r̂_b`` restricted to the zone's support)."""

    z: np.ndarray
    centres: np.ndarray
    ell: float
    trunc: float = 0.0
    weights: str = "p"

    def __post_init__(self) -> None:
        self.z = np.asarray(self.z, float)
        self.centres = np.asarray(self.centres, float)
        if self.z.ndim != 2 or self.centres.ndim != 2 or self.z.shape[1] != self.centres.shape[1]:
            raise ValueError("z must be (M, d) and centres (K, d)")
        if not self.ell > 0:
            raise ValueError("ell must be positive")
        if not 0.0 <= self.trunc < 1.0:
            raise ValueError("trunc must be in [0, 1)")
        if self.weights not in ("p", "r_hat"):
            raise ValueError("weights must be 'p' or 'r_hat'")

    def log_kernel(self) -> np.ndarray:
        """``(M, K)``: ``log K_ik = -‖z_i - c_k‖²/2ℓ²`` (as
        :func:`~pmedm_vb.latent.basis.kernel_features`, before the exp)."""
        z, c = self.z, self.centres
        d2 = np.square(z).sum(1)[:, None] - 2 * z @ c.T + np.square(c).sum(1)[None, :]
        return -np.maximum(d2, 0.0) / (2.0 * self.ell * self.ell)


@dataclass
class _KernelTables:
    """Integer tables of proposal (e) (module notes)."""

    ccum: np.ndarray        # (M, K+1) int64: centre given atom
    tab: np.ndarray         # (S,) int64: slot -> table row
    atoms: np.ndarray       # (Σ L_k,) int64: atoms kept per centre, ascending
    aoff: np.ndarray        # (K+1,) int64
    cum: np.ndarray         # (T, Σ (L_k+1)) int64: draws of j given (t, k)
    coff: np.ndarray        # (K,) int64: start of centre k's cumulative sums
    aw: np.ndarray          # (M, K) float64: widths of ccum (exact integers)
    wid: np.ndarray         # (T, M, K) float64: widths of cum by atom (exact; 0 if not kept)
    kept: np.ndarray        # (K,) atoms kept per centre
    empty: np.ndarray       # (T, K) bool: tables without mass

    @property
    def nbytes(self) -> int:
        return int(self.ccum.nbytes + self.atoms.nbytes + self.cum.nbytes + self.aw.nbytes
                   + self.wid.nbytes)

    def describe(self) -> dict:
        M, K = self.aw.shape
        return {"centres": int(K), "atoms": int(M), "tables": int(self.cum.shape[0]),
                "kept_per_centre_median": float(np.median(self.kept)),
                "kept_per_centre_min": int(self.kept.min()),
                "kept_share": float(self.kept.sum() / (M * K)),
                "empty_tables": int(self.empty.sum()), "megabytes": self.nbytes / 2 ** 20}


def _kernel_tables(spec: KernelSpec, w_slots: np.ndarray) -> _KernelTables:
    """The tables of proposal (e) for weight rows ``w_slots`` ``(S, M)``
    (module notes); identical rows share one table row."""
    M = w_slots.shape[1]
    logK = spec.log_kernel()
    if logK.shape[0] != M:
        raise ValueError(f"KernelSpec.z has {logK.shape[0]} atoms, the sampler {M}")
    nk = logK.shape[1]
    ccum = _integer_cdf(np.exp(logK - logK.max(axis=1, keepdims=True)))      # a_ik, softmax
    rel = logK - logK.max(axis=0, keepdims=True)                             # log K_jk/max_j K_jk
    keep = rel >= np.log(spec.trunc) if spec.trunc > 0 else np.ones_like(rel, dtype=bool)
    lists = [np.flatnonzero(keep[:, k]) for k in range(nk)]
    kept = np.array([a.size for a in lists], dtype=np.int64)
    aoff = np.zeros(nk + 1, dtype=np.int64)
    aoff[1:] = np.cumsum(kept)
    atoms = np.concatenate(lists).astype(np.int64)
    coff = aoff[:-1] + np.arange(nk, dtype=np.int64)
    uniq, inv = np.unique(np.ascontiguousarray(w_slots), axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1).astype(np.int64)
    T = uniq.shape[0]
    cum = np.zeros((T, int(aoff[-1]) + nk), dtype=np.int64)
    wid = np.zeros((T, M, nk))
    empty = np.zeros((T, nk), dtype=bool)
    for k, a in enumerate(lists):
        Kn = np.exp(rel[a, k])
        for t in range(T):
            v = uniq[t, a] * Kn
            if not v.sum() > 0:
                empty[t, k] = True                     # stays all 0: a null proposal
                continue
            c = _integer_cdf(v[None, :])[0]
            cum[t, coff[k]:coff[k] + a.size + 1] = c
            wid[t, a, k] = np.diff(c).astype(float)    # exact: integers below 2⁵³
    aw = np.diff(ccum, axis=1).astype(float)
    return _KernelTables(ccum=np.ascontiguousarray(ccum), tab=inv, atoms=atoms, aoff=aoff,
                         cum=cum, coff=coff, aw=np.ascontiguousarray(aw), wid=wid, kept=kept,
                         empty=empty)


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
    (outside ``seconds_per_sweep``); ``threads``: numba threads used.
    ``piles``: :meth:`NumbaSampler.pile_stats` over the recorded sweeps
    (``None`` without pile moves); ``piles_per_sweep``: pile moves per chain
    per sweep (the largest over chains)."""

    log_target: np.ndarray | None = None
    totals_error: np.ndarray | None = None
    seconds_functionals: float = 0.0
    threads: int = 0
    piles: dict | None = None
    piles_per_sweep: int = 0


class NumbaSampler:
    """The sampler of :mod:`pmedm_vb.latent.sampler` in numba (module notes).
    Same constructor as :class:`~pmedm_vb.latent.sampler.RealizedSampler`
    (``device`` is not taken; ``threads`` sets numba's thread count). State
    persists across :meth:`run` calls.

    Version 2 (module notes): ``mix`` may be a :class:`ProposalMix4` (kernel
    weight); ``kernel``: the :class:`KernelSpec`, needed when the kernel
    weight is positive or there are pile moves; ``pile_moves``: pile moves
    per zone per sweep (an integer, or ``(Z,)`` or ``(n, Z)``; default 0);
    ``single_moves=False`` leaves the single-unit moves out of a sweep (the
    gate's "pile moves alone"); ``pile_drop_likelihood``: the gate's broken
    pile variant.

    Version 3 (module notes): ``pile_first`` ``"r_hat"`` (default: version
    2's pile move, unchanged) or ``"occupied"`` (the occupied-record
    Metropolis–Hastings pile move); ``pile_second`` ``"kernel"`` (default)
    or ``"r_hat"`` (the ablation; only with ``pile_first="occupied"``);
    ``pile_drop_ratio``: the gate's broken variant of the occupied-record move
    (accepted without its pair-choice ratio)."""

    def __init__(self, inputs: SamplerInputs, cov: BlockCovariance | LowRankCovariance,
                 mix: ProposalMix | ProposalMix4 | None = None, *, seed: int = 0,
                 start_counts: np.ndarray | None = None, threads: int | None = None,
                 kernel: KernelSpec | None = None, pile_moves=0, single_moves: bool = True,
                 pile_drop_likelihood: bool = False, pile_first: str = "r_hat",
                 pile_second: str = "kernel", pile_drop_ratio: bool = False):
        if not HAVE_NUMBA:
            raise ImportError("numba is not installed: NumbaSampler needs it "
                              "(the torch RealizedSampler does not)")
        self.inputs, self.cov = inputs, cov
        mix = mix or ProposalMix()
        self.mix = mix if isinstance(mix, ProposalMix4) else ProposalMix4.from_mix(mix)
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
        self.thr, self.weff = _thresholds(self.w)
        # version 2: pile moves and the kernel tables
        self.single_moves = bool(single_moves)
        piles = np.broadcast_to(np.asarray(pile_moves, dtype=np.int64), (n, Z))
        if np.any(piles < 0):
            raise ValueError("pile_moves must be non-negative")
        self.piles = np.ascontiguousarray(piles.reshape(S))
        self.pile_drop_likelihood = bool(pile_drop_likelihood)
        self.has_piles = bool(self.piles.max(initial=0) > 0)
        if not self.single_moves and not self.has_piles:
            raise ValueError("single_moves=False needs pile moves")
        # version 3: the occupied-record pile move
        if pile_first not in ("r_hat", "occupied"):
            raise ValueError("pile_first must be 'r_hat' or 'occupied'")
        if pile_second not in ("kernel", "r_hat"):
            raise ValueError("pile_second must be 'kernel' or 'r_hat'")
        if pile_first == "r_hat" and pile_second != "kernel":
            raise ValueError("pile_second='r_hat' is implemented for pile_first='occupied' only")
        if pile_drop_ratio and pile_first != "occupied":
            raise ValueError("pile_drop_ratio applies to pile_first='occupied' only")
        self.pile_first, self.pile_second = pile_first, pile_second
        self.pile_drop_ratio = bool(pile_drop_ratio)
        needs_kernel = self.w[3] > 0 or (self.has_piles and pile_second == "kernel")
        needs_ki = self.has_piles and (pile_first == "r_hat" or pile_second == "r_hat")
        self.kernel = kernel
        self.kt = None
        if needs_kernel:
            if kernel is None:
                raise ValueError("the kernel proposal and the pile moves need a KernelSpec")
            if kernel.weights == "p":
                wk = inputs.p.reshape(S, M)
            else:
                wk = self._start_weights(inputs.r_hat.reshape(S, M), self.support)
            self.kt = _kernel_tables(kernel, np.ascontiguousarray(wk, dtype=float))
            kt = self.kt
            self._kn = (kt.ccum, kt.tab, kt.atoms, kt.aoff, kt.cum, kt.coff, kt.aw, kt.wid)
        else:
            self._kn = (dummy, np.zeros(S, dtype=np.int64), np.zeros(1, dtype=np.int64),
                        np.zeros(2, dtype=np.int64), dummy, np.zeros(1, dtype=np.int64),
                        np.zeros((1, 1)), np.zeros((1, 1, 1)))
        self.Ki = (_integer_cdf(self._start_weights(inputs.r_hat.reshape(S, M), self.support))
                   if needs_ki else dummy)
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
        self.stats = np.zeros((S, 4, 3), dtype=np.int64)     # slot x comp x (rej, acc, null)
        self.pstats = np.zeros((S, _NP + len(PILE_N_BINS)), dtype=np.int64)

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
        """Single-unit moves per chain per sweep (the largest over chains)."""
        if not self.single_moves:
            return 0
        return int(self.inputs.N.sum(1).max())

    @property
    def piles_per_sweep(self) -> int:
        """Pile moves per chain per sweep (the largest over chains)."""
        return int(self.piles.reshape(self.n, self.Z).sum(1).max())

    def _moves(self, moves_per_unit: np.ndarray) -> None:
        old = numba.get_parallel_chunksize()
        numba.set_parallel_chunksize(1)              # dynamic scheduling, one slot a chunk
        kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid = self._kn
        kwrong = bool(self.mix.kernel_reverse_wrong)
        try:
            if self.cov.kind == "block":
                _block_moves(moves_per_unit, self.labels, self.counts, self.Nslot, self.zone_of,
                             self.e, self.g, self.P, self.diagq, self.Tidx, self.Tval, self.Tnnz,
                             self.prior, self.finite, self.Kb, self.Kc, self.thr, self.weff,
                             bool(self.mix.drop_correction), kn_ccum, kn_tab, kn_atoms, kn_aoff,
                             kn_cum, kn_coff, kn_aw, kn_wid, kwrong, self.rng, self.stats)
            else:
                _lowrank_moves(moves_per_unit, self.labels, self.counts, self.Nslot, self.cumN,
                               self.Ntot, self.Z, self.e, self.zr, self.yr, self.dinv, self.LD,
                               self.Kinv, self.Tidx, self.Tval, self.Tnnz, self.prior,
                               self.finite, self.Kb, self.Kc, self.thr, self.weff,
                               bool(self.mix.drop_correction), kn_ccum, kn_tab, kn_atoms,
                               kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid, kwrong, self.rng,
                               self.stats)
        finally:
            numba.set_parallel_chunksize(old)

    def _pile_moves(self, piles: np.ndarray) -> None:
        """``piles[s]`` pile moves in slot ``s``, then the labels' rebuild."""
        old = numba.get_parallel_chunksize()
        numba.set_parallel_chunksize(1)
        kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid = self._kn
        try:
            if self.pile_first == "occupied":
                second_kernel = self.pile_second == "kernel"
                if self.cov.kind == "block":
                    _block_piles_occ(piles, self.counts, self.Nslot, self.zone_of, self.e, self.g,
                                     self.P, self.diagq, self.Tidx, self.Tval, self.Tnnz,
                                     self.prior, self.finite, second_kernel, self.Ki, kn_ccum,
                                     kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid,
                                     self.pile_drop_likelihood, self.pile_drop_ratio, self.rng,
                                     self.pstats)
                else:
                    _lowrank_piles_occ(piles, self.counts, self.Nslot, self.Z, self.e, self.zr,
                                       self.yr, self.dinv, self.LD, self.Kinv, self.Tidx,
                                       self.Tval, self.Tnnz, self.prior, self.finite,
                                       second_kernel, self.Ki, kn_ccum, kn_tab, kn_atoms,
                                       kn_aoff, kn_cum, kn_coff, kn_aw, kn_wid,
                                       self.pile_drop_likelihood, self.pile_drop_ratio, self.rng,
                                       self.pstats)
            elif self.cov.kind == "block":
                _block_piles(piles, self.counts, self.Nslot, self.zone_of, self.e, self.g, self.P,
                             self.diagq, self.Tidx, self.Tval, self.Tnnz, self.prior, self.finite,
                             self.Ki, kn_ccum, kn_tab, kn_atoms, kn_aoff, kn_cum, kn_coff,
                             self.pile_drop_likelihood, self.rng, self.pstats)
            else:
                _lowrank_piles(piles, self.counts, self.Nslot, self.Z, self.e, self.zr, self.yr,
                               self.dinv, self.LD, self.Kinv, self.Tidx, self.Tval, self.Tnnz,
                               self.prior, self.finite, self.Ki, kn_ccum, kn_tab, kn_atoms,
                               kn_aoff, kn_cum, kn_coff, self.pile_drop_likelihood, self.rng,
                               self.pstats)
            _rebuild_labels(self.counts, self.labels)
        finally:
            numba.set_parallel_chunksize(old)

    def _sweep_moves(self) -> np.ndarray:
        if self.cov.kind == "block":
            return self.Nslot.copy()
        return self.Ntot.astype(np.int64).copy()

    def run_sweeps(self, sweeps: int) -> tuple[np.ndarray, float]:
        """Advance ``sweeps`` sweeps (single-unit moves, then pile moves,
        module notes), refreshing after each; returns the seconds of each
        sweep and the largest drift."""
        self._set_threads()
        moves = self._sweep_moves()
        secs, drift = np.zeros(sweeps), 0.0
        for k in range(sweeps):
            t0 = time.perf_counter()
            if self.single_moves:
                self._moves(moves)
            if self.has_piles:
                self._pile_moves(self.piles)
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

    def time_piles(self, piles: int) -> float:
        """Wall seconds for ``piles`` pile moves in every slot (block) or in
        every zone of every chain (low rank), the labels' rebuild included,
        without refresh; for timing. Needs the kernel tables (unless the second
        record is drawn from ``r̂``)."""
        if self.kt is None and self.pile_second == "kernel":
            raise ValueError("no kernel tables: construct with a KernelSpec and pile moves")
        if self.Ki.shape[0] != self.S and (self.pile_first == "r_hat"
                                           or self.pile_second == "r_hat"):
            self.Ki = _integer_cdf(self._start_weights(self.inputs.r_hat.reshape(self.S, self.M),
                                                       self.support))
        self._set_threads()
        m = np.full(self.S, int(piles), dtype=np.int64)
        t0 = time.perf_counter()
        self._pile_moves(m)
        return time.perf_counter() - t0

    def pile_stats(self) -> dict:
        """Pile moves since the last reset (module notes): counts of
        :data:`PILE_STATS`, ``mean_abs_delta`` (over splits drawn),
        ``mean_n`` (over splits drawn), ``share_changed`` (of splits drawn)
        and the histogram of ``n`` over non-null pairs (:data:`PILE_N_BINS`).
        Version 3: ``mode`` (``pile_first/pile_second``); ``acceptance``, the
        share of drawn splits accepted (a split equal to the current one
        counts as accepted: its ratio is 1), and ``acceptance_of_changes``,
        the share of proposed changes accepted (both 1 for the version 2
        move, a Gibbs step); ``share_null`` (of pairs tried: ``j = i`` or an
        empty kernel table, ``n = 0``, off the support)."""
        st = self.pstats.sum(0)
        out = {k: int(st[c]) for c, k in enumerate(PILE_STATS)}
        out["mode"] = f"{self.pile_first}/{self.pile_second}"
        drawn = out["drawn"]
        out["mean_abs_delta"] = out["sum_abs_delta"] / drawn if drawn else float("nan")
        out["mean_n"] = out["sum_n"] / drawn if drawn else float("nan")
        out["share_changed"] = out["changed"] / drawn if drawn else float("nan")
        prop = out["changed"] + out["rejected"]
        out["acceptance"] = (drawn - out["rejected"]) / drawn if drawn else float("nan")
        out["acceptance_of_changes"] = out["changed"] / prop if prop else float("nan")
        tried = out["tried"]
        out["share_null"] = ((out["null_pair"] + out["n_zero"] + out["off_support"]) / tried
                             if tried else float("nan"))
        out["n_histogram"] = {b: int(st[_NP + k]) for k, b in enumerate(PILE_N_BINS)}
        return out

    def kernel_info(self) -> dict | None:
        """The kernel tables' size (module notes), or ``None``."""
        if self.kt is None:
            return None
        return {**self.kt.describe(), "trunc": self.kernel.trunc, "ell": self.kernel.ell,
                "weights": self.kernel.weights}

    def acceptance(self) -> dict:
        st = self.stats.sum(0)
        out = {}
        for c, name in enumerate(PROPOSALS4):
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
        self.pstats[:] = 0
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
            seconds_functionals=t_fun, threads=threads,
            piles=self.pile_stats() if self.has_piles else None,
            piles_per_sweep=self.piles_per_sweep)


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
    "PILE_N_BINS",
    "PILE_STATS",
    "PROPOSALS4",
    "TWO53",
    "BlockCovariance",
    "KernelSpec",
    "LowRankCovariance",
    "NumbaSampler",
    "NumbaSamplerResult",
    "ProposalMix",
    "ProposalMix4",
    "SamplerInputs",
    "log_target_numpy",
    "multinomial_counts",
    "restrict_support",
    "stream_states",
    "xoshiro_reference",
]
