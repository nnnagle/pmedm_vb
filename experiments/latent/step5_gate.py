"""Step 5, phase 1 (and 1b): the exactness gate of the realized-population
sampler (``PLAN_latent_step5.md`` version 2, §4 items 1 and 1b, §5 first
bullet; :mod:`pmedm_vb.latent.sampler`, and its numba version
:mod:`pmedm_vb.latent.sampler_numba`).

    PYTHONPATH=src python experiments/latent/step5_gate.py [--quick] [--device cpu]
    PYTHONPATH=src python experiments/latent/step5_gate.py --engine numba [--threads 4]

``--engine`` picks the sampler: ``torch`` (default, the phase 1 sampler) or
``numba`` (the compiled one; needs numba installed). Parts 1 and 2 are the
same for both; a numba *sweep* is ``N_b`` moves in zone ``b`` (block) or the
chain's ``Σ_b N_b`` moves (low rank), where the torch one moves every zone
``max N_b`` times (block), which leaves the checks unchanged. Part 3 differs
by engine (below).

Prints a report; writes nothing. Three parts:

1. **Enumeration.** A toy where the exact posterior over ``C`` is
   enumerated: 3 zones, 4 atoms, ``N = (3, 4, 3)``, 3 cells per zone (14,000
   joint states), ``κ`` finite (concentrations about 2 per zone) and one atom
   off zone 2's support (``p = 0``) that ``r̂`` still proposes. Both
   covariance backends: block-diagonal (random dense zone blocks) and
   ``D + LL'`` with a rank-2 factor on every row (so the zones are coupled
   and the posterior does not factorize). For each proposal alone and the
   mixture, with ``κ`` finite and ``κ = ∞`` (mixture), the joint state
   frequencies of 256 chains are compared with the exact posterior by total
   variation at two run lengths 10× apart (the short one is the first tenth
   of the long run). **Noise reference:** the total variation between the
   frequencies of two halves of the chains, halved (if the sampler is exact
   both halves estimate ``π`` independently, and ``TV(half₁, half₂) ≈
   2 TV(all, π)``; derived here, a heuristic for ``E|·|`` of independent
   errors). Exact: ``TV / noise ≈ 1`` at both lengths and TV shrinking by
   about ``√10 ≈ 3.2``. Occupancy alone is reducible (the module notes): it
   is started from one fixed state and compared with the posterior
   restricted to that state's set of occupied atoms (12 states); its row
   against the full posterior shows the reducibility. A broken variant (the
   proposal correction ``Q_i(C')/Q_j(C)`` dropped, mixture) shows the check
   fails when it should.

   **Phase 1b** (``--engine numba`` only; the torch sampler lacks these
   moves and skips them): the version 2 moves on the same toy, with a small
   synthetic kernel for it (:func:`enumeration_kernel`: 2-D positions of
   the 4 atoms, 3 centres, ``ℓ = 0.7``; the third centre's nearest atom is
   atom 3, off zone 2's support, so with truncation zone 2 has an empty
   table there and the kernel proposal's null branch is exercised). Rows:
   the kernel proposal (e) alone (irreducible on the toy: every ``K_ik > 0``
   and ``w = p > 0`` on each zone's support, so ``Q_e(j | i) > 0`` for every
   pair of support atoms); the mixture with (e) (0.3/0.2/0.2/0.3), with
   ``κ`` finite, with truncated tables (``ε = 0.3``) and with ``κ = ∞``; pile
   moves alone (no single-unit moves; 3 per zone per sweep; irreducible:
   ``r̂ > 0`` on every support atom and ``Q_e > 0``, so every pair of support
   atoms can exchange any number of units); pile moves (2 per zone per
   sweep) with the mixture with (e), ``κ`` finite and ``κ = ∞``. Broken
   variants: (e) alone with its reverse probability ``Q_e(i | j)`` replaced
   by ``Q_e(j | i)``; pile moves alone without the likelihood term. The
   state is recorded once per sweep at every sweep, whatever the moves did
   (a pile move with ``n = 0`` included). The last columns give the share of
   pile splits that changed the counts and the mean ``|Δ|``.
2. **Agreement with stage 2.** A synthetic problem where the Gaussian is
   accurate (run also with ``N`` and ``κ`` scaled by 1/4, where it is less
   so: a gap that is the Gaussian's should shrink about 2× from that run to
   the full one, a gap that is the sampler's need not): 3 zones,
   ``N = (3000, 4000, 5000)``, 20 atoms (every ``p_bj ≥ 0.025``, so ``N_b p_bj ≥ 75`` and ``κ s_b p_bj ≥ 50``), 10 cells,
   ``κ = 8000`` (``κ s_b`` from 2,000 to 3,333, ``φ`` about 2.5),
   ``Y`` drawn from the model, ``Σ_b`` diagonal plus rank 2. The closed-form
   Gaussian conditioning (``E C_b = N_b p_b``, ``Cov C_b = φ_b N_b (diag p_b -
   p_b p_b')``, zones independent, conditioned on ``Y``; built densely here,
   as :func:`~pmedm_vb.latent.free_shares.free_share_conditional_moments`
   does by blocks) against the sampler's posterior means and variances of
   the atom counts, the cells and two aggregates per zone, with Monte Carlo
   errors by the jackknife over chains, and the jackknife errors checked by
   comparing two halves of the chains with each other.
3. **Timing** (CPU): seconds per step against the number of chains for both
   backends at a larger synthetic size (10 zones, 5,000 atoms, 100 cells
   per zone, 8–24 nonzero cells per atom; rank 80 for ``D + LL'``), and the
   implied seconds per sweep against ``N_b`` (a sweep is ``max N_b`` steps
   for the block backend and ``Σ_b N_b`` for the low-rank one); one real
   sweep checks the extrapolation. With ``--engine numba``, the same sizes
   by number of threads (``--threads-grid``): microseconds per move, moves
   per second over all slots (block) or chains (low rank), the implied
   seconds per sweep (numba's sweep, above) against ``N_b``, and one real
   sweep against the extrapolation. The first call of each kernel compiles
   it (cached on disk afterwards); a warm-up call is not timed.
"""

from __future__ import annotations

import argparse
import itertools
import time
import zlib

import numpy as np
import scipy.sparse as sp
import torch
from scipy.special import gammaln

from pmedm_vb.latent.sampler import (
    BlockCovariance,
    LowRankCovariance,
    ProposalMix,
    RealizedSampler,
    SamplerInputs,
)


def make_sampler(args, inputs: SamplerInputs, cov, mix, *, seed: int,
                 start_counts: np.ndarray | None = None, **extra):
    """The ``--engine``'s sampler (same constructor and ``run`` for both).
    ``extra``: the numba sampler's version 2 settings (``kernel``,
    ``pile_moves``, ``single_moves``, ``pile_drop_likelihood``)."""
    if args.engine == "torch":
        if extra:
            raise ValueError("the torch sampler has no kernel proposal or pile moves")
        return RealizedSampler(inputs, cov, mix, device=args.device, seed=seed,
                               start_counts=start_counts)
    from pmedm_vb.latent.sampler_numba import NumbaSampler  # needs numba
    return NumbaSampler(inputs, cov, mix, seed=seed, start_counts=start_counts,
                        threads=args.threads, **extra)

# --------------------------------------------------------------------------
# Part 1: enumeration
# --------------------------------------------------------------------------


def compositions(N: int, M: int) -> np.ndarray:
    return np.array([c for c in itertools.product(range(N + 1), repeat=M) if sum(c) == N])


def log_prior_zone(comps: np.ndarray, N: int, alpha: np.ndarray, p: np.ndarray,
                   kappa: float) -> np.ndarray:
    """``log DirMult(c; N, α)`` (``κ = ∞``: ``log Multinomial(c; N, p)``);
    ``-inf`` where an atom with ``α = 0`` is occupied."""
    out = gammaln(N + 1) - gammaln(comps + 1).sum(1)
    if np.isfinite(kappa):
        a = alpha[None, :]
        pos = a > 0
        terms = np.where(pos, gammaln(comps + np.where(pos, a, 1.0)) - gammaln(np.where(pos, a, 1.0)),
                         np.where(comps > 0, -np.inf, 0.0))
        out = out + gammaln(alpha.sum()) - gammaln(N + alpha.sum()) + terms.sum(1)
    else:
        with np.errstate(divide="ignore"):
            lp = np.log(p)
        safe = np.where(p > 0, lp, 0.0)[None, :]
        out = out + np.where(p[None, :] > 0, comps * safe,
                             np.where(comps > 0, -np.inf, 0.0)).sum(1)
    return out


class Enumerated:
    """The exact joint posterior of a toy (all zones' compositions)."""

    def __init__(self, inputs: SamplerInputs, Sigma: np.ndarray):
        _, Z, M, C = inputs.shape
        self.Z, self.M = Z, M
        self.N = inputs.N[0]
        self.zone_comps = [compositions(int(self.N[b]), M) for b in range(Z)]
        self.sizes = [len(c) for c in self.zone_comps]
        T0 = inputs.T0.toarray()
        alpha = inputs.alpha()[0]
        lp = 0.0
        grids = np.meshgrid(*[np.arange(s) for s in self.sizes], indexing="ij")
        flat = [g.ravel() for g in grids]
        rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
        fit = np.zeros((flat[0].size, C * Z))
        for b in range(Z):
            comps = self.zone_comps[b]
            lpb = log_prior_zone(comps, int(self.N[b]), alpha[b], inputs.p[0, b], inputs.kappa)
            lp = lp + lpb[flat[b]]
            fit[:, rows[b]] = (comps @ T0)[flat[b]]
        e = inputs.Y[None, :] - fit
        lp = lp - 0.5 * np.einsum("si,ij,sj->s", e, np.linalg.inv(Sigma), e)
        self.logpost = lp
        pi = np.exp(lp - lp[np.isfinite(lp)].max())
        self.pi = pi / pi.sum()
        self.lookup = []
        for b in range(Z):
            radix = (self.N[b] + 1) ** np.arange(M)
            table = np.full((self.N[b] + 1) ** M, -1, dtype=np.int64)
            table[self.zone_comps[b] @ radix] = np.arange(self.sizes[b])
            self.lookup.append((radix, table))

    def encode(self, counts: np.ndarray) -> np.ndarray:
        """``counts`` ``(..., Z, M)`` -> joint state index ``(...)``."""
        code = np.zeros(counts.shape[:-2], dtype=np.int64)
        for b in range(self.Z):
            radix, table = self.lookup[b]
            k = table[counts[..., b, :] @ radix]
            if np.any(k < 0):
                raise ValueError("a recorded state is not a composition of N")
            code = code * self.sizes[b] + k
        return code

    def restricted(self, start: np.ndarray) -> np.ndarray:
        """``π`` restricted to the states with ``start``'s occupied atoms."""
        keep = np.ones(self.pi.size, dtype=bool)
        grids = np.unravel_index(np.arange(self.pi.size), self.sizes)
        for b in range(self.Z):
            occ = self.zone_comps[b] > 0
            same = (occ == (start[b] > 0)[None, :]).all(1)
            keep &= same[grids[b]]
        q = np.where(keep, self.pi, 0.0)
        return q / q.sum()


def tv(f: np.ndarray, g: np.ndarray) -> float:
    return 0.5 * float(np.abs(f - g).sum())


def frequencies(codes: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(codes.ravel(), minlength=K) / codes.size


def enumeration_problem(kappa: float) -> tuple[SamplerInputs, np.ndarray, np.ndarray, dict]:
    """The toy of part 1 (module notes): inputs for one chain, and the dense
    ``Σ`` of each backend."""
    rng = np.random.default_rng(20261124)
    Z, C = 3, 3
    T0 = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 1], [2, 1, 0], [1, 2, 1]], float))
    N = np.array([[3, 4, 3]])
    p = np.array([[0.40, 0.30, 0.20, 0.10],
                  [0.15, 0.25, 0.25, 0.35],
                  [0.50, 0.30, 0.20, 0.00]])          # atom 3 off zone 2's support
    r_hat = np.array([[0.10, 0.20, 0.30, 0.40],
                      [0.40, 0.30, 0.20, 0.10],
                      [0.25, 0.25, 0.25, 0.25]])      # mass on the off-support atom
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    true = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [2, 0, 1, 0]])
    fitted = np.zeros(C * Z)
    for b in range(Z):
        fitted[rows[b]] = true[b] @ T0.toarray()
    Y = fitted + rng.normal(0.0, 1.0, C * Z)
    blocks = []
    for b in range(Z):
        A = rng.normal(0.0, 0.6, (C, C))
        blocks.append(A @ A.T + 0.8 * np.eye(C))
    block = BlockCovariance(np.array(blocks))
    d = rng.uniform(0.8, 1.2, C * Z)
    L = rng.normal(0.0, 0.7, (C * Z, 2))
    lowrank = LowRankCovariance(d, L)
    inputs = SamplerInputs(T0, Y, N, p, r_hat, kappa)
    return inputs, block, lowrank, {"true": true}


def enumeration_kernel(trunc: float = 0.0):
    """The toy's kernel for the version 2 moves (module notes)."""
    from pmedm_vb.latent.sampler_numba import KernelSpec

    z = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.2, 1.2]])
    centres = np.array([[0.0, 0.0], [0.5, 1.2], [1.5, 1.5]])
    return KernelSpec(z, centres, 0.7, trunc=trunc)


def phase1b_configs() -> list:
    """Phase 1b rows (module notes): ``(name, mix, kappa, target, extra)``."""
    from pmedm_vb.latent.sampler_numba import ProposalMix4

    ker, ker_t = enumeration_kernel(), enumeration_kernel(0.3)
    mix_e = ProposalMix4(0.3, 0.2, 0.2, 0.3)
    return [
        ("(e) kernel alone", ProposalMix4(0, 0, 0, 1), "finite", "full", {"kernel": ker}),
        ("mixture + (e) 0.3/0.2/0.2/0.3", mix_e, "finite", "full", {"kernel": ker}),
        ("mixture + (e), truncated eps 0.3", mix_e, "finite", "full", {"kernel": ker_t}),
        ("mixture + (e), kappa = inf", mix_e, "inf", "full", {"kernel": ker}),
        ("pile moves alone (3/zone/sweep)", mix_e, "finite", "full",
         {"kernel": ker, "pile_moves": 3, "single_moves": False}),
        ("pile (2/zone) + mixture with (e)", mix_e, "finite", "full",
         {"kernel": ker, "pile_moves": 2}),
        ("pile (2/zone) + mixture (e), kappa=inf", mix_e, "inf", "full",
         {"kernel": ker, "pile_moves": 2}),
        ("BROKEN: (e) alone, reverse Q_e(i|j)->Q_e(j|i)",
         ProposalMix4(0, 0, 0, 1, kernel_reverse_wrong=True), "finite", "full", {"kernel": ker}),
        ("BROKEN: pile alone, likelihood dropped", mix_e, "finite", "full",
         {"kernel": ker, "pile_moves": 3, "single_moves": False, "pile_drop_likelihood": True}),
    ]


def run_enumeration(args) -> None:
    n_chains, long, burn = args.enum_chains, args.enum_sweeps, args.enum_burn
    short = long // 10
    print("=" * 100)
    print("PART 1. Enumeration: TV(sampler state frequencies, exact posterior)")
    print(f"  {n_chains} chains, burn-in {burn} sweeps, then {short} (short) and {long} (long) "
          f"recorded sweeps, one state per chain per sweep")
    print("  noise = TV(half of the chains, other half)/2; exact sampler: TV/noise ~ 1, "
          "TV_short/TV_long ~ sqrt(10) = 3.16")
    configs = [("occupancy (fixed start; restricted)", ProposalMix(1, 0, 0), "finite", "restricted"),
               ("occupancy (fixed start) vs FULL posterior", ProposalMix(1, 0, 0), "finite", "full"),
               ("realized r_hat", ProposalMix(0, 1, 0), "finite", "full"),
               ("prior p", ProposalMix(0, 0, 1), "finite", "full"),
               ("mixture 0.4/0.3/0.3", ProposalMix(0.4, 0.3, 0.3), "finite", "full"),
               ("mixture, kappa = inf", ProposalMix(0.4, 0.3, 0.3), "inf", "full"),
               ("BROKEN: mixture, correction dropped",
                ProposalMix(0.4, 0.3, 0.3, drop_correction=True), "finite", "full")]
    configs = [(*c, {}) for c in configs]
    if args.engine == "numba" and "b" in args.enum_sets:
        if "a" not in args.enum_sets:
            configs = []
        configs += phase1b_configs()
    elif args.engine == "torch":
        print("  (torch engine: phase 1b's kernel proposal and pile moves are numba-only, skipped)")
    problems = {}
    for kap in ("finite", "inf"):
        inp, block, lowrank, extra = enumeration_problem(6.0 if kap == "finite" else np.inf)
        problems[kap] = (inp, block, lowrank, extra)
    header = (f"  {'backend':8s} {'proposal':46s} {'states':>6s} | {'TV short':>8s} "
              f"{'noise':>7s} {'ratio':>5s} | {'TV long':>8s} {'noise':>7s} {'ratio':>5s} | "
              f"{'shrink':>6s} | accept a/b/c" + ("/e; piles changed, mean|D|"
                                                  if args.engine == "numba" else ""))
    print(header)
    print("  " + "-" * (len(header) - 2))
    start_fixed = np.array([[2, 1, 0, 0], [2, 1, 1, 0], [0, 2, 1, 0]])
    for backend in ("block", "lowrank"):
        enums = {}
        for name, mix, kap, target, extra in configs:
            inp1, block, lowrank, _ = problems[kap]
            cov = block if backend == "block" else lowrank
            if kap not in enums:
                enums[kap] = Enumerated(inp1, cov.dense())
            en = enums[kap]
            n, Z, M = n_chains, 3, 4
            inputs = SamplerInputs(inp1.T0, inp1.Y, np.repeat(inp1.N, n, 0), inp1.p[0],
                                   inp1.r_hat[0], inp1.kappa)
            start = None
            w = mix.weights()
            if w[0] == 1.0 and not extra.get("pile_moves"):        # occupancy alone
                start = np.broadcast_to(start_fixed, (n, Z, M)).copy()
            sampler = make_sampler(args, inputs, cov, mix,
                                   seed=zlib.crc32(f"{backend}:{name}".encode()), start_counts=start,
                                   **extra)
            res = sampler.run(long, burn=burn, keep_counts="dense")
            counts = np.stack(res.counts)                               # (long, n, Z, M)
            codes = en.encode(counts)                                    # (long, n)
            pi = en.restricted(start_fixed) if target == "restricted" else en.pi
            K = pi.size
            out = []
            for L_ in (short, long):
                c = codes[:L_]
                f = frequencies(c, K)
                h = n // 2
                noise = tv(frequencies(c[:, :h], K), frequencies(c[:, h:], K)) / 2
                out.append((tv(f, pi), noise))
            acc = res.acceptance
            rates = "/".join("-" if np.isnan(acc[k]["rate"]) else f"{acc[k]['rate']:.2f}"
                             for k in acc)
            piles = getattr(res, "piles", None)
            if piles is not None:
                rates += (f"; {piles['share_changed']:.2f}, {piles['mean_abs_delta']:.2f}")
            n_states = int((pi > 0).sum())
            print(f"  {backend:8s} {name:46s} {n_states:6d} | {out[0][0]:8.4f} {out[0][1]:7.4f} "
                  f"{out[0][0] / out[0][1]:5.2f} | {out[1][0]:8.4f} {out[1][1]:7.4f} "
                  f"{out[1][0] / out[1][1]:5.2f} | {out[0][0] / out[1][0]:6.2f} | {rates}"
                  f"   (drift {res.drift:.1e})", flush=True)
    print()


# --------------------------------------------------------------------------
# Part 2: agreement with stage 2
# --------------------------------------------------------------------------


def gaussian_conditioning(inputs: SamplerInputs, Sigma: np.ndarray, G: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2's conditioning for chain 0 (module notes): ``(Z, F)`` means and
    variances of ``G'C_b`` given ``Y``."""
    _, Z, M, C = inputs.shape
    N, p = inputs.N[0].astype(float), inputs.p[0]
    a0 = inputs.kappa * inputs.shares[0]
    phi = (N + a0) / (1.0 + a0) if np.isfinite(inputs.kappa) else np.ones(Z)
    T0 = inputs.T0.toarray()
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    mu = np.concatenate([N[b] * p[b] for b in range(Z)])               # (Z M,)
    V = np.zeros((Z * M, Z * M))
    A = np.zeros((C * Z, Z * M))
    for b in range(Z):
        sl = slice(b * M, (b + 1) * M)
        V[sl, sl] = phi[b] * N[b] * (np.diag(p[b]) - np.outer(p[b], p[b]))
        A[rows[b], sl] = T0.T
    S = A @ V @ A.T + Sigma
    gain = np.linalg.solve(S, A @ V).T                                  # V A' S⁻¹
    m = mu + gain @ (inputs.Y - A @ mu)
    P = V - gain @ A @ V
    mean, var = np.zeros((Z, G.shape[1])), np.zeros((Z, G.shape[1]))
    for b in range(Z):
        sl = slice(b * M, (b + 1) * M)
        mean[b] = G.T @ m[sl]
        var[b] = np.einsum("mf,mk,kf->f", G, P[sl, sl], G)
    return mean, var


def jackknife(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``draws`` ``(D, n, ...)``: pooled mean and variance over draws and
    chains, and their jackknife standard errors over chains."""
    D, n = draws.shape[:2]
    mean = draws.mean((0, 1))
    var = draws.var((0, 1), ddof=1)
    s1 = draws.sum(0)                                                   # (n, ...)
    s2 = (draws ** 2).sum(0)
    S1, S2 = s1.sum(0), s2.sum(0)
    tot = D * (n - 1)
    m_loo = (S1[None] - s1) / tot
    v_loo = ((S2[None] - s2) - tot * m_loo ** 2) / (tot - 1)
    se_m = np.sqrt((n - 1) / n * ((m_loo - m_loo.mean(0)) ** 2).sum(0))
    se_v = np.sqrt((n - 1) / n * ((v_loo - v_loo.mean(0)) ** 2).sum(0))
    return mean, var, se_m, se_v


def agreement_problem(scale: float = 1.0, seed: int = 7):
    rng = np.random.default_rng(seed)
    Z, M, C = 3, 20, 10
    N = np.rint(scale * np.array([3000, 4000, 5000])).astype(int)
    dense = np.zeros((M, C))
    for a in range(M):
        k = rng.integers(2, 5)
        dense[a, rng.choice(C, k, replace=False)] = rng.integers(1, 4, k)
    T0 = sp.csr_matrix(dense)
    p = np.stack([0.5 * rng.dirichlet(np.full(M, 4.0)) + 0.5 / M for _ in range(Z)])
    shares = N / N.sum()
    kappa = 8000.0 * scale                               # κ s_b from 2,000 to 3,333 at scale 1
    # truth from the model, then Y ~ N(A(C), Σ)
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    fitted = np.zeros(C * Z)
    for b in range(Z):
        g = rng.gamma(kappa * shares[b] * p[b])
        c = rng.multinomial(N[b], g / g.sum())
        fitted[rows[b]] = c @ dense
    blocks = []
    for b in range(Z):
        f = fitted[rows[b]]
        B = rng.normal(0.0, 1.0, (C, 2)) * np.sqrt(0.3 * f.mean())
        blocks.append(np.diag(0.7 * np.maximum(f, 10.0)) + B @ B.T)
    cov = BlockCovariance(np.array(blocks))
    Sigma = cov.dense()
    Y = fitted + np.linalg.cholesky(Sigma) @ rng.standard_normal(C * Z)
    G = np.hstack([np.eye(M), dense, (np.arange(M) < 5)[:, None].astype(float),
                   dense[:, :1] * (np.arange(M) % 2 == 0)[:, None]])
    classes = np.array(["atom"] * M + ["cell"] * C + ["aggregate"] * 2)
    return {"T0": T0, "Y": Y, "N": N, "p": p, "kappa": kappa, "cov": cov, "Sigma": Sigma,
            "G": G, "classes": classes}


def run_agreement(args) -> None:
    print("=" * 100)
    print("PART 2. Agreement with stage 2's Gaussian conditioning (block backend)")
    print("  at two scales: N_b and kappa x 1/4, then x 1 (the Gaussian's own error should "
          "shrink about 2x from the first to the second)")
    for scale in args.agree_scales:
        agreement_at(args, scale)


def agreement_at(args, scale: float) -> None:
    pr = agreement_problem(scale)
    print(f"  -- scale {scale:g}")
    n = args.agree_chains
    Zn = pr["N"].size
    proto = SamplerInputs(pr["T0"], pr["Y"], pr["N"][None, :], pr["p"], pr["p"], pr["kappa"])
    mean_g, var_g = gaussian_conditioning(proto, pr["Sigma"], pr["G"])
    # r̂: the Gaussian's posterior mean of the atom counts, normalized (a stand-in for
    # the realized MAP; any positive r̂ leaves the target unchanged)
    M = pr["p"].shape[1]
    r_hat = np.clip(mean_g[:, :M], 1e-3, None)
    a0 = pr["kappa"] * proto.shares[0]
    phi = (pr["N"] + a0) / (1 + a0)
    print(f"  Z = {Zn}, N_b = {pr['N'].tolist()}, M = {M} atoms, C = {pr['T0'].shape[1]} cells, "
          f"kappa s_b = {np.round(a0).astype(int).tolist()}, phi_b = {np.round(phi, 2).tolist()}, "
          f"min N_b p_bj = {float((pr['N'][:, None] * pr['p']).min()):.1f}")
    inputs = SamplerInputs(pr["T0"], pr["Y"], np.repeat(pr["N"][None, :], n, 0), pr["p"], r_hat,
                           pr["kappa"])
    sampler = make_sampler(args, inputs, pr["cov"], ProposalMix(0.4, 0.3, 0.3), seed=11)
    t0 = time.perf_counter()
    res = sampler.run(args.agree_sweeps, burn=args.agree_burn, thin=1, G=sp.csr_matrix(pr["G"]))
    secs = time.perf_counter() - t0
    draws = res.functionals                                             # (D, n, Z, F)
    mean_s, var_s, se_m, se_v = jackknife(draws)
    sd_g = np.sqrt(var_g)
    z_mean = (mean_s - mean_g) / se_m
    z_var = (var_s - var_g) / se_v
    acc = res.acceptance
    print(f"  {n} chains, burn-in {args.agree_burn} sweeps, {args.agree_sweeps} recorded sweeps "
          f"({res.steps_per_sweep} steps each); {secs:.0f} s; median {np.median(res.seconds_per_sweep):.2f} "
          f"s/sweep; accept a/b/c = " + "/".join(f"{acc[k]['rate']:.3f}" for k in acc)
          + f"; drift {res.drift:.1e}")
    # effective sample size per chain, from the jackknife se (pooled draws)
    ess = var_s / np.maximum(se_m, 1e-300) ** 2
    print(f"  effective draws (var/se^2 of the mean, all chains): median {np.median(ess):.0f}, "
          f"min {ess.min():.0f}")
    print(f"  {'class':10s} {'n':>4s} | {'max|dm|/sd':>10s} {'mean|dm|/sd':>11s} {'max|z_m|':>8s} "
          f"{'%|z_m|>2':>8s} | {'var ratio':>16s} {'max|z_v|':>8s} {'%|z_v|>2':>8s} | "
          f"{'se_m/sd':>7s}")
    cls = pr["classes"]
    for c in ("atom", "cell", "aggregate", "all"):
        sel = np.ones(cls.size, bool) if c == "all" else cls == c
        dm = np.abs(mean_s - mean_g)[:, sel] / sd_g[:, sel]
        ratio = (var_s / var_g)[:, sel]
        zm, zv = np.abs(z_mean[:, sel]), np.abs(z_var[:, sel])
        print(f"  {c:10s} {dm.size:4d} | {dm.max():10.3f} {dm.mean():11.3f} {zm.max():8.2f} "
              f"{100 * (zm > 2).mean():7.1f}% | {ratio.min():7.3f}-{ratio.max():7.3f} "
              f"{zv.max():8.2f} {100 * (zv > 2).mean():7.1f}% | "
              f"{np.median(se_m[:, sel] / sd_g[:, sel]):7.3f}")
    print("  (z = (sampler - Gaussian)/jackknife se over chains; within Monte Carlo error: "
          "about 5% of |z| > 2, max |z| near 3 for this many functionals)")
    # calibration of the se itself: the same z between two halves of the chains
    h = n // 2
    m1, v1, sm1, sv1 = jackknife(draws[:, :h])
    m2, v2, sm2, sv2 = jackknife(draws[:, h:])
    zh_m = np.abs(m1 - m2) / np.sqrt(sm1 ** 2 + sm2 ** 2)
    zh_v = np.abs(v1 - v2) / np.sqrt(sv1 ** 2 + sv2 ** 2)
    print(f"  se check, half of the chains vs the other half (same z, all {zh_m.size}): "
          f"means max|z| {zh_m.max():.2f}, {100 * (zh_m > 2).mean():.1f}% > 2; "
          f"variances max|z| {zh_v.max():.2f}, {100 * (zh_v > 2).mean():.1f}% > 2")
    print()


# --------------------------------------------------------------------------
# Part 3: timing
# --------------------------------------------------------------------------


def timing_problem(n: int, N_b: int, backend: str, seed: int = 3):
    rng = np.random.default_rng(seed)
    Z, M, C = 10, 5000, 100
    nnz = rng.integers(8, 25, M)
    rows, cols, vals = [], [], []
    for a in range(M):
        cc = rng.choice(C, nnz[a], replace=False)
        rows += [a] * nnz[a]
        cols += cc.tolist()
        vals += rng.integers(1, 3, nnz[a]).tolist()
    T0 = sp.csr_matrix((vals, (rows, cols)), shape=(M, C), dtype=float)
    p = rng.dirichlet(np.full(M, 1.0), Z)
    N = np.full((n, Z), N_b)
    Nm = N_b * p
    fitted = np.concatenate([[(Nm[b] @ T0.toarray())[k] for b in range(Z)] for k in range(C)])
    Y = fitted + rng.normal(0, 1, C * Z) * np.sqrt(np.maximum(fitted, 1.0))
    if backend == "block":
        blocks = []
        for b in range(Z):
            B = rng.normal(0, 1, (C, 5))
            blocks.append(np.diag(np.maximum(fitted[b::Z], 1.0)) + B @ B.T)
        cov = BlockCovariance(np.array(blocks))
    else:
        cov = LowRankCovariance(np.maximum(fitted, 1.0), rng.normal(0, 1, (C * Z, 80)))
    inputs = SamplerInputs(T0, Y, N, p, p, kappa=0.25 * N_b * Z)
    return inputs, cov, int(T0.getnnz(axis=1).max())


def run_timing(args) -> None:
    print("=" * 100)
    print(f"PART 3. Timing on {args.device} (torch threads {torch.get_num_threads()}): "
          "Z = 10 zones, M = 5,000 atoms, C = 100 cells per zone")
    chains_grid = [1, 4, 16, 64] if not args.quick else [1, 16]
    sizes = [1_000, 10_000, 100_000]
    for backend in ("block", "lowrank"):
        print(f"  backend {backend}" + (" (rank 80)" if backend == "lowrank" else ""))
        print(f"    {'chains':>6s} {'slots/step':>10s} {'ms/step':>8s} {'moves/s':>10s} | "
              + " ".join(f"{'s/sweep N_b=' + format(s, ','):>18s}" for s in sizes))
        for n in chains_grid:
            inputs, cov, K = timing_problem(n, 1000, backend)
            sm = RealizedSampler(inputs, cov, ProposalMix(0.4, 0.3, 0.3), device=args.device,
                                 seed=1)
            sm.time_steps(50)
            per = sm.time_steps(args.timing_steps)
            slots = n * 10 if backend == "block" else n
            per_sweep = [per * (s if backend == "block" else 10 * s) for s in sizes]
            print(f"    {n:6d} {slots:10d} {1e3 * per:8.3f} {slots / per:10.0f} | "
                  + " ".join(f"{x:18.1f}" for x in per_sweep))
        # one real sweep, N_b = 1,000, 16 chains
        inputs, cov, K = timing_problem(16, 1000, backend)
        sm = RealizedSampler(inputs, cov, ProposalMix(0.4, 0.3, 0.3), device=args.device, seed=2)
        secs, drift = sm.run_sweeps(2)
        per = sm.time_steps(args.timing_steps)
        print(f"    check: 16 chains, N_b = 1,000: a real sweep {secs[-1]:.2f} s vs "
              f"{per * sm.steps_per_sweep:.2f} s extrapolated ({sm.steps_per_sweep} steps; "
              f"largest row nnz K = {K}; drift {drift:.1e})")
    print("  (moves/s counts proposals; a sweep is N_b moves per zone. ms/step is dominated by "
          "the ~100 small torch operations of a step, so it grows slowly with the batch.)")
    print()


def run_timing_numba(args) -> None:
    import numba

    from pmedm_vb.latent.sampler_numba import NumbaSampler

    avail = numba.config.NUMBA_NUM_THREADS
    grid = sorted({t for t in args.threads_grid if 1 <= t <= avail}) or [avail]
    print("=" * 100)
    print(f"PART 3. Timing, numba {numba.__version__} (threading layer "
          f"{numba.config.THREADING_LAYER}, NUMBA_NUM_THREADS {avail}; threads {grid}): "
          "Z = 10 zones, M = 5,000 atoms, C = 100 cells per zone")
    chains_grid = [1, 4, 16, 64] if not args.quick else [1, 16]
    sizes = [1_000, 10_000, 100_000]
    moves = args.timing_moves
    for backend in ("block", "lowrank"):
        unit = "slots" if backend == "block" else "chains"
        print(f"  backend {backend}" + (" (rank 80)" if backend == "lowrank" else "")
              + f": {moves} moves per {unit[:-1]}; a sweep is "
              + ("N_b moves per zone" if backend == "block" else "10 N_b moves per chain"))
        print(f"    {'threads':>7s} {'chains':>6s} {unit:>6s} {'us/move':>8s} {'us/step':>8s} "
              f"{'moves/s':>11s} "
              f"{'accept':>6s} | "
              + " ".join(f"{'s/sweep N_b=' + format(s, ','):>18s}" for s in sizes))
        for threads in grid:
            for n in chains_grid:
                inputs, cov, K = timing_problem(n, 1000, backend)
                sm = NumbaSampler(inputs, cov, ProposalMix(0.4, 0.3, 0.3), seed=1,
                                  threads=threads)
                sm.time_moves(20)                                     # compile / warm up
                sm.stats[:] = 0
                secs = sm.time_moves(moves)
                units = n * 10 if backend == "block" else n
                rate = units * moves / secs
                # CPU time of one move (threads / rate), and wall time of one move
                # of every slot (chain) at once (a torch "step")
                us = 1e6 * threads / rate
                us_step = 1e6 * secs / moves
                acc = sm.stats[:, :, 1].sum() / max(sm.stats.sum(), 1)
                per_sweep = [s * units / rate if backend == "block" else 10 * s * n / rate
                             for s in sizes]
                print(f"    {threads:7d} {n:6d} {units:6d} {us:8.3f} {us_step:8.2f} {rate:11.0f} "
                      f"{acc:6.3f} | "
                      + " ".join(f"{x:18.3f}" for x in per_sweep))
        threads = grid[-1]
        inputs, cov, K = timing_problem(16, 1000, backend)
        sm = NumbaSampler(inputs, cov, ProposalMix(0.4, 0.3, 0.3), seed=2, threads=threads)
        secs, drift = sm.run_sweeps(3)
        m = sm._sweep_moves()
        units = m.size
        t = sm.time_moves(moves)
        est = t * float(m.max()) / moves
        print(f"    check ({threads} threads): 16 chains, N_b = 1,000: a real sweep "
              f"{np.median(secs):.4f} s (refresh included) vs {est:.4f} s extrapolated "
              f"({int(m.max())} moves per {unit[:-1]}, {units} {unit}; largest row nnz "
              f"K = {K}; drift {drift:.1e})")
    print("  (us/move: CPU time of one move, threads / (moves/s); us/step: wall time for every "
          "slot (block) or chain (low rank) to move once, comparable to torch's ms/step; "
          "moves/s: over all of them.)")
    run_timing_v2(args, grid[-1])
    print()


def run_timing_v2(args, threads: int) -> None:
    """Phase 1b timing: the version 2 moves at part 3's size (block backend,
    16 chains, 160 slots): CPU microseconds per single-unit move with the
    kernel proposal in the mixture (0.3/0.2/0.2/0.3) against version 1's
    mixture, and per pile move (labels' rebuild included), for 30 and 100
    centres on made-up 16-D positions (no truncation)."""
    from pmedm_vb.latent.basis import nearest_distance, weighted_kmeans
    from pmedm_vb.latent.sampler_numba import KernelSpec, NumbaSampler, ProposalMix4

    moves = args.timing_moves
    inputs, cov, _K = timing_problem(16, 1000, "block")
    M = inputs.shape[2]
    z = np.random.default_rng(4).normal(0.0, 1.0, (M, 16))
    slots = 16 * 10
    print(f"  version 2 moves, block backend, 16 chains ({slots} slots), {threads} threads, "
          f"{moves} moves per slot:")
    base = NumbaSampler(inputs, cov, ProposalMix(0.4, 0.3, 0.3), seed=1, threads=threads)
    base.time_moves(20)
    t_base = base.time_moves(moves)
    print(f"    version 1 mixture 0.4/0.3/0.3: {1e6 * threads * t_base / (slots * moves):.3f} "
          "us per move (CPU)")
    for k in (30, 100):
        centres, _ = weighted_kmeans(z, np.ones(M), k, seed=1, max_iter=20)
        spec = KernelSpec(z, centres, float(np.median(nearest_distance(z, centres))))
        sm = NumbaSampler(inputs, cov, ProposalMix4(0.3, 0.2, 0.2, 0.3), seed=1,
                          threads=threads, kernel=spec, pile_moves=1)
        sm.time_moves(20)
        t_mix = sm.time_moves(moves)
        sm.time_piles(20)
        t_pile = sm.time_piles(moves)
        st = sm.pile_stats()
        print(f"    {k:3d} centres: mixture with (e) {1e6 * threads * t_mix / (slots * moves):.3f} "
              f"us per move ({t_mix / t_base:.2f} x version 1); pile move "
              f"{1e6 * threads * t_pile / (slots * moves):.3f} us (mean n {st['mean_n']:.2f}); "
              f"kernel tables {sm.kernel_info()['megabytes']:.0f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cpu", help="torch engine only")
    parser.add_argument("--engine", default="torch", choices=("torch", "numba"))
    parser.add_argument("--threads", type=int, default=None,
                        help="numba engine: threads for parts 1 and 2 (default: numba's)")
    parser.add_argument("--threads-grid", type=lambda x: [int(v) for v in x.split(",")],
                        default=[1, 2, 4, 8, 16], help="numba engine, part 3: thread counts")
    parser.add_argument("--quick", action="store_true", help="shorter runs (a smoke test)")
    parser.add_argument("--parts", default="123")
    parser.add_argument("--enum-sets", default="ab",
                        help="part 1 rows: a = phase 1 (version 1 moves), b = phase 1b (numba "
                             "only: kernel proposal and pile moves); default both")
    args = parser.parse_args()
    q = args.quick
    args.enum_chains = 256
    args.enum_sweeps = 400 if q else 4000
    args.enum_burn = 50 if q else 200
    args.agree_chains = 32 if not q else 16
    args.agree_sweeps = 100 if not q else 10
    args.agree_burn = 20 if not q else 5
    args.agree_scales = (0.25, 1.0)
    args.timing_steps = 300 if not q else 50
    args.timing_moves = 20000 if not q else 2000
    torch.manual_seed(0)
    t0 = time.perf_counter()
    if args.engine == "torch":
        print(f"step 5 phase 1 gate; torch {torch.__version__}, device {args.device}")
    else:
        import numba
        print(f"step 5 phase 1 gate; engine numba {numba.__version__}, threads "
              f"{args.threads or numba.config.NUMBA_NUM_THREADS}")
    if "1" in args.parts:
        run_enumeration(args)
    if "2" in args.parts:
        run_agreement(args)
    if "3" in args.parts:
        if args.engine == "torch":
            run_timing(args)
        else:
            run_timing_numba(args)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
