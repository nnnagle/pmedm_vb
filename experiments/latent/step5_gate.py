"""Step 5, phase 1: the exactness gate of the realized-population sampler
(``PLAN_latent_step5.md`` version 1, §4 item 1 and §5 first bullet;
:mod:`pmedm_vb.latent.sampler`).

    PYTHONPATH=src python experiments/latent/step5_gate.py [--quick] [--device cpu]

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
   sweep checks the extrapolation.
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
    problems = {}
    for kap in ("finite", "inf"):
        inp, block, lowrank, extra = enumeration_problem(6.0 if kap == "finite" else np.inf)
        problems[kap] = (inp, block, lowrank, extra)
    header = (f"  {'backend':8s} {'proposal':42s} {'states':>6s} | {'TV short':>8s} "
              f"{'noise':>7s} {'ratio':>5s} | {'TV long':>8s} {'noise':>7s} {'ratio':>5s} | "
              f"{'shrink':>6s} | accept a/b/c")
    print(header)
    print("  " + "-" * (len(header) - 2))
    start_fixed = np.array([[2, 1, 0, 0], [2, 1, 1, 0], [0, 2, 1, 0]])
    for backend in ("block", "lowrank"):
        enums = {}
        for name, mix, kap, target in configs:
            inp1, block, lowrank, _ = problems[kap]
            cov = block if backend == "block" else lowrank
            if kap not in enums:
                enums[kap] = Enumerated(inp1, cov.dense())
            en = enums[kap]
            n, Z, M = n_chains, 3, 4
            inputs = SamplerInputs(inp1.T0, inp1.Y, np.repeat(inp1.N, n, 0), inp1.p[0],
                                   inp1.r_hat[0], inp1.kappa)
            start = None
            if mix.weights()[1] == 0 and mix.weights()[2] == 0:
                start = np.broadcast_to(start_fixed, (n, Z, M)).copy()
            sampler = RealizedSampler(inputs, cov, mix, device=args.device,
                                      seed=zlib.crc32(f"{backend}:{name}".encode()), start_counts=start)
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
                             for k in ("occupancy", "realized", "prior"))
            n_states = int((pi > 0).sum())
            print(f"  {backend:8s} {name:42s} {n_states:6d} | {out[0][0]:8.4f} {out[0][1]:7.4f} "
                  f"{out[0][0] / out[0][1]:5.2f} | {out[1][0]:8.4f} {out[1][1]:7.4f} "
                  f"{out[1][0] / out[1][1]:5.2f} | {out[0][0] / out[1][0]:6.2f} | {rates}"
                  f"   (drift {res.drift:.1e})")
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
    sampler = RealizedSampler(inputs, pr["cov"], ProposalMix(0.4, 0.3, 0.3), device=args.device,
                              seed=11)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true", help="shorter runs (a smoke test)")
    parser.add_argument("--parts", default="123")
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
    torch.manual_seed(0)
    t0 = time.perf_counter()
    print(f"step 5 phase 1 gate; torch {torch.__version__}, device {args.device}")
    if "1" in args.parts:
        run_enumeration(args)
    if "2" in args.parts:
        run_agreement(args)
    if "3" in args.parts:
        run_timing(args)
    print(f"total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
