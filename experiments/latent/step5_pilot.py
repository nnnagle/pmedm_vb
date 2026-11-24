"""Step 5, phase 2: the pilot of the realized-population sampler
(``PLAN_latent_step5.md`` version 1, §4 item 2, §5 second bullet;
:mod:`pmedm_vb.latent.sampler_numba`; model ``MODEL_latent_v5_draft.md``
§7, the cut form).

    R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
    PYTHONPATH=src python experiments/latent/step5_pilot.py \\
        --config experiments/embed/stage1_v2.json --regions experiments/latent/regions_tn.json \\
        --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \\
        --step3 $R/latent/step3 --step3b $R/latent/step3b --step4 $R/latent/step4 \\
        --step4c $R/latent/step4c --out $R/latent/step5_pilot --threads 16
    # the dry run, no project data needed:
    PYTHONPATH=src python experiments/latent/step5_pilot.py --synthetic --out /tmp/pilot \\
        --sweeps 300 --burn 100 --threads 4

**Unit** (per ``--region``, default ``knoxville_core`` (4 zones) and
``nashville_core`` (6 zones, the most of any region); replicate
``--replicate`` (0) of simulation S, at step 4c's chosen ``(τ, c)`` for S,
``--step4c``'s ``choice.json``), built as ``step4e_cut.fit_unit`` builds it:
S's data (step 4c's ``Replicate``), the totals module
(:func:`~pmedm_vb.latent.free_shares.cut_totals`), the engine, the MAP of
``U`` given the module's mean (:func:`~pmedm_vb.latent.free_shares.fit_cut_U`),
``p`` = the engine's probabilities at ``Û``, ``κ = c n_A``, ``Y``, and ``Σ``
= S's ``BlockSigma`` (zones independent: the block backend). The sampler's
zone totals are the module's mean rounded to integers summing to ``N_A``
exactly (:func:`round_totals`: floors, then one more unit to the zones with
the largest fractional parts; derived here, the largest-remainder rule).
The shares are those integers over ``N_A``, and the realized MAP ``r̂``
(proposal (b)) and stage 2's moments are computed at them (``Û`` is the MAP
at the unrounded mean; the change is under one unit per zone). Atoms with
``p = 0`` in every zone and parameter draw are dropped
(:func:`~pmedm_vb.latent.sampler.restrict_support`).

**Functionals** traced: step 4e's (fitted cells, B1, B2, rare groups, joint,
the zone total), at most ``--max-per-class`` (100; 0 for all) columns per
class, evenly spaced in the class's order; every rare group.

**At the MAP parameters:** ``--chains`` (4) chains from dispersed starts,
chain ``k`` drawn ``Multinomial(N_b, q_k)`` with ``q`` cycling through
``r̂_b``, ``p_b``, ``½(r̂_b + p_b)`` and ``p_b^{1/2}`` (normalized; flatter
than ``p``), each restricted to the zone's support. ``--burn`` + ``--sweeps``
sweeps (a sweep is ``N_b`` moves in zone ``b``, every zone of every chain in
parallel), every sweep recorded: the log target (up to a constant), the zone
totals check (the largest ``|Σ_j C_bj - N_b|``, which must be 0) and the
functionals. Diagnostics on the sweeps after ``--burn``: rank-normalized
split ``R̂`` and bulk and tail ESS (arviz), per class the median and minimum
ESS per sweep (ESS over all chains' kept sweeps divided by their number,
``chains × sweeps``: the plan's "effective sample size per draw", with one
draw per sweep) and ESS per second of wall time (all chains run together);
seconds per sweep, acceptance per proposal. Functionals constant over the
kept draws (the zone total, and cells no occupied atom reaches) are counted
and left out. **Burn-in** from the log target's trace: for each chain the
first sweep at which it crosses the median of the pooled second half of the
run, and ``R̂`` of the log target over ``[t, end)`` for ``t`` on a grid; the
suggested burn-in is the smallest grid ``t`` after every chain's crossing
with that ``R̂ < 1.01`` (a heuristic, derived here). **Decision rule** (§5,
second bullet): median ESS per sweep ≥ 0.1 in every class and ``R̂ < 1.01``
(the largest over the non-constant functionals reported beside the share
below 1.01).

**Agreement with stage 2** (§5's third-bullet thresholds, applied here at
the MAP parameters): the pooled chains' means and variances against
:func:`~pmedm_vb.latent.free_shares.free_share_conditional_moments` at the
MAP parameters, for functionals whose stage 2 mean is at least 50: the share
with ``|mean difference| ≤ 0.1`` stage 2 sd, with variance ratio in
``[0.8, 1.25]``, and with both (pass: both ≥ 90%), with the Monte Carlo
errors (mean: ``sd/√ESS_bulk``; variance ratio: ``ratio·√(2/ESS_bulk)``, a
Gaussian approximation, derived here) so that a gap can be told from noise.

**Parameter draws** (``--param-draws``, 4): draws of the cut posterior as
:func:`~pmedm_vb.latent.free_shares.cut_posterior_samples` makes them
(totals from the module, rounded as above; ``U`` from its Laplace
approximation), one chain each, all run together as one batch, with their
own ``r̂`` (``--draw-rhat refit``, the default) or the MAP's (``map``), each
started from ``Multinomial(N_b, r̂_b)``: ESS per sweep per class for each
draw, to see whether mixing differs from the MAP parameters.

**Torch** (``--torch-chains k``, default 0): ``k`` chains of the torch
sampler at the MAP parameters, ``--torch-steps`` timed steps after a warm-up,
for seconds per sweep (torch's sweep: ``max N_b`` steps, each moving every
zone) and moves per second, beside numba's. ``--engine torch`` runs the main
chains with the torch sampler instead (slow; traces computed from its counts).

**Outputs** (``<out>/<region>/``): ``map_traces.npz`` (functionals ``(T,
chains, Z, F)`` float32, log target ``(T, chains)``, seconds and totals check
per sweep, the classes and names), ``map_log_target.csv``,
``map_functionals.csv`` (per zone and functional: means, variances, ``R̂``,
ESS, stage 2's moments), ``draws_traces.npz``, ``draws_functionals.csv``,
``summary.txt`` and ``pilot.json`` (written last; a region with it is
skipped, so a timed-out job can be resubmitted unchanged). Regions run one
after another, each with all ``--threads`` (``--workers`` is not used).

**Synthetic** (``--synthetic``): the same pipeline on made-up units with the
real ones' structure (a ``Σ`` block per zone, leave-zone-out supports, one
category per table per atom, rare groups, κ small against ``N_b``), two
"regions" of 4 and 6 zones; ``r̂`` from the same conditional PMEDM; stage 2's
moments by the same Gaussian conditioning written densely
(:func:`dense_conditional_moments`). For checking the driver without the
project data; its numbers say nothing about simulation S.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.latent.sampler import (
    PROPOSALS,
    BlockCovariance,
    ProposalMix,
    SamplerInputs,
    restrict_support,
)

#: Default regions: Knoxville core (4 zones) and Nashville core (6, the most).
DEFAULT_REGIONS = ("knoxville_core", "nashville_core")
#: Tag of the pilot's seeds, apart from steps 4c–4e.
PILOT_TAG = 5002
#: The plan's thresholds (§5).
ESS_PER_SWEEP_MIN = 0.1
RHAT_MAX = 1.01
AGREE_MEAN_SD = 0.1
AGREE_VAR = (0.8, 1.25)
AGREE_SHARE = 0.9
AGREE_MIN_MEAN = 50.0


# --------------------------------------------------------------------------
# The unit
# --------------------------------------------------------------------------


@dataclass
class PilotUnit:
    """One region's sampler inputs at the MAP parameters, the parameter
    draws' inputs, the traced functionals and stage 2's moments (all on the
    restricted support)."""

    name: str
    T0: sp.csr_matrix                    # (M, C)
    Y: np.ndarray                        # (C Z,) stacked
    N: np.ndarray                        # (Z,) integer totals
    p: np.ndarray                        # (Z, M)
    r_hat: np.ndarray                    # (Z, M)
    kappa: float
    cov: BlockCovariance
    G: sp.csr_matrix                     # (M, F)
    cls: np.ndarray                      # (F,)
    names: np.ndarray                    # (F,)
    stage2_mean: np.ndarray              # (Z, F)
    stage2_var: np.ndarray               # (Z, F)
    draws: list = field(default_factory=list)   # [{"N", "p", "r_hat"}]
    info: dict = field(default_factory=dict)

    @property
    def Z(self) -> int:
        return self.N.size

    def inputs(self, n: int) -> SamplerInputs:
        return SamplerInputs(self.T0, self.Y, np.repeat(self.N[None, :], n, 0), self.p,
                             self.r_hat, self.kappa)

    def draw_inputs(self) -> SamplerInputs:
        N = np.stack([d["N"] for d in self.draws])
        return SamplerInputs(self.T0, self.Y, N, np.stack([d["p"] for d in self.draws]),
                             np.stack([d["r_hat"] for d in self.draws]), self.kappa)


def round_totals(x: np.ndarray, N_A: float) -> np.ndarray:
    """Integers summing to ``N_A`` exactly: floors, then one more unit to the
    ``N_A - Σ floor`` zones with the largest fractional parts (ties: the
    lower index)."""
    x = np.asarray(x, float)
    total = int(round(float(N_A)))
    if abs(total - float(N_A)) > 1e-6:
        raise ValueError("N_A is not an integer")
    base = np.floor(x).astype(np.int64)
    short = total - int(base.sum())
    if not 0 <= short <= x.size:
        raise ValueError(f"the totals sum to {x.sum():.3f}, not N_A = {total}")
    order = np.argsort(-(x - base), kind="stable")
    base[order[:short]] += 1
    if np.any(base < 1):
        raise ValueError("a rounded zone total is below 1")
    return base


def select_columns(cls: np.ndarray, max_per_class: int) -> np.ndarray:
    """At most ``max_per_class`` columns per class (0: all), evenly spaced in
    the class's order; every rare group and the total."""
    keep = []
    for c in pd.unique(cls):
        idx = np.flatnonzero(cls == c)
        if max_per_class > 0 and idx.size > max_per_class and c not in ("rare", "total"):
            idx = idx[np.unique(np.linspace(0, idx.size - 1, max_per_class).round().astype(int))]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def restrict_unit(name, T0, Y, N, p, r_hat, kappa, cov, G, cls, names, mean_g, var_g,
                  draws, info) -> PilotUnit:
    """Drop atoms with ``p = 0`` in every zone of the MAP and every draw."""
    stack = np.concatenate([p[None]] + [d["p"][None] for d in draws])
    keep = restrict_support(stack, T0)["keep"]
    info = {**info, "atoms": int(p.shape[1]), "atoms_on_support": int(keep.size)}
    return PilotUnit(
        name=name, T0=sp.csr_matrix(T0)[keep], Y=np.asarray(Y, float), N=np.asarray(N),
        p=np.ascontiguousarray(p[:, keep]), r_hat=np.ascontiguousarray(r_hat[:, keep]),
        kappa=float(kappa), cov=cov,
        G=sp.csr_matrix(G)[keep], cls=np.asarray(cls), names=np.asarray(names),
        stage2_mean=mean_g, stage2_var=var_g,
        draws=[{"N": d["N"], "p": np.ascontiguousarray(d["p"][:, keep]),
                "r_hat": np.ascontiguousarray(d["r_hat"][:, keep])} for d in draws],
        info=info)


def build_real_unit(region_id: str, args) -> PilotUnit:
    """Simulation S's unit, as ``step4e_cut.fit_unit`` builds it (module notes)."""
    import step4c_joint as s4c
    import step4e_cut as s4e
    from scipy.linalg import cholesky, solve_triangular

    from pmedm_vb.latent.free_shares import (
        cut_information,
        cut_totals,
        draw_cut_totals,
        fit_cut_U,
        free_share_conditional_moments,
        free_share_engine,
        shares_to_psi,
        solve_conditional_fixed_shares,
    )
    from pmedm_vb.latent.overdispersed import dirmult_phi
    from pmedm_vb.latent.posterior import Functionals
    from pmedm_vb.latent.region_reference import zone_log_bases

    s = s4c._SHARED
    pick = s["choice"][f"S/{region_id}"]
    c, m, k = pick["c"], pick["tau_multiple"], args.replicate
    t0 = time.perf_counter()
    rep = s4c.Replicate("S", region_id, k)
    kappa = c * rep.n_A
    tau = m * s["tau_ref_g"]
    x, Sx = s4e.direct_unit_totals(rep)
    mean, tcov = cut_totals(x, Sx, rep.N_A)
    if not np.all(mean > 0):
        raise ValueError(f"{region_id} r{k}: the totals module gave a non-positive total")
    N_int = round_totals(mean, rep.N_A)
    shares = mean / rep.N_A
    psi = shares_to_psi(shares)
    engine = free_share_engine(rep.area, s["H"], rep.ref["theta"], s["G"],
                               zone_log_bases(rep.ref["logb"], rep.ref["zone"], rep.Z, s4c.MODE),
                               s["pairs"], rep.N_A, kappa)
    fit, _starts = fit_cut_U(engine, rep.theta0, tau, psi, gtol=args.map_gtol,
                             max_iter=args.map_max_iter)
    seconds_map = time.perf_counter() - t0
    phi = dirmult_phi(rep.N_A, kappa)
    T0 = s["X"]["A"]
    Y = rep.pA.inputs.targets()

    def realized(p_mz, N):
        res = solve_conditional_fixed_shares(T0, p_mz, N / N.sum(), rep.N_A, rep.N_A / phi, Y,
                                             rep.sigma_dense, tol=args.cond_tol,
                                             max_iter=args.cond_max_iter)
        return res.r, res

    p = engine.probabilities(rep.theta0, fit.U)                           # (M, Z)
    r_hat, res = realized(p, N_int)
    # stage 2 at the MAP parameters, the traced columns only
    fun = s["fun"]
    cols = select_columns(fun.cls, args.max_per_class)
    fsub = Functionals(G=fun.G[:, cols].tocsc(), cls=fun.cls[cols], names=fun.names[cols])
    psi_int = shares_to_psi(N_int / N_int.sum())
    ev = engine.evaluate(rep.theta0, fit.U, psi_int, gradient=False)
    mean_g, var_g, negative = free_share_conditional_moments(engine, ev, fsub, T0)

    # parameter draws of the cut posterior (as cut_posterior_samples)
    rng = np.random.default_rng([args.seed, s4c.SIM_INDEX["S"], rep.index, k, 2,
                                 int(round(m * 1000)), int(round(kappa)), PILOT_TAG])
    draws = []
    if args.param_draws > 0:
        ev0 = engine.evaluate(rep.theta0, fit.U, fit.psi, gradient=False)
        Rf = cholesky(cut_information(engine, ev0, fit.tau), lower=False)
        xi = rng.standard_normal((fit.U.size, args.param_draws))
        Us = (fit.U.ravel()[:, None] + solve_triangular(Rf, xi, lower=False)).T
        totals = draw_cut_totals(mean, tcov, args.param_draws, rng)
        for d in range(args.param_draws):
            Nd = round_totals(totals[d] + (rep.N_A - totals[d].sum()) / rep.Z, rep.N_A)
            pd_ = engine.probabilities(rep.theta0, Us[d].reshape(engine.Z, engine.K))
            rd = realized(pd_, Nd)[0] if args.draw_rhat == "refit" else r_hat
            draws.append({"N": Nd, "p": pd_.T, "r_hat": rd})
    cov = BlockCovariance.from_factors(rep.sigma, rep.Z)
    info = {"simulation": "S", "region": region_id, "replicate": k, "c": c, "tau_multiple": m,
            "tau": tau, "kappa": kappa, "phi": phi, "N_A": rep.N_A, "n_A": rep.n_A,
            "Z": rep.Z, "cells_per_zone": int(T0.shape[1]),
            "totals_mean": mean.tolist(), "totals_int": N_int.tolist(),
            "totals_sd": np.sqrt(np.clip(np.diag(tcov), 0, None)).tolist(),
            "map_objective": fit.objective, "map_converged": bool(fit.converged),
            "map_grad_max": fit.grad_max, "map_start": fit.start,
            "realized_converged": bool(res.converged), "realized_gap": float(res.gap),
            "stage2_negative_variances": int(negative), "seconds_map": seconds_map,
            "seconds_unit": time.perf_counter() - t0, "draw_rhat": args.draw_rhat,
            "functionals_traced": int(cols.size), "functionals_all": int(fun.F)}
    return restrict_unit(region_id, T0, Y, N_int, p.T, r_hat, kappa, cov, fsub.G, fsub.cls,
                         fsub.names, mean_g, var_g, draws, info)


# --------------------------------------------------------------------------
# The synthetic unit (dry runs)
# --------------------------------------------------------------------------


def dense_conditional_moments(T0, Y, N, p, kappa, Sigma, G) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2's Gaussian conditioning given the totals (the formulas of
    :func:`~pmedm_vb.latent.free_shares.free_share_conditional_moments`,
    with ``S = Σ + Σ_b c_b D_b V_b D_b'`` built densely): ``(Z, F)`` means
    and variances of ``G'C_b`` given ``Y``."""
    T = sp.csr_matrix(T0).toarray()
    G = sp.csr_matrix(G).toarray()
    Z = p.shape[0]
    C = T.shape[1]
    N = np.asarray(N, float)
    s = N / N.sum()
    a = kappa * s
    c = N * (N + a) / (1.0 + a) if np.isfinite(kappa) else N.copy()
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    S = np.array(Sigma, float, copy=True)
    m = np.zeros(C * Z)
    mu = p @ T                                                            # (Z, C)
    for b in range(Z):
        V = T.T @ (p[b][:, None] * T) - np.outer(mu[b], mu[b])
        S[np.ix_(rows[b], rows[b])] += c[b] * V
        m[rows[b]] = N[b] * mu[b]
    Sinv = np.linalg.inv(0.5 * (S + S.T))
    r = Sinv @ (np.asarray(Y, float) - m)
    mean, var = np.zeros((Z, G.shape[1])), np.zeros((Z, G.shape[1]))
    for b in range(Z):
        Eg = G.T @ p[b]
        E2 = (G * G).T @ p[b]
        kg = T.T @ (p[b][:, None] * G) - np.outer(mu[b], Eg)              # (C, F)
        Gbb = Sinv[np.ix_(rows[b], rows[b])]
        mean[b] = N[b] * Eg + c[b] * (kg.T @ r[rows[b]])
        var[b] = np.maximum(c[b] * (E2 - Eg ** 2)
                            - c[b] ** 2 * np.einsum("ij,ij->j", kg, Gbb @ kg), 0.0)
    return mean, var


def build_synthetic_unit(name: str, Z: int, args, seed: int) -> PilotUnit:
    """A made-up unit with the real one's structure (module notes)."""
    from pmedm_vb.latent.free_shares import solve_conditional_fixed_shares
    from pmedm_vb.latent.overdispersed import dirmult_phi

    t0 = time.perf_counter()
    rng = np.random.default_rng([seed, Z, PILOT_TAG])
    M = args.synthetic_atoms
    n_cat = 5                                             # categories per table
    home = rng.integers(0, Z, M)                          # each atom's own zone (LZO)

    def onehot_tables(k, weights=None):
        cols = []
        for t in range(k):
            cat = rng.integers(0, n_cat, M)
            v = np.ones(M) if weights is None else weights
            cols.append(sp.csr_matrix((v, (np.arange(M), cat)), shape=(M, n_cat)))
        return sp.hstack(cols).tocsr()

    persons = rng.integers(1, 5, M).astype(float)
    T0 = sp.hstack([onehot_tables(4), onehot_tables(4, persons)]).tocsr()  # (M, 40)
    C = T0.shape[1]
    B1, B2 = onehot_tables(6), onehot_tables(4)
    t1, t2 = rng.integers(0, n_cat, M), rng.integers(0, n_cat, M)
    joint = sp.csr_matrix((np.ones(M), (np.arange(M), t1 * n_cat + t2)), shape=(M, n_cat ** 2))
    rare = sp.csr_matrix((rng.random((M, 15)) < 0.004).astype(float))
    total = sp.csr_matrix(np.ones((M, 1)))
    Gfull = sp.hstack([T0, B1, B2, rare, joint, total]).tocsr()
    cls = np.array(["fitted"] * C + ["B1"] * B1.shape[1] + ["B2"] * B2.shape[1]
                   + ["rare"] * 15 + ["joint"] * joint.shape[1] + ["total"])
    names = np.array([f"{c}:{k}" for c, k in zip(cls, range(cls.size))])
    cols = select_columns(cls, args.max_per_class)

    N = rng.integers(args.synthetic_N // 2, args.synthetic_N + 1, Z)
    N_A = int(N.sum())
    n_A = N_A // 8
    kappa = 0.25 * n_A
    logq = rng.normal(0.0, 1.0, M)
    p = np.exp(logq[None, :] + rng.normal(0.0, 0.7, (Z, M)))
    p[home[None, :] == np.arange(Z)[:, None]] = 0.0       # leave-zone-out supports
    p /= p.sum(1, keepdims=True)
    shares = N / N_A
    truth = np.stack([rng.multinomial(N[b], rng.dirichlet(kappa * shares[b] * p[b] + 1e-300))
                      for b in range(Z)])
    fitted = truth @ T0.toarray()                          # (Z, C)
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    blocks = []
    for b in range(Z):
        f = fitted[b]
        Bf = rng.normal(0.0, 1.0, (C, 2)) * np.sqrt(0.5 * f.mean())
        blocks.append(np.diag(7.0 * np.maximum(f, 5.0)) + Bf @ Bf.T + 4.0 * np.eye(C))
    cov = BlockCovariance(np.array(blocks))
    Sigma = cov.dense()
    Y = np.zeros(C * Z)
    for b in range(Z):
        Y[rows[b]] = fitted[b] + np.linalg.cholesky(blocks[b]) @ rng.standard_normal(C)
    phi = dirmult_phi(N_A, kappa)

    def realized(pz, Nz):
        res = solve_conditional_fixed_shares(T0, pz.T, Nz / Nz.sum(), N_A, N_A / phi, Y, Sigma,
                                             tol=args.cond_tol, max_iter=args.cond_max_iter)
        return res.r, res

    r_hat, res = realized(p, N)
    mean_g, var_g = dense_conditional_moments(T0, Y, N, p, kappa, Sigma, Gfull[:, cols])
    draws = []
    for d in range(args.param_draws):
        x = N + rng.normal(0.0, 0.01 * N.mean(), Z)
        Nd = round_totals(x + (N_A - x.sum()) / Z, N_A)
        pd_ = p * np.exp(rng.normal(0.0, 0.05, (Z, M)))
        pd_ /= pd_.sum(1, keepdims=True)
        rd = realized(pd_, Nd)[0] if args.draw_rhat == "refit" else r_hat
        draws.append({"N": Nd, "p": pd_, "r_hat": rd})
    info = {"simulation": "synthetic", "region": name, "Z": Z, "N_A": N_A, "n_A": n_A,
            "kappa": kappa, "phi": phi, "cells_per_zone": C, "totals_int": N.tolist(),
            "realized_converged": bool(res.converged), "realized_gap": float(res.gap),
            "seconds_unit": time.perf_counter() - t0, "draw_rhat": args.draw_rhat,
            "functionals_traced": int(cols.size), "functionals_all": int(cls.size)}
    return restrict_unit(name, T0, Y, N, p, r_hat, kappa, cov, Gfull[:, cols], cls[cols],
                         names[cols], mean_g, var_g, draws, info)


# --------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------


def dispersed_starts(unit: PilotUnit, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, Z, M)`` start counts (module notes)."""
    from pmedm_vb.latent.sampler_numba import multinomial_counts

    support = unit.p > 0
    rh = np.where(support, unit.r_hat, 0.0)
    rh = np.where(rh.sum(1, keepdims=True) > 0, rh, unit.p)
    weights = [rh, unit.p, 0.5 * rh / rh.sum(1, keepdims=True) + 0.5 * unit.p,
               np.sqrt(unit.p)]
    out = np.empty((n, unit.Z, unit.p.shape[1]), dtype=np.int64)
    for k in range(n):
        out[k] = multinomial_counts(weights[k % len(weights)], unit.N[None, :], rng)[0]
    return out


def start_names(n: int) -> list[str]:
    names = ["r_hat", "p", "mix", "sqrt_p"]
    return [names[k % len(names)] for k in range(n)]


def run_chains(engine: str, inputs: SamplerInputs, cov, mix: ProposalMix, *, seed: int,
               start_counts: np.ndarray | None, sweeps: int, G: sp.csr_matrix,
               threads: int | None) -> dict:
    """Every sweep recorded: functionals, log target, totals check, seconds."""
    if engine == "numba":
        from pmedm_vb.latent.sampler_numba import NumbaSampler

        t0 = time.perf_counter()
        sm = NumbaSampler(inputs, cov, mix, seed=seed, start_counts=start_counts,
                          threads=threads)
        setup = time.perf_counter() - t0
        res = sm.run(sweeps, thin=1, G=G, trace=True)
        return {"functionals": res.functionals.astype(np.float32), "log_target": res.log_target,
                "totals_error": res.totals_error, "seconds": res.seconds_per_sweep,
                "acceptance": res.acceptance, "drift": res.drift,
                "moves_per_sweep": res.steps_per_sweep, "threads": res.threads,
                "seconds_setup": setup, "seconds_trace": res.seconds_functionals}
    from pmedm_vb.latent.sampler import RealizedSampler
    from pmedm_vb.latent.sampler_numba import log_target_numpy

    t0 = time.perf_counter()
    sm = RealizedSampler(inputs, cov, mix, seed=seed, start_counts=start_counts)
    setup = time.perf_counter() - t0
    n, Z, M, _ = inputs.shape
    fun, lt, te, secs, drift, t_trace = [], [], [], [], 0.0, 0.0
    for _ in range(sweeps):
        s, d = sm.run_sweeps(1)
        secs.append(s[0])
        drift = max(drift, d)
        t1 = time.perf_counter()
        counts = sm.counts_numpy()
        fun.append((sp.csr_matrix(counts.reshape(n * Z, M)) @ G).toarray()
                   .reshape(n, Z, -1).astype(np.float32))
        lt.append(log_target_numpy(inputs, cov, counts))
        te.append(-1 if counts.min() < 0 else int(np.abs(counts.sum(2) - inputs.N).max()))
        t_trace += time.perf_counter() - t1
    return {"functionals": np.stack(fun), "log_target": np.stack(lt),
            "totals_error": np.asarray(te), "seconds": np.asarray(secs),
            "acceptance": sm.acceptance(), "drift": drift,
            # torch: a block step moves every zone once, a low-rank step one zone
            "moves_per_sweep": sm.steps_per_sweep * (Z if cov.kind == "block" else 1),
            "threads": None, "seconds_setup": setup, "seconds_trace": t_trace}


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def _arviz():
    import logging
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import arviz as az
    logging.getLogger("arviz").setLevel(logging.ERROR)    # one-chain shape warnings
    return az


def mixing(x: np.ndarray) -> dict:
    """``x``: ``(chains, draws, ...)``. Rank-normalized split R̂, bulk and tail
    ESS (arviz), pooled means and variances; NaN where ``x`` is constant."""
    import warnings

    az = _arviz()
    x = np.asarray(x, float)
    ds = az.convert_to_dataset({"x": x})
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")
    var = x.reshape(-1, *x.shape[2:]).var(axis=0, ddof=1)
    const = ~(var > 1e-12 * np.maximum(np.abs(x.mean((0, 1))), 1.0) ** 2)
    out = {"mean": x.mean((0, 1)), "var": var, "constant": const}
    for key, kw in (("rhat", {"method": "rank"}), ("ess_bulk", {"method": "bulk"}),
                    ("ess_tail", {"method": "tail"})):
        f = az.rhat if key == "rhat" else az.ess
        v = np.asarray(f(ds, **kw)["x"].values, float)
        out[key] = np.where(const, np.nan, v)
    return out


def class_table(cls: np.ndarray, mix_: dict, draws_total: int, seconds_kept: float) -> pd.DataFrame:
    """Per class: median and minimum ESS per sweep (bulk, tail), ESS per
    second, R̂ (max, share < 1.01), constants."""
    rows = []
    Zc = np.broadcast_to(cls[None, :], mix_["rhat"].shape)
    for c in list(pd.unique(cls)) + ["all"]:
        sel = np.ones(Zc.shape, bool) if c == "all" else Zc == c
        live = sel & ~mix_["constant"]
        eb, et, rh = mix_["ess_bulk"][live], mix_["ess_tail"][live], mix_["rhat"][live]
        row = {"cls": c, "n": int(sel.sum()), "constant": int((sel & mix_["constant"]).sum())}
        if live.any():
            row.update({
                "ess_bulk_per_sweep_median": float(np.median(eb)) / draws_total,
                "ess_bulk_per_sweep_min": float(np.min(eb)) / draws_total,
                "ess_tail_per_sweep_median": float(np.median(et)) / draws_total,
                "ess_tail_per_sweep_min": float(np.min(et)) / draws_total,
                "ess_bulk_per_second_median": float(np.median(eb)) / max(seconds_kept, 1e-300),
                "rhat_max": float(np.max(rh)), "rhat_share_below": float(np.mean(rh < RHAT_MAX))})
        rows.append(row)
    return pd.DataFrame(rows)


def burn_in(lt: np.ndarray) -> dict:
    """``lt``: ``(T, chains)`` log target. Each chain's first crossing of the
    pooled second half's median, and R̂ of the log target over ``[t, T)``
    (module notes)."""
    T, n = lt.shape
    med = float(np.median(lt[T // 2:]))
    crossing = []
    for k in range(n):
        side = np.sign(lt[:, k] - med)
        changed = np.flatnonzero(side != side[0])
        crossing.append(int(changed[0]) if changed.size else T)
    grid = sorted({int(t) for t in np.linspace(0, T // 2, 11)})
    rh = {}
    for t in grid:
        if T - t >= 8 and n >= 2:
            rh[t] = float(mixing(lt[t:].T[:, :, None])["rhat"][0])
    ok = [t for t, r in rh.items() if t >= max(crossing) and np.isfinite(r) and r < RHAT_MAX]
    return {"median_second_half": med, "first_crossing": crossing, "rhat_from": rh,
            "suggested_burn": min(ok) if ok else None}


def agreement(mean_s, var_s, ess, mean_g, var_g, cls) -> tuple[pd.DataFrame, dict]:
    """Stage 2 against the chains (module notes), functionals with stage 2
    mean ≥ 50 and positive variance."""
    Zc = np.broadcast_to(cls[None, :], mean_g.shape)
    sel = (mean_g >= AGREE_MIN_MEAN) & (var_g > 0) & np.isfinite(ess)
    sd_g = np.sqrt(np.where(var_g > 0, var_g, np.nan))
    dm = (mean_s - mean_g) / sd_g
    ratio = var_s / np.where(var_g > 0, var_g, np.nan)
    se_dm = np.sqrt(var_s / np.maximum(ess, 1.0)) / sd_g
    se_ratio = ratio * np.sqrt(2.0 / np.maximum(ess, 1.0))
    ok_m = np.abs(dm) <= AGREE_MEAN_SD
    ok_v = (ratio >= AGREE_VAR[0]) & (ratio <= AGREE_VAR[1])
    rows = []
    for c in list(pd.unique(cls)) + ["all"]:
        s_ = sel & ((Zc == c) if c != "all" else True)
        if not s_.any():
            continue
        rows.append({"cls": c, "n": int(s_.sum()),
                     "share_mean_ok": float(ok_m[s_].mean()), "share_var_ok": float(ok_v[s_].mean()),
                     "share_both_ok": float((ok_m & ok_v)[s_].mean()),
                     "median_abs_dm_sd": float(np.median(np.abs(dm[s_]))),
                     "max_abs_dm_sd": float(np.max(np.abs(dm[s_]))),
                     "median_mc_se_dm_sd": float(np.median(se_dm[s_])),
                     "var_ratio_median": float(np.median(ratio[s_])),
                     "var_ratio_min": float(np.min(ratio[s_])),
                     "var_ratio_max": float(np.max(ratio[s_])),
                     "median_mc_se_ratio": float(np.median(se_ratio[s_]))})
    tab = pd.DataFrame(rows)
    allrow = tab[tab["cls"] == "all"]
    verdict = {"n": int(sel.sum()),
               "pass": bool(len(allrow) and allrow["share_both_ok"].iloc[0] >= AGREE_SHARE)}
    return tab, verdict


def acceptance_text(acc: dict) -> str:
    return ", ".join(f"{k} {acc[k]['rate']:.3f} (non-null {acc[k]['rate_non_null']:.3f})"
                     for k in PROPOSALS if acc[k]["proposed"])


# --------------------------------------------------------------------------
# One region
# --------------------------------------------------------------------------


def pilot_region(unit: PilotUnit, args, out: Path) -> dict:
    from pmedm_vb.progress import logger, save_npz, write_text

    out.mkdir(parents=True, exist_ok=True)
    mix = ProposalMix(*args.mix)
    n = args.chains
    T = args.burn + args.sweeps
    seed_base = [args.seed if args.seed is not None else 0, PILOT_TAG,
                 sum(map(ord, unit.name))]
    rng = np.random.default_rng(seed_base)
    lines = [f"step 5 pilot: {unit.name} ({unit.info.get('simulation')}), engine {args.engine}",
             "unit: " + json.dumps({k: v for k, v in unit.info.items()
                                    if not isinstance(v, list)}, default=float),
             f"zone totals N_b = {unit.N.tolist()} (sum {int(unit.N.sum())}); atoms on the "
             f"support {unit.T0.shape[0]}, cells per zone {unit.T0.shape[1]}, largest row nnz "
             f"{int(np.diff(unit.T0.indptr).max())}; kappa {unit.kappa:.1f}; functionals traced "
             f"{unit.G.shape[1]} per zone; proposal mix "
             f"{[round(float(w), 3) for w in mix.weights()]}"]

    # -- MAP parameters, dispersed starts -----------------------------------
    inputs = unit.inputs(n)
    starts = dispersed_starts(unit, n, rng)
    t0 = time.perf_counter()
    ch = run_chains(args.engine, inputs, unit.cov, mix, seed=int(rng.integers(2 ** 31)),
                    start_counts=starts, sweeps=T, G=unit.G, threads=args.threads)
    wall = time.perf_counter() - t0
    fun, lt = ch["functionals"], ch["log_target"]
    kept = slice(args.burn, T)
    mx = mixing(np.moveaxis(fun[kept], 0, 1))                        # (Z, F)
    draws_total = n * (T - args.burn)
    secs = ch["seconds"]
    seconds_kept = float(secs[kept].sum())
    tab = class_table(unit.cls, mx, draws_total, seconds_kept)
    lt_mix = mixing(lt[kept].T[:, :, None])
    bi = burn_in(lt)
    agree_tab, agree = agreement(mx["mean"], mx["var"], mx["ess_bulk"], unit.stage2_mean,
                                 unit.stage2_var, unit.cls)
    live = tab[(tab["cls"] != "all") & tab["ess_bulk_per_sweep_median"].notna()]
    rh_all = tab.loc[tab["cls"] == "all", "rhat_max"].iloc[0] if "rhat_max" in tab else np.nan
    mixing_pass = bool(len(live) and (live["ess_bulk_per_sweep_median"] >= ESS_PER_SWEEP_MIN).all()
                       and np.isfinite(rh_all) and rh_all < RHAT_MAX)
    Zi, Fi = np.meshgrid(np.arange(unit.Z), np.arange(unit.G.shape[1]), indexing="ij")
    frame = pd.DataFrame({"zone": Zi.ravel(), "cls": unit.cls[Fi.ravel()],
                          "functional": unit.names[Fi.ravel()],
                          "mean": mx["mean"].ravel(), "var": mx["var"].ravel(),
                          "rhat": mx["rhat"].ravel(), "ess_bulk": mx["ess_bulk"].ravel(),
                          "ess_tail": mx["ess_tail"].ravel(), "constant": mx["constant"].ravel(),
                          "stage2_mean": unit.stage2_mean.ravel(),
                          "stage2_var": unit.stage2_var.ravel()})
    write_text(out / "map_functionals.csv", frame.to_csv(index=False))
    write_text(out / "map_log_target.csv", pd.DataFrame(
        {"sweep": np.arange(1, T + 1), "seconds": secs, "totals_error": ch["totals_error"],
         **{f"chain{k}_{nm}": lt[:, k] for k, nm in enumerate(start_names(n))}}
    ).to_csv(index=False))
    save_npz(out / "map_traces.npz", functionals=fun, log_target=lt, seconds=secs,
             totals_error=ch["totals_error"], cls=unit.cls, names=unit.names, N=unit.N,
             starts=np.array(start_names(n)), burn=args.burn)
    lines += ["", f"== MAP parameters: {n} chains (starts {start_names(n)}), {T} sweeps "
              f"({args.burn} burn-in), {ch['moves_per_sweep']} moves per chain per sweep, "
              f"threads {ch['threads']}",
              f"  seconds per sweep: median {np.median(secs):.3f}, mean {secs.mean():.3f} "
              f"(moves + refresh); traces {ch['seconds_trace']:.1f} s in all; set-up "
              f"{ch['seconds_setup']:.1f} s; wall {wall:.0f} s",
              f"  moves per second (all chains): {n * ch['moves_per_sweep'] / np.median(secs):.3g}",
              f"  acceptance: {acceptance_text(ch['acceptance'])}",
              f"  zone totals check: max |sum C_b - N_b| over sweeps = "
              f"{int(np.max(ch['totals_error']))} (0 = holds); drift {ch['drift']:.1e}",
              f"  log target: R-hat {lt_mix['rhat'][0]:.4f}, bulk ESS {lt_mix['ess_bulk'][0]:.0f} "
              f"({lt_mix['ess_bulk'][0] / draws_total:.3f} per sweep)",
              f"  burn-in: first crossing of the second half's median per chain "
              f"{bi['first_crossing']}; R-hat of the log target from sweep t: "
              + ", ".join(f"{t}: {r:.3f}" for t, r in bi["rhat_from"].items())
              + f"; suggested burn-in {bi['suggested_burn']} (used {args.burn})",
              "  mixing by class (ESS per sweep = ESS / (chains x kept sweeps)):",
              _indent(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}")),
              f"  DECISION (plan §5): median bulk ESS per sweep >= {ESS_PER_SWEEP_MIN} in every "
              f"class and max R-hat < {RHAT_MAX}: {'PASS' if mixing_pass else 'FAIL'}",
              "", f"== agreement with stage 2 at the MAP parameters (stage 2 mean >= "
              f"{AGREE_MIN_MEAN:g}; |dm| <= {AGREE_MEAN_SD} sd, variance ratio in {AGREE_VAR}; "
              f"MC se from bulk ESS):",
              _indent(agree_tab.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
              if len(agree_tab) else "  (no functional with stage 2 mean >= 50)",
              f"  {AGREE_SHARE:.0%} of the {agree['n']} with both: "
              f"{'PASS' if agree['pass'] else 'FAIL'}"]

    # -- parameter draws ----------------------------------------------------
    draws_summary = None
    if unit.draws:
        dinp = unit.draw_inputs()
        nd = len(unit.draws)
        Td = args.burn + (args.draw_sweeps or args.sweeps)
        t1 = time.perf_counter()
        dch = run_chains(args.engine, dinp, unit.cov, mix, seed=int(rng.integers(2 ** 31)),
                         start_counts=None, sweeps=Td, G=unit.G, threads=args.threads)
        dk = slice(args.burn, Td)
        per = []
        rows = []
        for d in range(nd):
            m1 = mixing(np.moveaxis(dch["functionals"][dk, d:d + 1], 0, 1))
            t_ = class_table(unit.cls, m1, Td - args.burn, float(dch["seconds"][dk].sum()))
            t_.insert(0, "draw", d)
            per.append(t_)
            rows.append(pd.DataFrame({"draw": d, "zone": Zi.ravel(), "cls": unit.cls[Fi.ravel()],
                                      "functional": unit.names[Fi.ravel()],
                                      "mean": m1["mean"].ravel(), "var": m1["var"].ravel(),
                                      "rhat_split": m1["rhat"].ravel(),
                                      "ess_bulk": m1["ess_bulk"].ravel(),
                                      "ess_tail": m1["ess_tail"].ravel()}))
        per = pd.concat(per, ignore_index=True)
        write_text(out / "draws_functionals.csv", pd.concat(rows).to_csv(index=False))
        save_npz(out / "draws_traces.npz", functionals=dch["functionals"],
                 log_target=dch["log_target"], seconds=dch["seconds"],
                 totals_error=dch["totals_error"], N=np.stack([d["N"] for d in unit.draws]),
                 cls=unit.cls, names=unit.names, burn=args.burn)
        med = per[per["cls"] != "all"].pivot(index="draw", columns="cls",
                                             values="ess_bulk_per_sweep_median")
        draws_summary = {"draws": nd, "sweeps": Td, "acceptance": dch["acceptance"],
                         "seconds_per_sweep": float(np.median(dch["seconds"])),
                         "ess_bulk_per_sweep_median_by_draw": med.round(4).to_dict(orient="index"),
                         "totals_error_max": int(np.max(dch["totals_error"])),
                         "drift": dch["drift"], "wall": time.perf_counter() - t1}
        lines += ["", f"== {nd} parameter draws of the cut posterior, one chain each (start "
                  f"Multinomial(N_b, r_hat_b); r_hat {unit.info.get('draw_rhat')}), {Td} sweeps "
                  f"({args.burn} burn-in); median {np.median(dch['seconds']):.3f} s per sweep for "
                  "all draws together",
                  f"  totals per draw: {[d['N'].tolist() for d in unit.draws]}",
                  f"  acceptance: {acceptance_text(dch['acceptance'])}; totals check "
                  f"{draws_summary['totals_error_max']}; drift {dch['drift']:.1e}",
                  "  median bulk ESS per sweep by class (rows: draws; MAP chains for reference "
                  "below):",
                  _indent(med.to_string(float_format=lambda v: f"{v:.4f}")),
                  _indent(live.set_index("cls")["ess_bulk_per_sweep_median"].to_frame("MAP")
                          .T.to_string(float_format=lambda v: f"{v:.4f}"))]

    # -- torch comparison -----------------------------------------------------
    torch_summary = None
    if args.torch_chains > 0:
        from pmedm_vb.latent.sampler import RealizedSampler

        k = args.torch_chains
        sm = RealizedSampler(unit.inputs(k), unit.cov, mix, seed=1)
        sm.time_steps(10)
        per_step = sm.time_steps(args.torch_steps)
        tsweep = per_step * sm.steps_per_sweep
        torch_summary = {"chains": k, "seconds_per_step": per_step,
                         "steps_per_sweep": sm.steps_per_sweep, "seconds_per_sweep": tsweep,
                         "moves_per_second": k * unit.Z / per_step}
        nmps = n * ch["moves_per_sweep"] / float(np.median(secs))
        lines += ["", f"== torch, {k} chains: {1e3 * per_step:.2f} ms per step (every zone of "
                  f"every chain moves once), {tsweep:.1f} s per torch sweep ({sm.steps_per_sweep} "
                  f"steps); {k * unit.Z / per_step:.3g} moves/s vs numba's {nmps:.3g} with {n} "
                  "chains"]

    summary = {
        "region": unit.name, "engine": args.engine, "info": unit.info, "chains": n,
        "sweeps": args.sweeps, "burn": args.burn, "mix": list(mix.weights()),
        "moves_per_sweep": ch["moves_per_sweep"], "threads": ch["threads"],
        "seconds_per_sweep_median": float(np.median(secs)), "wall": wall,
        "acceptance": ch["acceptance"], "drift": ch["drift"],
        "totals_error_max": int(np.max(ch["totals_error"])),
        "log_target": {"rhat": float(lt_mix["rhat"][0]), "ess_bulk": float(lt_mix["ess_bulk"][0])},
        "burn_in": bi, "mixing": tab.to_dict(orient="records"), "mixing_pass": mixing_pass,
        "agreement": agree_tab.to_dict(orient="records"), "agreement_pass": agree["pass"],
        "draws": draws_summary, "torch": torch_summary}
    text = "\n".join(lines) + "\n"
    write_text(out / "summary.txt", text)
    write_text(out / "pilot.json", json.dumps(summary, indent=1, default=_json_default))
    print(text, flush=True)
    logger.info("pilot %s: mixing %s, agreement %s", unit.name,
                "PASS" if mixing_pass else "FAIL", "PASS" if agree["pass"] else "FAIL")
    return summary


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    import step4d_free_shares as s4d

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    s4d.common_arguments(parser)
    parser.set_defaults(region=list(DEFAULT_REGIONS))
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=2000, help="kept sweeps (after --burn)")
    parser.add_argument("--burn", type=int, default=500, help="burn-in sweeps (also traced)")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--engine", default="numba", choices=("numba", "torch"))
    parser.add_argument("--mix", type=lambda x: [float(v) for v in x.split(",")],
                        default=[0.4, 0.3, 0.3], help="proposal weights a,b,c")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--param-draws", type=int, default=4)
    parser.add_argument("--draw-sweeps", type=int, default=None,
                        help="kept sweeps per parameter draw (default --sweeps)")
    parser.add_argument("--draw-rhat", default="refit", choices=("refit", "map"))
    parser.add_argument("--torch-chains", type=int, default=0)
    parser.add_argument("--torch-steps", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true",
                        help="made-up units, no project data (a dry run)")
    parser.add_argument("--synthetic-atoms", type=int, default=3000)
    parser.add_argument("--synthetic-N", type=int, default=6000)
    if "--synthetic" in argv:                          # the data arguments are not needed
        for action in parser._actions:
            if action.required and action.dest != "out":
                action.required = False
    args = parser.parse_args(argv)
    if len(args.mix) != 3:
        parser.error("--mix takes three weights")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    from pmedm_vb.progress import logger

    if args.engine == "numba":
        import numba

        if args.threads:
            numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
        logger.info("numba %s, %d threads (NUMBA_NUM_THREADS %d)", numba.__version__,
                    numba.get_num_threads(), numba.config.NUMBA_NUM_THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        if args.seed is None:
            args.seed = 0
        todo = [(f"synthetic_z{Z}", Z) for Z in (4, 6)]
        for name, Z in todo:
            out = args.out / name
            if (out / "pilot.json").exists():
                logger.info("%s: done", name)
                continue
            unit = build_synthetic_unit(name, Z, args, args.seed)
            pilot_region(unit, args, out)
        return
    import step4c_joint as s4c
    import step4d_free_shares as s4d

    pending = [r for r in args.region if not (args.out / r / "pilot.json").exists()]
    for r in set(args.region) - set(pending):
        logger.info("%s: done", r)
    if not pending:
        return
    args.region = pending
    _regions, chosen, choice = s4d.setup(args, {
        "step": "5 pilot", "pilot_tag": PILOT_TAG, "replicate": args.replicate,
        "sweeps": args.sweeps, "burn": args.burn, "chains": args.chains, "engine": args.engine,
        "mix": args.mix, "max_per_class": args.max_per_class, "param_draws": args.param_draws,
        "draw_rhat": args.draw_rhat})
    s4c._SHARED["choice"] = choice
    for rid in chosen:
        unit = build_real_unit(rid, args)
        pilot_region(unit, args, args.out / rid)


if __name__ == "__main__":
    main()
