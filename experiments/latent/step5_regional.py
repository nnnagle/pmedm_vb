"""Step 5, phase 3: the regional run of the realized-population sampler, arm
C3 (``PLAN_latent_step5.md`` version 3, §4 item 3, §5's phase 3 bullets,
§8c's settings; :mod:`pmedm_vb.latent.sampler_numba`; model
``MODEL_latent_v5_draft.md`` §7, the cut form).

    R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
    PYTHONPATH=src python experiments/latent/step5_regional.py \\
        --config experiments/embed/stage1_v2.json --regions experiments/latent/regions_tn.json \\
        --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \\
        --step3 $R/latent/step3 --step3b $R/latent/step3b --step4 $R/latent/step4 \\
        --step4c $R/latent/step4c --out $R/latent/step5 --threads 16 [--region <id> ...]
    python experiments/latent/collect_step5.py --root $R/latent/step5 \\
        --step4e $R/latent/step4e --step4d $R/latent/step4d --step4c $R/latent/step4c
    # the dry run, no project data needed:
    PYTHONPATH=src python experiments/latent/step5_regional.py --synthetic --out /tmp/c3 \\
        --threads 4 --param-draws 8 --map-burn 100 --map-sweeps 200

**Units:** every ``(region, replicate)`` of simulation S at step 4c's chosen
``(τ, c)`` for S (``--step4c``'s ``choice.json``;
:func:`step4d_free_shares.units_at_s_pairs`: all regions, ``--replicates``
(10) replicates; ``--region`` narrows the regions), one unit at a time in one
process, the sampler on ``--threads`` (16) numba threads. Output
``<out>/C3/<region>/r<k>/``; a unit with ``done.json`` is skipped, so a
timed-out job can be resubmitted unchanged.

**The unit** is built by the pilot's :func:`step5_pilot.build_real_unit`
(its module notes: S's data by step 4c's ``Replicate``, the totals module,
``Û`` the MAP given the module's mean, zone totals rounded to integers
summing to ``N_A``, the realized MAP ``r̂`` with those totals fixed, stage
2's moments at the MAP parameters on the traced functionals, the version 2
inputs of :func:`step5_pilot.prepare_version2`: the ``hhsize`` class and
the ``kern30`` kernel), with ``--param-draws`` (50) parameter draws of the
cut posterior, each with its own ``r̂`` (``--draw-rhat refit``), the draws'
seed tagged :data:`STEP5_TAG` (seeded as step 4e's and the pilot's:
``(seed, S, region, replicate, 2, τ multiple, κ, tag)``).

**Point scores** of the realized MAP, as ``step4e_cut.fit_unit`` writes them
(:func:`step4d_free_shares.write_scores`: ``scores.csv``, ``rare.csv``,
``misfit.csv``, ``transfer.csv``; ``fits.csv``, ``starts.csv``). One
difference from step 4e: C3's realized MAP is at the *rounded* totals (the
sampler's), step 4e's at the module's mean (under one unit per zone apart),
and the prior ``q`` is taken at the same rounded shares.

**MAP chains** (``--chains`` 4, the pilot's dispersed starts,
``--map-burn`` 300 + ``--map-sweeps`` 700 sweeps, every sweep traced on the
pilot's traced functionals, at most ``--max-per-class`` 100 per class):
per class the median bulk and tail ESS per sweep, ``R̂`` (max, share below
1.01 and 1.05) (:func:`step5_pilot.mixing`, :func:`step5_pilot.class_table`);
the log target's burn-in diagnostic (:func:`step5_pilot.burn_in`); and the
agreement with stage 2 at the MAP parameters (:func:`step5_pilot.agreement`,
§5's phase 3 rule: cells with stage 2 mean ≥ 50, ``|mean difference| ≤ 0.1``
stage 2 sd and variance ratio in ``[0.8, 1.25]``; the share passing both,
the rule's threshold 90%). Written: ``mixing.csv``, ``agreement.csv`` (the
class tables) and ``map_functionals.csv.gz`` (per zone and traced
functional: the chains' mean, variance, ``R̂``, ESS, stage 2's moments; what
``collect_step5.py`` pools), ``map_log_target.csv``.

**Draw chains:** one chain per parameter draw, warm-started from the MAP
chains' final state (:func:`step5_pilot.warm_starts`; draw ``d`` from MAP
chain ``d mod chains``), ``--reburn`` (30) sweeps, then ``--draws-per-chain``
(4) stored states ``--thin`` (25) sweeps apart (the states after sweeps
``reburn + thin, …, reburn + 4 thin``): 200 states of ``C`` per unit. Moves
as §8c: single-unit mix ``--mix`` (0.3/0.2/0.2/0.3, kernel ``kern30``), one
pile move per unit per sweep (``--pile-moves 1N``), first record from the
occupied ones, second from ``r̂_b``. The log target and the traced
functionals are recorded at every sweep (whatever the moves did), and the
pilot's re-burn diagnostic (:func:`step5_pilot.reburn`) and the draws'
one-chain ESS per sweep after the re-burn are reported.

*Batches and streams.* The draw chains run ``--draw-batch`` (25) at a time
in one :class:`~pmedm_vb.latent.sampler_numba.NumbaSampler` (memory: the
kernel tables are ``16 K M`` bytes per (draw, zone) weight row, ``K = 30``
centres, plus about ``728 M`` bytes shared; the third pilot's 1,811 MiB for
24 rows gives ``M`` = 155,000 atoms on nashville_core's support: 74 MB per
row, so 6 zones take 11.2 GB for a batch of 25 and 22.3 GB for all 50). The
sampler binds one xoshiro256** stream to each slot (chain × zone), seeded
from the sampler's seed and the slot's *position*, so a chain's stream would
depend on the batch. So the streams are replaced after construction (the
sampler's ``rng`` array, before any move): draw ``d``'s zone ``b`` gets the
state :func:`~pmedm_vb.latent.sampler_numba.stream_states` gives stream ``b``
of seed ``s_d``, ``s_d`` drawn from ``(seed, S, region, replicate,
STEP5_TAG, 4, d)``; the warm start's adjustment uses its own generator
``(…, 3, d)``. Slots are independent in the block backend, so each draw's
chain, and every output, is the same for any ``--draw-batch`` (checked by
the dry run with batches of 1, 4 and 6 draws: identical states, functionals
and files, except that with batches of one draw the log target's trace
differs by about 1e-16 relative: the sampler's refresh computes ``e P_b``
for all of a zone's slots in one BLAS call, a matrix-vector product when
there is one slot, whose rounding differs; such a difference could in
principle flip an acceptance decision, which was not seen). No change to
the sampler's code was needed.

**Posterior:** each stored state's functionals over *all* the functionals of
step 4c–4e (``G'C_b``, the ``hhsize`` class appended; on the support, i.e.
the same values as on all atoms since the others hold no units), the
``(200, Z, F)`` array written by :func:`step4d_free_shares.write_posterior`
(``posterior.csv.gz``, scored exactly as steps 4c–4e). Checked per stored
state: zone totals equal to that draw's ``N_b`` exactly, no negative count,
and its traced columns equal to the trace at that sweep. ``--save-counts``
also writes the stored states' counts (``counts.npz``: CSR parts of the
``(200 Z, M)`` int32 matrix, rows ``(state, zone)``, columns the support's
atoms, ``keep`` their indices among all atoms; default off).

**Files:** as step 4e's units (``posterior.csv.gz``, the scores, ``fits``,
``starts``, ``map.npz``, ``done.json``) and the ones above;
``draws_log_target.csv`` (per draw chain and sweep). ``done.json`` (written
last) carries step 4e's keys (so ``collect_step4d.report`` reads it) and the
timings (set-up and its parts, the ``r̂`` refits, MAP chains, draw chains),
peak memory, the mixing and agreement summaries, the re-burn, the checks
and the settings.

**Synthetic** (``--synthetic``): the same pipeline on the pilot's made-up
units (:func:`step5_pilot.build_synthetic_unit`, ``synthetic_z4`` and
``synthetic_z6``, ``--replicates`` default 1, with ``--param-draws`` made-up
parameter draws, ``κ = --synthetic-kappa-c × n_A``), no project data.
``write_scores`` and ``write_posterior`` need the project's data, so they
are not called: ``posterior.csv.gz`` holds the same interval statistics
against the made-up truth (:func:`~pmedm_vb.latent.calibration.interval_stats`),
``samples.npz`` the ``(200, Z, F)`` array, and, as an integration check,
``agreement_draws.csv`` compares the 200 states with stage 2's Gaussian at
each of the *same* parameter draws (:func:`step5_pilot.dense_conditional_moments`):
the pooled mixture's mean and variance (§5's thresholds) and z-scores of
the differences against their Monte Carlo errors (module notes of
:func:`draws_agreement`). Not exercised: Replicate, the MAP of ``U``, the
totals module, the scores, step 4e's posterior file layout.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import step5_pilot as s5p
from pmedm_vb.latent.sampler import SamplerInputs
from pmedm_vb.latent.sampler_numba import PILE_N_BINS, PILE_STATS, ProposalMix4, stream_states

ARM = "C3"
#: Tag of the parameter draws' and the chains' seeds, apart from steps 4c–4e
#: (4003–4005) and the pilot (5002).
STEP5_TAG = 4006
#: The synthetic "regions" (name, zones), as the pilot's dry run.
SYNTHETIC = (("synthetic_z4", 4), ("synthetic_z6", 6))


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------


def parse_args(argv=None):
    import step4d_free_shares as s4d

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    s4d.common_arguments(parser)
    parser.set_defaults(threads=16, replicates=None, draws=None)
    parser.add_argument("--chains", type=int, default=4, help="MAP chains")
    parser.add_argument("--map-burn", type=int, default=300)
    parser.add_argument("--map-sweeps", type=int, default=700, help="kept MAP sweeps")
    parser.add_argument("--mix", type=lambda x: [float(v) for v in x.split(",")],
                        default=[0.3, 0.2, 0.2, 0.3], help="proposal weights a,b,c,e")
    parser.add_argument("--kernel-basis", default="kern30", choices=("kern30", "kern100"))
    parser.add_argument("--kernel-trunc", type=float, default=0.0)
    parser.add_argument("--kernel-weights", default="p", choices=("p", "r_hat"))
    parser.add_argument("--pile-moves", default="1N")
    parser.add_argument("--pile-first", default="occupied", choices=("r_hat", "occupied"))
    parser.add_argument("--pile-second", default="r_hat", choices=("kernel", "r_hat"))
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--param-draws", type=int, default=50)
    parser.add_argument("--draw-rhat", default="refit", choices=("refit", "map"))
    parser.add_argument("--reburn", type=int, default=30)
    parser.add_argument("--draws-per-chain", type=int, default=4)
    parser.add_argument("--thin", type=int, default=25)
    parser.add_argument("--draw-batch", type=int, default=25,
                        help="draw chains per sampler (memory; results do not depend on it)")
    parser.add_argument("--save-counts", action="store_true",
                        help="also write the stored states' counts (counts.npz)")
    parser.add_argument("--save-traces", action="store_true",
                        help="also write the MAP and draw chains' functional traces")
    parser.add_argument("--synthetic", action="store_true",
                        help="made-up units, no project data (a dry run)")
    parser.add_argument("--synthetic-atoms", type=int, default=3000)
    parser.add_argument("--synthetic-N", type=int, default=6000)
    parser.add_argument("--synthetic-kappa-c", type=float, default=0.25,
                        help="synthetic units: kappa = c n_A (the pilot's 0.25)")
    if "--synthetic" in argv:                          # the data arguments are not needed
        for action in parser._actions:
            if action.required and action.dest != "out":
                action.required = False
    args = parser.parse_args(argv)
    if len(args.mix) == 3:
        args.mix = [*args.mix, 0.0]
    if len(args.mix) != 4:
        parser.error("--mix takes four weights a,b,c,e (or three, with e = 0)")
    try:
        s5p.parse_pile_moves(args.pile_moves)
    except ValueError as err:
        parser.error(f"--pile-moves: {err}")
    if args.pile_first == "r_hat" and args.pile_second != "kernel":
        parser.error("--pile-second r_hat needs --pile-first occupied")
    for name in ("chains", "param_draws", "draws_per_chain", "thin", "draw_batch", "map_sweeps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.reburn < 0 or args.map_burn < 0:
        parser.error("--reburn and --map-burn must be non-negative")
    n_states = args.param_draws * args.draws_per_chain
    if args.draws is None:
        args.draws = n_states
    elif args.draws != n_states:
        parser.error(f"--draws {args.draws} differs from --param-draws x --draws-per-chain "
                     f"= {n_states}")
    if args.replicates is None:
        args.replicates = 1 if args.synthetic else 10
    if args.workers != 1:
        parser.error("--workers: units run one at a time in one process (use --threads)")
    return args


def settings(args) -> dict:
    return {"step": "5 regional (phase 3)", "arm": ARM, "step5_tag": STEP5_TAG,
            "chains": args.chains, "map_burn": args.map_burn, "map_sweeps": args.map_sweeps,
            "mix": args.mix, "kernel_basis": args.kernel_basis,
            "kernel_trunc": args.kernel_trunc, "kernel_weights": args.kernel_weights,
            "pile_moves": args.pile_moves, "pile_first": args.pile_first,
            "pile_second": args.pile_second, "max_per_class": args.max_per_class,
            "param_draws": args.param_draws, "draw_rhat": args.draw_rhat,
            "draw_start": "warm", "reburn": args.reburn,
            "draws_per_chain": args.draws_per_chain, "thin": args.thin,
            "draw_batch": args.draw_batch, "save_counts": args.save_counts,
            "threads": args.threads, "synthetic": args.synthetic,
            **({"synthetic_atoms": args.synthetic_atoms, "synthetic_N": args.synthetic_N,
                "synthetic_kappa_c": args.synthetic_kappa_c} if args.synthetic else {})}


# --------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------


def peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def moves(args, unit: s5p.PilotUnit):
    """The proposal mix, the kernel (needed: kernel weight > 0 or kernel pile
    pairs) and the pile-move spec."""
    mix = ProposalMix4(*args.mix)
    spec = s5p.parse_pile_moves(args.pile_moves)
    kspec = None
    if mix.kernel > 0 or (s5p.pile_counts(spec, unit.N).max() > 0
                          and args.pile_second == "kernel"):
        kspec = unit.kernel_spec(args.kernel_trunc, args.kernel_weights)
    return mix, kspec, spec


def map_chains(unit: s5p.PilotUnit, args, rng: np.random.Generator) -> dict:
    """The MAP chains and their diagnostics (module notes)."""
    mix, kspec, spec = moves(args, unit)
    n = args.chains
    T = args.map_burn + args.map_sweeps
    inputs = unit.inputs(n)
    starts = s5p.dispersed_starts(unit, n, rng)
    t0 = time.perf_counter()
    ch = s5p.run_chains("numba", inputs, unit.cov, mix, seed=int(rng.integers(2 ** 31)),
                        start_counts=starts, sweeps=T, G=unit.G, threads=args.threads,
                        kernel=kspec, piles=s5p.pile_counts(spec, inputs.N),
                        pile_first=args.pile_first, pile_second=args.pile_second)
    wall = time.perf_counter() - t0
    kept = slice(args.map_burn, T)
    mx = s5p.mixing(np.moveaxis(ch["functionals"][kept], 0, 1))           # (Z, F)
    secs = ch["seconds"]
    tab = s5p.class_table(unit.cls, mx, n * args.map_sweeps, float(secs[kept].sum()))
    shares105 = []
    for c in tab["cls"]:
        sel = (np.broadcast_to(unit.cls[None, :], mx["rhat"].shape) == c) if c != "all" \
            else np.ones(mx["rhat"].shape, bool)
        live = sel & ~mx["constant"]
        shares105.append(float(np.mean(mx["rhat"][live] < s5p.RHAT_ALL_MAX)) if live.any()
                         else np.nan)
    tab["rhat_share_below_1.05"] = shares105
    lt_mix = s5p.mixing(ch["log_target"][kept].T[:, :, None])
    bi = s5p.burn_in(ch["log_target"])
    agree_tab, agree = s5p.agreement(mx["mean"], mx["var"], mx["ess_bulk"], unit.stage2_mean,
                                     unit.stage2_var, unit.cls)
    return {"ch": ch, "mx": mx, "table": tab, "lt_mix": lt_mix, "burn_in": bi,
            "agreement_table": agree_tab, "agreement": agree, "wall": wall,
            "starts": s5p.start_names(n)}


def batch_inputs(unit: s5p.PilotUnit, idx: list[int]) -> SamplerInputs:
    dr = [unit.draws[d] for d in idx]
    return SamplerInputs(unit.T0, unit.Y, np.stack([d["N"] for d in dr]),
                         np.stack([d["p"] for d in dr]), np.stack([d["r_hat"] for d in dr]),
                         unit.kappa)


def draw_seed(seed_base: list, d: int) -> int:
    """Draw ``d``'s sampler seed (module notes): a function of the unit's seed
    base and ``d`` only."""
    return int(np.random.default_rng([*seed_base, 4, d]).integers(2 ** 63))


def warm_start(map_counts: np.ndarray, unit: s5p.PilotUnit, d: int, seed_base: list
               ) -> tuple[np.ndarray, dict]:
    """Draw ``d``'s start: the pilot's :func:`~step5_pilot.warm_starts` from
    MAP chain ``d mod chains``, with the generator ``(seed_base, 3, d)``."""
    k = d % map_counts.shape[0]
    one = types.SimpleNamespace(draws=[unit.draws[d]])
    start, info = s5p.warm_starts(map_counts[k:k + 1], one,
                                  np.random.default_rng([*seed_base, 3, d]))
    return start[0], {"source_chain": k, "added": info["added"][0],
                      "removed": info["removed"][0],
                      "removed_off_support": info["removed_off_support"][0]}


def draw_chains(unit: s5p.PilotUnit, args, map_counts: np.ndarray, seed_base: list,
                G_all: sp.csr_matrix, cols: np.ndarray) -> dict:
    """The parameter draws' chains in batches (module notes): the stored
    states' functionals ``(D·K, Z, F)`` (draw-major), the log target at every
    sweep ``(T, D)``, the checks and the statistics. ``cols``: the traced
    functionals' columns of ``G_all`` (``unit.G = G_all[:, cols]``)."""
    from pmedm_vb.latent.sampler_numba import NumbaSampler

    mix, kspec, spec = moves(args, unit)
    D, Z, K = len(unit.draws), unit.Z, args.draws_per_chain
    T = args.reburn + K * args.thin
    store = {args.reburn + args.thin * (j + 1): j for j in range(K)}      # sweep (1-based) -> j
    G_all = sp.csr_matrix(G_all, dtype=float)
    F, Ft = G_all.shape[1], unit.G.shape[1]
    cols = np.asarray(cols)
    if (G_all[:, cols] != sp.csr_matrix(unit.G)).nnz:
        raise RuntimeError("the traced functionals are not G_all's columns cols")
    Gt = sp.csr_matrix(unit.G, dtype=float)
    samples = np.empty((D * K, Z, F))
    lt = np.empty((T, D))
    te = np.empty((T, D), dtype=np.int64)
    secs = np.empty((T, (D + args.draw_batch - 1) // args.draw_batch))
    trace = np.empty((T, D, Z, Ft), dtype=np.float32) if args.save_traces else None
    counts_rows = [None] * (D * K) if args.save_counts else None
    ess_rows, warm, acc, piles, kinfo = [], [], [], [], []
    checks = {"totals_max_abs": 0, "count_min": None, "stored_vs_traced_max_abs": 0.0,
              "stored_sweeps": sorted(store), "drift": 0.0}
    t_setup = t_sweeps = t_fun = 0.0
    for bi, lo in enumerate(range(0, D, args.draw_batch)):
        idx = list(range(lo, min(lo + args.draw_batch, D)))
        b = len(idx)
        starts = []
        for d in idx:
            s_, w_ = warm_start(map_counts, unit, d, seed_base)
            starts.append(s_)
            warm.append(w_)
        inputs = batch_inputs(unit, idx)
        t0 = time.perf_counter()
        sm = NumbaSampler(inputs, unit.cov, mix, seed=0, start_counts=np.stack(starts),
                          threads=args.threads, kernel=kspec,
                          pile_moves=s5p.pile_counts(spec, inputs.N),
                          pile_first=args.pile_first, pile_second=args.pile_second)
        del starts
        # streams bound to (draw, zone), not to the slot's position in the batch
        states = np.concatenate([stream_states(draw_seed(seed_base, d), Z) for d in idx])
        if sm.rng.shape != states.shape or sm.rng.dtype != states.dtype:
            raise RuntimeError("the sampler's stream array is not (slots, 4) uint64")
        sm.rng = np.ascontiguousarray(states)
        kinfo.append(sm.kernel_info())
        t_setup += time.perf_counter() - t0
        Nb = inputs.N
        tr = np.empty((T, b, Z, Ft))
        for t in range(1, T + 1):
            s, drift = sm.run_sweeps(1)
            t_sweeps += float(s[0])
            secs[t - 1, bi] = s[0]
            checks["drift"] = max(checks["drift"], drift)
            t1 = time.perf_counter()
            lt[t - 1, idx] = sm.log_target()
            te[t - 1, idx] = sm.totals_error()
            tr[t - 1] = sm.functionals(Gt)                                # every sweep
            if t in store:
                j = store[t]
                c = sm.counts.reshape(b, Z, -1)
                tot = int(np.abs(c.sum(2) - Nb).max())
                cmin = int(c.min())
                checks["totals_max_abs"] = max(checks["totals_max_abs"], tot)
                checks["count_min"] = cmin if checks["count_min"] is None \
                    else min(checks["count_min"], cmin)
                fa = sm.functionals(G_all)                                # (b, Z, F)
                diff = float(np.abs(fa[:, :, cols] - tr[t - 1]).max()) if Ft else 0.0
                checks["stored_vs_traced_max_abs"] = max(checks["stored_vs_traced_max_abs"],
                                                         diff)
                for i, d in enumerate(idx):
                    samples[d * K + j] = fa[i]
                    if counts_rows is not None:
                        counts_rows[d * K + j] = sp.csr_matrix(c[i].astype(np.int32))
            t_fun += time.perf_counter() - t1
        acc.append(sm.acceptance())
        piles.append(sm.pile_stats() if sm.has_piles else None)
        if trace is not None:
            trace[:, idx] = tr
        # the draws' one-chain mixing after the re-burn
        for i, d in enumerate(idx):
            m1 = s5p.mixing(np.moveaxis(tr[args.reburn:, i:i + 1], 0, 1))
            t_ = s5p.class_table(unit.cls, m1, T - args.reburn, float("nan"))
            t_.insert(0, "draw", d)
            ess_rows.append(t_)
        del sm, tr
    rb = s5p.reburn(lt)
    ess = pd.concat(ess_rows, ignore_index=True)
    return {"samples": samples, "log_target": lt, "totals_error": te, "seconds": secs,
            "trace": trace, "counts": counts_rows, "checks": checks, "reburn": rb,
            "warm": warm, "acceptance": merge_acceptance(acc), "piles": merge_piles(piles),
            "kernel": kinfo, "ess": ess, "sweeps": T, "store": sorted(store),
            "seconds_setup": t_setup, "seconds_sweeps": t_sweeps, "seconds_trace": t_fun}


def merge_acceptance(parts: list[dict]) -> dict:
    out = {}
    for name in parts[0]:
        prop = sum(p[name]["proposed"] for p in parts)
        accd = sum(p[name]["accepted"] for p in parts)
        null = sum(p[name]["null"] for p in parts)
        out[name] = {"proposed": prop, "accepted": accd, "null": null,
                     "rate": accd / prop if prop else float("nan"),
                     "rate_non_null": accd / (prop - null) if prop > null else float("nan")}
    return out


def merge_piles(parts: list) -> dict | None:
    """Pile statistics summed over batches, the derived shares recomputed as
    :meth:`NumbaSampler.pile_stats` does."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    out = {k: sum(p[k] for p in parts) for k in PILE_STATS}
    out["mode"] = parts[0]["mode"]
    drawn, tried = out["drawn"], out["tried"]
    out["mean_abs_delta"] = out["sum_abs_delta"] / drawn if drawn else float("nan")
    out["mean_n"] = out["sum_n"] / drawn if drawn else float("nan")
    out["share_changed"] = out["changed"] / drawn if drawn else float("nan")
    prop = out["changed"] + out["rejected"]
    out["acceptance"] = (drawn - out["rejected"]) / drawn if drawn else float("nan")
    out["acceptance_of_changes"] = out["changed"] / prop if prop else float("nan")
    out["share_null"] = ((out["null_pair"] + out["n_zero"] + out["off_support"]) / tried
                         if tried else float("nan"))
    out["n_histogram"] = {b: sum(p["n_histogram"][b] for p in parts) for b in PILE_N_BINS}
    return out


# --------------------------------------------------------------------------
# Synthetic integration check: the states against stage 2 at the same draws
# --------------------------------------------------------------------------


def draws_agreement(unit: s5p.PilotUnit, G_all, cls_all, samples: np.ndarray, K: int
                    ) -> tuple[pd.DataFrame, dict]:
    """C3's states against stage 2's Gaussian at each of the same parameter
    draws (synthetic units; dense conditioning). Per cell (zone, functional),
    over cells whose mixture mean is at least 50 and not constant:

    * the pooled comparison of §5's rule: ``dm = (x̄ - μ)/σ`` and the ratio
      ``s²/σ²``, ``μ``, ``σ²`` the equal-weight mixture's mean and variance
      over the draws (``mean_d μ_d``; ``mean_d σ²_d + var_d μ_d``), ``x̄``,
      ``s²`` the 200 states';
    * z-scores against Monte Carlo error (derived here): the states are
      stratified by draw (``K`` per draw), so ``Var x̄ = Σ_d σ²_d/(K D²)``
      with the states of a chain taken as independent (they are ``--thin``
      sweeps apart), estimated with the chains' own sample variances,
      ``z_mean = (x̄ - μ)/se``; and the within-draw variance
      ``W = mean_d s²_d`` against ``mean_d σ²_d``, ``se(W) = √(Σ_d 2 s⁴_d/(K - 1))/D``
      (a Gaussian approximation), ``z_var``. If the sampler and the
      Gaussian agree, the z-scores are about standard normal: RMS about 1,
      about 5% beyond 2."""
    D = len(unit.draws)
    Sigma = unit.cov.dense()
    Z, F = samples.shape[1:]
    mu = np.empty((D, Z, F))
    s2 = np.empty((D, Z, F))
    for d, dr in enumerate(unit.draws):
        mu[d], s2[d] = s5p.dense_conditional_moments(unit.T0, unit.Y, dr["N"], dr["p"],
                                                     unit.kappa, Sigma, G_all)
    x = samples.reshape(D, K, Z, F)
    xv = x.var(1, ddof=1)
    mean_c3, var_c3 = x.mean((0, 1)), x.reshape(D * K, Z, F).var(0, ddof=1)
    mean_mix = mu.mean(0)
    var_mix = s2.mean(0) + mu.var(0)
    se_m = np.sqrt(xv.sum(0) / K) / D
    W, Wg = xv.mean(0), s2.mean(0)
    se_w = np.sqrt((2 * xv ** 2 / (K - 1)).sum(0)) / D
    with np.errstate(divide="ignore", invalid="ignore"):
        z_m = (mean_c3 - mean_mix) / se_m
        z_w = (W - Wg) / se_w
        dm = (mean_c3 - mean_mix) / np.sqrt(var_mix)
        ratio = var_c3 / var_mix
    sel = (mean_mix >= s5p.AGREE_MIN_MEAN) & (var_mix > 0) & (se_m > 0) & (se_w > 0)
    clsZ = np.broadcast_to(np.asarray(cls_all)[None, :], (Z, F))
    rows = []
    for c in list(pd.unique(np.asarray(cls_all))) + ["all"]:
        s_ = sel & ((clsZ == c) if c != "all" else True)
        if not s_.any():
            continue
        ok_m = np.abs(dm[s_]) <= s5p.AGREE_MEAN_SD
        ok_v = (ratio[s_] >= s5p.AGREE_VAR[0]) & (ratio[s_] <= s5p.AGREE_VAR[1])
        rows.append({"cls": c, "n": int(s_.sum()),
                     "rms_z_mean": float(np.sqrt(np.mean(z_m[s_] ** 2))),
                     "share_abs_z_mean_gt2": float(np.mean(np.abs(z_m[s_]) > 2)),
                     "max_abs_z_mean": float(np.max(np.abs(z_m[s_]))),
                     "rms_z_var": float(np.sqrt(np.mean(z_w[s_] ** 2))),
                     "share_abs_z_var_gt2": float(np.mean(np.abs(z_w[s_]) > 2)),
                     "max_abs_z_var": float(np.max(np.abs(z_w[s_]))),
                     "median_abs_dm_sd": float(np.median(np.abs(dm[s_]))),
                     "median_se_mean_sd": float(np.median(se_m[s_] / np.sqrt(var_mix[s_]))),
                     "var_ratio_median": float(np.median(ratio[s_])),
                     "var_ratio_min": float(np.min(ratio[s_])),
                     "var_ratio_max": float(np.max(ratio[s_])),
                     "share_mean_ok": float(ok_m.mean()), "share_var_ok": float(ok_v.mean()),
                     "share_both_ok": float((ok_m & ok_v).mean())})
    tab = pd.DataFrame(rows)
    allrow = tab[tab["cls"] == "all"].iloc[0].to_dict() if len(tab) else {}
    return tab, {"n": int(sel.sum()), **{k: v for k, v in allrow.items() if k != "cls"}}


# --------------------------------------------------------------------------
# One unit
# --------------------------------------------------------------------------


def run_unit(unit: s5p.PilotUnit, args, out: Path, tag: dict, seed_base: list,
             G_all: sp.csr_matrix, cols: np.ndarray) -> dict:
    """MAP chains, draw chains, the shared files (module notes); returns what
    the mode-specific writers need."""
    from pmedm_vb.progress import save_npz, write_text

    rng = np.random.default_rng(seed_base)
    t0 = time.perf_counter()
    mp = map_chains(unit, args, rng)
    ch, mx = mp["ch"], mp["mx"]
    Zi, Fi = np.meshgrid(np.arange(unit.Z), np.arange(unit.G.shape[1]), indexing="ij")
    draws_total = args.chains * args.map_sweeps
    frame = pd.DataFrame({"zone_index": Zi.ravel(), "cls": unit.cls[Fi.ravel()],
                          "functional": unit.names[Fi.ravel()],
                          "mean": mx["mean"].ravel(), "var": mx["var"].ravel(),
                          "rhat": mx["rhat"].ravel(), "ess_bulk": mx["ess_bulk"].ravel(),
                          "ess_tail": mx["ess_tail"].ravel(),
                          "ess_bulk_per_sweep": mx["ess_bulk"].ravel() / draws_total,
                          "ess_tail_per_sweep": mx["ess_tail"].ravel() / draws_total,
                          "constant": mx["constant"].ravel(),
                          "stage2_mean": unit.stage2_mean.ravel(),
                          "stage2_var": unit.stage2_var.ravel()})
    frame.assign(**tag).to_csv(out / "map_functionals.csv.gz", index=False)
    write_text(out / "mixing.csv", mp["table"].assign(**tag).to_csv(index=False))
    write_text(out / "agreement.csv", mp["agreement_table"].assign(**tag).to_csv(index=False))
    T = args.map_burn + args.map_sweeps
    write_text(out / "map_log_target.csv", pd.DataFrame(
        {"sweep": np.arange(1, T + 1), "seconds": ch["seconds"],
         "totals_error": ch["totals_error"],
         **{f"chain{k}_{nm}": ch["log_target"][:, k] for k, nm in enumerate(mp["starts"])}}
    ).to_csv(index=False))
    if args.save_traces:
        save_npz(out / "map_traces.npz", functionals=ch["functionals"],
                 log_target=ch["log_target"], cls=unit.cls, names=unit.names)
    seconds_map_chains = time.perf_counter() - t0

    t1 = time.perf_counter()
    dc = draw_chains(unit, args, ch["final_counts"], seed_base, G_all, cols)
    del ch["final_counts"]
    seconds_draw_chains = time.perf_counter() - t1
    D, K = len(unit.draws), args.draws_per_chain
    write_text(out / "draws_log_target.csv", pd.DataFrame(
        {"sweep": np.arange(1, dc["sweeps"] + 1),
         **{f"draw{d}": dc["log_target"][:, d] for d in range(D)}}).to_csv(index=False))
    if args.save_traces:
        save_npz(out / "draws_traces.npz", functionals=dc["trace"], log_target=dc["log_target"],
                 cls=unit.cls, names=unit.names)
    if args.save_counts:
        C = sp.vstack(dc["counts"]).tocsr()
        save_npz(out / "counts.npz", data=C.data, indices=C.indices, indptr=C.indptr,
                 shape=np.array(C.shape), keep=unit.keep, N=np.stack([d["N"] for d in unit.draws]),
                 draws_per_chain=K, store_sweeps=np.array(dc["store"]))
    ess = dc["ess"]
    ess_med = (ess[ess["cls"] != "all"].groupby("cls")[["ess_bulk_per_sweep_median",
                                                        "ess_tail_per_sweep_median"]]
               .median().round(4).to_dict(orient="index"))
    checks = dict(dc["checks"])
    checks["totals_error_trace_max"] = int(dc["totals_error"].max())
    checks["map_totals_error_max"] = int(np.max(ch["totals_error"]))
    checks["ok"] = bool(checks["totals_max_abs"] == 0 and checks["count_min"] >= 0
                        and checks["stored_vs_traced_max_abs"] == 0.0
                        and checks["totals_error_trace_max"] == 0
                        and checks["map_totals_error_max"] == 0)
    if not checks["ok"]:
        raise RuntimeError(f"{out}: a check of the stored states failed: {checks}")
    rb = dc["reburn"]
    summary = {
        "map_chains": {"chains": args.chains, "burn": args.map_burn, "sweeps": args.map_sweeps,
                       "starts": mp["starts"], "wall": mp["wall"],
                       "seconds_per_sweep_median": float(np.median(ch["seconds"])),
                       "moves_per_sweep": ch["moves_per_sweep"], "threads": ch["threads"],
                       "acceptance": ch["acceptance"], "piles": ch["piles"],
                       "kernel": ch["kernel"], "drift": ch["drift"],
                       "log_target": {"rhat": float(mp["lt_mix"]["rhat"][0]),
                                      "ess_bulk": float(mp["lt_mix"]["ess_bulk"][0])},
                       "burn_in": mp["burn_in"]},
        "mixing": mp["table"].to_dict(orient="records"),
        "agreement": mp["agreement_table"].to_dict(orient="records"),
        "agreement_pass": mp["agreement"]["pass"], "agreement_n": mp["agreement"]["n"],
        "draw_chains": {"draws": D, "draws_per_chain": K, "thin": args.thin,
                        "reburn": args.reburn, "sweeps": dc["sweeps"],
                        "store_sweeps": dc["store"], "batch": args.draw_batch,
                        "seconds_per_sweep_median_by_batch":
                            np.median(dc["seconds"], axis=0).tolist(),
                        "acceptance": dc["acceptance"], "piles": dc["piles"],
                        "kernel_tables_megabytes": [k["megabytes"] if k else 0.0
                                                    for k in dc["kernel"]],
                        "warm": dc["warm"], "ess_per_sweep_median_over_draws": ess_med},
        "reburn": {**rb, "share_crossing_after_reburn":
                   float(np.mean(np.asarray(rb["first_crossing"]) > args.reburn))},
        "checks": checks,
        "timings": {"map_chains": seconds_map_chains, "draw_chains": seconds_draw_chains,
                    "draw_chains_setup": dc["seconds_setup"],
                    "draw_chains_sweeps": dc["seconds_sweeps"],
                    "draw_chains_trace": dc["seconds_trace"]}}
    return {"samples": dc["samples"], "summary": summary}


def unit_tag(info: dict, region: str, k: int) -> dict:
    return {"simulation": ARM, "region": region, "replicate": k, "c": info.get("c"),
            "phase": "final", "tau_multiple": info.get("tau_multiple"), "tau": info.get("tau"),
            "kappa": info["kappa"], "phi": info["phi"]}


def log_unit(name: str, info: dict) -> None:
    from pmedm_vb.progress import logger

    agree = [r for r in info["agreement"] if r["cls"] == "all"]
    logger.info("C3 %s: agreement %s (both ok %.3f of %d), re-burn max %d, %.0fs, peak %.1f GB",
                name, "PASS" if info["agreement_pass"] else "FAIL",
                agree[0]["share_both_ok"] if agree else float("nan"), info["agreement_n"],
                info["reburn"]["max"], info["seconds"], info["peak_gb"])


def fit_real_unit(unit_key: tuple, args) -> str:
    import step4c_joint as s4c
    import step4d_free_shares as s4d
    from pmedm_vb.progress import save_npz, write_text

    region_id, c, m, k = unit_key
    out = args.out / ARM / region_id / f"r{k}"
    if (out / "done.json").exists():
        return f"{region_id} r{k}: done"
    out.mkdir(parents=True, exist_ok=True)
    s = s4c._SHARED
    t0 = time.perf_counter()
    parts: dict = {}
    unit = s5p.build_real_unit(region_id, args, tag=STEP5_TAG, replicate=k, parts=parts)
    seconds_setup = time.perf_counter() - t0
    info0 = unit.info
    if (info0["c"], info0["tau_multiple"]) != (c, m):
        raise RuntimeError(f"{region_id} r{k}: the unit's pair differs from S's chosen pair")
    rep, fit, starts, res = parts["rep"], parts["fit"], parts["starts"], parts["res"]
    tag = unit_tag(info0, region_id, k)
    N_int = parts["N_int"]
    q = (N_int / N_int.sum())[:, None] * parts["p"].T
    b1 = s4d.write_scores(rep, res.r, q, tag, out)
    values = [f.objective for f in starts]
    write_text(out / "starts.csv", pd.DataFrame(
        [{**tag, **f.info(), "selected": f is fit} for f in starts]).to_csv(index=False))
    write_text(out / "fits.csv", pd.DataFrame(
        [{**tag, "model": "cut_fixed_shares_rounded", **res.info()}]).to_csv(index=False))

    fun = s["fun"]                                         # with hhsize (prepare_version2)
    G_all = fun.G.tocsr()[unit.keep]
    seed_base = [args.seed, s4c.SIM_INDEX["S"], rep.index, k, STEP5_TAG]
    t1 = time.perf_counter()
    run = run_unit(unit, args, out, tag, seed_base, G_all, parts["cols"])
    t2 = time.perf_counter()
    b1_logdens = s4d.write_posterior(rep, run["samples"], tag, out)
    seconds_write = time.perf_counter() - t2

    mean, tcov, x, Sx = parts["totals_mean"], parts["totals_cov"], parts["x"], parts["Sx"]
    N_true = rep.true_pop.N.astype(float)
    sd = np.sqrt(np.clip(np.diag(tcov), 0.0, None))
    Nd = np.stack([d["N"] for d in unit.draws]).astype(float)
    save_npz(out / "map.npz", U=fit.U, shares=parts["shares"], totals_mean=mean,
             totals_cov=tcov, totals_int=N_int, x_direct=x, x_var=np.diag(Sx),
             shares_survey=rep.shares, shares_realized=res.shares, N_hat=rep.N_hat,
             N_true=N_true, Y=rep.draw.Y, var=rep.draw.var, N_draws=Nd)

    def rel_rms(est):
        return float(np.sqrt(np.mean(((est - N_true) / N_true) ** 2)))

    sec = parts["seconds"]
    info = {**tag, "map_start": fit.start, "map_objective": fit.objective,
            "map_grad_max": fit.grad_max, "map_grad_max_U": fit.grad_max_U,
            "map_grad_max_psi": 0.0, "map_converged": fit.converged,
            "spread": max(values) - min(values),
            "evaluations_all_starts": sum(f.evaluations for f in starts),
            "b1_rms_z": b1, "b1_logdens": b1_logdens, "seconds_map": sec["map"],
            "realized_outer_iterations": 1, "realized_converged": bool(res.converged),
            "realized_share_change_units": float(np.abs(res.shares - N_int / N_int.sum()).max()
                                                 * rep.N_A),
            "conditional_converged": bool(res.converged), "conditional_gap": res.gap,
            "totals_rel_rms": {"map": rel_rms(mean), "survey": rel_rms(rep.N_A * rep.shares),
                               "realized_map": rel_rms(rep.N_A * res.shares),
                               "direct": rel_rms(x), "rounded": rel_rms(N_int)},
            "totals_posterior_sd_rel_mean": float(np.mean(sd / N_true)),
            "totals_draws_sd_rel_mean": float(np.mean(Nd.std(0, ddof=1) / N_true)),
            "draws": int(run["samples"].shape[0]),
            "seconds_posterior": time.perf_counter() - t1,
            "seconds_per_draw": (time.perf_counter() - t1) / max(run["samples"].shape[0], 1),
            # stage 2's at the MAP parameters (the agreement's reference); C3 has none
            "negative_variances": int(info0["stage2_negative_variances"]),
            "unit": info0,
            **run["summary"], "settings": settings(args)}
    info["timings"] = {"setup": seconds_setup, "setup_map": sec["map"],
                       "setup_realized_map": sec["realized"], "setup_stage2": sec["stage2"],
                       "setup_param_draws": sec["draws"], "setup_rhat_refits": sec["draws_rhat"],
                       **info["timings"], "write_posterior": seconds_write}
    info["seconds"] = time.perf_counter() - t0
    info["peak_gb"] = peak_gb()
    write_text(out / "done.json", json.dumps(info, default=s5p._json_default))
    log_unit(f"{region_id} r{k}", info)
    return f"{region_id} r{k}"


def fit_synthetic_unit(name: str, Z: int, k: int, args) -> str:
    from pmedm_vb.latent.calibration import interval_stats
    from pmedm_vb.progress import save_npz, write_text

    out = args.out / ARM / name / f"r{k}"
    if (out / "done.json").exists():
        return f"{name} r{k}: done"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    parts: dict = {}
    unit = s5p.build_synthetic_unit(name, Z, args, args.seed + 1000 * k,
                                    kappa_c=args.synthetic_kappa_c, parts=parts)
    seconds_setup = time.perf_counter() - t0
    tag = unit_tag(unit.info, name, k)
    write_text(out / "fits.csv", pd.DataFrame(
        [{**tag, "model": "synthetic_conditional", **parts["res"].info()}]).to_csv(index=False))
    G_all = sp.csr_matrix(parts["G_all"])[unit.keep]
    cls_all, names_all = parts["cls_all"], parts["names_all"]
    seed_base = [args.seed, Z, k, STEP5_TAG]
    t1 = time.perf_counter()
    run = run_unit(unit, args, out, tag, seed_base, G_all, parts["cols"])
    samples = run["samples"]
    truth = np.asarray(sp.csr_matrix(parts["G_all"]).T @ np.asarray(parts["truth"], float).T).T
    st = interval_stats(samples, truth)
    frame = pd.DataFrame({"zone": np.repeat(np.arange(unit.Z), cls_all.size),
                          "cls": np.tile(cls_all, unit.Z),
                          "functional": np.tile(names_all, unit.Z), "truth": truth.ravel(),
                          **{key: np.asarray(v).ravel() for key, v in st.items()}})
    frame["width90_over_truth"] = np.where(frame["truth"] > 0,
                                           frame["width90"] / frame["truth"], np.nan)
    frame.assign(**tag).to_csv(out / "posterior.csv.gz", index=False)
    save_npz(out / "samples.npz", samples=samples, cls=cls_all, names=names_all)
    agree_tab, agree = draws_agreement(unit, G_all, cls_all, samples, args.draws_per_chain)
    write_text(out / "agreement_draws.csv", agree_tab.assign(**tag).to_csv(index=False))
    save_npz(out / "map.npz", totals_int=unit.N, N_draws=np.stack([d["N"] for d in unit.draws]),
             N_true=np.asarray(parts["truth"]).sum(1), keep=unit.keep)
    post = pd.DataFrame({"cls": np.tile(cls_all, unit.Z), "below_zero": st["below_zero"].ravel(),
                         "cover90": st["cover90"].ravel(), "cover50": st["cover50"].ravel()})
    info = {**tag, "draws": int(samples.shape[0]), "seconds_posterior": time.perf_counter() - t1,
            "unit": unit.info,
            **run["summary"], "draws_agreement": agree,
            "coverage": post.groupby("cls")[["cover50", "cover90", "below_zero"]].mean()
            .round(4).to_dict(orient="index"),
            "settings": settings(args)}
    info["timings"] = {"setup": seconds_setup, **info["timings"]}
    info["seconds"] = time.perf_counter() - t0
    info["peak_gb"] = peak_gb()
    write_text(out / "done.json", json.dumps(info, default=s5p._json_default))
    log_unit(f"{name} r{k}", info)
    return f"{name} r{k}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)
    import numba

    from pmedm_vb.progress import logger, record_run, write_text

    numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
    args.threads = numba.get_num_threads()
    logger.info("numba %s, %d threads (NUMBA_NUM_THREADS %d)", numba.__version__,
                numba.get_num_threads(), numba.config.NUMBA_NUM_THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        if args.seed is None:
            args.seed = 0
        record_run(args.out, args)
        write_text(args.out / "settings.json", json.dumps(
            {**settings(args), "seed": args.seed, "draws": args.draws,
             "replicates": args.replicates}, indent=1, default=str))
        names = dict(SYNTHETIC)
        chosen = args.region or list(names)
        for name in chosen:
            if name not in names:
                raise SystemExit(f"--synthetic: no unit {name!r} (have {list(names)})")
        for k in range(args.replicates):
            for name in chosen:
                logger.info("C3 %s", fit_synthetic_unit(name, names[name], k, args))
        return
    import step4c_joint as s4c
    import step4d_free_shares as s4d

    choice = json.loads((args.step4c / "choice.json").read_text())
    regions_all = [k.split("/", 1)[1] for k in choice if k.startswith("S/")]
    wanted = args.region or regions_all
    pending = [r for r in wanted if not all((args.out / ARM / r / f"r{k}" / "done.json").exists()
                                            for k in range(args.replicates))]
    for r in set(wanted) - set(pending):
        logger.info("C3 %s: all %d replicates done", r, args.replicates)
    if not pending:
        return
    args.region = pending
    regions, chosen, choice = s4d.setup(args, settings(args))
    s4c._SHARED["choice"] = choice
    s5p.prepare_version2(s4c._SHARED, args)
    units = s4d.units_at_s_pairs(regions, chosen, choice, args.replicates)
    logger.info("C3: %d units at S's chosen pairs (%s)", len(units), ", ".join(chosen))
    for u in units:
        logger.info("C3 unit %s", fit_real_unit(u, args))


if __name__ == "__main__":
    main()
