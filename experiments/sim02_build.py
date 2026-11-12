"""Draw the borrowing simulation's truths, surveys and PUMS samples and write its
synthetic inputs (``experiments/sim/sim02.json``; ``sim02_common.py``).

**Truths.** For each kind in ``truths``, ``count`` populations of the PUMA,
integer counts ``T`` (zones x units) of each unit of the kind's source inputs
in each block group, as sim01 draws them (``sim_build.hmc_truths``): draws of
the source experiment's HMC run (``run``: ``short`` or ``ref``) evenly spaced
over its kept draws (chains and iterations flattened), each draw's
``p(lambda)`` made a population by one multinomial draw of ``round(N)`` units.

- ``local``: the PUMA-only experiment (``tn01_gpu``); units are the PUMA's
  own PUMS records;
- ``borrowed``: the pool A experiment (``tn04_poola_e0.01``,
  ``sim02_pools.py``); units are the merged rows of the PUMA's records and
  pool A's, so the truth holds kinds of household the PUMA's sample lacks.

**Surveys**, as sim01: each truth's units sampled independently with
probability ``f = sampling_fraction`` within each block group, ``S ~
Binomial(T, f)``; every published line (constraint and held-out, both levels)
estimated as ``S``'s loadings over ``f``. Sigma is the source's published
covariance (``v`` and the replicate deviations, every cell of the five tables:
no line is dropped) times one scalar ``c`` per truth kind, calibrated as
sim01's (``sim_build.py``): the median, over the constraint cells with a
sampled published estimate and the kind's truths, of the survey's exact
Bernoulli variance ``(1 - f) / f sum_r x_r^2 T_r`` over the published variance.
The held-out tables are the survey's estimates with ``c`` times their
published variances. The fits' ``variance_floor: "zero"`` reads each area's
zero-count variance off the cells estimated as zero with no replicate spread
(``PMEDMInputs.zero_cell_variances``), so every truth's survey must leave one
such cell at each level; all truths are drawn and checked before anything is
written, and the build stops, writing nothing, if one does not.

**Simulated PUMS.** A subsample of the survey, ``P ~ Binomial(S, r)``, with
``r = target / (f round(N))``, so that the expected sample size is ``target``
(``pums_size``; ``null`` is the PUMA's real sample size, its inputs' ``n``),
each sampled unit carrying the design weight ``1 / (f r)``. The PUMS does not
say where a unit lives, so only ``P`` summed over zones reaches the fits.

**Two fits per truth**, the same survey constraints, Sigma and held-out tables:

- **local** (``<fits local area>/<sim puma>``): the units are the distinct
  source units with ``P > 0`` (unmerged, as the real PUMA-only problem is),
  each weighted ``count / (f r)`` and spread evenly over the zones
  (``design_weights``). ``n`` is the number of PUMS units, ``N`` the sum of
  their weights -- as the real pipeline sets them, the sample size and the
  sum of the own design weights -- so ``N`` is an estimate of the truth's
  total, not the total itself;
- **statewide** (``<fits statewide area>-state-e<E>/<sim puma>``): the local
  PUMS plus pool B's out-of-PUMA records with their real design weights,
  weighted by ``support_weights`` at ``epsilon`` (the PUMS units as the PUMA's
  own records) and merged where rows of ``X`` agree (``collapse_units``), as
  ``build_puma(..., epsilon=E)`` builds a statewide support. ``unit_members``
  has one row per PUMS unit (``SERIALNO`` its source unit's, ``#k``) and per
  pool B record; ``n`` and ``N`` are the local fit's. A merged row's held-out
  counts are its members' averaged by prior weight (``assemble/heldout.py``):
  a PUMS unit's are its source unit's, a pool B record's its pool B row's.
  Pool B is the same for both kinds and never holds a borrowed truth's records.

**Held-out counts of the source units.** The borrowed kind's are pool A's
(``sim02_pools.py``). The local kind's are the PUMA-only inputs' own held-out
tables (``run_map.py heldout`` on the PUMA-only area) when they exist with the
same tables as pool B's; otherwise each record takes the held-out counts of its
merged row in the real statewide inputs (the prior-weight average over the
row's records), which is recorded in calibration.json as ``heldout_source``.

**Several PUMAs, other fits, no pools** (``sim04.json``; ``sim02_common.py``):
everything above is done PUMA by PUMA, with that PUMA's source inputs, trace,
held-out tables, real sample size and out-of-PUMA records; ``c`` is
calibrated per kind *and PUMA*, over the kind's truths of that PUMA, since
each PUMA's published variances are its own (with one PUMA that is sim02's
per-kind ``c``). Without a ``pools`` block the statewide fits' out-of-PUMA
records are the real statewide inputs' (``statewide``), every one with its
real design weight, in place of pool B's; a source on those same inputs (a
statewide experiment) then shares out-of-PUMA records with the fits, and each
truth's share of ``N`` on out-of-PUMA records is recorded
(``truth_borrowed``). The local inputs are written only if a fit JSON fits
the local problem, the statewide ones only if one fits the statewide problem,
once per inputs folder. With several PUMAs each truth's random draws come
from ``default_rng([seed, k, <puma>, i])`` (one PUMA: ``[seed, k, i]``, as
before), and calibration.json is ``{"sampling_fraction", "epsilon", "pumas":
{<puma>: {"pool": ..., "kinds": {<kind>: ...}}}}``, each kind's entry as
sim02's plus ``truth_borrowed`` per truth.
A source with Poisson rows (``+poisson<c>``, tn10) is drawn exactly as a
Gaussian one: its trace's ``lam`` holds the multipliers the data see, in
today's stacked layout (``run_mcmc.draw_arrays``; the sampled coordinates,
``u`` on the Poisson rows, are ``xi``), so ``p(lambda)`` is the same function
of it [assumed: the allocation under Poisson rows is ``q exp(-X lam)`` with
those multipliers, as :mod:`pmedm_vb.assemble.poisson` defines it].
The same command builds sim04 with ``--sim experiments/sim/sim04.json``.

Writes the two fits' inputs (with ``heldout/``),
``<root>/<name>/simulation/truths/<sim puma>.npz`` (``T``, ``S``, ``P``, the
local fit's units as source unit indices, ``local_units``, and each one's row in
the statewide fit, ``state_rows``; ``c``, ``f``, ``r``) and, last,
``calibration.json``::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log experiments/run_python.sbatch \\
        sim02_build.py --sim experiments/sim/sim02.json --root $R
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim02_common import Sim02, record_design_weights  # noqa: E402
from sim_common import Tabulator, stacked, stacked_rows  # noqa: E402

LEVELS = ("block group", "tract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/sim02.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--overwrite", action="store_true",
                        help="redraw truths that exist (default: refuse)")
    return parser.parse_args()


# -- the truths' sources ----------------------------------------------------------


def truth_source(sim: Sim02, kind: str, names: dict, puma: str | None = None):
    """The kind's source inputs for ``puma`` (default the simulation's one PUMA),
    their held-out tables, and where those came from. ``names`` are the fits'
    out-of-PUMA records' held-out names (pool B's), which the source's must match."""
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    path = sim.truth_inputs(kind, puma)
    inputs = PMEDMInputs.load(path)
    if HeldOut.exists(path):
        held = HeldOut.load(path)
        if held.names == names:
            return inputs, held, str(path / "heldout")
        if inputs.unit_members is not None:
            raise SystemExit(f"{path}/heldout: its tables are not pool B's")
        print(f"{path}/heldout: its tables are not pool B's; using the statewide rows'",
              flush=True)
    elif inputs.unit_members is not None:
        raise SystemExit(f"{path}/heldout: missing (sim02_pools.py writes it)")
    # A PUMA-only problem: each record takes its merged row's held-out counts in
    # the real statewide inputs.
    state_path = sim.statewide_inputs(puma)
    state, state_held = PMEDMInputs.load(state_path), HeldOut.load(state_path)
    members = state.unit_members
    own = members[members["in_puma"].to_numpy(dtype=bool) & (members["row"] >= 0)]
    row_of = pd.Series(own["row"].to_numpy(), index=own["SERIALNO"].to_numpy())
    rows = row_of.reindex(inputs.units["SERIALNO"].to_numpy())
    weight = inputs.units["weight"].to_numpy(dtype=float)
    lost = rows.isna().to_numpy() & (weight > 0)
    if lost.any():
        raise SystemExit(f"{path}: {int(lost.sum())} records with weight are not among "
                         f"{state_path}'s own records")
    found = np.flatnonzero(rows.notna().to_numpy())
    pick = sp.csr_matrix((np.ones(found.size), (found, rows.to_numpy()[found].astype(int))),
                         shape=(inputs.n_units, state.n_units))
    held = HeldOut(X={level: sp.csr_matrix(pick @ X) for level, X in state_held.X.items()},
                   Y=state_held.Y, v=state_held.v, names=state_held.names)
    return inputs, held, f"{state_path / 'heldout'} (each record its statewide merged row's)"


def posterior_truths(inputs, trace: Path, count: int, rngs: list) -> list[tuple[int, np.ndarray]]:
    """``count`` draws of the trace's ``lam`` evenly spaced over its kept draws,
    each as ``(flat draw index, integer T)``: one multinomial draw of
    ``round(N)`` units from ``p(lambda)``, with ``rngs[i]`` for the ``i``-th."""
    from pmedm_vb.solvers.base import ConstraintOperator

    with np.load(trace) as saved:
        kept = saved["lam"]
    flat = kept.reshape(-1, kept.shape[-1])
    if flat.shape[1] != inputs.n_constraints:
        raise SystemExit(f"{trace}: {flat.shape[1]} multipliers, the inputs have "
                         f"{inputs.n_constraints} constraint rows")
    picks = np.unique(np.linspace(0, len(flat) - 1, count).astype(int))
    if len(picks) < count:
        raise SystemExit(f"{trace}: {len(flat)} draws, fewer than the {count} truths asked for")
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    total = int(round(inputs.N))
    out = []
    for d, rng in zip(picks, rngs):
        logits = log_q - op.adjoint(flat[d])
        p = np.exp(logits - logsumexp(logits)).ravel()
        out.append((int(d), rng.multinomial(total, p / p.sum()).reshape(log_q.shape)))
    return out


# -- the synthetic problems -----------------------------------------------------


def scaled_sigma(inputs, c: float, name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """``c`` times the source's covariance, as sim_build.sim_inputs scales it."""
    from pmedm_vb.data.variance import SDR_FACTOR

    v = c * inputs.sigma_v
    dev = None if inputs.sigma_l is None else np.sqrt(c) * inputs.sigma_l
    if dev is not None:
        s = SDR_FACTOR * np.square(dev).sum(axis=1)
        if np.any(s - v > 1e-9 * np.maximum(s, 1.0)):
            raise SystemExit(f"{name}: scaled variances below their replicate variance "
                             "by more than rounding")
        v = np.maximum(v, s)
    return v, dev


def local_fit(source, held, name: str, survey: dict, sigma: tuple, c: float,
              counts: np.ndarray, weight_each: float):
    """The local fit's problem and held-out tables: the source units the PUMS
    sampled, ``counts`` (per source unit) times ``weight_each`` each."""
    from pmedm_vb.assemble.design import design_weights
    from pmedm_vb.assemble.heldout import HeldOut

    types = np.flatnonzero(counts > 0)
    weight = counts[types] * weight_each
    units = pd.DataFrame({
        "SERIALNO": source.units["SERIALNO"].to_numpy()[types],
        "weight": weight,
        "is_group_quarters": source.units["is_group_quarters"].to_numpy(dtype=bool)[types],
        "source_unit": types,
        "pums_units": counts[types].astype(int),
    })
    nT, nB = len(source.tract_constraints), len(source.bg_constraints)
    out = dataclasses.replace(
        source, q=design_weights(units["weight"], n_zones=source.n_zones),
        X_T=sp.csr_matrix(source.X_T)[types], X_B=sp.csr_matrix(source.X_B)[types],
        Y_T=survey["tract"][:, :nT], Y_B=survey["block group"][:, :nB],
        sigma_v=sigma[0], sigma_l=sigma[1], puma=name, n=int(counts.sum()),
        N=float(weight.sum()), units=units, unit_members=None, support=None,
        support_columns=None)
    out.validate()
    out.zero_cell_variances()  # raises if a level has no zero cell to read the floor off
    split = {"tract": nT, "block group": nB}
    heldout = HeldOut(X={level: sp.csr_matrix(X)[types] for level, X in held.X.items()},
                      Y={level: survey[level][:, split[level]:] for level in LEVELS},
                      v={level: c * held.v[level] for level in LEVELS}, names=held.names)
    return out, heldout, types


class PoolB:
    """Pool B's out-of-PUMA records: their rows, design weights and held-out counts.

    Any statewide inputs serve (every out-of-PUMA record with a row and a
    positive design weight is taken); ``key`` is the name under which the
    statewide fits' ``support`` records the inputs' path (``pool_b`` for
    sim02's pool B, ``pool`` for real statewide inputs)."""

    def __init__(self, path: Path, key: str = "pool_b") -> None:
        from pmedm_vb.assemble.heldout import HeldOut
        from pmedm_vb.assemble.inputs import PMEDMInputs

        self.path, self.key = path, key
        self.inputs = PMEDMInputs.load(path)
        self.held = HeldOut.load(path)
        members = self.inputs.unit_members
        d = record_design_weights(self.inputs)
        keep = (~members["in_puma"].to_numpy(dtype=bool)) & (members["row"].to_numpy() >= 0) \
            & (d > 0)
        self.serials = members["SERIALNO"].to_numpy()[keep]
        self.d = d[keep]
        record_rows = members["row"].to_numpy()[keep]
        self.rows, self.item = np.unique(record_rows, return_inverse=True)
        self.item = self.item.reshape(-1)


def statewide_fit(local, local_held, types: np.ndarray, counts: np.ndarray,
                  weight_each: float, pool: PoolB, epsilon: float):
    """The statewide fit's problem and held-out tables: the local fit's PUMS units
    and pool B's records, support-weighted and merged; and each local unit's row."""
    from pmedm_vb.assemble.build import support_report, unit_frame
    from pmedm_vb.assemble.design import collapse_units, design_weights, support_weights
    from pmedm_vb.assemble.heldout import HeldOut

    pX_T, pX_B = sp.csr_matrix(pool.inputs.X_T), sp.csr_matrix(pool.inputs.X_B)
    k_local = types.size
    X_T = sp.vstack([sp.csr_matrix(local.X_T), pX_T[pool.rows]]).tocsr()
    X_B = sp.vstack([sp.csr_matrix(local.X_B), pX_B[pool.rows]]).tocsr()
    gq = np.concatenate([local.units["is_group_quarters"].to_numpy(dtype=bool),
                         pool.inputs.units["is_group_quarters"].to_numpy(dtype=bool)[pool.rows]])
    H = {level: sp.vstack([sp.csr_matrix(local_held.X[level]),
                           sp.csr_matrix(pool.held.X[level])[pool.rows]]).tocsr()
         for level in LEVELS}

    # One member per PUMS unit, then pool B's records.
    per_type = counts[types].astype(int)
    item_local = np.repeat(np.arange(k_local), per_type)
    serial = pd.Series(np.repeat(local.units["SERIALNO"].astype(str).to_numpy(), per_type))
    serial = serial + "#" + serial.groupby(item_local).cumcount().astype(str)
    item = np.concatenate([item_local, k_local + pool.item])
    d = np.concatenate([np.full(item_local.size, weight_each), pool.d])
    in_puma = np.concatenate([np.ones(item_local.size, bool), np.zeros(pool.d.size, bool)])
    weight = support_weights(pd.Series(d), in_puma, epsilon)
    item_weight = np.bincount(item, weights=weight, minlength=X_T.shape[0])
    X_T, X_B, merged, item_row = collapse_units(X_T, X_B, gq, item_weight)
    members = pd.DataFrame({
        "SERIALNO": np.concatenate([serial.to_numpy(), pool.serials.astype(str)]),
        "row": item_row[item], "weight": weight, "in_puma": in_puma, "design_weight": d})
    units = unit_frame(members, gq[item], merged.size)
    support, columns = support_report(members, d, X_T, X_B, list(local.tract_constraints),
                                      list(local.bg_constraints), epsilon)
    support[pool.key] = str(pool.path)
    out = dataclasses.replace(
        local, q=design_weights(units["weight"], n_zones=local.n_zones), X_T=X_T, X_B=X_B,
        units=units, unit_members=members, support=support, support_columns=columns,
        n=int(in_puma.sum()), N=float(d[in_puma].sum()))
    out.validate()
    # A merged row's held-out counts: its members' (each item's) by prior weight.
    ok = np.flatnonzero(item_row >= 0)
    share = item_weight[ok] / merged[item_row[ok]]
    average = sp.csr_matrix((share, (item_row[ok], ok)), shape=(merged.size, item_row.size))
    heldout = HeldOut(X={level: sp.csr_matrix(average @ H[level]) for level in LEVELS},
                      Y=local_held.Y, v=local_held.v, names=local_held.names)
    return out, heldout, item_row[:k_local]


# -- the build ------------------------------------------------------------------


def draw_puma(sim: Sim02, puma: str, pool: PoolB, record: dict, kinds: dict,
              no_zero: list[str]) -> None:
    """Draw and check every kind's truths, surveys and PUMS samples of ``puma``
    (nothing written): each kind's ``(source, held, tab, c, draws, r,
    weight_each)`` into ``kinds[(kind, puma)]``, its calibration into
    ``record[kind]``, and the truths without a zero cell at a level into
    ``no_zero``."""
    from compare_support import borrowed_fraction
    from pmedm_vb.data.variance import SDR_FACTOR

    f = sim.fraction
    single = sim.puma is not None
    for k, kind in enumerate(sim.kinds()):
        source, held, held_note = truth_source(sim, kind, pool.held.names, puma)
        for label, a, b in (("tract", source.tract_constraints, pool.inputs.tract_constraints),
                            ("block group", source.bg_constraints, pool.inputs.bg_constraints)):
            if list(a) != list(b):
                raise SystemExit(f"{kind}: the source's {label} constraints are not pool B's")
        if source.n_zones != pool.inputs.n_zones:
            raise SystemExit(f"{kind}: the source's block groups are not pool B's")
        tab = Tabulator(source, held)
        columns = {"tract": list(range(len(source.tract_constraints))),
                   "block group": list(range(len(source.bg_constraints)))}
        rows = stacked_rows(source, columns)
        replicate = (np.zeros(rows.size) if source.sigma_l is None
                     else SDR_FACTOR * np.square(source.sigma_l[rows]).sum(axis=1))
        sampled = (source.targets()[rows] > 0) & (replicate > 0) & (source.sigma_v[rows] > 0)
        split = source.Y_T.size

        count = sim.count(kind)
        seed = int(sim.settings["seed"])
        rngs = [np.random.default_rng([seed, k, i] if single else [seed, k, int(puma), i])
                for i in range(1, count + 1)]
        trace = sim.truth_trace(kind, puma)
        truths = posterior_truths(source, trace, count, rngs)
        total = int(round(source.N))
        target = sim.settings.get("pums_size") or source.n
        r = float(target) / (f * total)
        if not 0 < r <= 1:
            raise SystemExit(f"{kind}: PUMS rate {r:g} for {target} units of {total} at f = {f:g}")
        weight_each = 1.0 / (f * r)
        draws, ratios = [], []
        for i, ((source_draw, T), rng) in enumerate(zip(truths, rngs), start=1):
            S = rng.binomial(T, f)
            P = rng.binomial(S, r)
            draws.append((source_draw, T.astype(np.int32), S.astype(np.int32),
                          P.astype(np.int32)))
            variance = stacked(tab(T, squared=True), columns) * (1 - f) / f
            ok = sampled & (variance > 0)
            ratios.append(variance[ok] / source.sigma_v[rows][ok])
            # The fits' variance floor ("zero") is read off cells estimated as
            # zero with no replicate spread; each level needs one.
            zero = (stacked(tab(S / f), columns) == 0) & (replicate == 0)
            for label, part in (("tract", zero[:split]), ("block group", zero[split:])):
                if not part.any():
                    no_zero.append(f"{sim.sim_puma(kind, i, puma)} ({label})")
        ratios = np.concatenate(ratios)
        c = float(np.median(ratios))
        record[kind] = {
            "c": c, "cells": int(ratios.size),
            "ratio_quantiles": {f"q{q:g}": float(np.quantile(ratios, q))
                                for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "source_inputs": str(sim.truth_inputs(kind, puma)), "source_trace": str(trace),
            "source_draws": [d for d, *_ in draws], "heldout_source": held_note,
            "pums_target": int(target), "pums_rate": r, "pums_weight": weight_each,
            "total": total}
        if not single:
            # The truth's share of N on records from outside the PUMA, which
            # without pools may be among the fits' own out-of-PUMA records.
            frac = borrowed_fraction(source)
            record[kind]["truth_borrowed"] = [float(T.sum(0) @ frac / T.sum())
                                              for _, T, _, _ in draws]
        print(f"{kind}{'' if single else f' {puma}'}: {source.n_units:,} source units; "
              f"c = {c:.4g} over {ratios.size:,} cell-truths; PUMS rate {r:.4g} (target "
              f"{target:,} of {total:,} at f = {f:g}), weight {weight_each:.2f}\n  held-out "
              f"counts: {held_note}", flush=True)
        kinds[(kind, puma)] = (source, held, tab, c, draws, r, weight_each)


def main() -> None:
    args = parse_args()
    sim = Sim02(args.sim, args.root)

    names = sim.sim_pumas()
    existing = [n for n in names if sim.truth_file(n).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{sim.truth_file(existing[0])} exists; --overwrite to redraw")
    single = sim.puma is not None
    f = sim.fraction
    calibration = {"sampling_fraction": f, "epsilon": sim.epsilon}
    if not single:
        calibration["pumas"] = {}
    # Everything is drawn and checked before anything is written, so a build
    # that fails leaves the inputs and truths of an earlier one as they were.
    pools, kinds, no_zero = {}, {}, []
    for puma in sim.pumas:
        pool = PoolB(sim.fit_pool_inputs(puma), "pool_b" if sim.pools is not None else "pool")
        pools[puma] = pool
        label = "pool B" if sim.pools is not None else f"{puma}: out-of-PUMA records"
        print(f"{label} ({pool.path}): {pool.d.size:,} records in {pool.rows.size:,} rows, "
              f"design weight {pool.d.sum():,.0f}", flush=True)
        pool_record = {"inputs": str(pool.path), "records": int(pool.d.size),
                       "rows": int(pool.rows.size)}
        if single:
            calibration["pool_b"], calibration["kinds"] = pool_record, {}
            record = calibration["kinds"]
        else:
            calibration["pumas"][puma] = {"pool": pool_record, "kinds": {}}
            record = calibration["pumas"][puma]["kinds"]
        draw_puma(sim, puma, pool, record, kinds, no_zero)
    if no_zero:
        raise SystemExit("no constraint cell estimated as zero with no replicate spread, so no "
                         "zero-count variance for the fits' variance floor, in "
                         + ", ".join(no_zero) + "; nothing written (change the seed)")

    sim.folder.mkdir(parents=True, exist_ok=True)
    (sim.folder / "truths").mkdir(exist_ok=True)
    # The inputs each fit needs, written once per inputs folder.
    supports = sim.input_fits()
    for (kind, puma), (source, held, tab, c, draws, r, weight_each) in kinds.items():
        sigma = scaled_sigma(source, c, kind)
        for i, (source_draw, T, S, P) in enumerate(draws, start=1):
            name = sim.sim_puma(kind, i, puma)
            survey = tab(S / f)
            counts = P.sum(axis=0)
            local, local_held, types = local_fit(source, held, name, survey, sigma, c, counts,
                                                 weight_each)
            state, state_held, state_rows = statewide_fit(local, local_held, types, counts,
                                                          weight_each, pools[puma], sim.epsilon)
            for support, problem, h in (("local", local, local_held),
                                        ("statewide", state, state_held)):
                for folder in sorted({sim.fit_inputs(fit, name) for fit in supports[support]}):
                    problem.save(folder)
                    h.save(folder)
            np.savez_compressed(
                sim.truth_file(name), T=T, S=S, P=P, local_units=types, state_rows=state_rows,
                c=c, kind=kind, source_draw=source_draw, fraction=f, pums_rate=r,
                pums_weight=weight_each)
            print(f"  {name}: T {T.sum():,} units, survey {S.sum():,}, PUMS {P.sum():,} in "
                  f"{types.size:,} distinct units (N {local.N:,.0f}); statewide "
                  f"{state.n_units:,} rows, {100 * state.support['prior_share_added']:.2f}% of "
                  f"the prior on rows without a PUMS unit", flush=True)
    (sim.folder / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"wrote {sim.folder / 'calibration.json'}")


if __name__ == "__main__":
    main()
