"""Draw a simulation's truths and surveys and write its synthetic inputs
(``experiments/sim/<name>.json``; ``sim_common.py``).

**Truths.** For each kind, ``truths[kind]`` populations of the source PUMA,
integer counts of each PUMS weighting unit in each block group:

- ``hmc_ref``: draws of the source experiment's reference HMC at the source
  alpha, spread evenly over its trace; each draw's ``W = N p`` made integer by
  one multinomial draw of ``round(N)`` units;
- ``ipf_multinomial``: multinomial draws of ``round(sum W)`` units from the
  source's IPF fit, as ``compare_methods.py`` draws IPF multinomial.

**Surveys.** Each truth's units are sampled independently with probability
``sampling_fraction`` (Bernoulli sampling, within each block group), and every
published line is estimated as the sampled units' loadings over the fraction.

**Sigma.** The source's published covariance (``v`` and the replicate
deviations), restricted to the simulation's constraint cells and multiplied by
one scalar ``c`` per truth kind: the median, over the constraint cells with a
sampled published estimate in the source and the kind's truths, of the
survey's exact variance under Bernoulli sampling, ``(1 - f) / f * sum_r
x_r^2 T_r``, over the published variance. So ``c Sigma`` matches the survey's
variances on the median cell while keeping the published correlations.

**Constraints.** The tables of ``constraints`` at each level, every published
line, less the deepest lines of each table in ``drop_deepest``; the tables'
trees follow (the pruned roll-up reads them). The held-out tables are written
as the survey's estimates with ``c`` times their published variances.

Writes the synthetic inputs (``<data>/processed/inputs/<fits area>/<sim puma>``),
``<root>/<fits name>/simulation/truths/<sim puma>.npz`` (the truth ``T`` and
the sample ``S``, sparse counts) and ``calibration.json``::

    sbatch --account=acf-utk0011 experiments/run_python.sbatch sim_build.py \\
        --sim experiments/sim/sim01.json --root $R

Then fit with ``run_experiment.py experiments/sim/sim01_fits.json --root $R``
and tabulate with ``sim_collect.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402
from sim_common import (Simulation, Tabulator, constraint_columns, pruned_tree,  # noqa: E402
                        stacked, stacked_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/<name>.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--overwrite", action="store_true",
                        help="redraw truths and surveys that exist (default: refuse)")
    return parser.parse_args()


def hmc_truths(sim: Simulation, inputs, rng, count: int) -> list[tuple[int, np.ndarray]]:
    """``count`` draws of the source's reference HMC, evenly spaced over its
    trace, each as ``(flat draw index, integer T)``."""
    from compare_methods import log_p_of
    from pmedm_vb.solvers.base import ConstraintOperator

    path = (sim.source_folder / sim.puma / "hmc_ref"
            / f"{rx.fit_name(sim.source, sim.puma, sim.alpha)}_trace.npz")
    with np.load(path) as saved:
        kept = saved["lam"]
    flat = kept.reshape(-1, kept.shape[-1])
    picks = np.unique(np.linspace(0, len(flat) - 1, count).astype(int))
    if len(picks) < count:
        raise SystemExit(f"{path}: {len(flat)} draws, fewer than the {count} truths asked for")
    op = ConstraintOperator(inputs)
    total = int(round(inputs.N))
    out = []
    for d in picks:
        p = np.exp(log_p_of(inputs, op, flat[d]))
        out.append((int(d), rng.multinomial(total, (p / p.sum()).ravel()).reshape(p.shape)))
    return out


def ipf_truths(sim: Simulation, rng, count: int) -> list[tuple[int, np.ndarray]]:
    """``count`` multinomial draws from the source's IPF fit, as ``(i, T)``."""
    from pmedm_vb.rake import rake_name

    rollup = sim.source["model"].get("rollup")
    path = sim.source_folder / sim.puma / "ipf" / f"{rake_name(sim.puma, 'ipf', rollup)}.npz"
    with np.load(path) as saved:
        W = np.asarray(saved["W"], dtype=float)
    total = int(round(W.sum()))
    p = np.clip(W.ravel() / W.sum(), 0.0, None)
    p /= p.sum()
    return [(i, rng.multinomial(total, p).reshape(W.shape)) for i in range(count)]


def sim_inputs(inputs, heldout, sim_puma: str, columns: dict, rows: np.ndarray, c: float,
               survey: dict, T: np.ndarray, drop: set[str]):
    """The synthetic problem: the survey's estimates of the constraint cells,
    ``c`` times their published covariance, and the trees without ``drop``."""
    from pmedm_vb.assemble.heldout import HeldOut

    kT, kB = columns["tract"], columns["block group"]
    trees = {}
    for table, entry in inputs.trees.items():
        if not any(inputs.tract_constraints[k].split(".")[0] == table for k in kT):
            continue
        bare = {name.split(".", 1)[1] for name in drop if name.split(".")[0] == table}
        trees[table] = pruned_tree(entry, bare) if bare else entry
    n_T = len(inputs.tract_constraints)
    out = dataclasses.replace(
        inputs,
        X_T=sp.csr_matrix(inputs.X_T)[:, kT], X_B=sp.csr_matrix(inputs.X_B)[:, kB],
        Y_T=survey["tract"][:, kT], Y_B=survey["block group"][:, kB],
        sigma_v=c * inputs.sigma_v[rows],
        sigma_l=None if inputs.sigma_l is None else np.sqrt(c) * inputs.sigma_l[rows],
        puma=sim_puma, N=float(T.sum()),
        tract_constraints=[inputs.tract_constraints[k] for k in kT],
        bg_constraints=[inputs.bg_constraints[k] for k in kB],
        trees=trees,
    )
    out.validate()
    out.zero_cell_variances()  # raises if a level has no zero cell to read the floor off
    held = HeldOut(
        X=heldout.X,
        Y={level: survey[level][:, n_T if level == "tract" else len(inputs.bg_constraints):]
           for level in ("block group", "tract")},
        v={level: c * heldout.v[level] for level in ("block group", "tract")},
        names=heldout.names)
    return out, held


def main() -> None:
    args = parse_args()
    sim = Simulation(args.sim, args.root)
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.data.variance import SDR_FACTOR

    source = sim.source_inputs()
    inputs = PMEDMInputs.load(source)
    heldout = HeldOut.load(source)
    tab = Tabulator(inputs, heldout)
    columns = constraint_columns(sim, inputs)
    rows = stacked_rows(inputs, columns)
    published = inputs.targets()[rows]
    replicate = (np.zeros(rows.size) if inputs.sigma_l is None
                 else SDR_FACTOR * np.square(inputs.sigma_l[rows]).sum(axis=1))
    sampled = (published > 0) & (replicate > 0) & (inputs.sigma_v[rows] > 0)
    f = sim.fraction
    print(f"{source}: {len(columns['block group'])} block group and {len(columns['tract'])} "
          f"tract constraint columns of {len(inputs.bg_constraints)} and "
          f"{len(inputs.tract_constraints)}; dropped lines: {sorted(columns['dropped'])}",
          flush=True)

    sim.folder.mkdir(parents=True, exist_ok=True)
    (sim.folder / "truths").mkdir(exist_ok=True)
    rng = np.random.default_rng(sim.settings["seed"])
    calibration = {"sampling_fraction": f, "kinds": {}}
    for kind in sim.kinds():
        count = sim.settings["truths"][kind]
        names = [sim.sim_puma(kind, i) for i in range(1, count + 1)]
        existing = [n for n in names if sim.truth_file(n).exists()]
        if existing and not args.overwrite:
            raise SystemExit(f"{sim.truth_file(existing[0])} exists; --overwrite to redraw")
        truths = (hmc_truths(sim, inputs, rng, count) if kind == "hmc_ref"
                  else ipf_truths(sim, rng, count))
        samples, ratios = [], []
        for _, T in truths:
            S = rng.binomial(T, f)
            samples.append(S)
            variance = stacked(tab(T, squared=True), columns) * (1 - f) / f
            ok = sampled & (variance > 0)
            ratios.append(variance[ok] / inputs.sigma_v[rows][ok])
        ratios = np.concatenate(ratios)
        c = float(np.median(ratios))
        calibration["kinds"][kind] = {
            "c": c, "cells": int(ratios.size),
            "ratio_quantiles": {f"q{q:g}": float(np.quantile(ratios, q))
                                for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}
        print(f"{kind}: c = {c:.4g} over {ratios.size:,} cell-truths (ratio p10 "
              f"{np.quantile(ratios, 0.1):.3g}, p90 {np.quantile(ratios, 0.9):.3g})", flush=True)
        for name, (source_draw, T), S in zip(names, truths, samples):
            survey = tab(S / f)
            out, held = sim_inputs(inputs, heldout, name, columns, rows, c, survey, T,
                                   columns["dropped"])
            out.save(sim.sim_inputs(name))
            held.save(sim.sim_inputs(name))
            np.savez_compressed(sim.truth_file(name), T=T.astype(np.int32), S=S.astype(np.int32),
                                c=c, kind=kind, source_draw=source_draw, fraction=f)
            print(f"  {name}: N {out.N:,.0f}, sampled {S.sum():,} of {T.sum():,} units",
                  flush=True)
    calibration["dropped_lines"] = sorted(columns["dropped"])
    (sim.folder / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    print(f"wrote {sim.folder / 'calibration.json'}")


if __name__ == "__main__":
    main()
