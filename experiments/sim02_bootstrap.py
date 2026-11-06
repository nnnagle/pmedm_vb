"""Write Bayesian-bootstrap replicates of the borrowing simulation's local-fit
inputs (``experiments/sim/sim02.json``, its ``bootstrap`` block; ``sim02_common.py``).

**Why.** The local fit's prior ``q`` is the simulated PUMS's design weights
spread over the block groups (``assemble/design.py: design_weights``), and the
posterior treats it as known: the uncertainty of which households the PUMS
happened to sample, and so of the joint distribution ``q`` carries (race x
income above all), is nowhere in it. Averaging the posterior over a posterior
for that sample puts it back.

**The Bayesian bootstrap** (Rubin 1981, *The Bayesian bootstrap*, Annals of
Statistics 9(1), 130-134): given ``n`` sampled households, the population's
distribution over them is drawn as Dirichlet(1, ..., 1) weights, one per
household -- the posterior of a multinomial over the observed values under a
non-informative prior. A local-fit unit is a source unit the PUMS sampled
``c_i`` times (``units.pums_units``, ``sim02_build.local_fit``), weighted
``c_i / (f r)``; its households' Dirichlet components sum to
``G_i / sum_k G_k`` with ``G_i ~ Gamma(shape=c_i, scale=1)`` independent.
With design weights the households are not exchangeable, so each household's
component multiplies its design weight (the *weighted* Bayesian bootstrap, an
approximation to the finite-population posterior): the unit's weight becomes
``(w_i / c_i) G_i``, rescaled so that the weights sum to the fit's ``N`` again.
The unit's multiplier ``m_i = G_i / c_i`` (before the rescaling) has mean 1
and variance ``1 / c_i``: a unit seen once may lose its weight or double it; a
unit seen often keeps about its own. In sim02 every PUMS household has the
same design weight ``1 / (f r)``, so the replicate's weights are exactly ``N``
times the households' Dirichlet shares, summed by unit (multiplier variance
``(n - c_i) / (c_i (n + 1))``).

Each replicate keeps the local fit's units, ``n``, ``N``, ``X``, survey
targets, Sigma and held-out tables; only the unit weights and so ``q``
(``design_weights``) change. Fitting the ``R`` replicates and pooling their
draws in equal numbers (``sim02_score.py``, fit ``boot``) samples, approximately,
the posterior averaged over the bootstrap's posterior for ``q``.

For each truth in ``bootstrap.truths`` and ``j = 1 .. bootstrap.replicates``
this writes the PUMA ``<sim puma>r<j>`` (e.g. ``4701502l01r3``) under the local
fits' area (``<data>/processed/inputs/tnsim-2024-5yr/``) with its held-out
tables and a ``bootstrap.json`` (truth, replicate, seed, the multipliers'
summary); ``units`` gains ``design_weight`` (the local fit's weight) and
``bootstrap_multiplier`` (the new weight over it). Replicate ``j`` of truth
``i`` of the ``k``-th kind is drawn with ``numpy.random.default_rng([seed, k,
i, j])``, so a replicate does not depend on which other truths are listed.
Everything is drawn and checked before anything is written; a record of every
replicate goes to ``<root>/<name>/simulation/bootstrap.json``. It reads only
the local fits' saved inputs (``sim02_build.py``'s), on a CPU::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log experiments/run_python.sbatch \\
        sim02_bootstrap.py --sim experiments/sim/sim02.json --root $R

then ``run_experiment.py experiments/sim/sim02_fits_boot.json --root $R``.
Existing replicates are refused unless ``--overwrite``, which also leaves any
fits of the old replicates (``<root>/<bootstrap fits name>/<puma>``) to be
deleted by hand.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim02_common import Sim02  # noqa: E402

METHOD = ("weighted Bayesian bootstrap (Rubin 1981) over the simulated PUMS households: "
          "unit weight (w_i / c_i) Gamma(c_i, 1), rescaled to sum to N")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/sim02.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--overwrite", action="store_true",
                        help="redraw replicates that exist (default: refuse)")
    return parser.parse_args()


def bootstrap_weights(design: np.ndarray, counts: np.ndarray, N: float,
                      rng: np.random.Generator) -> np.ndarray:
    """One weighted-Bayesian-bootstrap draw of the units' weights.

    ``design`` are the units' design weights, ``counts`` the sampled households
    each stands for (``c_i >= 1``). Each household's Dirichlet(1, ..., 1)
    component times its design weight ``w_i / c_i``, summed over the unit's
    households, is ``(w_i / c_i) G_i`` with ``G_i ~ Gamma(c_i, 1)``; the result
    is rescaled to sum to ``N``.
    """
    design, counts = np.asarray(design, float), np.asarray(counts, float)
    if np.any(counts < 1) or np.any(counts != np.round(counts)):
        raise ValueError("every unit must stand for a whole number (>= 1) of households")
    weight = design / counts * rng.gamma(shape=counts, scale=1.0)
    return weight * (N / weight.sum())


def replicate(local, name: str, rng: np.random.Generator):
    """The local fit's problem with bootstrap weights, as PUMA ``name``."""
    from pmedm_vb.assemble.design import design_weights

    units = local.units.copy()
    design = units["weight"].to_numpy(dtype=float)
    weight = bootstrap_weights(design, units["pums_units"].to_numpy(), local.N, rng)
    units["design_weight"] = design
    units["bootstrap_multiplier"] = weight / design
    units["weight"] = weight
    out = dataclasses.replace(local, q=design_weights(units["weight"], n_zones=local.n_zones),
                              units=units, puma=name)
    out.validate()
    if not np.isclose(weight.sum(), out.N, rtol=1e-9):
        raise SystemExit(f"{name}: weights sum to {weight.sum():,.3f}, not N = {out.N:,.3f}")
    out.zero_cell_variances()  # unchanged Sigma, but the fits need it: fail here, not there
    return out


def main() -> None:
    args = parse_args()
    sim = Sim02(args.sim, args.root)
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    if sim.bootstrap is None:
        raise SystemExit(f"{args.sim}: no 'bootstrap' block")
    names = sim.boot_pumas()
    existing = [n for n in names if (sim.fit_inputs("boot", n) / "manifest.json").exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{sim.fit_inputs('boot', existing[0])} exists "
                         f"({len(existing)} of {len(names)} replicates); --overwrite to redraw")
    seed, R = int(sim.bootstrap["seed"]), sim.replicates()

    # Draw and check everything before writing anything.
    drawn, record = [], {"method": METHOD, "seed": seed, "replicates": R,
                         "fits": sim.bootstrap["fits"], "truths": {}}
    for truth in sim.boot_truths():
        source = sim.fit_inputs("local", truth)
        if not (source / "manifest.json").exists() or not HeldOut.exists(source):
            raise SystemExit(f"{source}: the local fit's inputs or held-out tables are missing "
                             "(sim02_build.py writes them)")
        local = PMEDMInputs.load(source)
        if local.unit_members is not None or "pums_units" not in local.units:
            raise SystemExit(f"{source}: not a sim02 local fit (statewide, or no pums_units)")
        if "bootstrap_multiplier" in local.units:
            raise SystemExit(f"{source}: already a bootstrap replicate")
        counts = local.units["pums_units"].to_numpy()
        if int(counts.sum()) != local.n or not np.isclose(local.units["weight"].sum(), local.N):
            raise SystemExit(f"{source}: pums_units do not sum to n ({local.n}) or the weights "
                             f"to N ({local.N:,.1f})")
        k, i = sim.truth_index(truth)
        entry = {"source": str(source), "units": int(local.n_units), "n": int(local.n),
                 "N": float(local.N), "replicates": {}}
        for j in range(1, R + 1):
            name = sim.replicate_puma(truth, j)
            entropy = [seed, k, i, j]
            out = replicate(local, name, np.random.default_rng(entropy))
            m = out.units["bootstrap_multiplier"].to_numpy()
            summary = {"seed": entropy, "multiplier_min": float(m.min()),
                       "multiplier_max": float(m.max()),
                       "multiplier_sd": float(m.std()),
                       "weight_sum": float(out.units["weight"].sum())}
            entry["replicates"][name] = summary
            drawn.append((name, out, source, {"truth": truth, "replicate": j, "method": METHOD,
                                              "source": str(source), **summary}))
            print(f"  {name}: multipliers {m.min():.3g} to {m.max():.3g} (sd {m.std():.3g}) "
                  f"over {out.n_units:,} units, n {out.n:,}, N {out.N:,.0f}; seed {entropy}",
                  flush=True)
        record["truths"][truth] = entry

    for name, out, source, note in drawn:
        path = sim.fit_inputs("boot", name)
        if path.exists():
            shutil.rmtree(path)  # no file of an earlier replicate survives a redraw
        out.save(path)
        HeldOut.load(source).save(path)
        (path / "bootstrap.json").write_text(json.dumps(note, indent=2) + "\n")
    print(f"wrote {len(drawn)} replicates under {sim.fit_inputs('boot', '')}")
    fitted = [n for n in existing if (args.root / sim.fits["boot"]["name"] / n).exists()]
    if fitted:
        print(f"note: {len(fitted)} redrawn replicate(s) have fits from the old weights under "
              f"{args.root / sim.fits['boot']['name']}; delete them to refit: {' '.join(fitted)}")
    sim.folder.mkdir(parents=True, exist_ok=True)
    (sim.folder / "bootstrap.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {sim.folder / 'bootstrap.json'}")


if __name__ == "__main__":
    main()
