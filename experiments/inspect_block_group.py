"""Why is one block group's population concentrated on a few household records?

For one experiment, PUMA and alpha, finds the block group with the smallest
effective number of household records (n_eff = 1 / sum s^2 over the records'
shares s of its households, as compare_methods.py; block groups with more than
``--hh-min`` published occupied households, B25003's owner and renter cells) in
the IPF fit, or takes ``--zone``, and reports, for the IPF fit and for draws of
the HMC reference:

- the block group: its GEOID, tract, published households and population, its
  n_eff in the IPF fit, across the HMC draws, and under the base weights alone
  (each record's design weight spread evenly, before any constraint), and how
  many household records may be placed there;
- its published leaf cells, block group and tract, with SE: the targets the
  allocation has to meet, the zeros and small counts first;
- the records with the largest shares, in the IPF fit and on average over the
  HMC draws: each share, how often the record is the block group's largest in
  the HMC draws, its design weight and the published categories it contributes
  to (its row of X, the people or units it adds to each).

Writes the report to ``--out`` and prints it. Reads the experiment's saved fits,
so it needs ``PMEDM_VB_DATA`` pointing at the experiment's data::

    PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
    $CONDA_PREFIX/bin/python experiments/inspect_block_group.py \\
        --root $RUNS/paper_v3 --experiment exp01_baseline --puma 4701504 --alpha 0.01 \\
        --out $RUNS/paper_v3/block_group_4701504.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import area_slug, fit_name, rake_output  # noqa: E402

from pmedm_vb.assemble.allcells import is_subtotal, table_leaves  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import weights_from_lambda  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiment", default="exp01_baseline")
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--zone", type=int, default=None,
                        help="a block group's row (default: the IPF fit's smallest n_eff)")
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--draws", type=int, default=400, help="HMC reference draws used")
    parser.add_argument("--top", type=int, default=8, help="records listed")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def shares(W: np.ndarray, households: np.ndarray) -> np.ndarray:
    """Each household record's share of every block group's households."""
    W_hh = W[:, households]
    return W_hh / W_hh.sum(axis=1, keepdims=True)


def n_eff(s: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(s).sum(axis=-1)


def hmc_draws(path: Path, draws: int) -> np.ndarray:
    """``(m, <= draws)`` evenly spaced multipliers, today's layout, from a trace."""
    with np.load(path) as saved:
        kept = saved["lam"]
    flat = kept.reshape(-1, kept.shape[-1])
    pick = np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))
    return flat[pick].T


def describe_record(unit: int, inputs: PMEDMInputs, x_b: sp.csr_matrix) -> str:
    """The published leaf categories a record adds to, with how many."""
    row = x_b[unit].toarray().ravel()
    parts = [f"{inputs.bg_constraints[k]}={row[k]:g}" for k in np.flatnonzero(row)
             if not is_subtotal(inputs.bg_constraints[k])]
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    exp = args.root / args.experiment
    spec = json.loads((exp / "experiment.json").read_text())
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / area_slug(spec["area"]) / args.puma)
    households = ~inputs.units["is_group_quarters"].to_numpy(bool)
    hh_index = np.flatnonzero(households)
    tenure = table_leaves(inputs.bg_constraints, "B25003")
    published_hh = inputs.Y_B[:, tenure].sum(axis=1)
    eligible = published_hh > args.hh_min

    with np.load(rake_output(spec, exp, args.puma, "ipf")) as saved:
        W_ipf = np.asarray(saved["W"], dtype=float)
    s_ipf = shares(W_ipf, households)
    neff_ipf = n_eff(s_ipf)
    zone = (args.zone if args.zone is not None
            else int(np.flatnonzero(eligible)[np.argmin(neff_ipf[eligible])]))

    lam = hmc_draws(exp / args.puma / "hmc_ref" / f"{fit_name(spec, args.puma, args.alpha)}_trace.npz",
                    args.draws)
    s_hmc = np.empty((lam.shape[1], hh_index.size))
    for d in range(lam.shape[1]):
        W = inputs.N * weights_from_lambda(inputs, lam[:, d])
        s_hmc[d] = shares(W[[zone]], households)[0]
    neff_hmc = n_eff(s_hmc)
    s_base = shares(inputs.q[[zone]], households)[0]

    tract = int(inputs.zone_tracts()[zone])
    zone_id = inputs.zones.iloc[zone].to_dict() if hasattr(inputs.zones, "iloc") else zone
    tract_id = inputs.tracts.iloc[tract].to_dict() if hasattr(inputs.tracts, "iloc") else tract
    population = table_leaves(inputs.bg_constraints, "B01001")
    lines = [f"{args.experiment}, PUMA {args.puma}, alpha {args.alpha:g}: block group row {zone}",
             f"  block group {zone_id}, tract {tract_id}",
             f"  published occupied households {published_hh[zone]:,.0f} (B25003 owner + renter); "
             f"population {inputs.Y_B[zone, population].sum():,.0f} (B01001 leaves)",
             f"  household records that may be placed here: "
             f"{int((inputs.q[zone, households] > 0).sum()):,} of {hh_index.size:,}",
             f"  n_eff: IPF fit {neff_ipf[zone]:.2f}; HMC reference median "
             f"{np.median(neff_hmc):.2f} (90%: {np.quantile(neff_hmc, 0.05):.2f} to "
             f"{np.quantile(neff_hmc, 0.95):.2f}, {neff_hmc.size} draws); base weights "
             f"{n_eff(s_base):.1f}",
             f"  the IPF fit's smallest n_eff over the {int(eligible.sum())} block groups over "
             f"{args.hh_min:g} households: this one" if args.zone is None else "",
             ""]

    # Published leaf cells: the block group's and its tract's, zeros and small first.
    n_tracts = inputs.Y_T.shape[0]
    variance = inputs.floored_variances(None)
    rows = []
    for level, names, Y, offset, area, n_areas in (
            ("block group", inputs.bg_constraints, inputs.Y_B, inputs.Y_T.size, zone,
             inputs.Y_B.shape[0]),
            ("tract", inputs.tract_constraints, inputs.Y_T, 0, tract, n_tracts)):
        for k, name in enumerate(names):
            if is_subtotal(name):
                continue
            se = float(np.sqrt(variance[offset + k * n_areas + area]))
            rows.append(dict(level=level, cell=name, published=float(Y[area, k]), se=se))
    cells = pd.DataFrame(rows).sort_values(["level", "published"])
    lines += ["Published leaf cells (smallest first):",
              cells.to_string(index=False, float_format=lambda v: f"{v:,.1f}"), ""]

    # The records with the largest shares.
    x_b = sp.csr_matrix(inputs.X_B)
    mean_hmc = s_hmc.mean(axis=0)
    top_hmc = np.bincount(s_hmc.argmax(axis=1), minlength=hh_index.size) / s_hmc.shape[0]
    order = np.unique(np.concatenate([np.argsort(-s_ipf[zone])[:args.top],
                                      np.argsort(-mean_hmc)[:args.top]]))
    order = order[np.argsort(-(s_ipf[zone, order] + mean_hmc[order]))]
    lines.append("Records with the largest shares of the block group's households:")
    for j in order:
        unit = int(hh_index[j])
        lines += [f"  record {unit} (design weight {inputs.units['weight'].iloc[unit]:,.1f}): "
                  f"IPF share {s_ipf[zone, j]:.3f}; HMC mean share {mean_hmc[j]:.3f} "
                  f"(90%: {np.quantile(s_hmc[:, j], 0.05):.3f} to "
                  f"{np.quantile(s_hmc[:, j], 0.95):.3f}), largest in {top_hmc[j]:.0%} of draws; "
                  f"base share {s_base[j]:.4f}",
                  f"    {describe_record(unit, inputs, x_b)}"]
    report = "\n".join(line for line in lines if line is not None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
