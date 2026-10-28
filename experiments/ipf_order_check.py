"""Does the order of IPF's sweep decide which margins it misses?

IPF (``pmedm_vb.rake.rake_ipf``) visits the tract-only tables' tract margins
and then the block group margins in each sweep, and stops at ``max_sweeps``
when the margins cannot all be met. The hypothesis: the margins visited last
are met most nearly, so the misfit lands on the tract-only tables. For each
PUMA of the experiment, rakes twice with the experiment's roll-up, tract
margins first (as the paper's IPF) and block group margins first, and scores
both fits on the published constraint leaf cells as the paper does (sampled
cells only, ``compare_methods.outcome_sets``): |z| p50 and p90 by level and
ACS table, the PUMAs pooled. Writes ``<out>/ipf_order_cells.csv`` (every
sampled leaf cell's |z| under both orders) and ``ipf_order.txt``::

    sbatch --account=acf-utk0011 --time=06:00:00 experiments/run_python.sbatch \\
        ipf_order_check.py --spec experiments/paper_v3/exp01_baseline.json \\
        --out $R/ipf_order
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402

ORDERS = (("tract_first", True), ("bg_first", False))
SUBSETS = {"constrained_block_group": "block group", "constrained_tract": "tract"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True, help="the experiment's JSON")
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--max-sweeps", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def cells(inputs, W: np.ndarray) -> pd.DataFrame:
    """|z| of ``W``'s fitted margins on the sampled constraint leaf cells."""
    from compare_methods import outcome_sets
    from pmedm_vb.assemble.allcells import is_subtotal

    frames = []
    for s in outcome_sets(inputs, None):
        if s["subset"] not in SUBSETS:
            continue
        fitted = np.asarray((s["X"].T @ W.T).T)  # (zones, cells)
        if s["level"] == "tract":
            fitted = np.asarray(inputs.A_T @ fitted)
        z = np.abs(fitted - s["Y"]) / np.maximum(s["se"], 1e-9)
        names = np.array(s["names"])
        leaf = np.array([not is_subtotal(n) for n in names])
        keep = s["sampled"] & leaf[None, :]
        areas, columns = np.nonzero(keep)
        frames.append(pd.DataFrame({"level": SUBSETS[s["subset"]], "area": areas,
                                    "cell": names[columns], "abs_z": z[areas, columns]}))
    frame = pd.concat(frames, ignore_index=True)
    frame["table"] = frame.cell.str.split(".", n=1).str[0]
    return frame


def summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    group = frame.groupby(by + ["order"]).abs_z
    out = pd.DataFrame({"n": group.size(), "p50": group.quantile(0.5),
                        "p90": group.quantile(0.9)})
    return out.unstack("order").swaplevel(axis=1).sort_index(axis=1)


def main() -> None:
    args = parse_args()
    spec = rx.load_experiment(args.spec)
    os.environ["PMEDM_VB_DATA"] = str(rx.data_dir(spec))
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.config import processed_dir
    from pmedm_vb.rake import rake_ipf, tract_only_categories

    args.out.mkdir(parents=True, exist_ok=True)
    root = processed_dir() / "inputs" / rx.study_area(spec["area"]).slug
    rollup = spec["model"].get("rollup")
    frames, fits = [], []
    for puma in args.pumas or spec["pumas"]:
        inputs = PMEDMInputs.load(root / puma)
        tract_only = sorted({inputs.tract_constraints[k].split(".")[0]
                             for k in tract_only_categories(inputs)})
        for order, tract_first in ORDERS:
            result = rake_ipf(inputs, rollup=rollup, max_sweeps=args.max_sweeps,
                              tract_first=tract_first)
            fits.append(dict(puma=puma, order=order, converged=result.converged,
                             sweeps=result.n_sweeps, max_residual=float(result.trace[-1]),
                             seconds=round(result.seconds)))
            frame = cells(inputs, result.W).assign(puma=puma, order=order)
            frame["tract_only"] = frame.table.isin(tract_only)
            frames.append(frame)
            print(f"{puma} {order}: {'converged' if result.converged else 'NOT converged'} "
                  f"after {result.n_sweeps} sweeps, max |residual| {result.trace[-1]:.3g}",
                  flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(args.out / "ipf_order_cells.csv", index=False)
    pd.set_option("display.width", 200)
    fmt = {"float_format": lambda x: f"{x:.3g}"}
    text = ["IPF fits:", pd.DataFrame(fits).to_string(index=False), "",
            "|z| of the sampled constraint leaf cells, PUMAs pooled, by level:",
            summary(frame, ["level"]).to_string(**fmt), "",
            "by level and whether the table is tract-only:",
            summary(frame, ["level", "tract_only"]).to_string(**fmt), "",
            "by level and ACS table:",
            summary(frame, ["level", "table"]).to_string(**fmt)]
    (args.out / "ipf_order.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
