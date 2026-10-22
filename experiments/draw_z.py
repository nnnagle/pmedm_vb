"""How far single HMC draws stray from the published tables, cell by cell.

For each experiment, PUMA and alpha, takes the HMC reference draws the scoring
uses (the trace thinned evenly to ``--draws``) and, in every draw, each leaf
cell's z = (allocated - published) / SE -- the SE the scoring uses, a zero
cell's modelled one included -- over every block group and tract. The cells are
grouped by level (block group, tract), kind (constraint, held out) and whether
the published value is zero; per draw and group it keeps the largest |z|, the
cells, and the cells over ``--z``. Writes one row per (experiment, PUMA, alpha,
draw, level, kind, zero) to ``--out`` and prints, per group, the share of draws
with a cell over ``--z`` and quantiles of the draws' largest |z|.

Per cell, over all the draws, it also writes ``--cells-out`` (default
``<out>_cells.csv``): the area and cell, published value and SE, the median
allocation, the median and 5% and 95% z, the largest |z| and the share of draws
over ``--z``; and it prints, per group, how many cells are **persistent
misfits** (median |z| over ``--z``: off in most draws) and how many
**excursions** (median |z| within it but some draw over), the share of the
draws' cells over ``--z`` that persistent misfits account for, the median
published value and SE of each kind of cell, and the ``--top-cells`` most
persistent misfits by median |z|::

    sbatch --export=ALL,PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
        experiments/run_python.sbatch draw_z.py --root $RUNS/paper_v3 \\
        --out $RUNS/paper_v3/draw_z.csv
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

import compare_methods as cm  # noqa: E402
from run_experiment import area_slug, fit_name  # noqa: E402

from pmedm_vb.assemble.allcells import is_subtotal  # noqa: E402
from pmedm_vb.assemble.heldout import HeldOut  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--pumas", nargs="+", default=None, help="default: each experiment's")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="default: each experiment's")
    parser.add_argument("--kind", default="hmc_ref", choices=["hmc_ref", "hmc_short"])
    parser.add_argument("--draws", type=int, default=4000, help="as the scoring thins the trace")
    parser.add_argument("--z", type=float, default=5.9, help="count the cells over this |z|")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cells-out", type=Path, default=None,
                        help="per-cell statistics (default: <out>_cells.csv)")
    parser.add_argument("--top-cells", type=int, default=15,
                        help="persistent misfits listed per experiment and alpha")
    args = parser.parse_args()
    if args.cells_out is None:
        args.cells_out = args.out.with_name(args.out.stem + "_cells.csv")
    return args


def cell_groups(inputs: PMEDMInputs, heldout: HeldOut | None) -> list[dict]:
    """Each (level, kind) set of leaf cells: X columns, published Y and SE,
    ``(areas, cells)``, and whether the areas are tracts."""
    variance = inputs.floored_variances(None)
    n_t, n_b = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    se_T = np.sqrt(variance[:inputs.Y_T.size]).reshape(-1, n_t).T
    se_B = np.sqrt(variance[inputs.Y_T.size:]).reshape(-1, n_b).T
    areas = {"block group": inputs.block_groups.iloc[:, 0].astype(str).to_numpy(),
             "tract": inputs.tracts.iloc[:, 0].astype(str).to_numpy()}
    sets = [("block group", "constraint", inputs.bg_constraints, inputs.X_B, inputs.Y_B, se_B),
            ("tract", "constraint", inputs.tract_constraints, inputs.X_T, inputs.Y_T, se_T)]
    if heldout is not None:
        for level in ("block group", "tract"):
            sets.append((level, "held out", heldout.names[level], heldout.X[level],
                         heldout.Y[level], np.sqrt(heldout.v[level])))
    out = []
    for level, kind, names, X, Y, se in sets:
        leaves = np.array([k for k, name in enumerate(names) if not is_subtotal(name)])
        out.append(dict(level=level, kind=kind, X=sp.csr_matrix(X)[:, leaves],
                        Y=np.asarray(Y)[:, leaves], se=se[:, leaves], tract=level == "tract",
                        names=np.asarray(names)[leaves], areas=areas[level]))
    return out


def cell_statistics(g: dict, values: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per cell of one group, over the draws ``values`` ``(draws, areas, cells)``
    of its signed z (NaN where the cell has no SE)."""
    keep = g["se"] > 0
    area, cell = np.nonzero(keep)
    z = values[:, area, cell]
    median = np.median(z, axis=0)
    return pd.DataFrame({
        "level": g["level"], "kind": g["kind"], "area": g["areas"][area], "cell": g["names"][cell],
        "published": g["Y"][area, cell], "se": g["se"][area, cell],
        "median_allocated": g["Y"][area, cell] + median * g["se"][area, cell],
        "median_z": median, "z_p05": np.quantile(z, 0.05, axis=0),
        "z_p95": np.quantile(z, 0.95, axis=0), "max_abs_z": np.abs(z).max(axis=0),
        "share_over": (np.abs(z) > threshold).mean(axis=0)})


def report_cells(cells: pd.DataFrame, args) -> None:
    """Persistent misfits against excursions, per group, and the worst misfits."""
    cells = cells.assign(published_zero=np.where(cells.published == 0, "zero", "nonzero"),
                         persistent=cells.median_z.abs() > args.z,
                         excursion=(cells.median_z.abs() <= args.z) & (cells.share_over > 0))
    keys = ["experiment", "alpha", "level", "kind", "published_zero"]
    rows = []
    for key, group in cells.groupby(keys):
        events = group.share_over.sum()  # expected cells over --z per draw, summed over PUMAs
        p, e = group[group.persistent], group[group.excursion]
        rows.append(dict(zip(keys, key), cells=len(group), persistent=len(p), excursion=len(e),
                         over_from_persistent=p.share_over.sum() / events if events else np.nan,
                         published_persistent=p.published.median(), se_persistent=p.se.median(),
                         published_all=group.published.median(), se_all=group.se.median()))
    pd.set_option("display.width", 220)
    print(f"\nPer cell over the draws, by group (four PUMAs together): persistent misfits "
          f"(median |z| > {args.z:g}), excursions (median |z| within it, some draw over), the "
          f"share of the draws' cells over {args.z:g} that persistent misfits account for, and "
          f"the median published value and SE of the persistent misfits and of all cells:")
    print(pd.DataFrame(rows).set_index(keys).round(3).to_string())
    for (experiment, alpha), group in cells[cells.persistent].groupby(["experiment", "alpha"]):
        top = group.reindex(group.median_z.abs().sort_values(ascending=False).index)
        print(f"\n{experiment}, alpha {alpha:g}: the {args.top_cells} persistent misfits with the "
              f"largest median |z| (of {len(group):,}):")
        print(top.head(args.top_cells)[["puma", "level", "kind", "area", "cell", "published", "se",
                                        "median_allocated", "median_z", "share_over"]]
              .round(2).to_string(index=False))


def main() -> None:
    args = parse_args()
    rows, cell_frames = [], []
    for name in args.experiments:
        exp = args.root / name
        spec = json.loads((exp / "experiment.json").read_text())
        pumas = args.pumas or spec["pumas"]
        alphas = args.alphas or spec["alphas"]
        for puma in pumas:
            path = processed_dir() / "inputs" / area_slug(spec["area"]) / puma
            inputs = PMEDMInputs.load(path)
            heldout = HeldOut.load(path) if HeldOut.exists(path) else None
            groups = cell_groups(inputs, heldout)
            op = ConstraintOperator(inputs)
            A_T = sp.csr_matrix(inputs.A_T)
            for alpha in alphas:
                trace = exp / puma / args.kind / f"{fit_name(spec, puma, alpha)}_trace.npz"
                if not trace.exists():
                    print(f"skipping {name} {puma} alpha {alpha:g}: no {trace}", flush=True)
                    continue
                with np.load(trace) as saved:
                    lam = cm.thinned(saved["lam"], args.draws)
                # Every cell's signed z in every draw, for the per-cell statistics.
                signed = [np.full((lam.shape[1], *g["Y"].shape), np.nan, np.float32)
                          for g in groups]
                for d in range(lam.shape[1]):
                    W = inputs.N * np.exp(cm.log_p_of(inputs, op, lam[:, d]))
                    for i, g in enumerate(groups):
                        allocated = np.asarray((g["X"].T @ W.T).T)
                        if g["tract"]:
                            allocated = np.asarray(A_T @ allocated)
                        ok = g["se"] > 0
                        z = (allocated - g["Y"]) / np.where(ok, g["se"], 1)
                        signed[i][d][ok] = z[ok]
                        z = np.abs(np.where(ok, z, 0))
                        zero = g["Y"] == 0
                        for is_zero in (True, False):
                            cells = ok & (zero if is_zero else ~zero)
                            values = z[cells]
                            rows.append(dict(experiment=name, puma=puma, alpha=alpha, draw=d,
                                             level=g["level"], kind=g["kind"],
                                             published="zero" if is_zero else "nonzero",
                                             cells=int(values.size),
                                             max_abs_z=float(values.max()) if values.size else np.nan,
                                             over=int((values > args.z).sum())))
                for g, values in zip(groups, signed):
                    cell_frames.append(cell_statistics(g, values, args.z).assign(
                        experiment=name, puma=puma, alpha=alpha))
                print(f"{name} {puma} alpha {alpha:g}: {lam.shape[1]} draws", flush=True)
    if not rows:
        raise SystemExit("no traces found for these experiments, PUMAs and alphas")
    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    keys = ["experiment", "alpha", "level", "kind", "published"]
    summary = frame.groupby(keys).agg(
        draws=("draw", "size"), cells=("cells", "median"),
        share_over=("over", lambda x: (x > 0).mean()), mean_over=("over", "mean"),
        p50=("max_abs_z", "median"), p90=("max_abs_z", lambda x: x.quantile(0.90)),
        p99=("max_abs_z", lambda x: x.quantile(0.99)), max=("max_abs_z", "max"))
    pd.set_option("display.width", 200)
    print(f"\nPer draw (one PUMA's draw), by cell group: share of draws with a cell over |z| "
          f"{args.z:g}, mean such cells, and quantiles of the draw's largest |z| (cells: per "
          f"PUMA, the median over PUMAs):")
    print(summary.round(3).to_string())
    print(f"wrote {args.out}")

    cells = pd.concat(cell_frames, ignore_index=True)
    cells.to_csv(args.cells_out, index=False)
    report_cells(cells, args)
    print(f"wrote {args.cells_out}")


if __name__ == "__main__":
    main()
