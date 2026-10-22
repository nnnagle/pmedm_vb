"""Are the HMC reference draws' errors consistent with the model's Sigma?

For each experiment, PUMA and alpha, takes the HMC reference draws the scoring
uses (the trace thinned evenly to ``--draws``) and each draw's error on the
model's own rows, ``e = N X'p - Y`` extended as the fit extends it (the PUMA
rows, a roll-up's cells), exactly as ``solve_map`` measures its ``mahalanobis``.
Two checks, against the model's Sigma -- the published covariance at that
alpha plus the multiplier prior's ``tau^2`` on the diagonal (``prior.py``) --
and, beside it, against the published variances alone:

1. **Per row, its diagonal.** ``z = e / sqrt(diag Sigma)``, pooled over draws
   and rows by group (tract, block group, PUMA rows; published zero or not).
   The **bulk** -- median |z| and the share within 1 and 1.96 -- and the
   **tails** -- the share over 3, 4, 5 and 5.9 and the 99.9% and 99.999%
   points of |z| -- each beside a standard normal's.
2. **Correlations included.** Per draw ``d = e' Sigma^-1 e`` over the ``m``
   rows, as ``d / m``. For the MAP the fitted residuals are shrunk by the
   penalty and ``d / m`` is below 1 when the model holds (``map_dual``); a draw
   far above 1 has errors Sigma does not allow.

Writes ``--out`` (per draw: ``d``, ``m``), ``<out>_hist.csv`` (per group, the
counts of |z| in log-spaced bins, both scalings, for plotting) and prints the
summaries::

    sbatch --export=ALL,PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
        experiments/run_python.sbatch draw_sigma.py --root $RUNS/paper_v3 \\
        --out $RUNS/paper_v3/draw_sigma.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_methods as cm  # noqa: E402
from score_experiment import cell_calls  # noqa: E402

from pmedm_vb.assemble.hierarchy import Hierarchy  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402

BULK = (1.0, 1.96)
TAILS = (3.0, 4.0, 5.0, 5.9)
POINTS = (0.999, 0.99999)
#: |z| bins: log-spaced from 1e-4 to 1e4, the thresholds above exact edges, with
#: 0 and +inf at the ends.
EDGES = np.unique(np.concatenate([[0.0], np.logspace(-4, 4, 1601), BULK, TAILS, [np.inf]]))
SCALINGS = ("model", "published")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--pumas", nargs="+", default=None, help="default: each experiment's")
    parser.add_argument("--alphas", nargs="+", type=float, default=None,
                        help="default: each experiment's")
    parser.add_argument("--draws", type=int, default=4000, help="as the scoring thins the trace")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def model_args(spec: dict, exp: Path, puma: str, alpha: float) -> argparse.Namespace | None:
    """compare_methods.py's arguments for this cell's HMC reference, as scored."""
    calls, _ = cell_calls(spec, exp, puma, alpha, Path("unused"), 1)
    for call in calls:
        tokens = call.split()
        if tokens[tokens.index("--method") + 1] == "hmc_ref":
            argv = sys.argv
            try:
                sys.argv = ["compare_methods.py", *tokens]
                return cm.parse_args()
            finally:
                sys.argv = argv
    return None


def row_groups(h, m: int, split: int) -> np.ndarray:
    """Each model row's level: tract, block group, or PUMA (the appended rows)."""
    level = np.full(h.size, "PUMA", dtype=object)
    tract = h.cell_is_tract if hasattr(h, "cell_is_tract") else np.arange(m) < split
    level[:m] = np.where(tract, "tract", "block group")
    return level


def quantile_from_hist(counts: np.ndarray, p: float) -> float:
    """The ``p`` quantile of |z| from its binned counts, log-interpolated in its bin."""
    cum = np.cumsum(counts) / counts.sum()
    i = int(np.searchsorted(cum, p))
    lo, hi = EDGES[i], EDGES[i + 1]
    if not np.isfinite(hi) or lo == 0:
        return float(hi if lo == 0 else lo)
    before = cum[i - 1] if i else 0.0
    frac = (p - before) / (cum[i] - before) if cum[i] > before else 0.5
    return float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))


def summary(hist: pd.DataFrame) -> pd.DataFrame:
    """Bulk and tails of |z| per group and scaling, with a standard normal's."""
    rows = []
    keys = ["experiment", "alpha", "scaling", "level", "published"]
    for key, group in hist.groupby(keys):
        counts = np.zeros(EDGES.size - 1)
        counts[group["bin"].to_numpy()] = group["count"].to_numpy(float)
        total = counts.sum()
        upper = EDGES[1:]
        row = dict(zip(keys, key), values=int(total),
                   median=quantile_from_hist(counts, 0.5))
        for b in BULK:
            row[f"within_{b:g}"] = counts[upper <= b].sum() / total
        for t in TAILS:
            row[f"over_{t:g}"] = counts[EDGES[:-1] >= t].sum() / total
        for p in POINTS:
            row[f"q{p:g}"] = quantile_from_hist(counts, p)
        rows.append(row)
    out = pd.DataFrame(rows).set_index(keys)
    reference = dict(values=np.nan, median=norm.ppf(0.75),
                     **{f"within_{b:g}": 2 * norm.cdf(b) - 1 for b in BULK},
                     **{f"over_{t:g}": 2 * norm.sf(t) for t in TAILS},
                     **{f"q{p:g}": norm.ppf(0.5 + p / 2) for p in POINTS})
    out.loc[("standard normal", np.nan, "", "", ""), :] = reference
    return out


def main() -> None:
    args = parse_args()
    draws, hists = [], []
    for name in args.experiments:
        exp = args.root / name
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            inputs = None
            for alpha in args.alphas or spec["alphas"]:
                margs = model_args(spec, exp, puma, alpha)
                if margs is None:
                    print(f"skipping {name} {puma} alpha {alpha:g}: no HMC reference", flush=True)
                    continue
                if inputs is None:
                    inputs = PMEDMInputs.load(processed_dir() / "inputs" / margs.area / puma)
                    op = ConstraintOperator(inputs)
                h = Hierarchy.build(inputs, margs.hierarchy, alpha=alpha,
                                    taper=cm.taper_spec(margs.taper),
                                    variance_floor=cm.floor_spec(margs.variance_floor))
                sigma = h.sigma(inputs, alpha, cm.taper_spec(margs.taper),
                                cm.floor_spec(margs.variance_floor))
                targets = h.targets_ext
                sd = {"model": np.sqrt(sigma.diagonal()), "published": np.sqrt(h.v_ext)}
                level = row_groups(h, h.m, inputs.Y_T.size)
                zero = np.where(targets == 0, "zero", "nonzero")
                labels = pd.MultiIndex.from_arrays([level, zero]).unique()
                masks = {lab: (level == lab[0]) & (zero == lab[1]) for lab in labels}
                counts = {(s, lab): np.zeros(EDGES.size - 1) for s in SCALINGS for lab in labels}
                with np.load(margs.run / f"{cm.fit_name(puma, alpha, margs.hierarchy, margs.taper)}"
                                         f"_trace.npz") as saved:
                    lam = cm.thinned(saved["lam"], args.draws)
                for d in range(lam.shape[1]):
                    W = inputs.N * np.exp(cm.log_p_of(inputs, op, lam[:, d]))
                    e = h.extend(op.forward(W)) - targets
                    draws.append(dict(experiment=name, puma=puma, alpha=alpha, draw=d,
                                      d=float(e @ sigma.solve(e)), m=e.size))
                    for s in SCALINGS:
                        z = np.abs(e / sd[s])
                        for lab, mask in masks.items():
                            counts[(s, lab)] += np.histogram(z[mask], EDGES)[0]
                for (s, (lev, pub)), c in counts.items():
                    hists.append(pd.DataFrame(dict(experiment=name, puma=puma, alpha=alpha,
                                                   scaling=s, level=lev, published=pub,
                                                   bin=np.arange(c.size), lo=EDGES[:-1],
                                                   hi=EDGES[1:], count=c)))
                print(f"{name} {puma} alpha {alpha:g}: {lam.shape[1]} draws, {h.size:,} rows, "
                      f"tau^2 {sigma.prior_var:.1f}", flush=True)
    if not draws:
        raise SystemExit("no traces found for these experiments, PUMAs and alphas")
    frame = pd.DataFrame(draws)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    hist = pd.concat(hists, ignore_index=True)
    hist = hist[hist["count"] > 0]
    hist.to_csv(args.out.with_name(args.out.stem + "_hist.csv"), index=False)

    pd.set_option("display.width", 220)
    frame["ratio"] = frame.d / frame.m
    print("\nCorrelations included: d / m = e' Sigma^-1 e / m per draw (the model's Sigma); for "
          "the MAP below 1 when the model holds:")
    print(frame.groupby(["experiment", "alpha"]).agg(
        draws=("d", "size"), m=("m", "median"), p05=("ratio", lambda x: x.quantile(0.05)),
        median=("ratio", "median"), p95=("ratio", lambda x: x.quantile(0.95)),
        max=("ratio", "max")).round(3).to_string())
    pooled = hist.groupby(["experiment", "alpha", "scaling", "level", "published", "bin"],
                          as_index=False)["count"].sum()
    print("\nPer row, z = e / sqrt(diag Sigma), all draws and PUMAs pooled; 'model' is the "
          "model's Sigma (published v + tau^2), 'published' v alone. Bulk: median |z|, share "
          "within 1 and 1.96; tails: share over 3-5.9, and the 99.9% and 99.999% points:")
    print(summary(pooled).to_string(float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
