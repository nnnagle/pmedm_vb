"""Diagnostics for a stricter VB truncation: how many draws each rule turns away.

Today's truncated VB (``compare_methods.truncated_draws``) rejects a draw with
any cell (one record in one block group) over ``--share`` of N, or any block
group (over ``--hh-min`` published households) whose D/H is under
``--distinct``. The candidate rule also rejects a draw with any block group's
allocated occupied households more than ``--hh-z`` SEs from its published
count (B25003, the SE the square root of its cells' summed variances): Table
5's "households |z|", ``compare_methods.household_consistency``.

For each experiment, PUMA and alpha it takes ``--draws`` draws of the VB
Gaussian fit and as many of the HMC reference's (evenly thinned), and records
per draw its largest cell share, smallest D/H and largest households |z|. The
reference shows what each test turns away of the posterior itself. Nothing is
scored and nothing the paper tables read is written. Writes to ``--out``:

``trunc_draws.csv``
    One row per (experiment, PUMA, alpha, method, draw): ``max_share``,
    ``dh_min``, ``hh_z_max`` and the tests each fails.
``trunc_summary.csv``, printed too
    Per (experiment, alpha, method), and per PUMA: the share of draws each test
    rejects, today's rule (share or D/H), the new rule (share, D/H or |z|),
    and, for VB, the tries the new rule needs for the scoring's 4,000 draws
    against the cap of 20 times that.

    sbatch --account=acf-utk0011 experiments/run_python.sbatch trunc_check.py \\
        --root $R --out $R/trunc_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402

#: The scoring's draws per method and compare_methods.py's --trunc-max-factor.
SCORING_DRAWS, MAX_FACTOR = 4000, 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiments", nargs="+", default=["exp01_baseline", "exp03_rollup"])
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--alphas", nargs="+", type=float, default=None)
    parser.add_argument("--draws", type=int, default=2000, help="draws per fit and method")
    parser.add_argument("--share", type=float, default=0.01, help="a cell over this share of N")
    parser.add_argument("--distinct", type=float, default=0.04, help="a block group's D/H under this")
    parser.add_argument("--hh-z", type=float, default=6.0, help="households |z| over this")
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


class Checks:
    """One fit's per-draw statistics, from ``lam`` in the inputs' layout."""

    def __init__(self, inputs, variance_floor: str, hh_min: float) -> None:
        from compare_methods import floor_spec

        from pmedm_vb.assemble.allcells import table_leaves
        from pmedm_vb.solvers.base import ConstraintOperator

        self.inputs = inputs
        self.op = ConstraintOperator(inputs)
        tenure = table_leaves(inputs.bg_constraints, "B25003")  # owner + renter
        if not tenure:
            raise SystemExit("no B25003 leaves among the block group constraints")
        self.hh = ~inputs.units["is_group_quarters"].to_numpy(bool)
        self.published = inputs.Y_B[:, tenure].sum(axis=1)
        self.zones = self.published > hh_min
        n_zones = inputs.Y_B.shape[0]
        rows = inputs.Y_T.size + np.array(tenure)[None, :] * n_zones + np.arange(n_zones)[:, None]
        self.se = np.sqrt(inputs.floored_variances(floor_spec(variance_floor))[rows].sum(axis=1))
        # Each unit's occupied households, as the B25003 leaves count them.
        self.households = np.asarray(inputs.X_B[:, tenure].sum(axis=1)).ravel()

    def __call__(self, lam: np.ndarray) -> dict:
        from compare_methods import expected_distinct, log_p_of

        log_p = log_p_of(self.inputs, self.op, lam)                     # (zones, units)
        W = self.inputs.N * np.exp(log_p)
        z = (W @ self.households - self.published) / self.se
        dh = np.nan
        if self.zones.any():
            w = W[np.ix_(self.zones, self.hh)]
            s = w / w.sum(axis=1, keepdims=True)
            H = self.published[self.zones]
            dh = float((expected_distinct(s, H) / H).min())
        return dict(max_share=float(np.exp(log_p.max())), dh_min=dh,
                    hh_z_max=float(np.abs(z).max()))


def fit_draws(spec: dict, exp: Path, puma: str, alpha: float, inputs, draws: int, rng):
    """``{method: lam (m, draws)}``: the VB Gaussian fit's draws and the reference's."""
    from compare_methods import check_level, floor_spec, taper_spec, thinned, with_lambda
    from laplace_diagnostic import load_vb

    from pmedm_vb.assemble.hierarchy import Hierarchy

    name = rx.fit_name(spec, puma, alpha)
    level = rx.level(spec)
    out = {}
    path = exp / puma / "vb_gaussian" / f"{name}.npz"
    if path.exists():
        with np.load(path) as saved:
            check_level(path, saved, level)
        q, _, _ = load_vb(path)
        model = spec["model"]
        h = Hierarchy.build(inputs, level, alpha=alpha, taper=taper_spec(model["taper"]),
                            variance_floor=floor_spec(str(model["variance_floor"])))
        out["vb_gaussian"] = with_lambda(h, q.sample(rng, draws), q)["lam"]
    trace = exp / puma / "hmc_ref" / f"{name}_trace.npz"
    if trace.exists():
        with np.load(trace) as saved:
            out["hmc_ref"] = thinned(saved["lam"], draws)
    return out


def summarise(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(by, sort=False):
        row = dict(zip(by, key), draws=len(group))
        for test in ("share", "distinct", "hh_z", "today", "new"):
            row[f"rejected_{test}"] = float(group[f"fails_{test}"].mean())
        accept = 1 - row["rejected_new"]
        row["new_tries_for_4000"] = SCORING_DRAWS / accept if accept > 0 else np.inf
        row["new_within_cap"] = bool(accept * MAX_FACTOR >= 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.config import processed_dir

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.experiments:
        exp = args.root / name
        spec = json.loads((exp / "experiment.json").read_text())
        os.environ["PMEDM_VB_DATA"] = str(rx.data_dir(spec))
        pumas = args.pumas or spec["pumas"]
        alphas = args.alphas or spec["alphas"]
        for puma in pumas:
            inputs = PMEDMInputs.load(processed_dir() / "inputs" / rx.area_slug(spec["area"]) / puma)
            checks = Checks(inputs, str(spec["model"]["variance_floor"]), args.hh_min)
            for alpha in alphas:
                rng = np.random.default_rng([args.seed, pumas.index(puma), int(round(1000 * alpha))])
                for method, lam in fit_draws(spec, exp, puma, alpha, inputs, args.draws, rng).items():
                    for d in range(lam.shape[1]):
                        records.append(dict(experiment=name, puma=puma, alpha=alpha,
                                            method=method, draw=d + 1, **checks(lam[:, d])))
                    print(f"{name} {puma} alpha {alpha:g} {method}: {lam.shape[1]} draws",
                          flush=True)
    if not records:
        raise SystemExit("no fits found")
    frame = pd.DataFrame(records)
    frame["fails_share"] = frame.max_share > args.share
    frame["fails_distinct"] = frame.dh_min < args.distinct
    frame["fails_hh_z"] = frame.hh_z_max > args.hh_z
    frame["fails_today"] = frame.fails_share | frame.fails_distinct
    frame["fails_new"] = frame.fails_today | frame.fails_hh_z
    frame.to_csv(args.out / "trunc_draws.csv", index=False)
    summary = pd.concat([
        summarise(frame, ["experiment", "alpha", "method"]).assign(puma="all"),
        summarise(frame, ["experiment", "alpha", "method", "puma"]),
    ], ignore_index=True)
    summary.to_csv(args.out / "trunc_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nshare > {args.share:g}, D/H < {args.distinct:g}, households |z| > {args.hh_z:g}")
        print(summary.round(3).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
