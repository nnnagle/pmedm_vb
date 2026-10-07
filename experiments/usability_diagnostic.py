"""Which draws are usable? Two block-group rules, measured on every method's draws.

A draw is *usable* when every block group has a varied representation of
records and an allocation consistent with its published count. Two rules, each
on a grid of thresholds, applied to nothing yet -- this only measures them:

- **variety**: in every block group with more than ``--hh-min`` published
  occupied households (B25003), the household records' Kish effective number
  ``n_eff = 1 / sum s^2`` (``s``: each record's share of the block group's
  allocated households) is at least ``f``, for ``f`` in ``--neff-floors``.
  ``n_eff >= f`` also caps any one record's share at ``1 / sqrt(f)``.
- **consistency**: in every block group, empty and small ones included, the
  allocated occupied households (the model's B25003 total, ``N`` times the
  forward map of ``p``) are within ``z`` standard errors of the published
  total, for ``z`` in ``--z``. The standard error is the square root of the
  summed variances of the block group's B25003 rows, with the zero-count floor
  the model uses (``--variance-floor``); the rows' covariance is ignored.

For each fit (PUMA x alpha) and method -- the experiment's VB families and both
HMC runs -- it draws ``--draws`` (VB) or thins the trace to that many (HMC),
and reports the share of draws each rule, and both, would keep; and of the VB
draws with a wall (a cell over ``--wall`` of N, the scoring's definition), the
share each rule would reject.

Writes ``<out>/usability_draws.csv`` (one row per draw), ``usability_grid.csv``
(one row per fit, method and threshold pair) and ``usability.txt``, each with a
``_<pumas>`` suffix under ``--pumas``. One job per PUMA, then ``--combine``
reads the per-PUMA draws and writes the report over all of them::

    sbatch --export=ALL,PMEDM_VB_DATA=$PMEDM_VB_DATA experiments/run_python.sbatch \\
        usability_diagnostic.py $PAPER/exp01_baseline --pumas 4701501
    $CONDA_PREFIX/bin/python experiments/usability_diagnostic.py $PAPER/exp01_baseline --combine
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from laplace_diagnostic import load_vb  # noqa: E402
from run_experiment import area_slug, data_dir, fit_name  # noqa: E402

from pmedm_vb.assemble.hierarchy import Hierarchy  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402

NAME = re.compile(r"^(?P<puma>\d+)_(?P<taper>[^_]+)_a(?P<alpha>[0-9.e-]+)(?:_h(?P<level>.+))?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", type=Path, help="the experiment's folder (<root>/<name>)")
    parser.add_argument("--pumas", nargs="+", help="default: the experiment's")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--neff-floors", type=float, nargs="+", default=[5, 10, 20])
    parser.add_argument("--z", type=float, nargs="+", default=[3, 4, 5])
    parser.add_argument("--wall", type=float, default=0.01, help="a wall is a cell over this share of N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="default: <experiment>/usability")
    parser.add_argument("--combine", action="store_true",
                        help="read every usability_draws*.csv in --out and write one report")
    return parser.parse_args()


class Usability:
    """One PUMA's problem, and the two rules' statistics of a draw."""

    def __init__(self, inputs: PMEDMInputs, hh_min: float, variance_floor) -> None:
        self.inputs = inputs
        self.op = ConstraintOperator(inputs)
        with np.errstate(divide="ignore"):
            self.log_q = np.log(inputs.q)
        self.support = inputs.q > 0
        gq = (inputs.units["is_group_quarters"].to_numpy(bool)
              if "is_group_quarters" in inputs.units else np.zeros(inputs.n_units, bool))
        self.hh = ~gq
        tenure = [k for k, c in enumerate(inputs.bg_constraints) if c.startswith("B25003.")]
        if not tenure:
            raise SystemExit("no B25003 rows in the block-group constraints")
        n_zones = inputs.Y_B.shape[0]
        # Stacked rows of each block group's B25003 cells: [vec(Y_T); vec(Y_B)], column-major.
        self.rows = inputs.Y_T.size + np.array(tenure)[None, :] * n_zones + np.arange(n_zones)[:, None]
        variance = inputs.floored_variances(variance_floor)
        self.published = inputs.Y_B[:, tenure].sum(axis=1)
        self.se = np.sqrt(variance[self.rows].sum(axis=1))
        self.zones = np.flatnonzero(self.published > hh_min)

    def stats(self, lam: np.ndarray) -> dict:
        with np.errstate(divide="ignore"):
            logits = self.log_q - self.op.adjoint(lam)
        log_p = logits - logsumexp(logits[self.support])
        p = np.where(self.support, np.exp(log_p), 0.0)
        top = np.unravel_index(np.argmax(p), p.shape)
        allocated = self.inputs.N * self.op.forward(p)[self.rows].sum(axis=1)
        z = (allocated - self.published) / self.se
        k = int(np.argmax(np.abs(z)))
        row = dict(max_cell_share=float(p[top]), max_cell_zone=int(top[0]),
                   max_abs_z=float(abs(z[k])), z=float(z[k]), z_zone=int(k),
                   z_published=float(self.published[k]), z_allocated=float(allocated[k]),
                   z_se=float(self.se[k]))
        if self.zones.size:
            w = p[np.ix_(self.zones, self.hh)]
            total = w.sum(axis=1, keepdims=True)
            s = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
            sum_sq = np.square(s).sum(axis=1)
            n_eff = np.divide(1.0, sum_sq, out=np.zeros_like(sum_sq), where=sum_sq > 0)
            j = int(np.argmin(n_eff))
            row.update(min_n_eff=float(n_eff[j]), min_n_eff_zone=int(self.zones[j]),
                       min_n_eff_published=float(self.published[self.zones[j]]),
                       min_n_eff_top_share=float(s[j].max()),
                       max_hh_share=float(s.max()))
        return row


def draws_of(exp: Path, puma: str, method: str, name: str, h, args, rng) -> np.ndarray | None:
    """``(m, draws)`` multipliers in the data's layout, or None if the fit is missing."""
    if method.startswith("vb_"):
        path = exp / puma / method / f"{name}.npz"
        if not path.exists():
            return None
        q, _, _ = load_vb(path)
        xi = q.sample(rng, args.draws)
        return xi if h.is_trivial else h.lambda_data(xi)
    path = exp / puma / method / f"{name}_trace.npz"
    if not path.exists():
        return None
    with np.load(path) as saved:
        kept = saved["lam"]                                                  # (kept, chains, m)
    flat = kept.reshape(-1, kept.shape[-1])
    pick = np.unique(np.linspace(0, len(flat) - 1, min(args.draws, len(flat))).astype(int))
    return flat[pick].T


def grid(draws: pd.DataFrame, args) -> pd.DataFrame:
    """Per fit and method, the share kept under each (n_eff floor, z) pair -- 0 and
    inf meaning that rule is off -- and the share of wall draws rejected."""
    floors = [0.0] + list(args.neff_floors)
    zs = list(args.z) + [np.inf]
    rows = []
    for (puma, alpha, method), part in draws.groupby(["puma", "alpha", "method"], sort=True):
        wall = part.max_cell_share > args.wall
        for f in floors:
            variety = (part.min_n_eff >= f) if "min_n_eff" in part else pd.Series(True, part.index)
            for z in zs:
                keep = variety & (part.max_abs_z <= z)
                rows.append(dict(puma=puma, alpha=alpha, method=method, neff_floor=f, z=z,
                                 draws=len(part), walls=int(wall.sum()), kept=float(keep.mean()),
                                 walls_rejected=float((~keep[wall]).mean()) if wall.any() else np.nan))
    return pd.DataFrame(rows)


def label_f(f: float) -> str:
    return "off" if f == 0 else f"{f:g}"


def label_z(z: float) -> str:
    return "off" if np.isinf(z) else f"{z:g}"


def report(draws: pd.DataFrame, table: pd.DataFrame, args, methods: list[str]) -> str:
    lines = [f"Usable draws. variety: n_eff >= floor in every block group over {args.hh_min:g} "
             f"published households (n_eff >= f caps one record at 1/sqrt(f) of a block group); "
             f"consistency: every block group's allocated households within z SE of B25003 "
             f"(zero-count floor, covariance ignored). Up to "
             f"{int(draws.groupby(['puma', 'alpha', 'method']).size().max())} draws per fit; "
             f"wall = a cell over {args.wall:g} of N.",
             "Medians over PUMAs [min, max]."]

    def cell(x: pd.Series, fmt: str = "{:.3f}") -> str:
        x = x.dropna()
        if x.empty:
            return "-"
        return (fmt.format(x.median()) if x.min() == x.max()
                else f"{fmt.format(x.median())} [{fmt.format(x.min())},{fmt.format(x.max())}]")

    order = {m: i for i, m in enumerate(methods)}
    for alpha in sorted(draws.alpha.unique()):
        part = draws[draws.alpha == alpha]
        lines.append(f"\n== alpha {alpha:g}")
        lines.append("Per draw, by method: the walls; the smallest n_eff (p1, p50); its block group's "
                     "top record share (p50); the largest |z| (p50, p99):")
        rows = []
        for method in sorted(part.method.unique(), key=order.get):
            by = part[part.method == method].groupby("puma")
            row = {"method": method,
                   "walls": cell(by.max_cell_share.apply(lambda x: (x > args.wall).mean()))}
            if "min_n_eff" in part:
                row["neff_min_p1"] = cell(by.min_n_eff.quantile(0.01), "{:.1f}")
                row["neff_min_p50"] = cell(by.min_n_eff.median(), "{:.1f}")
                row["its_top_share_p50"] = cell(by.min_n_eff_top_share.median())
            row["absz_max_p50"] = cell(by.max_abs_z.median(), "{:.2f}")
            row["absz_max_p99"] = cell(by.max_abs_z.quantile(0.99), "{:.2f}")
            rows.append(row)
        lines.append(pd.DataFrame(rows).set_index("method").to_string())

        sub = table[table.alpha == alpha]
        lines.append("\nShare of draws kept: rows n_eff floor, columns z ('off': rule not applied).")
        for method in sorted(sub.method.unique(), key=order.get):
            m = sub[sub.method == method]
            pivot = {label_z(z): {label_f(f): cell(g.kept) for f, g in mz.groupby("neff_floor")}
                     for z, mz in m.groupby("z")}
            frame = pd.DataFrame(pivot)
            frame = frame[[label_z(z) for z in sorted(m.z.unique())]]
            lines.append(f"  {method}:")
            lines += ["    " + r for r in frame.to_string().splitlines()]
        walls = sub[sub.walls > 0]
        if not walls.empty:
            lines.append("\nShare of wall draws rejected (methods with walls):")
            for method in sorted(walls.method.unique(), key=order.get):
                m = walls[walls.method == method]
                pivot = {label_z(z): {label_f(f): cell(g.walls_rejected) for f, g in mz.groupby("neff_floor")}
                         for z, mz in m.groupby("z")}
                frame = pd.DataFrame(pivot)
                frame = frame[[label_z(z) for z in sorted(m.z.unique())]]
                lines.append(f"  {method}:")
                lines += ["    " + r for r in frame.to_string().splitlines()]

    lines.append("\nWhere the largest |z| falls, over all draws: the published households of its "
                 "block group (quartiles), the share of draws where it is a block group with "
                 "none published, and the share where it is over-allocated (z > 0):")
    rows = []
    for (alpha, method), part in draws.groupby(["alpha", "method"], sort=True):
        rows.append(dict(alpha=alpha, method=method,
                         published_q25=part.z_published.quantile(0.25),
                         published_q50=part.z_published.median(),
                         published_q75=part.z_published.quantile(0.75),
                         empty_bg=(part.z_published == 0).mean(),
                         over_allocated=(part.z > 0).mean()))
    lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    exp = args.experiment.resolve()
    spec = json.loads((exp / "experiment.json").read_text())
    os.environ.setdefault("PMEDM_VB_DATA", str(data_dir(spec)))
    out = args.out or exp / "usability"
    out.mkdir(parents=True, exist_ok=True)
    methods = [f"vb_{f}" for f in spec["vb_families"]] + ["hmc_short", "hmc_ref"]
    if args.combine:
        parts = sorted(out.glob("usability_draws*.csv"))
        parts = [path for path in parts if path.name != "usability_draws.csv"] or parts
        if not parts:
            raise SystemExit(f"no usability_draws*.csv in {out}")
        draws = pd.concat([pd.read_csv(path, dtype={"puma": str}) for path in parts],
                          ignore_index=True)
        write(draws, out, "", args, methods)
        return
    rng = np.random.default_rng(args.seed)
    model = spec["model"]
    rows = []
    for puma in args.pumas or spec["pumas"]:
        inputs = PMEDMInputs.load(processed_dir() / "inputs" / area_slug(spec["area"]) / puma)
        use = Usability(inputs, args.hh_min, model["variance_floor"])
        for alpha in spec["alphas"]:
            name = fit_name(spec, puma, alpha)
            match = NAME.match(name)
            level = match.group("level") or "none"
            h = Hierarchy.build(inputs, level, alpha=alpha,
                                taper=None if model["taper"] == "none" else model["taper"],
                                variance_floor=model["variance_floor"])
            for method in methods:
                lam = draws_of(exp, puma, method, name, h, args, rng)
                if lam is None:
                    print(f"missing {puma} a{alpha:g} {method}", flush=True)
                    continue
                for d in range(lam.shape[1]):
                    rows.append(dict(puma=puma, alpha=alpha, method=method, draw=d,
                                     **use.stats(lam[:, d])))
                print(f"done {puma} a{alpha:g} {method}: {lam.shape[1]} draws", flush=True)
    if not rows:
        raise SystemExit("no fits found")
    draws = pd.DataFrame(rows)
    suffix = "" if args.pumas is None else "_" + "_".join(args.pumas)
    draws.to_csv(out / f"usability_draws{suffix}.csv", index=False)
    write(draws, out, suffix, args, methods)


def write(draws: pd.DataFrame, out: Path, suffix: str, args, methods: list[str]) -> None:
    table = grid(draws, args)
    table.to_csv(out / f"usability_grid{suffix}.csv", index=False)
    text = report(draws, table, args, methods)
    (out / f"usability{suffix}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
