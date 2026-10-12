"""Which constraints and households do the wide whitened directions move?

The whitened coordinates ``x = G' (xi - mu)`` (HMC's space; ``mu``, ``G`` the
skewed VB fit's median and Gaussian-part factor) along which the posterior is
far wider than VB's Gaussian are combinations of multipliers, not single ones
(``skew_check.py``). This asks what they are made of, to test the account that
they come from households with rare combinations of attributes.

For a whitened coordinate ``j`` with posterior sd ``s_j`` (from the reference
trace), a one-sd move is ``d_j = s_j G'^{-1} e_j`` in the sampling coordinates,
mapped to the multipliers the data see (``hierarchy.lambda_data``) and to each
(zone, unit) cell's tilt ``eta = X lambda`` (on the support). Directions come
in three groups by ``s_j``: the widest (over ``--wide-sd``, default 3, the
symmetric ones that set the run length), the moderate (1.5 to ``--wide-sd``,
the skewed ones) and the bulk (under 1.5); up to ``--per-group`` of each, drawn
at random where there are more.

Per direction:

- **constraints**: the one it moves most (its level, published value and SE),
  and the share of the move's squared length in its top 5 (1 = concentrated
  on a few);
- **how far it reaches**: the constraints carrying 90% of the move's squared
  length, how many there are and how many distinct tracts they lie in (a
  block group's tract is the first 11 digits of its GEOID). The variational
  family's correlations are per tract plus a low-rank term, so a direction
  spread over many tracts is one it can represent only through that term;
- **households**: the unit whose tilt changes most (largest ``|d eta|`` over
  zones), that change, the unit's MAP weight in that zone and in all, and its
  *rarest category*: of the table columns (tract and block-group) on which it
  is nonzero, the one the fewest units in the PUMA share, with that count and
  the share of all units whose rarest category is at least as rare (a
  percentile: small = a rare household).

Writes ``<out>/wide_directions.csv`` (medians per experiment, PUMA and group),
``wide_directions_detail.csv`` (every direction examined) and
``wide_directions.txt`` (the medians, then the ``--detail`` widest directions
of each fit with their top constraints and households). Needs the assembled
inputs, so ``PMEDM_VB_DATA`` must point at the paper's data::

    $PY experiments/wide_directions.py --root $PAPER --alpha 0.01 --pumas 4701501 \\
        --out $PAPER/oracle_scale/wide_directions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.progress import logger, record_run

from laplace_diagnostic import constraint_table
from run_experiment import area_slug, fit_name, level
from skew_check import shape

GROUPS = ("widest", "moderate", "bulk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' results root")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--kind", default="ref", help="the HMC run whose sds are used, hmc_<kind>")
    parser.add_argument("--wide-sd", type=float, default=3.0)
    parser.add_argument("--per-group", type=int, default=300)
    parser.add_argument("--detail", type=int, default=8, help="widest directions listed per fit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load(spec: dict, exp: Path, puma: str, alpha: float):
    from run_mcmc import load_problem

    model = spec["model"]
    problem = argparse.Namespace(
        area=area_slug(spec["area"]), puma=puma, alpha=alpha, taper=model["taper"],
        variance_floor=str(model["variance_floor"]), hierarchy=level(spec),
        vb_run=exp / puma / "vb_skewed")
    return load_problem(problem)


class Tilt:
    """``eta = X lambda`` on the support, for multipliers in the data layout."""

    def __init__(self, inputs) -> None:
        self.split = inputs.Y_T.size
        (self.n_t, self.c_t), (self.n_b, self.c_b) = inputs.Y_T.shape, inputs.Y_B.shape
        self.A_T = sp.csr_matrix(inputs.A_T)
        self.X_T, self.X_B = sp.csr_matrix(inputs.X_T), sp.csr_matrix(inputs.X_B)
        self.support = inputs.q > 0

    def __call__(self, lam: np.ndarray) -> np.ndarray:
        v_t = lam[: self.split].reshape(self.c_t, self.n_t).T          # (tracts, columns)
        v_b = lam[self.split:].reshape(self.c_b, self.n_b).T           # (zones, columns)
        eta = self.A_T.T @ (self.X_T @ v_t.T).T + (self.X_B @ v_b.T).T  # (zones, units)
        return np.where(self.support, eta, 0.0)


def rarity(inputs) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Per unit: its rarest category's column, how many units share it, and
    the share of units whose rarest category is at least as rare."""
    X = sp.hstack([sp.csr_matrix(inputs.X_T), sp.csr_matrix(inputs.X_B)]).tocsr()
    names = [f"T:{c}" for c in inputs.tract_constraints] + [f"B:{c}" for c in inputs.bg_constraints]
    present = (X != 0).astype(np.int64)
    column_n = np.asarray(present.sum(axis=0)).ravel()
    rarest_col = np.full(X.shape[0], -1)
    rarest_n = np.full(X.shape[0], np.iinfo(np.int64).max)
    for u in range(X.shape[0]):
        cols = present.indices[present.indptr[u]:present.indptr[u + 1]]
        if cols.size:
            k = cols[np.argmin(column_n[cols])]
            rarest_col[u], rarest_n[u] = k, column_n[k]
    order = np.sort(rarest_n)
    percentile = np.searchsorted(order, rarest_n, side="right") / rarest_n.size
    return rarest_col, rarest_n, percentile, names


def one_fit(args, spec: dict, exp: Path, puma: str, rng: np.random.Generator):
    from pmedm_vb.solvers.base import weights_from_lambda

    inputs, _, q, map_lam, hierarchy = load(spec, exp, puma, args.alpha)
    name = fit_name(spec, puma, args.alpha)
    with np.load(exp / puma / f"hmc_{args.kind}" / f"{name}_trace.npz") as saved:
        draws = saved["xi"] if "xi" in saved.files else saved["lam"]
    m = draws.shape[2]
    x = q.whiten((draws.reshape(-1, m) - q.mean).T).T
    del draws
    post = shape(x)
    del x
    sd = post["sd"]

    groups = {"widest": np.flatnonzero(sd > args.wide_sd),
              "moderate": np.flatnonzero((sd >= 1.5) & (sd <= args.wide_sd)),
              "bulk": np.flatnonzero((sd > 0) & (sd < 1.5))}
    chosen = {g: (idx if idx.size <= args.per_group
                  else np.sort(rng.choice(idx, args.per_group, replace=False)))
              for g, idx in groups.items()}

    table = constraint_table(inputs)
    tract_of = table["geoid"].astype(str).str[:11].to_numpy()
    tilt = Tilt(inputs)
    weights = weights_from_lambda(inputs, map_lam) * inputs.N               # people
    unit_total = weights.sum(axis=0)
    rarest_col, rarest_n, rarest_pct, names = rarity(inputs)
    zones = inputs.block_groups.iloc[:, 0].to_numpy()
    unit_ids = inputs.units.iloc[:, 0].to_numpy()

    rows, details = [], []
    for group in GROUPS:
        idx = chosen[group]
        if not idx.size:
            continue
        E = np.zeros((m, idx.size))
        E[idx, np.arange(idx.size)] = sd[idx]
        move = q.gaussian_part(E)                                            # sampling coords
        move = move if hierarchy.is_trivial else hierarchy.lambda_data(move)  # data layout
        for i, j in enumerate(idx):
            dl = move[:, i]
            size = np.abs(dl)
            order = np.argsort(-size)
            top = order[:5]
            cumulative = np.cumsum(size[order] ** 2) / (size ** 2).sum()
            carry = order[: int(np.searchsorted(cumulative, 0.9)) + 1]
            eta = np.abs(tilt(dl))
            per_unit = eta.max(axis=0)
            units = np.argsort(-per_unit)[:3]
            u = units[0]
            z = int(np.argmax(eta[:, u]))
            first = table.iloc[top[0]]
            record = {
                "group": group, "coordinate": int(j), "sd": float(sd[j]),
                "kelly": float(post["kelly"][j]), "qkurt": float(post["qkurt"][j]),
                "top_level": first["level"], "top_constraint": first["constraint"],
                "top_geoid": first["geoid"], "top_published": float(first["published"]),
                "top_se": float(first["se"]), "top_move": float(dl[top[0]]),
                "top5_share": float((size[top] ** 2).sum() / (size ** 2).sum()),
                "n90_constraints": int(carry.size), "n90_tracts": int(np.unique(tract_of[carry]).size),
                "unit": unit_ids[u], "unit_eta": float(per_unit[u]), "unit_zone": zones[z],
                "unit_weight_zone": float(weights[z, u]), "unit_weight": float(unit_total[u]),
                "unit_rarest": names[rarest_col[u]] if rarest_col[u] >= 0 else "",
                "unit_rarest_n": int(rarest_n[u]), "unit_rarest_pct": float(rarest_pct[u]),
            }
            details.append(record)
            if group == "widest":
                record["_constraints"] = table.iloc[top].assign(move=dl[top])
                record["_units"] = pd.DataFrame({
                    "unit": unit_ids[units], "rarest": [names[k] if k >= 0 else "" for k in rarest_col[units]],
                    "rarest_n": rarest_n[units], "rarest_pct": rarest_pct[units],
                    "zone": [zones[int(np.argmax(eta[:, v]))] for v in units],
                    "d_eta": per_unit[units],
                    "weight_zone": [weights[int(np.argmax(eta[:, v])), v] for v in units],
                    "weight_unit": unit_total[units]})
        frame = pd.DataFrame([d for d in details if d["group"] == group])
        rows.append({
            "group": group, "available": int(groups[group].size), "n": int(idx.size),
            "sd": frame["sd"].median(), "abs_kelly": frame["kelly"].abs().median(),
            "qkurt": frame["qkurt"].median(),
            "top_tract_share": float((frame["top_level"] == "tract").mean()),
            "top_published": frame["top_published"].median(),
            "top_published_zero": float((frame["top_published"] == 0).mean()),
            "top_se": frame["top_se"].median(), "top5_share": frame["top5_share"].median(),
            "n90_constraints": frame["n90_constraints"].median(),
            "n90_tracts": frame["n90_tracts"].median(),
            "unit_eta": frame["unit_eta"].median(),
            "unit_rarest_n": frame["unit_rarest_n"].median(),
            "unit_rarest_pct": frame["unit_rarest_pct"].median(),
            "unit_weight": frame["unit_weight"].median(),
        })
    all_units = {"all_units_rarest_n_median": float(np.median(rarest_n)), "n_units": int(rarest_n.size),
                 "n_tracts": int(np.unique(tract_of).size), "vb_rank": int(q.W.shape[1])}
    return rows, details, all_units


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)

    summary, detail_rows, lines_detail = [], [], []
    for experiment in args.experiments:
        exp = args.root / experiment
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            name = fit_name(spec, puma, args.alpha)
            if not (exp / puma / f"hmc_{args.kind}" / f"{name}_trace.npz").exists():
                logger.info("%s %s: no %s trace; skipped", experiment, puma, args.kind)
                continue
            logger.info("%s %s", experiment, puma)
            rows, details, all_units = one_fit(args, spec, exp, puma, rng)
            ident = {"experiment": experiment, "puma": puma}
            summary += [{**ident, **row, **all_units} for row in rows]
            lines_detail.append(f"=== {experiment} {puma}: the {args.detail} widest directions")
            widest = sorted((d for d in details if d["group"] == "widest"), key=lambda d: -d["sd"])
            for d in widest[: args.detail]:
                lines_detail.append(f"--- coordinate {d['coordinate']}: sd {d['sd']:.1f}, "
                                    f"Kelly {d['kelly']:+.3f}, quantile ratio {d['qkurt']:.2f}; "
                                    f"top-5 constraints carry {d['top5_share']:.0%} of the move; "
                                    f"90% of it lies in {d['n90_constraints']} constraints over "
                                    f"{d['n90_tracts']} tracts")
                with pd.option_context("display.width", 200, "display.max_columns", 20):
                    lines_detail.append(d["_constraints"].to_string(index=False, float_format=lambda v: f"{v:,.3g}"))
                    lines_detail.append(d["_units"].to_string(index=False, float_format=lambda v: f"{v:,.3g}"))
            detail_rows += [{**ident, **{k: v for k, v in d.items() if not k.startswith("_")}}
                            for d in details]
            pd.DataFrame(summary).to_csv(args.out / "wide_directions.csv", index=False)
            pd.DataFrame(detail_rows).to_csv(args.out / "wide_directions_detail.csv", index=False)
    if not summary:
        raise SystemExit("no traces found")

    table = pd.DataFrame(summary)
    lines = [f"What the wide whitened directions move, alpha {args.alpha:g}, {args.kind} sds, "
             f"from {args.root}",
             f"Groups by posterior sd: widest > {args.wide_sd:g}, moderate 1.5-{args.wide_sd:g}, "
             "bulk < 1.5. Medians over each group's directions. top_*: the constraint each "
             "direction moves most; unit_*: the household whose tilt it moves most (rarest_n: "
             "units sharing its rarest category; rarest_pct: share of all units at least as rare)",
             ""]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        lines.append(table.to_string(index=False, float_format=lambda v: f"{v:,.3g}"))
    text = "\n".join(lines + [""] + lines_detail)
    (args.out / "wide_directions.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out / 'wide_directions.csv'}, wide_directions_detail.csv and wide_directions.txt")


if __name__ == "__main__":
    main()
