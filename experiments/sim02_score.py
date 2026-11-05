"""Score the borrowing simulation's fits against its truths (``experiments/sim/sim02.json``).

For each truth kind (``local``, ``borrowed``), truth and fit (``local``: the
simulated PUMS alone; ``statewide``: the PUMS plus pool B), ``draws`` evenly
spaced draws of the fit's short HMC (chains and iterations flattened,
``compare_support.thinned``) are made populations ``W = N p(lambda)`` and set
against the truth's ``T``, cell by cell:

- **cells**: the race x income cross-tabulation (persons of each ``B03002``
  category in households of each ``B19001`` band, ``pmedm_vb.compare.outcome_matrix``;
  no table constrains it) per block group and for the PUMA; every constraint
  cell (block group and tract); and the truth-based held-out cells
  (``sim02_build.py``: the held-out lines' loadings of the truth's units).
  Per cell: the error of the posterior mean, whether the 90% interval
  (5% to 95% quantiles of the draws) covers the truth, its width, and the CRPS
  of the draws as an ensemble (``sim_score.ensemble_scores``: ``mean|X - T| -
  mean|X - X'| / 2``; lower is better). Constraint and held-out cells also in
  units of their ``se``, the square root of the fit's (``c``-scaled) variance;
  the cross-tabulation has none, so it is in persons;
- **income within race**: for each race, the PUMA's persons over income
  bands as shares, the posterior mean of each draw's shares against the
  truth's: the total variation distance (``compare_support.tvd``). The mean is
  over the draws with persons of the race in households (``draws_defined``,
  their share); a fit with none in any draw scores 1, having no income
  distribution for the race at all, and a race the truth has none of is left
  out (NaN);
- **rare groups** (``rare_groups``): per group, the persons a draw places in
  block groups where the truth has none of the group (posterior mean and 90%
  interval), with the group's true and fitted totals;
- **borrowed weight** (statewide fit): the share of ``N`` on pool B's
  records, prior and posterior (``compare_support.borrowed_fraction``); and,
  for the borrowed kind, the truth's own share on pool A's records.

Writes, under ``<root>/<name>/simulation/scores``:

``scores.csv``
    One row per (kind, truth_id, fit, level, set): ``set`` is ``crosstab``,
    ``constrained`` or ``heldout``, ``level`` block group, tract or puma
    (the cross-tabulation summed over block groups); the number of cells,
    mean |error| and RMSE, mean CRPS, 90% coverage and mean width, and for
    constraint and held-out cells the mean |error| and CRPS over ``se``.
``puma_crosstab.csv``, ``income_within_race.csv``, ``rare_groups.csv``, ``borrowed.csv``
    The PUMA's cross-tabulation cells (truth, posterior mean and interval),
    each race's TVD, the rare groups and the borrowed weight, per kind,
    truth and fit.
``per_truth.csv``, ``paired.csv``, ``summary.txt``
    One row per (kind, truth, fit) of every measure; per kind and measure the
    two fits over the truths both have (means, the paired difference statewide
    minus local: mean, sd, min and max over truths, and the truths where the
    statewide fit is lower); and that comparison as text.

A fit whose HMC trace is missing is left out and listed::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log experiments/run_python.sbatch \\
        sim02_score.py --sim experiments/sim/sim02.json --root $R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_support import borrowed_fraction, thinned, tvd  # noqa: E402
from sim02_build import truth_source  # noqa: E402
from sim02_common import FITS, Sim02  # noqa: E402
from sim_common import Tabulator  # noqa: E402
from sim_score import ensemble_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/sim02.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--draws", type=int, default=None,
                        help="draws per fit (default: the simulation's 'draws')")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <root>/<name>/simulation/scores")
    return parser.parse_args()


def crosstab(inputs) -> tuple[sp.csr_matrix, list[str]]:
    """The race x income columns of :func:`pmedm_vb.compare.outcome_matrix`."""
    from pmedm_vb import compare

    X, meta = compare.outcome_matrix(inputs)
    keep = np.flatnonzero((meta["kind"] == "crosstab").to_numpy())
    if not keep.size:
        raise SystemExit(f"PUMA {inputs.puma}: no B03002 x B19001 columns at block group")
    return sp.csr_matrix(sp.csc_matrix(X)[:, keep]), list(meta["outcome"].iloc[keep])


def tabulate(tab: Tabulator, cross: sp.csr_matrix, W: np.ndarray) -> dict[str, np.ndarray]:
    """A population's cells: block group and tract lines (constraints, then
    held-out) and the block group cross-tabulation."""
    out = tab(W)
    out["crosstab"] = np.asarray((cross.T @ np.asarray(W, float).T).T)
    return out


def fit_cells(inputs, held, lam: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """``(draws, areas, k)`` cells of each column of ``lam`` and each draw's share
    of ``N`` on records from outside the PUMA."""
    from pmedm_vb.solvers.base import ConstraintOperator

    tab, (cross, _) = Tabulator(inputs, held), crosstab(inputs)
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    frac = borrowed_fraction(inputs)
    parts, borrowed = [], np.empty(lam.shape[1])
    for d in range(lam.shape[1]):
        logits = log_q - op.adjoint(lam[:, d])
        W = inputs.N * np.exp(logits - logsumexp(logits))
        parts.append({k: v.astype(np.float32) for k, v in tabulate(tab, cross, W).items()})
        borrowed[d] = float((W @ frac).sum() / inputs.N)
    return {k: np.stack([p[k] for p in parts]) for k in parts[0]}, borrowed


def score_set(X: np.ndarray, T: np.ndarray, se: np.ndarray | None, rng) -> dict:
    """``X`` (draws, cells) against ``T`` (cells,): the summary of one cell set."""
    X, T = np.asarray(X, float).T, np.asarray(T, float).ravel()
    s = ensemble_scores(X, T, rng)
    q05, q95 = np.quantile(X, [0.05, 0.95], axis=1)
    error = s["mean"] - T
    out = {"cells": int(T.size), "mae": float(np.abs(error).mean()),
           "rmse": float(np.sqrt(np.square(error).mean())), "crps": float(s["crps"].mean()),
           "cover90": float(((q05 <= T) & (T <= q95)).mean()), "width90": float((q95 - q05).mean())}
    if se is not None:
        se = np.asarray(se, float).ravel()
        ok = se > 0
        out["mae_se"] = float((np.abs(error[ok]) / se[ok]).mean())
        out["crps_se"] = float((s["crps"][ok] / se[ok]).mean())
    return out


def shares(x: np.ndarray) -> np.ndarray:
    total = x.sum(-1, keepdims=True)
    return np.divide(x, total, out=np.full(x.shape, np.nan), where=total > 0)


def score_fit(truth: dict, draws: dict, names: dict, se: dict, rare: list[str],
              rng) -> tuple[list[dict], dict, list[dict], list[dict], list[dict]]:
    """One fit of one truth: the cell sets' scores, the per-truth measures, the
    PUMA cross-tabulation, the races' TVDs and the rare groups."""
    n_cons = {"block group": len(names["bg"]), "tract": len(names["tract"])}
    rows, measures = [], {}
    sets = [("block group", "crosstab", draws["crosstab"], truth["crosstab"], None),
            ("puma", "crosstab", draws["crosstab"].sum(1), truth["crosstab"].sum(0), None)]
    for level in ("block group", "tract"):
        k = n_cons[level]
        for cells, part in (("constrained", slice(0, k)), ("heldout", slice(k, None))):
            sets.append((level, cells, draws[level][:, :, part], truth[level][:, part],
                         se[level][:, part]))
    for level, cells, X, T, s in sets:
        if np.size(T) == 0:
            continue
        result = score_set(X.reshape(X.shape[0], -1), T, s, rng)
        rows.append({"level": level, "set": cells, **result})
        for key in ("mae", "crps", "cover90", "mae_se", "crps_se"):
            if key in result:
                measures[f"{level.replace(' ', '_')}_{cells}_{key}"] = result[key]

    puma = draws["crosstab"].sum(1).astype(float)
    true_puma = truth["crosstab"].sum(0)
    q05, q95 = np.quantile(puma, [0.05, 0.95], axis=0)
    cross = [{"outcome": name, "truth": float(t), "mean": float(m), "q05": float(a),
              "q95": float(b)} for name, t, m, a, b in
             zip(names["crosstab"], true_puma, puma.mean(0), q05, q95)]
    races = pd.Series([name.split(" x ")[0] for name in names["crosstab"]])
    within, tvds = [], {}
    for race, idx in races.groupby(races).groups.items():
        cols = np.asarray(list(idx))
        per_draw, true_shares = shares(puma[:, cols]), shares(true_puma[cols])
        defined = ~np.isnan(per_draw).any(axis=1)
        if np.isnan(true_shares).any():
            value = float("nan")  # the truth has none of the race in households
        elif not defined.any():
            value = 1.0  # no draw has the race in households: no income distribution at all
        else:
            value = tvd(per_draw[defined].mean(axis=0), true_shares)
        tvds[race] = value
        within.append({"race": race.split(".", 1)[1], "truth_persons": float(true_puma[cols].sum()),
                       "fit_persons": float(puma[:, cols].sum(1).mean()),
                       "draws_defined": float(defined.mean()), "tvd": value})
    measures["tvd_mean"] = float(np.nanmean(list(tvds.values())))
    for race, value in tvds.items():
        measures[f"tvd_{race.split('.', 1)[1]}"] = value

    groups = []
    for group in rare:
        name = f"B03002.{group}"
        if name not in names["bg"]:
            continue
        k = names["bg"].index(name)
        zero = truth["block group"][:, k] == 0
        placed = draws["block group"][:, :, k].astype(float)
        misplaced = placed[:, zero].sum(1)
        a, b = np.quantile(misplaced, [0.05, 0.95])
        groups.append({"group": group, "truth_total": float(truth["block group"][:, k].sum()),
                       "fit_total": float(placed.sum(1).mean()),
                       "truth_zero_block_groups": int(zero.sum()),
                       "misplaced_mean": float(misplaced.mean()), "misplaced_q05": float(a),
                       "misplaced_q95": float(b)})
        measures[f"misplaced_{group}"] = float(misplaced.mean())
    return rows, measures, cross, within, groups


def paired(per_truth: pd.DataFrame) -> pd.DataFrame:
    """Per kind and measure: the two fits over the truths both have."""
    keys = ["kind", "truth_id", "fit"]
    measures = [c for c in per_truth.columns if c not in keys]
    out = []
    for kind, group in per_truth.groupby("kind", sort=False):
        wide = group.pivot(index="truth_id", columns="fit", values=measures)
        for measure in measures:
            if (measure, "local") not in wide or (measure, "statewide") not in wide:
                continue
            both = wide[[(measure, "local"), (measure, "statewide")]].dropna()
            if both.empty:
                continue
            local, state = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
            diff = state - local
            out.append({"kind": kind, "measure": measure, "truths": len(diff),
                        "local": local.mean(), "statewide": state.mean(),
                        "diff_mean": diff.mean(),
                        "diff_sd": diff.std(ddof=1) if len(diff) > 1 else np.nan,
                        "diff_min": diff.min(), "diff_max": diff.max(),
                        "statewide_lower": int((diff < 0).sum())})
    return pd.DataFrame(out)


#: The summary's measures, in order, with how to read them.
SUMMARY = [
    ("block_group_crosstab_mae", "race x income per block group: mean |error| (persons)"),
    ("block_group_crosstab_crps", "race x income per block group: CRPS (persons)"),
    ("block_group_crosstab_cover90", "race x income per block group: 90% coverage"),
    ("puma_crosstab_mae", "race x income, PUMA: mean |error| (persons)"),
    ("puma_crosstab_crps", "race x income, PUMA: CRPS (persons)"),
    ("puma_crosstab_cover90", "race x income, PUMA: 90% coverage"),
    ("tvd_mean", "income within race: TVD, mean over races"),
    ("block_group_constrained_crps_se", "constraint cells, block group: CRPS / se"),
    ("block_group_constrained_cover90", "constraint cells, block group: 90% coverage"),
    ("tract_constrained_crps_se", "constraint cells, tract: CRPS / se"),
    ("block_group_heldout_crps_se", "held-out cells, block group: CRPS / se"),
    ("block_group_heldout_cover90", "held-out cells, block group: 90% coverage"),
    ("tract_heldout_crps_se", "held-out cells, tract: CRPS / se"),
    ("tract_heldout_cover90", "held-out cells, tract: 90% coverage"),
]


def main() -> None:
    args = parse_args()
    sim = Sim02(args.sim, args.root)
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    draws_wanted = args.draws or sim.draws
    out = args.out or sim.folder / "scores"
    out.mkdir(parents=True, exist_ok=True)
    rare = list(sim.settings.get("rare_groups", []))
    pool_names = HeldOut.load(sim.pool_inputs("B")).names
    rng = np.random.default_rng(0)
    tables = {k: [] for k in ("scores", "per_truth", "cross", "within", "rare", "borrowed")}
    missing = []
    for kind in sim.kinds():
        source, held, _ = truth_source(sim, kind, pool_names)
        src_tab, (src_cross, cross_names) = Tabulator(source, held), crosstab(source)
        src_frac = borrowed_fraction(source)
        names = {"bg": list(source.bg_constraints), "tract": list(source.tract_constraints),
                 "crosstab": cross_names}
        for i in range(1, sim.count(kind) + 1):
            puma = sim.sim_puma(kind, i)
            with np.load(sim.truth_file(puma)) as saved:
                T = saved["T"].astype(float)
            truth = tabulate(src_tab, src_cross, T)
            truth_borrowed = float(T.sum(0) @ src_frac / T.sum())
            for fit in FITS:
                trace = sim.fit_trace(fit, puma)
                if not trace.exists():
                    missing.append(f"{puma} {fit}")
                    continue
                path = sim.fit_inputs(fit, puma)
                inputs, fit_held = PMEDMInputs.load(path), HeldOut.load(path)
                with np.load(trace) as saved:
                    lam = thinned(saved["lam"], draws_wanted)
                cells, borrowed = fit_cells(inputs, fit_held, lam)
                fit_tab = Tabulator(inputs, fit_held)
                _, fit_cross = crosstab(inputs)
                if fit_cross != cross_names or list(inputs.bg_constraints) != names["bg"]:
                    raise SystemExit(f"{path}: its cells are not the source's")
                se = {level: np.sqrt(fit_tab.v[level]) for level in ("block group", "tract")}
                rows, measures, cross, within, groups = score_fit(truth, cells, names, se, rare,
                                                                  rng)
                base = {"kind": kind, "truth_id": i, "fit": fit}
                tables["scores"] += [{**base, **row} for row in rows]
                tables["cross"] += [{**base, **row} for row in cross]
                tables["within"] += [{**base, **row} for row in within]
                tables["rare"] += [{**base, **row} for row in groups]
                prior = float(inputs.units["weight"].to_numpy(float) @ borrowed_fraction(inputs)
                              / inputs.N)
                q05, q95 = np.quantile(borrowed, [0.05, 0.95])
                tables["borrowed"].append({**base, "prior": prior, "mean": float(borrowed.mean()),
                                           "q05": float(q05), "q95": float(q95),
                                           "truth_borrowed": truth_borrowed})
                tables["per_truth"].append({**base, **measures,
                                            "borrowed_posterior": float(borrowed.mean())})
                print(f"{puma} {fit}: {lam.shape[1]} draws, {inputs.n_units:,} rows; race x "
                      f"income CRPS {measures['block_group_crosstab_crps']:.3g}, TVD "
                      f"{measures['tvd_mean']:.3g}", flush=True)
    if not tables["per_truth"]:
        raise SystemExit(f"no fit to score; missing: {', '.join(missing)}")
    files = {"scores": "scores.csv", "cross": "puma_crosstab.csv",
             "within": "income_within_race.csv", "rare": "rare_groups.csv",
             "borrowed": "borrowed.csv", "per_truth": "per_truth.csv"}
    frames = {key: pd.DataFrame(rows) for key, rows in tables.items()}
    for key, name in files.items():
        frames[key].to_csv(out / name, index=False)
    pairs = paired(frames["per_truth"])
    pairs.to_csv(out / "paired.csv", index=False)

    lines = [f"{sim.name}: PUMA {sim.puma}, alpha {sim.alpha:g}, epsilon {sim.epsilon:g}; "
             f"{draws_wanted} HMC short draws per fit",
             "local fit: the simulated PUMS alone; statewide fit: the PUMS plus pool B",
             "diff = statewide - local, per truth; mean, sd, min and max over truths; "
             "'lower' counts truths where the statewide value is lower",
             "(lower is better for errors, CRPS and TVD; coverage should be near 0.90)", ""]
    if missing:
        lines += [f"missing fits, left out ({len(missing)}): {', '.join(missing)}", ""]
    rare_measures = [(f"misplaced_{g}", f"{g}: persons in block groups where the truth has none")
                     for g in rare]
    for kind in sim.kinds():
        part = pairs[pairs["kind"] == kind].set_index("measure")
        if part.empty:
            lines += [f"== {kind} truths: no truth with both fits", ""]
            continue
        lines.append(f"== {kind} truths ({int(part['truths'].max())} with both fits)")
        lines.append(f"   {'measure':<58} {'local':>9} {'state':>9} {'diff':>9} {'sd':>8} "
                     f"{'min':>9} {'max':>9} {'lower':>6}")
        for measure, label in SUMMARY + rare_measures + [
                ("borrowed_posterior", "share of N on pool B records (posterior mean)")]:
            if measure not in part.index:
                continue
            r = part.loc[measure]
            lines.append(f"   {label:<58} {r['local']:>9.4g} {r['statewide']:>9.4g} "
                         f"{r['diff_mean']:>9.3g} {r['diff_sd']:>8.3g} {r['diff_min']:>9.3g} "
                         f"{r['diff_max']:>9.3g} {int(r['statewide_lower']):>3}/{int(r['truths'])}")
        within = frames["within"][frames["within"]["kind"] == kind]
        table = within.pivot_table(index="race", columns="fit", values="tvd", aggfunc="mean")
        persons = within.groupby("race")["truth_persons"].mean().rename("truth persons")
        lines += ["   income within race, TVD from the truth (mean over truths):"]
        lines += ["   " + line for line in table.join(persons).round(4).to_string().splitlines()]
        b = frames["borrowed"][(frames["borrowed"]["kind"] == kind)
                               & (frames["borrowed"]["fit"] == "statewide")]
        if len(b):
            lines.append(f"   pool B share of N, statewide fit: prior {b['prior'].mean():.4f}, "
                         f"posterior {b['mean'].mean():.4f} (truth's own share on records from "
                         f"outside the PUMA: {b['truth_borrowed'].mean():.4f})")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
