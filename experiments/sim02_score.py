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
- **rare groups** (``rare_groups``, and always :data:`SCORED_RARE`): per
  group, the persons a draw places in block groups where the truth has none of
  the group (posterior mean and 90% interval), with the group's true and fitted
  totals; and the group's count per block group (its ``B03002`` block group
  constraint column, persons) against the truth's over *every* block group,
  those where the truth has none included: CRPS of the draws as an ensemble
  (the formula above), 90% coverage, interval width and mean |error|. Why a
  proper score and not the misplaced persons alone: a published zero or a
  count resting on one sampled case says little about where the group lives,
  so a fit that spreads it over the plausible block groups can be honest
  rather than wrong; the CRPS rewards that spread only when it is calibrated
  (a spread that misses the truth's block groups, or too narrow a one, scores
  worse than a calibrated one), and coverage says which way it errs;
- **borrowed weight** (statewide fit): the share of ``N`` on pool B's
  records, prior and posterior (``compare_support.borrowed_fraction``); and,
  for the borrowed kind, the truth's own share on pool A's records.

**Bootstrap fit** (``boot``; ``sim02.json``'s ``bootstrap`` block,
``sim02_bootstrap.py``): for each truth in ``bootstrap.truths``, the ``R``
replicates' short HMC (fits ``bootstrap.fits``) pooled, ``draws / R`` evenly
spaced draws of each, each replicate's draws made populations with its own
``q``: so the pooled draws sample, approximately, the posterior averaged over
the Bayesian bootstrap's posterior for the prior (Rubin 1981). Scored with the
same measures as the other two fits, and compared with each over the truths
both have. A truth with a replicate's trace missing is left out and listed;
without a ``bootstrap`` block or any bootstrap trace, nothing changes.

Writes, under ``<root>/<name>/simulation/scores``:

``scores.csv``
    One row per (kind, truth_id, fit, level, set): ``set`` is ``crosstab``,
    ``constrained`` or ``heldout``, ``level`` block group, tract or puma
    (the cross-tabulation summed over block groups); the number of cells,
    mean |error| and RMSE, mean CRPS, 90% coverage and mean width, and for
    constraint and held-out cells the mean |error| and CRPS over ``se``.
``puma_crosstab.csv``, ``income_within_race.csv``, ``rare_groups.csv``, ``borrowed.csv``
    The PUMA's cross-tabulation cells (truth, posterior mean and interval),
    each race's TVD, the rare groups (with ``bg_*``: the per-block-group
    count's CRPS, coverage, width and mean |error| over ``block_groups``) and
    the borrowed weight, per kind, truth and fit. The rare groups' measures
    are also in ``per_truth.csv`` (``misplaced_<g>``, ``rare_crps_<g>``,
    ``rare_cover90_<g>``), hence in the paired comparisons and summary.txt.
``per_truth.csv``, ``paired.csv``, ``summary.txt``
    One row per (kind, truth, fit) of every measure; per kind and measure the
    two fits over the truths both have (means, the paired difference statewide
    minus local: mean, sd, min and max over truths, and the truths where the
    statewide fit is lower); and that comparison as text.
``paired_boot.csv``
    With bootstrap fits: the same per kind and measure for ``boot`` against
    ``local`` and against ``statewide`` (``comparison``; ``base`` is the
    other fit's mean, the difference is boot minus it, ``boot_lower`` the
    truths where boot is lower); in summary.txt after the kinds, with the
    replicates pooled per truth.

A fit whose HMC trace is missing is left out and listed::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log experiments/run_python.sbatch \\
        sim02_score.py --sim experiments/sim/sim02.json --root $R

**Named fits** (``--fits``): any set of fit experiments of the same truths
scored with the same measures and compared with one of them, ``--base``
(default the first)::

    sim02_score.py --sim experiments/sim/sim02.json --root $R \\
        --fits local_prior3 state_prior3 local_prior3_decor2 state_prior3_decor2 \\
        --base local_prior3

A name is ``local`` or ``statewide`` (``sim02.json``'s ``fits``), ``boot`` (the
pooled bootstrap fit, as above), a fits JSON ``sim02_fits_<name>.json`` next to
``sim02.json`` or in ``experiments/sim`` (so ``state`` is
``sim02_fits_state.json``), or a path to a fits JSON (named by its file name
less ``sim02_fits_``). Each must fit the simulation's truths at its alpha with
an HMC short run, on inputs with the source's cells (``sim02_build.py``'s: the
prior and decorrelation options change the model, not the inputs). The
tables are those above, with ``fit`` the name; ``paired.csv`` has, per kind,
comparison (``<fit> - <base>``) and measure, the base's and the fit's means
(``base``, ``fit``), the paired difference fit minus base and ``fit_lower``,
the truths where the fit is lower; summary.txt has one such table per kind and
fit, with the rare groups' per-block-group CRPS and coverage, then the
income-within-race TVDs of every fit. They are written to ``--out``, by default
``<root>/<name>/simulation/scores_<fit>+<fit>...``, so the default run's
``scores`` is left as it is; without ``--fits`` nothing changes.

**Other simulations** (``sim02_common.py``: several PUMAs, any named fits,
no pools; ``experiments/sim/sim04.json``). A simulation whose ``fits`` are not
sim02's ``local`` and ``statewide``, or that has several PUMAs, is always
scored as named fits: ``--fits`` defaults to its ``fits``, in order, and
``--base`` to the first. A ``--fits`` name is first one of the simulation's
``fits``, then ``<sim name>_fits_<name>.json`` as above. With several PUMAs
every table has a ``puma`` column (the real PUMA the truth was drawn for), and
``paired.csv`` and summary.txt compare the fits per kind for each PUMA and for
all PUMAs pooled (``puma`` ``all``: the paired differences over every truth of
the kind, each truth one PUMA's); the truths are matched by PUMA and number.
``truth_borrowed`` (``borrowed.csv``) is the truth's share of ``N`` on records
from outside the PUMA, which, without pools, may include the fits' own
out-of-PUMA records. Coverage counts a truth of zero as covered when the
interval includes zero, as for sim02.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_support import borrowed_fraction, thinned, tvd  # noqa: E402
from sim02_build import truth_source  # noqa: E402
import run_experiment as rx  # noqa: E402
from sim02_common import FITS, REPO, Sim02  # noqa: E402
from sim_common import Tabulator  # noqa: E402
from sim_score import ensemble_scores  # noqa: E402

#: The rare ``B03002`` groups whose block group counts are always scored, with
#: the simulation's ``rare_groups`` (persons; tn01's one-case tract cells were
#: in these groups).
SCORED_RARE = ("nh_nhpi", "nh_aian", "nh_other_race", "nh_asian")
#: The rare-group scores' random draws (``ensemble_scores``' rank ties) come from
#: their own generator, so the other scores are those of a run without them.
RARE_SEED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/sim02.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--draws", type=int, default=None,
                        help="draws per fit (default: the simulation's 'draws')")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <root>/<name>/simulation/scores, with --fits "
                             "scores_<fit>+<fit>...")
    parser.add_argument("--fits", nargs="+", default=None, metavar="NAME",
                        help="score these fits and compare each with --base: local, statewide, "
                             "boot, <name> for sim02_fits_<name>.json, or a fits JSON path")
    parser.add_argument("--base", default=None,
                        help="with --fits: the fit the others are compared with (default the first)")
    args = parser.parse_args()
    if args.fits is not None:
        if len(set(args.fits)) != len(args.fits):
            parser.error("--fits: a name is given twice")
        args.base = args.base or args.fits[0]
        if args.base not in args.fits:
            parser.error(f"--base {args.base}: not among --fits")
    return args


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


def rare_groups(rare: list[str]) -> list[str]:
    """The simulation's rare groups, then those of :data:`SCORED_RARE` it lacks."""
    return list(rare) + [g for g in SCORED_RARE if g not in rare]


def score_fit(truth: dict, draws: dict, names: dict, se: dict, rare: list[str],
              rng, rare_rng=None) -> tuple[list[dict], dict, list[dict], list[dict], list[dict]]:
    """One fit of one truth: the cell sets' scores, the per-truth measures, the
    PUMA cross-tabulation, the races' TVDs and the rare groups (scored with
    ``rare_rng``, default ``rng``)."""
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
    for group in rare_groups(rare):
        name = f"B03002.{group}"
        if name not in names["bg"]:
            continue
        k = names["bg"].index(name)
        zero = truth["block group"][:, k] == 0
        placed = draws["block group"][:, :, k].astype(float)
        misplaced = placed[:, zero].sum(1)
        a, b = np.quantile(misplaced, [0.05, 0.95])
        # The count in every block group, truth-zero ones included (persons).
        bg = score_set(placed, truth["block group"][:, k], None,
                       rng if rare_rng is None else rare_rng)
        groups.append({"group": group, "truth_total": float(truth["block group"][:, k].sum()),
                       "fit_total": float(placed.sum(1).mean()),
                       "truth_zero_block_groups": int(zero.sum()),
                       "misplaced_mean": float(misplaced.mean()), "misplaced_q05": float(a),
                       "misplaced_q95": float(b), "block_groups": bg["cells"],
                       "bg_crps": bg["crps"], "bg_cover90": bg["cover90"],
                       "bg_width90": bg["width90"], "bg_mae": bg["mae"]})
        measures[f"misplaced_{group}"] = float(misplaced.mean())
        measures[f"rare_crps_{group}"] = bg["crps"]
        measures[f"rare_cover90_{group}"] = bg["cover90"]
    return rows, measures, cross, within, groups


def truth_groups(per_truth: pd.DataFrame):
    """The sets of truths compared: ``(labels, rows, index)`` per kind, and with a
    ``puma`` column per kind and PUMA and per kind over all PUMAs (``puma``
    ``all``), ``index`` the columns that identify a truth in the set."""
    for kind, group in per_truth.groupby("kind", sort=False):
        if "puma" not in per_truth:
            yield {"kind": kind}, group, "truth_id"
            continue
        for puma, part in group.groupby("puma", sort=False):
            yield {"kind": kind, "puma": puma}, part, "truth_id"
        yield {"kind": kind, "puma": "all"}, group, ["puma", "truth_id"]


def paired(per_truth: pd.DataFrame, base: str = "local",
           other: str = "statewide") -> pd.DataFrame:
    """Per kind (and PUMA, :func:`truth_groups`) and measure: two fits over the
    truths both have, the difference ``other - base`` per truth."""
    keys = ["kind", "puma", "truth_id", "fit"]
    measures = [c for c in per_truth.columns if c not in keys]
    out = []
    for labels, group, index in truth_groups(per_truth):
        wide = group.pivot(index=index, columns="fit", values=measures)
        for measure in measures:
            if (measure, base) not in wide or (measure, other) not in wide:
                continue
            both = wide[[(measure, base), (measure, other)]].dropna()
            if both.empty:
                continue
            a, b = both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy()
            diff = b - a
            out.append({**labels, "measure": measure, "truths": len(diff),
                        base: a.mean(), other: b.mean(),
                        "diff_mean": diff.mean(),
                        "diff_sd": diff.std(ddof=1) if len(diff) > 1 else np.nan,
                        "diff_min": diff.min(), "diff_max": diff.max(),
                        f"{other}_lower": int((diff < 0).sum())})
    return pd.DataFrame(out)


def paired_boot(per_truth: pd.DataFrame) -> pd.DataFrame:
    """``boot`` against ``local`` and against ``statewide`` (:func:`paired`)."""
    parts = []
    for base in ("local", "statewide"):
        part = paired(per_truth, base, "boot")
        if len(part):
            part = part.rename(columns={base: "base"})
            part.insert(1, "comparison", f"boot - {base}")
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def table_lines(part: pd.DataFrame, labels: list[tuple[str, str]], first: str, second: str,
                lower: str, headers: tuple[str, str]) -> list[str]:
    """A paired comparison (:func:`paired`, indexed by measure) as summary lines:
    ``first``, ``second`` and ``lower`` are the columns of the two means and of
    the count of lower truths, ``headers`` the means' column headings."""
    lines = [f"   {'measure':<58} {headers[0]:>9} {headers[1]:>9} {'diff':>9} {'sd':>8} "
             f"{'min':>9} {'max':>9} {'lower':>6}"]
    for measure, label in labels:
        if measure not in part.index:
            continue
        r = part.loc[measure]
        lines.append(f"   {label:<58} {r[first]:>9.4g} {r[second]:>9.4g} "
                     f"{r['diff_mean']:>9.3g} {r['diff_sd']:>8.3g} {r['diff_min']:>9.3g} "
                     f"{r['diff_max']:>9.3g} {int(r[lower]):>3}/{int(r['truths'])}")
    return lines


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


def fit_json(sim: Sim02, name: str) -> Path:
    """The fits JSON a ``--fits`` name stands for (module docstring)."""
    path = Path(name)
    if path.suffix == ".json":
        if not path.exists():
            raise SystemExit(f"--fits {name}: no such file")
        return path
    for folder in (sim.path.parent, REPO / "experiments" / "sim"):
        candidate = folder / f"{sim.name}_fits_{name}.json"
        if candidate.exists():
            return candidate
    raise SystemExit(f"--fits {name}: not among {', '.join(sim.settings['fits'])} or boot, and "
                     f"no {sim.name}_fits_{name}.json in {sim.path.parent} or "
                     f"{REPO / 'experiments' / 'sim'}")


def fit_label(name: str, prefix: str = "sim02") -> str:
    """A ``--fits`` name as the tables call it: a path's file name less
    ``<prefix>_fits_`` (the simulation's name)."""
    path = Path(name)
    if path.suffix != ".json":
        return name
    stem, head = path.stem, f"{prefix}_fits_"
    return stem[len(head):] if stem.startswith(head) else stem


def named_fits(sim: Sim02, names: list[str]) -> list[str]:
    """Register the ``--fits`` names as fits of ``sim`` (``sim.fits[label]``, so
    its traces and inputs are found as the built-in fits' are) and check that
    each fits the simulation's truths; the labels, in order. The simulation's
    ``fits`` (sim02's ``local`` and ``statewide``) and ``boot`` are its own."""
    labels, problems = [], []
    for name in names:
        label = fit_label(name, sim.name)
        if label in labels:
            raise SystemExit(f"--fits: two names stand for the fit {label!r}")
        labels.append(label)
        if label in sim.settings["fits"]:
            continue
        if label == "boot":
            if sim.bootstrap is None:
                raise SystemExit(f"--fits boot: {sim.path} has no bootstrap block")
            continue
        path = fit_json(sim, name)
        spec = rx.load_experiment(path)
        expected = sim.sim_pumas()
        if spec["pumas"] != expected:
            problems.append(f"{path}: pumas must be {expected[0]} ... {expected[-1]} "
                            f"({len(expected)}), the simulation's truths")
        if [float(a) for a in spec["alphas"]] != [sim.alpha]:
            problems.append(f"{path}: alphas must be [{sim.alpha:g}]")
        if "short" not in (spec["hmc"] or {}).get("kinds", []):
            problems.append(f"{path}: the scores read HMC short (hmc.kinds)")
        epsilon = spec["model"].get("epsilon")
        if epsilon is not None and float(epsilon) != sim.epsilon:
            problems.append(f"{path}: model.epsilon must be unset or {sim.epsilon:g}")
        if str(rx.data_dir(spec)) != str(rx.data_dir(next(iter(sim.fits.values())))):
            problems.append(f"{path}: data_dir must be the simulation's")
        sim.fits[label] = spec
    if problems:
        raise SystemExit("--fits:\n  " + "\n  ".join(problems))
    return labels


def paired_named(per_truth: pd.DataFrame, base: str, others: list[str]) -> pd.DataFrame:
    """Each of ``others`` against ``base`` (:func:`paired`): per kind, comparison
    (``<fit> - <base>``) and measure, with the two means as ``base`` and ``fit``
    and the truths where the fit is lower as ``fit_lower``."""
    parts = []
    for other in others:
        part = paired(per_truth, base, other)
        if len(part):
            part = part.rename(columns={base: "base", other: "fit", f"{other}_lower": "fit_lower"})
            part.insert(1, "comparison", f"{other} - {base}")
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def named_summary(sim: Sim02, labels: list[str], base: str, pairs: pd.DataFrame,
                  frames: dict[str, pd.DataFrame], measures: list[tuple[str, str]],
                  boot_pooled: dict, boot_skipped: list[str]) -> list[str]:
    """summary.txt's comparisons for ``--fits``: per kind, every other fit
    against ``base``; then income within race for every fit, and the borrowed
    weight of every statewide-support fit."""
    lines = []
    if "boot" in labels and boot_pooled:
        lines.append("boot pools Bayesian-bootstrap replicates per truth: " + ", ".join(
            f"{puma} {len(n)} x {n[0]}" if len(set(n)) == 1
            else f"{puma} {len(n)} ({'+'.join(map(str, n))} draws)"
            for puma, n in boot_pooled.items()))
    if boot_skipped:
        lines.append(f"boot: truths left out, replicates incomplete ({len(boot_skipped)}): "
                     + "; ".join(boot_skipped))
    if lines:
        lines.append("")
    # One PUMA: per kind; several: per kind, each PUMA and then all PUMAs pooled.
    groups = [None] if sim.puma is not None else [*sim.pumas, "all"]
    for kind in sim.kinds():
        for group in groups:
            lines += kind_summary(sim, kind, group, labels, base, pairs, frames, measures)
    return lines


def kind_summary(sim: Sim02, kind: str, group: str | None, labels: list[str], base: str,
                 pairs: pd.DataFrame, frames: dict[str, pd.DataFrame],
                 measures: list[tuple[str, str]]) -> list[str]:
    """summary.txt's lines for one kind's truths, of one PUMA (``group``), of all
    PUMAs (``"all"``) or of the one PUMA of a single-PUMA simulation (``None``):
    every other fit against ``base``, income within race for every fit and the
    out-of-PUMA weight of every statewide-support fit."""
    def rows(frame: pd.DataFrame, pooled: bool = False) -> pd.DataFrame:
        """The kind's rows of the group; ``pooled``: the frame has rows of its
        own for all PUMAs (``puma`` ``all``, as ``paired.csv``)."""
        if not len(frame):
            return frame
        keep = frame["kind"] == kind
        if group == "all" and pooled:
            keep &= frame["puma"] == "all"
        elif group not in (None, "all"):
            keep &= frame["puma"] == group
        return frame[keep]

    where = {None: "", "all": ", all PUMAs"}.get(group, f", PUMA {group}")
    lines, chosen = [], rows(pairs, pooled=True)
    for other in labels:
        if other == base:
            continue
        part = (chosen[chosen["comparison"] == f"{other} - {base}"] if len(chosen) else chosen)
        if part.empty:
            lines += [f"== {kind} truths{where}, {other} vs {base}: no truth with both fits", ""]
            continue
        part = part.set_index("measure")
        lines.append(f"== {kind} truths{where}, {other} vs {base} ({int(part['truths'].max())} "
                     "with both fits; diff = fit - base)")
        lines += table_lines(part, measures, "base", "fit", "fit_lower", ("base", "fit"))
        lines.append("")
    within = rows(frames["within"])
    if len(within):
        table = within.pivot_table(index="race", columns="fit", values="tvd", aggfunc="mean")
        table = table[[f for f in labels if f in table]]
        truth = [c for c in ("puma", "truth_id") if c in within]
        persons = (within.drop_duplicates([*truth, "race"]).groupby("race")
                   ["truth_persons"].mean().rename("truth persons"))
        lines.append(f"   {kind} truths{where}, income within race, TVD from the truth (mean "
                     "over truths):")
        lines += ["   " + line for line in table.join(persons).round(4).to_string().splitlines()]
    b = rows(frames["borrowed"])
    for fit in labels:
        spec = sim.fits.get(fit)
        if spec is None or spec["model"].get("epsilon") is None:
            continue
        part = b[b["fit"] == fit]
        if len(part):
            lines.append(f"   {sim.pool_label} share of N, {fit}: prior {part['prior'].mean():.4f}, "
                         f"posterior {part['mean'].mean():.4f} (truth's own share on records "
                         f"from outside the PUMA: {part['truth_borrowed'].mean():.4f})")
    lines.append("")
    return lines


def load_fit(path: Path, trace: Path, draws: int, names: dict):
    """One fit's inputs at ``path`` and ``draws`` draws of its ``trace``: the
    cells, each draw's borrowed share, the prior's borrowed share, the cells'
    ``se`` and the number of rows."""
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    inputs, fit_held = PMEDMInputs.load(path), HeldOut.load(path)
    with np.load(trace) as saved:
        lam = thinned(saved["lam"], draws)
    cells, borrowed = fit_cells(inputs, fit_held, lam)
    fit_tab = Tabulator(inputs, fit_held)
    _, fit_cross = crosstab(inputs)
    if fit_cross != names["crosstab"] or list(inputs.bg_constraints) != names["bg"]:
        raise SystemExit(f"{path}: its cells are not the source's")
    se = {level: np.sqrt(fit_tab.v[level]) for level in ("block group", "tract")}
    prior = float(inputs.units["weight"].to_numpy(float) @ borrowed_fraction(inputs) / inputs.N)
    return cells, borrowed, prior, se, inputs.n_units


def record_fit(tables: dict, base: dict, truth: dict, cells: dict, borrowed: np.ndarray,
               prior: float, truth_borrowed: float, names: dict, se: dict, rare: list[str],
               rng, rare_rng=None) -> dict:
    """Score one fit of one truth into ``tables``; its per-truth measures."""
    rows, measures, cross, within, groups = score_fit(truth, cells, names, se, rare, rng,
                                                      rare_rng)
    tables["scores"] += [{**base, **row} for row in rows]
    tables["cross"] += [{**base, **row} for row in cross]
    tables["within"] += [{**base, **row} for row in within]
    tables["rare"] += [{**base, **row} for row in groups]
    q05, q95 = np.quantile(borrowed, [0.05, 0.95])
    tables["borrowed"].append({**base, "prior": prior, "mean": float(borrowed.mean()),
                               "q05": float(q05), "q95": float(q95),
                               "truth_borrowed": truth_borrowed})
    tables["per_truth"].append({**base, **measures, "borrowed_posterior": float(borrowed.mean())})
    return measures


def boot_traces(sim: Sim02, puma: str) -> tuple[list[tuple[str, Path]], list[str]]:
    """The truth's replicates with their traces, and the replicates without one."""
    reps = [(rep, sim.fit_trace("boot", rep)) for rep in sim.replicate_pumas(puma)]
    return reps, [rep for rep, trace in reps if not trace.exists()]


def boot_summary(sim: Sim02, pairs: pd.DataFrame, within: pd.DataFrame,
                 pooled: dict[str, list[int]], skipped: list[str], per_replicate: int,
                 labels: list[tuple[str, str]]) -> list[str]:
    """summary.txt's bootstrap section: boot against local and statewide per kind,
    income within race over the bootstrap truths, and what was pooled."""
    name, R = sim.fits["boot"]["name"], sim.replicates()
    if not pooled:
        lines = [f"== bootstrap fits ({name}): none scored"]
        if skipped:
            lines.append(f"   truths left out, replicates incomplete ({len(skipped)}): "
                         + "; ".join(skipped))
        else:
            lines.append("   no replicate's HMC trace exists yet")
        return lines + [""]
    lines = [f"== bootstrap fits ({name}): boot pools {R} Bayesian-bootstrap replicates of the "
             f"local fit's prior per truth, {per_replicate} draws each",
             "   diff = boot - local (or boot - statewide), per truth; 'lower' counts truths "
             "where boot is lower",
             "   replicates pooled: " + ", ".join(
                 f"{puma} {len(n)} x {n[0]}" if len(set(n)) == 1
                 else f"{puma} {len(n)} ({'+'.join(map(str, n))} draws)"
                 for puma, n in pooled.items())]
    if skipped:
        lines.append(f"   truths left out, replicates incomplete ({len(skipped)}): "
                     + "; ".join(skipped))
    lines.append("")
    for kind in sim.kinds():
        for base, heading in (("local", "local"), ("statewide", "state")):
            part = pairs[(pairs["kind"] == kind) & (pairs["comparison"] == f"boot - {base}")]
            if part.empty:
                continue
            part = part.set_index("measure")
            lines.append(f"== {kind} truths, boot vs {base} ({int(part['truths'].max())} "
                         "with both fits)")
            lines += table_lines(part, labels, "base", "boot", "boot_lower", (heading, "boot"))
            lines.append("")
        ids = [sim.truth_index(puma)[1] for puma in pooled
               if sim.truth_index(puma)[0] == sim.kinds().index(kind)]
        rows = within[(within["kind"] == kind) & within["truth_id"].isin(ids)]
        if len(rows):
            table = rows.pivot_table(index="race", columns="fit", values="tvd", aggfunc="mean")
            table = table[[f for f in ("local", "statewide", "boot") if f in table]]
            lines.append(f"   {kind} truths, income within race, TVD from the truth (mean over "
                         f"the {len(ids)} bootstrap truths):")
            lines += ["   " + line for line in table.round(4).to_string().splitlines()]
            lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    sim = Sim02(args.sim, args.root)
    from pmedm_vb.assemble.heldout import HeldOut

    if args.fits is None and (set(sim.settings["fits"]) != set(FITS) or sim.puma is None):
        # Not sim02's two fits of one PUMA: always the named-fits comparison.
        args.fits = list(sim.settings["fits"])
        args.base = args.base or args.fits[0]
    elif args.base is not None and args.fits is None:
        raise SystemExit("--base needs --fits")

    # The fits scored, in order: the two built-in ones and (once started) boot,
    # or the --fits names.
    labels = named_fits(sim, args.fits) if args.fits is not None else None
    draws_wanted = args.draws or sim.draws
    if labels is None:
        out = args.out or sim.folder / "scores"
    else:
        out = args.out or sim.folder / ("scores_" + "+".join(labels))
    out.mkdir(parents=True, exist_ok=True)
    rare = list(sim.settings.get("rare_groups", []))
    pool_names = {puma: HeldOut.load(sim.fit_pool_inputs(puma)).names for puma in sim.pumas}
    rng, rare_rng = np.random.default_rng(0), np.random.default_rng(RARE_SEED)
    tables = {k: [] for k in ("scores", "per_truth", "cross", "within", "rare", "borrowed")}
    missing = []
    # The bootstrap fit is scored once some replicate has a trace (or when named);
    # then each truth with a replicate missing is left out and listed.
    R = sim.replicates()
    if labels is None:
        started = any(len(boot_traces(sim, t)[1]) < R for t in sim.boot_truths())
        boot = sim.boot_truths() if started else []
        fit_list = [*FITS, *(["boot"] if boot else [])]
    else:
        boot = sim.boot_truths() if "boot" in labels else []
        fit_list = labels
    per_replicate = draws_wanted // R if boot else 0
    if boot and per_replicate < 1:
        raise SystemExit(f"--draws {draws_wanted}: fewer than one per bootstrap replicate ({R})")
    boot_pooled, boot_skipped = {}, []
    sources = [(kind, area) for kind in sim.kinds() for area in sim.pumas]
    for kind, area in sources:
        source, held, _ = truth_source(sim, kind, pool_names[area], area)
        src_tab, (src_cross, cross_names) = Tabulator(source, held), crosstab(source)
        src_frac = borrowed_fraction(source)
        names = {"bg": list(source.bg_constraints), "tract": list(source.tract_constraints),
                 "crosstab": cross_names}
        # The rows' identity: the real PUMA too when there are several.
        where = {"kind": kind} if sim.puma is not None else {"kind": kind, "puma": area}
        for i, puma in sim.truths(kind, area):
            with np.load(sim.truth_file(puma)) as saved:
                T = saved["T"].astype(float)
            truth = tabulate(src_tab, src_cross, T)
            truth_borrowed = float(T.sum(0) @ src_frac / T.sum())
            for fit in fit_list:
                if fit == "boot":
                    if puma not in boot:
                        continue
                    reps, absent = boot_traces(sim, puma)
                    if absent:
                        boot_skipped.append(f"{puma} ({R - len(absent)} of {R}; missing "
                                            f"{', '.join(absent)})")
                        continue
                    # Equal draws from each replicate, each made populations with its own q.
                    parts = [load_fit(sim.fit_inputs("boot", rep), trace, per_replicate, names)
                             for rep, trace in reps]
                    cells = {k: np.concatenate([part[0][k] for part in parts])
                             for k in parts[0][0]}
                    borrowed = np.concatenate([part[1] for part in parts])
                    prior = float(np.mean([part[2] for part in parts]))
                    measures = record_fit(tables, {**where, "truth_id": i, "fit": "boot"},
                                          truth, cells, borrowed, prior, truth_borrowed, names,
                                          parts[0][3], rare, rng, rare_rng)
                    boot_pooled[puma] = [part[1].size for part in parts]
                    print(f"{puma} boot: {borrowed.size} draws "
                          f"({' + '.join(map(str, boot_pooled[puma]))} from {R} replicates); "
                          f"race x income CRPS {measures['block_group_crosstab_crps']:.3g}, TVD "
                          f"{measures['tvd_mean']:.3g}", flush=True)
                    continue
                trace = sim.fit_trace(fit, puma)
                if not trace.exists():
                    missing.append(f"{puma} {fit}")
                    continue
                cells, borrowed, prior, se, rows = load_fit(sim.fit_inputs(fit, puma), trace,
                                                            draws_wanted, names)
                measures = record_fit(tables, {**where, "truth_id": i, "fit": fit}, truth,
                                      cells, borrowed, prior, truth_borrowed, names, se, rare,
                                      rng, rare_rng)
                print(f"{puma} {fit}: {borrowed.size} draws, {rows:,} rows; race x "
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
    rare_measures = [(f"misplaced_{g}", f"{g}: persons in block groups where the truth has none")
                     for g in rare_groups(rare)]
    rare_measures += [(f"rare_{m}_{g}", f"{g} per block group (all): {label}")
                      for g in rare_groups(rare)
                      for m, label in (("crps", "CRPS (persons)"), ("cover90", "90% coverage"))]
    if labels is not None:
        pairs = paired_named(frames["per_truth"], args.base,
                             [f for f in labels if f != args.base])
        pairs.to_csv(out / "paired.csv", index=False)
        areas = (f"PUMA {sim.puma}" if sim.puma is not None
                 else f"PUMAs {', '.join(sim.pumas)}")
        lines = [f"{sim.name}: {areas}, alpha {sim.alpha:g}, epsilon {sim.epsilon:g}; "
                 f"{draws_wanted} HMC short draws per fit",
                 "fits: " + "; ".join(
                     f"{f} ({sim.fits[f]['name']}, level {rx.level(sim.fits[f])})"
                     if f in sim.fits else f for f in labels),
                 f"each compared with {args.base}: diff = fit - base, per truth; mean, sd, min and "
                 "max over truths; 'lower' counts truths where the fit is lower",
                 "(lower is better for errors, CRPS and TVD; coverage should be near 0.90)", ""]
        if missing:
            lines += [f"missing fits, left out ({len(missing)}): {', '.join(missing)}", ""]
        lines += named_summary(sim, labels, args.base, pairs, frames, SUMMARY + rare_measures + [
            ("borrowed_posterior", f"share of N on {sim.pool_label} records (posterior mean)")],
            boot_pooled, boot_skipped)
        text = "\n".join(lines) + "\n"
        (out / "summary.txt").write_text(text)
        (out / "fits.json").write_text(json.dumps(
            {"fits": labels, "base": args.base,
             "experiments": {f: sim.fits[f]["name"] for f in labels if f in sim.fits}},
            indent=2) + "\n")
        print(text)
        print(f"wrote {out}")
        return
    pairs = paired(frames["per_truth"])
    pairs.to_csv(out / "paired.csv", index=False)
    boot_pairs = paired_boot(frames["per_truth"]) if boot_pooled else pd.DataFrame()
    if boot_pooled:
        boot_pairs.to_csv(out / "paired_boot.csv", index=False)

    lines = [f"{sim.name}: PUMA {sim.puma}, alpha {sim.alpha:g}, epsilon {sim.epsilon:g}; "
             f"{draws_wanted} HMC short draws per fit",
             "local fit: the simulated PUMS alone; statewide fit: the PUMS plus pool B",
             "diff = statewide - local, per truth; mean, sd, min and max over truths; "
             "'lower' counts truths where the statewide value is lower",
             "(lower is better for errors, CRPS and TVD; coverage should be near 0.90)", ""]
    if missing:
        lines += [f"missing fits, left out ({len(missing)}): {', '.join(missing)}", ""]
    for kind in sim.kinds():
        part = pairs[pairs["kind"] == kind].set_index("measure")
        if part.empty:
            lines += [f"== {kind} truths: no truth with both fits", ""]
            continue
        lines.append(f"== {kind} truths ({int(part['truths'].max())} with both fits)")
        lines += table_lines(part, SUMMARY + rare_measures + [
            ("borrowed_posterior", "share of N on pool B records (posterior mean)")],
            "local", "statewide", "statewide_lower", ("local", "state"))
        within = frames["within"][(frames["within"]["kind"] == kind)
                                  & (frames["within"]["fit"] != "boot")]
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
    if sim.bootstrap is not None:
        lines += boot_summary(sim, boot_pairs, frames["within"], boot_pooled, boot_skipped,
                              per_replicate, SUMMARY + rare_measures)
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
