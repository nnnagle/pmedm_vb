"""Compare model variants PUMA by PUMA: HMC cost and health, convergence, extreme multipliers, held-out scores.

Given several experiment folders (``run_experiment.py``; e.g. ``tn05_long_base``,
``tn05_long_prior3``, ``tn05_long_tweedie18``, ``tn05_long_state_e0.01`` and
``tn01_gpu``) and PUMAs, tabulates per experiment x PUMA, from the HMC run
(``hmc_<kind>``; by default the experiment's ``ref`` run if it has one, else
its ``short`` run):

- **cost** (the trace): the sampler's wall time ``seconds`` and its warmup
  part ``warmup_seconds`` (summed over the jobs a resumed run took), the
  iterations done of those planned, peak GPU memory, and the learned diagonal
  (``scale``: median and max) with ``--adapt-diag``;
- **sampler and convergence** (the ``_report.txt``, parsed): step size,
  leapfrog steps per iteration (median, max), divergent transitions, mean
  acceptance; over the kept ``lambda`` coordinates R-hat (median, p90, p99,
  max), the number over 1.01 and 1.1, the smallest bulk and tail ESS; and the
  largest cell's share of ``N`` (median, p90, p99, max). A trace without a
  report (a run still going) gets the trace's own sampler figures and no
  report figures;
- **extreme multipliers**: the MAP's (``map/<name>.npz``, ``lam``, the
  multipliers the data see) largest ``|lambda|``, its cell and how many exceed
  10; and the PUMA's worst-R-hat coordinates (``--worst`` per experiment,
  R-hat computed here from the trace's ``lam`` as the report does): for every
  experiment, those coordinates' posterior median and 90% interval, R-hat,
  bulk ESS, MAP value, published estimate and SE and ``k_hat =
  (estimate / SE)^2``, about the number of sampled cases behind the cell
  (``cell_cases.py``). Coordinates are matched by (level, area, constraint),
  so a statewide support's runs line up with the PUMA-only ones;
- **held-out tables**, where the experiment's inputs folder has them
  (``run_map.py heldout``, with ``--epsilon`` for a statewide support):
  ``--draws`` evenly spaced draws made populations and scored against every
  held-out cell as ``compare_support.py`` does (``heldout_scores.py``): mean
  |error| and RMSE of the posterior mean, the mean and mean square of the
  predictive z, 90% predictive coverage, the mean Gaussian log score and the
  mean count score (the log predictive probability of the published value
  under a scaled-Poisson survey model, ``heldout_scores.count_log_score``;
  the case size behind a published zero is an assumption there), by level
  and relation to the fitted tables, plus the B17010 race x poverty cells,
  plus the cell's own evidence (published zero, ``0 < k_hat < 2``,
  ``k_hat >= 2``); and against ``--base`` (default the first ``--exp``), over
  the cells both have, in the same groups: the mean log score and count score
  differences and the share of cells where the experiment's mean is nearer the
  published value (a tie counting half).

Experiments, PUMAs or runs that are missing are reported in summary.txt and
``missing.csv`` (reason: no ``experiment.json``, PUMA not in the experiment, no
trace, no report, no MAP, no held-out tables); everything else is still
tabulated.

Usage, on a CPU node::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log experiments/run_python.sbatch \\
        compare_variants.py --exp $R/tn01_gpu $R/tn05_long_base $R/tn05_long_prior3 \\
        $R/tn05_long_tweedie18 $R/tn05_long_state_e0.01 \\
        --puma 4703000 4702300 4702900 4701900 --base tn05_long_base

Writes ``runs.csv`` (one row per experiment x PUMA), ``coordinates.csv``,
``heldout.csv``, ``heldout_vs_base.csv``, ``missing.csv`` and ``summary.txt``
to ``--out`` (default ``<first --exp>/compare_variants``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment as rx  # noqa: E402
from cell_cases import cell_frame  # noqa: E402
from compare_support import thinned  # noqa: E402
from heldout_scores import (LEVELS, area_names, closer_share, group_summary,  # noqa: E402
                            level_frame, predict, score_cells, score_groups, zero_case_size)

#: The coordinate key shared by every experiment's runs of a PUMA.
CELL = ["level", "area", "constraint"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=Path, nargs="+", required=True,
                        help="experiment folders (<root>/<name>)")
    parser.add_argument("--puma", nargs="+", default=None,
                        help="default: every PUMA of any of the experiments")
    parser.add_argument("--kind", default="auto",
                        help="the HMC run to read, hmc_<kind>: 'short', 'ref', or 'auto' (each "
                             "experiment's ref run if it has one, else its short run)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="default: each experiment's first alpha")
    parser.add_argument("--worst", type=int, default=3,
                        help="worst-R-hat coordinates taken from each experiment's run of a PUMA")
    parser.add_argument("--draws", type=int, default=400,
                        help="draws per run for the held-out scores, evenly spaced over its kept draws")
    parser.add_argument("--base", default=None,
                        help="experiment name the held-out scores are compared with "
                             "(default: the first --exp)")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


# -- the report and the trace -------------------------------------------------

NUMBER = r"([-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?|nan|inf)"


def number(text: str) -> float:
    return float(text.replace(",", ""))


#: ``_report.txt`` lines (run_mcmc.py report) and the fields they give.
REPORT_PATTERNS = [
    (rf"step size {NUMBER}; leapfrog steps per iteration median {NUMBER}, max {NUMBER}",
     ("step_size", "leapfrog_median", "leapfrog_max")),
    (rf"mean acceptance probability {NUMBER}", ("accept",)),
    (rf"divergent transitions {NUMBER} of {NUMBER}", ("divergent", "transitions")),
    (rf"R-hat\s+median {NUMBER}\s+p90 {NUMBER}\s+p99 {NUMBER}\s+max {NUMBER}",
     ("rhat_median", "rhat_p90", "rhat_p99", "rhat_max")),
    (rf"R-hat over 1\.01: {NUMBER}; over 1\.1: {NUMBER}", ("rhat_over_1.01", "rhat_over_1.1")),
    (rf"ESS bulk min {NUMBER}, median {NUMBER}; tail min {NUMBER}",
     ("ess_bulk_min", "ess_bulk_median", "ess_tail_min")),
    (rf"share of N: median {NUMBER}%\s+p90 {NUMBER}%\s+p99 {NUMBER}%\s+max {NUMBER}%",
     ("share_median_pct", "share_p90_pct", "share_p99_pct", "share_max_pct")),
]


def parse_report(text: str) -> dict[str, float]:
    """The report's figures (:data:`REPORT_PATTERNS`); a line not found leaves its fields out."""
    out = {}
    for pattern, fields in REPORT_PATTERNS:
        match = re.search(pattern, text)
        if match:
            out.update({field: number(value) for field, value in zip(fields, match.groups())})
    return out


def trace_figures(trace: dict, planned: int | None) -> dict:
    """Cost and sampler figures from the trace itself (also for a run still going)."""
    warmup = int(trace["warmup"])
    done = int(trace["log_pi"].shape[0])
    post = slice(warmup, None)
    out = {"seconds": float(trace.get("seconds", np.nan)),
           "warmup_seconds": float(trace.get("warmup_seconds", np.nan)),
           "iterations": done, "planned": planned, "warmup": warmup,
           "chains": int(trace["log_pi"].shape[1]),
           "kept_draws": int(trace["lam"].shape[0]) if "lam" in trace else 0,
           "peak_gpu_gb": float(trace.get("peak_gpu_gb", np.nan))}
    if "scale" in trace and np.size(trace["scale"]):
        scale = np.asarray(trace["scale"], float)
        out.update(scale_median=float(np.median(scale)), scale_max=float(scale.max()))
    if done > warmup:  # the report's sampler figures, from the trace
        leap = np.asarray(trace["n_leapfrog"])[post]
        div = np.asarray(trace["divergent"])[post]
        out.update(t_step_size=float(np.asarray(trace["step_size"])[-1]),
                   t_leapfrog_median=float(np.median(leap)), t_leapfrog_max=float(leap.max()),
                   t_divergent=int(div.sum()), t_transitions=int(div.size))
    return out


def coordinate_table(inputs, puma: str, lam: np.ndarray, map_lam: np.ndarray | None,
                     az) -> pd.DataFrame:
    """Per multiplier coordinate (the data's layout, ``cell_cases.cell_frame``'s
    order): published estimate, SE and ``k_hat``, the MAP value, and over the
    kept draws ``lam`` (kept, chains, m) R-hat, bulk ESS, median and 90% interval.
    Coordinates a hierarchy fixes (no spread) get no R-hat or ESS."""
    frame = cell_frame(inputs, puma)[CELL + ["estimate", "se", "k_hat"]]
    if lam.shape[2] != len(frame):
        raise ValueError(f"the trace has {lam.shape[2]} coordinates, the inputs {len(frame)}")
    flat = lam.reshape(-1, lam.shape[2])
    q05, q50, q95 = np.quantile(flat, [0.05, 0.5, 0.95], axis=0)
    frame = frame.assign(map_lam=np.nan if map_lam is None else map_lam,
                         median=q50, q05=q05, q95=q95, rhat=np.nan, ess_bulk=np.nan)
    varying = flat.std(axis=0) > 0
    if lam.shape[0] >= 4 and varying.any():
        chains_first = np.transpose(lam, (1, 0, 2))[:, :, varying]
        dataset = az.convert_to_dataset({"lam": chains_first})
        frame.loc[varying, "rhat"] = az.rhat(dataset)["lam"].values
        frame.loc[varying, "ess_bulk"] = az.ess(dataset, method="bulk")["lam"].values
    return frame


# -- one run ------------------------------------------------------------------


class Run:
    """One experiment's run of one PUMA: what was found, and its figures."""

    def __init__(self, folder: Path, spec: dict, puma: str, kind: str, alpha: float | None):
        self.name, self.puma = folder.name, puma
        self.spec = spec
        kinds = (spec.get("hmc") or {}).get("kinds", ["short", "ref"])
        self.kind = ("ref" if "ref" in kinds else "short") if kind == "auto" else kind
        self.alpha = float(spec["alphas"][0] if alpha is None else alpha)
        self.fit = rx.fit_name(spec, puma, self.alpha)
        hmc = folder / puma / f"hmc_{self.kind}"
        self.report = hmc / f"{self.fit}_report.txt"
        self.trace = hmc / f"{self.fit}_trace.npz"
        self.map = folder / puma / "map" / f"{self.fit}.npz"
        self.inputs_path = rx.inputs_dir(spec) / puma
        settings = (spec.get("hmc") or {}).get(self.kind) or {}
        self.planned = (int(settings["warmup"]) + int(settings["samples"])
                        if "warmup" in settings and "samples" in settings else None)

    def label(self) -> dict:
        return {"experiment": self.name, "puma": self.puma, "kind": self.kind,
                "alpha": self.alpha, "level": rx.level(self.spec)}


def load_experiment(folder: Path) -> dict | None:
    path = folder / "experiment.json"
    return json.loads(path.read_text()) if path.exists() else None


def heldout_tables(run: Run, inputs, lam: np.ndarray, draws: int) -> pd.DataFrame | None:
    """Every held-out cell of the run's inputs, scored (``heldout_scores``, with
    the count score, its zeros' case sizes from these inputs:
    ``zero_case_size``); ``None`` without held-out tables there."""
    from pmedm_vb.assemble.heldout import HeldOut

    if not HeldOut.exists(run.inputs_path):
        return None
    held = HeldOut.load(run.inputs_path)
    predictions = predict(inputs, held, thinned(lam, draws))
    areas = area_names(inputs)
    own_inputs, own_held = own_records(run, inputs, held)
    return pd.concat([level_frame(held, level, areas[level],
                                  {"post": score_cells(predictions[level], held.Y[level],
                                                       held.v[level],
                                                       zero_case_size(own_inputs, own_held, level))})
                      for level in LEVELS], ignore_index=True)


def own_records(run: Run, inputs, held):
    """The inputs and held-out tables the published zeros' case sizes are taken
    from (``heldout_scores.zero_case_size``): for a run on a statewide support,
    the PUMA's own-records inputs (the experiment's area without
    ``model.epsilon``) where they are assembled with the same held-out cells, so
    every run of a PUMA is scored with the same ones; otherwise the run's own."""
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    if inputs.unit_members is None:
        return inputs, held
    local = {**run.spec, "model": {**run.spec["model"], "epsilon": None}}
    path = rx.inputs_dir(local) / run.puma
    if (path / "manifest.json").exists() and HeldOut.exists(path):
        own = HeldOut.load(path)
        if own.names == held.names:
            return PMEDMInputs.load(path), own
    print(f"  {run.name} {run.puma}: no PUMA-only inputs with these held-out tables at {path}; "
          "the zeros' case sizes come from the statewide support's merged rows", flush=True)
    return inputs, held


def heldout_summary(cells: pd.DataFrame) -> list[dict]:
    """Per level and group (relation, race x poverty, evidence): :func:`group_summary`."""
    rows = []
    for level in LEVELS:
        for group, part in score_groups(cells[cells["level"] == level]):
            rows.append({"level": level, "group": group, **group_summary(part, "post")})
    return rows


def versus_base(cells: pd.DataFrame, base: pd.DataFrame) -> list[dict]:
    """Against the base run, over the cells both have: per level and group
    (:func:`heldout_scores.score_groups`, the evidence groups by this run's
    published values, which both runs share), the mean log score difference and
    the mean count score difference (this minus base; positive favours this
    run; the count score over the cells where both have one) and the share of
    cells where this run's mean is nearer the published value, a tie counting
    half (``heldout_scores.closer_share``)."""
    keys = ["level", "area", "cell"]
    columns = ["post_error", "post_log_score"] + (["post_count_score"]
                                                  if "post_count_score" in base else [])
    both = cells.merge(base[keys + columns], on=keys, suffixes=("", "_base"))
    rows = []
    for level in LEVELS:
        for group, part in score_groups(both[both["level"] == level]):
            ok = part.dropna(subset=["post_log_score", "post_log_score_base"])
            row = {"level": level, "group": group, "cells": len(ok),
                   "log_score_diff": float((ok["post_log_score"]
                                            - ok["post_log_score_base"]).mean())}
            if "post_count_score_base" in ok and "post_count_score" in ok:
                counted = ok.dropna(subset=["post_count_score", "post_count_score_base"])
                row["count_score_diff"] = float((counted["post_count_score"]
                                                 - counted["post_count_score_base"]).mean())
            row["closer"] = closer_share(ok["post_error"], ok["post_error_base"])
            rows.append(row)
    return rows


# -- the summary --------------------------------------------------------------

RUN_COLUMNS = [
    ("experiment", "experiment", "{}"), ("status", "status", "{}"),
    ("seconds", "min", "{:.1f}"), ("warmup_seconds", "warm min", "{:.1f}"),
    ("iterations", "iter", "{:.0f}"), ("step_size", "step", "{:.3g}"),
    ("leapfrog_median", "leap med", "{:.0f}"), ("leapfrog_max", "leap max", "{:.0f}"),
    ("divergent", "div", "{:.0f}"), ("rhat_max", "R-hat max", "{:.3f}"),
    ("rhat_over_1.01", ">1.01", "{:.0f}"), ("ess_bulk_min", "bulk min", "{:.0f}"),
    ("ess_tail_min", "tail min", "{:.0f}"), ("share_max_pct", "share max %", "{:.3f}"),
    ("map_max_abs_lam", "MAP |lam| max", "{:.1f}"), ("map_over_10", "MAP >10", "{:.0f}"),
]


def text_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> list[str]:
    """``frame`` as aligned text: (column, heading, format) per column; blank for missing."""
    cells = []
    for _, row in frame.iterrows():
        line = []
        for column, _, fmt in columns:
            value = row.get(column, np.nan)
            if column in ("seconds", "warmup_seconds"):
                value = value / 60 if pd.notna(value) else value
            line.append("" if value is None or (isinstance(value, float) and np.isnan(value))
                        else fmt.format(value))
        cells.append(line)
    heads = [heading for _, heading, _ in columns]
    widths = [max([len(h)] + [len(c[i]) for c in cells]) for i, h in enumerate(heads)]
    out = ["  ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(heads, widths)))]
    out += ["  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(line, widths)))
            for line in cells]
    return out


def main() -> None:
    args = parse_args()
    from pmedm_vb.assemble.inputs import PMEDMInputs
    from pmedm_vb.compare import import_arviz

    az = import_arviz()
    out = args.out or args.exp[0] / "compare_variants"
    out.mkdir(parents=True, exist_ok=True)
    specs = {folder.name: (folder, load_experiment(folder)) for folder in args.exp}
    missing = [{"experiment": name, "puma": "", "reason": f"no {folder / 'experiment.json'}"}
               for name, (folder, spec) in specs.items() if spec is None]
    pumas = args.puma or sorted({p for _, spec in specs.values() if spec for p in spec["pumas"]})
    base_name = args.base or args.exp[0].name
    if base_name not in specs:
        raise SystemExit(f"--base {base_name}: not among the --exp folders")

    runs, coords, held_rows, versus_rows = [], [], [], []
    held_cells: dict[tuple[str, str], pd.DataFrame] = {}
    for puma in pumas:
        for name, (folder, spec) in specs.items():
            if spec is None:
                continue
            if puma not in spec["pumas"]:
                missing.append({"experiment": name, "puma": puma,
                                "reason": "PUMA not in the experiment"})
                continue
            run = Run(folder, spec, puma, args.kind, args.alpha)
            row = {**run.label(), "status": "ok"}
            if not run.trace.exists():
                missing.append({"experiment": name, "puma": puma,
                                "reason": f"no trace ({run.trace})"})
                runs.append({**row, "status": "no trace"})
                continue
            with np.load(run.trace) as saved:
                trace = {key: saved[key] for key in saved.files}
            row.update(trace_figures(trace, run.planned))
            if run.report.exists():
                row.update(parse_report(run.report.read_text()))
            else:
                row["status"] = "running (no report)"
                missing.append({"experiment": name, "puma": puma,
                                "reason": f"no report ({run.report})"})
                for key in ("step_size", "leapfrog_median", "leapfrog_max", "divergent",
                            "transitions"):
                    row[key] = row.get(f"t_{key}", np.nan)
            map_lam = None
            if run.map.exists():
                with np.load(run.map) as saved:
                    map_lam = np.asarray(saved["lam"], float)
            else:
                missing.append({"experiment": name, "puma": puma, "reason": f"no MAP ({run.map})"})
            inputs = PMEDMInputs.load(run.inputs_path)
            if map_lam is not None and map_lam.size == inputs.n_constraints:
                frame = cell_frame(inputs, puma)
                k = int(np.argmax(np.abs(map_lam)))
                row.update(map_max_abs_lam=float(np.abs(map_lam[k])),
                           map_max_cell=" ".join(str(frame[c].iloc[k]) for c in CELL),
                           map_over_10=int((np.abs(map_lam) > 10).sum()))
            else:
                map_lam = None
            if "lam" in trace and trace["lam"].shape[0] > 0:
                table = coordinate_table(inputs, puma, trace["lam"], map_lam, az)
                table.insert(0, "experiment", name)
                table.insert(1, "puma", puma)
                worst = table.dropna(subset=["rhat"]).nlargest(args.worst, "rhat")
                table["worst_here"] = table.index.isin(worst.index)
                coords.append(table)
                if len(worst):
                    top = worst.iloc[0]
                    row.update(worst_cell=" ".join(str(top[c]) for c in CELL),
                               worst_rhat=float(top["rhat"]))
                cells = heldout_tables(run, inputs, trace["lam"], args.draws)
                if cells is None:
                    missing.append({"experiment": name, "puma": puma,
                                    "reason": f"no held-out tables in {run.inputs_path}"})
                else:
                    held_cells[(name, puma)] = cells
                    held_rows += [{"experiment": name, "puma": puma, **r}
                                  for r in heldout_summary(cells)]
            runs.append(row)
            print(f"{puma} {name}: {row['status']}", flush=True)
        if (base_name, puma) in held_cells:
            for (name, p), cells in held_cells.items():
                if p == puma and name != base_name:
                    versus_rows += [{"experiment": name, "puma": puma, "base": base_name, **r}
                                    for r in versus_base(cells, held_cells[(base_name, puma)])]

    runs = pd.DataFrame(runs)
    runs.to_csv(out / "runs.csv", index=False)
    missing = pd.DataFrame(missing, columns=["experiment", "puma", "reason"])
    missing.to_csv(out / "missing.csv", index=False)
    coords = pd.concat(coords, ignore_index=True) if coords else pd.DataFrame()
    held = pd.DataFrame(held_rows)
    held.to_csv(out / "heldout.csv", index=False)
    versus = pd.DataFrame(versus_rows)
    versus.to_csv(out / "heldout_vs_base.csv", index=False)

    lines = [f"compare_variants: {len(specs)} experiment(s) x {len(pumas)} PUMA(s); HMC run "
             f"{args.kind}; worst {args.worst} R-hat coordinate(s) from each run; held-out "
             f"scores from {args.draws} draws, compared with {base_name}", ""]
    if len(missing):
        lines.append(f"missing ({len(missing)}):")
        lines += [f"  {r.experiment} {r.puma}: {r.reason}" for r in missing.itertuples()]
        lines.append("")
    lines += ["min: HMC wall time (trace seconds), warm min: its warmup part; leap: leapfrog "
              "steps per sampling iteration; div: divergent sampling transitions; R-hat, ESS over "
              "the kept lambda draws; share max: largest cell's share of N (report); MAP |lam|: "
              "the MAP multipliers",
              "held-out groups: relation to the fitted tables, B17010 race x poverty, and the "
              "cell's own evidence, k_hat = (estimate/SE)^2 (published zero, 0 < k_hat < 2, "
              "k_hat >= 2); count score: log posterior predictive probability of the published "
              "value under a scaled-Poisson survey model (k_hat cases of size SE^2/estimate; a "
              "zero's case size assumed, N/n per unit times persons per carrying unit)", ""]
    selected = []
    for puma in pumas:
        part = runs[runs["puma"] == puma] if len(runs) else runs
        lines.append(f"== PUMA {puma}")
        if part.empty:
            lines += ["   no run found", ""]
            continue
        lines += ["   " + line for line in text_table(part, RUN_COLUMNS)]
        if len(coords):
            here = coords[coords["puma"] == puma]
            keys = here.loc[here["worst_here"], CELL].drop_duplicates()
            if len(keys):
                chosen = here.merge(keys, on=CELL)
                selected.append(chosen)
                lines.append("   worst-R-hat coordinates (each run's worst; all runs shown): "
                             "posterior median [5%, 95%], R-hat, bulk ESS, MAP")
                for key, group in chosen.groupby(CELL, sort=False):
                    first = group.iloc[0]
                    lines.append(f"     {' '.join(map(str, key))}: published {first['estimate']:g} "
                                 f"+- {first['se']:.3g} (k_hat {first['k_hat']:.2f})")
                    for r in group.itertuples():
                        lines.append(f"       {r.experiment:<28} {r.median:8.2f} [{r.q05:8.2f}, "
                                     f"{r.q95:8.2f}]  R-hat {r.rhat:.3f}  ESS {r.ess_bulk:6.0f}  "
                                     f"MAP {r.map_lam:8.2f}")
        hp = held[held["puma"] == puma] if len(held) else held
        if len(hp):
            lines.append("   held-out cells (mean |error|, RMSE, mean z, mean z^2, 90% coverage, "
                         "mean Gaussian log score, mean count score):")
            table = hp.drop(columns="puma").round(3).to_string(index=False)
            lines += ["     " + line for line in table.splitlines()]
        vp = versus[versus["puma"] == puma] if len(versus) else versus
        if len(vp):
            lines.append(f"   against {base_name}, same cells (log score and count score "
                         "differences per cell, positive favours the experiment; share of cells "
                         "nearer, ties half):")
            table = vp.drop(columns=["puma", "base"]).round(4).to_string(index=False)
            lines += ["     " + line for line in table.splitlines()]
        lines.append("")
    if selected:
        pd.concat(selected, ignore_index=True).to_csv(out / "coordinates.csv", index=False)
    else:
        pd.DataFrame(columns=["experiment", "puma", *CELL]).to_csv(out / "coordinates.csv",
                                                                    index=False)
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
