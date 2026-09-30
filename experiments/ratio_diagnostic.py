"""Where do the weight ratios sit, in VB and HMC draws? The evidence for a ratio cap.

A unit's weight ratio in a zone is ``w / (N q)``, with ``q`` the design
probabilities normalised to sum to 1. Its log,

    l = -X lambda - log Z   (the logits less their log-sum-exp, less log q),

is what a cap ``(tau/2) (l - b)_+^2`` would act on. For each fit this takes
draws -- from a VB family (``--runs``) and from HMC traces (``--hmc``) -- and
the MAP, and reports:

- per draw, the largest ``l``, split into draws with and without a *wall* (a
  zone x unit weight over ``--wall`` of N, the scoring's definition);
- per candidate bound (``--bounds``, as ratios), how many unit-zones exceed
  it;
- for each draw's largest cell, what drives it. ``l`` splits exactly into one
  term per constraint row the unit contributes to, ``-x_k lambda_k``, less
  ``log Z``; the rows whose term grew most against the same cell at the MAP
  are its drivers, reported with their published estimate, standard error and
  design-expected count (``N x forward(q)``), and the cell's design
  probability against the median.

HMC samples the posterior itself, so if its draws stay well below VB's, the
walls are VB's, and a cap above HMC's range would leave the posterior nearly
unchanged while trimming VB's tails.

``--runs`` are job directories of ``run_map_bundle.sbatch`` (``map/`` and
``vb_<family>/``); ``--hmc`` directories holding ``<name>_trace.npz``. A trace
is matched to the MAP of the same name among ``--runs``. Writes
``<out>/ratio_diagnostic.txt``, ``ratio_draws.csv`` (one row per draw) and
``ratio_drivers.csv`` (the top driver rows of each draw's largest cell)::

    $CONDA_PREFIX/bin/python experiments/ratio_diagnostic.py \\
        --runs $RUNS/<fit job> ... --hmc $RUNS/nullspace_grid/hmc/*_ref \\
        --out $RUNS/ratio_diagnostic_hmc
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from laplace_diagnostic import load_vb  # noqa: E402

from pmedm_vb.assemble.hierarchy import Hierarchy  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402

NAME = re.compile(r"^(?P<puma>\d+)_(?P<taper>[^_]+)_a(?P<alpha>[0-9.e-]+)(?:_h(?P<level>.+))?$")
TOP_DRIVERS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="*", type=Path, default=[],
                        help="fit job directories (map/, vb_<family>/)")
    parser.add_argument("--hmc", nargs="*", type=Path, default=[],
                        help="directories holding HMC <name>_trace.npz")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--family", default="skewed")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--bounds", nargs="+", type=float, default=[10, 50, 100, 500, 1000],
                        help="candidate ratio caps (w / N q)")
    parser.add_argument("--wall", type=float, default=0.01, help="a wall is a cell over this share of N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


class Ratios:
    """Log weight ratios of one PUMA's problem, and their split by constraint row."""

    def __init__(self, inputs: PMEDMInputs) -> None:
        self.inputs = inputs
        self.op = ConstraintOperator(inputs)
        q = inputs.q / inputs.q.sum()
        self.support = q > 0
        with np.errstate(divide="ignore"):
            self.log_q = np.log(q)
        self.median_log_q = float(np.median(self.log_q[self.support]))
        self.design = inputs.N * self.op.forward(q)          # design-expected count per row
        self.targets = inputs.targets()
        self.se = np.sqrt(inputs.sigma_v)
        self.names = ([f"T:{n}" for n in inputs.tract_constraints for _ in range(inputs.Y_T.shape[0])]
                      + [f"B:{n}" for n in inputs.bg_constraints for _ in range(inputs.Y_B.shape[0])])
        self.x_t = inputs.X_T.tocsr()
        self.x_b = inputs.X_B.tocsr()
        self.zone_tract = inputs.zone_tracts()

    def log_ratio(self, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """``(l, log p, log Z)``, the first two as ``(n_zones, n_units)``."""
        logits = self.log_q - self.op.adjoint(lam)
        log_z = float(logsumexp(logits[self.support]))
        log_p = logits - log_z
        return np.where(self.support, log_p - self.log_q, -np.inf), log_p, log_z

    def terms(self, zone: int, unit: int, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rows unit ``unit`` contributes to in zone ``zone``, and each row's
        term ``-x lambda`` of its log ratio there."""
        n_tracts, n_zones = self.inputs.Y_T.shape[0], self.inputs.Y_B.shape[0]
        tract = int(self.zone_tract[zone])
        row_t, row_b = self.x_t.getrow(unit), self.x_b.getrow(unit)
        rows = np.concatenate([row_t.indices * n_tracts + tract,
                               self.inputs.Y_T.size + row_b.indices * n_zones + zone])
        x = np.concatenate([row_t.data, row_b.data])
        return rows, -x * lam[rows]


def fits_in(run: Path, family: str):
    for path in sorted((run / f"vb_{family}").glob("*.npz")):
        match = NAME.match(path.stem)
        if match:
            yield path, match


def traces_in(directory: Path):
    for path in sorted(directory.glob("*_trace.npz")):
        match = NAME.match(path.stem[: -len("_trace")])
        if match:
            yield path, match


def analyse(ratios: Ratios, key: dict, source: str, lam_draws: np.ndarray, map_lam: np.ndarray | None,
            args, draw_rows: list, driver_rows: list) -> None:
    """Statistics of each draw (columns of ``lam_draws``) and the drivers of its largest cell."""
    log_bounds = np.log(args.bounds)
    if map_lam is not None:
        l_map, log_p_map, log_z_map = ratios.log_ratio(map_lam)
        if source == "vb":   # the MAP row once per fit
            draw_rows.append(dict(**key, source="map", draw=-1,
                                  wall=bool(np.exp(log_p_map[ratios.support].max()) > args.wall),
                                  max_log_ratio=float(l_map[ratios.support].max()),
                                  **{f"over_{b:g}": int((l_map > lb).sum())
                                     for b, lb in zip(args.bounds, log_bounds)}))
    for d in range(lam_draws.shape[1]):
        lam = lam_draws[:, d]
        l, log_p, log_z = ratios.log_ratio(lam)
        zone, unit = np.unravel_index(np.argmax(np.where(ratios.support, log_p, -np.inf)), log_p.shape)
        wall = bool(np.exp(log_p[zone, unit]) > args.wall)
        draw_rows.append(dict(**key, source=source, draw=d, wall=wall,
                              max_log_ratio=float(l[ratios.support].max()),
                              top_log_ratio=float(l[zone, unit]),
                              top_share=float(np.exp(log_p[zone, unit])),
                              top_log_q_offset=float(ratios.log_q[zone, unit] - ratios.median_log_q),
                              **{f"over_{b:g}": int((l > lb).sum())
                                 for b, lb in zip(args.bounds, log_bounds)}))
        rows, terms = ratios.terms(zone, unit, lam)
        if map_lam is not None:
            _, terms_map = ratios.terms(zone, unit, map_lam)
            growth = terms - terms_map
            log_z_change = log_z - log_z_map
        else:
            growth = terms
            log_z_change = np.nan
        for rank, i in enumerate(np.argsort(growth)[::-1][:TOP_DRIVERS]):
            row = int(rows[i])
            driver_rows.append(dict(**key, source=source, draw=d, wall=wall, rank=rank + 1,
                                    name=ratios.names[row], row=row,
                                    term=float(terms[i]), growth=float(growth[i]),
                                    log_z_change=float(log_z_change),
                                    published=float(ratios.targets[row]), se=float(ratios.se[row]),
                                    design=float(ratios.design[row])))


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    draw_rows: list[dict] = []
    driver_rows: list[dict] = []
    maps = {path.stem: path for run in args.runs for path in (run / "map").glob("*.npz")}
    cache: dict[str, Ratios] = {}

    def ratios_for(puma: str) -> Ratios:
        if puma not in cache:
            cache.clear()   # one PUMA's problem in memory at a time
            cache[puma] = Ratios(PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma))
        return cache[puma]

    def map_lam(name: str) -> np.ndarray | None:
        if name not in maps:
            return None
        with np.load(maps[name]) as saved:
            return saved["lam"]

    jobs = [("vb", path, match) for run in args.runs for path, match in fits_in(run, args.family)]
    jobs += [("hmc", path, match) for d in args.hmc for path, match in traces_in(d)]
    jobs.sort(key=lambda job: job[2].group("puma"))
    for source, path, match in jobs:
        puma, alpha = match.group("puma"), float(match.group("alpha"))
        level, taper = match.group("level") or "none", match.group("taper")
        name = path.stem if source == "vb" else path.stem[: -len("_trace")]
        key = dict(puma=puma, alpha=alpha, level=level)
        ratios = ratios_for(puma)
        if source == "vb":
            h = Hierarchy.build(ratios.inputs, level, alpha=alpha,
                                taper=None if taper == "none" else taper, variance_floor="zero")
            q, _, _ = load_vb(path)
            xi = q.sample(rng, args.draws)                                   # (size, draws)
            lam = xi if h.is_trivial else h.lambda_data(xi)
        else:
            with np.load(path) as saved:
                kept = saved["lam"]                                          # (kept, chains, m)
            flat = kept.reshape(-1, kept.shape[-1])
            pick = np.unique(np.linspace(0, len(flat) - 1, min(args.draws, len(flat))).astype(int))
            lam = flat[pick].T
        analyse(ratios, key, source, lam, map_lam(name), args, draw_rows, driver_rows)
        print(f"done {source} {name}: {lam.shape[1]} draws", flush=True)

    draws = pd.DataFrame(draw_rows)
    draws.to_csv(args.out / "ratio_draws.csv", index=False)
    drivers = pd.DataFrame(driver_rows)
    drivers.to_csv(args.out / "ratio_drivers.csv", index=False)
    write_report(args, draws, drivers)


def group_of(frame: pd.DataFrame) -> pd.Series:
    return np.where(frame.source == "map", "MAP",
                    np.where(frame.source == "hmc", np.where(frame.wall, "HMC, wall", "HMC, no wall"),
                             np.where(frame.wall, "VB, wall", "VB, no wall")))


def write_report(args, draws: pd.DataFrame, drivers: pd.DataFrame) -> None:
    over = [f"over_{b:g}" for b in args.bounds]
    order = ["MAP", "HMC, no wall", "HMC, wall", "VB, no wall", "VB, wall"]
    lines = [f"Log weight ratios l = log(w / N q); bounds as ratios {args.bounds}; wall = a cell over "
             f"{args.wall:g} of N; up to {args.draws} draws per fit (VB-{args.family}, HMC thinned)"]
    draws = draws.assign(group=group_of(draws))
    for (puma, alpha, level), part in draws.groupby(["puma", "alpha", "level"], sort=True):
        lines.append(f"\n== {puma} alpha {alpha:g} {level}")
        for group in order:
            sub = part[part.group == group]
            if sub.empty:
                continue
            m = sub.max_log_ratio
            counts = "  ".join(f">{b:g}: {sub[c].mean():.1f}" for b, c in zip(args.bounds, over))
            lines.append(f"   {group:13s}: {len(sub):4d} draws; max l median {m.median():.2f} "
                         f"(ratio {np.exp(m.median()):,.0f}), range {m.min():.2f}-{m.max():.2f}; "
                         f"mean unit-zones over bound  {counts}")
    lines.append("\nThe largest cell of each draw: its log ratio, share of N, and design "
                 "probability against the median (log), medians by group:")
    cells = draws[draws.source != "map"].groupby("group")[
        ["top_log_ratio", "top_share", "top_log_q_offset"]].median()
    lines.append(cells.reindex([g for g in order if g in cells.index]).round(4).to_string())
    if not drivers.empty:
        drivers = drivers.assign(group=group_of(drivers))
        lines.append(f"\nTop driver of each draw's largest cell (the row whose term -x lambda grew most "
                     f"against the MAP), by group; growth and the log Z change in log-ratio units:")
        top = drivers[drivers["rank"] == 1]
        for group in order:
            sub = top[top.group == group]
            if sub.empty:
                continue
            table = sub.groupby("name").agg(draws=("draw", "size"), growth=("growth", "median"),
                                            log_z_change=("log_z_change", "median"),
                                            published=("published", "median"), se=("se", "median"),
                                            design=("design", "median"))
            lines.append(f"\n   {group} ({len(sub)} draws), most frequent top drivers:")
            lines += ["     " + r for r in
                      table.sort_values("draws", ascending=False).head(15).round(2).to_string().splitlines()]
            lines.append(f"     all top drivers: published median {sub.published.median():.1f}, "
                         f"quartiles {sub.published.quantile(0.25):.1f}-{sub.published.quantile(0.75):.1f}; "
                         f"design median {sub.design.median():.1f}")
    text = "\n".join(lines) + "\n"
    (args.out / "ratio_diagnostic.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
