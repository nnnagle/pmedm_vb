"""Where do the weight ratios of wall draws sit? The evidence for placing a ratio cap.

A unit's weight ratio in a zone is ``w / (N q)``, with ``q`` the design
probabilities normalised to sum to 1. Its log,

    l = log q - X lambda - log Z - log q  (the logits less their log-sum-exp),

is what a cap ``(tau/2) (l - b)_+^2`` would act on. For each fit this draws
from the VB family (and takes the MAP), maps each draw to the multipliers the
data see, and reports:

- per draw, the largest ``l``, split by whether the draw has a *wall* (a
  zone x unit weight over ``--wall`` of N, the scoring's definition);
- per candidate bound ``b`` (``--bounds``, as ratios), how many unit-zones
  exceed it, in wall and wall-free draws, and at the MAP;
- for each wall draw's largest cell, the unit's constraint rows: how often
  each row appears, with its published estimate and its design-expected count
  (``N x forward(q)``), to check that walls ride on rare rows.

A good ``b`` sits above everything the MAP and wall-free draws use and below
what wall draws reach; if the two overlap, a cap would bend ordinary fits.

``--runs`` are job directories of ``run_map_bundle.sbatch`` (each holding
``map/`` and ``vb_<family>/``); every fit found there is analysed. Writes
``<out>/ratio_diagnostic.txt``, ``ratio_draws.csv`` (one row per draw) and
``ratio_wall_rows.csv`` (one row per constraint row seen under a wall)::

    $CONDA_PREFIX/bin/python experiments/ratio_diagnostic.py \\
        --runs $RUNS/6293070 $RUNS/6293071 $RUNS/6293072 $RUNS/6293073 \\
        --out $RUNS/ratio_diagnostic
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
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
    """Log weight ratios of one PUMA's problem."""

    def __init__(self, inputs: PMEDMInputs) -> None:
        self.inputs = inputs
        self.op = ConstraintOperator(inputs)
        q = inputs.q / inputs.q.sum()
        self.support = q > 0
        with np.errstate(divide="ignore"):
            self.log_q = np.log(q)
        self.design = inputs.N * self.op.forward(q)          # design-expected count per row
        self.targets = inputs.targets()
        self.names = ([f"T:{n}" for n in inputs.tract_constraints for _ in range(inputs.Y_T.shape[0])]
                      + [f"B:{n}" for n in inputs.bg_constraints for _ in range(inputs.Y_B.shape[0])])

    def log_ratio(self, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(l, log p)`` over the supported unit-zones, as ``(n_zones, n_units)``."""
        logits = self.log_q - self.op.adjoint(lam)
        log_p = logits - logsumexp(logits[self.support])
        return np.where(self.support, log_p - self.log_q, -np.inf), log_p

    def rows_of(self, zone: int, unit: int) -> np.ndarray:
        """The stacked constraint rows unit ``unit`` contributes to in zone ``zone``."""
        inputs = self.inputs
        n_tracts = inputs.Y_T.shape[0]
        n_zones = inputs.Y_B.shape[0]
        tract = int(inputs.zone_tracts()[zone])
        cols_t = inputs.X_T[unit].nonzero()[1] if hasattr(inputs.X_T, "nonzero") else np.flatnonzero(inputs.X_T[unit])
        cols_b = inputs.X_B[unit].nonzero()[1] if hasattr(inputs.X_B, "nonzero") else np.flatnonzero(inputs.X_B[unit])
        return np.concatenate([cols_t * n_tracts + tract, inputs.Y_T.size + cols_b * n_zones + zone])


def fits_in(run: Path, family: str):
    for path in sorted((run / f"vb_{family}").glob("*.npz")):
        match = NAME.match(path.stem)
        if match:
            yield path, match.group("puma"), float(match.group("alpha")), match.group("level") or "none"


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    log_bounds = np.log(args.bounds)
    draw_rows, wall_rows, lines = [], [], []
    for run in args.runs:
        for path, puma, alpha, level in fits_in(run, args.family):
            inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma)
            ratios = Ratios(inputs)
            h = Hierarchy.build(inputs, level, alpha=alpha, taper=None, variance_floor="zero")
            with np.load(path) as saved:
                found = str(saved["hierarchy"]) if "hierarchy" in saved.files else "none"
                if found != level:
                    raise SystemExit(f"{path}: fitted at {found}, name says {level}")
            q, _, _ = load_vb(path)
            xi = q.sample(rng, args.draws)                       # (size, draws)
            lam = xi if h.is_trivial else h.lambda_data(xi)
            map_path = run / "map" / path.name
            label = f"{puma} a{alpha:g} {level}"

            if map_path.exists():
                with np.load(map_path) as saved:
                    l_map, log_p_map = ratios.log_ratio(saved["lam"])
                draw_rows.append(dict(puma=puma, alpha=alpha, level=level, source="map", draw=-1,
                                      wall=bool(np.exp(log_p_map[ratios.support].max()) > args.wall),
                                      max_log_ratio=float(l_map[ratios.support].max()),
                                      **{f"over_{b:g}": int((l_map > lb).sum())
                                         for b, lb in zip(args.bounds, log_bounds)}))
            for d in range(lam.shape[1]):
                l, log_p = ratios.log_ratio(lam[:, d])
                top = np.unravel_index(np.argmax(np.where(ratios.support, log_p, -np.inf)), log_p.shape)
                wall = bool(np.exp(log_p[top]) > args.wall)
                draw_rows.append(dict(puma=puma, alpha=alpha, level=level, source=args.family, draw=d,
                                      wall=wall, max_log_ratio=float(l[ratios.support].max()),
                                      **{f"over_{b:g}": int((l > lb).sum())
                                         for b, lb in zip(args.bounds, log_bounds)}))
                if wall:
                    for row in ratios.rows_of(*top):
                        wall_rows.append(dict(puma=puma, alpha=alpha, level=level, draw=d,
                                              row=int(row), name=ratios.names[row],
                                              log_ratio=float(l[top]), share=float(np.exp(log_p[top])),
                                              published=float(ratios.targets[row]),
                                              design=float(ratios.design[row])))
            print(f"done {label}", flush=True)

    draws = pd.DataFrame(draw_rows)
    draws.to_csv(args.out / "ratio_draws.csv", index=False)
    walls = pd.DataFrame(wall_rows)
    walls.to_csv(args.out / "ratio_wall_rows.csv", index=False)

    over = [f"over_{b:g}" for b in args.bounds]
    lines.append(f"Log weight ratios l = log(w / N q); bounds as ratios {args.bounds} "
                 f"(log {', '.join(f'{x:.2f}' for x in log_bounds)}); wall = a cell over "
                 f"{args.wall:g} of N; VB-{args.family}, {args.draws} draws per fit")
    for (puma, alpha, level), part in draws.groupby(["puma", "alpha", "level"], sort=True):
        lines.append(f"\n== {puma} alpha {alpha:g} {level}")
        for name, sel in (("MAP", part.source == "map"),
                          ("VB, no wall", (part.source != "map") & ~part.wall),
                          ("VB, wall", (part.source != "map") & part.wall)):
            sub = part[sel]
            if sub.empty:
                lines.append(f"   {name:12s}: none")
                continue
            m = sub.max_log_ratio
            counts = "  ".join(f">{b:g}: {sub[c].mean():.1f}" for b, c in zip(args.bounds, over))
            lines.append(f"   {name:12s}: {len(sub):4d} draws; max l min {m.min():.2f} "
                         f"median {m.median():.2f} max {m.max():.2f} (ratio {np.exp(m.median()):,.0f}); "
                         f"mean unit-zones over bound  {counts}")
    if not walls.empty:
        lines.append("\nConstraint rows under the largest cell of wall draws (all fits), most frequent:")
        rows = walls.groupby("name").agg(wall_draws=("draw", "size"),
                                         published=("published", "median"),
                                         design=("design", "median"),
                                         log_ratio=("log_ratio", "median"))
        rows = rows.sort_values("wall_draws", ascending=False).head(30)
        lines.append(rows.round(2).to_string())
        lines.append("\nAll rows under wall cells, published and design-expected counts (quartiles):")
        lines.append(walls[["published", "design"]].describe(percentiles=[0.25, 0.5, 0.75]).round(1).to_string())
    text = "\n".join(lines) + "\n"
    (args.out / "ratio_diagnostic.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
