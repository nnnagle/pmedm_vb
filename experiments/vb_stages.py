"""What the skewed VB fit's second stage does to the Gaussian part, and whether the ELBO wants it.

``solve_vb(family="skewed")`` fits a Gaussian first, then re-optimises the
Gaussian part together with the sinh-arcsinh layers. Against the reference
HMC, the skewed fit's Gaussian part is 2.6-6.8 times narrower than the
Gaussian-only fit along the widest whitened directions (``skew_check.py``
section 4), which sit in the PUMA-level multipliers. This reads only the VB
result files and reports, per fit:

1. **ELBOs**: Laplace, the skewed run's Gaussian stage, its final (skewed)
   ELBO, and the separate Gaussian-only run's, each with its Monte Carlo SE;
   the iterations and whether each stage converged. A final ELBO clearly
   above the Gaussian stage's means the objective itself prefers the
   narrower fit, not an optimisation artefact.
2. **Widths**: each sampling coordinate's sd under the skewed fit's final
   Gaussian part and under the whole skewed family, over its Gaussian
   stage's sd -- the ``scale`` the skewed layers were given, which is that
   stage's marginal sd exactly -- for the block holding the last coordinates
   (the PUMA rows under the ``puma``, ``nullspace`` and roll-up levels) and
   for every other coordinate, as quantiles; and the Gaussian-only run's sd
   over the same stage-1 sd, to show how close that run is to stage 1.

Writes ``<out>/vb_stages.csv`` and ``vb_stages.txt``::

    $PY experiments/vb_stages.py --root $PAPER --alpha 0.01 --pumas 4701501 \\
        --out $PAPER/oracle_scale/vb_stages
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import logger, record_run

from laplace_diagnostic import load_vb
from run_experiment import fit_name

KEYS = ("laplace_elbo", "laplace_elbo_se", "gaussian_elbo", "gaussian_elbo_se", "elbo",
        "elbo_se", "converged", "n_iter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' results root")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def saved_values(path: Path) -> dict:
    with np.load(path) as saved:
        return {k: (float(saved[k]) if k != "converged" else bool(saved[k]))
                for k in KEYS if k in saved.files}


def quantiles(prefix: str, ratio: np.ndarray) -> dict:
    qs = np.quantile(ratio, [0.1, 0.5, 0.9]) if ratio.size else [np.nan] * 3
    return {f"{prefix}_p10": float(qs[0]), f"{prefix}_p50": float(qs[1]),
            f"{prefix}_p90": float(qs[2])}


def one_fit(skewed_path: Path, gaussian_path: Path, draws: int, rng: np.random.Generator) -> dict:
    q, _, _ = load_vb(skewed_path)
    if q.scale is None:
        raise SystemExit(f"{skewed_path} has no skew layer")
    stage1_sd = q.scale                                            # the Gaussian stage's marginal sd
    eps = rng.standard_normal((q.size, draws))
    gauss = q.gaussian_part(eps)
    final_sd = gauss.std(axis=1)
    family_sd = q.skew_forward(gauss)[0].std(axis=1)

    last = np.zeros(q.size, bool)
    last[q.rows[int(np.argmax([r.max() for r in q.rows]))]] = True   # the block with the last row
    row = {"m": q.size, "last_block": int(last.sum())}
    for group, mask in (("last", last), ("other", ~last)):
        row.update(quantiles(f"{group}_gauss_part_over_stage1", final_sd[mask] / stage1_sd[mask]))
        row.update(quantiles(f"{group}_family_over_stage1", family_sd[mask] / stage1_sd[mask]))

    s = saved_values(skewed_path)
    row.update({f"skewed_{k}": v for k, v in s.items()})
    if gaussian_path.exists():
        g = saved_values(gaussian_path)
        row.update({f"gaussian_run_{k}": v for k, v in g.items()
                    if k in ("elbo", "elbo_se", "converged", "n_iter")})
        q_g, _, _ = load_vb(gaussian_path)
        g_sd = q_g.gaussian_part(rng.standard_normal((q_g.size, draws))).std(axis=1)
        for group, mask in (("last", last), ("other", ~last)):
            row.update(quantiles(f"{group}_gaussian_run_over_stage1", g_sd[mask] / stage1_sd[mask]))
    return row


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)

    rows = []
    for experiment in args.experiments:
        exp = args.root / experiment
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            name = fit_name(spec, puma, args.alpha)
            skewed = exp / puma / "vb_skewed" / f"{name}.npz"
            if not skewed.exists():
                logger.info("%s %s: no skewed fit; skipped", experiment, puma)
                continue
            logger.info("%s %s", experiment, puma)
            rows.append({"experiment": experiment, "puma": puma,
                         **one_fit(skewed, exp / puma / "vb_gaussian" / f"{name}.npz",
                                   args.draws, rng)})
            pd.DataFrame(rows).to_csv(args.out / "vb_stages.csv", index=False)
    if not rows:
        raise SystemExit("no fits found")

    table = pd.DataFrame(rows)
    ident = ["experiment", "puma"]
    sections = {
        "1. ELBOs (Monte Carlo SE): Laplace; the skewed run's Gaussian stage and final; the "
        "separate Gaussian-only run":
            ident + [c for c in table.columns if "elbo" in c or "converged" in c or "n_iter" in c],
        "2. Widths over the Gaussian stage's marginal sd (p10, p50, p90): the skewed fit's final "
        "Gaussian part, the whole skewed family, and the Gaussian-only run; 'last' is the block "
        "holding the last coordinates (the PUMA rows), 'other' every other coordinate":
            ident + ["m", "last_block"] + [c for c in table.columns if c.startswith(("last_", "other_"))],
    }
    lines = [f"The skewed VB fit's two stages, alpha {args.alpha:g}, from {args.root}", ""]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for title, columns in sections.items():
            lines += [title, table[columns].to_string(index=False, float_format=lambda v: f"{v:,.4g}"), ""]
    text = "\n".join(lines)
    (args.out / "vb_stages.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out / 'vb_stages.csv'} and vb_stages.txt")


if __name__ == "__main__":
    main()
