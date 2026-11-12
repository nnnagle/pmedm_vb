"""Fit the encoder-free baseline embedding, one setting or a grid, with check 1.

    python experiments/embed/fit_baseline.py --config experiments/embed/stage1_v1.json \\
        --out $PMEDM_VB_RUNS/embed/baseline --d 8 --gamma 0 0.5 1 --ridge 1e-4 1e-3 1e-2

Every combination of ``--d``, ``--gamma`` and ``--ridge`` is fitted in turn,
each into its own directory under ``--out`` (``d8_g0.5_r0.001``, with ``_unw``
for ``--unweighted``). A directory holding ``result.json`` is complete and is
skipped, so a job that runs out of time can simply be resubmitted. The target
arrays are read from the cache (``experiments/embed/build_targets.py`` builds
them).
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from pmedm_vb.embed.check import reconstruction
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.fit import fit_baseline, save_fit
from pmedm_vb.embed.units import load_targets
from pmedm_vb.progress import logger, record_run, stage


def run_name(d: int, gamma: float, ridge: float, weighted: bool) -> str:
    return f"d{d}_g{gamma:g}_r{ridge:g}" + ("" if weighted else "_unw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="root for the fit directories")
    parser.add_argument("--d", type=int, nargs="+", required=True)
    parser.add_argument("--gamma", type=float, nargs="+", required=True)
    parser.add_argument("--ridge", type=float, nargs="+", required=True)
    parser.add_argument("--unweighted", action="store_true",
                        help="the diagnostic: every unit weight 1 in the fit")
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if present)")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--calibrate-iter", type=int, default=500)
    args = parser.parse_args()

    config = load_config(args.config)
    arrays = load_targets(config)
    record_run(args.out, args)
    for d, gamma, ridge in itertools.product(args.d, args.gamma, args.ridge):
        directory = args.out / run_name(d, gamma, ridge, not args.unweighted)
        if (directory / "result.json").exists():
            logger.info("%s: already complete, skipped", directory.name)
            continue
        with stage(f"fit {directory.name}"):
            fit = fit_baseline(arrays, d=d, gamma=gamma, ridge=ridge,
                               weighted=not args.unweighted,
                               heldout_fraction=args.heldout_fraction, seed=args.seed,
                               device=args.device, max_iter=args.max_iter, tol=args.tol,
                               calibrate_iter=args.calibrate_iter)
        with stage(f"check 1 {directory.name}"):
            checks = reconstruction(arrays, fit)
        result = save_fit(directory, arrays, fit, checks, args.config)
        logger.info("%s: held-out mean deviance explained %.4f, %d iterations, %.0fs",
                    directory.name, result["heldout_mean_deviance_explained"],
                    result["iterations"], result["seconds"])


if __name__ == "__main__":
    main()
