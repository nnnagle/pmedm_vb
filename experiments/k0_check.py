"""How far does the K0 term with k > 0 move the model from Sigma, and does its MAP
converge? (``OPEN_QUESTIONS.md``, #5.)

The K0 term's covariance at its centre is ``Sigma + 2 k S o S`` and its mean
``a^2 + k diag(S)``, with ``S = A^-1 Sigma A^-1 / 4`` (``pmedm_vb.assemble.k0``).
For each ``--k``, on the model's rows (published rows, roll-up cells, PUMA rows)
at ``--alpha``, untapered:

1. variance inflation, ``2 k S_ii^2 / v_i``, by published zero / nonzero;
2. the change in correlation, ``|rho' - rho|``, over ``--pairs`` random row
   pairs, and over those whose ``|rho|`` is at least 0.1;
3. the mean shift ``k S_ii``, in people, by published zero / nonzero;
4. the MAP with ``+k0w<w>k<k>``: converged, Newton iterations, seconds.

    sbatch --account=acf-utk0011 \\
        --export=ALL,PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
        experiments/run_python.sbatch k0_check.py --area knoxall-2024-5yr \\
        --puma 4701502 --alpha 0.01 --level puma-rpruned+prior3 --weight 18 \\
        --k 0.01 0.1 1
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from pmedm_vb.assemble.hierarchy import Hierarchy
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.k0 import k0_level
from pmedm_vb.config import processed_dir
from pmedm_vb.solvers.map_dual import solve_map

QUANTILES = (0.5, 0.9, 0.99, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", required=True)
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--level", required=True, help="the hierarchy level before the K0 suffix")
    parser.add_argument("--weight", type=float, default=18.0, help="w for the zero rows")
    parser.add_argument("--k", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    parser.add_argument("--pairs", type=int, default=2_000_000)
    parser.add_argument("--no-map", action="store_true", help="skip the MAP solves")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def quantiles(x: np.ndarray) -> dict:
    return {f"q{q:g}": float(np.quantile(x, q)) for q in QUANTILES} if x.size else {}


def main() -> None:
    args = parse_args()
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    rng = np.random.default_rng(args.seed)
    pd.set_option("display.width", 200)
    print(f"{args.puma}, {args.level}, alpha {args.alpha}, w {args.weight:g}")

    # S, Sigma and a do not depend on k: build them once, at k = 0.
    h = Hierarchy.build(inputs, k0_level(args.level, args.weight), alpha=args.alpha, taper=None)
    term = h.tweedie
    zero = term.y == 0
    v = term.sigma0.v
    s_diag = term.d_s + (0 if term.b_s is None else np.square(term.b_s).sum(1))
    rows = term.a.size
    print(f"{rows:,} rows, {int(zero.sum()):,} published zero; S's diagonal (people): "
          f"median {np.median(s_diag):.3g} (zero rows {np.median(s_diag[zero]):.3g})")

    i = rng.integers(0, rows, args.pairs)
    j = rng.integers(0, rows, args.pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    b = term.sigma0.b
    sigma_ij = np.zeros(i.size) if b is None else np.einsum("pr,pr->p", b[i], b[j])
    s_ij = sigma_ij / (4 * term.a[i] * term.a[j])
    rho = sigma_ij / np.sqrt(v[i] * v[j])
    strong = np.abs(rho) >= 0.1

    stats, maps = [], []
    for k in args.k:
        inflation = 2 * k * s_diag**2 / v
        v_k = v + 2 * k * s_diag**2
        rho_k = (sigma_ij + 2 * k * s_ij**2) / np.sqrt(v_k[i] * v_k[j])
        change = np.abs(rho_k - rho)
        shift = k * s_diag
        for name, values in [("variance inflation, nonzero", inflation[~zero]),
                             ("variance inflation, zero", inflation[zero]),
                             ("mean shift (people), nonzero", shift[~zero]),
                             ("mean shift (people), zero", shift[zero]),
                             ("|correlation change|, all pairs", change),
                             ("|correlation change|, |rho| >= 0.1", change[strong])]:
            stats.append({"k": k, "measure": name, **quantiles(values)})
        flips = int(np.sum(np.sign(rho_k[strong]) != np.sign(rho[strong])))
        stats.append({"k": k, "measure": f"sign flips among {int(strong.sum()):,} pairs "
                      f"with |rho| >= 0.1", "q1": float(flips)})
        if args.no_map:
            continue
        level = k0_level(args.level, args.weight, k)
        logging.disable(logging.INFO)
        start = time.time()
        try:
            result = solve_map(inputs, alpha=args.alpha, taper=None, hierarchy=level, max_iter=100)
            maps.append({"k": k, "converged": result.converged, "iterations": result.n_iter,
                         "seconds": round(time.time() - start, 1), "error": ""})
        except Exception as error:  # report and go on to the next k
            maps.append({"k": k, "converged": False, "iterations": np.nan,
                         "seconds": round(time.time() - start, 1), "error": str(error)[:120]})
        logging.disable(logging.NOTSET)

    print("\nThe K0 term's covariance and mean against Sigma and the published values:")
    print(pd.DataFrame(stats).to_string(index=False, float_format=lambda x: f"{x:.4g}",
                                        na_rep=""))
    if maps:
        print("\nMAP with +k0w<w>k<k>:")
        print(pd.DataFrame(maps).to_string(index=False))


if __name__ == "__main__":
    main()
