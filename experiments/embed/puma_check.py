"""Leave-one-PUMA-out check of an embedding (see :mod:`pmedm_vb.embed.puma_check`).

    python experiments/embed/puma_check.py --config experiments/embed/stage1_v2.json \\
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d8_g0.5_r0.01 \\
        --out $PMEDM_VB_RUNS/embed/puma_check [--puma 4701502 4702300] [--shard 1/3]

``--run`` is any φ source keyed by ``SERIALNO``: a fit directory (its
``phi_units.parquet``) or a parquet/CSV file; ``--label`` names it in the
output (default: the run directory's name). Results go to
``<out>/<label>/<puma>/`` (``scores.csv``, ``misfit.csv``, ``rare.csv``;
``done.json`` last), and a PUMA already done is skipped, so a timed-out job
can be resubmitted as it was. ``--shard i/n`` takes every n-th PUMA starting
at the i-th, to split the 58 PUMAs across jobs; ``--workers k`` runs k PUMAs
at once in forked processes that share the loaded data (the S2 fits dominate:
minutes per PUMA on one core).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

from pmedm_vb.config import ensure_dir
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import check_puma, load_check_data, load_phi, published_tables
from pmedm_vb.progress import logger, record_run, stage, write_text

#: Shared with forked workers: set once in the parent before the pool starts.
_SHARED: dict = {}


def run_puma(puma: str) -> str:
    """Check one PUMA and write its files; skipped if already done."""
    s = _SHARED
    directory = s["root"] / puma
    if (directory / "done.json").exists():
        logger.info("PUMA %s: already done, skipped", puma)
        return puma
    start = time.perf_counter()
    with stage(f"PUMA {puma}"):
        frames = check_puma(s["data"], puma, s["phi"], s["binary"], s["published"],
                            tau=s["tau"], sets=s["sets"])
    ensure_dir(directory)
    for name, frame in frames.items():
        write_text(directory / f"{name}.csv", frame.assign(source=s["label"]).to_csv(index=False))
    write_text(directory / "done.json",
               json.dumps({"seconds": time.perf_counter() - start, "sets": list(s["sets"]),
                           "tau": s["tau"]}, indent=1))
    return puma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="φ source keyed by SERIALNO")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--puma", nargs="+", default=None, help="PUMA GEOIDs (default: all)")
    parser.add_argument("--shard", default=None, help="i/n: every n-th PUMA from the i-th (1-based)")
    parser.add_argument("--sets", nargs="+", default=["S0", "S1", "S2"])
    parser.add_argument("--tau", type=float, default=3.0, help="prior sd of β")
    parser.add_argument("--workers", type=int, default=1, help="PUMAs run at once")
    args = parser.parse_args()

    label = args.label or args.run.name
    root = ensure_dir(args.out / label)
    record_run(root, args)
    config = load_config(args.config)
    data = load_check_data(config)
    with stage(f"read positions {args.run}"):
        phi, columns, binary = load_phi(args.run, data.units.index)
    write_text(root / "phi_columns.json", json.dumps({"run": str(args.run), "columns": columns}, indent=1))
    published = published_tables(config.area)

    pumas = args.puma or data.pumas
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        pumas = pumas[i - 1::n]
    logger.info("%d PUMA(s): %s", len(pumas), " ".join(pumas))
    _SHARED.update(data=data, phi=phi, binary=binary, published=published, root=root,
                   label=label, tau=args.tau, sets=tuple(args.sets))
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for done in pool.imap_unordered(run_puma, pumas):
                logger.info("PUMA %s finished", done)
    else:
        for puma in pumas:
            run_puma(puma)


if __name__ == "__main__":
    main()
