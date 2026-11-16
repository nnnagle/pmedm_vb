"""Leave-one-PUMA-out check of an embedding (see :mod:`pmedm_vb.embed.puma_check`).

    python experiments/embed/puma_check.py --config experiments/embed/stage1_v2.json \\
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d8_g0.5_r0.01 \\
        --out $PMEDM_VB_RUNS/embed/puma_check [--puma 4701502 4702300] [--shard 1/3]

``--run`` is any φ source keyed by ``SERIALNO``: a fit directory (its
``phi_units.parquet``) or a parquet/CSV file; ``--label`` names it in the
output (default: the run directory's name). ``--h`` picks the feature maps
(``linear``, ``quadratic``, ``decoder``; the decoder map needs a fit directory)
and ``--tilts`` the tilts (``none x phi both oracle``); ``phi``, ``both`` and
``oracle`` are run once per h, labelled with h's suffix (``phi_quad``...).
``--alpha`` and ``--tau`` (every α with every τ) or ``--grid a:t ...``
(``--grid default`` for the standard grid) give
the penalty settings of the fitted tilts ``x``, ``phi``, ``both``: prior sd τ
on β and the entropy penalty α n KL(p ‖ q) (:mod:`pmedm_vb.embed.tilt`); each
setting is labelled ``_a<α>_t<τ>`` except α = 0, τ = 3, the earlier results.
The oracles do not depend on them and are not refitted. Default sets: S0 S1.

Results go to ``<out>/<label>/<puma>/`` (``scores.csv``, ``misfit.csv``,
``rare.csv``), and ``done.json`` records which ``tilt|set`` pairs are done.
Only pairs not yet done are fitted, and their rows are appended, so new h can
be added to an existing output (e.g. ``--h quadratic decoder --tilts phi both
oracle`` beside a finished linear run) and a timed-out job can be resubmitted
as it was. A ``done.json`` from before feature maps counts as every linear
tilt done for its sets. ``--shard i/n`` takes every n-th PUMA starting at the
i-th; ``--workers k`` runs k PUMAs at once in forked processes that share the
loaded data (the S2 ``x``/``both`` fits dominate: minutes per PUMA on one
core).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from pathlib import Path

from pmedm_vb.config import ensure_dir
from pmedm_vb.embed.config import load_config
import pandas as pd

from pmedm_vb.embed.puma_check import (
    DECODER_ATTRIBUTES, DEFAULT_PENALTY, FEATURE_MAPS, FITTED_TILTS, PENALTY_GRID, TILTS,
    check_puma,
    feature_maps, load_check_data, load_phi, published_tables, tilt_label,
)
from pmedm_vb.progress import logger, record_run, stage, write_text

#: Shared with forked workers: set once in the parent before the pool starts.
_SHARED: dict = {}


def done_keys(directory: Path) -> set[str]:
    """``"tilt|set"`` pairs already done for one PUMA. A ``done.json`` without
    ``keys`` predates feature maps: every linear tilt was run for its sets."""
    path = directory / "done.json"
    if not path.exists():
        return set()
    done = json.loads(path.read_text())
    if "keys" in done:
        return set(done["keys"])
    return {f"{t}|{s}" for t in TILTS for s in done.get("sets", [])}


def requested_keys(maps, tilts, sets, penalties=(DEFAULT_PENALTY,)) -> set[str]:
    labels = {tilt_label(t, h, pen) for t in tilts for h in maps
              for pen in (penalties if t in FITTED_TILTS else [DEFAULT_PENALTY])}
    return {f"{label}|{s}" for label in labels for s in sets}


def parse_penalties(alpha, tau, grid) -> tuple[tuple[float, float], ...]:
    """(α, τ) pairs: ``--grid a:t ...`` if given (``default`` adds
    :data:`~pmedm_vb.embed.puma_check.PENALTY_GRID`), else every α with every τ."""
    if grid:
        pairs = []
        for item in grid:
            if item == "default":
                pairs.extend(PENALTY_GRID)
                continue
            a, _, t = item.partition(":")
            if not t:
                raise SystemExit(f"--grid takes alpha:tau pairs, got {item!r}")
            pairs.append((float(a), float(t)))
    else:
        pairs = [(float(a), float(t)) for a in alpha for t in tau]
    if any(a < 0 or t <= 0 for a, t in pairs):
        raise SystemExit(f"need alpha >= 0 and tau > 0: {pairs}")
    return tuple(dict.fromkeys(pairs))


def run_puma(puma: str) -> str:
    """Check one PUMA, fitting only the tilt/set pairs not yet done, and
    append their rows to its files."""
    s = _SHARED
    directory = s["root"] / puma
    done = done_keys(directory)
    if requested_keys(s["maps"], s["tilts"], s["sets"], s["penalties"]) <= done:
        logger.info("PUMA %s: already done, skipped", puma)
        return puma
    start = time.perf_counter()
    with stage(f"PUMA {puma}"):
        frames = check_puma(s["data"], puma, s["maps"], s["published"],
                            sets=s["sets"], tilts=s["tilts"], skip=frozenset(done),
                            solver=s["solver"], penalties=s["penalties"])
    ensure_dir(directory)
    new = {f"{t}|{st}" for t, st in zip(frames["misfit"]["tilt"], frames["misfit"]["set"])} \
        if len(frames["misfit"]) else set()
    for name, frame in frames.items():
        frame = frame.assign(source=s["label"])
        path = directory / f"{name}.csv"
        if path.exists():
            # Rows of pairs being written now are replaced, not duplicated (a
            # run killed between these files and done.json leaves them behind).
            old = pd.read_csv(path, dtype={"puma": str})
            old = old[~(old["tilt"].astype(str) + "|" + old["set"].astype(str)).isin(new)]
            frame = pd.concat([old, frame], ignore_index=True)
        write_text(path, frame.to_csv(index=False))
    previous = json.loads((directory / "done.json").read_text()) if (directory / "done.json").exists() else {}
    write_text(directory / "done.json", json.dumps({
        "keys": sorted(done | new),
        "penalties": sorted({tuple(x) for x in previous.get("penalties", [list(DEFAULT_PENALTY)])}
                            | set(s["penalties"])),
        "seconds": previous.get("seconds", 0.0) + time.perf_counter() - start,
        "h": sorted(set(previous.get("h", ["linear"] if done else [])) | set(s["maps"])),
        "decoder_attributes": s["decoder_attributes"]}, indent=1))
    return puma


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, help="φ source keyed by SERIALNO")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--label", default=None)
    parser.add_argument("--puma", nargs="+", default=None, help="PUMA GEOIDs (default: all)")
    parser.add_argument("--shard", default=None, help="i/n: every n-th PUMA from the i-th (1-based)")
    parser.add_argument("--sets", nargs="+", default=["S0", "S1"],
                        help="constraint sets (default S0 S1; S2's fits are slow)")
    parser.add_argument("--tau", type=float, nargs="+", default=[3.0],
                        help="prior sd(s) of β for the fitted tilts")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0],
                        help="entropy weight(s): alpha * n * KL(p||q), n = P's sample units; "
                             "every alpha is combined with every tau")
    parser.add_argument("--grid", nargs="+", default=None, metavar="ALPHA:TAU",
                        help="explicit (alpha, tau) pairs instead of the alpha x tau product; "
                             "'default' is " + " ".join(f"{a:g}:{t:g}" for a, t in PENALTY_GRID))
    parser.add_argument("--workers", type=int, default=1, help="PUMAs run at once")
    parser.add_argument("--h", nargs="+", default=["linear"], choices=list(FEATURE_MAPS),
                        help="feature maps for the phi, both and oracle tilts")
    parser.add_argument("--tilts", nargs="+", default=list(TILTS), choices=list(TILTS))
    parser.add_argument("--solver", default="auto", choices=["auto", "lbfgs", "lm"],
                        help="auto: L-BFGS for x and linear tilts, Levenberg-Marquardt for "
                             "quadratic and decoder ones")
    parser.add_argument("--decoder-attributes", nargs="+", default=list(DECODER_ATTRIBUTES),
                        help="attributes of the decoder feature map ('all' for every target)")
    args = parser.parse_args()

    label = args.label or args.run.name
    root = ensure_dir(args.out / label)
    record_run(root, args)
    config = load_config(args.config)
    data = load_check_data(config)
    with stage(f"read positions {args.run}"):
        phi, columns, binary = load_phi(args.run, data.units.index)
    with stage(f"feature maps {' '.join(args.h)}"):
        maps = feature_maps(tuple(args.h), phi, columns, binary, run=args.run,
                            weight=data.weight, decoder_attributes=tuple(args.decoder_attributes))
    record = root / "phi_columns.json"
    previous = json.loads(record.read_text()) if record.exists() else {}
    write_text(record, json.dumps(
        {"run": str(args.run), "columns": columns,
         "feature_maps": {**previous.get("feature_maps", {}),
                          **{h: m.columns for h, m in maps.items()}}}, indent=1))
    published = published_tables(config.area)

    pumas = args.puma or data.pumas
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        pumas = pumas[i - 1::n]
    logger.info("%d PUMA(s): %s", len(pumas), " ".join(pumas))
    penalties = parse_penalties(args.alpha, args.tau, args.grid)
    logger.info("penalty settings (alpha, tau): %s", penalties)
    _SHARED.update(data=data, maps=maps, published=published, root=root, label=label,
                   penalties=penalties, sets=tuple(args.sets), tilts=tuple(args.tilts),
                   decoder_attributes=list(args.decoder_attributes), solver=args.solver)
    if args.workers > 1:
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for done in pool.imap_unordered(run_puma, pumas):
                logger.info("PUMA %s finished", done)
    else:
        for puma in pumas:
            run_puma(puma)


if __name__ == "__main__":
    main()
