"""PUMA neighbour diagnostic and cluster maps (see :mod:`pmedm_vb.embed.puma_neighbors`).

    python experiments/embed/puma_neighbors.py --config experiments/embed/stage1_v2.json \\
        --run $PMEDM_VB_RUNS/embed/stage1_v2/d8_g0.5_r0.01 $PMEDM_VB_RUNS/embed/stage1_v2/d16_g0.5_r0.01 \\
        --out $PMEDM_VB_RUNS/embed/puma_neighbors

``--run`` takes one or more φ sources keyed by ``SERIALNO`` (fit directories or
parquet/CSV files); only the learned ``z`` columns are used. An SVD baseline is
added for each distinct d. Writes, under ``--out``:

* ``summary.txt`` -- the three summary tables printed below;
* ``mantel.csv`` (alignment with the table distance), ``joint.csv``
  (correlation with the joint-structure distance), ``ari.csv`` (cluster
  agreement for each k);
* ``clusters.csv`` -- each PUMA's cluster in each space and k, raw and aligned
  to the table-space clusters (what the map colours by);
* ``distance_<space>.csv`` -- the 58 x 58 distance matrices;
* ``clusters_k<k>.png`` -- the map, if matplotlib and the cached PUMA
  boundaries are available (``experiments/embed/puma_cluster_map.py`` redraws it
  from ``clusters.csv`` alone).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.config import ensure_dir
from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.puma_check import load_check_data, load_phi
from pmedm_vb.embed.puma_map import (
    cached_boundaries, map_panels, plot_cluster_maps, read_boundaries,
)
from pmedm_vb.embed.puma_neighbors import run_diagnostic
from pmedm_vb.progress import logger, record_run, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path, nargs="+",
                        help="φ sources keyed by SERIALNO (fit directories or files)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, nargs="+", default=[4, 5, 6], help="cluster counts")
    parser.add_argument("--map-k", type=int, default=5, help="k for the map")
    parser.add_argument("--sample", type=int, default=500,
                        help="weighted subsample per PUMA for the energy distance")
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-map", action="store_true")
    args = parser.parse_args()

    out = ensure_dir(args.out)
    record_run(out, args)
    config = load_config(args.config)
    data = load_check_data(config)

    runs = {}
    for run in args.run:
        phi, columns, _ = load_phi(run, data.units.index)
        keep = [j for j, c in enumerate(columns) if c.startswith("z")]
        label = run.name if run.is_dir() else run.stem
        runs[label] = phi[:, keep]
        logger.info("%s: d = %d", label, len(keep))

    result = run_diagnostic(data, runs, ks=tuple(args.k), sample=args.sample,
                            permutations=args.permutations, seed=args.seed)

    for space, matrix in result.distances.items():
        name = space.replace(":", "_")
        frame = pd.DataFrame(matrix, index=result.pumas, columns=result.pumas)
        write_text(out / f"distance_{name}.csv", frame.to_csv(index_label="puma"))
    write_text(out / "mantel.csv", result.mantel.to_csv(index=False))
    write_text(out / "joint.csv", result.joint.to_csv(index=False))
    write_text(out / "ari.csv", result.ari.to_csv(index=False))
    write_text(out / "clusters.csv", result.clusters.to_csv(index=False))

    ari = result.ari.pivot(index="space", columns="k")
    ari.columns = [f"{m.replace('ari_vs_', 'ARI vs ')} k={k}" for m, k in ari.columns]
    text = "\n\n".join([
        "1. Alignment with the table distance (Spearman over PUMA pairs, Mantel p)\n"
        + result.mantel.round(3).to_string(index=False),
        "2. Cluster agreement (average linkage; adjusted Rand index)\n"
        + ari.round(3).to_string(),
        "3. Correlation with the joint-structure distance (Spearman, Mantel p)\n"
        + result.joint.round(3).to_string(index=False),
    ])
    write_text(out / "summary.txt", text + "\n")
    print(text)

    if not args.no_map:
        try:
            boundaries = read_boundaries(cached_boundaries(config.area))
            spaces, titles = map_panels(result.distances)
            path = plot_cluster_maps(boundaries, result.clusters, spaces, args.map_k,
                                     out / f"clusters_k{args.map_k}.png", titles)
            print(f"\nmap: {path}")
        except (FileNotFoundError, ImportError) as err:
            logger.warning("map skipped (%s); redraw with puma_cluster_map.py", err)
    print(f"\ntables in {out}")


if __name__ == "__main__":
    main()
