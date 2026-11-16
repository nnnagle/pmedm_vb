"""Draw the PUMA cluster map from a diagnostic's ``clusters.csv`` alone.

    python experiments/embed/puma_cluster_map.py --clusters $OUT/clusters.csv --k 5 \\
        --out clusters_k5.png [--boundaries tl_2024_47_puma20.zip] [--state 47]

Needs only numpy, pandas and matplotlib: the boundary shapefile is read by
:func:`pmedm_vb.embed.puma_map.read_boundaries`. Without ``--boundaries`` the
cached TIGER file for ``--state`` is used (fetched by the embedding prefetch).
``--spaces`` picks the panels (default: tables, each z and SVD by energy
distance, and the joint structure).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.embed.puma_map import (
    cached_boundaries, map_panels, plot_cluster_maps, read_boundaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--clusters", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--boundaries", type=Path, default=None, help="TIGER PUMA zip")
    parser.add_argument("--state", default="47")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--spaces", nargs="+", default=None)
    args = parser.parse_args()

    clusters = pd.read_csv(args.clusters, dtype={"puma": str})
    path = args.boundaries or cached_boundaries(
        StudyArea(name="state", state=args.state, year=args.year))
    boundaries = read_boundaries(path)
    spaces, titles = map_panels(clusters["space"].unique())
    if args.spaces:
        spaces = args.spaces
    print(plot_cluster_maps(boundaries, clusters, spaces, args.k, args.out, titles))


if __name__ == "__main__":
    main()
