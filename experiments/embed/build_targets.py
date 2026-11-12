"""Build (or rebuild) and cache the embedding's target arrays.

    python experiments/embed/build_targets.py --config experiments/embed/stage1_v1.json

Reads only cached PUMS files, so it runs on a compute node after the prefetch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.units import load_targets, targets_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rebuild", action="store_true", help="ignore an existing cache")
    args = parser.parse_args()
    config = load_config(args.config)
    arrays = load_targets(config, rebuild=args.rebuild)
    meta = arrays.meta
    print(f"{targets_dir(config)}\n{meta['n_units']:,} units ({meta['n_group_quarters']:,} "
          f"group quarters) -> {meta['n_rows']:,} merged rows; {meta['n_categories']} "
          f"categories in {len(meta['targets'])} targets")


if __name__ == "__main__":
    main()
