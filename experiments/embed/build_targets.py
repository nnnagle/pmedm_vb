"""Build (or rebuild) and cache the embedding's target arrays.

    python experiments/embed/build_targets.py --config experiments/embed/stage1_v2.json

Reads only cached PUMS files, so it runs on a compute node after the prefetch.

``--check-refinement`` (attribute targets, v2) also checks, on the statewide
PUMS, that every attribute partitions its universe and that every carried
table's categories are unions of its attributes' categories
(:func:`~pmedm_vb.embed.attribute_targets.refinement_violations`), and writes
both reports beside the arrays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pmedm_vb.embed.config import load_config, resolve
from pmedm_vb.embed.units import load_targets, targets_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rebuild", action="store_true", help="ignore an existing cache")
    parser.add_argument("--check-refinement", action="store_true",
                        help="check attribute partitions and table refinement (v2)")
    args = parser.parse_args()
    config = load_config(args.config)
    arrays = load_targets(config, rebuild=args.rebuild)
    meta = arrays.meta
    print(f"{targets_dir(config)}\n{meta['n_units']:,} units ({meta['n_group_quarters']:,} "
          f"group quarters) -> {meta['n_rows']:,} merged rows; {meta['n_categories']} "
          f"categories in {len(meta['targets'])} targets")
    if args.check_refinement:
        check_refinement(config)


def check_refinement(config) -> None:
    import pandas as pd

    from pmedm_vb.assemble.households import build_units
    from pmedm_vb.embed.attribute_targets import (
        ATTRIBUTE_TARGETS, partition_report, refinement_violations,
    )

    attrs = resolve(config.area, [i for i in config.targets.ids if i in ATTRIBUTE_TARGETS])
    tables = resolve(config.area, config.table_ids)
    frame = build_units(config.area, None, [*attrs.values(), *tables.values()])
    partition = partition_report(frame.units, frame.persons, list(attrs), attrs)
    refinement = refinement_violations(frame.units, frame.persons, tables, attrs)
    directory = targets_dir(config)
    partition.to_csv(directory / "partition.csv", index=False)
    refinement.to_csv(directory / "refinement.csv", index=False)
    with pd.option_context("display.width", 200):
        print(partition.to_string(index=False))
        print(refinement.to_string(index=False))


if __name__ == "__main__":
    main()
