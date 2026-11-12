"""Prefetch everything the embedding reads, on a login or data transfer node.

ISAAC's compute nodes cannot reach census.gov, so this runs first, on the login
node or dtn1/dtn2, with the Census API key loaded::

    . ~/.config/census/env
    $CONDA_PREFIX/bin/python experiments/embed/prefetch.py \\
        --config experiments/embed/stage1_v2.json

``--no-values`` skips the API estimates (PUMA-level, for scoring), which are
the only part that needs the key. See :mod:`pmedm_vb.embed.prefetch`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pmedm_vb.embed.config import load_config
from pmedm_vb.embed.prefetch import prefetch_embedding

DEFAULT_CONFIG = Path(__file__).with_name("stage1_v2.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help=f"embedding configuration (default {DEFAULT_CONFIG.name})")
    parser.add_argument("--no-values", action="store_true",
                        help="skip the API estimates and MOEs (no key needed)")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = prefetch_embedding(config, values=not args.no_values)
    print(f"{len(paths)} files cached for {config.name}")


if __name__ == "__main__":
    main()
