#!/usr/bin/env python
"""Check every constraint definition at once, against the published numbers.

The end-to-end test of the PUMS side. A weighted PUMS total is compared with
the published estimate for the same area and constraint, which exercises the
whole mapping chain in one pass: the ``RAC1P`` collapse onto ``B03002``'s seven
race categories, ``HISP`` compared as a string, ``JWTRNS`` against ``B08301``'s
subtotals, the ``OCCP`` prefix crosswalk and ``ESR`` universe behind ``C24010``,
the ``POVPIP`` bands of ``B17024``, the two group shapes of ``B23001``, the
``ADJINC`` multiply in ``B19001``, and the ``WGTP``/``PWGTP`` split for group
quarters.

**Read the standardised gap, not the ratio.** A ratio is meaningless on a small
cell -- 14 against 54 looks catastrophic and is one sample record. The gap is
divided by the published estimate's own standard error, taken from the replicate
deviations so that the covariance *between* the areas being summed is included;
a sum of per-area margins of error would overstate it.

How to read the result:

* A whole table with ``mean z`` well away from zero is a definition error. The
  mapping for that table is systematically wrong.
* Scattered large ``|z|`` with ``mean z`` near zero is sampling.
* A mild negative bias across every *person*-level table, with the
  household-level ones near zero, is expected and benign: person totals are
  estimated as ``WGTP x (persons in unit)``, and ACS calibrates household and
  person weights separately. ``q`` is a prior, and the solver reweights.
* ``B26001`` runs about 11 percent low by construction. PUMS weights pin
  household population exactly and leave group quarters unpinned -- see
  ``census_data_sources.md``. The constraint is what corrects it.

``z`` standardises by the published error only; the PUMS estimate has its own,
roughly ``1/sqrt(records)`` on a total. So the real spread is wider than ``z``
implies and ``|z|`` up to about 3 on a small cell is unremarkable. This is a
screen for finding a broken mapping, not a hypothesis test.

Usage::

    $CONDA_PREFIX/bin/python tools/check_mapping.py [PUMA] [STATE] [COUNTY]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from pmedm_vb.assemble.constraints import default_tables
from pmedm_vb.assemble.households import build_units, person_counts
from pmedm_vb.assemble.targets import build_targets
from pmedm_vb.config import StudyArea
from pmedm_vb.data.geography import puma_crosswalk_whole
from pmedm_vb.data.variance import SDR_FACTOR


def check(area: StudyArea, puma: str, geography: str = "tract") -> pd.DataFrame:
    """Per-constraint PUMS total, published total, standard error and ``z``."""
    tables = default_tables(area)
    at_level = [t for t in tables if t.geography == geography]

    units = build_units(area, puma, tables)
    estimate = person_counts(units, at_level).mul(units.units["weight"], axis=0).sum()

    zones = puma_crosswalk_whole(area)
    own = zones[zones["puma_geoid"] == puma]
    column = "tract_geoid" if geography == "tract" else "block_group_geoid"
    block = build_targets(
        area, at_level, geography=geography, geoids=sorted(own[column].unique())
    )

    rows = []
    for position, name in enumerate(block.names):
        # Column-major: constraint `position` occupies rows
        # position*n_areas .. (position+1)*n_areas. Summing those deviations
        # gives the variance of the summed estimate, covariance included.
        start = position * block.n_areas
        deviations = block.L[start : start + block.n_areas].sum(axis=0)
        rows.append(
            (
                name,
                float(estimate.get(name, np.nan)),
                float(block.Y[:, position].sum()),
                float(np.sqrt(SDR_FACTOR * np.square(deviations).sum())),
            )
        )

    frame = pd.DataFrame(rows, columns=["name", "pums", "published", "se"])
    frame = frame.set_index("name")
    frame["z"] = (frame["pums"] - frame["published"]) / frame["se"]
    frame["table"] = [name.split(".")[0] for name in frame.index]
    return frame


def main() -> int:
    puma = sys.argv[1] if len(sys.argv) > 1 else "4701501"
    state = sys.argv[2] if len(sys.argv) > 2 else "47"
    county = sys.argv[3] if len(sys.argv) > 3 else "093"
    area = StudyArea(name="check", state=state, year=2024, counties=(county,))

    frame = check(area, puma)
    worst = frame.reindex(frame["z"].abs().sort_values(ascending=False).index)
    print(f"PUMA {puma}: largest standardised gaps\n")
    print(
        worst.head(12)[["pums", "published", "se", "z"]].to_string(
            float_format=lambda v: f"{v:,.1f}"
        )
    )
    print("\nBy table -- off-centre mean means a definition error:\n")
    print(
        frame.groupby("table")["z"]
        .agg(["count", "mean", "median", lambda s: s.abs().max()])
        .rename(columns={"<lambda_0>": "max|z|"})
        .to_string(float_format=lambda v: f"{v:.2f}")
    )
    beyond = int((frame["z"].abs() > 2).sum())
    print(
        f"\noverall: mean z {frame['z'].mean():.2f}, "
        f"median {frame['z'].median():.2f}, {beyond} of {len(frame)} beyond 2 se"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
