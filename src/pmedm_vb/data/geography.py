"""Crosswalks between block groups, tracts and PUMAs.

PMEDM needs two distinct geographic relationships:

* **Nesting**, block group into tract, which builds the aggregation operators
  ``A_B`` and ``A_T`` mapping zones to constraint rows. Census GEOIDs nest by
  construction, so this is string slicing rather than a spatial join.
* **PUMA membership** for each zone, which decides *which* PUMS records may
  receive weight in a given zone and so sets the sparsity of the design
  weights ``q``. PUMA boundaries do not nest within tracts, and PUMA
  definitions change between vintages, so this needs a published relationship
  file rather than GEOID arithmetic.
"""

from __future__ import annotations

import pandas as pd

from pmedm_vb.config import StudyArea


def block_groups(area: StudyArea) -> pd.DataFrame:
    """Return the block group GEOIDs in ``area``, with their parent tract GEOID."""
    raise NotImplementedError


def tracts(area: StudyArea) -> pd.DataFrame:
    """Return the tract GEOIDs in ``area``."""
    raise NotImplementedError


def puma_crosswalk(area: StudyArea) -> pd.DataFrame:
    """Map each block group in ``area`` to the PUMA containing it.

    The vintage matters: PUMA definitions are redrawn after each decennial
    census, and an ACS period spanning a redraw publishes PUMS coded to a
    definition that may not match the one the summary-file geographies use.
    Which relationship file is correct for a given ``area.year`` needs
    confirming against Census documentation before this is filled in.
    """
    raise NotImplementedError
