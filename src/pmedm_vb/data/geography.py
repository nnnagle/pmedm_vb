"""Crosswalks between block groups, tracts and PUMAs.

PMEDM needs two distinct geographic relationships:

* **Nesting**, block group into tract, which builds the aggregation operators
  ``A_B`` and ``A_T`` mapping zones to constraint rows. Census GEOIDs nest by
  construction, so this is string slicing rather than a spatial join.
* **PUMA membership** for each zone, which decides *which* PUMS records may
  receive weight in a given zone and so sets the sparsity of the design
  weights ``q``. PUMA boundaries do not nest within tracts, so this needs a
  published relationship file.

**PUMA membership is also what decomposes the problem.** A 2020 PUMA is built
from whole tracts -- which is why the relationship file is a plain 1:1 map
rather than an allocation table -- so block group nests in tract nests in PUMA
with no overlap, and both constraint levels partition by PUMA. Since ``q`` is
zero outside a record's own PUMA, so is every weight, and the joint problem is
block diagonal by PUMA. Assembly therefore builds one problem per PUMA rather
than one per study area.

That makes the *whole* PUMA the unit that matters, not the part inside the
study area: PUMS records represent all of a PUMA, so constraining only part of
one leaves the rest of its population with nowhere to go.
:func:`puma_crosswalk_whole` returns whole PUMAs and :func:`puma_coverage`
reports which of them reach beyond the area.

Both are satisfied without geometry, and therefore without a GIS dependency.
The published tract-to-PUMA file supplies the second relationship directly, and
block groups reach it through their parent tract.

The vintage question the stub raised is settled for 2020-vintage PUMAs: the
2020-2024 PUMS dictionary codes a single ``PUMA`` column on the 2020 Census
definition, which is what ``rel2020`` maps to. A vintage whose PUMS spans the
redraw would need the matching earlier file; check the dictionary for that
vintage rather than assuming this module covers it.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.cache import fetch_cached
from pmedm_vb.data.variance import POPULATION_TABLE, fetch_replicates

#: Published relationship file for 2020-vintage PUMAs. Columns are
#: ``STATEFP,COUNTYFP,TRACTCE,PUMA5CE``; roughly 85,000 national rows.
TRACT_TO_PUMA_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/"
    "2020_Census_Tract_to_2020_PUMA.txt"
)

#: Characters of a block group GEOID that name its parent tract.
TRACT_GEOID_WIDTH = 11


def _tract_to_puma() -> pd.DataFrame:
    """The national tract-to-PUMA table, with GEOIDs assembled.

    Every column is read as text: the components are zero-padded fixed-width
    codes, and inferring them as integers destroys the padding.
    """
    path = fetch_cached(TRACT_TO_PUMA_URL, subdir="geo")
    # The published file begins with a UTF-8 BOM (confirmed by inspecting the
    # bytes). pandas strips it anyway, but utf-8-sig says so rather than
    # relying on that, and is equally correct when the BOM is absent -- if it
    # ever leaked through, the first column would be named "\ufeffSTATEFP"
    # and every lookup here would fail on a name that looks right.
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    frame["tract_geoid"] = frame["STATEFP"] + frame["COUNTYFP"] + frame["TRACTCE"]
    frame["puma_geoid"] = frame["STATEFP"] + frame["PUMA5CE"]
    return frame


def _restrict(frame: pd.DataFrame, area: StudyArea, geoid_column: str) -> pd.DataFrame:
    """Keep rows whose GEOID falls in ``area``'s state and counties."""
    geoids = frame[geoid_column]
    selected = frame[geoids.str[:2] == area.state]
    counties = set(area.county_geoids())
    if counties:
        selected = selected[selected[geoid_column].str[:5].isin(counties)]
    return selected.reset_index(drop=True)


def tracts(area: StudyArea) -> pd.DataFrame:
    """Return the tract GEOIDs in ``area``, with the PUMA containing each.

    Returns
    -------
    pandas.DataFrame
        Columns ``tract_geoid`` and ``puma_geoid``, one row per tract.
    """
    frame = _tract_to_puma()[["tract_geoid", "puma_geoid"]]
    return _restrict(frame, area, "tract_geoid")


def block_groups(area: StudyArea) -> pd.DataFrame:
    """Return the block group GEOIDs in ``area``, with their parent tract GEOID.

    The universe of block groups is taken from the published total-population
    replicate file rather than from a TIGER extract: it is exactly the set of
    block groups the constraint tables are published for, which is the set a
    run can actually use, and it is a file the run downloads anyway to assign
    k-values.
    """
    frame = fetch_replicates(area, [POPULATION_TABLE], geography="block group")
    geoids = frame.index.get_level_values("geoid").unique()
    result = pd.DataFrame({"block_group_geoid": geoids})
    result["tract_geoid"] = result["block_group_geoid"].str[:TRACT_GEOID_WIDTH]
    return result.sort_values("block_group_geoid").reset_index(drop=True)


def puma_crosswalk(area: StudyArea) -> pd.DataFrame:
    """Map each block group in ``area`` to the PUMA containing it.

    Returns
    -------
    pandas.DataFrame
        Columns ``block_group_geoid``, ``tract_geoid`` and ``puma_geoid``.

    Raises
    ------
    KeyError
        If a block group's parent tract is absent from the relationship file,
        which would mean the tract vintage and the PUMA vintage disagree.
    """
    zones = block_groups(area)
    lookup = _tract_to_puma().set_index("tract_geoid")["puma_geoid"]
    zones["puma_geoid"] = zones["tract_geoid"].map(lookup)

    unmatched = zones.loc[zones["puma_geoid"].isna(), "tract_geoid"].unique()
    if len(unmatched):
        raise KeyError(
            f"{len(unmatched)} tract(s) absent from the PUMA relationship file, "
            f"e.g. {sorted(unmatched)[:5]} -- check that the file vintage matches "
            f"the {area.year} {area.span}-year geographies"
        )
    return zones


def puma_geoids(area: StudyArea) -> tuple[str, ...]:
    """PUMA GEOIDs that any tract in the study area belongs to.

    A PUMA is built from whole tracts, so "touches the area" and "contains a
    tract of the area" are the same question.
    """
    inside = _restrict(_tract_to_puma(), area, "tract_geoid")
    return tuple(sorted(inside["puma_geoid"].unique()))


def puma_coverage(area: StudyArea) -> pd.DataFrame:
    """How much of each touched PUMA lies inside the study area.

    A PUMA reaching beyond the area is the case that makes a per-PUMA run
    ill-posed: its PUMS records represent the whole PUMA, so constraining only
    the part inside leaves the rest of its population with nowhere to go.

    Returns
    -------
    pandas.DataFrame
        One row per PUMA, indexed by ``puma_geoid``, with the tract counts
        inside and in total, whether it is wholly contained, and the counties
        it spans.
    """
    national = _tract_to_puma()
    inside = _restrict(national, area, "tract_geoid")
    counted = inside.groupby("puma_geoid")["tract_geoid"].nunique()

    rows = []
    for puma in counted.index:
        tracts = national[national["puma_geoid"] == puma]
        rows.append(
            {
                "puma_geoid": puma,
                "tracts_in_area": int(counted[puma]),
                "tracts_total": int(tracts["tract_geoid"].nunique()),
                "counties": tuple(sorted(tracts["tract_geoid"].str[:5].unique())),
            }
        )
    frame = pd.DataFrame(rows).set_index("puma_geoid")
    frame["whole"] = frame["tracts_in_area"] == frame["tracts_total"]
    return frame


def puma_crosswalk_whole(area: StudyArea) -> pd.DataFrame:
    """Every block group of every PUMA the study area touches.

    The difference from :func:`puma_crosswalk` is what happens at the edge.
    That function returns the area's own block groups; this one returns whole
    PUMAs, which may reach into neighbouring counties. Per-PUMA assembly needs
    the latter: the PUMS records for a PUMA represent all of it, so all of its
    zones have to be available to receive weight or the population does not
    allocate.

    The block group universe still comes from the published total-population
    file, so it remains exactly the set constraints exist for -- but fetched
    statewide rather than for the area's counties, since a PUMA may cross a
    county line and PUMAs never cross a state one.

    Returns
    -------
    pandas.DataFrame
        Columns ``block_group_geoid``, ``tract_geoid``, ``puma_geoid``.
    """
    targets = set(puma_geoids(area))
    statewide = replace(area, counties=())

    frame = fetch_replicates(statewide, [POPULATION_TABLE], geography="block group")
    geoids = frame.index.get_level_values("geoid").unique()
    zones = pd.DataFrame({"block_group_geoid": geoids})
    zones["tract_geoid"] = zones["block_group_geoid"].str[:TRACT_GEOID_WIDTH]

    lookup = _tract_to_puma().set_index("tract_geoid")["puma_geoid"]
    zones["puma_geoid"] = zones["tract_geoid"].map(lookup)

    unmatched = zones.loc[zones["puma_geoid"].isna(), "tract_geoid"].unique()
    if len(unmatched):
        raise KeyError(
            f"{len(unmatched)} tract(s) absent from the PUMA relationship file, "
            f"e.g. {sorted(unmatched)[:5]} -- check that the file vintage matches "
            f"the {area.year} {area.span}-year geographies"
        )

    zones = zones[zones["puma_geoid"].isin(targets)]
    return zones.sort_values("block_group_geoid").reset_index(drop=True)
