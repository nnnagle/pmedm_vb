"""ACS summary-table estimates: the constraint targets ``Y``.

The access route is settled: the estimates come from the Variance Replicate
Estimate Tables, which carry the estimate, the margin of error and the 80
replicates in one file. For any table those tables cover, a separate summary
download would add nothing, and the API cannot supply the replicates a
non-diagonal ``Sigma`` needs.

This module is therefore a view over :mod:`pmedm_vb.data.variance` rather than
a second downloader, reshaping the tidy replicate rows into the zone-by-cell
layout the assembly step wants.

The route's limit is coverage, not mechanism: 131 tables at tract and 73 at
block group for 2019-2023. A constraint outside that set has to come from the
detailed-table API with an MOE-derived diagonal, and adding that would mean a
second implementation behind this same entry point -- which is why callers get
one function rather than a choice of route.
"""

from __future__ import annotations

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.variance import fetch_replicates


def fetch_summary_tables(
    area: StudyArea,
    tables: list[str],
    *,
    geography: str,
) -> pd.DataFrame:
    """Return estimates and margins of error for ``tables`` over ``area``.

    Parameters
    ----------
    area:
        Study area; supplies the state, counties and vintage.
    tables:
        ACS detailed-table IDs, e.g. ``["B01001", "B19001"]``.
    geography:
        ``"tract"`` or ``"block group"``.

    Returns
    -------
    pandas.DataFrame
        One row per geography, indexed by GEOID, with two-level columns: level
        zero is ``"estimate"`` or ``"moe"`` and level one is the cell key, so
        that ``frame["estimate"]`` is the zone-by-cell target matrix directly
        and ``frame["moe"]`` is its parallel of margins.
    """
    frame = fetch_replicates(area, tables, geography=geography)
    wide = frame[["estimate", "moe"]].unstack("cell")
    return wide.sort_index(axis="columns")


def cell_labels(area: StudyArea, tables: list[str], *, geography: str) -> pd.Series:
    """Human-readable label per cell key, for checking against PUMS categories.

    Matching constraint cells to the PUMS columns that feed them is the fiddliest
    part of setting up a run and the usual source of silent misfit, so the
    published cell titles are worth carrying alongside the numbers.
    """
    frame = fetch_replicates(area, tables, geography=geography)
    labels = frame["title"].groupby(level="cell").first()
    return labels.rename("title")
