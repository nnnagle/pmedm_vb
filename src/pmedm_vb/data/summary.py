"""ACS summary-table estimates: the constraint targets ``Y``.

The access route is **deliberately undecided**. Two are viable and the choice
depends on whether a full covariance is wanted:

* The Microdata/detailed-table API (``api.census.gov/data/<year>/acs/acs5/...``)
  returns exactly the variables and geographies asked for, but does not serve
  the variance replicate tables.
* The published table files carry the same estimates, and for the subset of
  tables covered by :mod:`pmedm_vb.data.variance` they carry the estimate, the
  margin of error and the 80 replicates together -- so if a non-diagonal
  ``Sigma`` is wanted, that one download supplies both ``Y`` and ``Sigma`` and
  the API adds nothing.

This module therefore exposes a single entry point. Settling the route changes
only the body of :func:`fetch_summary_tables`, never its callers.
"""

from __future__ import annotations

import pandas as pd

from pmedm_vb.config import StudyArea


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
        One row per geography, indexed by GEOID, with a column per table cell
        and a parallel column of margins of error.
    """
    raise NotImplementedError
