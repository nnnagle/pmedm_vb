"""ACS Variance Replicate Estimate Tables: the raw material for a full ``Sigma``.

These files carry the estimate, the margin of error and 80 variance replicates
for a *selected subset* of the 5-year Detailed Tables, published for areas down
to the block group. The tract and block group files are named by table ID plus
state FIPS code. They are download-only; the API does not serve them. See
https://www.census.gov/programs-surveys/acs/data/variance-tables.html.

Two consequences shape the design:

* Coverage is partial. Any constraint table chosen for a PMEDM run has to be
  checked against the published list, and a run mixing covered and uncovered
  tables can only have a full covariance over the covered block.
* Because the replicate files also contain the estimates, fetching them makes a
  separate summary-table download redundant for those tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea


def available_tables(area: StudyArea) -> list[str]:
    """Return the table IDs published with variance replicates for this vintage."""
    raise NotImplementedError


def fetch_replicates(
    area: StudyArea,
    tables: list[str],
    *,
    geography: str,
) -> pd.DataFrame:
    """Return estimates plus the 80 replicate estimates for ``tables``.

    Returns
    -------
    pandas.DataFrame
        Indexed by GEOID and table cell, with an ``estimate`` column, a
        ``moe`` column, and eighty replicate columns.
    """
    raise NotImplementedError


def covariance_from_replicates(replicates: np.ndarray) -> np.ndarray:
    """Form the design-based covariance from replicate estimates.

    ACS uses the successive-differences replication estimator, whose variance is
    a scaled sum of squared deviations of the replicates from the full-sample
    estimate. The scale factor is given in the documentation accompanying the
    replicate tables (linked from the page above) and must be read from there
    rather than assumed -- it was not verified when this stub was written.
    """
    raise NotImplementedError
