"""ACS Public Use Microdata Sample (PUMS) person and housing records.

Obtained as bulk published files rather than through the Microdata API: there
is no key or query-count limit, and a whole state's person records arrive in a
single request.

The Census Bureau documents PUMS file access at
https://www.census.gov/programs-surveys/acs/microdata/access.html. The exact
directory layout and per-state filename pattern on the FTP site were *not*
confirmed while this module was written -- census.gov was unreachable from the
development sandbox -- so both need checking against a live listing before
:func:`download_pums` is filled in.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pmedm_vb.config import StudyArea


def download_pums(area: StudyArea, *, record_type: str = "person") -> Path:
    """Download the raw PUMS file covering ``area`` and return its cached path.

    Parameters
    ----------
    area:
        Study area; supplies the state FIPS code and the ACS vintage.
    record_type:
        ``"person"`` or ``"housing"``.
    """
    raise NotImplementedError


def load_pums(
    area: StudyArea,
    variables: list[str],
    *,
    record_type: str = "person",
) -> pd.DataFrame:
    """Read the cached PUMS file, keeping ``variables`` plus the identifiers.

    The identifier columns (the record serial number, the person number for
    person records, the PUMA, and the replicate-weight columns) are always
    retained regardless of ``variables``, since the assembly step needs them to
    build the design weights and the PUMA-to-zone allocation.
    """
    raise NotImplementedError
