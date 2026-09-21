"""ACS Public Use Microdata Sample (PUMS) person and housing records.

Obtained as bulk published files rather than through the Microdata API: there
is no key or query-count limit, and a whole state's records arrive in a single
request.

The 5-year tree was confirmed against a live listing: one flat directory per
vintage holding one zip per state, named by lowercase postal abbreviation --
``csv_p<st>.zip`` for person records and ``csv_h<st>.zip`` for housing. Note
the capitalised ``5-Year`` here, against the lowercase ``5-year`` used in the
variance replicate tree.

The 2020-2024 data dictionary codes PUMA in a single five-character ``PUMA``
column on the 2020 Census definition, to be combined with ``ST`` for a unique
code. There is no ``PUMA10``/``PUMA20`` split to reconcile, which is what lets
:mod:`pmedm_vb.data.geography` work from the 2020 crosswalk alone. A vintage
spanning the PUMA redraw may not be so simple; check the dictionary for that
vintage before assuming it is.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.cache import fetch_cached

PUMS_BASE = "https://www2.census.gov/programs-surveys/acs/data/pums"

RECORD_TYPES = {"person": "p", "housing": "h"}

#: Columns kept regardless of what the caller asks for. ``SERIALNO`` joins
#: person records to their housing record, ``SPORDER`` identifies the person
#: within it, and ``ST`` plus ``PUMA`` place the record geographically.
IDENTIFIER_COLUMNS = {
    "person": ("SERIALNO", "SPORDER", "ST", "PUMA"),
    "housing": ("SERIALNO", "ST", "PUMA"),
}

#: Base weight column by record type; the replicate weights append 1-80.
WEIGHT_PREFIXES = {"person": "PWGTP", "housing": "WGTP"}

#: Read as text so that leading zeros survive. PUMS writes ``SERIALNO`` as a
#: string in any case (``"2020GQ0000001"``), but ``ST`` and ``PUMA`` are
#: numeric-looking and would silently lose theirs.
TEXT_COLUMNS = ("RT", "SERIALNO", "ST", "PUMA")


def weight_columns(record_type: str) -> tuple[str, ...]:
    """Base weight followed by the 80 replicate weights."""
    prefix = WEIGHT_PREFIXES[record_type]
    return (prefix, *(f"{prefix}{i}" for i in range(1, 81)))


def pums_url(area: StudyArea, *, record_type: str = "person") -> str:
    """URL of the bulk PUMS file covering ``area``."""
    try:
        letter = RECORD_TYPES[record_type]
    except KeyError:
        raise ValueError(
            f"record_type must be one of {sorted(RECORD_TYPES)}, got {record_type!r}"
        ) from None
    state = area.state_abbrev.lower()
    return f"{PUMS_BASE}/{area.year}/{area.span}-Year/csv_{letter}{state}.zip"


def download_pums(
    area: StudyArea,
    *,
    record_type: str = "person",
    force: bool = False,
) -> Path:
    """Download the raw PUMS file covering ``area`` and return its cached path.

    Parameters
    ----------
    area:
        Study area; supplies the state and the ACS vintage.
    record_type:
        ``"person"`` or ``"housing"``.
    force:
        Re-download even when the file is already cached.
    """
    return fetch_cached(
        pums_url(area, record_type=record_type),
        subdir=f"pums/{area.year}-{area.span}yr",
        force=force,
    )


def load_pums(
    area: StudyArea,
    variables: list[str],
    *,
    record_type: str = "person",
) -> pd.DataFrame:
    """Read the cached PUMS file, keeping ``variables`` plus the identifiers.

    The identifier columns (the record serial number, the person number for
    person records, the state, the PUMA, and the replicate-weight columns) are
    always retained regardless of ``variables``, since the assembly step needs
    them to build the design weights and the PUMA-to-zone allocation.

    A ``puma_geoid`` column is added: ``ST`` and ``PUMA`` concatenated, which
    is the form :mod:`pmedm_vb.data.geography` matches zones on.
    """
    path = download_pums(area, record_type=record_type)
    wanted = list(
        dict.fromkeys(
            [*IDENTIFIER_COLUMNS[record_type], *weight_columns(record_type), *variables]
        )
    )
    dtypes = {column: str for column in TEXT_COLUMNS}

    frames = []
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV member in {path}")
        for member in sorted(members):
            with archive.open(member) as handle:
                frames.append(
                    pd.read_csv(
                        handle,
                        usecols=lambda column: column in wanted,
                        dtype=dtypes,
                        low_memory=False,
                    )
                )
    frame = pd.concat(frames, ignore_index=True)

    missing = [column for column in wanted if column not in frame.columns]
    if missing:
        raise KeyError(f"columns absent from {path.name}: {missing}")

    # Widths are fixed by the data dictionary; pad defensively in case a
    # vintage publishes them unquoted and a reader drops the leading zero.
    frame["ST"] = frame["ST"].str.zfill(2)
    frame["PUMA"] = frame["PUMA"].str.zfill(5)
    frame["puma_geoid"] = frame["ST"] + frame["PUMA"]
    return frame
