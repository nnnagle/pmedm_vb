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
column on the 2020 Census definition, to be combined with the state code for a
unique code. There is no ``PUMA10``/``PUMA20`` split to reconcile, which is what
lets :mod:`pmedm_vb.data.geography` work from the 2020 crosswalk alone. A
vintage spanning the PUMA redraw may not be so simple; check the dictionary for
that vintage before assuming it is.

That state code is spelled ``STATE`` in the 2020-2024 files, not ``ST``. This
module read ``ST`` until the first run that opened a zip, having taken the name
from documentation rather than from a header -- hence
:data:`STATE_COLUMN_ALIASES`, which resolves it against the file and raises
naming both candidates if neither is there.

A state's archive holds one CSV (``psam_{p,h}{st}.csv``, 241 columns for
housing) beside a README PDF. ``load_pums`` reads every ``.csv`` member and
concatenates, which is correct for one member and would stay correct for
several.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.cache import PUBLISHED_ENCODINGS, fetch_cached

PUMS_BASE = "https://www2.census.gov/programs-surveys/acs/data/pums"

#: The machine-readable data dictionary lives outside the PUMS data tree.
PUMS_DICT_BASE = "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict"

RECORD_TYPES = {"person": "p", "housing": "h"}

#: Name the state column is normalised to, whatever the file calls it.
STATE_COLUMN = "STATE"

#: How the state column has been spelled, in preference order. The 2020-2024
#: files publish ``STATE``; ``ST`` is kept because older vintages used it and
#: this module is parameterised by year. Resolved against the file rather than
#: assumed -- it *was* assumed, and the assumption went unchallenged until the
#: first run that opened a zip, because nothing else reads the header.
STATE_COLUMN_ALIASES = ("STATE", "ST")

#: Columns kept regardless of what the caller asks for. ``SERIALNO`` joins
#: person records to their housing record and ``SPORDER`` identifies the person
#: within it. The state column is handled separately, through
#: :data:`STATE_COLUMN_ALIASES`, since its name varies.
IDENTIFIER_COLUMNS = {
    "person": ("SERIALNO", "SPORDER", "PUMA"),
    "housing": ("SERIALNO", "PUMA"),
}

#: Base weight column by record type; the replicate weights append 1-80.
WEIGHT_PREFIXES = {"person": "PWGTP", "housing": "WGTP"}

#: Always read as text, whatever the dictionary says. PUMS writes ``SERIALNO``
#: as a string in any case (``"2020GQ0000001"``), but the state code and
#: ``PUMA`` are numeric-looking and would silently lose their leading zeros.
#: Requested variables get the same treatment when the dictionary declares them
#: character -- see :func:`text_columns`.
TEXT_COLUMNS = ("RT", "SERIALNO", "PUMA", *STATE_COLUMN_ALIASES)


def weight_columns(record_type: str) -> tuple[str, ...]:
    """Base weight followed by the 80 replicate weights."""
    prefix = WEIGHT_PREFIXES[record_type]
    return (prefix, *(f"{prefix}{i}" for i in range(1, 81)))


def text_columns(area: StudyArea, wanted: list[str]) -> set[str]:
    """Which of ``wanted`` must be read as text rather than inferred.

    :data:`TEXT_COLUMNS` always, plus every requested variable the data
    dictionary declares character (``C``).

    Inferring a character column's type is silently destructive in a way no
    error reports. ``HISP`` is ``C`` width 2, so ``"01"`` -- not Hispanic --
    becomes ``1.0``, and a constraint written as ``HISP != "01"`` is then true
    for every record in the file: wrong numbers, no failure.

    This does *not* address the dictionary's ``b`` codes, which are a separate
    thing and are not fixed by any dtype. ``b`` repeated to the column width
    (``TEN`` ``"b"``, ``JWTRNS`` ``"bb"``, ``OCCP`` ``"bbbb"``) denotes a
    *blank*, and the published field really is empty, so it arrives as ``NaN``
    whatever dtype is asked for. Test those with ``.isna()``; ``TEN == "b"``
    matches nothing.

    The declaration is read rather than hand-listed on purpose: a list has to
    be kept in step with both the constraint set and each vintage's renames,
    which is the same maintenance the ``ST``/``STATE`` rename already defeated.
    """
    declared = variables(area).set_index("variable")["dtype"]
    character = {column for column in wanted if declared.get(column) == "C"}
    return set(TEXT_COLUMNS) | character


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

    Every requested variable the dictionary declares character is read as text;
    see :func:`text_columns` for why that is not optional.

    The state column is normalised to :data:`STATE_COLUMN` whatever the file
    spells it, and a ``puma_geoid`` column is added: state and ``PUMA``
    concatenated, which is the form :mod:`pmedm_vb.data.geography` matches
    zones on.
    """
    path = download_pums(area, record_type=record_type)
    required = list(
        dict.fromkeys(
            [*IDENTIFIER_COLUMNS[record_type], *weight_columns(record_type), *variables]
        )
    )
    # Every alias is read; exactly one is expected back, and which one is a
    # property of the file rather than something the caller should know.
    wanted = [*required, *STATE_COLUMN_ALIASES]
    dtypes = {column: str for column in text_columns(area, wanted)}

    frames = []
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ValueError(f"no CSV member in {path}")
        for member in sorted(members):
            # Same publisher, same encoding question as the replicate files, but
            # these run to hundreds of megabytes so they cannot be read into
            # memory to decode. Re-opening the member and retrying costs nothing
            # in the normal case, where the first attempt succeeds.
            for encoding in PUBLISHED_ENCODINGS:
                try:
                    with archive.open(member) as handle:
                        frames.append(
                            pd.read_csv(
                                handle,
                                usecols=lambda column: column in wanted,
                                dtype=dtypes,
                                low_memory=False,
                                encoding=encoding,
                            )
                        )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(
                    PUBLISHED_ENCODINGS[-1], b"", 0, 1,
                    f"none of {list(PUBLISHED_ENCODINGS)} decodes {path}::{member}",
                )
    frame = pd.concat(frames, ignore_index=True)

    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"columns absent from {path.name}: {missing}")

    found = [column for column in STATE_COLUMN_ALIASES if column in frame.columns]
    if not found:
        raise KeyError(
            f"no state column in {path.name}: looked for {list(STATE_COLUMN_ALIASES)}. "
            f"Check the {area.year - area.span + 1}-{area.year} data dictionary via "
            f"variables() -- a vintage may have renamed it again"
        )
    frame = frame.rename(columns={found[0]: STATE_COLUMN})
    frame = frame.drop(columns=[c for c in found[1:] if c in frame.columns])

    # Widths are fixed by the data dictionary; pad defensively in case a
    # vintage publishes them unquoted and a reader drops the leading zero.
    frame[STATE_COLUMN] = frame[STATE_COLUMN].str.zfill(2)
    frame["PUMA"] = frame["PUMA"].str.zfill(5)
    frame["puma_geoid"] = frame[STATE_COLUMN] + frame["PUMA"]
    return frame


def dictionary_url(area: StudyArea) -> str:
    """URL of the machine-readable PUMS data dictionary for this vintage."""
    start = area.year - area.span + 1
    return f"{PUMS_DICT_BASE}/PUMS_Data_Dictionary_{start}-{area.year}.csv"


def data_dictionary(area: StudyArea, *, force: bool = False) -> pd.DataFrame:
    """The PUMS data dictionary as a frame, one row per declaration.

    The published file is a ragged CSV -- two row shapes sharing no width, which
    is why it is parsed with :mod:`csv` rather than handed to pandas:

    ``NAME,<var>,<type>,<width>,<description>`` declares a variable;
    ``VAL,<var>,<type>,<width>,<lo>,<hi>,<label>`` declares one of its values.

    Returns
    -------
    pandas.DataFrame
        Columns ``record`` (``"NAME"`` or ``"VAL"``), ``variable``, ``dtype``,
        ``width``, ``lo``, ``hi`` and ``label``. A ``NAME`` row carries its
        description in ``label`` and nothing in ``lo``/``hi``.

    Notes
    -----
    ``lo`` and ``hi`` stay text. They are usually numeric ranges but not always
    -- a PUMA bound is ``"00101"``, and reading it as a number destroys the
    padding the same way it would in the data itself.

    A variable declared for both record types appears twice, since the
    dictionary describes each separately. :func:`variables` and
    :func:`variable_labels` de-duplicate; this function does not, so that the
    file is represented as published.

    Raises
    ------
    ValueError
        If no declaration parses at all, which means the published format
        changed rather than that the vintage has no dictionary.
    """
    path = fetch_cached(
        dictionary_url(area),
        subdir=f"pums/{area.year}-{area.span}yr",
        force=force,
    )
    # utf-8-sig for the same reason geography.py uses it: a BOM would otherwise
    # end up inside the first field. errors="replace" because a stray byte in a
    # value label should not cost the whole dictionary.
    text = path.read_text(encoding="utf-8-sig", errors="replace")

    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        kind = fields[0].strip()
        if kind == "NAME" and len(fields) >= 5:
            rows.append(("NAME", *fields[1:4], None, None, fields[4]))
        elif kind == "VAL" and len(fields) >= 7:
            rows.append(("VAL", *fields[1:7]))

    if not rows:
        raise ValueError(
            f"no NAME or VAL rows parsed from {path} -- the dictionary format "
            f"has probably changed; inspect the file and re-run with force=True"
        )

    frame = pd.DataFrame(
        rows,
        columns=["record", "variable", "dtype", "width", "lo", "hi", "label"],
    )
    frame["width"] = pd.to_numeric(frame["width"], errors="coerce").astype("Int64")
    return frame


def variables(area: StudyArea) -> pd.DataFrame:
    """One row per declared variable: ``variable``, ``dtype``, ``width``, ``label``.

    The answer to "does this vintage carry the column I think it does", which is
    worth asking before writing a constraint against it -- PUMS renames columns
    between vintages, and a missing one otherwise surfaces as a ``KeyError``
    from :func:`load_pums` after the download.
    """
    frame = data_dictionary(area)
    names = frame[frame["record"] == "NAME"]
    return (
        names[["variable", "dtype", "width", "label"]]
        .drop_duplicates(subset="variable")
        .sort_values("variable")
        .reset_index(drop=True)
    )


def variable_labels(area: StudyArea, variable: str) -> pd.DataFrame:
    """Value ranges and their labels for one variable: ``lo``, ``hi``, ``label``.

    This is what a constraint cell has to be written against. A published table
    cell is a set of PUMS codes, and the mapping from one to the other is the
    break-for-break correspondence that silent misfit comes from -- so read the
    labels rather than assuming the coding.

    Raises
    ------
    KeyError
        If the variable declares no values. A continuous column such as
        ``AGEP`` or ``HINCP`` legitimately has none; check :func:`variables`
        to tell that apart from a name that does not exist in this vintage.
    """
    frame = data_dictionary(area)
    values = frame[(frame["record"] == "VAL") & (frame["variable"] == variable)]
    if values.empty:
        declared = variable in set(frame["variable"])
        raise KeyError(
            f"{variable!r} declares no values in the "
            f"{area.year - area.span + 1}-{area.year} dictionary"
            + (
                " -- it is declared, so it is probably continuous; see variables()"
                if declared
                else " -- and is not declared at all in this vintage"
            )
        )
    return (
        values[["lo", "hi", "label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
