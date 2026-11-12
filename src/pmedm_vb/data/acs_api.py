"""ACS detailed tables from the Census Data API: estimates, MOEs and cell labels.

The second route to published tables, beside the Variance Replicate Estimate
files of :mod:`pmedm_vb.data.variance`. It exists for tables those files do not
carry, and for geographies they do not publish (the PUMA, summary level 795).
It returns estimates and margins of error only -- the API serves no
replicates, so a non-diagonal ``Sigma`` cannot come from here, and the VRE
route stays the one PMEDM's constraints use.

Two kinds of request, with different needs:

* **Cell labels** (``/groups/<TABLE>.json``) need no key. They are what new
  table specifications are validated against: every published cell claimed or
  waived, none claimed twice (:func:`pmedm_vb.embed.tables.validate_against_api`).
* **Values** (``get=group(<TABLE>)``) need a key. Without one, or with an
  invalid one, the API answers with a redirect to ``missing_key.html`` or
  ``invalid_key.html`` -- both seen from the live endpoint -- so redirects are
  not followed and are reported as a key problem.

**The key** is read from :data:`KEY_ENV` at download time and nowhere else.
It is sent as the ``key`` query parameter, as the API requires; it is never
part of a cache path, never written to disk and never logged (logged URLs go
through :func:`_redact`). A cached response needs no key to read, so batch jobs
on compute nodes never see it.

**Where requests may run.** As for every other download here: on a login or
data transfer node, never a compute node, which cannot reach census.gov.
:data:`~pmedm_vb.data.cache.OFFLINE_ENV` is honoured exactly as
:func:`~pmedm_vb.data.cache.fetch` honours it -- a batch job sets it, and a
response that is not cached then raises at once, naming itself.

**Jam values.** The API encodes the absence of a number as large negative
integers, documented in the Census "Jam Value Specifications" workbook
(``tech_docs/jam_values/2024_Jam_Values.xlsx`` on www2.census.gov; read for
this module, not recalled). For estimates and MOEs:

==============  ========  ====================================================
value           applies   meaning
==============  ========  ====================================================
``-666666666``  estimate  not computed: insufficient sample cases
``-888888888``  both      not applicable or not available
``-999999999``  both      not displayed: insufficient sample for the geography
``-222222222``  MOE       not computed: insufficient sample cases
``-333333333``  MOE       not computed for medians in lower or upper interval
``-555555555``  MOE       estimate is controlled; "MOE may be treated as zero
                          since estimate has no sampling error"
==============  ========  ====================================================

Every jam becomes ``NaN`` except ``-555555555`` on an MOE, which becomes 0 as
the specification allows. The count tables used here should carry none of the
median codes; they are handled anyway rather than assumed away.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pmedm_vb.config import StudyArea, ensure_dir, raw_dir
from pmedm_vb.data.cache import OFFLINE_ENV, TIMEOUT_SECONDS
from pmedm_vb.progress import logger

API_BASE = "https://api.census.gov/data"

#: Environment variable holding the Census API key. On ISAAC it is exported
#: from ``~/.config/census/env`` on the login and data transfer nodes.
KEY_ENV = "CENSUS_API_KEY"

#: Geographies served, as ``(API "for" clause, summary level)``. The ``in``
#: clause is the state alone for both: the dataset's ``geography.json`` lists
#: county as a wildcard for tract (``optionalWithWCFor: county``) and state as
#: the only requirement for a PUMA.
GEOGRAPHIES = {
    "puma": ("public use microdata area", "795"),
    "tract": ("tract", "140"),
}

#: Estimate jam values, all meaning "no number".
ESTIMATE_JAMS = (-666666666, -888888888, -999999999)

#: MOE jam values meaning "no number".
MOE_JAMS = (-222222222, -333333333, -888888888, -999999999)

#: MOE jam value meaning "controlled estimate, no sampling error": treated as 0.
MOE_CONTROLLED = -555555555

#: An estimate or MOE variable of a table, e.g. ``B25032_003E``.
_VALUE_VARIABLE = re.compile(r"^(?P<cell>[A-Z]\d{5}[A-Z]*_\d{3})(?P<kind>[EM])$")


def _dataset(area: StudyArea) -> str:
    """``2024/acs/acs5`` -- the dataset path for the area's vintage and span."""
    return f"{area.year}/acs/acs{area.span}"


def _redact(url: str) -> str:
    """``url`` with any ``key=`` value removed, for logs and error messages."""
    return re.sub(r"([?&]key=)[^&]*", r"\1<redacted>", url)


def labels_path(area: StudyArea, table: str) -> Path:
    """Cache location of a table's ``groups`` metadata."""
    return raw_dir() / "api" / f"{area.year}" / f"acs{area.span}" / "groups" / f"{table}.json"


def values_path(area: StudyArea, table: str, geography: str) -> Path:
    """Cache location of a table's statewide values at ``geography``.

    Statewide whatever ``area.counties`` says: one request serves every county
    subset, and the subset is applied when the file is read.
    """
    _geography(geography)
    return (
        raw_dir() / "api" / f"{area.year}" / f"acs{area.span}" / geography
        / f"{table}_{area.state}.json"
    )


def _geography(geography: str) -> tuple[str, str]:
    try:
        return GEOGRAPHIES[geography]
    except KeyError:
        raise ValueError(
            f"geography must be one of {sorted(GEOGRAPHIES)}, got {geography!r}"
        ) from None


def _offline_check(dest: Path, what: str) -> None:
    if os.environ.get(OFFLINE_ENV):
        raise FileNotFoundError(
            f"{dest} is not cached, and {OFFLINE_ENV} is set so {what} will not be "
            f"requested. Run `python experiments/embed/prefetch.py` on a login or "
            f"data transfer node first"
        )


def _write_atomic(dest: Path, text: str) -> None:
    """Write via a ``.part`` sibling renamed into place, as ``cache.fetch`` does,
    so an interrupted write never leaves a truncated file that looks cached."""
    ensure_dir(dest.parent)
    partial = dest.with_name(dest.name + ".part")
    try:
        partial.write_text(text, encoding="utf-8")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)


def _get_json(url: str, params: dict, *, what: str) -> object:
    """GET ``url`` and parse JSON, without following redirects.

    The API redirects a request lacking a valid key to an HTML page, which
    ``requests`` would otherwise follow and hand back as a 200. A redirect is
    therefore an error here, and so is a body that is not JSON.
    """
    response = requests.get(
        url, params=params, timeout=TIMEOUT_SECONDS, allow_redirects=False
    )
    shown = _redact(response.url)
    if response.is_redirect or 300 <= response.status_code < 400:
        location = response.headers.get("Location", "")
        raise RuntimeError(
            f"{what}: the API redirected ({response.status_code}) to {location!r} "
            f"instead of answering {shown}. A redirect to missing_key.html or "
            f"invalid_key.html means {KEY_ENV} is unset, empty or not a valid key"
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"{what}: HTTP {response.status_code} from {shown}: "
            f"{response.text[:300]!r}"
        )
    try:
        return response.json()
    except ValueError:
        raise RuntimeError(
            f"{what}: response from {shown} is not JSON: {response.text[:300]!r}"
        ) from None


# --------------------------------------------------------------------------
# Cell labels (no key)
# --------------------------------------------------------------------------


def download_labels(area: StudyArea, table: str, *, force: bool = False) -> Path:
    """Fetch a table's ``groups`` metadata into the cache; no key needed."""
    dest = labels_path(area, table)
    if dest.exists() and not force:
        logger.info("cached   %s", dest.name)
        return dest
    _offline_check(dest, f"the {table} labels")
    url = f"{API_BASE}/{_dataset(area)}/groups/{table}.json"
    logger.info("download api labels %s (%s)", table, _dataset(area))
    payload = _get_json(url, {}, what=f"{table} labels")
    if not isinstance(payload, dict) or "variables" not in payload:
        raise RuntimeError(f"{table} labels: unexpected payload from {url}")
    _write_atomic(dest, json.dumps(payload))
    return dest


def api_cell_labels(area: StudyArea, table: str) -> pd.Series:
    """Published cells of ``table`` and their titles, from the API metadata.

    Indexed by cell key in the form the VRE route uses (``B25032_003``),
    ordered by cell number, with the ``Estimate!!`` prefix and the ``!!``
    separators of the API label turned into the `` | `` used for display.
    """
    path = download_labels(area, table)
    variables = json.loads(path.read_text(encoding="utf-8"))["variables"]
    labels = {}
    for name, meta in variables.items():
        match = _VALUE_VARIABLE.match(name)
        if match and match["kind"] == "E" and match["cell"].startswith(f"{table}_"):
            label = str(meta.get("label", ""))
            label = re.sub(r"^Estimate!!", "", label).replace("!!", " | ")
            labels[match["cell"]] = label
    if not labels:
        raise RuntimeError(f"{table}: no estimate variables in {path}")
    return pd.Series(labels, name="title").sort_index()


# --------------------------------------------------------------------------
# Values (key required)
# --------------------------------------------------------------------------


def download_values(
    area: StudyArea, table: str, *, geography: str, force: bool = False
) -> Path:
    """Fetch a table's estimates and MOEs for every ``geography`` in the state.

    The cached file is the API's JSON array exactly as returned (header row,
    then one row per area); the key is in neither its path nor its contents.
    """
    dest = values_path(area, table, geography)
    if dest.exists() and not force:
        logger.info("cached   %s", dest.name)
        return dest
    _offline_check(dest, f"the {table} values")
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY_ENV} is not set, so {table} at {geography} cannot be requested "
            f"from the Census API. On ISAAC: `. ~/.config/census/env`, then rerun "
            f"on a login or data transfer node"
        )
    level, _ = _geography(geography)
    params = {
        "get": f"group({table})",
        "for": f"{level}:*",
        "in": f"state:{area.state}",
        "key": key,
    }
    url = f"{API_BASE}/{_dataset(area)}"
    logger.info("download api %s %s state %s (%s)", table, geography, area.state, _dataset(area))
    payload = _get_json(url, params, what=f"{table} at {geography}")
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"{table} at {geography}: empty or unexpected payload")
    _write_atomic(dest, json.dumps(payload))
    return dest


def _clean(values: pd.Series, jams: tuple[int, ...], zero: int | None = None) -> pd.Series:
    numbers = pd.to_numeric(values, errors="coerce")
    if zero is not None:
        numbers = numbers.mask(numbers == zero, 0.0)
    return numbers.mask(numbers.isin(jams), np.nan).astype(float)


def fetch_api_tables(
    area: StudyArea,
    tables: list[str],
    *,
    geography: str,
) -> pd.DataFrame:
    """Estimates and MOEs for ``tables`` over ``area``, from the API.

    Returns
    -------
    pandas.DataFrame
        The layout of :func:`pmedm_vb.data.summary.fetch_summary_tables`: one
        row per area, indexed by GEOID (the code after ``US`` in the Census
        ``GEO_ID``, e.g. ``4700100`` for a PUMA, state plus ``PUMA``, matching
        the PUMS ``puma_geoid``), with two-level columns -- ``"estimate"`` or
        ``"moe"`` over the cell key. Jam values are ``NaN`` (a controlled MOE
        is 0); see the module notes. Tract rows are restricted to
        ``area.counties`` when set.
    """
    frames = []
    for table in tables:
        path = download_values(area, table, geography=geography)
        rows = json.loads(path.read_text(encoding="utf-8"))
        raw = pd.DataFrame(rows[1:], columns=rows[0], dtype=str)
        geoid = raw["GEO_ID"].str.split("US").str[-1]
        estimates, moes = {}, {}
        for column in raw.columns:
            match = _VALUE_VARIABLE.match(column)
            if not match:
                continue
            if match["kind"] == "E":
                estimates[match["cell"]] = _clean(raw[column], ESTIMATE_JAMS)
            else:
                moes[match["cell"]] = _clean(raw[column], MOE_JAMS, zero=MOE_CONTROLLED)
        if not estimates:
            raise RuntimeError(f"{table}: no estimate columns in {path}")
        part = pd.concat(
            {"estimate": pd.DataFrame(estimates), "moe": pd.DataFrame(moes)}, axis=1
        )
        part.index = pd.Index(geoid, name="geoid")
        frames.append(part)

    frame = pd.concat(frames, axis=1)
    counties = set(area.county_geoids())
    if counties and geography == "tract":
        frame = frame[frame.index.str[:5].isin(counties)]
    return frame.sort_index(axis="columns")
