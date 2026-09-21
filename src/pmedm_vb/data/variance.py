"""ACS Variance Replicate Estimate Tables: the raw material for a full ``Sigma``.

These files carry the estimate, the margin of error and 80 variance replicates
for a *selected subset* of the 5-year Detailed Tables, published for areas down
to the block group. They are download-only; the API does not serve them. See
https://www.census.gov/programs-surveys/acs/data/variance-tables.html.

Because the replicate files also carry the estimates, this module is the single
download for both ``Y`` and ``Sigma``; :mod:`pmedm_vb.data.summary` is a view
over what is fetched here rather than a separate request.

Layout and formulas below were confirmed against the published 2019-2023 files
and documentation rather than inferred:

* Data path is ``<year>/data/<span>-year/<sumlevel>/<TBLID>_<stfips>.csv.zip``
  -- note the lowercase ``5-year`` here against the capitalised ``5-Year`` in
  the PUMS tree. Summary level ``140`` is tract, ``150`` is block group.
* Coverage thins with geography. For 2019-2023, 131 tables are published at
  tract and 73 at block group, out of 132 in the master list.
* ``Var(X) = (4/80) * sum_r (X_r - X)^2``, the squared differences taken
  against the published full-sample estimate rather than the replicate mean,
  and ``MOE = 1.645 * sqrt(Var)``. The 4/80 is an artifact of using the
  successive-differences replication estimator with 80 replicates (Fay and
  Train 1995).
* A zero count has all 80 replicates equal to the estimate and so a variance of
  exactly zero. Census models it instead as ``MOE = 1.645 * sqrt(w * k)``, for
  ``w`` the state average weight and ``k`` a step function of the area's total
  population -- equivalently ``Var = w * k``. Both parameters come from files
  published beside the documentation, and ``k`` is keyed on total population,
  so :data:`POPULATION_TABLE` is fetched on any run using that policy.

The published documentation notes that the average weights and k-values are
derived from internal files, so an MOE recomputed this way may not reproduce
the published one exactly. That is expected, not a defect here.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.cache import fetch_cached

VRE_BASE = "https://www2.census.gov/programs-surveys/acs/replicate_estimates"

#: Census summary level codes for the geographies a PMEDM run constrains.
SUMMARY_LEVELS = {"tract": "140", "block group": "150"}

#: Number of variance replicates published per estimate.
N_REPLICATES = 80

#: Multiplier outside the summation in the SDR variance.
SDR_FACTOR = 4 / N_REPLICATES

#: Normal deviate for the 90 percent confidence level the ACS publishes MOEs at.
Z_90 = 1.645

#: Total population, used to assign a k-value. Not a constraint table.
POPULATION_TABLE = "B01003"

#: Upper population bound (exclusive) mapped to its k-value, from Table APP2 of
#: Appendix A. The final entry is the open-ended top bracket.
K_VALUE_BRACKETS = (
    (5_000, 4),
    (10_000, 8),
    (20_000, 10),
    (30_000, 14),
    (50_000, 18),
    (np.inf, 22),
)

#: How a cell whose replicate-based variance is exactly zero is treated.
ZERO_POLICIES = ("model", "drop", "floor")

REPLICATE_COLUMNS = tuple(f"rep_{i}" for i in range(1, N_REPLICATES + 1))


def _span_label(area: StudyArea) -> str:
    """``"5-year"`` -- the lowercase spelling the replicate tree uses."""
    return f"{area.span}-year"


def _summary_level(geography: str) -> str:
    try:
        return SUMMARY_LEVELS[geography]
    except KeyError:
        raise ValueError(
            f"geography must be one of {sorted(SUMMARY_LEVELS)}, got {geography!r}"
        ) from None


def documentation_url(area: StudyArea, filename: str) -> str:
    """URL of a file in the vintage's documentation directory."""
    return f"{VRE_BASE}/{area.year}/documentation/{_span_label(area)}/{filename}"


def replicate_url(area: StudyArea, table: str, *, geography: str) -> str:
    """URL of one table's replicate file for this state and geography."""
    level = _summary_level(geography)
    return (
        f"{VRE_BASE}/{area.year}/data/{_span_label(area)}/{level}/"
        f"{table}_{area.state}.csv.zip"
    )


def available_tables(area: StudyArea) -> list[str]:
    """Return the table IDs published with variance replicates for this vintage.

    Read from the published ``VRE_TABLE_LIST`` rather than by scraping a
    directory index. This is the *master* list; coverage at tract and block
    group is narrower, and :func:`is_available` checks a specific geography.
    """
    path = fetch_cached(
        documentation_url(area, f"VRE_TABLE_LIST_{area.year}.csv"),
        subdir=f"vre/{area.year}/documentation",
    )
    return pd.read_csv(path, dtype=str)["TBLID"].tolist()


def is_available(area: StudyArea, table: str, *, geography: str) -> bool:
    """Whether ``table`` is published at ``geography``, by asking the server.

    One HEAD request per call. The published table-by-geography list is an
    ``.xlsx``, which would pull in an Excel reader for a question a 200 or a
    404 already answers.
    """
    import requests

    response = requests.head(
        replicate_url(area, table, geography=geography),
        timeout=30,
        allow_redirects=True,
    )
    return response.status_code == 200


def download_replicates(
    area: StudyArea,
    table: str,
    *,
    geography: str,
    force: bool = False,
) -> Path:
    """Download one table's replicate file and return its cached path."""
    level = _summary_level(geography)
    return fetch_cached(
        replicate_url(area, table, geography=geography),
        subdir=f"vre/{area.year}/{level}",
        force=force,
    )


def _read_replicate_file(path: Path) -> pd.DataFrame:
    """Parse one replicate zip into tidy rows, one per geography and cell.

    The published columns are ``TBLID, GEOID, NAME, ORDER, TITLE, ESTIMATE,
    MOE, CME, SE, Var_Rep1..Var_Rep80``. The first two rows of each file are
    the table title and its universe, carrying a ``TBLID`` but no ``GEOID``;
    they are dropped. ``CME`` is the MOE as a display string (``"+/-251"``) and
    is not kept.
    """
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path}, found {members}")
        with archive.open(members[0]) as handle:
            raw = pd.read_csv(handle, dtype=str, keep_default_na=False)

    raw = raw[raw["GEOID"].str.len() > 0].copy()

    frame = pd.DataFrame(index=raw.index)
    frame["tblid"] = raw["TBLID"]
    # Published as a Census GEO_ID: a summary-level prefix, "US", then the code.
    frame["geoid"] = raw["GEOID"].str.split("US").str[-1]
    frame["order"] = pd.to_numeric(raw["ORDER"], errors="coerce").astype("Int64")
    frame["title"] = raw["TITLE"]
    for column, source in (("estimate", "ESTIMATE"), ("moe", "MOE"), ("se", "SE")):
        frame[column] = pd.to_numeric(raw[source], errors="coerce")
    for index, column in enumerate(REPLICATE_COLUMNS, start=1):
        frame[column] = pd.to_numeric(raw[f"Var_Rep{index}"], errors="coerce")

    # Mirrors the Census variable naming (B25003_001), so a cell key is
    # recognisable against published documentation.
    frame["cell"] = (
        frame["tblid"] + "_" + frame["order"].astype("Int64").astype(str).str.zfill(3)
    )
    return frame.reset_index(drop=True)


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
        Indexed by ``(geoid, cell)``, with ``tblid``, ``order``, ``title``,
        ``estimate``, ``moe`` and ``se`` columns followed by ``rep_1``
        through ``rep_80``. Restricted to ``area.counties`` when set.
    """
    frames = [
        _read_replicate_file(download_replicates(area, table, geography=geography))
        for table in tables
    ]
    frame = pd.concat(frames, ignore_index=True)

    counties = set(area.county_geoids())
    if counties:
        frame = frame[frame["geoid"].str[:5].isin(counties)]

    return frame.set_index(["geoid", "cell"]).sort_index()


def replicate_matrix(frame: pd.DataFrame) -> np.ndarray:
    """The ``(n_cells, 80)`` replicate array from a :func:`fetch_replicates` frame."""
    return frame.loc[:, list(REPLICATE_COLUMNS)].to_numpy(dtype=float)


def average_weight(area: StudyArea) -> float:
    """State average weight ``w`` for the zero-count model.

    Block groups and tracts nest within states, so the national row is never
    the right one here and is not consulted.
    """
    path = fetch_cached(
        documentation_url(area, f"VRE_AVERAGE_WEIGHT_{area.year}.csv"),
        subdir=f"vre/{area.year}/documentation",
    )
    weights = pd.read_csv(path, dtype={"STATE": str})
    row = weights.loc[weights["STATE"] == area.state, "AVERAGE_WEIGHT"]
    if row.empty:
        raise KeyError(f"no average weight published for state {area.state!r}")
    return float(row.iloc[0])


def k_value(population: float) -> int:
    """k-value for an area of this total population, per Table APP2."""
    for upper, k in K_VALUE_BRACKETS:
        if population < upper:
            return k
    raise AssertionError("K_VALUE_BRACKETS must end with an open-ended bracket")


def population_estimates(area: StudyArea, *, geography: str) -> pd.Series:
    """Total population per geography, the input to :func:`k_value`."""
    frame = fetch_replicates(area, [POPULATION_TABLE], geography=geography)
    totals = frame.xs(f"{POPULATION_TABLE}_001", level="cell")["estimate"]
    return totals.rename("population")


def zero_cell_variance(area: StudyArea, *, geography: str) -> pd.Series:
    """Modelled variance ``w * k`` for a zero count, per geography.

    Neither ``w`` nor ``k`` varies by cell, so one value covers every zero cell
    within a geography.
    """
    weight = average_weight(area)
    populations = population_estimates(area, geography=geography)
    return (populations.map(k_value) * weight).rename("variance")


def variances(
    frame: pd.DataFrame,
    *,
    area: StudyArea,
    geography: str,
    policy: str = "model",
    floor: float | None = None,
) -> pd.Series:
    """Per-cell SDR variance, with zero-variance cells handled by ``policy``.

    A zero count has every replicate equal to its estimate, so the SDR variance
    is exactly zero and ``Sigma`` is singular through those cells. At block
    group these are common rather than exceptional.

    Parameters
    ----------
    policy:
        ``"model"`` applies the published ``w * k`` model; ``"drop"`` removes
        the affected cells from the result; ``"floor"`` replaces them with
        ``floor``.
    floor:
        Required when ``policy`` is ``"floor"``.
    """
    if policy not in ZERO_POLICIES:
        raise ValueError(f"policy must be one of {ZERO_POLICIES}, got {policy!r}")

    deviations = replicate_matrix(frame) - frame["estimate"].to_numpy(float)[:, None]
    result = pd.Series(
        SDR_FACTOR * np.square(deviations).sum(axis=1),
        index=frame.index,
        name="variance",
    )

    degenerate = result == 0.0
    if not degenerate.any():
        return result

    if policy == "drop":
        return result[~degenerate]
    if policy == "floor":
        if floor is None:
            raise ValueError("policy='floor' requires a floor value")
        result[degenerate] = floor
        return result

    modelled = zero_cell_variance(area, geography=geography)
    geoids = result.index.get_level_values("geoid")[degenerate]
    result[degenerate] = modelled.reindex(geoids).to_numpy()
    return result


def covariance_from_replicates(
    replicates: np.ndarray,
    estimates: np.ndarray,
) -> np.ndarray:
    """Form the design-based covariance from replicate estimates.

    ``Cov = (4/80) * D D'`` for ``D`` the deviations of each replicate from the
    full-sample estimate. This is the bilinear form of the same estimator the
    documentation gives for a variance; the documentation states the diagonal
    case and licenses any derived statistic through it, but does not write the
    cross-cell covariance out.

    ``estimates`` is a required argument rather than the replicate mean: the
    published formula takes deviations against the full-sample estimate, and
    the two differ.

    Notes
    -----
    A sum of 80 outer products has rank at most 80 however many cells there
    are, so for a block-group run this matrix is large and singular. Prefer
    carrying the ``(n, 80)`` deviations and letting the solver exploit the
    low-rank structure; materialise this only when a dense ``Sigma`` is
    genuinely wanted.
    """
    replicates = np.asarray(replicates, dtype=float)
    estimates = np.asarray(estimates, dtype=float)
    if replicates.ndim != 2 or replicates.shape[1] != N_REPLICATES:
        raise ValueError(
            f"replicates must be (n, {N_REPLICATES}), got {replicates.shape}"
        )
    if estimates.shape != (replicates.shape[0],):
        raise ValueError(
            f"estimates must be ({replicates.shape[0]},), got {estimates.shape}"
        )
    deviations = replicates - estimates[:, None]
    return SDR_FACTOR * (deviations @ deviations.T)
