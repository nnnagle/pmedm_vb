"""Filesystem layout and the definition of a study area.

Every downloader and assembler takes a :class:`StudyArea` so that a run is
described by one object rather than by six loose arguments, and writes beneath
:func:`data_dir` so that the cache root is a single deployment-time decision.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Environment variable holding the cache root. On ISAAC this should point at
#: Lustre scratch: PUMS state files are large and home directories are quota'd.
ENV_DATA_DIR = "PMEDM_VB_DATA"

#: Used when :data:`ENV_DATA_DIR` is unset -- a ``data/`` directory under the
#: current working directory, which is convenient on a laptop and wrong on a
#: cluster.
DEFAULT_DATA_DIR = "data"


def data_dir() -> Path:
    """Return the cache root, from ``$PMEDM_VB_DATA`` or the local default.

    Pure path resolution: nothing is created on disk. Callers that are about to
    write should pass the result through :func:`ensure_dir`.
    """
    return Path(os.environ.get(ENV_DATA_DIR, DEFAULT_DATA_DIR)).expanduser()


def raw_dir() -> Path:
    """Downloads exactly as published, never edited in place."""
    return data_dir() / "raw"


def interim_dir() -> Path:
    """Parsed and cleaned, but not yet arranged into solver inputs."""
    return data_dir() / "interim"


def processed_dir() -> Path:
    """Saved :class:`~pmedm_vb.assemble.inputs.PMEDMInputs` directories."""
    return data_dir() / "processed"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class StudyArea:
    """One PMEDM problem: a place, a vintage, and a target geography.

    Attributes
    ----------
    name:
        Human-readable label, used to build directory names via :attr:`slug`.
    state:
        Two-digit state FIPS code, as a string so that leading zeros survive
        (``"47"`` for Tennessee, ``"01"`` for Alabama).
    year:
        End year of the ACS period estimate, e.g. ``2023`` for 2019-2023.
    counties:
        Three-digit county FIPS codes to restrict to. Empty means the whole
        state.
    span:
        Length of the ACS period in years. PMEDM needs block group estimates,
        which are published only for the 5-year period, so 5-year is the
        default and the only span expected to be useful here.
    """

    name: str
    state: str
    year: int
    counties: tuple[str, ...] = ()
    span: int = 5

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier, unique per place and vintage."""
        cleaned = "".join(c if c.isalnum() else "-" for c in self.name.lower())
        cleaned = "-".join(part for part in cleaned.split("-") if part)
        return f"{cleaned}-{self.year}-{self.span}yr"

    def county_geoids(self) -> tuple[str, ...]:
        """Five-digit state+county GEOIDs, empty when the area is statewide."""
        return tuple(f"{self.state}{county}" for county in self.counties)
