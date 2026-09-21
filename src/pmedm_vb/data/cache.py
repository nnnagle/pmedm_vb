"""Fetch-once-and-reuse helper shared by every downloader.

Census files are large and the endpoints are slow, so a download is treated as
an expensive pure function of its URL: fetched once into
:func:`~pmedm_vb.config.raw_dir`, reused thereafter, and never edited in place.
"""

from __future__ import annotations

from pathlib import Path


def fetch(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it is already there.

    Parameters
    ----------
    url:
        Absolute URL of the published file.
    dest:
        Destination path beneath :func:`~pmedm_vb.config.raw_dir`.
    force:
        Re-download even when ``dest`` exists.

    Returns
    -------
    Path
        ``dest``, now guaranteed to exist.

    Notes
    -----
    Should stream to a temporary file and rename on completion, so that an
    interrupted download cannot leave a truncated file that later looks cached.
    """
    raise NotImplementedError


def cached_path(url: str, subdir: str = "") -> Path:
    """Return the canonical cache location for ``url`` without fetching it."""
    raise NotImplementedError
