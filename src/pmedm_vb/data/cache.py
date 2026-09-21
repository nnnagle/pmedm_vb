"""Fetch-once-and-reuse helper shared by every downloader.

Census files are large and the endpoints are slow, so a download is treated as
an expensive pure function of its URL: fetched once into
:func:`~pmedm_vb.config.raw_dir`, reused thereafter, and never edited in place.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

from pmedm_vb.config import ensure_dir, raw_dir

#: Bytes per streamed chunk. Large enough that a multi-hundred-megabyte PUMS
#: file does not spend its time in the loop rather than on the socket.
CHUNK_BYTES = 1 << 20

#: Seconds to wait for the *response headers*, not for the whole transfer --
#: requests applies a read timeout per chunk, so a slow but live download is
#: not killed by it.
TIMEOUT_SECONDS = 120


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
    The download streams to a ``.part`` sibling and is renamed into place only
    once the transfer completes. A rename within a directory is atomic, so an
    interrupted download leaves a ``.part`` file rather than a truncated one
    that a later call would mistake for a complete cache entry.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest

    ensure_dir(dest.parent)
    partial = dest.with_name(dest.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            with open(partial, "wb") as handle:
                for chunk in response.iter_content(CHUNK_BYTES):
                    handle.write(chunk)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    partial.replace(dest)
    return dest


def cached_path(url: str, subdir: str = "") -> Path:
    """Return the canonical cache location for ``url`` without fetching it.

    The basename alone is not unique across vintages or summary levels -- every
    year publishes its own ``B01003_47.csv.zip`` -- so callers pass a ``subdir``
    that carries whatever distinguishes them.
    """
    name = unquote(Path(urlparse(url).path).name)
    if not name:
        raise ValueError(f"no filename in URL: {url!r}")
    root = raw_dir()
    return (root / subdir / name) if subdir else (root / name)


def fetch_cached(url: str, subdir: str = "", *, force: bool = False) -> Path:
    """Fetch ``url`` into its canonical cache location and return the path."""
    return fetch(url, cached_path(url, subdir), force=force)
