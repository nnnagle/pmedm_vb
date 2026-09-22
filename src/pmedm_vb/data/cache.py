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

#: Encodings tried, in order, when decoding a published Census file. These are
#: not UTF-8: a replicate table carrying an accented place name has a raw
#: ``0xFA`` in it, which is ``u`` with an acute accent in the Windows codepage
#: and not valid UTF-8 at all. Trying UTF-8 first keeps a genuinely UTF-8 file
#: correct; cp1252 then decodes the legacy ones without loss. Decoding straight
#: to cp1252, or passing ``errors="replace"``, would quietly corrupt characters
#: instead of reading them.
#:
#: This is a fallback order, not encoding detection: cp1252 leaves only five
#: byte values undefined, so it decodes almost anything and the raise below is
#: close to unreachable. What is guaranteed is "UTF-8 where the file really is
#: UTF-8, cp1252 otherwise" -- a file in some third encoding would be decoded
#: wrongly rather than rejected.
PUBLISHED_ENCODINGS = ("utf-8-sig", "cp1252")


def decode(raw: bytes, *, source: str = "") -> str:
    """Decode bytes from a published file, trying :data:`PUBLISHED_ENCODINGS`.

    Raises
    ------
    UnicodeDecodeError
        If no candidate encoding decodes the bytes, which means the publisher
        changed encoding again rather than that this particular file is odd.
    """
    for encoding in PUBLISHED_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as error:
            last = error
    raise UnicodeDecodeError(
        last.encoding,
        last.object,
        last.start,
        last.end,
        f"none of {list(PUBLISHED_ENCODINGS)} decodes {source or 'this file'}",
    )


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
