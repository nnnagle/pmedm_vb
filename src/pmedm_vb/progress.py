"""Progress messages for the slow steps: downloads, file reads, assembly, solving.

A first run downloads and parses statewide files and can go quiet for a long
time, which is indistinguishable from a hang without some sign of life. So
messages are **on by default**, as timestamped lines on stderr from the
``pmedm_vb`` logger::

    15:50:02 pmedm_vb  read B01001 replicates (block group) ...
    15:50:09 pmedm_vb  read B01001 replicates (block group): done in 7.1s (190,512 rows)

Silence them with ``pmedm_vb.set_verbosity("WARNING")``, or get more with
``"DEBUG"``. They go to stderr, so ``2>&1 | tee`` captures them alongside a
script's own output, and they are flushed per line, so a pipe does not hold
them back the way it buffers ``print``.

The handler is attached to the ``pmedm_vb`` logger only, with propagation off,
so a script that configures the root logger itself does not see each line
twice.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

logger = logging.getLogger("pmedm_vb")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s pmedm_vb  %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def set_verbosity(level: str | int) -> None:
    """Set the threshold: ``"DEBUG"``, ``"INFO"`` (default) or ``"WARNING"``."""
    logger.setLevel(level.upper() if isinstance(level, str) else level)


@contextmanager
def stage(message: str) -> Iterator[SimpleNamespace]:
    """Log ``message ...`` on entry and ``message: done in Ns`` on exit.

    The entry line is the point: when a run stalls, the last unmatched one
    names the step it is stuck in. The yielded object's ``detail`` attribute,
    if set inside the block, is appended to the exit line -- a row count, say.
    A step that raises is logged as failed, with its time, and the exception
    propagates unchanged.
    """
    note = SimpleNamespace(detail="")
    logger.info("%s ...", message)
    start = time.perf_counter()
    try:
        yield note
    except BaseException:
        logger.info("%s: failed after %.1fs", message, time.perf_counter() - start)
        raise
    suffix = f" ({note.detail})" if note.detail else ""
    logger.info("%s: done in %.1fs%s", message, time.perf_counter() - start, suffix)


def record_run(directory, args) -> None:
    """Append this invocation to ``<directory>/run_args.jsonl``.

    One JSON line per call: time, command line, every parsed argument (defaults
    included), git commit, Slurm job id and host -- so a results directory says
    how it was made. Appending keeps a resumed job's history rather than
    overwriting the first record.
    """
    import json
    import os
    import socket
    import subprocess
    from pathlib import Path

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": socket.gethostname(),
    }
    path = Path(directory) / "run_args.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _part_path(path):
    """A temporary name next to ``path``: hidden, unique to this process, and not
    ending in the final suffix, so a glob for finished results never sees it."""
    import os
    from pathlib import Path

    path = Path(path)
    return path.with_name(f".{path.name}.{os.getpid()}.part")


def save_npz(path, **arrays) -> None:
    """``np.savez(path, **arrays)``, appearing all at once.

    The archive is written under a temporary name in the same directory and
    renamed into place: a rename within a directory is atomic, so a job killed
    mid-write leaves a ``.part`` file and no truncated ``.npz`` that a resumed
    run would take for a finished one. Writing through an open handle keeps
    numpy from appending ``.npz`` to the temporary name. The file's name and
    contents are those ``np.savez(path, ...)`` writes.
    """
    save_npz_files({path: arrays})


def save_npz_files(files: dict) -> None:
    """:func:`save_npz` for several files, ``{path: arrays}``, that belong together
    (a sampler's state and its trace): every one is written under its temporary
    name first, then all are renamed, so a kill leaves them out of step only in
    the moment between two renames, not for the length of a write."""
    import os

    import numpy as np

    parts = {path: _part_path(path) for path in files}
    try:
        for path, arrays in files.items():
            with open(parts[path], "wb") as handle:
                np.savez(handle, **arrays)
        for path, part in parts.items():
            os.replace(part, path)
    finally:
        for part in parts.values():
            if part.exists():
                part.unlink()


def write_text(path, text: str) -> None:
    """``Path(path).write_text(text)``, appearing all at once (see :func:`save_npz`)."""
    import os

    part = _part_path(path)
    try:
        part.write_text(text)
        os.replace(part, path)
    finally:
        if part.exists():
            part.unlink()
