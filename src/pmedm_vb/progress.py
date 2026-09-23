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
