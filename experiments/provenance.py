"""Which code wrote an output, so that a driver can tell a stale output from a current one.

A script writes ``<output>.stamp.json`` beside its output once everything is
written: a digest of the scripts that decide the output's content and the
settings it was run with. :func:`stale` says why an output is not current --
missing, unstamped, written by other code, or with other settings -- or
``None``. An output's age against its inputs is the caller's to check.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parent

#: The scripts that decide each output's content, relative to experiments/.
TABLES_SCRIPTS = ("paper_tables.py",)
EXAMPLES_SCRIPTS = ("draw_examples.py", "compare_methods.py")
INVENTORY_SCRIPTS = ("table_inventory.py", "design_cells.py")
SIM_COLLECT_SCRIPTS = ("sim_collect.py", "sim_common.py")
SIM_SCORE_SCRIPTS = ("sim_score.py",)


def digest(scripts) -> str:
    h = hashlib.sha256()
    for name in scripts:
        h.update(name.encode())
        h.update((EXPERIMENTS / name).read_bytes())
    return h.hexdigest()[:16]


def stamp_path(output: Path) -> Path:
    return output.with_name(output.name + ".stamp.json")


def _plain(settings: dict) -> dict:
    return json.loads(json.dumps(settings))


def write_stamp(output: Path, scripts, **settings) -> None:
    record = {"code": digest(scripts), "scripts": list(scripts), "settings": _plain(settings),
              "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    stamp_path(output).write_text(json.dumps(record, indent=2) + "\n")


def stale(output: Path, scripts, **settings) -> str | None:
    """Why ``output`` is not current with ``scripts`` and ``settings``, or None."""
    if not output.exists():
        return f"{output} is missing"
    path = stamp_path(output)
    if not path.exists():
        return f"{output} has no stamp: written by older code"
    stamp = json.loads(path.read_text())
    if stamp.get("code") != digest(scripts):
        return f"{output} was written by another version of {', '.join(scripts)}"
    for key, value in _plain(settings).items():
        if stamp.get("settings", {}).get(key) != value:
            return (f"{output} was written with {key} = {stamp.get('settings', {}).get(key)!r}, "
                    f"not {value!r}")
    return None
