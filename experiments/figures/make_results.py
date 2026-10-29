"""Pack the figure scripts and their data into ``results.zip`` for the Quarto project.

Reads ``results_manifest.txt`` (next to this script): each line names a figure
script in the repository or a data file in the paper tables' folder. Every file
lands flat in ``results/`` inside the zip, beside a ``MANIFEST.txt`` that
records where each came from, its size and modification time, and the
repository commit. A manifest line that matches no file stops the script, so a
missing table is noticed here rather than when the paper renders; so does a
table or data file written by older code than the repository's (its stamp,
experiments/provenance.py): rerun the driver (run_paper.py), which brings them
up to date and builds ``<root>/results.zip`` itself. By hand::

    $CONDA_PREFIX/bin/python experiments/figures/make_results.py \\
        --tables $RUNS/paper_v3/tables_compare --out $RUNS/paper_v3/results.zip

Unzipped in the Quarto project's folder, it gives ``results/``, from which a
chunk draws a figure, e.g. ``source("results/fig_cost.R");
plot_cost(cost_data("results"))``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = HERE / "results_manifest.txt"

sys.path.insert(0, str(HERE.parent))

from provenance import EXAMPLES_SCRIPTS, INVENTORY_SCRIPTS, TABLES_SCRIPTS, stale  # noqa: E402

#: The tables-folder files written by other scripts than paper_tables.py: name ->
#: (the output whose stamp covers it, the scripts). Any other tables-folder file
#: is paper_tables.py's, covered by tables.md's stamp.
STAMPED = {"draw_examples.csv": ("draw_examples.csv", EXAMPLES_SCRIPTS),
           **{f"{stem}.{ext}": ("table_inventory.md", INVENTORY_SCRIPTS)
              for stem in ("table_inventory", "table_cells", "puma_cells", "parameters")
              for ext in ("md", "tex", "csv")}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tables", type=Path, required=True,
                        help="the paper tables' folder (paper_tables.py --out)")
    parser.add_argument("--out", type=Path, default=Path("results.zip"))
    return parser.parse_args()


def entries(tables: Path, manifest: Path = MANIFEST,
            roots: dict | None = None) -> list[tuple[Path, str, str]]:
    """(file, source, used by) for every manifest line, globs expanded. ``roots``
    maps a line's source to its folder (default: ``repo`` and ``tables``)."""
    roots = {"repo": REPO, "tables": tables} if roots is None else {"repo": REPO, **roots}
    out = []
    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        source, pattern, *used = line.split(maxsplit=2)
        if source not in roots:
            raise SystemExit(f"{manifest}:{number}: unknown source {source!r}")
        matches = sorted(roots[source].glob(pattern))
        if not matches:
            raise SystemExit(f"{manifest}:{number}: nothing matches {roots[source] / pattern}")
        out += [(path, source, used[0] if used else "") for path in matches]
    names = [path.name for path, _, _ in out]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise SystemExit(f"two files would land on the same name in the zip: {clashes}")
    return out


def staleness(tables: Path, files) -> list[str]:
    """Why any of ``files`` from the tables' folder was written by older code."""
    reasons = set()
    for path, source, _ in files:
        if source != "tables":
            continue
        if path.name in STAMPED:
            stamped, scripts = STAMPED[path.name]
            why = stale(tables / stamped, scripts)
        else:
            why = stale(tables / "tables.md", TABLES_SCRIPTS)
        if why:
            reasons.add(why)
    return sorted(reasons)


def inputs(tables: Path) -> list[Path]:
    """Every file results.zip is built from, the manifest included."""
    return [MANIFEST, *(path for path, _, _ in entries(tables))]


def build(tables: Path, out: Path, manifest: Path = MANIFEST,
          roots: dict | None = None) -> list[str]:
    """Write ``out`` from ``tables`` (or ``roots``, see :func:`entries`); the
    MANIFEST.txt lines. The files land in ``<out's stem>/``: ``results/`` for
    results.zip, ``sim/`` for run_sim.py's sim.zip."""
    tables = tables.resolve()
    files = entries(tables, manifest, roots)
    reasons = staleness(tables, files)
    if reasons:
        raise SystemExit(f"not building {out.name}; rerun the driver first:\n  "
                         + "\n  ".join(reasons))
    folder = out.stem
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    lines = [f"{out.name}, made {time.strftime('%Y-%m-%d %H:%M')} from commit {commit}",
             *(f"{name}: {path}" for name, path in
               ({"tables": tables} if roots is None else roots).items()), "",
             f"{'file':<62} {'bytes':>10}  {'modified':<16}  used by"]
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, source, used in files:
            archive.write(path, f"{folder}/{path.name}")
            stat = path.stat()
            lines.append(f"{path.name:<62} {stat.st_size:>10,}  "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))}  "
                         f"{used} ({source})")
        archive.writestr(f"{folder}/MANIFEST.txt", "\n".join(lines) + "\n")
    partial.replace(out)
    return lines


def main() -> None:
    args = parse_args()
    lines = build(args.tables, args.out)
    print("\n".join(lines))
    print(f"\nwrote {args.out} ({len(lines) - 4} files)")


if __name__ == "__main__":
    main()
