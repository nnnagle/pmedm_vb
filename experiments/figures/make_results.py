"""Pack the figure scripts and their data into ``results.zip`` for the Quarto project.

Reads ``results_manifest.txt`` (next to this script): each line names a figure
script in the repository or a data file in the paper tables' folder. Every file
lands flat in ``results/`` inside the zip, beside a ``MANIFEST.txt`` that
records where each came from, its size and modification time, and the
repository commit. A manifest line that matches no file stops the script, so a
missing table is noticed here rather than when the paper renders::

    $CONDA_PREFIX/bin/python experiments/figures/make_results.py \\
        --tables $RUNS/paper_v3/tables_compare --out $RUNS/paper_v3/results.zip

Unzipped in the Quarto project's folder, it gives ``results/``, from which a
chunk draws a figure, e.g. ``source("results/fig_cost.R");
plot_cost(cost_data("results"))``.
"""

from __future__ import annotations

import argparse
import subprocess
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MANIFEST = HERE / "results_manifest.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tables", type=Path, required=True,
                        help="the paper tables' folder (paper_tables.py --out)")
    parser.add_argument("--out", type=Path, default=Path("results.zip"))
    return parser.parse_args()


def entries(tables: Path) -> list[tuple[Path, str, str]]:
    """(file, source, used by) for every manifest line, globs expanded."""
    roots = {"repo": REPO, "tables": tables}
    out = []
    for number, line in enumerate(MANIFEST.read_text().splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        source, pattern, *used = line.split(maxsplit=2)
        if source not in roots:
            raise SystemExit(f"{MANIFEST}:{number}: unknown source {source!r}")
        matches = sorted(roots[source].glob(pattern))
        if not matches:
            raise SystemExit(f"{MANIFEST}:{number}: nothing matches {roots[source] / pattern}")
        out += [(path, source, used[0] if used else "") for path in matches]
    names = [path.name for path, _, _ in out]
    clashes = sorted({name for name in names if names.count(name) > 1})
    if clashes:
        raise SystemExit(f"two files would land on the same name in results/: {clashes}")
    return out


def main() -> None:
    args = parse_args()
    files = entries(args.tables.resolve())
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    lines = [f"results.zip, made {time.strftime('%Y-%m-%d %H:%M')} from commit {commit}",
             f"tables: {args.tables.resolve()}", "",
             f"{'file':<62} {'bytes':>10}  {'modified':<16}  used by"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, source, used in files:
            archive.write(path, f"results/{path.name}")
            stat = path.stat()
            lines.append(f"{path.name:<62} {stat.st_size:>10,}  "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(stat.st_mtime))}  "
                         f"{used} ({source})")
        archive.writestr("results/MANIFEST.txt", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out} ({len(files)} files)")


if __name__ == "__main__":
    main()
