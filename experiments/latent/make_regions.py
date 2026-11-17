"""Group Tennessee's 58 PUMAs (2020) into regions for the stage 2 regional run.

    python experiments/latent/make_regions.py --out experiments/latent/regions_tn.json \
        [--map regions_tn.png]

A region plays a PUMA's role in stage 2 and its PUMAs play block groups' roles:
the x multipliers are shared across a region and each PUMA gets its own θ on
φ. So a region should be a contiguous group of 3-6 PUMAs that one would
plausibly treat as a single area: a metro core, a metro's suburban ring, or
neighbouring rural PUMAs.

The grouping is written out by hand below (:data:`REGIONS`), from the PUMA
names in the TIGER 2024 PUMA20 file. This script checks it rather than
deriving it: every PUMA in exactly one region, 3-6 PUMAs per region, and each
region contiguous, where two PUMAs are adjacent when their boundary rings share
at least two vertices (TIGER polygons that share an edge share its vertices
exactly). It then writes the region list (id, name, PUMAs, rationale) and,
optionally, a map. Reads the cached boundary zip (fetched by
``experiments/embed/prefetch.py``); needs matplotlib only for ``--map``.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from pmedm_vb.config import StudyArea
from pmedm_vb.embed.puma_map import PALETTE, _read_dbf, cached_boundaries, read_boundaries
from pmedm_vb.latent.geography import adjacency

#: (id, name, PUMA GEOIDs, rationale). Metro cores are kept apart from their
#: suburban rings, as block groups of different character sit within one PUMA;
#: rural PUMAs are grouped with neighbours of similar settlement.
REGIONS = (
    ("memphis_core", "Memphis core",
     ("4702501", "4702502", "4702507", "4702508"),
     "Memphis city PUMAs: Central Riverside, Central Midtown, Southeast, Southwest."),
    ("memphis_ring", "Memphis suburbs and Tipton-Fayette",
     ("4702503", "4702504", "4702505", "4702506", "4702600"),
     "North and east Shelby (Bartlett, Millington, Lakeland, Arlington, Collierville, "
     "Germantown) with the Tipton and Fayette counties PUMA."),
    ("nashville_core", "Nashville-Davidson",
     ("4702401", "4702402", "4702403", "4702404", "4702405", "4702406"),
     "The six Nashville/Davidson County PUMAs."),
    ("nashville_south_west", "Williamson, Maury and Dickson",
     ("4702101", "4702102", "4702200", "4702300"),
     "Southern and western suburbs: Williamson County (2), Maury County, and "
     "Dickson-Cheatham-Hickman."),
    ("nashville_east", "Rutherford, Wilson and Sumner (Hendersonville)",
     ("4702001", "4702002", "4702003", "4701900", "4700501"),
     "Eastern and northern suburbs: Rutherford County (3), Wilson County, and "
     "Hendersonville-Gallatin."),
    ("clarksville", "Clarksville and northwest Middle Tennessee",
     ("4700301", "4700302", "4700401", "4700200"),
     "Montgomery County (2) with Robertson-Sumner (Springfield, Portland) and "
     "Henry-Carroll-Humphreys."),
    ("knoxville_core", "Knox County",
     ("4701501", "4701502", "4701503", "4701504"),
     "The four Knox County PUMAs."),
    ("knoxville_ring", "Knoxville ring",
     ("4701600", "4701700", "4703700", "4701400"),
     "Blount, Anderson-Roane-Morgan, Loudon-Monroe and Sevier-Cocke."),
    ("chattanooga", "Chattanooga and southeast",
     ("4703401", "4703402", "4703403", "4703600", "4703500"),
     "Hamilton County (3) with Bradley-Polk (Cleveland) and McMinn-Rhea-Bledsoe."),
    ("northeast", "Tri-Cities and Morristown",
     ("4701000", "4701101", "4701201", "4700900", "4701300"),
     "Sullivan, Washington (2) and Greene-Hawkins, with Hamblen-Jefferson-Grainger."),
    ("west", "West Tennessee",
     ("4700100", "4702700", "4702800", "4702900"),
     "Dyer-Weakley-Obion, Gibson-Lauderdale-Haywood, Madison-Chester (Jackson) and "
     "Henderson-Hardin-McNairy."),
    ("upper_cumberland", "Upper Cumberland and northern Plateau",
     ("4700600", "4700700", "4701800", "4700800"),
     "Macon-DeKalb-Cannon, Putnam-Overton, Cumberland-White-Fentress and "
     "Campbell-Claiborne-Scott."),
    ("south_central", "South Central",
     ("4703000", "4703100", "4703200", "4703300"),
     "Lawrence-Giles-Wayne, Bedford-Lincoln-Marshall, Coffee-Franklin-Moore and "
     "Warren-Marion-Sequatchie."),
)

MIN_SIZE, MAX_SIZE = 3, 6


def connected(members: tuple[str, ...], adjacent: dict[str, set[str]]) -> bool:
    seen, todo = set(), [members[0]]
    while todo:
        g = todo.pop()
        if g in seen:
            continue
        seen.add(g)
        todo.extend(x for x in adjacent[g] if x in members and x not in seen)
    return seen == set(members)


def check(regions, rings, adjacent) -> list[str]:
    problems = []
    listed = [g for _, _, members, _ in regions for g in members]
    if sorted(listed) != sorted(rings):
        missing = sorted(set(rings) - set(listed))
        extra = sorted(set(listed) - set(rings))
        twice = sorted({g for g in listed if listed.count(g) > 1})
        problems.append(f"coverage: missing {missing}, unknown {extra}, repeated {twice}")
    for rid, _, members, _ in regions:
        if not MIN_SIZE <= len(members) <= MAX_SIZE:
            problems.append(f"{rid}: {len(members)} PUMAs, outside {MIN_SIZE}-{MAX_SIZE}")
        if all(g in rings for g in members) and not connected(members, adjacent):
            problems.append(f"{rid}: not contiguous")
    return problems


def draw(regions, rings, names, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = [c for c in PALETTE if c not in ("#000000", "#999999", "#DDDDDD")]
    palette += ["#CC6677", "#88CCEE", "#117733", "#DDCC77", "#AA4499", "#882255", "#44AA99"]
    colour = {}
    for k, (_, _, members, _) in enumerate(regions):
        for g in members:
            colour[g] = palette[k % len(palette)]
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=130)
    for g, r in rings.items():
        for ring in r:
            ax.fill(ring[:, 0], ring[:, 1], facecolor=colour[g], edgecolor="white", linewidth=0.6)
    for rid, name, members, _ in regions:
        pts = np.vstack([ring for g in members for ring in rings[g]])
        ax.annotate(name, pts.mean(0), ha="center", va="center", fontsize=7.5,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "alpha": 0.75, "lw": 0})
    ax.set_aspect(1 / np.cos(np.deg2rad(36)))
    ax.axis("off")
    ax.legend(handles=[Patch(color=colour[m[0]], label=f"{n} ({len(m)})")
                       for _, n, m, _ in regions],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=8, frameon=False)
    ax.set_title("Stage 2 regional run: regions of Tennessee PUMAs (2020)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("experiments/latent/regions_tn.json"))
    parser.add_argument("--map", type=Path, default=None, help="PNG of the regions")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    area = StudyArea(name="state", state="47", year=args.year)
    zpath = cached_boundaries(area)
    rings = read_boundaries(zpath)
    with zipfile.ZipFile(zpath) as z:
        dbf = z.read(next(n for n in z.namelist() if n.endswith(".dbf")))
    names = {r["GEOID20"]: r["NAMELSAD20"] for r in _read_dbf(dbf)}
    adjacent = adjacency(rings)
    problems = check(REGIONS, rings, adjacent)
    if problems:
        raise SystemExit("region list fails its checks:\n  " + "\n  ".join(problems))

    payload = {
        "state": "47", "puma_vintage": 2020, "boundaries": zpath.name,
        "regions": [{"id": rid, "name": name, "pumas": list(members),
                     "puma_names": [names[g] for g in members], "rationale": why}
                    for rid, name, members, why in REGIONS],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"{len(REGIONS)} regions, {sum(len(m) for _, _, m, _ in REGIONS)} PUMAs -> {args.out}")
    for rid, name, members, _ in REGIONS:
        print(f"  {rid:22s} {len(members)}  {name}")
    if args.map is not None:
        draw(REGIONS, rings, names, args.map)
        print(f"map -> {args.map}")


if __name__ == "__main__":
    main()
