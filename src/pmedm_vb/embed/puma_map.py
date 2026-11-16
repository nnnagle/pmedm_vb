"""Maps of PUMA cluster assignments.

The PUMA boundaries are the Census TIGER/Line 2020 PUMA polygons for the state
(:func:`boundary_url`), fetched by the embedding prefetch on a login node and
read offline from the cache in jobs. They are read here by a small reader of
the zipped shapefile (polygons and the ``GEOID20`` attribute only), so that no
geospatial package is needed: the ISAAC environment has matplotlib (through
arviz) but not geopandas or pyshp.

Coordinates are longitude and latitude (NAD83), drawn with the x axis scaled by
the cosine of the state's mean latitude, which is close enough to equal-area
for a state the size of Tennessee.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea
from pmedm_vb.data.cache import cached_path, fetch_cached

#: TIGER/Line vintage of the 2020 PUMA polygons (the 2020 PUMAs do not change
#: between vintages; 2024 is the PUMS file's last year).
BOUNDARY_YEAR = 2024

#: Distinct colours for up to ten clusters (Okabe-Ito, then two greys);
#: colour-blind safe for the first eight.
PALETTE = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00",
           "#CC79A7", "#000000", "#999999", "#DDDDDD")


def boundary_url(area: StudyArea) -> str:
    return (f"https://www2.census.gov/geo/tiger/TIGER{BOUNDARY_YEAR}/PUMA20/"
            f"tl_{BOUNDARY_YEAR}_{area.state}_puma20.zip")


def boundary_subdir() -> str:
    return f"geo/tiger{BOUNDARY_YEAR}/puma20"


def fetch_boundaries(area: StudyArea) -> Path:
    """Download (or find cached) the PUMA boundary zip; offline-safe."""
    return fetch_cached(boundary_url(area), boundary_subdir())


def cached_boundaries(area: StudyArea) -> Path:
    return cached_path(boundary_url(area), boundary_subdir())


def _read_dbf(raw: bytes) -> list[dict]:
    """Records of a dBase III table: fields of type C/N as stripped strings."""
    n_records = struct.unpack("<I", raw[4:8])[0]
    header_len, record_len = struct.unpack("<HH", raw[8:12])
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        name = raw[pos:pos + 11].split(b"\x00")[0].decode("ascii")
        length = raw[pos + 16]
        fields.append((name, length))
        pos += 32
    records = []
    for r in range(n_records):
        start = header_len + r * record_len + 1   # skip the deletion flag
        rec, offset = {}, start
        for name, length in fields:
            rec[name] = raw[offset:offset + length].decode("latin-1").strip()
            offset += length
        records.append(rec)
    return records


def _read_shp(raw: bytes) -> list[list[np.ndarray]]:
    """Polygon records (shape type 5) of a shapefile as lists of rings."""
    shapes, pos = [], 100
    while pos < len(raw):
        _, length_words = struct.unpack(">ii", raw[pos:pos + 8])
        content = raw[pos + 8:pos + 8 + 2 * length_words]
        pos += 8 + 2 * length_words
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            shapes.append([])
            continue
        if shape_type != 5:
            raise ValueError(f"shape type {shape_type} is not a polygon")
        n_parts, n_points = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts]))
        start = 44 + 4 * n_parts
        points = np.frombuffer(content[start:start + 16 * n_points], dtype="<f8").reshape(-1, 2)
        bounds = [*parts, n_points]
        shapes.append([points[bounds[i]:bounds[i + 1]] for i in range(n_parts)])
    return shapes


def read_boundaries(path: str | Path, key: str = "GEOID20") -> dict[str, list[np.ndarray]]:
    """PUMA GEOID -> list of polygon rings (longitude, latitude), from a zipped
    shapefile."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        shp = z.read(next(n for n in names if n.endswith(".shp")))
        dbf = z.read(next(n for n in names if n.endswith(".dbf")))
    records, shapes = _read_dbf(dbf), _read_shp(shp)
    if len(records) != len(shapes):
        raise ValueError(f"{len(records)} attribute records for {len(shapes)} shapes")
    return {rec[key]: rings for rec, rings in zip(records, shapes)}


def map_panels(spaces) -> tuple[list[str], dict[str, str]]:
    """Map one panel per space: tables, each z and SVD by energy distance, joint."""
    chosen = ["tables", *[s for s in spaces if s.endswith(":energy") and not s.startswith("svd")],
              *[s for s in spaces if s.startswith("svd") and s.endswith(":energy")], "joint"]
    titles = {"tables": "S1 tables", "joint": "joint structure (truth: scoring crosses)"}
    for s in chosen:
        if s.startswith("svd"):
            titles[s] = f"SVD baseline, d = {s.split(':')[0][3:]} (energy)"
        elif s.endswith(":energy"):
            titles[s] = f"z: {s.split(':')[0]} (energy)"
    return chosen, titles


def plot_cluster_maps(boundaries: dict[str, list[np.ndarray]], clusters: pd.DataFrame,
                      spaces: list[str], k: int, out: str | Path,
                      titles: dict[str, str] | None = None) -> Path:
    """One panel per space, PUMAs coloured by their aligned cluster at ``k``.

    ``clusters`` has columns ``puma``, ``space``, ``k``, ``aligned`` (as written
    by the diagnostic). Colours follow the aligned labels, so cluster 1 has the
    same colour in every panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch

    sub = clusters[clusters["k"] == k]
    labels = sorted(sub["aligned"].unique())
    colour = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(labels)}
    allpts = np.vstack([r for rings in boundaries.values() for r in rings])
    lat0 = np.deg2rad(allpts[:, 1].mean())
    xmin, ymin = allpts.min(axis=0)
    xmax, ymax = allpts.max(axis=0)
    width = (xmax - xmin) * np.cos(lat0)
    height = ymax - ymin

    n = len(spaces)
    cols = 1 if n == 1 else 2
    rows = int(np.ceil(n / cols))
    panel_w = 7.0
    panel_h = panel_w * height / width + 0.6
    fig, axes = plt.subplots(rows, cols, figsize=(panel_w * cols, panel_h * rows + 0.8),
                             squeeze=False)
    for ax, space in zip(axes.flat, spaces):
        here = sub[sub["space"] == space].set_index("puma")["aligned"]
        polys, faces = [], []
        for geoid, rings in boundaries.items():
            face = colour.get(here.get(geoid), "#FFFFFF")
            for ring in rings:
                xy = ring.copy()
                xy[:, 0] = xy[:, 0] * np.cos(lat0)
                polys.append(xy)
                faces.append(face)
        ax.add_collection(PolyCollection(polys, facecolors=faces, edgecolors="#333333",
                                         linewidths=0.3))
        ax.set_xlim(xmin * np.cos(lat0), xmax * np.cos(lat0))
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title((titles or {}).get(space, space), fontsize=11)
        missing = sorted(set(boundaries) - set(here.index))
        if missing:
            ax.text(0.01, 0.01, f"{len(missing)} PUMA(s) without a label", fontsize=8,
                    transform=ax.transAxes)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    handles = [Patch(facecolor=colour[g], edgecolor="#333333", label=f"cluster {g}")
               for g in labels]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               frameon=False, fontsize=10)
    fig.suptitle(f"PUMA clusters, k = {k} (average linkage; colours matched to the "
                 f"table-space clusters)", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = Path(out)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
