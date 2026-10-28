"""Shared pieces of the design-based simulation (``experiments/sim/<name>.json``):
the settings, the synthetic PUMAs' names and paths, the smaller constraint set,
and the tabulation of a population ``W`` (zones x units) on every published
line of the source PUMA's tables, constraint and held-out, at both levels.

``sim_build.py`` draws the truths and their surveys and writes the synthetic
inputs; ``run_experiment.py`` fits them (``experiments/sim/<name>_fits.json``);
``sim_collect.py`` tabulates the truths, surveys and fits.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LEVELS = ("block group", "tract")


class Simulation:
    """The settings of ``experiments/sim/<name>.json`` and what follows from them."""

    def __init__(self, path: Path, root: Path) -> None:
        self.settings = json.loads(Path(path).read_text())
        self.root = Path(root)
        self.source = rx.load_experiment(REPO / self.settings["source"]["spec"])
        self.fits = rx.load_experiment(REPO / self.settings["fits"])
        self.puma = self.settings["source"]["puma"]
        self.alpha = float(self.settings["source"]["alpha"])
        self.fraction = float(self.settings["sampling_fraction"])
        self.draws = int(self.settings["draws"])
        expected = self.sim_pumas()
        if self.fits["pumas"] != expected:
            raise SystemExit(f"{self.settings['fits']}: pumas must be {expected[0]} ... "
                             f"{expected[-1]}, as the truths in {path} name them")
        if [float(a) for a in self.fits["alphas"]] != [self.alpha]:
            raise SystemExit(f"{self.settings['fits']}: alphas must be [{self.alpha:g}]")
        os.environ["PMEDM_VB_DATA"] = str(rx.data_dir(self.source))

    # -- names and paths ------------------------------------------------------

    def kinds(self) -> list[str]:
        return list(self.settings["truths"])

    def sim_puma(self, kind: str, i: int) -> str:
        """``<puma><letter><i>``, e.g. ``4701502h07``: the ``i``-th truth (from 1) of ``kind``."""
        return f"{self.puma}{self.settings['puma_letters'][kind]}{i:02d}"

    def sim_pumas(self) -> list[str]:
        return [self.sim_puma(kind, i) for kind in self.kinds()
                for i in range(1, self.settings["truths"][kind] + 1)]

    def source_inputs(self) -> Path:
        from pmedm_vb.config import processed_dir

        return processed_dir() / "inputs" / rx.area_slug(self.source["area"]) / self.puma

    def sim_inputs(self, sim_puma: str) -> Path:
        from pmedm_vb.config import processed_dir

        return processed_dir() / "inputs" / rx.area_slug(self.fits["area"]) / sim_puma

    @property
    def source_folder(self) -> Path:
        return self.root / self.source["name"]

    @property
    def fits_folder(self) -> Path:
        return self.root / self.fits["name"]

    @property
    def folder(self) -> Path:
        """``<root>/<fits name>/simulation``: the truths, the calibration and the output."""
        return self.fits_folder / "simulation"

    def truth_file(self, sim_puma: str) -> Path:
        return self.folder / "truths" / f"{sim_puma}.npz"


# -- the constraint set --------------------------------------------------------


def depths(entry: dict) -> list[int]:
    """Each line's depth in a published tree (the total 0)."""
    parents, out = entry["parents"], []
    for parent in parents:
        out.append(0 if parent < 0 else out[parent] + 1)
    return out


def dropped_lines(trees: dict, tables: list[str]) -> set[str]:
    """``"{table}.{line}"`` of the deepest lines of each of ``tables``."""
    out = set()
    for table in tables:
        entry = trees[table]
        d = depths(entry)
        out |= {f"{table}.{line}" for line, depth in zip(entry["lines"], d) if depth == max(d)}
    return out


def pruned_tree(entry: dict, drop: set[str]) -> dict:
    """``entry`` without the lines named in ``drop`` (bare line names), parents re-indexed."""
    keep = [i for i, line in enumerate(entry["lines"]) if line not in drop]
    new = {old: new for new, old in enumerate(keep)}
    out = dict(entry)
    for key in ("lines", "labels", "additive"):
        out[key] = [entry[key][i] for i in keep]
    out["parents"] = [-1 if entry["parents"][i] < 0 else new[entry["parents"][i]] for i in keep]
    return out


def constraint_columns(sim: Simulation, inputs) -> dict:
    """Per level, the source's constraint columns the simulation fits, and the
    deepest lines dropped (``drop_deepest``)."""
    drop = dropped_lines(inputs.trees, sim.settings["drop_deepest"])
    out = {"dropped": drop}
    for level, names in (("tract", inputs.tract_constraints),
                         ("block group", inputs.bg_constraints)):
        tables = set(sim.settings["constraints"][level])
        out[level] = [k for k, name in enumerate(names)
                      if name.split(".")[0] in tables and name not in drop]
    return out


# -- tabulation ---------------------------------------------------------------


class Tabulator:
    """Every published line of the source PUMA (its constraint tables, all of
    them, and its held-out tables) at both levels, as ``(areas, lines)``
    arrays of a population ``W``."""

    def __init__(self, inputs, heldout) -> None:
        self.inputs = inputs
        self.names = {
            "block group": list(inputs.bg_constraints) + list(heldout.names["block group"]),
            "tract": list(inputs.tract_constraints) + list(heldout.names["tract"])}
        self.heldout = {"block group": set(heldout.names["block group"]),
                        "tract": set(heldout.names["tract"])}
        self.X = {"block group": sp.hstack([inputs.X_B, heldout.X["block group"]]).tocsc(),
                  "tract": sp.hstack([inputs.X_T, heldout.X["tract"]]).tocsc()}
        self.X2 = {level: X.multiply(X).tocsc() for level, X in self.X.items()}
        self.A_T = sp.csr_matrix(inputs.A_T)
        split = inputs.Y_T.size
        v_T = inputs.sigma_v[:split].reshape(inputs.Y_T.shape, order="F")
        v_B = inputs.sigma_v[split:].reshape(inputs.Y_B.shape, order="F")
        #: The published variances of every line (constraint and held-out).
        self.v = {"block group": np.hstack([v_B, heldout.v["block group"]]),
                  "tract": np.hstack([v_T, heldout.v["tract"]])}

    def __call__(self, W: np.ndarray, squared: bool = False) -> dict[str, np.ndarray]:
        X = self.X2 if squared else self.X
        W = np.asarray(W, dtype=float)
        block_group = np.asarray((X["block group"].T @ W.T).T)
        tract = np.asarray(self.A_T @ np.asarray((X["tract"].T @ W.T).T))
        return {"block group": block_group, "tract": tract}


def stacked(values: dict[str, np.ndarray], columns: dict[str, list[int]]) -> np.ndarray:
    """The constraint vector ``[vec(Y_T); vec(Y_B)]`` (column-major) of ``columns``."""
    return np.concatenate([values["tract"][:, columns["tract"]].ravel(order="F"),
                           values["block group"][:, columns["block group"]].ravel(order="F")])


def stacked_rows(inputs, columns: dict[str, list[int]]) -> np.ndarray:
    """The source's stacked rows of ``columns``, in the same order as :func:`stacked`."""
    n_tracts, n_zones, split = inputs.Y_T.shape[0], inputs.Y_B.shape[0], inputs.Y_T.size
    rows_T = (np.asarray(columns["tract"])[:, None] * n_tracts + np.arange(n_tracts)).ravel()
    rows_B = split + (np.asarray(columns["block group"])[:, None] * n_zones
                      + np.arange(n_zones)).ravel()
    return np.concatenate([rows_T, rows_B]).astype(int)
