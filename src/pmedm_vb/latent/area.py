"""A PUMA's area inputs for the collapsed likelihood (step 2, pieces A3 and A4
of ``PLAN_latent.md``; ``PLAN_latent_step2.md`` version 2, §4.1).

**Published tables (A3).** ``Y``, ``Σ`` and the nesting come from
:func:`~pmedm_vb.assemble.build.build_puma` over the PUMA's constraint tables
(imported, unchanged). ``Y`` is the stacked vector ``[vec(Y_T); vec(Y_B)]``,
column-major. The table covariance (``MODEL_latent.qmd`` version 2) is

    Σ = Σ_survey + ω² I,    ω = N_P / (n_P s),  s = 3,

with ``Σ_survey = PMEDMInputs.sigma(alpha=0.01, taper=None,
variance_floor="zero")``, the standard PMEDM's (untapered: ``D + LL'`` with
``L`` the scaled replicate deviations over every row), and ``ω²`` the
standard PMEDM's multiplier prior in covariance form
(:func:`~pmedm_vb.assemble.prior.prior_variance` with the PUMA build's ``N``
and ``n``, its units and own records), carried as ``Sigma.prior_var``.

**Contributions.** Step 1's statewide ``T`` (:class:`~pmedm_vb.latent.atoms.Contributions`)
is restricted to the constraint cells and its columns mapped one to one onto
``build_puma``'s ``X_T`` and ``X_B`` columns, by name
(:func:`map_columns`). A table published at both levels has identical
columns at each, so the engine works on **base cells**: the distinct columns
among the tract and block-group constraint cells (``J0`` of them). Each tract
cell and each block-group cell points to its base cell (``tract_src``,
``bg_src``); a block-group cell shares the base cell of the tract cell with
the same name only when the two columns of ``T`` are identical.

**Nesting** (``D_b``). Block group ``b`` in tract ``t`` contributes its cell
totals to tract ``t``'s rows (row ``k * n_tracts + t`` for tract cell ``k``)
and to its own block-group rows (``J_T n_tracts + k * n_bg + b``), as
``build_puma``'s ``A_T`` and identity ``A_B`` stack them. Per tract, the
engine uses a local layout: the tract's ``J_T`` rows, then ``J_B`` rows for
each of its block groups in order (:meth:`AreaInputs.tract_rows`).

**Units per block group (A4)**, :func:`block_group_units` (``PLAN_latent_step2.md``
§4.1, the working rule):

* households: the block group's published occupied units, the sum of
  ``B25003``'s leaf cells (owner, renter) in ``Y_B``;
* group-quarters persons: the tract's published group-quarters total
  (``B26001``'s leaf cells in ``Y_T``), split across its block groups in
  proportion to their occupied units (equally when the tract has none).
  ``build_puma`` carries no block-group unit totals, so the split is used.

``N_b`` is their sum, treated as known.

**Step 1's reference** (:func:`load_step1`, :func:`load_basis`). The basis
``h`` is rebuilt over the atoms from what step 1 saved under
``inputs/bases/``: the kept columns, their means and scales, and for a kernel
basis its centres and ``ℓ`` (the other bases are recomputed by
:func:`~pmedm_vb.latent.basis.raw_features`, which is deterministic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.allcells import table_leaves
from pmedm_vb.assemble.build import build_puma
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.prior import prior_variance, with_prior
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.latent.atoms import GQ_COLUMNS, Atoms, Contributions
from pmedm_vb.latent.basis import KERNEL_CENTRES, kernel_features, raw_features

#: Tables giving ``N_b``: occupied units (block group) and group-quarters persons (tract).
HOUSEHOLD_TABLE = "B25003"
GQ_TABLE = "B26001"


# --------------------------------------------------------------------------
# Column mapping
# --------------------------------------------------------------------------


@dataclass
class ColumnMap:
    """``T``'s constraint cells against ``build_puma``'s columns."""

    base_columns: np.ndarray   # (J0,) columns of step 1's T forming the base cells
    tract_src: np.ndarray      # (J_T,) base cell of each tract cell (X_T's order)
    bg_src: np.ndarray         # (J_B,) base cell of each block-group cell (X_B's order)
    checks: dict


def map_columns(contributions: Contributions, tract_names: list[str],
                bg_names: list[str]) -> ColumnMap:
    """Map ``X_T``'s and ``X_B``'s columns (``build_puma``'s names) one to one
    onto ``T``'s constraint cells at the same level, then share base cells
    between levels where the columns are identical."""
    cell = np.array(contributions.cell)
    found = {}
    for level, names in (("tract", tract_names), ("block group", bg_names)):
        block = contributions.block(level, "constraint")
        position = {}
        for j in block:
            if cell[j] in position:
                raise ValueError(f"{level}: cell {cell[j]!r} appears twice in T")
            position[cell[j]] = j
        missing = [n for n in names if n not in position]
        extra = sorted(set(position) - set(names))
        if missing or extra or len(set(names)) != len(names):
            raise ValueError(f"{level}: T's constraint cells and build_puma's columns are not "
                             f"one to one (missing {missing[:5]}, extra {extra[:5]})")
        found[level] = np.array([position[n] for n in names])
    T = contributions.T.tocsc()
    base = list(found["tract"])
    tract_src = np.arange(len(base))
    by_name = {n: k for k, n in enumerate(tract_names)}
    bg_src, shared, differing = [], 0, []
    for name, j in zip(bg_names, found["block group"]):
        k = by_name.get(name)
        if k is not None and (T[:, j] != T[:, base[k]]).nnz == 0:
            bg_src.append(k)
            shared += 1
        else:
            if k is not None:
                differing.append(name)
            base.append(j)
            bg_src.append(len(base) - 1)
    checks = {"J_T": len(tract_names), "J_B": len(bg_names), "J0": len(base),
              "one_to_one": True, "bg_cells_sharing_tract_column": shared,
              "bg_cells_same_name_different_column": differing}
    return ColumnMap(base_columns=np.array(base), tract_src=tract_src,
                     bg_src=np.array(bg_src, dtype=int), checks=checks)


# --------------------------------------------------------------------------
# Units per block group
# --------------------------------------------------------------------------


def block_group_units(problem: PMEDMInputs, bg_tract: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Households and group-quarters persons per block group (module notes)."""
    hh_cols = table_leaves(problem.bg_constraints, HOUSEHOLD_TABLE)
    gq_cols = table_leaves(problem.tract_constraints, GQ_TABLE)
    if not hh_cols:
        raise ValueError(f"no {HOUSEHOLD_TABLE} cells among the block-group constraints")
    if not gq_cols:
        raise ValueError(f"no {GQ_TABLE} cells among the tract constraints")
    households = np.maximum(problem.Y_B[:, hh_cols].sum(axis=1), 0.0)
    gq_tract = np.maximum(problem.Y_T[:, gq_cols].sum(axis=1), 0.0)
    n_tracts = problem.Y_T.shape[0]
    hh_tract = np.bincount(bg_tract, weights=households, minlength=n_tracts)
    n_bg_tract = np.bincount(bg_tract, minlength=n_tracts)
    share = np.where(hh_tract[bg_tract] > 0,
                     households / np.where(hh_tract[bg_tract] > 0, hh_tract[bg_tract], 1.0),
                     1.0 / n_bg_tract[bg_tract])
    return households, gq_tract[bg_tract] * share


# --------------------------------------------------------------------------
# The area
# --------------------------------------------------------------------------


@dataclass
class AreaInputs:
    """One PUMA's ``Y``, ``Σ``, nesting, cell mapping and ``N_b``."""

    puma: str
    tracts: list[str]
    block_groups: list[str]
    bg_tract: np.ndarray          # (B,) tract index of each block group
    tract_cells: list[str]
    bg_cells: list[str]
    columns: ColumnMap
    Y: np.ndarray                 # stacked [vec(Y_T); vec(Y_B)]
    sigma: Sigma                  # Σ_survey + ω² I (prior_var = ω²)
    households: np.ndarray
    gq: np.ndarray
    N_puma: float                 # build_puma's N (the PUMA's PUMS weight total)
    n_puma: int                   # build_puma's n (the PUMA's records)
    checks: dict = field(default_factory=dict)

    @property
    def N_b(self) -> np.ndarray:
        return self.households + self.gq

    @property
    def B(self) -> int:
        return len(self.block_groups)

    @property
    def n_tracts(self) -> int:
        return len(self.tracts)

    @property
    def J_T(self) -> int:
        return len(self.tract_cells)

    @property
    def J_B(self) -> int:
        return len(self.bg_cells)

    @property
    def n_rows(self) -> int:
        return self.Y.size

    def tract_members(self, t: int) -> np.ndarray:
        """Block groups of tract ``t``, in order."""
        return np.flatnonzero(self.bg_tract == t)

    def tract_rows(self, t: int) -> np.ndarray:
        """Rows of the stacked vector in tract ``t``'s local layout: its
        ``J_T`` tract rows, then ``J_B`` rows per block group."""
        tract = np.arange(self.J_T) * self.n_tracts + t
        offset = self.J_T * self.n_tracts
        bgs = [offset + np.arange(self.J_B) * self.B + b for b in self.tract_members(t)]
        return np.concatenate([tract, *bgs])

    def local_positions(self, j: int) -> np.ndarray:
        """Local positions of the ``j``-th block group of a tract: the tract
        rows and its own block-group rows (``D_b`` in the local layout)."""
        return np.concatenate([np.arange(self.J_T),
                               self.J_T + j * self.J_B + np.arange(self.J_B)])

    @property
    def sources(self) -> np.ndarray:
        """Base cell of each of a block group's local positions (tract cells,
        then block-group cells)."""
        return np.concatenate([self.columns.tract_src, self.columns.bg_src])

    def row_labels(self) -> pd.DataFrame:
        """Level, table, cell and area of every stacked row."""
        rows = []
        for k, name in enumerate(self.tract_cells):
            for t, g in enumerate(self.tracts):
                rows.append(("tract", name.split(".", 1)[0], name, g))
        for k, name in enumerate(self.bg_cells):
            for b, g in enumerate(self.block_groups):
                rows.append(("block group", name.split(".", 1)[0], name, g))
        return pd.DataFrame(rows, columns=["level", "table", "cell", "geoid"])


def area_inputs(area, puma: str, tables, contributions: Contributions, *, alpha: float = 0.01,
                variance_floor="zero", prior_sd: float = 3.0,
                zones: pd.DataFrame | None = None) -> AreaInputs:
    """Build PUMA ``puma``'s area inputs and run the §4.1 checks: the column
    mapping one to one (and ``T``'s rows equal to ``X`` for the PUMA's own
    records), the tracts' local layouts partitioning the rows, and
    ``Σ_b N_b`` against the PUMA's ``N``."""
    problem = build_puma(area, str(puma), tables, zones=zones)
    columns = map_columns(contributions, problem.tract_constraints, problem.bg_constraints)
    checks = dict(columns.checks)
    if contributions.serial is not None:
        rows = contributions.serial.get_indexer(problem.units["SERIALNO"].astype(str))
        if (rows < 0).any():
            raise ValueError(f"{int((rows < 0).sum())} of the PUMA's records are not atoms")
        T = contributions.T[rows]
        cells = contributions.block("tract", "constraint"), contributions.block(
            "block group", "constraint")
        by_name = [{n: j for j, n in zip(c, np.array(contributions.cell)[c])} for c in cells]
        diff = 0.0
        for X, names, lookup in ((problem.X_T, problem.tract_constraints, by_name[0]),
                                 (problem.X_B, problem.bg_constraints, by_name[1])):
            cols = [lookup[n] for n in names]
            diff = max(diff, float(abs(T[:, cols] - sp.csr_matrix(X)).max()))
        checks["own_rows_max_abs_T_minus_X"] = diff
    bg_tract = problem.zone_tracts()
    omega2 = prior_variance(prior_sd, problem.n, problem.N)
    sigma = with_prior(problem.sigma(alpha, taper=None, variance_floor=variance_floor), omega2)
    groups = problem.constraint_tracts()
    households, gq = block_group_units(problem, bg_tract)
    out = AreaInputs(
        puma=str(puma), tracts=list(problem.tracts["tract_geoid"].astype(str)),
        block_groups=list(problem.block_groups["block_group_geoid"].astype(str)),
        bg_tract=bg_tract, tract_cells=list(problem.tract_constraints),
        bg_cells=list(problem.bg_constraints), columns=columns, Y=problem.targets(),
        sigma=sigma, households=households, gq=gq, N_puma=float(problem.N),
        n_puma=int(problem.n), checks=checks)
    partition = all(np.array_equal(np.sort(out.tract_rows(t)), np.flatnonzero(groups == t))
                    for t in range(out.n_tracts))
    N_sum = float(out.N_b.sum())
    checks.update({
        "n_tracts": out.n_tracts, "n_block_groups": out.B, "n_rows": out.n_rows,
        "alpha": alpha, "taper": None, "variance_floor": variance_floor,
        "prior_sd_s": prior_sd, "omega": float(np.sqrt(omega2)), "omega2": float(omega2),
        "tract_rows_partition_rows": bool(partition),
        "households_total": float(households.sum()), "gq_total": float(gq.sum()),
        "N_b_sum": N_sum, "N_puma": out.N_puma, "n_puma": out.n_puma,
        "N_b_sum_over_N_puma": N_sum / out.N_puma if out.N_puma > 0 else float("nan"),
        "n_block_groups_N_zero": int((out.N_b <= 0).sum()),
    })
    if not partition:
        raise ValueError("the tracts' local layouts do not partition the rows")
    return out


# --------------------------------------------------------------------------
# Step 1's reference
# --------------------------------------------------------------------------


@dataclass
class Step1:
    """What step 1 hands to step 2 (``STEP1_latent.qmd``, "Record")."""

    root: Path
    h: str
    gamma: float
    sigma: float
    tables: object
    pumas: list[str]
    theta: np.ndarray          # (n_pumas, K) θ̂_P
    q: np.ndarray              # (n_pumas, M) q^P
    serial: pd.Index
    columns: list[str]
    distances: pd.DataFrame
    notes: list[str]

    def theta_of(self, puma: str) -> np.ndarray:
        return self.theta[self.pumas.index(str(puma))]

    def q_of(self, puma: str) -> np.ndarray:
        return self.q[self.pumas.index(str(puma))]

    def tau_ref(self) -> float:
        """``(mean_k Var_P(θ̂_{P,k}))^{1/2}`` over the PUMAs (variance with
        divisor ``n_pumas``)."""
        return float(np.sqrt(self.theta.var(axis=0).mean()))


def load_step1(root: Path) -> Step1:
    """Settings, reference fits and distances from step 1's run root. The
    settings are those the final fits were made with (``final/settings.json``);
    a disagreement with ``decisions.json`` is noted."""
    root = Path(root)
    settings = json.loads((root / "final" / "settings.json").read_text())
    notes = []
    decisions_path = root / "decisions.json"
    if decisions_path.exists():
        decisions = json.loads(decisions_path.read_text())
        for key in ("h", "gamma", "sigma"):
            if key in decisions and decisions[key] != settings[key]:
                notes.append(f"final/settings.json {key} = {settings[key]} differs from "
                             f"decisions.json {decisions[key]}; the final fits' value is used")
    else:
        notes.append("no decisions.json; the final fits' settings are used")
    tables = "all"
    for name in ("grid_final.json", "grid_grid.json"):
        if (root / name).exists():
            tables = json.loads((root / name).read_text()).get("tables", "all")
            break
    ref = np.load(root / "final" / "reference.npz")
    D = pd.read_csv(root / "inputs" / "distances.csv", dtype={"puma": str}).set_index("puma")
    D.columns = D.columns.astype(str)
    return Step1(root=root, h=str(settings["h"]), gamma=float(settings["gamma"]),
                 sigma=float(settings["sigma"]), tables=tables,
                 pumas=[str(p) for p in ref["pumas"]], theta=ref["theta"], q=ref["q"],
                 serial=pd.Index([str(s) for s in ref["serial"]], name="SERIALNO"),
                 columns=[str(c) for c in ref["columns"]], distances=D, notes=notes)


def load_basis(directory: Path, name: str, atoms: Atoms, run: Path) -> np.ndarray:
    """``H`` (``M × K``) of step 1's basis ``name``, rebuilt from
    ``<directory>/<name>.{npz,json}`` (see the module notes)."""
    directory = Path(directory)
    saved = np.load(directory / f"{name}.npz")
    meta = json.loads((directory / f"{name}.json").read_text())
    columns = [str(c) for c in saved["columns"]]
    if name in KERNEL_CENTRES:
        centres = saved["centres"]
        F = np.column_stack([atoms.gq, kernel_features(atoms.z, centres, float(meta["ell"]))])
        names = [*GQ_COLUMNS, *(f"k{j + 1}" for j in range(len(centres)))]
    else:
        F, names, _, _ = raw_features(name, atoms, run)
    position = {c: j for j, c in enumerate(names)}
    missing = [c for c in columns if c not in position]
    if missing:
        raise ValueError(f"basis {name}: saved columns {missing[:5]} not rebuilt")
    F = F[:, [position[c] for c in columns]]
    return np.ascontiguousarray((F - saved["mean"]) / saved["scale"])
