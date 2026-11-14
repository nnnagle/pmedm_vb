"""Leave-one-PUMA-out check: can a PUMA's units be recovered by tilting the
other PUMAs' units to its tables?

The PUMS records each unit's PUMA, which gives a truth for joint distributions
that no block-group table provides. For each PUMA P:

* **Support**: the units of every other PUMA, base weights ``q`` their design
  weights (``WGTP`` for households, ``PWGTP`` for group-quarters persons, as
  :func:`~pmedm_vb.assemble.households.build_units` defines them).
* **Truth**: P's own units with their design weights; ``N`` is P's weighted
  unit total.
* **Targets**: P's table cells computed from P's own records with the same
  unit weights (so person tables are counted with household weights, as the
  PMEDM's unit frame counts them), **unperturbed**. Standard errors come from
  the 80 replicate weights (``WGTP1``-``80`` for households, the occupant's
  ``PWGTP1``-``80`` for group quarters): SE² = (4/80) Σ_r (X_r − X)², the
  ACS successive-difference formula.
* **SE floor**: a cell whose replicate SE is below P's mean unit weight -- a
  zero cell, or one no replicate moves -- gets that mean weight as its SE: the
  size of one sampled case, the smallest difference the sample can resolve.

PUMA is used only here, never in fitting the embedding, so one embedding
serves every PUMA, and P's records are excluded from the support it is
recovered from.

**Tilts** (:mod:`pmedm_vb.embed.tilt`), each fitted per constraint set:

``none``    the untilted support (q alone), for reference;
``x``       features = the unit's counts in the set's cells (the PMEDM's form);
``phi``     features = h(φ), h linear, φ including the group-quarters indicators;
``both``    features = [x, h(φ)];
``oracle``  φ only, β such that the tilted mean of h(φ) equals P's true mean
            exactly (independent of the constraint set; what the
            representation could reach if the tables identified θ exactly).

**Feature maps h** (:func:`feature_maps`; ``phi``, ``both`` and ``oracle`` are
run for each requested h, labelled ``phi``/``both``/``oracle`` for linear and
with a suffix otherwise, so results for different h never mix):

``linear``     φ itself: the two group-quarters indicators and z1..zd;
``quadratic``  linear plus every square and cross-product of z (not of the
               indicators), d(d+1)/2 more (``_quad``);
``decoder``    the indicators plus the run's calibrated decoder's predicted
               category probabilities for a set of attributes, one category
               per attribute dropped (``_dec``).

A linear tilt exp(−θ′φ) is monotone along every direction of φ: it cannot
raise an interior region of φ-space, change the dispersion of positions, or
represent interactions. Squares and cross-products can; decoder probabilities
give each category's region of φ-space, interior ones included, its own
coefficient. Non-indicator features are design-weighted standardised over
the support (:mod:`pmedm_vb.embed.tilt`, *feature scale*), for every h
including linear, so the prior sd τ means the same thing across h.

``x``, ``phi`` and ``both`` share one MAP objective (Gaussian error on the
cells with the replicate SEs, prior sd τ = 3 on β); see
:func:`~pmedm_vb.embed.tilt.fit_tilt`. The PMEDM's own penalized dual for one
zone is not run: the solvers need the zone structure and VRE ``Sigma`` of a
real PMEDM input, which this check does not build.

**Constraint sets** are nested (:data:`S0`, :data:`S1_ADD`, S2 = S1 plus the
added tables of the configuration less :data:`S2_EXCLUDE` and the scoring
set). A published table whose cells are unions of an earlier set's cells is
not added again: under independent errors it would enter the same data twice.

**Scoring tables** (:data:`SCORING_TABLES`, :data:`SCORING_CROSSES`) are never
in any constraint set; this is checked. Per table: total variation distance
and Kullback-Leibler divergence KL(truth ‖ prediction) between the
within-table distributions, and for two-way tables the error in log odds
ratios against reference categories (:func:`log_odds_ratios`), which isolates
joint from marginal error: a tilt on one-way margins alone leaves the
support's odds ratios unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.constraints import (
    NOT_HISPANIC,
    RAC1P_TO_RACE,
    RACE_POVERTY_TABLES,
    Category,
    ConstraintTable,
)
from pmedm_vb.assemble.households import build_units
from pmedm_vb.config import StudyArea
from pmedm_vb.data.pums import load_pums, weight_columns
from pmedm_vb.embed.config import EmbedConfig, resolve, table_kind
from pmedm_vb.embed.model import EMPTY_LOGIT
from pmedm_vb.embed.tilt import Features, fit_oracle, fit_tilt
from pmedm_vb.embed.units import GQ_TYPES, _table_counts
from pmedm_vb.progress import logger, stage

# --------------------------------------------------------------------------
# Sets
# --------------------------------------------------------------------------

#: S0: one-way margins of key attributes (attribute targets of configuration
#: v2, plus the unit type).
S0 = ("age_sex", "race_ethnicity", "income", "tenure", "household_type",
      "household_size", "unit_type")

#: Added to S0 to make S1: the PMEDM constraint tables S0 does not contain.
S1_ADD = ("B08301", "C24010", "B17024", "B23001")

#: PMEDM constraint tables left out of S1 because their cells are unions of
#: S0's: B01001 of age_sex, B03002 of race_ethnicity, B19001 of income,
#: B25003 of tenure, B26001 of the unit type.
S1_COARSENINGS = ("B01001", "B03002", "B19001", "B25003", "B26001")

#: Added tables kept out of S2: B11001 is a coarsening of household_type.
S2_EXCLUDE = ("B11001",)

#: Scoring tables (never constrained): the PMEDM held-out tables.
SCORING_TABLES = ("B25044", "C17002", "C24030", "B12001", "B15002", "B25040",
                  *RACE_POVERTY_TABLES)

#: Two-way scoring crosses of household attributes: (row table, column table).
#: ``hh_race8`` is the householder's race/ethnicity in B03002's eight
#: categories (Hispanic of any race; not Hispanic by race).
SCORING_CROSSES = (("hh_race8", "B19001"), ("hh_race8", "B25003"),
                   ("B25003", "B19001"), ("household_type", "B19001"))

#: Tables computed only for reporting (rare groups, published comparison).
REPORT_TABLES = ("B01001", "B03002", "B19001", "B25003", "B26001")

#: Scoring tables with a two-way structure read from their category names:
#: the part before the first underscore is the row (tenure or sex).
PREFIX_TWO_WAY = ("B25044", "B12001", "B15002", "C24030")

#: The race-alone iterations of B17010, combined into one race x poverty table.
B17010_RACE_ALONE = tuple(f"B17010{s}" for s in "ABCDEFG")

#: Rare groups reported against truth and, where published, the PUMA tables.
RARE = (
    ("B03002", "nh_aian"), ("B03002", "nh_asian"), ("B03002", "nh_nhpi"),
    ("B03002", "nh_other_race"),
    ("B01001", "male_80_to_85"), ("B01001", "male_ge_85"),
    ("B01001", "female_80_to_85"), ("B01001", "female_ge_85"),
    ("B26001", "group_quarters"),
    ("unit_type", "institutional"), ("unit_type", "noninstitutional"),
    ("B17010C", "below_poverty"), ("B17010C", "at_or_above_poverty"),
    ("B17010E", "below_poverty"), ("B17010E", "at_or_above_poverty"),
)

TILTS = ("none", "x", "phi", "both", "oracle")

#: Feature maps h, with the suffix their tilts carry.
FEATURE_MAPS = {"linear": "", "quadratic": "_quad", "decoder": "_dec"}

#: Tilts that depend on h; ``none`` and ``x`` do not.
H_TILTS = ("phi", "both", "oracle")

#: Default attributes of the decoder feature map: S0's attributes and the
#: householder's race and Hispanic origin.
DECODER_ATTRIBUTES = ("age_sex", "race_ethnicity", "income", "tenure", "household_type",
                      "household_size", "householder_race_ethnicity")


def tilt_label(tilt: str, h: str) -> str:
    """``phi``, ``both``, ``oracle`` carry h's suffix; ``none`` and ``x`` none."""
    return tilt + FEATURE_MAPS[h] if tilt in H_TILTS else tilt


def unit_type_table() -> ConstraintTable:
    """Household, institutional and noninstitutional group quarters."""
    codes = {"household": "1", **GQ_TYPES}
    return ConstraintTable(
        table="unit_type", universe="household", geography="none", variables=(),
        categories=tuple(Category(name=name, select=(lambda c: lambda u: u["TYPEHUGQ"] == c)(code))
                         for name, code in codes.items()),
    )


def householder_race8() -> ConstraintTable:
    """The householder's race/ethnicity in ``B03002``'s eight categories."""
    def nh(codes):
        return lambda u: u["householder_RAC1P"].isin(codes) & (u["householder_HISP"] == NOT_HISPANIC)

    categories = [Category(name=f"nh_{race}", select=nh(codes))
                  for race, codes in RAC1P_TO_RACE.items()]
    categories.append(Category(
        name="hispanic",
        select=lambda u: u["householder_HISP"].notna() & (u["householder_HISP"] != NOT_HISPANIC)))
    return ConstraintTable(table="hh_race8", universe="household", geography="none",
                           variables=(), categories=tuple(categories),
                           householder=("RAC1P", "HISP"))


def cross(a: ConstraintTable, b: ConstraintTable) -> ConstraintTable:
    """Two household-universe tables crossed; categories named ``"row|col"``."""
    if a.universe != "household" or b.universe != "household":
        raise ValueError(f"cross needs household tables: {a.table}, {b.table}")
    categories = tuple(
        Category(name=f"{ca.name}|{cb.name}",
                 select=(lambda x, y: lambda u: x.select(u) & y.select(u))(ca, cb))
        for ca in a.categories for cb in b.categories)
    return ConstraintTable(
        table=f"{a.table}x{b.table}", universe="household", geography="none",
        variables=tuple(dict.fromkeys((*a.variables, *b.variables))),
        categories=categories,
        householder=tuple(dict.fromkeys((*a.householder, *b.householder))))


@dataclass
class Sets:
    """Constraint sets (cumulative table lists) and the scoring tables."""

    constraint: dict[str, tuple[str, ...]]
    scoring: tuple[str, ...]


def constraint_sets(config: EmbedConfig) -> Sets:
    added = tuple(t for t in config.tables
                  if table_kind(t) == "added" and t not in S2_EXCLUDE)
    s1 = (*S0, *S1_ADD)
    sets = {"S0": S0, "S1": s1, "S2": (*s1, *added)}
    scoring = (*SCORING_TABLES, *(f"{a}x{b}" for a, b in SCORING_CROSSES))
    for name, tables in sets.items():
        clash = set(tables) & set(scoring)
        if clash:
            raise ValueError(f"{name} constrains scoring tables {sorted(clash)}")
    return Sets(constraint=sets, scoring=scoring)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass
class CheckData:
    """Statewide units, their PUMA, weights, replicate weights and counts."""

    units: pd.DataFrame
    weight: np.ndarray
    replicates: np.ndarray
    puma: np.ndarray
    counts: dict[str, sp.csr_matrix]
    tables: dict[str, ConstraintTable]
    sets: Sets
    two_way: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def pumas(self) -> list[str]:
        return sorted(set(self.puma))


def _replicate_weights(area: StudyArea, units: pd.DataFrame) -> np.ndarray:
    """Units x 80: ``WGTP1``-``80`` for households, the occupant's
    ``PWGTP1``-``80`` for group quarters."""
    housing = load_pums(area, [], record_type="housing").set_index("SERIALNO")
    reps_h = list(weight_columns("housing")[1:])
    out = housing.reindex(units.index)[reps_h].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64, copy=True)
    gq = units["is_group_quarters"].to_numpy()
    if gq.any():
        persons = load_pums(area, [], record_type="person")
        persons = persons[persons["SERIALNO"].isin(set(units.index[gq]))]
        reps_p = list(weight_columns("person")[1:])
        occupant = persons.set_index("SERIALNO")[reps_p].apply(pd.to_numeric, errors="coerce")
        out[gq] = occupant.reindex(units.index[gq]).to_numpy()
    if np.isnan(out).any():
        raise ValueError(f"{int(np.isnan(out).any(axis=1).sum())} unit(s) without replicate weights")
    return out


def load_check_data(config: EmbedConfig) -> CheckData:
    """Build every table the check needs over the statewide units."""
    area = config.area
    sets = constraint_sets(config)
    needed = list(dict.fromkeys(
        t for t in (*sets.constraint["S2"], *SCORING_TABLES, *REPORT_TABLES,
                    *(x for pair in SCORING_CROSSES for x in pair))
        if t not in ("unit_type", "hh_race8")))
    with stage(f"resolve {len(needed)} tables"):
        tables = resolve(area, needed)
    tables["unit_type"] = unit_type_table()
    tables["hh_race8"] = householder_race8()
    for a, b in SCORING_CROSSES:
        tables[f"{a}x{b}"] = cross(tables[a], tables[b])

    with stage("build statewide units") as step:
        frame = build_units(area, None, list(tables.values()))
        step.detail = f"{frame.n_units:,} units"
    units, persons = frame.units, frame.persons
    person_row = units.index.get_indexer(persons["SERIALNO"])

    counts = {}
    with stage(f"count {len(tables)} tables"):
        for name, table in tables.items():
            counts[name] = sp.csr_matrix(_table_counts(table, units, persons, person_row))

    # B17010A-G partition families by the householder's race alone, so they
    # stack into one race x poverty table.
    counts["B17010A-G"] = sp.hstack([counts[t] for t in B17010_RACE_ALONE]).tocsr()
    two_way = {
        "B17010A-G": (np.repeat(np.arange(len(B17010_RACE_ALONE)), 2),
                      np.tile(np.arange(2), len(B17010_RACE_ALONE)))}
    for name in PREFIX_TWO_WAY:
        two_way[name] = _two_way_from_names([c.name for c in tables[name].categories], "_")
    for a, b in SCORING_CROSSES:
        name = f"{a}x{b}"
        two_way[name] = _two_way_from_names([c.name for c in tables[name].categories], "|")
    scoring = (*sets.scoring, "B17010A-G")
    sets = replace(sets, scoring=scoring)

    with stage("replicate weights"):
        replicates = _replicate_weights(area, units)
    return CheckData(
        units=units, weight=units["weight"].to_numpy(dtype=np.float64), replicates=replicates,
        puma=units["puma_geoid"].astype(str).to_numpy(), counts=counts, tables=tables,
        sets=sets, two_way=two_way)


def _two_way_from_names(names, sep) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = zip(*(n.split(sep, 1) for n in names))
    row_codes = pd.Index(list(dict.fromkeys(rows)))
    col_codes = pd.Index(list(dict.fromkeys(cols)))
    return row_codes.get_indexer(rows), col_codes.get_indexer(cols)


def load_phi(path: Path, units: pd.Index, columns=None) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Positions keyed by ``SERIALNO`` from a parquet (or CSV) file, aligned to
    ``units``; by default the group-quarters indicators and ``z1``... Returns
    the matrix, its column names and which columns are 0/1 indicators."""
    path = Path(path)
    if path.is_dir():
        path = path / "phi_units.parquet"
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    frame["SERIALNO"] = frame["SERIALNO"].astype(str)
    frame = frame.set_index("SERIALNO")
    if columns is None:
        columns = [c for c in ("gq_institutional", "gq_noninstitutional") if c in frame]
        columns += sorted((c for c in frame if c.startswith("z") and c[1:].isdigit()),
                          key=lambda c: int(c[1:]))
    missing = units.difference(frame.index)
    if len(missing):
        raise ValueError(f"{len(missing)} unit(s) have no position in {path}")
    phi = frame.reindex(units)[columns].to_numpy(dtype=np.float64)
    binary = np.array([set(np.unique(phi[:, j])) <= {0.0, 1.0} for j in range(phi.shape[1])])
    return phi, list(columns), binary


@dataclass
class FeatureMap:
    """One feature map h(φ) over the statewide units."""

    name: str
    F: np.ndarray
    columns: list[str]
    binary: np.ndarray


def quadratic_features(phi: np.ndarray, columns: list[str], binary: np.ndarray) -> FeatureMap:
    """φ plus every square and cross-product of its non-indicator columns."""
    z = np.flatnonzero(~binary)
    products, names = [], []
    for a, j in enumerate(z):
        for k in z[a:]:
            products.append(phi[:, j] * phi[:, k])
            names.append(f"{columns[j]}*{columns[k]}")
    F = np.column_stack([phi, *products]) if products else phi
    return FeatureMap("quadratic", F, [*columns, *names],
                      np.concatenate([binary, np.zeros(len(names), dtype=bool)]))


def decoder_features(run: Path, phi: np.ndarray, columns: list[str], binary: np.ndarray,
                     weight: np.ndarray, attributes=DECODER_ATTRIBUTES) -> FeatureMap:
    """The group-quarters indicators plus the run's calibrated decoder's
    predicted category probabilities p_ak(φ) for ``attributes`` (``"all"``:
    every target).

    The calibrated decoder (``W_cal``, ``b_cal``, ``O_cal`` in ``decoder.npz``)
    acts on the whitened coordinates that ``phi_units.parquet`` holds, so the
    probabilities are those check 1 scores. They are computed for every unit,
    including units outside an attribute's universe (a group-quarters person's
    tenure probabilities, say): they are features of the position, not
    targets. Per attribute, empty categories and the one with the largest
    weighted mean probability (the reference) are dropped, since an
    attribute's probabilities sum to one.
    """
    run = Path(run)
    decoder = np.load(run / "decoder.npz")
    meta = _targets_meta(run)
    by_id = {t["id"]: t for t in meta["targets"]}
    if attributes == "all" or attributes == ("all",) or attributes == ["all"]:
        attributes = tuple(by_id)
    missing = [a for a in attributes if a not in by_id]
    if missing:
        raise ValueError(f"decoder attributes not in the run's targets: {missing}")
    gq_cols = [j for j, c in enumerate(columns) if c in ("gq_institutional", "gq_noninstitutional")]
    z_cols = [j for j in range(len(columns)) if not binary[j]]
    gq = np.zeros((len(phi), 2))
    for j in gq_cols:
        gq[:, 0 if columns[j] == "gq_institutional" else 1] = phi[:, j]
    z = phi[:, z_cols]
    W, b, O = decoder["W_cal"], decoder["b_cal"], decoder["O_cal"]
    empty = decoder["empty"].astype(bool)
    if W.shape[1] != z.shape[1]:
        raise ValueError(f"decoder has {W.shape[1]} coordinates, positions have {z.shape[1]}")
    blocks, names = [phi[:, gq_cols]], [columns[j] for j in gq_cols]
    for attribute in attributes:
        t = by_id[attribute]
        start, end = int(t["start"]), int(t["end"])
        logits = z @ W[start:end].T + b[start:end] + gq @ O[start:end].T \
            + EMPTY_LOGIT * empty[start:end]
        logits -= logits.max(1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(1, keepdims=True)
        mean = weight @ prob / weight.sum()
        keep = ~empty[start:end]
        keep[int(np.argmax(np.where(keep, mean, -1.0)))] = False
        blocks.append(prob[:, keep])
        names += [f"{attribute}.{c}" for c, k in zip(t["categories"], keep) if k]
    F = np.column_stack(blocks)
    return FeatureMap("decoder", F, names,
                      np.array([j < len(gq_cols) for j in range(F.shape[1])]))


def _targets_meta(run: Path) -> dict:
    """The target metadata (ids, categories, column ranges) the run was fitted
    on, from the cached target arrays its ``version.json`` names."""
    import json

    from pmedm_vb.config import processed_dir

    version = json.loads((run / "version.json").read_text())
    name = json.loads((run / "config.json").read_text())["name"]
    path = processed_dir() / "embed" / "targets" / f"{name}_{version['targets_key']}" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"target metadata for {run} not found at {path}")
    return json.loads(path.read_text())


def feature_maps(h: tuple[str, ...], phi: np.ndarray, columns: list[str], binary: np.ndarray,
                 *, run: Path | None = None, weight: np.ndarray | None = None,
                 decoder_attributes=DECODER_ATTRIBUTES) -> dict[str, FeatureMap]:
    """The requested feature maps, keyed by name (see the module notes)."""
    unknown = [x for x in h if x not in FEATURE_MAPS]
    if unknown:
        raise ValueError(f"unknown feature maps {unknown}; known: {list(FEATURE_MAPS)}")
    out = {}
    for name in h:
        if name == "linear":
            out[name] = FeatureMap("linear", phi, list(columns), np.asarray(binary))
        elif name == "quadratic":
            out[name] = quadratic_features(phi, columns, np.asarray(binary))
        else:
            if run is None or weight is None:
                raise ValueError("the decoder feature map needs the run directory and weights")
            out[name] = decoder_features(Path(run), phi, columns, np.asarray(binary), weight,
                                         decoder_attributes)
    return out


def support_scale(F: np.ndarray, binary: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Design-weighted sd of each column over the support (weights ``q``, zero
    off the support); 1 for indicators and for columns constant there."""
    w = q / q.sum()
    mean = w @ F
    sd = np.sqrt(np.maximum(w @ np.square(F) - np.square(mean), 0.0))
    return np.where(binary | (sd <= 1e-12), 1.0, sd)


# --------------------------------------------------------------------------
# Targets and scores
# --------------------------------------------------------------------------


def targets(data: CheckData, tables, in_p: np.ndarray, floor: float):
    """Stacked ``X`` (all units), ``y`` and ``se`` of P for ``tables``."""
    X = sp.hstack([data.counts[t] for t in tables]).tocsr()
    Xp = X[in_p]
    y = Xp.T @ data.weight[in_p]
    reps = np.asarray((Xp.T @ data.replicates[in_p]))
    se = np.sqrt(4.0 / 80.0 * np.square(reps - y[:, None]).sum(axis=1))
    return X, y, np.maximum(se, floor)


def _reference(table: np.ndarray) -> tuple[int, int]:
    """Reference row and column of a two-way table: its largest margins."""
    return int(np.argmax(table.sum(1))), int(np.argmax(table.sum(0)))


def _contrasts(values: np.ndarray, r: int, s: int) -> np.ndarray:
    keep = np.ones_like(values, dtype=bool)
    keep[r, :] = False
    keep[:, s] = False
    return values[keep]


def log_odds_ratios(table: np.ndarray, continuity: float,
                    reference: tuple[int, int] | None = None) -> np.ndarray:
    """Log odds ratios of a two-way table against a reference row r and column
    s (by default its largest margins):
    LOR_jk = log((n_jk + c)(n_rs + c) / ((n_js + c)(n_rk + c))) for every
    j != r, k != s. ``c`` is a continuity correction on the table's scale."""
    r, s = reference if reference is not None else _reference(table)
    n = table + continuity
    lor = np.log(n) + np.log(n[r, s]) - np.log(n[:, [s]]) - np.log(n[[r], :])
    return _contrasts(lor, r, s)


def lor_precision(unweighted: np.ndarray, reference: tuple[int, int]) -> np.ndarray:
    """1 / Σ 1/(u + 0.5) over each odds ratio's four cells, from P's sample
    (unweighted) counts: the approximate precision of the true log odds
    ratio, on the same contrasts as :func:`log_odds_ratios`."""
    r, s = reference
    u = unweighted + 0.5
    var = 1 / u + 1 / u[r, s] + 1 / u[:, [s]] + 1 / u[[r], :]
    return _contrasts(1 / var, r, s)


def _as_grid(values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    grid = np.zeros((rows.max() + 1, cols.max() + 1))
    np.add.at(grid, (rows, cols), values)
    return grid


def score_tables(data: CheckData, p: np.ndarray, N: float, in_p: np.ndarray,
                 continuity: float) -> list[dict]:
    """TVD, KL and log-odds-ratio error of every scoring table."""
    rows = []
    for name in data.sets.scoring:
        C = data.counts[name]
        truth = C[in_p].T @ data.weight[in_p]
        pred = N * (C.T @ p)
        row = {"table": name, "truth_total": float(truth.sum()), "pred_total": float(pred.sum())}
        if truth.sum() > 0 and pred.sum() > 0:
            t, m = truth / truth.sum(), pred / pred.sum()
            row["tvd"] = 0.5 * float(np.abs(t - m).sum())
            pos = t > 0
            row["kl"] = float((t[pos] * np.log(t[pos] / np.maximum(m[pos], 1e-300))).sum())
        if name in data.two_way and truth.sum() > 0:
            ri, ci = data.two_way[name]
            truth_grid = _as_grid(truth, ri, ci)
            pred_grid = _as_grid(pred, ri, ci)
            unweighted = _as_grid(np.asarray(C[in_p].sum(axis=0)).ravel(), ri, ci)
            reference = _reference(truth_grid)
            lt = log_odds_ratios(truth_grid, continuity, reference)
            lp = log_odds_ratios(pred_grid, continuity, reference)
            w = lor_precision(unweighted, reference)
            err = lp - lt
            row["lor_n"] = int(err.size)
            row["lor_mae"] = float(np.abs(err).mean()) if err.size else np.nan
            row["lor_wrms"] = float(np.sqrt((w * err**2).sum() / w.sum())) if err.size else np.nan
        rows.append(row)
    return rows


def misfit(X: sp.csr_matrix, y: np.ndarray, se: np.ndarray, p: np.ndarray, N: float) -> dict:
    z = (y - N * (X.T @ p)) / se
    return {"rms_z": float(np.sqrt(np.mean(z**2))), "share_within_1se": float(np.mean(np.abs(z) <= 1)),
            "max_abs_z": float(np.max(np.abs(z))), "n_cells": int(z.size)}


def rare_groups(data: CheckData, p: np.ndarray, N: float, in_p: np.ndarray,
                published: pd.DataFrame | None, puma: str) -> list[dict]:
    rows = []
    for table, category in RARE:
        names = [c.name for c in data.tables[table].categories]
        j = names.index(category)
        col = data.counts[table][:, j]
        truth = float((col[in_p].T @ data.weight[in_p])[0])
        pred = float(N * (col.T @ p)[0])
        row = {"table": table, "category": category, "truth": truth, "pred": pred}
        cells = data.tables[table].categories[j].published
        if published is not None and cells and puma in published.index:
            est = published.loc[puma, [("estimate", c) for c in cells]].astype(float)
            moe = published.loc[puma, [("moe", c) for c in cells]].astype(float)
            if not (est.isna().any() or moe.isna().any()):
                se = float(np.sqrt(np.square(moe.to_numpy() / 1.645).sum()))
                row.update(published=float(est.sum()), published_se=se)
                if se > 0:
                    row.update(z_pred=(pred - row["published"]) / se,
                               z_truth=(truth - row["published"]) / se)
        rows.append(row)
    return rows


def published_tables(area: StudyArea) -> pd.DataFrame | None:
    """PUMA-level estimates and MOEs of the rare groups' tables, from the
    cache (offline); ``None`` with a warning when they are not cached."""
    tables = sorted({t for t, _ in RARE if t != "unit_type"})
    try:
        from pmedm_vb.data.acs_api import fetch_api_tables

        return fetch_api_tables(area, tables, geography="puma")
    except Exception as error:  # noqa: BLE001 -- reported, the check goes on
        logger.warning("published PUMA tables unavailable (%s); rare groups are "
                       "scored against PUMS truth only", error)
        return None


# --------------------------------------------------------------------------
# One PUMA
# --------------------------------------------------------------------------


def check_puma(data: CheckData, puma: str, maps: dict[str, FeatureMap],
               published: pd.DataFrame | None, *, tau: float = 3.0,
               sets: tuple[str, ...] = ("S0", "S1", "S2"),
               tilts: tuple[str, ...] = TILTS, skip: frozenset = frozenset(),
               standardise: bool = True, solver: str = "auto") -> dict[str, pd.DataFrame]:
    """Every requested tilt for every constraint set and feature map for PUMA
    ``puma``. ``skip`` holds ``"tilt_label|set"`` keys already done, which are
    not refitted. ``standardise=False`` leaves the prior on the raw feature
    scale (the behaviour before feature maps were added). ``solver``:
    ``"auto"`` fits ``x`` and the linear tilts by L-BFGS, as before feature
    maps were added, and the quadratic and decoder tilts by
    Levenberg-Marquardt, which converges where L-BFGS stalls on their many
    nearly collinear features (same objective and minimiser:
    :func:`~pmedm_vb.embed.tilt.fit_tilt`); ``"lbfgs"`` or ``"lm"`` use one
    solver throughout."""
    in_p = data.puma == puma
    if not in_p.any():
        raise ValueError(f"no units in PUMA {puma}")
    log_q = np.where(in_p, -np.inf, np.log(np.maximum(data.weight, 1e-300)))
    N = float(data.weight[in_p].sum())
    floor = float(data.weight[in_p].mean())
    continuity = 0.5 * floor
    q_support = np.where(in_p, 0.0, data.weight)

    # (tilt label, h or None, tilt) for everything still to do in some set.
    work = []
    for tilt in tilts:
        for h in (maps if tilt in H_TILTS else [None]):
            label = tilt if h is None else tilt_label(tilt, h)
            todo = [s_ for s_ in sets if f"{label}|{s_}" not in skip]
            if todo:
                work.append((label, h, tilt, todo))
    scores, misfits, rare = [], [], []
    if not work:
        return {"scores": pd.DataFrame(), "misfit": pd.DataFrame(), "rare": pd.DataFrame()}

    def record(set_name, label, p, X, y, se, info):
        base = {"puma": puma, "set": set_name, "tilt": label}
        scores.extend({**base, **r} for r in score_tables(data, p, N, in_p, continuity))
        rare.extend({**base, **r} for r in rare_groups(data, p, N, in_p, published, puma))
        misfits.append({**base, **misfit(X, y, se, p, N), **info})

    scales = {h: (support_scale(m.F, m.binary, q_support) if standardise
                  else np.ones(m.F.shape[1])) for h, m in maps.items()}
    oracles = {}
    for label, h, tilt, _ in work:
        if tilt != "oracle":
            continue
        m = maps[h]
        mu = m.F[in_p].T @ data.weight[in_p] / N
        start = time.perf_counter()
        fit, error = fit_oracle(log_q, m.F, mu, binary=m.binary)
        scaled = np.abs(error) / scales[h] if standardise else np.abs(error)
        oracles[h] = (fit, {"iterations": fit.iterations, "converged": fit.converged,
                            "seconds": time.perf_counter() - start,
                            "oracle_max_moment_error": float(np.max(np.abs(error))),
                            "oracle_max_moment_error_sd": float(np.max(scaled)),
                            "message": fit.message})
        logger.info("PUMA %s oracle h=%s: %d iterations, max moment error %.2e sd (%s)",
                    puma, h, fit.iterations, oracles[h][1]["oracle_max_moment_error_sd"],
                    fit.message)

    base_p = q_support / q_support.sum()
    for set_name in sets:
        tables = data.sets.constraint[set_name]
        X, y, se = targets(data, tables, in_p, floor)
        for label, h, tilt, todo in work:
            if set_name not in todo:
                continue
            if tilt == "none":
                record(set_name, label, base_p, X, y, se,
                       {"iterations": 0, "converged": True, "seconds": 0.0})
                continue
            if tilt == "oracle":
                fit, info = oracles[h]
                record(set_name, label, fit.p, X, y, se, info)
                continue
            if tilt == "x":
                F, scale = Features([X]), None
            elif tilt == "phi":
                F, scale = Features([maps[h].F]), scales[h]
            else:
                F = Features([X, maps[h].F])
                scale = np.concatenate([np.ones(X.shape[1]), scales[h]])
            method = solver if solver != "auto" else (
                "lm" if h in ("quadratic", "decoder") else "lbfgs")
            start = time.perf_counter()
            fit = fit_tilt(log_q, F, X, y, se, N, tau=tau, scale=scale, method=method)
            record(set_name, label, fit.p, X, y, se,
                   {"iterations": fit.iterations, "converged": fit.converged,
                    "seconds": time.perf_counter() - start, "solver": method})
            logger.info("PUMA %s %s %s: %d iterations, %.1fs, rms z %.2f", puma, set_name,
                        label, fit.iterations, misfits[-1]["seconds"], misfits[-1]["rms_z"])
    return {"scores": pd.DataFrame(scores), "misfit": pd.DataFrame(misfits),
            "rare": pd.DataFrame(rare)}


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------


def summarise(scores: pd.DataFrame, misfit_frame: pd.DataFrame, rare: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pool per-PUMA results: means over PUMAs by (set, tilt[, table]); rare
    groups pooled as sums, and against published tables as the z of the sums
    and the RMS of per-PUMA z."""
    keys = ["source", "set", "tilt"] if "source" in scores else ["set", "tilt"]
    metrics = [c for c in ("tvd", "kl", "lor_mae", "lor_wrms") if c in scores]
    by_table = scores.groupby([*keys, "table"])[metrics].mean().reset_index()
    overall = scores.groupby(keys)[metrics].mean().reset_index()
    fit = misfit_frame.groupby(keys)[["rms_z", "share_within_1se", "seconds"]].mean().reset_index()
    overall = overall.merge(fit, on=keys, how="left")

    sums = rare.groupby([*keys, "table", "category"])[["truth", "pred"]].sum()
    if "published" in rare:
        has = rare.dropna(subset=["published", "published_se"])
        pub = has.assign(var=has["published_se"] ** 2).groupby(
            [*keys, "table", "category"]).agg(published=("published", "sum"), var=("var", "sum"),
                                              pred_pub=("pred", "sum"),
                                              rms_z_pred=("z_pred", lambda z: float(np.sqrt(np.mean(z**2)))),
                                              n_pumas=("puma", "nunique"))
        pub["z_pooled"] = (pub["pred_pub"] - pub["published"]) / np.sqrt(pub["var"])
        sums = sums.join(pub[["published", "z_pooled", "rms_z_pred", "n_pumas"]], how="left")
    sums["pred_over_truth"] = sums["pred"] / sums["truth"]
    return {"overall": overall, "tables": by_table, "rare": sums.reset_index()}
