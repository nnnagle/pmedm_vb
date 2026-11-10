"""Held-out table scores of a posterior's draws, shared by ``compare_support.py`` and ``compare_variants.py``.

The held-out tables (``run_map.py heldout``; :class:`pmedm_vb.assemble.heldout.HeldOut`)
are published estimates no fit sees. A posterior draw ``W`` (persons per block
group and record) predicts each held-out cell: ``X_b' W`` per block group, and
per tract the block groups' ``X_t' W`` summed by ``A_T``
(:class:`HeldoutPredictor`). Over the draws, each cell is scored against the
published estimate ``Y`` and its sampling variance ``v``
(:func:`score_cells`): the error of the posterior mean, the predictive
``z = (mean - Y) / sqrt(v + sd^2)`` (the published value's sampling error plus
the posterior's own spread), whether ``|z| <= 1.645`` (90% predictive
coverage) and the Gaussian log score ``log N(Y; mean, v + sd^2)``. A cell with
no spread at all (zero variance and sd) has no score (NaN).

**Count score** (``count_score``, :func:`count_log_score`): the log of the
posterior predictive probability of the published value under a scaled-Poisson
survey model, ``log mean_draws L(T)`` with ``T`` the drawn count. A published
``Y > 0`` with variance ``v`` is read as ``k = Y^2 / v`` sampled cases of size
``w = v / Y`` (so ``Y = k w`` and ``v = k w^2``), and
``L(T) = (T/w)^k exp(-T/w) / Gamma(k + 1)``: the probability of ``k`` cases
when ``T/w`` are expected. A published zero is no cases:
``L(T) = exp(-T/w)``. The Gaussian log score treats a published zero as a
measurement with the zero-cell policy's SE (about 8.5 persons) around 0, so a
posterior that spreads a rare group over the block groups where it may live
loses on every zero it touches; the count score asks instead how probable it is
that the survey sampled none of the ``T`` predicted, which a small ``T``
leaves likely. A zero's case size is not published, so ``w`` there is an
**assumption** (:func:`zero_case_size`): the PUMA's average weight ``N / n``
for a household-universe table, and for a person-universe one ``N / n`` times
the persons in the category per unit carrying it (the design-weighted mean of
the units' counts over the units with a positive count, the PUMA's own records
where its PUMA-only inputs are there, else the support's rows). Cells with
``Y > 0`` and no published variance have no count score (NaN); a draw's ``T``
is floored at ``1e-12 w`` so that predicting none of a published positive
scores very low rather than ``-inf``.

Cells are grouped by level and by their relation to the fitted tables
(``HeldOut.relation``) and, as their own group, the race x poverty cells
(``B17010A-I``, :data:`RACE_POVERTY`): the joint structure the prior, not the
tables, decides; and by the cell's own evidence (:data:`EVIDENCE`): published
zero, ``0 < k_hat < 2`` (about one sampled case) and ``k_hat >= 2``, with
``k_hat = (Y / SE)^2`` from the published variance (:func:`score_groups`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import gammaln, logsumexp

#: The extra group of held-out cells reported on their own: B17010A-I, families
#: by poverty status by race of householder.
RACE_POVERTY = ("B17010", "B17010 race x poverty")
#: The per-cell measures of :func:`score_cells` (``count_score`` only when
#: zeros' case sizes are given).
MEASURES = ("mean", "sd", "error", "pred_z", "covers", "log_score", "count_score")
LEVELS = ("block group", "tract")
#: The groups by the held-out cell's own evidence, ``k_hat = (Y / SE)^2`` from
#: the published variance (:func:`evidence_group`).
EVIDENCE = ("published zero", "0 < k_hat < 2", "k_hat >= 2")
#: The universe of each held-out table: what one of its cases counts. Read off
#: the specifications in ``pmedm_vb.assemble.constraints`` (``heldout_tables``,
#: ``race_poverty``) and ``pmedm_vb.assemble.allcells`` (``all_heldout_tables``),
#: which declare the same universes: tenure by vehicles (``B25044``), heating
#: fuel (``B25040``) and families by poverty by race of householder
#: (``B17010A-I``) count occupied housing units; poverty ratio (``C17002``),
#: industry by sex (``C24030``), marital status (``B12001``) and educational
#: attainment (``B15002``) count persons. Kept here rather than built from the
#: specifications, two of which need the PUMS dictionary or the replicate
#: files to construct.
HELDOUT_UNIVERSE = {"B25044": "household", "B25040": "household", "C17002": "person",
                    "C24030": "person", "B12001": "person", "B15002": "person",
                    **{f"B17010{s}": "household" for s in "ABCDEFGHI"}}
#: The floor on ``T / w`` in :func:`count_log_score`.
COUNT_FLOOR = 1e-12


class HeldoutPredictor:
    """A population's held-out cells: ``(block group cells (zones, k_b), tract
    cells (tracts, k_t))`` for ``W`` (zones, rows)."""

    def __init__(self, inputs, heldout):
        self.h_b, self.h_t = (sp.csr_matrix(heldout.X[level]) for level in LEVELS)
        self.A_T = sp.csr_matrix(inputs.A_T)

    def __call__(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (self.h_b.T @ W.T).T, self.A_T @ (self.h_t.T @ W.T).T


def predict(inputs, heldout, lam: np.ndarray) -> dict[str, np.ndarray]:
    """Each column of ``lam`` (``(m, draws)``, the multipliers the data see) made
    a population ``W = N p(lambda)``; its held-out cells per level, ``(draws,
    areas, k)`` float32."""
    from pmedm_vb.solvers.base import ConstraintOperator

    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    held = HeldoutPredictor(inputs, heldout)
    d = lam.shape[1]
    out = {"block group": np.empty((d, inputs.n_zones, held.h_b.shape[1]), dtype=np.float32),
           "tract": np.empty((d, held.A_T.shape[0], held.h_t.shape[1]), dtype=np.float32)}
    for k in range(d):
        logits = log_q - op.adjoint(lam[:, k])
        W = inputs.N * np.exp(logits - logsumexp(logits))
        out["block group"][k], out["tract"][k] = held(W)
    return out


def score_cells(draws: np.ndarray, Y: np.ndarray, v: np.ndarray,
                zero_w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per cell of ``(draws, areas, k)`` predictions against published ``Y`` and
    variance ``v``: mean, sd, error, predictive z, 90% predictive coverage and a
    Gaussian log score. Cells with no spread at all (zero variance and sd) are NaN.
    With ``zero_w`` (the case size of each column's published zeros,
    :func:`zero_case_size`), also the count score (:func:`count_log_score`)."""
    mean, sd = draws.mean(0).astype(float), draws.std(0, ddof=1).astype(float)
    spread = np.sqrt(np.asarray(v, float) + sd**2)
    ok = spread > 0
    safe = np.where(ok, spread, 1.0)
    pred_z = np.where(ok, (mean - Y) / safe, np.nan)
    out = {"mean": mean, "sd": sd, "error": mean - Y, "pred_z": pred_z,
           "covers": np.where(ok, np.abs(pred_z) <= 1.645, np.nan),
           "log_score": np.where(ok, -0.5 * np.log(2 * np.pi * safe**2) - 0.5 * pred_z**2, np.nan)}
    if zero_w is not None:
        out["count_score"] = count_log_score(draws, Y, v, zero_w)
    return out


def count_log_score(draws: np.ndarray, Y: np.ndarray, v: np.ndarray, zero_w: np.ndarray,
                    chunk: int = 64) -> np.ndarray:
    """Per cell ``(areas, k)``: the log posterior predictive probability of the
    published value under the scaled-Poisson survey model (module docstring),
    ``log mean_draws L(T)`` over the ``(draws, areas, k)`` predictions ``T``.

    A published ``Y > 0`` with ``v > 0`` is ``k = Y^2 / v`` cases of size
    ``w = v / Y``: ``log L(T) = k log(T/w) - T/w - log Gamma(k + 1)``. A
    published zero is no cases of size ``zero_w`` (per column, broadcast over
    areas; an assumption, :func:`zero_case_size`): ``log L(T) = -T/w``. ``T/w``
    is floored at :data:`COUNT_FLOOR`. NaN where ``Y > 0`` has no variance.
    The draws are taken ``chunk`` areas at a time, in float64."""
    Y, v = np.asarray(Y, float), np.asarray(v, float)
    positive, zero = (Y > 0) & (v > 0), Y == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(positive, Y * Y / np.where(positive, v, 1.0), 0.0)
        w = np.where(positive, v / np.where(positive, Y, 1.0),
                     np.broadcast_to(np.asarray(zero_w, float), Y.shape))
    log_norm = np.where(positive, gammaln(k + 1), 0.0)
    out = np.full(Y.shape, np.nan)
    d = draws.shape[0]
    for start in range(0, Y.shape[0], chunk):
        part = slice(start, start + chunk)
        mu = np.maximum(np.asarray(draws[:, part], float), 0.0) / w[part]
        log_l = np.where(positive[part], k[part] * np.log(np.maximum(mu, COUNT_FLOOR))
                         - mu - log_norm[part], -mu)
        out[part] = logsumexp(log_l, axis=0) - np.log(d)
    out[~(positive | zero)] = np.nan
    return out


def table_universe(name: str) -> str:
    """The universe (``person`` or ``household``) of a held-out column
    ``<table>.<category>`` (:data:`HELDOUT_UNIVERSE`)."""
    table = name.split(".", 1)[0]
    if table not in HELDOUT_UNIVERSE:
        raise KeyError(f"held-out table {table}: no universe in HELDOUT_UNIVERSE")
    return HELDOUT_UNIVERSE[table]


def zero_case_size(inputs, heldout, level: str) -> np.ndarray:
    """Per held-out column of ``level``, the size ``w`` assumed for the case a
    published zero did not sample (:func:`count_log_score`).

    ``N / n`` is the PUMA's average weight (``n`` its own sample records, also
    under a statewide support). A household-universe table's case is one unit,
    so ``w = N / n``. A person-universe table's case is a unit carrying the
    category with all its persons in it, so ``w = N / n`` times the
    design-weighted mean of the column's count over the support's units with
    a positive count (``units.weight``; under a statewide support a merged
    row's count is its records' weighted mean, ``HeldOut``); ``N / n`` alone
    where no unit carries it, since ``T`` is then zero in every draw and ``w``
    does not matter. This is an assumption: the survey's own case sizes behind
    a zero are not published.

    Pass the PUMA's own-records inputs where there are any (callers do): ``w``
    is then a property of the PUMA's survey, the same for every run of the
    PUMA, so that runs on different supports are scored against the same
    model of the published zeros; and a statewide support's merged rows
    average their records' counts, which dilutes the persons per carrying
    unit (a merged row carries the category when any of its records does)."""
    base = float(inputs.N) / float(inputs.n)
    X = sp.csr_matrix(heldout.X[level])
    weight = inputs.units["weight"].to_numpy(dtype=float)
    carries = X.copy()
    carries.data = (carries.data > 0).astype(float)
    persons = X.T @ weight
    carriers = carries.T @ weight
    mean = np.divide(persons, carriers, out=np.ones_like(persons), where=carriers > 0)
    person = np.array([table_universe(name) == "person" for name in heldout.names[level]])
    return base * np.where(person, mean, 1.0)


def closer_share(error: pd.Series | np.ndarray, error_base: pd.Series | np.ndarray) -> float:
    """The share of cells where one posterior mean is nearer the published value
    than the base's, a tie (equal ``|error|``, as where both predict nothing)
    counting half: so two identical posteriors score 0.5, not 0 as a strict
    ``<`` would make them."""
    a, b = np.abs(np.asarray(error, float)), np.abs(np.asarray(error_base, float))
    if not a.size:
        return float("nan")
    return float(((a < b) + 0.5 * (a == b)).mean())


def area_names(inputs) -> dict[str, list[str]]:
    """The held-out cells' areas per level: block groups (zones) and tracts."""
    return {"block group": list(inputs.zones["block_group_geoid"]),
            "tract": list(inputs.tracts.iloc[:, 0].astype(str))}


def level_frame(heldout, level: str, areas: list[str],
                scores: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """One row per held-out cell of ``level`` (area-major): its area, name,
    relation, published value and SE, and ``<tag>_<measure>`` for each
    ``scores[tag]`` (:func:`score_cells`)."""
    Y, v = heldout.Y[level], heldout.v[level]
    names, relation = heldout.names[level], heldout.relation(level)
    return pd.DataFrame({
        "level": level, "area": np.repeat(areas, len(names)),
        "cell": np.tile(names, len(areas)),
        "relation": np.tile(relation, len(areas)),
        "published": Y.ravel(), "se": np.sqrt(v).ravel(),
        **{f"{tag}_{m}": vals.ravel() for tag, sc in scores.items() for m, vals in sc.items()},
    })


def evidence_group(published: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Each cell's :data:`EVIDENCE` group from its published value and SE:
    ``published zero``, ``0 < k_hat < 2`` or ``k_hat >= 2`` (a positive value
    with no published SE counts as ``k_hat >= 2``)."""
    Y, se = np.asarray(published, float), np.asarray(se, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_hat = np.where(se > 0, (Y / np.where(se > 0, se, 1.0)) ** 2, np.inf)
    return np.where(Y <= 0, EVIDENCE[0], np.where(k_hat < 2, EVIDENCE[1], EVIDENCE[2]))


def score_groups(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """One level's cells grouped by relation, then the race x poverty cells
    (:data:`RACE_POVERTY`) as a group of their own when there are any, then by
    the cell's own evidence (:data:`EVIDENCE`, from the ``published`` and ``se``
    columns), each evidence group that has cells. A cell is in one relation
    group and one evidence group."""
    groups = list(frame.groupby("relation"))
    prefix, label = RACE_POVERTY
    race_poverty = frame[frame["cell"].str.startswith(prefix)]
    if len(race_poverty):
        groups.append((label, race_poverty))
    evidence = evidence_group(frame["published"].to_numpy(), frame["se"].to_numpy())
    for name in EVIDENCE:
        part = frame[evidence == name]
        if len(part):
            groups.append((name, part))
    return groups


def group_summary(group: pd.DataFrame, tag: str) -> dict:
    """One posterior's scores (columns ``<tag>_<measure>``) over a group of cells:
    the scored cells, mean |error| and RMSE of the posterior mean, the mean and
    mean square of the predictive z, 90% predictive coverage and the mean
    Gaussian log score; with a ``<tag>_count_score`` column also the mean count
    score (:func:`count_log_score`) over the scored cells that have one."""
    ok = group.dropna(subset=[f"{tag}_pred_z"])
    error, z = ok[f"{tag}_error"], ok[f"{tag}_pred_z"]
    out = {"cells": len(ok), "mae": float(error.abs().mean()),
           "rmse": float(np.sqrt((error**2).mean())), "pred_z_mean": float(z.mean()),
           "pred_z2": float((z**2).mean()), "cover90": float(ok[f"{tag}_covers"].mean()),
           "log_score": float(ok[f"{tag}_log_score"].mean())}
    if f"{tag}_count_score" in ok:
        out["count_score"] = float(ok[f"{tag}_count_score"].mean())
    return out
