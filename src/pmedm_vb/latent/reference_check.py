"""The cross-fitted reproduction check of the PUMA reference (step 1, piece D1
of ``PLAN_latent.md``; ``PLAN_latent_step1.md`` §3, §4.6 and §6).

**Folds** (:func:`assign_folds`). Each PUMA's records are split into ``k``
folds at random by unit, stratified by household vs group quarters (within a
stratum the records are shuffled and dealt to the folds in turn, from a random
first fold). ``θ_P`` is fitted on the other folds and ``q^P`` scored against the
held-out fold. The held-out fold's atoms are off the support of that fit for
every ``γ`` (so at ``γ = 0`` the support is the training folds' own records,
the noise-floor baseline).

**Scores.** All quantities are shares within the PUMA: predicted ``C'q^P``,
observed the held-out fold's weighted share.

* **Cells** (:func:`heldout_cells`, :func:`score_cells`): the check's S1 set,
  the transfer set B (:func:`~pmedm_vb.latent.regions.transfer_sets`) and the
  published cells through ``T`` (one column per distinct cell, cells with no
  statewide contribution left out). ``z = (predicted - observed) / SE``. The
  SE is the replicate SE of the held-out share, each replicate's share being
  its own weighted total over its own weighted unit total
  (``SE² = 4/80 Σ_r (s_r - s)²``), floored at one sampled case, ``1 / n_f``
  (the check's floor, the fold's mean unit weight, as a share of the fold's
  total). Per fit, per group: the number of cells, ``Σ z²``, the number with
  ``|z| ≤ 1`` and the largest ``|z|``, so RMS z and the share within 1 SE pool
  exactly over fits.
* **Two-way tables:** TVD, KL and log-odds-ratio error of the check's scoring
  tables (:func:`~pmedm_vb.embed.puma_check.score_tables`, truth = the
  held-out fold, continuity half the fold's mean unit weight).
* **Rare groups** (:func:`~pmedm_vb.embed.puma_check.rare_groups`, truth =
  the held-out fold), and the Kish effective sample size of ``q^P`` within each
  group (weights ``q^P_i t_i``). The step 1 groups (:data:`RARE_STEP1`) drive
  the decisions; the check's other rare groups are reported too. Pooled over
  PUMAs: the ratio of predicted to held-out, and the RMS over PUMAs of the log
  ratio, each PUMA's prediction and truth summed over its folds (the §6
  criterion). Summed over folds, the own-records baseline reproduces the
  PUMA's own total almost exactly by construction (each record is in four
  training sets), so the RMS over (PUMA, fold) pairs of the per-fold log ratio
  is reported beside it, which keeps the held-out noise.

**Pooling and decisions** (:func:`pool_settings`, :func:`decide`): see
``PLAN_latent_step1.md`` §6 and the docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.data.variance import SDR_FACTOR
from pmedm_vb.embed.puma_check import RARE, CheckData, rare_groups, score_tables
from pmedm_vb.latent.atoms import Atoms, Contributions
from pmedm_vb.latent.regions import transfer_sets

#: Rare groups of ``PLAN_latent_step1.md`` §4.6: B03002 AIAN, NH/PI and other
#: race; B17010C and E; ages 80 and over; group-quarters type.
RARE_STEP1 = (
    ("B03002", "nh_aian"), ("B03002", "nh_nhpi"), ("B03002", "nh_other_race"),
    ("B17010C", "below_poverty"), ("B17010C", "at_or_above_poverty"),
    ("B17010E", "below_poverty"), ("B17010E", "at_or_above_poverty"),
    ("B01001", "male_80_to_85"), ("B01001", "male_ge_85"),
    ("B01001", "female_80_to_85"), ("B01001", "female_ge_85"),
    ("unit_type", "institutional"), ("unit_type", "noninstitutional"),
)
assert set(RARE_STEP1) <= set(RARE)

#: Groups whose pooled ratio the stop rule bounds, and the bounds.
STOP_GROUPS = (("B03002", "nh_aian"), ("B03002", "nh_nhpi"))
STOP_RATIO = (0.8, 1.25)

#: Decision thresholds (§6): within 10% of the best; the noise floor plus 50%.
WITHIN = 0.10
FLOOR_MARGIN = 0.50

CELL_GROUPS = ("S1", "B", "T")


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------


def assign_folds(atoms: Atoms, puma: str, k: int = 5, seed: int = 1) -> np.ndarray:
    """Fold of every atom of ``puma`` (``-1`` elsewhere); see the module notes."""
    rng = np.random.default_rng([int(seed), int(puma)])
    fold = np.full(atoms.M, -1, dtype=np.int64)
    own = atoms.puma == str(puma)
    for stratum in (atoms.unit_type == "household", atoms.unit_type != "household"):
        idx = np.flatnonzero(own & stratum)
        idx = idx[rng.permutation(idx.size)]
        fold[idx] = (np.arange(idx.size) + rng.integers(k)) % k
    return fold


# --------------------------------------------------------------------------
# Cells
# --------------------------------------------------------------------------


@dataclass
class CellGroup:
    name: str
    C: sp.csc_matrix
    cells: list[str]


def cell_groups(data: CheckData, contributions: Contributions) -> list[CellGroup]:
    """S1, B and T (see the module notes), columns over the atoms."""
    sets = transfer_sets(data)
    out = []
    for name, tables in (("S1", data.sets.constraint["S1"]), ("B", sets["B"])):
        C = sp.hstack([data.counts[t] for t in tables]).tocsc()
        cells = [f"{t}:{c.name}" for t in tables for c in data.tables[t].categories]
        out.append(CellGroup(name, C, cells))
    unique = contributions.unique_cells()
    T = contributions.T.tocsc()[:, unique]
    nonempty = np.flatnonzero(np.diff(T.indptr) > 0)
    out.append(CellGroup("T", T[:, nonempty].tocsc(),
                         [contributions.cell[j] for j in unique[nonempty]]))
    return out


@dataclass
class HeldOutCells:
    """Observed shares and SEs of one held-out fold, per cell group."""

    share: dict[str, np.ndarray]
    se: dict[str, np.ndarray]
    n: int
    total: float


def heldout_cells(atoms: Atoms, groups: list[CellGroup], held: np.ndarray) -> HeldOutCells:
    rows = np.flatnonzero(held)
    w = atoms.weight[rows]
    R = atoms.replicates[rows]
    total, rep_total = float(w.sum()), R.sum(0)
    floor = 1.0 / rows.size
    share, se = {}, {}
    for g in groups:
        Cf = g.C[rows]
        s = np.asarray(Cf.T @ w).ravel() / total
        s_r = np.asarray(Cf.T @ R) / rep_total[None, :]
        share[g.name] = s
        se[g.name] = np.maximum(np.sqrt(SDR_FACTOR * np.square(s_r - s[:, None]).sum(1)), floor)
    return HeldOutCells(share=share, se=se, n=int(rows.size), total=total)


def score_cells(p: np.ndarray, groups: list[CellGroup], obs: HeldOutCells) -> list[dict]:
    rows = []
    for g in groups:
        pred = np.asarray(g.C.T @ p).ravel()
        z = (pred - obs.share[g.name]) / obs.se[g.name]
        rows.append({"group": g.name, "n_cells": int(z.size), "sum_z2": float(np.square(z).sum()),
                     "n_within_1": int((np.abs(z) <= 1).sum()),
                     "max_abs_z": float(np.abs(z).max())})
    return rows


def rare_columns(data: CheckData) -> dict[tuple[str, str], np.ndarray]:
    """Each rare group's per-atom contributions (dense), for the within-group ESS."""
    out = {}
    for table, category in RARE:
        names = [c.name for c in data.tables[table].categories]
        col = data.counts[table][:, names.index(category)]
        out[(table, category)] = np.asarray(col.toarray(), dtype=float).ravel()
    return out


def score_fit(data: CheckData, p: np.ndarray, groups: list[CellGroup], obs: HeldOutCells,
              held: np.ndarray, puma: str, rare_cols: dict) -> dict[str, list[dict]]:
    """Every §4.6 score of one ``q^P`` against one held-out fold."""
    N = obs.total
    continuity = 0.5 * N / obs.n
    tables = score_tables(data, p, N, held, continuity)
    rare = rare_groups(data, p, N, held, None, puma)
    for row in rare:
        v = rare_cols[(row["table"], row["category"])] * p
        total = v.sum()
        row["ess"] = float(total ** 2 / np.square(v).sum()) if total > 0 else 0.0
    return {"cells": score_cells(p, groups, obs), "tables": tables, "rare": rare}


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------

SETTING = ["basis", "gamma", "sigma"]

#: Stands in for the baselines' σ (NaN) while pooling: pandas does not align
#: index keys holding NaN reliably.
_NO_SIGMA = -1.0


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(sigma=frame["sigma"].fillna(_NO_SIGMA))


def _unkeyed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(sigma=frame["sigma"].replace(_NO_SIGMA, np.nan))


def pool_cells(cells: pd.DataFrame) -> pd.DataFrame:
    """RMS z and share within 1 SE per setting, by group and over all groups."""
    by_group = cells.groupby([*SETTING, "group"], dropna=False)[
        ["n_cells", "sum_z2", "n_within_1"]].sum()
    out = pd.DataFrame({
        "rms_z": np.sqrt(by_group["sum_z2"] / by_group["n_cells"]),
        "within_1": by_group["n_within_1"] / by_group["n_cells"]}).unstack("group")
    out.columns = [f"{m}_{g}" for m, g in out.columns]
    total = cells.groupby(SETTING, dropna=False)[["n_cells", "sum_z2", "n_within_1"]].sum()
    out["rms_z_all"] = np.sqrt(total["sum_z2"] / total["n_cells"])
    out["within_1_all"] = total["n_within_1"] / total["n_cells"]
    return out


def _rms(x: pd.Series) -> float:
    return float(np.sqrt(np.nanmean(np.square(x)))) if x.notna().any() else np.nan


def _log_ratio(frame: pd.DataFrame) -> np.ndarray:
    ok = (frame["truth"] > 0) & (frame["pred"] > 0)
    return np.where(ok, np.log(np.where(ok, frame["pred"], 1.0)
                               / np.where(ok, frame["truth"], 1.0)), np.nan)


def pool_rare(rare: pd.DataFrame, groups=RARE_STEP1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per setting and rare group: the pooled ratio of predicted to held-out
    (sums over PUMAs and folds); the RMS over PUMAs of the log ratio, each
    PUMA's ratio pooled over its folds (``rms_log_ratio``; PUMAs where either
    is zero are left out and counted); and, reported beside it, the RMS over
    (PUMA, fold) pairs of the per-fold log ratio (``rms_log_ratio_fold``,
    pairs where either is zero left out and counted). Second frame, per
    setting over ``groups``: both RMS log ratios over every (group, PUMA) or
    (group, PUMA, fold) pair, the pooled AIAN and NH/PI ratios, and the mean
    within-group Kish ESS."""
    rare = _keyed(rare)
    keys = [*SETTING, "table", "category"]
    puma = rare.groupby([*keys, "puma"], dropna=False)[["truth", "pred"]].sum().reset_index()
    puma["log_ratio"] = _log_ratio(puma)
    fold = rare.copy()
    fold["log_ratio"] = _log_ratio(fold)
    per = puma.groupby(keys, dropna=False).agg(
        truth=("truth", "sum"), pred=("pred", "sum"), rms_log_ratio=("log_ratio", _rms),
        n_pumas=("log_ratio", "size"), n_skipped=("log_ratio", lambda x: int(x.isna().sum())))
    per = per.join(fold.groupby(keys, dropna=False).agg(
        rms_log_ratio_fold=("log_ratio", _rms),
        n_folds_skipped=("log_ratio", lambda x: int(x.isna().sum()))))
    per["pooled_ratio"] = per["pred"] / per["truth"]
    if "ess" in rare:
        per = per.join(rare.groupby(keys, dropna=False)["ess"].mean().rename("mean_ess"))

    def chosen(frame):
        return frame[frame.set_index(["table", "category"]).index.isin(list(groups))]

    overall = chosen(puma).groupby(SETTING, dropna=False).agg(
        rms_log_ratio_rare=("log_ratio", _rms))
    overall = overall.join(chosen(fold).groupby(SETTING, dropna=False).agg(
        rms_log_ratio_fold_rare=("log_ratio", _rms)))
    flat = per.reset_index()
    for table, category in STOP_GROUPS:
        sel = flat[(flat["table"] == table) & (flat["category"] == category)]
        overall = overall.join(sel.set_index(SETTING)["pooled_ratio"].rename(f"pooled_{category}"))
    if "ess" in rare:
        overall = overall.join(chosen(rare).groupby(SETTING, dropna=False)["ess"].mean()
                               .rename("rare_ess"))
    return _unkeyed(flat), overall


def pool_settings(fits: pd.DataFrame, cells: pd.DataFrame, tables: pd.DataFrame,
                  rare: pd.DataFrame, K: dict[str, int]) -> pd.DataFrame:
    """One row per setting (basis, γ, σ; baselines have basis ``baseline_*``
    and σ NaN): the pooled §4.6 metrics and the fits' diagnostics."""
    fits, cells, tables, rare = (_keyed(f) for f in (fits, cells, tables, rare))
    out = pool_cells(cells)
    metrics = [c for c in ("tvd", "lor_mae") if c in tables]
    out = out.join(tables.groupby(SETTING, dropna=False)[metrics].mean())
    out = out.join(pool_rare(rare)[1])
    diag = fits.groupby(SETTING, dropna=False).agg(
        n_fits=("ess", "size"), ess=("ess", "mean"), share_outside=("share_outside", "mean"),
        iterations=("iterations", "mean"), max_iterations=("iterations", "max"),
        converged=("converged", "mean"), grad_max=("grad_max", "max"),
        seconds=("seconds", "mean"))
    out = _unkeyed(out.join(diag).reset_index())
    out.insert(1, "K", out["basis"].map(K))
    return out


# --------------------------------------------------------------------------
# Decisions (§6)
# --------------------------------------------------------------------------


def decide(settings: pd.DataFrame) -> tuple[dict, list[str]]:
    """The §6 decision rules on the pooled settings (:func:`pool_settings`).

    * **h:** each basis's best (γ, σ) is the one with the lowest held-out RMS
      z over all scored cells; at it, RMS z and the rare groups' RMS log ratio
      are compared across bases, and the basis with the smallest K among those
      within 10% of the best on both is chosen. If none is within 10% on
      both, the basis with the smallest larger relative excess is taken and
      flagged.
    * **γ:** for the chosen h, the best σ per γ; the largest γ whose RMS z is
      within 10% of the best γ's. **σ_θ** is the best σ at that γ; flagged if
      at the edge of the grid.
    * **Variety:** for the chosen h and each γ at the chosen σ (γ = 0 has
      ``θ̂ = 0``, the own-records support), ESS and rare-group accuracy
      against γ = 0.
    * **Stop rule:** a basis passes when its RMS z at its best setting is
      within 50% of the noise floor's (the own-records baseline, γ = 0,
      θ = 0) and the pooled AIAN and NH/PI ratios there are in [0.8, 1.25].
      Step 1 fails if no basis passes.
    """
    lines = []
    grid = settings[~settings["basis"].str.startswith("baseline")].copy()
    base = settings[settings["basis"].str.startswith("baseline")]
    floor_row = base[base["gamma"] == 0]
    floor = float(floor_row["rms_z_all"].iloc[0]) if len(floor_row) else np.nan
    best = grid.loc[grid.groupby("basis")["rms_z_all"].idxmin()].set_index("basis")
    z0, r0 = best["rms_z_all"].min(), best["rms_log_ratio_rare"].min()
    best["rel_z"] = best["rms_z_all"] / z0 - 1
    best["rel_rare"] = best["rms_log_ratio_rare"] / r0 - 1
    eligible = best[(best["rel_z"] <= WITHIN) & (best["rel_rare"] <= WITHIN)]
    flags = []
    if len(eligible):
        h = str(eligible.sort_values(["K", "rms_z_all"]).index[0])
    else:
        worst = best[["rel_z", "rel_rare"]].max(axis=1)
        h = str(worst.idxmin())
        flags.append(f"no basis is within 10% of the best on both criteria; took {h}, "
                     "the smallest worse excess (review)")
    lines.append("Choosing h: each basis at its best (gamma, sigma) by RMS z over all cells")
    show = best[["K", "gamma", "sigma", "rms_z_all", "rms_log_ratio_rare", "rel_z", "rel_rare"]]
    lines.append(show.round(4).to_string())
    lines.append(f"  within 10% on both: {sorted(eligible.index.tolist()) or 'none'}; "
                 f"chosen h = {h} (smallest K)")
    if (best["gamma"] == 0).any():
        at0 = sorted(best.index[best["gamma"] == 0].tolist())
        flags.append(f"best setting at gamma = 0 for {at0}: theta = 0 there, so h has no "
                     "effect on those scores")

    sub = grid[grid["basis"] == h]
    per_gamma = sub.loc[sub.groupby("gamma")["rms_z_all"].idxmin()].set_index("gamma")
    zbest = per_gamma["rms_z_all"].min()
    ok = per_gamma[per_gamma["rms_z_all"] <= (1 + WITHIN) * zbest]
    gamma = float(ok.index.max())
    sigma = float(per_gamma.loc[gamma, "sigma"])
    sigmas = sorted(sub["sigma"].unique())
    lines += ["", f"Choosing gamma and sigma for h = {h}: best sigma per gamma",
              per_gamma[["sigma", "rms_z_all", "rms_log_ratio_rare", "ess",
                         "share_outside"]].round(4).to_string(),
              f"  best gamma {per_gamma['rms_z_all'].idxmin():g} (RMS z {zbest:.4f}); largest "
              f"within 10%: gamma = {gamma:g}, sigma = {sigma:g}"]
    if gamma == 0:
        flags.append("chosen gamma = 0: theta = 0 for every sigma (the own-records support), "
                     "so sigma is not identified by the grid")
    elif sigma in (sigmas[0], sigmas[-1]) and len(sigmas) > 1:
        flags.append(f"chosen sigma = {sigma:g} is at the edge of the grid "
                     f"{[float(x) for x in sigmas]}: extend the grid (section 6)")

    variety = sub[sub["sigma"] == sigma].set_index("gamma").sort_index()
    ref = variety.loc[0] if 0 in variety.index else None
    cols = ["rms_z_all", "rms_log_ratio_rare", "rms_log_ratio_fold_rare", "pooled_nh_aian",
            "pooled_nh_nhpi", "ess",
            "rare_ess", "share_outside"]
    cols = [c for c in cols if c in variety]
    lines += ["", f"Variety (h = {h}, sigma = {sigma:g}): against gamma = 0",
              variety[cols].round(4).to_string()]
    if ref is not None:
        rel = pd.DataFrame({
            "ess_ratio": variety["ess"] / ref["ess"],
            "rare_ess_ratio": (variety["rare_ess"] / ref["rare_ess"]) if "rare_ess" in variety
            else np.nan,
            "rms_log_ratio_rare_change": variety["rms_log_ratio_rare"] - ref["rms_log_ratio_rare"],
            "rms_log_ratio_fold_rare_change": variety["rms_log_ratio_fold_rare"]
            - ref["rms_log_ratio_fold_rare"],
            "rms_z_change": variety["rms_z_all"] - ref["rms_z_all"]})
        lines.append(rel.round(4).to_string())

    lo, hi = STOP_RATIO
    best["passes_floor"] = best["rms_z_all"] <= (1 + FLOOR_MARGIN) * floor
    best["passes_rare"] = best["pooled_nh_aian"].between(lo, hi) & \
        best["pooled_nh_nhpi"].between(lo, hi)
    best["passes"] = best["passes_floor"] & best["passes_rare"]
    failed = not best["passes"].any()
    lines += ["", f"Stop rule: noise floor RMS z = {floor:.4f} (own records, gamma = 0); "
              f"limit {(1 + FLOOR_MARGIN) * floor:.4f}; AIAN and NH/PI pooled ratio in "
              f"[{lo}, {hi}]",
              best[["rms_z_all", "pooled_nh_aian", "pooled_nh_nhpi", "passes_floor",
                    "passes_rare", "passes"]].round(4).to_string(),
              "  STEP 1 FAILS: revisit the stage 1 representation before step 2" if failed
              else "  step 1 passes the stop rule"]
    if flags:
        lines += ["", "Flags:", *(f"  - {f}" for f in flags)]
    decision = {"h": h, "gamma": gamma, "sigma": sigma, "noise_floor_rms_z": floor,
                "step1_fails": bool(failed), "flags": flags}
    return decision, lines
