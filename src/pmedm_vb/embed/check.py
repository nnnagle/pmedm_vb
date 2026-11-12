"""Check 1: how well the positions reconstruct held-out target entries.

Scored on the **held-out** (row, target) entries only -- held out by entry in
configuration v1, by attribute with its dependents in v2
(:mod:`pmedm_vb.embed.fit`) -- with the calibrated
decoder (``c_k = 1``; :mod:`pmedm_vb.embed.fit`) and design weights, against
the intercept-only null fitted on the training entries (household intercepts
plus group-quarters offsets, so the null already knows each unit's type and
the score measures what the learned coordinates add).

* **Per target:** multinomial deviance explained, ``1 − D_model / D_null``,
  where ``D = 2 Σ_h w_h Σ_k n_hk log(n_hk / (m_h p_hk))``. The same on the
  training entries is reported alongside, as an overfitting signal.
* **Per category:** binomial deviance explained (category against the rest of
  its target), and design-weighted predicted against observed totals on the
  held-out entries.
* **Rare categories** (:data:`RARE_CATEGORIES`) listed separately, and each
  target's held-out deviance explained among **group-quarters units** alone.

A target with a single category (``B26001``, the group-quarters count), or
any target or category whose null deviance is below :data:`NULL_FLOOR` per
weighted trial, has nothing to explain and is reported with ``NaN``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy

from pmedm_vb.embed.fit import column_group, universe_counts
from pmedm_vb.embed.model import EMPTY_LOGIT
from pmedm_vb.embed.units import TargetArrays

#: Categories reported separately, by ``"{target}.{category}"``: the rare race
#: and ethnicity groups and the oldest age bands -- of ``B03002`` and
#: ``B01001`` in configuration v1, of the ``race_ethnicity`` and ``age_sex``
#: attributes in v2. Those absent from a configuration are skipped.
#:
#: **Group quarters** are not categories here: whether a unit is group quarters
#: (``B26001``, ``B09019.group_quarters``) is fixed by its type indicators, so
#: there is nothing for the coordinates to predict and a deviance ratio there
#: is noise over a near-zero null. They are covered instead by
#: ``check1_gq_targets.csv``: each target's held-out deviance explained among
#: group-quarters units alone, i.e. how well the coordinates reconstruct the
#: attributes of the people living in them.
RARE_CATEGORIES = (
    "B03002.nh_aian", "B03002.nh_nhpi", "B03002.nh_other_race", "B03002.nh_asian",
    "B01001.male_80_to_85", "B01001.female_80_to_85",
    "B01001.male_ge_85", "B01001.female_ge_85",
    "race_ethnicity.nh_aian", "race_ethnicity.nh_nhpi", "race_ethnicity.nh_other_race",
    "race_ethnicity.nh_asian",
    "age_sex.male_80_to_85", "age_sex.female_80_to_85",
    "age_sex.male_ge_85", "age_sex.female_ge_85",
)


#: A null deviance below this, per weighted trial, leaves nothing to explain
#: (a category fixed by unit type, say: every group-quarters resident is in
#: ``B09019``'s group-quarters line); deviance explained is then ``NaN`` rather
#: than a ratio of rounding errors.
NULL_FLOOR = 1e-4


def _probs(z, W, b, O, gq, empty, start, end):
    logits = z @ W[start:end].T + b[start:end] + gq @ O[start:end].T + EMPTY_LOGIT * empty[start:end]
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def _multinomial_deviance(n, m, p, w):
    return 2.0 * (w[:, None] * (xlogy(n, n) - xlogy(n, m[:, None] * p))).sum()


def _binomial_deviance(n, m, p, w):
    p = np.clip(p, 1e-300, 1.0 - 1e-15)
    rest = m[:, None] - n
    return 2.0 * (w[:, None] * (xlogy(n, n) - xlogy(n, m[:, None] * p)
                                + xlogy(rest, rest) - xlogy(rest, m[:, None] * (1.0 - p)))).sum(0)


def reconstruction(arrays: TargetArrays, fit: dict) -> dict[str, pd.DataFrame]:
    """Check-1 tables for one fit: ``targets``, ``categories``, ``rare`` and
    ``gq_targets``."""
    tables = _reconstruction(arrays, fit, rows=None)
    gq_rows = arrays.gq.sum(1) > 0
    tables["gq_targets"] = _reconstruction(arrays, fit, rows=gq_rows)["targets"]
    return tables


def _reconstruction(arrays: TargetArrays, fit: dict, rows) -> dict[str, pd.DataFrame]:
    universe = universe_counts(arrays)
    held = fit["heldout"]
    train = (universe > 0) & ~held
    if rows is not None:
        held = held & rows[:, None]
        train = train & rows[:, None]
    w = arrays.weight
    z, gq, empty = fit["z"], arrays.gq.astype(np.float64), fit["empty"].astype(np.float64)
    names = arrays.categories
    target_rows, category_rows = [], []
    for g, meta in enumerate(arrays.meta["targets"]):
        start, end = arrays.groups[g]
        n_all = arrays.counts[:, start:end].astype(np.float64)
        row = {"target": meta["id"], "kind": meta["kind"], "categories": end - start}
        for split, sel in (("heldout", held[:, g]), ("train", train[:, g])):
            n, m, ww, zz, gg = n_all[sel], universe[sel, g].astype(np.float64), w[sel], z[sel], gq[sel]
            p_model = _probs(zz, fit["W_cal"], fit["b_cal"], fit["O_cal"], gg, empty, start, end)
            p_null = _probs(np.zeros((len(zz), 0)), np.zeros((len(names), 0)), fit["b_null"],
                            fit["O_null"], gg, empty, start, end)
            d_model = _multinomial_deviance(n, m, p_model, ww)
            d_null = _multinomial_deviance(n, m, p_null, ww)
            row[f"{split}_entries"] = int(sel.sum())
            floor = NULL_FLOOR * float(ww @ m)
            row[f"{split}_deviance_explained"] = (1.0 - d_model / d_null) if d_null > floor else np.nan
            if split == "heldout":
                row["heldout_weighted_universe"] = float(ww @ m)
                bd_model = _binomial_deviance(n, m, p_model, ww)
                bd_null = _binomial_deviance(n, m, p_null, ww)
                observed = ww @ n
                predicted = (ww * m) @ p_model
                for k in range(end - start):
                    category_rows.append({
                        "category": names[start + k], "target": meta["id"],
                        "heldout_deviance_explained":
                            (1.0 - bd_model[k] / bd_null[k]) if bd_null[k] > floor else np.nan,
                        "observed_total": observed[k], "predicted_total": predicted[k],
                        "share_observed": observed[k] / observed.sum() if observed.sum() > 0 else np.nan,
                    })
        target_rows.append(row)
    targets = pd.DataFrame(target_rows)
    categories = pd.DataFrame(category_rows)
    rare = categories[categories["category"].isin(RARE_CATEGORIES)].reset_index(drop=True)
    return {"targets": targets, "categories": categories, "rare": rare}


def overall(targets: pd.DataFrame) -> float:
    """Unweighted mean held-out deviance explained over targets that have any
    deviance (single-category targets excluded)."""
    return float(targets["heldout_deviance_explained"].dropna().mean())
