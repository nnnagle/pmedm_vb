"""Comparing two samplers' posteriors: what a simulator user would see differ.

A draw of ``lambda`` is a draw of the weights ``W = N p(lambda)``, and what a
user of the simulation reads off ``W`` are counts: how many people of some kind
live in each block group. So the comparison here is of **outcomes**, each a
per-unit loading ``o_i`` summed over the weights, ``Y_b = sum_i W_bi o_i``:

- **constrained**: every block group constraint column of ``X_B``. The fit
  targets these, so a sampler should get their centre right; the question is
  their spread.
- **cross-tabulations**: the product of two constrained columns, for example
  persons of each race and ethnicity (``B03002``) in households of each income
  band (``B19001``). No table constrains the product, so this is where the
  simulator's answer is its own -- and where a user most needs it. A group
  quarters person has no household income, so carries no ``B19001`` loading
  and drops out of these.

For each (block group, outcome) cell the posterior mean, sd and 5/50/95%
quantiles are summarised from each sampler's draws; :func:`compare_outcomes`
sets one against the other, in units of the reference's sd.

Two lower-level comparisons locate *where* an approximation goes wrong:
:func:`coordinate_comparison` in the whitened coordinates ``G'(lambda - mu)`` of
a Gaussian (standard normal if that Gaussian were exact), and
:func:`cell_log_shares` along the "walls" -- the cells an approximation piles
weight onto. :func:`psis_khat` is the Pareto-``k`` diagnostic of Yao et al.
(2018, ICML), via arviz.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.solvers.base import ConstraintOperator

#: Race and ethnicity by household income band, the default cross-tabulation.
RACE_BY_INCOME = ("B03002.", "B19001.")

#: Floor, in people, on the reference sd when standardising differences, so a
#: cell the reference pins at (nearly) zero does not turn a tiny difference into
#: an enormous one.
SD_FLOOR = 0.5


def outcome_matrix(
    inputs: PMEDMInputs, crosstabs: Sequence[tuple[str, str]] = (RACE_BY_INCOME,)
) -> tuple[sp.csc_matrix, pd.DataFrame]:
    """``(n_units, k)`` outcome loadings and a frame naming each column.

    Constrained outcomes are ``X_B``'s columns; each ``(prefix_a, prefix_b)`` in
    ``crosstabs`` adds the elementwise product of every column whose name starts
    with ``prefix_a`` with every one starting with ``prefix_b``. A pair with no
    matching columns adds nothing.
    """
    X = sp.csc_matrix(inputs.X_B)
    names = list(inputs.bg_constraints)
    columns = [X[:, k] for k in range(X.shape[1])]
    meta = [{"outcome": name, "kind": "constrained"} for name in names]
    for prefix_a, prefix_b in crosstabs:
        left = [k for k, name in enumerate(names) if name.startswith(prefix_a)]
        right = [k for k, name in enumerate(names) if name.startswith(prefix_b)]
        for a in left:
            for b in right:
                columns.append(X[:, a].multiply(X[:, b]))
                meta.append({"outcome": f"{names[a]} x {names[b]}", "kind": "crosstab"})
    return sp.csc_matrix(sp.hstack(columns)), pd.DataFrame(meta)


def outcome_draws(
    inputs: PMEDMInputs, lam: np.ndarray, outcomes: sp.spmatrix, batch: int = 64
) -> np.ndarray:
    """``(draws, n_zones, k)`` outcome counts, float32, one per column of ``lam``
    (``(m, draws)``): ``N p(lambda) @ outcomes``."""
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    outcomes = sp.csr_matrix(outcomes)
    out = np.empty((lam.shape[1], inputs.n_zones, outcomes.shape[1]), dtype=np.float32)
    for d in range(lam.shape[1]):
        logits = log_q - op.adjoint(lam[:, d])
        W = inputs.N * np.exp(logits - logsumexp(logits))
        out[d] = (outcomes.T @ W.T).T
    return out


def summarise(draws: np.ndarray, meta: pd.DataFrame, zones: Sequence) -> pd.DataFrame:
    """One row per (zone, outcome): mean, sd and 5/50/95% quantiles over draws."""
    q05, q50, q95 = np.quantile(draws, [0.05, 0.5, 0.95], axis=0)
    n_zones, k = draws.shape[1:]
    return pd.DataFrame({
        "zone": np.repeat(np.asarray(zones), k),
        "outcome": np.tile(meta["outcome"].to_numpy(), n_zones),
        "kind": np.tile(meta["kind"].to_numpy(), n_zones),
        "mean": draws.mean(0).ravel(),
        "sd": draws.std(0, ddof=1).ravel(),
        "q05": q05.ravel(),
        "q50": q50.ravel(),
        "q95": q95.ravel(),
    })


def compare_outcomes(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Set :func:`summarise` frames side by side, differences in reference sds.

    ``mean_diff`` and ``median_diff`` are ``(other - reference) / sd``, with the
    reference sd floored at :data:`SD_FLOOR`; ``sd_ratio`` and ``width_ratio``
    (of the 90% intervals) are other over reference. The median is there
    because a heavy tail in ``other`` moves its mean but not its median.
    """
    keys = ["zone", "outcome", "kind"]
    both = reference.merge(other, on=keys, suffixes=("_ref", "_alt"))
    scale = np.maximum(both["sd_ref"], SD_FLOOR)
    width_ref = np.maximum(both["q95_ref"] - both["q05_ref"], SD_FLOOR)
    return both.assign(
        mean_diff=(both["mean_alt"] - both["mean_ref"]) / scale,
        median_diff=(both["q50_alt"] - both["q50_ref"]) / scale,
        sd_ratio=both["sd_alt"] / scale,
        width_ratio=(both["q95_alt"] - both["q05_alt"]) / width_ref,
    )


def whitened(q, lam: np.ndarray) -> np.ndarray:
    """``G'(lambda - mu)`` for each column of ``lam``, ``q`` a
    :class:`~pmedm_vb.solvers.vb.StructuredGaussian`: standard normal if
    ``lambda`` were drawn from ``q``'s Gaussian part."""
    y = q._block_apply(lam - q.mean[:, None], transpose=True)
    return y + q.V @ (q.W.T @ y)


def coordinate_comparison(reference: np.ndarray, other: np.ndarray) -> pd.DataFrame:
    """Per coordinate (rows of the ``(m, draws)`` inputs): mean, sd, skewness and
    0.1/99.9% quantiles under each, and the sd ratio other/reference."""
    from scipy.stats import skew

    def stats(x: np.ndarray, tag: str) -> dict[str, np.ndarray]:
        lo, hi = np.quantile(x, [0.001, 0.999], axis=1)
        return {f"mean_{tag}": x.mean(1), f"sd_{tag}": x.std(1, ddof=1),
                f"skew_{tag}": skew(x, axis=1), f"q001_{tag}": lo, f"q999_{tag}": hi}

    frame = pd.DataFrame({**stats(reference, "ref"), **stats(other, "alt")})
    return frame.assign(sd_ratio=frame["sd_alt"] / frame["sd_ref"])


def largest_cells(inputs: PMEDMInputs, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column of ``lam``: the largest cell's share of ``p`` and its zone and unit."""
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    share, zone, unit = (np.empty(lam.shape[1]) for _ in range(3))
    for d in range(lam.shape[1]):
        logits = log_q - op.adjoint(lam[:, d])
        flat = int(np.argmax(logits))
        share[d] = np.exp(logits.flat[flat] - logsumexp(logits))
        zone[d], unit[d] = divmod(flat, inputs.n_units)
    return share, zone.astype(int), unit.astype(int)


def cell_log_shares(inputs: PMEDMInputs, lam: np.ndarray, cells: Sequence[tuple[int, int]]) -> np.ndarray:
    """``(draws, len(cells))`` log share of ``p`` on each (zone, unit) cell."""
    op = ConstraintOperator(inputs)
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    zones = np.array([z for z, _ in cells], dtype=int)
    units = np.array([u for _, u in cells], dtype=int)
    out = np.empty((lam.shape[1], len(cells)))
    for d in range(lam.shape[1]):
        logits = log_q - op.adjoint(lam[:, d])
        out[d] = logits[zones, units] - logsumexp(logits)
    return out


def psis_khat(log_ratio: np.ndarray) -> float:
    """Pareto-``k`` of the importance ratios ``log pi - log q`` (Yao et al.
    2018): under 0.5 good, 0.5-0.7 acceptable, above 0.7 unreliable."""
    import arviz as az

    _, khat = az.psislw(np.asarray(log_ratio, dtype=float))
    return float(khat)
