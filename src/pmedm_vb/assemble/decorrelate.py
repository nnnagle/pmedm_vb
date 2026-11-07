"""Decorrelate the cells resting on about one sampled case: the ``+decor<c>`` level suffix.

``Sigma(alpha) = D^1/2 [(1 - alpha) R + alpha I] D^1/2`` (:mod:`pmedm_vb.assemble.sigma`),
with ``R`` the correlation of the 80 replicate deviations. Outside the span of
the replicates (80 directions per taper group) every direction keeps only
``alpha`` of its variance, and the dual's Gaussian term is correspondingly
flat there. For a cell resting on about one sampled case -- ``k_hat =
(estimate / SE)^2`` below ``c`` -- that matters: its replicate pattern is one
household's, so the correlations it gets are a single record's weight
perturbations rather than an estimate of how the cell's error moves with its
neighbours'; and when the fit lets its allocation collapse to zero (the data
side of the dual goes flat too), only that weak curvature holds its
multiplier, which then drifts to 20-65 on real Tennessee PUMAs
(``experiments/cell_cases.py``).

With ``+decor<c>``, every tract and block group row with a positive estimate
and ``k_hat < c`` loses its replicate deviations (``L_i = 0``). Its residual
diagonal is then ``D_i = v_i`` exactly -- the published (or floored) variance
-- and it has no correlation with any other row: just how a published zero is
already carried. Its own direction then has the full curvature ``c v_i``
rather than ``alpha v_i``.

``k_hat`` is computed from the published variance (``sigma_v``), not the
floored one. The PUMA rows of the ``puma`` hierarchy keep their replicates,
which are summed from the tract rows' undecorrelated deviations
(:meth:`~pmedm_vb.assemble.hierarchy.Hierarchy.build`); only
:meth:`~pmedm_vb.assemble.hierarchy.Hierarchy.sigma` applies the change, so
every solver (MAP and its Hessian, VB, HMC) sees it. The Tweedie zero term
selects rows published as zero, so decorrelated positives are not in it.

It travels with the hierarchy level: ``"puma+decor2"``, or with other suffixes,
``"puma+prior3+decor2"``. The plain levels (``none``, ``tract``, ``puma``) only.
"""

from __future__ import annotations

import numpy as np

SEPARATOR = "+decor"


def split_decor(level: str) -> tuple[str, float | None]:
    """``"puma+decor2"`` -> ``("puma", 2.0)``; none -> ``(level, None)``.
    Another suffix after it (``"+prior..."``) is kept."""
    if SEPARATOR not in level:
        return level, None
    head, _, rest = level.partition(SEPARATOR)
    spec, plus, tail = rest.partition("+")
    try:
        c = float(spec)
    except ValueError:
        raise ValueError(f"decorrelation is +decor<c>, e.g. +decor2; got {level!r}") from None
    if not c > 0:
        raise ValueError(f"the decorrelation threshold must be positive, got {level!r}")
    return head + (plus + tail if plus else ""), c


def decor_level(level: str, c: float | None) -> str:
    """The level string with ``+decor<c>`` appended, for a CLI's ``--decorrelate-k``."""
    if c is None:
        return level
    if not c > 0:
        raise ValueError(f"--decorrelate-k must be positive, got {c}")
    return f"{level}{SEPARATOR}{c:g}"


def decorrelated_rows(inputs, c: float) -> np.ndarray:
    """Indices of the stacked rows with a positive estimate and ``k_hat < c``
    (``k_hat`` from the published variance)."""
    y = inputs.targets()
    v = np.asarray(inputs.sigma_v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_hat = np.where(v > 0, y * y / v, np.inf)
    return np.flatnonzero((y > 0) & (k_hat < c))


def decorrelate(sigma, rows: np.ndarray):
    """``sigma`` with these rows' replicate deviations set to zero."""
    import dataclasses

    if sigma.l is None or not rows.size:
        return sigma
    l = np.array(sigma.l, dtype=float)
    l[rows] = 0.0
    return dataclasses.replace(sigma, l=l)
