"""Raking baselines: one weight matrix fitted to the published margins.

Two methods, both starting from the design prior ``W0 = N q`` and both giving a
single ``W`` -- no posterior, so no ``alpha`` and no ``Sigma``.

**Hard raking to block group margins** (:func:`rake_ipf`). Classic iterative
proportional fitting (Deming & Stephan 1940) generalised to count loadings: one
cycle visits every block group constraint ``k`` and moves ``W`` to the
closest matrix in KL divergence that meets that one margin exactly (Csiszar
1975, *Annals of Probability* 3). A unit with loading ``x_ik`` is scaled by
``exp(-theta x_ik)``, with ``theta`` the root of a one-dimensional monotone
equation; for 0/1 loadings that is the familiar ratio update. Tract margins are
not fitted (they are the sums of the block group estimates, so they would add
no target). Published margins may still not be jointly attainable -- two
tables with one universe may publish different block group totals, and a
published zero removes every unit carrying that category from the block group
-- so an exact solution may not exist: the cycle stops at ``tol`` or
``max_sweeps`` and reports how far it got (``experiments/ipf_diagnosis.py``
checks which applies).

**Unbalanced Sinkhorn** (:func:`rake_sinkhorn`). Hard margins replaced by KL
penalties, as in the unbalanced scaling algorithms of Chizat, Peyre, Schmitzer
and Vialard (2018, *Mathematics of Computation* 87), over both levels:

.. math::

    \\min_W \\; \\frac{n}{N} KL(W \\| W_0) + \\sum_k c_k \\, KL(m_k(W) \\| Y_k),
    \\qquad \\sum W = N,

with ``m = X'W`` the fitted margins and ``KL(a || b) = a log(a/b) - a + b``.
``n/N`` is PMEDM's weight on the entropy term (``pmedm_derivation.md``), and
``c_k = Y_k / sigma_k^2`` gives each penalty the curvature of PMEDM's Gaussian
term at its target, since ``KL(a || b) ~ (a - b)^2 / (2b)`` near ``a = b``.
That choice of ``c_k`` is this project's, not the literature's; with it the
method differs from MAP at ``alpha = 1`` mainly in the shape of the penalty.
The total is held at ``N`` exactly, as PMEDM holds it. A cell published as 0
would make its penalty infinite; its target is replaced by its published
standard error divided by 1000 (about 1,000 persons per block group, so a
small but nonzero population).

At the optimum ``W = W0 exp(-sum_k theta_k x_k)`` with
``theta_k = rho_k log(m_k / Y_k)`` and ``rho_k = (N / n) c_k``. Each sweep
updates one ``theta_k`` at a time exactly, by the same one-dimensional root
as :func:`rake_ipf` with the extra ``theta / rho`` term -- cyclic coordinate
ascent on the dual -- and then rescales to the total. For 0/1 loadings the
update is Chizat et al.'s scaling with its ``rho / (rho + 1)`` damping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.progress import logger

#: Divisor turning a published zero's standard error into its Sinkhorn target.
ZERO_TARGET_DIVISOR = 1000.0


@dataclass
class RakeResult:
    """A raked weight matrix and how it got there.

    Attributes
    ----------
    W:
        ``(n_zones, n_units)`` weights.
    method:
        ``"ipf"`` or ``"sinkhorn"``.
    converged:
        Whether the largest residual fell below ``tol``.
    n_sweeps:
        Full cycles over the constraints.
    residual:
        Per stacked constraint row: ``log(m / Y)`` for IPF, the stationarity
        residual ``log(m / Y) - theta / rho`` for Sinkhorn. NaN where a row is
        not fitted (tract rows under IPF) or its target is zero (IPF).
    trace:
        Largest ``|residual|`` after each sweep.
    infeasible:
        Stacked rows with a positive target and no unit able to carry it.
    seconds:
        Wall time of the fit.
    """

    W: np.ndarray
    method: str
    converged: bool
    n_sweeps: int
    residual: np.ndarray
    trace: np.ndarray
    infeasible: np.ndarray
    seconds: float
    theta: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _rows(inputs: PMEDMInputs, level: str) -> list[tuple]:
    """Per constraint column at one level, one update: the stacked rows (one
    per area), the carriers and their loadings, the zones it touches (``None``:
    all) and each zone's area among those rows."""
    if level == "tract":
        X, areas, offset = sp.csc_matrix(inputs.X_T), inputs.zone_tracts(), 0
        n_areas = inputs.Y_T.shape[0]
    else:
        X, areas, offset = sp.csc_matrix(inputs.X_B), np.arange(inputs.n_zones), inputs.Y_T.size
        n_areas = inputs.Y_B.shape[0]
    out = []
    for k in range(X.shape[1]):
        start, stop = X.indptr[k], X.indptr[k + 1]
        out.append((offset + k * n_areas + np.arange(n_areas), X.indices[start:stop],
                    X.data[start:stop], None, areas))
    return out


def _cell_rows(inputs: PMEDMInputs, h, level: str) -> list[tuple]:
    """As :func:`_rows`, over the roll-up's cells (``h`` a
    :class:`~pmedm_vb.assemble.collapse.CollapsedHierarchy`): a cell's loading
    is the sum of its categories', and its rows are cell indices. Cells merging
    the same categories (in different areas) form one update, so the step stays
    vectorised over areas."""
    n_tracts, c_t = inputs.Y_T.shape
    n_zones, c_b = inputs.Y_B.shape
    split = inputs.Y_T.size
    if level == "tract":
        X, zone_area, n_areas = sp.csc_matrix(inputs.X_T), inputs.zone_tracts(), n_tracts
        stacked = np.arange(split)
        category, area = stacked // n_tracts, stacked % n_tracts
    else:
        X, zone_area, n_areas = sp.csc_matrix(inputs.X_B), np.arange(n_zones), n_zones
        stacked = split + np.arange(inputs.Y_B.size)
        category, area = (stacked - split) // n_zones, (stacked - split) % n_zones
    cell = h.row_cell[stacked]
    members: dict[int, list[int]] = {}
    cell_area: dict[int, int] = {}
    for c, k, a in zip(cell, category, area):
        members.setdefault(int(c), []).append(int(k))
        cell_area[int(c)] = int(a)
    groups: dict[tuple[int, ...], list[int]] = {}
    for c, cats in members.items():
        groups.setdefault(tuple(sorted(cats)), []).append(c)
    out = []
    for cats, cells in groups.items():
        x = np.asarray(X[:, list(cats)].sum(axis=1)).ravel()
        carriers = np.flatnonzero(x)
        areas = np.array([cell_area[c] for c in cells])
        local = np.full(n_areas, -1)
        local[areas] = np.arange(areas.size)
        zones = np.flatnonzero(local[zone_area] >= 0)
        out.append((np.array(cells), carriers, x[carriers], zones, local[zone_area[zones]]))
    return out


def _grouped_logsumexp(values: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    """``log sum exp`` of ``values`` (one per zone) within each group."""
    top = np.full(n_groups, -np.inf)
    np.maximum.at(top, groups, values)
    shift = np.where(np.isfinite(top), top, 0.0)
    total = np.zeros(n_groups)
    np.add.at(total, groups, np.exp(values - shift[groups]))
    with np.errstate(divide="ignore"):
        return shift + np.log(total)


def _solve_step(
    log_w: np.ndarray, x: np.ndarray, groups: np.ndarray, n_groups: int,
    log_target: np.ndarray, theta: np.ndarray, inv_rho: np.ndarray,
    iters: int = 50, tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Per area, the step ``delta`` solving
    ``log m(delta) = log Y + (theta + delta) / rho``.

    ``log_w`` is ``(n_zones, carriers)``, ``x`` the carriers' loadings. The left
    side is a log-sum-exp of affine functions of ``delta``, so convex and
    decreasing, and the right side increasing: Newton from 0 converges
    monotonically after its first step, and ``x >= 1`` on carriers keeps the
    derivative at least 1 in size. Returns ``delta`` and the log margins
    before the step.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return _newton(log_w, x, groups, n_groups, log_target, theta, inv_rho, iters, tol)


def _newton(log_w, x, groups, n_groups, log_target, theta, inv_rho, iters, tol):
    log_x = np.log(x)
    delta = np.zeros(n_groups)
    start = None
    for _ in range(iters):
        shifted = log_w - delta[groups][:, None] * x
        log_m = _grouped_logsumexp(logsumexp(shifted + log_x, axis=1), groups, n_groups)
        log_m2 = _grouped_logsumexp(logsumexp(shifted + 2 * log_x, axis=1), groups, n_groups)
        if start is None:
            start = log_m
        h = log_m - log_target - (theta + delta) * inv_rho
        slope = -np.exp(log_m2 - log_m) - inv_rho
        live = np.isfinite(log_m) & np.isfinite(log_target)
        step = np.where(live, h / slope, 0.0)
        delta = delta - step
        if np.all(np.abs(step) < tol):
            break
    return delta, start


def _fit(
    inputs: PMEDMInputs, columns: list[tuple], targets: np.ndarray, rho: np.ndarray | None,
    *, tol: float, max_sweeps: int, fix_total: bool, method: str, log_every: int,
) -> RakeResult:
    """``columns`` from :func:`_rows` or :func:`_cell_rows`; ``targets`` (and
    ``rho``) indexed by their rows."""
    start_time = time.perf_counter()
    with np.errstate(divide="ignore"):
        log_W = np.log(inputs.N * inputs.q)
    m = targets.size
    theta = np.zeros(m)
    inv_rho = np.zeros(m) if rho is None else 1.0 / rho
    with np.errstate(divide="ignore"):
        log_targets = np.log(targets)
    residual = np.full(m, np.nan)
    trace, infeasible = [], set()
    converged, sweep = False, 0
    for sweep in range(1, max_sweeps + 1):
        for rows, carriers, x, zones, areas in columns:
            if carriers.size == 0:
                infeasible.update(rows[targets[rows] > 0].tolist())
                continue
            at = np.ix_(zones, carriers) if zones is not None else (slice(None), carriers)
            block = log_W[at]
            zero = ~np.isfinite(log_targets[rows])
            if zero.any():  # hard zero (IPF only): its carriers leave those areas
                block[zero[areas]] = -np.inf
            delta, log_m = _solve_step(block, x, areas, rows.size, log_targets[rows],
                                       theta[rows], inv_rho[rows])
            infeasible.update(rows[~np.isfinite(log_m) & ~zero].tolist())
            with np.errstate(invalid="ignore"):
                residual[rows] = np.where(
                    zero, np.nan, log_m - log_targets[rows] - theta[rows] * inv_rho[rows]
                )
            theta[rows] += delta
            log_W[at] = block - delta[areas][:, None] * x
        if fix_total:
            log_W += np.log(inputs.N) - logsumexp(log_W)
        worst = float(np.nanmax(np.abs(np.where(np.isfinite(residual), residual, np.nan))))
        trace.append(worst)
        if sweep % log_every == 0 or sweep == 1:
            logger.info("  %s sweep %5d  max |residual| %.3g", method, sweep, worst)
        if worst < tol:
            converged = True
            break
    logger.info("%s: %s after %d sweeps, max |residual| %.3g, %d infeasible row(s)",
                method, "converged" if converged else "NOT converged", sweep, trace[-1],
                len(infeasible))
    return RakeResult(
        W=np.exp(log_W), method=method, converged=converged, n_sweeps=sweep,
        residual=residual, trace=np.asarray(trace),
        infeasible=np.array(sorted(infeasible), dtype=int),
        seconds=time.perf_counter() - start_time, theta=theta,
    )


def rake_ipf(
    inputs: PMEDMInputs, *, tol: float = 1e-3, max_sweeps: int = 2000, log_every: int = 50,
    rollup=None,
) -> RakeResult:
    """Hard raking of ``N q`` to the block group margins.

    ``tol`` is on ``|log(m / Y)|``, the relative error of each fitted margin;
    ``1e-3`` is 0.1%. The residual reported at a sweep's end is each row's
    error just before its own update, so it measures how far the other margins
    pulled it away.
    """
    if rollup is not None:
        h = _rolled(inputs, rollup)
        return _fit(inputs, _cell_rows(inputs, h, "block group"), h.Y_c.astype(float).copy(),
                    None, tol=tol, max_sweeps=max_sweeps, fix_total=False, method="ipf",
                    log_every=log_every)
    targets = inputs.targets().astype(float).copy()
    targets[: inputs.Y_T.size] = np.nan  # tract rows are not fitted
    return _fit(inputs, _rows(inputs, "block group"), np.nan_to_num(targets, nan=1.0), None,
                tol=tol, max_sweeps=max_sweeps, fix_total=False, method="ipf",
                log_every=log_every)


def sinkhorn_penalties(inputs: PMEDMInputs, variance_floor: str | float | None = "zero"
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Targets, with published zeros replaced, and ``rho_k = (N / n) Y_k / sigma_k^2``.

    ``sigma_k^2`` is the published variance with ``variance_floor`` applied, as
    in the MAP and VB fits. A zero's replacement target is its published
    standard error over :data:`ZERO_TARGET_DIVISOR`.
    """
    return _penalties(inputs, inputs.targets().astype(float),
                      inputs.floored_variances(variance_floor))


def _penalties(inputs: PMEDMInputs, targets: np.ndarray, variances: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    targets = targets.copy()
    zero = targets <= 0
    targets[zero] = np.sqrt(variances[zero]) / ZERO_TARGET_DIVISOR
    rho = (inputs.N / inputs.n) * targets / variances
    return targets, rho


def rake_name(puma: str, method: str, rollup=None) -> str:
    """The result name of a raking run: ``<puma>_<method>``, with ``_r<spec>``
    appended when it rakes to the roll-up's cells."""
    if rollup is None:
        return f"{puma}_{method}"
    from pmedm_vb.assemble.rollup import RollupSpec

    spec = rollup if isinstance(rollup, RollupSpec) else RollupSpec.parse(str(rollup))
    return f"{puma}_{method}_r{spec}"


def _rolled(inputs: PMEDMInputs, rollup):
    """The roll-up's cells for raking (``rollup`` a spec or its text, e.g. ``"50h20"``)."""
    from pmedm_vb.assemble.collapse import CollapsedHierarchy
    from pmedm_vb.assemble.rollup import RollupSpec

    spec = rollup if isinstance(rollup, RollupSpec) else RollupSpec.parse(str(rollup))
    return CollapsedHierarchy.build(inputs, "none", spec)


def _floored(h, variance_floor) -> np.ndarray:
    """The cells' variances with the floor applied, as the model's ``Sigma`` does."""
    if variance_floor == "zero":
        return np.maximum(h.v_c, h.floor_c)
    if variance_floor is not None:
        return np.maximum(h.v_c, float(variance_floor))
    return h.v_c


def rake_sinkhorn(
    inputs: PMEDMInputs, *, variance_floor: str | float | None = "zero",
    tol: float = 1e-6, max_sweeps: int = 5000, log_every: int = 50, rollup=None,
) -> RakeResult:
    """Unbalanced Sinkhorn over both levels; see the module docstring.

    ``tol`` is on the stationarity residual ``|log(m / Y) - theta / rho|``.
    """
    if rollup is not None:
        h = _rolled(inputs, rollup)
        targets, rho = _penalties(inputs, h.Y_c.astype(float), _floored(h, variance_floor))
        columns = _cell_rows(inputs, h, "tract") + _cell_rows(inputs, h, "block group")
    else:
        targets, rho = sinkhorn_penalties(inputs, variance_floor)
        columns = _rows(inputs, "tract") + _rows(inputs, "block group")
    return _fit(inputs, columns, targets, rho,
                tol=tol, max_sweeps=max_sweeps, fix_total=True, method="sinkhorn",
                log_every=log_every)
