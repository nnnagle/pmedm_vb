"""Calibration of the stage 2 posterior on the regional run with survey error
(step 4, D2 coverage of ``PLAN_latent.md``; ``PLAN_latent_step4.md`` version 2,
§3 and §4.4).

**Truth** (both simulations; :func:`draw_truth`): a realized population per
replicate. Zone ``b``'s population is ``N_b`` units (its design-weighted unit
total, rounded) drawn independently from the zone's own records with
probability ``∝ w_i``: ``c ~ Multinomial(N_b, w/Σw)`` over the records
(households carry their members, since ``t(x_i)`` is the whole unit's
contribution). Its tables ``A(C_true) = Σ_i c_i t(x_i)`` are the replicate's
target, and every functional's truth is computed from ``c``. The region's
records and ``θ_A`` are not perturbed; ``N_b`` is known.

**Simulation G** (:func:`draw_gaussian`): ``Y = A(C_true) + ε``,
``ε ~ N(0, Σ_survey)`` with the region's replicate covariance (untapered,
``α = 0.01``; ``build_region``'s variances with their one-case floor). The
draw is joint over the fitted cells and the held-out sets B1 and B2 (one
replicate covariance over all of them, ``Σ = D + BB'``:
``ε = D^{1/2} ξ + B η``), so the sets' errors are correlated as one survey's
are (a choice made here; the plan says B1 and B2 are drawn the same way as
the fitted tables). Fitting uses the same ``Σ_survey`` plus ``ω² I``
(:func:`~pmedm_vb.latent.region_engine.region_sigma`).

**Simulation S** (:func:`draw_bernoulli`): a 1-in-8 Bernoulli sample of the
replicate's realized population. Record ``i`` of zone ``b`` contributes
``n_i ~ Binomial(c_i, π)`` sampled units, ``π = 1/8`` (a household with its
members); no Gaussian error is added;
``Ŷ = π⁻¹ Σ_i n_i t(x_i)`` (Horvitz–Thompson) and
``Σ̂ = (1 - π)/π² Σ_i n_i t(x_i) t(x_i)'`` over all of the zone's cells
(fitted, B1, B2), zones independent. A cell with ``Ŷ = 0`` gets the variance
of one sampled unit, ``(1 - π)/π²``, on the diagonal. Fitting uses ``Σ̂`` of
the fitted cells plus ``ω² I`` (:class:`BlockSigma`); the held-out sets'
standard errors are ``sqrt(diag Σ̂)``.

:class:`BlockSigma` carries ``Σ̂ + ω² I`` in the form the engine reads
(``d``, a factor ``b`` with ``Σ = diag(d) + bb'``, untapered): each zone's
``Σ̂_b = V Λ V'`` (eigendecomposition, eigenvalues above ``1e-10`` of the
largest kept) gives that zone's columns of ``b`` on its own rows; the floor
and ``ω²`` are in ``d``.

**Interval statistics** (:func:`interval_stats`, §4.4), from the posterior
sample of each functional: the mean and sd over the sample, the 50% and 90%
intervals from its empirical 25/75% and 5/95% quantiles, coverage of the
truth, the standardized error ``z = (truth - mean)/sd``, the PIT (share of the
sample below the truth; ties counted half), widths, and whether the 90%
interval's lower end is below zero. :func:`summarize` pools them by
simulation and functional class (§4.4); :func:`decide` applies §6.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from pmedm_vb.latent.regions import RegionProblem

PI = 1.0 / 8.0
SIMULATIONS = ("G", "S")
LEVELS = (0.5, 0.9)
#: §6 coverage bands.
BANDS = {0.9: (0.85, 0.95), 0.5: (0.42, 0.58)}


# --------------------------------------------------------------------------
# Simulated surveys
# --------------------------------------------------------------------------


@dataclass
class SurveyDraw:
    """One replicate's published tables over all cells (``Z × C_all``), their
    variances, and per zone the estimated covariance (simulation S only)."""

    Y: np.ndarray
    var: np.ndarray                     # (Z, C_all) published variance
    blocks: list | None = None          # S: Σ̂_b (C_all × C_all) before the floor
    floor: np.ndarray | None = None     # S: (Z, C_all) floor added on the diagonal
    sampled_units: np.ndarray | None = None


@dataclass
class TruthDraw:
    """A replicate's realized population: ``counts`` (units per record of the
    region, in ``problem_all``'s unit order), the zones' ``N_b`` and tables
    (``Z × C_all``)."""

    counts: np.ndarray
    N: np.ndarray
    Y: np.ndarray


def draw_truth(problem_all: RegionProblem, rng: np.random.Generator) -> TruthDraw:
    """The realized population (module notes)."""
    inputs = problem_all.inputs
    X = inputs.X_B.tocsr()
    w = inputs.units["weight"].to_numpy(dtype=float)
    Z, C = inputs.Y_B.shape
    counts = np.zeros(w.size, dtype=np.int64)
    N = np.zeros(Z, dtype=np.int64)
    for b in range(Z):
        own = np.flatnonzero(problem_all.zone_of_unit == b)
        N[b] = int(np.rint(w[own].sum()))
        counts[own] = rng.multinomial(N[b], w[own] / w[own].sum())
    Y = np.zeros((Z, C))
    for b in range(Z):
        own = np.flatnonzero(problem_all.zone_of_unit == b)
        Y[b] = np.asarray(X[own].T @ counts[own].astype(float)).ravel()
    return TruthDraw(counts=counts, N=N, Y=Y)


def draw_gaussian(problem_all: RegionProblem, truth: TruthDraw, rng: np.random.Generator,
                  alpha: float = 0.01) -> SurveyDraw:
    """Simulation G (module notes)."""
    inputs = problem_all.inputs
    sigma = inputs.sigma(alpha, taper=None, variance_floor=None)
    n = sigma.d.size
    eps = np.sqrt(sigma.d) * rng.standard_normal(n)
    if sigma.b is not None:
        eps += sigma.b @ rng.standard_normal(sigma.b.shape[1])
    Z, C = inputs.Y_B.shape
    E = eps.reshape(C, Z).T                      # stacked: cell-major, zone fastest
    var = sigma.diagonal().reshape(C, Z).T
    return SurveyDraw(Y=truth.Y + E, var=var)


def draw_bernoulli(problem_all: RegionProblem, truth: TruthDraw, rng: np.random.Generator,
                   pi: float = PI) -> SurveyDraw:
    """Simulation S (module notes)."""
    inputs = problem_all.inputs
    X = inputs.X_B.tocsr()
    n = rng.binomial(truth.counts, pi)
    Z, C = inputs.Y_B.shape
    Y, var, blocks, floor = np.zeros((Z, C)), np.zeros((Z, C)), [], np.zeros((Z, C))
    c = (1.0 - pi) / pi ** 2
    for b in range(Z):
        own = np.flatnonzero(problem_all.zone_of_unit == b)
        Xb = X[own]
        nb = n[own].astype(float)
        Y[b] = np.asarray(Xb.T @ nb).ravel() / pi
        S = c * np.asarray((Xb.T @ Xb.multiply(nb[:, None])).todense())
        blocks.append(S)
        floor[b] = np.where(Y[b] == 0, c, 0.0)
        var[b] = np.diag(S) + floor[b]
    return SurveyDraw(Y=Y, var=var, blocks=blocks, floor=floor, sampled_units=n)


def with_tables(problem: RegionProblem, columns: np.ndarray, draw: SurveyDraw
                ) -> RegionProblem:
    """``problem`` with its tables (``Y_B``) and variances (``sigma_v``)
    replaced by the draw's for ``columns`` (positions in the joint cells)."""
    Y = draw.Y[:, columns]
    sigma_v = draw.var[:, columns].T.ravel()               # stacked order
    inputs = problem.inputs
    # Simulation S has no replicate factor: its covariance is Σ̂ (BlockSigma).
    sigma_l = inputs.sigma_l if draw.blocks is None else None
    return replace(problem, inputs=replace(inputs, Y_B=Y, sigma_v=sigma_v, sigma_l=sigma_l))


class BlockSigma:
    """``Σ̂ + ω² I`` in the engine's form (module notes)."""

    is_tapered = False
    groups = None

    def __init__(self, blocks: list[np.ndarray], floor: np.ndarray, Z: int, prior_var: float,
                 rel_tol: float = 1e-10):
        C = blocks[0].shape[0]
        self.prior_var = float(prior_var)
        self.d = floor.T.ravel() + self.prior_var          # stacked: cell-major, zone fastest
        factors = []
        for b, S in enumerate(blocks):
            lam, V = np.linalg.eigh(0.5 * (S + S.T))
            keep = lam > rel_tol * max(float(lam.max()), 1e-300)
            col = np.zeros((C * Z, int(keep.sum())))
            col[np.arange(C) * Z + b] = V[:, keep] * np.sqrt(lam[keep])
            factors.append(col)
        self.b = np.hstack(factors) if factors else None
        self.rank = 0 if self.b is None else self.b.shape[1]

    def diagonal(self) -> np.ndarray:
        return self.d + (np.square(self.b).sum(axis=1) if self.b is not None else 0.0)

    def to_dense(self) -> np.ndarray:
        out = np.diag(self.d)
        if self.b is not None:
            out += self.b @ self.b.T
        return out


# --------------------------------------------------------------------------
# Interval statistics
# --------------------------------------------------------------------------


def interval_stats(samples: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per (zone, functional) statistics of a posterior sample ``(S, Z, F)``
    against ``truth`` ``(Z, F)`` (module notes)."""
    s = np.asarray(samples, float)
    q05, q25, q75, q95 = np.quantile(s, [0.05, 0.25, 0.75, 0.95], axis=0)
    mean, sd = s.mean(axis=0), s.std(axis=0, ddof=1)
    pit = ((s < truth[None]).mean(axis=0) + 0.5 * (s == truth[None]).mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (truth - mean) / sd, np.nan)
    return {"mean": mean, "sd": sd, "q05": q05, "q25": q25, "q75": q75, "q95": q95,
            "cover50": (truth >= q25) & (truth <= q75), "cover90": (truth >= q05) & (truth <= q95),
            "z": z, "pit": pit, "width50": q75 - q25, "width90": q95 - q05,
            "below_zero": q05 < 0}


def summarize(frame: pd.DataFrame, keys=("simulation", "cls")) -> pd.DataFrame:
    """Pooled calibration by ``keys`` (§4.4): coverage, RMS z, widths, the
    share of 90% intervals below zero, the direct-estimate reference (fitted
    cells), and the variation of coverage across replicates."""
    keys = list(keys)
    g = frame.groupby(keys)
    out = pd.DataFrame({
        "n": g.size(),
        "cover50": g["cover50"].mean(), "cover90": g["cover90"].mean(),
        "rms_z": g["z"].apply(lambda z: float(np.sqrt(np.nanmean(np.square(z))))),
        "median_width90": g["width90"].median(),
        "median_width90_over_se": g["width90_over_se"].median()
        if "width90_over_se" in frame else np.nan,
        "median_width90_over_truth": g["width90_over_truth"].median(),
        "share_below_zero": g["below_zero"].mean(),
    })
    if "direct_cover90" in frame:
        out["direct_cover90"] = g["direct_cover90"].mean()
    per_rep = frame.groupby([*keys, "replicate"])[["cover50", "cover90"]].mean()
    spread = per_rep.groupby(level=list(range(len(keys)))).agg(["std", "min", "max"])
    spread.columns = [f"{a}_rep_{b}" for a, b in spread.columns]
    return out.join(spread)


def pit_histogram(frame: pd.DataFrame, bins: int = 10, keys=("simulation", "cls")
                  ) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = {}
    for key, sub in frame.groupby(list(keys)):
        counts, _ = np.histogram(np.clip(sub["pit"], 0, 1), bins=edges)
        rows[key] = counts / max(counts.sum(), 1)
    return pd.DataFrame(rows, index=[f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]).T


def decide(table: pd.DataFrame, primary: str = "S") -> tuple[dict, list[str]]:
    """§6 on :func:`summarize`'s table (keys simulation, cls)."""
    lines, verdict = [], {}
    for sim in [s for s in (primary, *[x for x in SIMULATIONS if x != primary])
                if s in table.index.get_level_values(0)]:
        sub = table.loc[sim]
        ok = {}
        for cls, row in sub.iterrows():
            ok[cls] = all(BANDS[lv][0] <= row[f"cover{int(lv * 100)}"] <= BANDS[lv][1]
                          for lv in LEVELS)
        others = {c: v for c, v in ok.items() if c != "rare"}
        others_pass = all(others.values()) if others else False
        rare = ok.get("rare")
        if others_pass and rare:
            out = "calibrated, rare groups included: stage 2 is the posterior; go to step 6"
        elif others_pass and rare is False:
            out = "calibrated except rare groups: step 5 (samplers)"
        else:   # other classes fail, whether rare groups pass or fail (plan v2, §6)
            out = ("not calibrated outside rare groups: stop and review the posterior "
                   "approximation (Laplace, tau fixed, theta_A fixed) before any sampler"
                   + (" (rare groups pass; plan version 2: also stop and review)"
                      if rare else ""))
        verdict[sim] = {"classes": ok, "others_pass": others_pass, "rare_pass": rare,
                        "decision": out}
        role = "primary" if sim == primary else "diagnostic"
        lines.append(f"Simulation {sim} ({role}): " + ", ".join(
            f"{c} {'pass' if v else 'FAIL'}" for c, v in ok.items()) + f" -> {out}")
    if primary in verdict and "G" in verdict and primary != "G":
        if not verdict[primary]["others_pass"] and verdict["G"]["others_pass"]:
            lines.append("  S fails where G passes: the failure is in the survey error's form "
                         "or in estimating Sigma, not in the posterior approximation.")
    decision = verdict.get(primary, {}).get("decision", "incomplete")
    lines.append(f"DECISION (section 6, simulation {primary}): {decision}")
    return {"decision": decision, "per_simulation": verdict}, lines
