"""The collapsed-likelihood engine for a region (step 3, ``PLAN_latent_step3.md``
version 1, §4.2): an adapter to :mod:`pmedm_vb.latent.collapsed`, which is
imported unchanged.

**Layout.** Each zone (a PUMA of the region) is its own tract, with no shared
tract rows: ``J_T = 0``, the block-group cells are the region's S1 cells
(:func:`~pmedm_vb.latent.regions.build_region`), and ``D_b`` reaches only
zone ``b``'s own rows. The stacked vector is ``build_region``'s ``vec(Y_B)``
(cell-major, zone fastest), which is :class:`~pmedm_vb.latent.area.AreaInputs`'
layout with ``J_T = 0``. The base cells are the S1 cells themselves
(``tract_src`` empty, ``bg_src`` the identity), so ``T0`` is the S1 cells'
contributions over the statewide atoms (``data.counts``; the atoms are the
check's units, in order).

**Σ** (:func:`region_sigma`, §3): the replicate covariance across zones as
``build_region`` builds it (its one-case floor covers zero cells), shrinkage
``α = 0.01``, untapered, no zero-count floor, plus the model-gap term
``ω² I`` with ``ω = N_A / (n_A s)``, ``s = 3``
(:func:`~pmedm_vb.assemble.prior.prior_variance` with the region's ``N`` and
``n``). This is the ``Σ`` that ``solve_map`` uses under
``hierarchy="none+prior3"``, so every variant of the ladder sees one ``Σ``.
``S = Σ + Σ_b N_b D_b V_b D_b'`` is then ``B + LL'`` with one block per zone
(:class:`~pmedm_vb.latent.collapsed.CollapsedLikelihood`).

**Per-zone support** (:class:`RegionLikelihood`). Under leave-zone-out each
zone has its own base (zone ``b``'s records at weight zero), so ``p_b ∝
exp{log base_b + (θ_A + u_b)'h}`` with a base per zone
(:func:`~pmedm_vb.latent.region_reference.zone_log_bases`). Only
:meth:`~pmedm_vb.latent.collapsed.CollapsedLikelihood.probabilities` changes;
the gradient's derivation (``dp_bi = p_bi (h_i - h̄_b)' du_b``) holds with a
base per zone, so the rest of the engine is used as it stands.

**Zone covariances** (:func:`zone_covariances`):
``C_b = Cov_{p_b}(t0, h)`` per zone, for the information
(:func:`~pmedm_vb.latent.information.puma_information`). Step 2's
:func:`~pmedm_vb.latent.information.block_covariances` computes one per
distinct ``θ_b``; with a base per zone two zones can share ``θ_b`` and still
differ, so each zone is given its own key.

**A separate deviation basis** (:func:`deviation_engine`, step 3b,
``MODEL_latent.qmd`` version 3): ``θ_A' h`` is folded into the zones' log
bases and the engine runs on ``g`` with ``θ_P = 0``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp

from pmedm_vb.assemble.prior import prior_variance, with_prior
from pmedm_vb.assemble.sigma import Sigma
from pmedm_vb.embed.puma_check import CheckData
from pmedm_vb.latent.area import AreaInputs, ColumnMap
from pmedm_vb.latent.collapsed import CollapsedLikelihood, Evaluation, PairOperator
from pmedm_vb.latent.information import block_covariances, puma_information
from pmedm_vb.latent.regions import RegionProblem


def region_contributions(data: CheckData, tables: tuple[str, ...]) -> sp.csr_matrix:
    """``T0``: the cells of ``tables`` over every statewide atom, in
    ``build_region``'s column order."""
    return sp.hstack([data.counts[t] for t in tables]).tocsr().astype(float)


def region_sigma(problem: RegionProblem, alpha: float = 0.01, prior_sd: float = 3.0) -> Sigma:
    """``Σ = Σ_survey + ω² I`` (module notes)."""
    inputs = problem.inputs
    sigma = inputs.sigma(alpha, taper=None, variance_floor=None)
    return with_prior(sigma, prior_variance(prior_sd, inputs.n, inputs.N))


def region_area(problem: RegionProblem, sigma: Sigma) -> AreaInputs:
    """The region as :class:`~pmedm_vb.latent.area.AreaInputs`: every zone its
    own tract, no tract cells, ``N_b`` the zone's weighted units (truth)."""
    region, inputs = problem.region, problem.inputs
    Z, C = len(region.pumas), len(problem.cells)
    columns = ColumnMap(base_columns=np.arange(C), tract_src=np.zeros(0, dtype=int),
                        bg_src=np.arange(C),
                        checks={"J_T": 0, "J_B": C, "J0": C, "one_to_one": True})
    Y = inputs.targets()
    if Y.size != Z * C:
        raise ValueError("the region's targets are not zones x cells")
    out = AreaInputs(
        puma=region.id, tracts=list(region.pumas), block_groups=list(region.pumas),
        bg_tract=np.arange(Z), tract_cells=[], bg_cells=list(problem.cells), columns=columns,
        Y=Y, sigma=sigma, households=np.asarray(problem.zone_weight, float),
        gq=np.zeros(Z), N_puma=float(inputs.N), n_puma=int(inputs.n),
        checks={"n_zones": Z, "n_cells": C, "n_rows": Y.size,
                "omega": float(np.sqrt(sigma.prior_var)),
                "N_b_sum_over_N": float(problem.zone_weight.sum() / inputs.N)})
    for t in range(Z):
        rows = out.tract_rows(t)
        if not np.array_equal(rows, np.arange(C) * Z + t):
            raise ValueError("zone rows are not the stacked layout's")
    return out


class RegionLikelihood(CollapsedLikelihood):
    """:class:`~pmedm_vb.latent.collapsed.CollapsedLikelihood` with a log base
    per zone (``logb_zones``: ``M × Z``, ``-inf`` off a zone's support)."""

    def __init__(self, area: AreaInputs, H: np.ndarray, logb_zones: np.ndarray,
                 pairs: PairOperator):
        logb_zones = np.asarray(logb_zones, float)
        if logb_zones.shape != (H.shape[0], area.B):
            raise ValueError("logb_zones must be atoms x zones")
        finite = np.where(np.isfinite(logb_zones), logb_zones, -np.inf).max(axis=1)
        super().__init__(area, H, finite, pairs)
        self.logb_zones = logb_zones

    def probabilities(self, theta_P, U) -> np.ndarray:
        eta = self.H @ self.thetas(theta_P, U).T
        eta += self.logb_zones
        eta -= logsumexp(eta, axis=0)[None, :]
        return np.exp(eta, out=eta)


def zone_covariances(engine: CollapsedLikelihood, ev: Evaluation) -> list[np.ndarray]:
    """``C_b = Cov_{p_b}(t0, h)`` per zone (module notes)."""
    B = engine.area.B
    return block_covariances(engine, ev, np.arange(B, dtype=float)[:, None])


def information(engine: CollapsedLikelihood, ev: Evaluation) -> np.ndarray:
    """The mean-part Fisher information of ``U`` (``ZK × ZK``) at ``ev``. An
    engine with its own ``gn_information`` (model version 4's
    :class:`~pmedm_vb.latent.overdispersed.PairLikelihood`) supplies it; every
    other engine is unchanged."""
    own = getattr(engine, "gn_information", None)
    if own is not None:
        return own(ev)
    return puma_information(engine, ev, zone_covariances(engine, ev))


def deviation_engine(area: AreaInputs, H: np.ndarray, theta: np.ndarray, G: np.ndarray,
                     logb_zones: np.ndarray, pairs: PairOperator) -> RegionLikelihood:
    """The engine for zone deviations on their own basis ``g`` (``MODEL_latent.qmd``
    version 3; step 3b): ``p_b ∝ exp{log base_b + θ_A'h + u_b'g}``, with
    ``θ_A'h(z_i)`` folded into each zone's log base. The returned engine is
    evaluated with ``θ_P = 0`` (``np.zeros(G.shape[1])``) and ``U`` on ``g``;
    with ``G = H`` it is step 3's engine at ``θ_A`` (step 3 is unchanged)."""
    offset = np.asarray(H, float) @ np.asarray(theta, float)
    return RegionLikelihood(area, G, np.asarray(logb_zones, float) + offset[:, None], pairs)
