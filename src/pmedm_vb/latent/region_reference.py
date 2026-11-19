"""Region references and the zones' base distributions (step 3, ``PLAN_latent_step3.md``
version 1, §3 and §4.1).

In the regional run the region ``A`` plays the PUMA's part and its PUMAs
(zones) play the block groups'. From ``MODEL_latent.qmd`` (version 2, notation
kept, ``P`` read as ``A``):

* ``q^A_i ∝ q_i γ^{d(A,i)} exp{θ_A' h(z_i)}`` on the statewide atoms, with step
  1's settings (``h``, ``γ``, ``σ_θ``);
* ``d(A,i)`` (:func:`region_distance`): adjacency steps from atom ``i``'s PUMA
  to the nearest PUMA of the region, 0 inside it;
* ``θ_A`` (:func:`fit_region_reference`): step 1's pseudo-likelihood fit
  (:func:`~pmedm_vb.latent.reference.fit_reference`) on **all** the region's
  records, in both support modes (§3).

**Reproduction check** (:func:`reproduction_check`, §4.1): step 1's in-sample
check on the region's own records. For each rare group of
:data:`~pmedm_vb.embed.puma_check.RARE`, ``N_A Σ_i q^A_i c_i`` against the
region's own weighted total (:func:`~pmedm_vb.embed.puma_check.rare_groups`),
with the ratio and whether step 1's stop range ``[0.8, 1.25]`` holds for the
AIAN and NH/PI groups. Pooled over regions by ``collect_step3.py``.

**Support modes** (§3):

* ``lzo`` (leave-zone-out, primary): when fitting zone ``b`` its own records
  get zero base weight, in every variant;
* ``own`` (with own records): every atom may sit in every zone.

The bases built here (``allowed[b, i]`` says whether atom ``i`` may sit in
zone ``b``):

* :func:`zone_log_bases`: the latent model's per-zone log base weights
  ``log q_i + d(A,i) log γ`` (``-inf`` where not allowed), so that
  ``p_b ∝ exp{log base_b + (θ_A + u_b)'h}``;
* :func:`zone_distributions`: ``p_b`` at ``U = 0`` (``q^A`` restricted to the
  allowed atoms and renormalized per zone);
* :func:`pair_base`: the standard PMEDM's base over (zone, atom) pairs: each
  atom's ``base_i`` spread evenly over the zones it may sit in,
  ``q_bi ∝ base_i 1[allowed_bi] / #{b': allowed_b'i}``, normalized over all
  pairs, so the atoms' marginal is ``base`` in both modes. With the design
  weights as ``base`` and every pair allowed it is
  :func:`~pmedm_vb.assemble.design.design_weights`. The leave-zone-out
  version is the simplest extension of it (chosen here; the plan does not
  prescribe one);
* :func:`pmedm_inputs`: the standard PMEDM's inputs with such a base, on the
  region's own records (``pmedm_own``) or on every statewide atom
  (``pmedm_ref``, base ``q^A``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.embed.puma_check import CheckData, rare_groups
from pmedm_vb.latent.geography import atom_distance
from pmedm_vb.latent.reference import ReferenceFit, fit_reference, log_base
from pmedm_vb.latent.reference_check import STOP_GROUPS, STOP_RATIO
from pmedm_vb.latent.regions import Region, RegionProblem

MODES = ("lzo", "own")


def region_distance(D: pd.DataFrame, pumas: list[str], atom_puma: np.ndarray) -> np.ndarray:
    """``d(A,i)``: the smallest distance from any of the region's PUMAs to each
    atom's PUMA."""
    return np.min([atom_distance(D, p, atom_puma) for p in pumas], axis=0)


def zone_of_atoms(region: Region, atom_puma: np.ndarray) -> np.ndarray:
    """Zone index (into ``region.pumas``) of every atom, ``-1`` outside the region."""
    return pd.Index(list(region.pumas)).get_indexer(atom_puma.astype(str))


@dataclass
class RegionReference:
    region: Region
    theta: np.ndarray           # θ̂_A (K,)
    q: np.ndarray               # q^A over all atoms
    logb: np.ndarray            # log q_i + d(A,i) log γ over all atoms
    distance: np.ndarray        # d(A,i)
    zone: np.ndarray            # zone of each atom, -1 outside the region
    n: int                      # n_A, the region's records
    N: float                    # N_A, their weight total
    fit: ReferenceFit
    gamma: float
    sigma: float

    @property
    def Z(self) -> int:
        return len(self.region.pumas)

    @property
    def in_region(self) -> np.ndarray:
        return self.zone >= 0

    def info(self) -> dict:
        f = self.fit
        return {"region": self.region.id, "n_A": self.n, "N_A": self.N, "gamma": self.gamma,
                "sigma_theta": self.sigma, "iterations": f.iterations, "converged": f.converged,
                "grad_max": f.grad_max, "objective": f.objective, "ess": f.ess,
                "share_outside": f.share_outside, "seconds": f.seconds, "message": f.message,
                "theta_norm": float(np.linalg.norm(self.theta)),
                "max_distance_in_support": int(self.distance.max())}


def fit_region_reference(region: Region, H: np.ndarray, q: np.ndarray, weight: np.ndarray,
                         atom_puma: np.ndarray, D: pd.DataFrame, gamma: float, sigma: float, *,
                         tol: float = 1e-8, max_iter: int = 100) -> RegionReference:
    """``θ̂_A`` on all the region's records (module notes)."""
    zone = zone_of_atoms(region, atom_puma)
    own = zone >= 0
    if not own.any():
        raise ValueError(f"region {region.id} has no atoms")
    distance = region_distance(D, region.pumas, atom_puma)
    if np.any(distance[own] != 0):
        raise ValueError("an atom of the region has a nonzero distance to it")
    logb = log_base(q, distance, gamma)
    w = weight[own]
    mu = (w / w.sum()) @ H[own]
    n = int(own.sum())
    rows = np.flatnonzero(own) if gamma == 0 else None
    fit = fit_reference(H, logb, mu, n, sigma, rows=rows, in_puma=own, tol=tol,
                        max_iter=max_iter)
    return RegionReference(region=region, theta=fit.theta, q=fit.p, logb=logb,
                           distance=distance, zone=zone, n=n, N=float(w.sum()), fit=fit,
                           gamma=float(gamma), sigma=float(sigma))


def reproduction_check(data: CheckData, ref: RegionReference) -> pd.DataFrame:
    """Rare groups: ``q^A`` against the region's own records (module notes)."""
    rows = rare_groups(data, ref.q, ref.N, ref.in_region, None, ref.region.id)
    frame = pd.DataFrame(rows).assign(region=ref.region.id)
    frame["ratio"] = frame["pred"] / frame["truth"].where(frame["truth"] > 0)
    lo, hi = STOP_RATIO
    stop = frame.set_index(["table", "category"]).index.isin(list(STOP_GROUPS))
    frame["stop_group"] = stop
    frame["in_stop_range"] = np.where(stop, frame["ratio"].between(lo, hi), np.nan)
    return frame


# --------------------------------------------------------------------------
# Bases per zone
# --------------------------------------------------------------------------


def allowed(zone: np.ndarray, Z: int, mode: str) -> np.ndarray:
    """``(Z, M)``: may atom ``i`` sit in zone ``b``? (module notes)"""
    if mode not in MODES:
        raise ValueError(f"unknown support mode {mode!r}; known {MODES}")
    out = np.ones((Z, zone.size), dtype=bool)
    if mode == "lzo":
        out[zone[None, :] == np.arange(Z)[:, None]] = False
    return out


def zone_log_bases(logb: np.ndarray, zone: np.ndarray, Z: int, mode: str) -> np.ndarray:
    """``(M, Z)``: log base weights of every zone, ``-inf`` where not allowed."""
    ok = allowed(zone, Z, mode)
    return np.where(ok.T, logb[:, None], -np.inf)


def zone_distributions(q: np.ndarray, zone: np.ndarray, Z: int, mode: str) -> np.ndarray:
    """``(Z, M)``: ``p_b`` at ``U = 0`` (``q`` on the allowed atoms, per zone sum 1)."""
    p = np.where(allowed(zone, Z, mode), q[None, :], 0.0)
    return p / p.sum(axis=1, keepdims=True)


def pair_base(base: np.ndarray, zone: np.ndarray, Z: int, mode: str) -> np.ndarray:
    """``(Z, M)``: the PMEDM's ``q`` over (zone, atom) pairs (module notes).
    ``zone`` is the zone of each of ``base``'s atoms."""
    ok = allowed(zone, Z, mode)
    spread = np.asarray(base, float) / np.maximum(ok.sum(axis=0), 1)
    q = np.where(ok, spread[None, :], 0.0)
    return q / q.sum()


def pmedm_inputs(problem: RegionProblem, q_pairs: np.ndarray, X: sp.spmatrix | None = None,
                 units: pd.DataFrame | None = None) -> PMEDMInputs:
    """The region's standard-PMEDM inputs (``build_region``'s) with the base
    ``q`` over (zone, atom) pairs replaced by ``q_pairs`` and, for the
    statewide support, the units replaced by every atom (``X``: their S1
    cells; ``units``: their frame). ``n`` and ``N`` stay the region's, so the
    entropy weight is ``n_A`` and ``Σ`` (with the multiplier prior) is
    unchanged."""
    inputs = problem.inputs
    changes = {"q": np.asarray(q_pairs, float)}
    if X is not None:
        changes.update(X_B=sp.csr_matrix(X), X_T=sp.csr_matrix((X.shape[0], 0)), units=units)
    out = replace(inputs, **changes)
    out.validate()
    return out
