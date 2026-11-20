"""The zone deviations' own basis ``g`` and its scale ``τ^g_ref`` (step 3b,
``PLAN_latent_step3b.md`` version 1, §2-§3; ``MODEL_latent.qmd`` version 3).

The model (version 3): ``p_bi ∝ q_i γ^{d(A,i)} exp{θ_A' h(z_i) + u_b' g(z_i)}``,
``u_b ~ N(0, τ² I_{K_g})``, with ``h`` = step 1's basis (``kern100``) and
``θ_A`` fixed at its records fit; version 2 is ``g = h``.

**Bases** (:func:`deviation_basis`): ``kern30``, ``kern100``, ``kern200``,
Gaussian kernels on the whitened ``z`` at design-weighted k-means centres
with the two group-quarters indicators, each with its own width ``ℓ`` (the
median distance from the atoms to their nearest centre), standardized and
made minimal exactly as step 1 builds a kernel basis
(:mod:`pmedm_vb.latent.basis`: weighted k-means++ start, seed 1, at most 100
Lloyd iterations, collinearity tolerance 1e-9). A basis step 1 saved under
its ``inputs/bases`` is rebuilt from those files
(:func:`~pmedm_vb.latent.area.load_basis`), so ``kern100`` is step 1's ``h``
exactly; one it did not save (``kern30``) is built here with the same recipe
(:func:`build_kernel_basis`).

**Scale** (:func:`puma_tilts`, :func:`tau_ref`, §3): for each ``g``, every
PUMA's tilt on ``g`` alone is fitted from its own records with step 1's
pseudo-likelihood (:func:`~pmedm_vb.latent.reference.fit_reference`,
``σ_θ = 1``, ``γ = 0.5``, support every atom), and
``τ^g_ref = (mean_k Var_P(θ̂_{P,k}))^{1/2}`` over the PUMAs (variance with
divisor ``n_PUMAs``), as step 2's ``τ_ref``
(:meth:`~pmedm_vb.latent.area.Step1.tau_ref`).

**The engine** for a region (:func:`~pmedm_vb.latent.region_engine.deviation_engine`)
folds ``θ_A' h(z_i)`` into each zone's log base weight, so step 2's engine runs
unchanged on ``g`` with its ``θ_P`` set to 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.latent.area import load_basis
from pmedm_vb.latent.atoms import GQ_COLUMNS, Atoms
from pmedm_vb.latent.basis import (
    COLLINEARITY_TOL,
    Basis,
    kernel_features,
    minimal_columns,
    nearest_distance,
    standardize,
    weighted_cov,
    weighted_kmeans,
)
from pmedm_vb.latent.geography import atom_distance
from pmedm_vb.latent.reference import fit_reference, log_base

#: Deviation bases and their numbers of kernel centres (§3).
DEVIATION_BASES = {"kern30": 30, "kern100": 100, "kern200": 200}


def build_kernel_basis(name: str, atoms: Atoms, *, seed: int = 1, kmeans_iter: int = 100,
                       tol: float = COLLINEARITY_TOL) -> Basis:
    """A kernel basis with ``DEVIATION_BASES[name]`` centres, by step 1's recipe
    (:func:`~pmedm_vb.latent.basis.raw_features` and
    :func:`~pmedm_vb.latent.basis.build_basis` for ``kern100``/``kern200``)."""
    k = DEVIATION_BASES[name]
    centres, iterations = weighted_kmeans(atoms.z, atoms.weight, k, seed=seed,
                                          max_iter=kmeans_iter)
    ell = float(np.median(nearest_distance(atoms.z, centres)))
    F = np.column_stack([atoms.gq, kernel_features(atoms.z, centres, ell)])
    columns = [*GQ_COLUMNS, *(f"k{j + 1}" for j in range(k))]
    binary = np.array([True] * len(GQ_COLUMNS) + [False] * k)
    q = atoms.q
    H, mean, scale = standardize(F, binary, q)
    del F
    cov = weighted_cov(H, q)
    keep = minimal_columns(cov, tol)
    kept = set(keep.tolist())
    values = np.linalg.eigvalsh(cov[np.ix_(keep, keep)])
    condition = float(values.max() / values.min()) if values.min() > 0 else float("inf")
    return Basis(name=name, H=np.ascontiguousarray(H[:, keep]),
                 columns=[columns[j] for j in keep], binary=binary[keep], mean=mean[keep],
                 scale=scale[keep], dropped=[c for j, c in enumerate(columns) if j not in kept],
                 condition=condition,
                 params={"kind": "kernel", "n_centres": k, "ell": ell, "kmeans_seed": seed,
                         "kmeans_iterations": iterations, "centres": centres,
                         "collinearity_tol": tol})


def deviation_basis(name: str, atoms: Atoms, step1_bases: Path, run: Path,
                    save_to: Path | None = None) -> tuple[np.ndarray, dict]:
    """``G`` (``M × K_g``) and a record of where it came from (module notes)."""
    if name not in DEVIATION_BASES:
        raise ValueError(f"unknown deviation basis {name!r}; known {list(DEVIATION_BASES)}")
    step1_bases = Path(step1_bases)
    if (step1_bases / f"{name}.npz").exists() and (step1_bases / f"{name}.json").exists():
        G = load_basis(step1_bases, name, atoms, run)
        return G, {"basis": name, "K": int(G.shape[1]), "source": str(step1_bases)}
    basis = build_kernel_basis(name, atoms)
    if save_to is not None:
        basis.save(Path(save_to))
    return basis.H, {**basis.summary(), "source": "built (step 1's recipe)"}


def puma_tilt(G: np.ndarray, q: np.ndarray, weight: np.ndarray, atom_puma: np.ndarray,
              D: pd.DataFrame, puma: str, *, gamma: float = 0.5, sigma: float = 1.0) -> dict:
    """One PUMA's tilt on ``g`` from its own records (module notes)."""
    own = atom_puma == str(puma)
    logb = log_base(q, atom_distance(D, puma, atom_puma), gamma)
    w = weight[own]
    mu = (w / w.sum()) @ G[own]
    fit = fit_reference(G, logb, mu, int(own.sum()), sigma, in_puma=own)
    return {"puma": str(puma), "theta": fit.theta, "iterations": fit.iterations,
            "converged": fit.converged, "grad_max": fit.grad_max, "seconds": fit.seconds}


def tau_ref(theta: np.ndarray) -> float:
    """``(mean_k Var_P(θ̂_{P,k}))^{1/2}`` (divisor ``n_PUMAs``), as step 2's."""
    return float(np.sqrt(np.asarray(theta).var(axis=0).mean()))
