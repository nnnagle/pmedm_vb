"""Bases ``h(z)`` on type space (step 1, piece A5 of ``PLAN_latent.md``).

The candidates (``PLAN_latent_step1.md`` §4.4), each with the two
group-quarters indicators appended (first, as in
:mod:`pmedm_vb.embed.puma_check`):

``dec7``     the stage 1 calibrated decoder's category probabilities for the
             seven attributes of :data:`~pmedm_vb.embed.puma_check.DECODER_ATTRIBUTES`,
             one category dropped per attribute
             (:func:`~pmedm_vb.embed.puma_check.decoder_features`);
``dec37``    the same for every attribute of the run's targets (37 for
             ``stage1_v2``);
``quad``     ``z``, its squares and cross-products
             (:func:`~pmedm_vb.embed.puma_check.quadratic_features`);
``kern100``, ``kern200``
             Gaussian features ``exp{-‖z - c_k‖² / 2ℓ²}`` at 100 or 200
             centres ``c_k``: a design-weighted k-means of ``z``
             (:func:`weighted_kmeans`, k-means++ start, fixed seed), and
             ``ℓ`` the median over atoms (unweighted) of the distance to the
             nearest centre.

**Standardization.** Every column other than the two indicators is centred
and scaled to unit sd with the statewide design weights ``q`` (all ``M``
atoms). The indicators stay 0/1. ``θ`` is then on the standardized scale, so
a prior sd ``σ_θ`` means the same for every basis.

**Minimality** (:func:`minimal_columns`). Columns are visited in order and a
column is dropped when it is constant on the support or its residual variance
after regression on the columns kept before it is below ``tol`` times its own
variance (``q``-weighted, i.e. ``R² > 1 - tol``): an incremental Cholesky
factorization of ``Cov_q(h)``. What is kept has ``Cov_q(h)`` positive
definite; ``K`` and its condition number are reported.

:class:`Basis` holds the standardized matrix and what step 2 needs to rebuild
it for new positions: the raw columns kept, their means and scales, and the
map's own parameters (decoder attributes, kernel centres and ``ℓ``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from pmedm_vb.embed.puma_check import (
    DECODER_ATTRIBUTES,
    decoder_features,
    quadratic_features,
)
from pmedm_vb.latent.atoms import GQ_COLUMNS, Atoms
from pmedm_vb.progress import save_npz, write_text

BASES = ("dec7", "dec37", "quad", "kern100", "kern200")

#: Number of kernel centres per kernel basis.
KERNEL_CENTRES = {"kern100": 100, "kern200": 200}

#: Default relative residual-variance threshold of :func:`minimal_columns`.
COLLINEARITY_TOL = 1e-9


@dataclass
class Basis:
    """One basis over the atoms: ``H`` (``M × K``), standardized and minimal."""

    name: str
    H: np.ndarray
    columns: list[str]
    binary: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    dropped: list[str]
    condition: float
    params: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return self.H.shape[1]

    def summary(self) -> dict:
        return {"basis": self.name, "K": self.K, "K_raw": self.K + len(self.dropped),
                "n_dropped": len(self.dropped), "dropped": self.dropped,
                "condition_cov": self.condition,
                **{k: v for k, v in self.params.items() if k not in ("centres",)}}

    def save(self, directory: Path) -> None:
        """``<name>.npz`` (columns, means, scales, centres) and ``<name>.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"columns": np.array(self.columns), "binary": self.binary,
                  "mean": self.mean, "scale": self.scale}
        if "centres" in self.params:
            arrays["centres"] = np.asarray(self.params["centres"])
        save_npz(directory / f"{self.name}.npz", **arrays)
        write_text(directory / f"{self.name}.json", json.dumps(self.summary(), indent=1,
                                                               default=str))


def weighted_kmeans(z: np.ndarray, w: np.ndarray, k: int, *, seed: int = 1,
                    max_iter: int = 100, tol: float = 1e-8) -> tuple[np.ndarray, int]:
    """Design-weighted k-means (Lloyd's algorithm) with a weighted k-means++
    start. Returns the centres and the iterations used. A centre that loses
    every atom is moved to the atom with the largest weighted squared
    distance to its centre."""
    rng = np.random.default_rng(seed)
    w = np.asarray(w, dtype=float)
    p = w / w.sum()
    centres = np.empty((k, z.shape[1]))
    centres[0] = z[rng.choice(len(z), p=p)]
    d2 = np.square(z - centres[0]).sum(1)
    for j in range(1, k):
        prob = w * d2
        centres[j] = z[rng.choice(len(z), p=prob / prob.sum())]
        d2 = np.minimum(d2, np.square(z - centres[j]).sum(1))
    z2 = np.square(z).sum(1)
    for it in range(1, max_iter + 1):
        dist = z2[:, None] - 2 * z @ centres.T + np.square(centres).sum(1)[None, :]
        label = dist.argmin(1)
        mass = np.bincount(label, weights=w, minlength=k)
        new = np.zeros_like(centres)
        for c in range(z.shape[1]):
            new[:, c] = np.bincount(label, weights=w * z[:, c], minlength=k)
        empty = mass <= 0
        new[~empty] /= mass[~empty, None]
        if empty.any():
            spread = w * dist[np.arange(len(z)), label]
            for j in np.flatnonzero(empty):
                far = int(spread.argmax())
                new[j] = z[far]
                spread[far] = -1.0
        shift = np.abs(new - centres).max()
        centres = new
        if shift <= tol:
            break
    return centres, it


def kernel_features(z: np.ndarray, centres: np.ndarray, ell: float) -> np.ndarray:
    """``exp{-‖z - c_k‖² / 2ℓ²}`` for every atom and centre."""
    d2 = (np.square(z).sum(1)[:, None] - 2 * z @ centres.T
          + np.square(centres).sum(1)[None, :])
    return np.exp(-np.maximum(d2, 0.0) / (2 * ell * ell))


def nearest_distance(z: np.ndarray, centres: np.ndarray) -> np.ndarray:
    d2 = (np.square(z).sum(1)[:, None] - 2 * z @ centres.T
          + np.square(centres).sum(1)[None, :])
    return np.sqrt(np.maximum(d2.min(1), 0.0))


def raw_features(name: str, atoms: Atoms, run: Path, *, seed: int = 1,
                 kmeans_iter: int = 100) -> tuple[np.ndarray, list[str], np.ndarray, dict]:
    """Unstandardized features of basis ``name``: matrix, column names,
    indicator flags (the two group-quarters columns) and parameters."""
    phi, columns, binary = atoms.phi()
    if name in ("dec7", "dec37"):
        attributes = DECODER_ATTRIBUTES if name == "dec7" else "all"
        fm = decoder_features(Path(run), phi, columns, binary, atoms.weight, attributes)
        params = {"kind": "decoder", "attributes": list(attributes) if name == "dec7" else "all",
                  "run": str(run)}
        return fm.F, list(fm.columns), np.asarray(fm.binary), params
    if name == "quad":
        fm = quadratic_features(phi, columns, binary)
        return fm.F, list(fm.columns), np.asarray(fm.binary), {"kind": "quadratic"}
    if name in KERNEL_CENTRES:
        k = KERNEL_CENTRES[name]
        centres, iterations = weighted_kmeans(atoms.z, atoms.weight, k, seed=seed,
                                              max_iter=kmeans_iter)
        ell = float(np.median(nearest_distance(atoms.z, centres)))
        F = np.column_stack([atoms.gq, kernel_features(atoms.z, centres, ell)])
        names = [*GQ_COLUMNS, *(f"k{j + 1}" for j in range(k))]
        binary_k = np.array([True] * len(GQ_COLUMNS) + [False] * k)
        params = {"kind": "kernel", "n_centres": k, "ell": ell, "kmeans_seed": seed,
                  "kmeans_iterations": iterations, "centres": centres}
        return F, names, binary_k, params
    raise ValueError(f"unknown basis {name!r}; known: {BASES}")


def standardize(F: np.ndarray, binary: np.ndarray, q: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale the non-indicator columns with weights ``q``
    (a column constant there keeps scale 1). Returns ``(H, mean, scale)``,
    with mean 0 and scale 1 for the indicators."""
    w = q / q.sum()
    mean = w @ F
    sd = np.sqrt(np.maximum(w @ np.square(F) - np.square(mean), 0.0))
    mean = np.where(binary, 0.0, mean)
    scale = np.where(binary | (sd <= 1e-12 * max(1.0, np.abs(mean).max())), 1.0, sd)
    return (F - mean) / scale, mean, scale


def weighted_cov(H: np.ndarray, q: np.ndarray) -> np.ndarray:
    w = q / q.sum()
    centred = H - w @ H
    return (centred * w[:, None]).T @ centred


def minimal_columns(cov: np.ndarray, tol: float = COLLINEARITY_TOL) -> np.ndarray:
    """Positions of the columns kept by the incremental Cholesky rule (see the
    module notes)."""
    K = cov.shape[0]
    scale = max(float(np.diag(cov).max()), 1e-300)
    L = np.zeros((K, K))
    kept = []
    for j in range(K):
        var = cov[j, j]
        if var <= 1e-12 * scale:
            continue
        if kept:
            idx = np.array(kept)
            r = len(kept)
            l = solve_triangular(L[:r, :r], cov[idx, j], lower=True)
            resid = var - l @ l
        else:
            l, resid = np.zeros(0), var
        if resid <= tol * var:
            continue
        r = len(kept)
        L[r, :r] = l
        L[r, r] = np.sqrt(resid)
        kept.append(j)
    return np.array(kept, dtype=int)


def build_basis(name: str, atoms: Atoms, run: Path, *, seed: int = 1, kmeans_iter: int = 100,
                tol: float = COLLINEARITY_TOL) -> Basis:
    """Basis ``name`` over the atoms: raw features, standardized over the
    statewide support, minimal."""
    F, columns, binary, params = raw_features(name, atoms, run, seed=seed,
                                              kmeans_iter=kmeans_iter)
    q = atoms.q
    H, mean, scale = standardize(F, binary, q)
    del F
    cov = weighted_cov(H, q)
    keep = minimal_columns(cov, tol)
    dropped = [c for j, c in enumerate(columns) if j not in set(keep.tolist())]
    H = np.ascontiguousarray(H[:, keep])
    cov_kept = cov[np.ix_(keep, keep)]
    values = np.linalg.eigvalsh(cov_kept)
    condition = float(values.max() / values.min()) if values.min() > 0 else float("inf")
    params = {**params, "collinearity_tol": tol}
    return Basis(name=name, H=H, columns=[columns[j] for j in keep], binary=binary[keep],
                 mean=mean[keep], scale=scale[keep], dropped=dropped, condition=condition,
                 params=params)
