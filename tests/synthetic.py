"""Small synthetic PMEDM problems with the same structure as the real ones.

Zones are block groups nested in tracts, ``A_B`` is the identity, ``q`` has no
structural zeros, and ``Sigma`` carries 80 replicate columns with some shared
structure so that its off-diagonal terms are not negligible. Small enough that
every operator can be checked against its dense form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.data.variance import SDR_FACTOR


def make_problem(
    zone_tract=(0, 0, 0, 1, 1, 2, 2, 2),
    n_units: int = 150,
    tract_cells: int = 4,
    block_group_cells: int = 3,
    noise: float = 0.1,
    seed: int = 1,
) -> PMEDMInputs:
    rng = np.random.default_rng(seed)
    zone_tract = np.asarray(zone_tract)
    n_zones, n_tracts = zone_tract.size, int(zone_tract.max()) + 1
    X_T = sp.csr_matrix(rng.poisson(0.6, (n_units, tract_cells)).astype(float))
    X_B = sp.csr_matrix(rng.poisson(0.6, (n_units, block_group_cells)).astype(float))
    A_T = sp.csr_matrix(
        (np.ones(n_zones), (zone_tract, np.arange(n_zones))), shape=(n_tracts, n_zones)
    )
    A_B = sp.identity(n_zones, format="csr")
    weights = rng.uniform(10, 60, n_units)
    q = np.tile(weights, (n_zones, 1))
    q /= q.sum()
    N = weights.sum()
    truth = rng.dirichlet(np.ones(n_zones * n_units)).reshape(n_zones, n_units) * N
    Y_T = np.asarray(A_T @ (truth @ X_T)) * rng.uniform(1 - noise, 1 + noise, (n_tracts, tract_cells))
    Y_B = np.asarray(A_B @ (truth @ X_B)) * rng.uniform(1 - noise, 1 + noise, (n_zones, block_group_cells))
    m = Y_T.size + Y_B.size
    L = rng.normal(0, 1, (m, 80)) * rng.uniform(1, 5, (m, 1))
    L[:, :40] += rng.normal(0, 2, (1, 40))
    v = SDR_FACTOR * np.square(L).sum(axis=1) * 1.2 + 1.0
    inputs = PMEDMInputs(
        q=q, X_T=X_T, X_B=X_B, A_T=A_T, A_B=A_B, Y_T=Y_T, Y_B=Y_B,
        sigma_v=v, sigma_l=L, puma="0000000", n=n_units, N=N,
        units=pd.DataFrame({"unit": range(n_units)}),
        zones=pd.DataFrame({"zone": range(n_zones)}),
        tracts=pd.DataFrame({"tract": range(n_tracts)}),
        block_groups=pd.DataFrame({"zone": range(n_zones)}),
        tract_constraints=[f"T.{i}" for i in range(tract_cells)],
        bg_constraints=[f"B.{i}" for i in range(block_group_cells)],
    )
    inputs.validate()
    return inputs


def kronecker(inputs: PMEDMInputs) -> np.ndarray:
    """``X_tilde'`` materialised, ``(n_constraints, n_zones * n_units)``."""
    return sp.vstack(
        [sp.kron(inputs.X_T.T, inputs.A_T), sp.kron(inputs.X_B.T, inputs.A_B)]
    ).toarray()


def dense_hessian(inputs: PMEDMInputs, p: np.ndarray, sigma) -> np.ndarray:
    """The dual Hessian built from dense pieces, for checking."""
    xt = kronecker(inputs)
    pv = p.ravel(order="F")
    u = xt @ pv
    c = inputs.n / inputs.N**2
    return xt @ (pv[:, None] * xt.T) - np.outer(u, u) + c * sigma.to_dense()


def make_pathological_problem(
    zone_tract=(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
    n_units: int = 1500,
    tract_cells: int = 5,
    block_group_cells: int = 4,
    rare_cells: int = 3,
    carriers: int = 4,
    max_loading: int = 8,
    rare_mean: float = 2.0,
    noise: float = 0.1,
    seed: int = 1,
) -> PMEDMInputs:
    """A problem with the Knox pathology built in: rare categories on few units.

    On top of common cells like :func:`make_problem`'s, each of ``rare_cells``
    categories is carried by ``carriers`` units only, with loadings (members in
    the category) drawn from ``1 .. max_loading`` and the largest forced to
    ``max_loading``, and one extra "bundle" unit carries every rare category at
    ``max_loading // 2`` at once. Each rare category is constrained at both
    tract and block group, as at Knox.

    Their published values are small counts, not the truth's: Poisson with mean
    ``rare_mean`` per block group, with standard errors of 1.4-2.6 where the
    count is positive and the modelled zero-count variance (72, the Knox block
    group value) where it is zero; a tract's is the sum of its block groups',
    with their variances summed. So the fit puts little mass on the carriers,
    the curvature along their multipliers is small, and lowering a multiplier
    multiplies its carriers' weight by ``exp(loading x shift)`` -- flat on one
    side, a wall on the other. Rare rows carry no replicate structure.

    Rare columns come after the common ones in ``X_T`` and ``X_B``, named
    ``T.rare{j}`` and ``B.rare{j}``; ``units`` flags the carriers and the bundle.
    Whether a given draw actually shows the VB tail is for the diagnostics to
    say, not this docstring.
    """
    rng = np.random.default_rng(seed)
    zone_tract = np.asarray(zone_tract)
    n_zones, n_tracts = zone_tract.size, int(zone_tract.max()) + 1

    common_T = rng.poisson(0.6, (n_units, tract_cells)).astype(float)
    common_B = rng.poisson(0.6, (n_units, block_group_cells)).astype(float)
    rare = np.zeros((n_units, rare_cells))
    chosen = rng.choice(n_units - 1, size=(rare_cells, carriers), replace=False)
    for j in range(rare_cells):
        loadings = rng.integers(1, max_loading + 1, carriers)
        loadings[0] = max_loading
        rare[chosen[j], j] = loadings
    bundle = n_units - 1
    rare[bundle, :] = max(1, max_loading // 2)
    X_T = sp.csr_matrix(np.hstack([common_T, rare]))
    X_B = sp.csr_matrix(np.hstack([common_B, rare]))

    A_T = sp.csr_matrix(
        (np.ones(n_zones), (zone_tract, np.arange(n_zones))), shape=(n_tracts, n_zones)
    )
    A_B = sp.identity(n_zones, format="csr")
    weights = rng.uniform(10, 60, n_units)
    q = np.tile(weights, (n_zones, 1))
    q /= q.sum()
    N = weights.sum()

    truth = rng.dirichlet(np.ones(n_zones * n_units)).reshape(n_zones, n_units) * N
    Y_T = np.asarray(A_T @ (truth @ common_T)) * rng.uniform(1 - noise, 1 + noise, (n_tracts, tract_cells))
    Y_B = np.asarray(truth @ common_B) * rng.uniform(1 - noise, 1 + noise, (n_zones, block_group_cells))

    rare_B = rng.poisson(rare_mean, (n_zones, rare_cells)).astype(float)
    rare_B_var = np.where(rare_B > 0, rng.uniform(1.4, 2.6, rare_B.shape) ** 2, 72.0)
    rare_T = np.asarray(A_T @ rare_B)
    rare_T_var = np.asarray(A_T @ rare_B_var)
    Y_T = np.hstack([Y_T, rare_T])
    Y_B = np.hstack([Y_B, rare_B])

    # Stacked column-major: all of X_T's columns over tracts, then X_B's over zones.
    m_T, m_B = Y_T.size, Y_B.size
    m = m_T + m_B
    common_rows = np.concatenate([
        np.arange(tract_cells * n_tracts),
        m_T + np.arange(block_group_cells * n_zones),
    ])
    L = np.zeros((m, 80))
    L[common_rows] = rng.normal(0, 1, (common_rows.size, 80)) * rng.uniform(1, 5, (common_rows.size, 1))
    L[common_rows, :40] += rng.normal(0, 2, (1, 40))
    v = SDR_FACTOR * np.square(L).sum(axis=1) * 1.2 + 1.0
    v[tract_cells * n_tracts : m_T] = rare_T_var.ravel(order="F")
    v[m_T + block_group_cells * n_zones :] = rare_B_var.ravel(order="F")

    role = np.full(n_units, "", dtype=object)
    role[chosen.ravel()] = "carrier"
    role[bundle] = "bundle"
    inputs = PMEDMInputs(
        q=q, X_T=X_T, X_B=X_B, A_T=A_T, A_B=A_B, Y_T=Y_T, Y_B=Y_B,
        sigma_v=v, sigma_l=L, puma="0000001", n=n_units, N=N,
        units=pd.DataFrame({"unit": range(n_units), "role": role}),
        zones=pd.DataFrame({"zone": range(n_zones)}),
        tracts=pd.DataFrame({"tract": range(n_tracts)}),
        block_groups=pd.DataFrame({"zone": range(n_zones)}),
        tract_constraints=[f"T.{i}" for i in range(tract_cells)] + [f"T.rare{j}" for j in range(rare_cells)],
        bg_constraints=[f"B.{i}" for i in range(block_group_cells)] + [f"B.rare{j}" for j in range(rare_cells)],
    )
    inputs.validate()
    return inputs
