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
