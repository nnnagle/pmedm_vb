"""Build the individual attribute matrices and the aggregation operators.

``X_T`` and ``X_B`` translate each PUMS record into the constraint cells it
contributes to; ``A_T`` and ``A_B`` translate zones into the areas the
constraints are published for. Together they are the four factors that
:class:`~pmedm_vb.assemble.inputs.PMEDMInputs` stores in place of the Kronecker
product.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.config import StudyArea


def build_attribute_matrix(
    pums: pd.DataFrame,
    constraints: list[str],
) -> tuple[sp.csr_matrix, list[str]]:
    """Expand PUMS records into indicator columns, one per constraint cell.

    The column definitions have to agree cell-for-cell with the summary tables
    they will be matched against -- same age breaks, same categories, same
    universe. That correspondence is the fiddliest part of setting up a PMEDM
    run and the usual source of silent misfit.

    Returns
    -------
    tuple
        The sparse matrix ``(n_individuals, n_cells)`` and its column labels.
    """
    raise NotImplementedError


def build_aggregation(
    zones: pd.DataFrame,
    areas: pd.DataFrame,
) -> sp.csr_matrix:
    """Build the ``(n_areas, n_zones)`` operator summing zones into areas.

    Census GEOIDs nest by construction, so membership is a string prefix rather
    than a spatial join. Degenerate when zones are the areas, in which case the
    result is the identity.
    """
    raise NotImplementedError


def design_weights(
    pums: pd.DataFrame,
    zones: pd.DataFrame,
    crosswalk: pd.DataFrame,
) -> np.ndarray:
    """Build ``q``, the ``(n_zones, n_individuals)`` design probabilities.

    Entries are zero where a record's PUMA does not contain the zone, which is
    what confines each record's weight to its own PUMA. The result is
    normalised to sum to one over all (zone, individual) pairs, matching the
    constraint ``sum_ij p_ij = 1`` in the derivation.
    """
    raise NotImplementedError
