"""Build the constraint targets ``Y`` and their covariance ``Sigma``.

``Sigma`` takes one of two forms, and which one is available decides the
summary-table access route (see :mod:`pmedm_vb.data.summary`):

* **Diagonal**, derived from published margins of error. The derivation in
  ``pmedm_derivation.md`` assumes exactly this, noting that independence
  "isn't quite right, but that's a fight for another day".
* **Full**, derived from variance replicates. Available only for the subset of
  tables covered by :mod:`pmedm_vb.data.variance`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmedm_vb.config import StudyArea


def stack_targets(
    estimates: pd.DataFrame,
    areas: pd.DataFrame,
    constraints: list[str],
) -> np.ndarray:
    """Arrange published estimates into the ``(n_areas, n_constraints)`` target matrix."""
    raise NotImplementedError


def sigma_from_moe(moe: np.ndarray, *, confidence: float = 0.90) -> np.ndarray:
    """Build a diagonal ``Sigma`` from published margins of error.

    Returns a one-dimensional array of variances over the stacked constraint
    vector, column-major within each of the tract and block group blocks. The
    ACS publishes margins of error at 90 percent confidence; the conversion
    factor from margin of error to standard error should be taken from Census
    documentation rather than assumed.
    """
    raise NotImplementedError


def sigma_from_replicates(replicates: np.ndarray) -> np.ndarray:
    """Build a full ``Sigma`` from variance replicate estimates.

    Delegates the variance formula to
    :func:`pmedm_vb.data.variance.covariance_from_replicates`, then arranges
    the result to match the stacked constraint ordering.
    """
    raise NotImplementedError
