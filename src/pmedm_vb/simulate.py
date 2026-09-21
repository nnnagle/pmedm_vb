"""Draw synthetic populations from a fitted solution.

Both solvers produce a distribution over individuals within each zone; this
turns that into integer counts of synthetic people. The distinction the paper
turns on is what uncertainty is propagated: the MAP path can only resample
given one weight matrix, whereas the VB path can draw a weight matrix from the
posterior first and then resample, carrying parameter uncertainty into the
synthetic population.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmedm_vb.assemble.inputs import PMEDMInputs


def simulate_population(
    inputs: PMEDMInputs,
    W: np.ndarray,
    *,
    n_draws: int = 1,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw synthetic individuals zone by zone from the weights ``W``.

    Returns
    -------
    pandas.DataFrame
        One row per synthetic person, carrying the draw index, the zone GEOID
        and the PUMS record identifiers it was cloned from.
    """
    raise NotImplementedError


def expected_counts(inputs: PMEDMInputs, W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Aggregate weights to expected counts for a set of attribute columns.

    Useful without drawing anything: this is what gets compared against held-out
    published estimates when scoring a fit.
    """
    raise NotImplementedError
