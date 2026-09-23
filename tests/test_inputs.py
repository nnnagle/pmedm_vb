"""Row bookkeeping on PMEDMInputs, and the aggregation operators."""

import numpy as np
import scipy.sparse as sp

from pmedm_vb.assemble.design import build_aggregation


def test_constraint_tracts_follows_column_major_stacking(problem):
    tracts = problem.constraint_tracts()
    n_t, n_b = problem.A_T.shape
    tract_part = tracts[: problem.Y_T.size].reshape(problem.Y_T.shape, order="F")
    bg_part = tracts[problem.Y_T.size :].reshape(problem.Y_B.shape, order="F")
    np.testing.assert_array_equal(tract_part, np.arange(n_t)[:, None].repeat(problem.Y_T.shape[1], 1))
    np.testing.assert_array_equal(bg_part[:, 0], problem.zone_tracts())
    assert (bg_part == bg_part[:, [0]]).all()


def test_targets_are_vec_column_major(problem):
    np.testing.assert_array_equal(
        problem.targets(),
        np.concatenate([problem.Y_T.ravel(order="F"), problem.Y_B.ravel(order="F")]),
    )


BLOCK_GROUPS = ["470930001001", "470930001002", "470930002001", "470930003001", "470930003002"]
TRACTS = ["47093000100", "47093000200", "47093000300"]


def test_aggregation_onto_the_zones_themselves_is_the_identity():
    # Regression: omitting membership used to map a block group to its tract.
    a = build_aggregation(BLOCK_GROUPS, BLOCK_GROUPS)
    assert (a != sp.identity(len(BLOCK_GROUPS))).nnz == 0


def test_aggregation_onto_tracts_nests_by_prefix():
    np.testing.assert_array_equal(
        build_aggregation(BLOCK_GROUPS, TRACTS).toarray(),
        [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 1]],
    )
