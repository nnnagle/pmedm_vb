"""Row bookkeeping on PMEDMInputs, and the aggregation operators."""

import numpy as np
import pytest
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


def with_zero_cells(problem):
    """A copy with planted zero cells whose modelled variance depends on the area."""
    import dataclasses

    inputs = dataclasses.replace(problem, Y_T=problem.Y_T.copy(), Y_B=problem.Y_B.copy(),
                                 sigma_v=problem.sigma_v.copy(), sigma_l=problem.sigma_l.copy())
    n_t, n_b = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    tract_v0 = 100.0 + 10 * np.arange(n_t)
    bg_v0 = 50.0 + 5 * np.arange(n_b)
    planted = []
    # One zero cell in category 0 of every tract but the last, and of every other block group.
    for t in range(n_t - 1):
        planted.append((t, tract_v0[t]))
    for b in range(0, n_b, 2):
        planted.append((inputs.Y_T.size + b, bg_v0[b]))
    for row, v0 in planted:
        inputs.sigma_l[row] = 0.0
        inputs.sigma_v[row] = v0
    y = inputs.targets()
    y[[row for row, _ in planted]] = 0.0
    inputs.Y_T[:] = y[: inputs.Y_T.size].reshape(inputs.Y_T.shape, order="F")
    inputs.Y_B[:] = y[inputs.Y_T.size :].reshape(inputs.Y_B.shape, order="F")
    return inputs, tract_v0, bg_v0


def test_zero_cell_variances_are_read_per_area(problem):
    inputs, tract_v0, bg_v0 = with_zero_cells(problem)
    v0 = inputs.zero_cell_variances()
    split = inputs.Y_T.size
    tract_rows = v0[:split].reshape(inputs.Y_T.shape, order="F")
    bg_rows = v0[split:].reshape(inputs.Y_B.shape, order="F")
    # Areas with a zero cell get their own value on every row; the rest the level median.
    np.testing.assert_allclose(tract_rows[:-1, :], tract_v0[:-1, None].repeat(tract_rows.shape[1], 1))
    np.testing.assert_allclose(tract_rows[-1, :], np.median(tract_v0[:-1]))
    np.testing.assert_allclose(bg_rows[::2, :], bg_v0[::2, None].repeat(bg_rows.shape[1], 1))
    np.testing.assert_allclose(bg_rows[1::2, :], np.median(bg_v0[::2]))


def test_the_floor_only_raises_variances(problem):
    inputs, _, _ = with_zero_cells(problem)
    for floor in ("zero", 400.0):
        floored = inputs.floored_variances(floor)
        base = inputs.zero_cell_variances() if floor == "zero" else 400.0
        np.testing.assert_allclose(floored, np.maximum(inputs.sigma_v, base))
        assert (floored >= inputs.sigma_v).all() and (floored > inputs.sigma_v).any()
        sigma = inputs.sigma(0.3, "tract", floor)
        np.testing.assert_allclose(sigma.diagonal(), floored, rtol=1e-12)
    np.testing.assert_array_equal(inputs.floored_variances(None), inputs.sigma_v)


def test_a_zero_floor_needs_zero_cells(problem):
    with pytest.raises(ValueError, match="published zero"):
        problem.zero_cell_variances()


def test_solvers_carry_the_floor(problem):
    from pmedm_vb.solvers.map_dual import laplace_precision, solve_map

    inputs, _, _ = with_zero_cells(problem)
    result = solve_map(inputs, alpha=0.3, variance_floor="zero")
    unfloored = solve_map(inputs, alpha=0.3)
    assert result.converged and result.variance_floor == "zero"
    assert result.objective != pytest.approx(unfloored.objective, rel=1e-6)
    x = np.random.default_rng(0).normal(size=inputs.n_constraints)
    assert not np.allclose(laplace_precision(inputs, result).matvec(x),
                           laplace_precision(inputs, unfloored).matvec(x))
