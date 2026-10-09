"""Do VB's walls lie in a low-dimensional subspace? The test before a wall-shaped flow.

A flow that adds skew only along a few directions can remove the walls only if
the walls lie along a few directions. This measures that, from a fitted VB
family and nothing else:

1. Draw ``--draws`` from the family (``--family gaussian`` by default).
2. In each draw, take the ``--top`` largest cells ``p_ib`` (record ``i``, block
   group ``b``) that hold at least ``--min-share`` of N.
3. For each, the gradient of ``log p_ib`` in the multipliers the data see,
   ``-a_ib + u``: the cell's loadings (its tract's and block group's rows) less
   their ``p``-weighted mean ``u = X'p``. Mapped to the solver's coordinates
   ``xi`` by the hierarchy's adjoint.
4. Whitened by the family's Gaussian part: with ``xi = mu + G'^{-1} eps`` the
   gradient in ``eps`` is ``G^{-1} grad_xi``. Each is scaled to unit length, so
   the analysis is of directions, and weighted by ``min(1, p_ib / --wall)``:
   cells nearer a wall count more.
5. A weighted SVD of the directions from the even-numbered draws.
6. Held out: for the odd-numbered draws' directions, the share of each one's
   squared length inside the top ``k`` singular vectors, ``k`` in ``--ks``.
   Walls (cells over ``--wall`` of N) and near-walls are reported apart. Fitting
   and judging on the same directions would flatter the subspace.

It also reports how many distinct records and block groups the walls involve,
and, for the leading singular vectors, the constraint rows they load on most
(mapped back to the multipliers the data see), to show what each direction is.

Writes ``<out>/wall_subspace.txt`` and ``wall_subspace.csv`` (one row per fit
and ``k``)::

    sbatch experiments/run_python.sbatch wall_subspace.py \\
        --runs $E/4701501 $E/4701502 $E/4701503 $E/4701504 --out $E/wall_subspace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

sys.path.insert(0, str(Path(__file__).resolve().parent))

from laplace_diagnostic import load_vb  # noqa: E402
from pmedm_vb.assemble.hierarchy import Hierarchy  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from ratio_diagnostic import Ratios, fits_in  # noqa: E402

#: Columns of the gradient matrix whitened at a time.
CHUNK = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="fit folders holding vb_<family>/ (an experiment's <puma>/ folders)")
    parser.add_argument("--family", default="gaussian")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--top", type=int, default=3, help="largest cells taken per draw")
    parser.add_argument("--wall", type=float, default=0.01, help="a wall is a cell over this share of N")
    parser.add_argument("--min-share", type=float, default=0.001,
                        help="cells under this share of N are left out")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 25, 50, 100])
    parser.add_argument("--interpret", type=int, default=5, help="leading directions described")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def loadings(ratios: Ratios, zone: int, unit: int) -> tuple[np.ndarray, np.ndarray]:
    """The stacked rows cell ``(zone, unit)`` loads on, and its loadings there."""
    inputs = ratios.inputs
    n_tracts, n_zones = inputs.Y_T.shape[0], inputs.Y_B.shape[0]
    tract = int(ratios.zone_tract[zone])
    row_t, row_b = ratios.x_t.getrow(unit), ratios.x_b.getrow(unit)
    rows = np.concatenate([row_t.indices * n_tracts + tract,
                           inputs.Y_T.size + row_b.indices * n_zones + zone])
    return rows, np.concatenate([row_t.data, row_b.data])


def whiten_gradients(q, g: np.ndarray) -> np.ndarray:
    """``G^{-1} g`` for the columns of ``g``, ``G = blockdiag(L) (I + W V')``: a
    gradient in ``xi`` as a gradient in the whitened ``eps``, ``xi = mu + G'^{-1} eps``."""
    y = np.empty_like(g)
    for rows, block in zip(q.rows, q.blocks):
        y[rows] = solve_triangular(block, g[rows], lower=True)
    r = q.W.shape[1]
    small = np.eye(r) + q.V.T @ q.W
    return y - q.W @ np.linalg.solve(small, q.V.T @ y)


def cell_directions(ratios: Ratios, h, q, xi: np.ndarray, args) -> tuple[np.ndarray, pd.DataFrame]:
    """Whitened unit directions of each draw's largest cells, and the cells."""
    lam = xi if h.is_trivial else h.lambda_data(xi)
    support = ratios.support
    grads, cells = [], []
    for d in range(lam.shape[1]):
        _, log_p, _ = ratios.log_ratio(lam[:, d])
        p = np.where(support, np.exp(log_p), 0.0)
        flat = p.ravel()
        top = np.argpartition(flat, -args.top)[-args.top:]
        top = top[flat[top] >= args.min_share]
        if not top.size:
            continue
        u = ratios.op.forward(p)
        for cell in top:
            zone, unit = np.unravel_index(cell, p.shape)
            rows, x = loadings(ratios, zone, unit)
            g = u.copy()
            np.subtract.at(g, rows, x)
            grads.append(g)
            cells.append(dict(draw=d, zone=int(zone), unit=int(unit), share=float(flat[cell])))
    if not grads:
        return np.zeros((0, q.size)), pd.DataFrame(cells)
    out = np.empty((len(grads), q.size), dtype=np.float32)
    for start in range(0, len(grads), CHUNK):
        g = np.stack(grads[start:start + CHUNK], axis=1)                   # (m_data, chunk)
        g_xi = g if h.is_trivial else h.lambda_data_T(g)
        z = whiten_gradients(q, g_xi)
        out[start:start + CHUNK] = (z / np.linalg.norm(z, axis=0, keepdims=True)).T
    return out, pd.DataFrame(cells)


def describe(ratios: Ratios, h, q, v: np.ndarray, top: int = 6) -> str:
    """The constraint rows a whitened direction ``v`` moves most, in the data's layout."""
    xi = q.gaussian_part(v[:, None])[:, 0]
    lam = xi if h.is_trivial else h.lambda_data(xi)
    lam = lam / np.abs(lam).max()
    order = np.argsort(-np.abs(lam))[:top]
    return "; ".join(f"{ratios.names[k]}#{area_of(ratios, k)} {lam[k]:+.2f}" for k in order)


def area_of(ratios: Ratios, row: int) -> int:
    """The tract or block group index of a stacked constraint row."""
    inputs = ratios.inputs
    if row < inputs.Y_T.size:
        return int(row % inputs.Y_T.shape[0])
    return int((row - inputs.Y_T.size) % inputs.Y_B.shape[0])


def analyse(ratios: Ratios, h, q, key: dict, args, rng) -> tuple[list[dict], list[str]]:
    xi = q.sample(rng, args.draws)
    Z, cells = cell_directions(ratios, h, q, xi, args)
    head = (f"\n== {key['puma']} alpha {key['alpha']:g} {key['level']} ({args.family}): "
            f"{args.draws} draws, {len(cells)} cells over {args.min_share:g} of N")
    if len(cells) < 4:
        return [], [head, "   too few cells"]
    share = cells.share.to_numpy()
    weight = np.minimum(1.0, share / args.wall)
    fit = (cells.draw % 2 == 0).to_numpy()
    test = ~fit
    _, s, vt = np.linalg.svd(Z[fit] * np.sqrt(weight[fit])[:, None], full_matrices=False)
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)
    walls = cells[share > args.wall]
    near = (share > 0.1 * args.wall) & (share <= args.wall)
    lines = [head,
             f"   walls (> {args.wall:g} of N): {len(walls)} cells in "
             f"{walls.draw.nunique()} draws; {walls.unit.nunique()} records, "
             f"{walls.zone.nunique()} block groups, "
             f"{len(walls.groupby(['unit', 'zone']))} record x block group pairs",
             f"   fitted on {fit.sum()} cells (even draws), judged on {test.sum()} (odd draws)",
             "      k   in-sample energy   held-out capture: walls median / p10 / share >= 0.8"
             "   near-walls median   all (weighted)"]
    rows = []
    for k in args.ks:
        if k > vt.shape[0]:
            continue
        capture = np.square(Z[test] @ vt[:k].T).sum(axis=1)
        w_test, s_test = weight[test], share[test]
        is_wall, is_near = s_test > args.wall, near[test]
        row = dict(**key, family=args.family, k=k, energy=float(energy[k - 1]),
                   wall_cells_test=int(is_wall.sum()),
                   wall_median=float(np.median(capture[is_wall])) if is_wall.any() else np.nan,
                   wall_p10=float(np.quantile(capture[is_wall], 0.1)) if is_wall.any() else np.nan,
                   wall_share_ge_0_8=float((capture[is_wall] >= 0.8).mean()) if is_wall.any() else np.nan,
                   near_median=float(np.median(capture[is_near])) if is_near.any() else np.nan,
                   weighted=float(np.sum(w_test * capture) / np.sum(w_test)))
        rows.append(row)
        lines.append(f"   {k:4d}   {row['energy']:16.3f}   {row['wall_median']:13.3f} / "
                     f"{row['wall_p10']:.3f} / {row['wall_share_ge_0_8']:.3f}"
                     f"   {row['near_median']:17.3f}   {row['weighted']:.3f}")
    lines.append(f"   leading directions (rows moved most, scaled to the largest; #area):")
    for j in range(min(args.interpret, vt.shape[0])):
        lines.append(f"     {j + 1} ({s[j] ** 2 / np.sum(s ** 2):.1%}): "
                     f"{describe(ratios, h, q, vt[j].astype(float))}")
    return rows, lines


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, lines = [], [
        f"Do the largest cells' log-share gradients lie in a few whitened directions? "
        f"{args.family} VB, {args.draws} draws per fit, top {args.top} cells per draw over "
        f"{args.min_share:g} of N, weights min(1, share / {args.wall:g}).",
        "capture = share of a held-out direction's squared length inside the top k singular vectors."]
    cache: dict[str, Ratios] = {}
    for run in args.runs:
        for path, match in fits_in(run, args.family):
            puma, alpha = match.group("puma"), float(match.group("alpha"))
            level, taper = match.group("level") or "none", match.group("taper")
            if puma not in cache:
                cache.clear()
                cache[puma] = Ratios(PMEDMInputs.load(processed_dir() / "inputs" / args.area / puma))
            ratios = cache[puma]
            h = Hierarchy.build(ratios.inputs, level, alpha=alpha,
                                taper=None if taper == "none" else taper, variance_floor="zero")
            q, _, _ = load_vb(path)
            fit_rows, fit_lines = analyse(ratios, h, q, dict(puma=puma, alpha=alpha, level=level),
                                          args, rng)
            rows += fit_rows
            lines += fit_lines
            print("\n".join(fit_lines), flush=True)
    pd.DataFrame(rows).to_csv(args.out / "wall_subspace.csv", index=False)
    (args.out / "wall_subspace.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
