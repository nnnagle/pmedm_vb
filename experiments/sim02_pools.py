"""Split a statewide support's borrowed records into two pools (``experiments/sim/sim02.json``).

From the real statewide inputs of the simulation's PUMA
(``sim02.json``'s ``statewide`` experiment, e.g. ``tennessee-5t-2024-5yr-state-e0.01/4701502``,
and its ``heldout/``), the records from outside the PUMA (``unit_members``
with ``in_puma`` false) are split at random, record by record, into two
halves, pools **A** and **B** (``pools.seed``). Each pool gives a derived
statewide problem: the PUMA's own records plus that pool's, written as the
inputs of the pool's area (``pools.areas``), ``<area>-<year>-<span>yr-state-e<E>/<puma>``,
where run_experiment.py finds them through ``inputs_name``.

**From the saved inputs alone**, no PUMS or census file:

- each record's own design weight ``d`` is recovered from its support weight
  (:func:`sim02_common.record_design_weights`) and kept as the
  ``design_weight`` column of the derived ``unit_members``;
- the support weights are recomputed over the PUMA's records and the pool's
  with :func:`pmedm_vb.assemble.design.support_weights` at the same epsilon,
  so the PUMA's ``N`` and ``n`` are unchanged and ``sum(d)`` is now the PUMA's
  plus the pool's;
- rows are re-merged: records in one merged row share their row of ``X``, so
  a derived row is a source row restricted to its records in the selection
  (a row holding records of both pools appears in both, each with its own
  records' weights); source rows with no selected record of positive weight
  are dropped, and the rest keep their source order. Since the source rows
  are distinct, so are the derived ones: nothing needs merging again;
- ``units``, ``support`` and ``support_columns`` are rebuilt as
  :func:`pmedm_vb.assemble.build.build_puma` builds them, ``q`` from the new
  row weights; the constraints, Sigma, geography and table trees are the
  source's;
- **held-out X.** A source row's held-out counts are its records' counts
  averaged by prior weight over *all* its records (own and both pools), and
  the per-record counts are not in the saved inputs, so a derived row keeps its
  source row's held-out counts. That is exact for rows whose records agree on
  the held-out columns (every row of one record among them), and otherwise
  the average over the row's statewide records, with the source's weights,
  rather than over the selection's records with the recomputed ones. The
  simulation needs only that truth and fits read one unit's held-out counts
  alike, which they do.

Each derived problem is checked with ``PMEDMInputs.validate()`` and saved with
``manifest.json``, ``heldout/`` and the table trees; ``support`` records the
pool, the seed and the source. Refuses to replace derived inputs that exist
(the pool A fit, tn04, depends on them) unless ``--overwrite``::

    sbatch --account=isaac-utk0533 --partition=campus --qos=campus \\
        --output=$PMEDM_VB_RUNS/slurm-%j.log \\
        experiments/run_python.sbatch sim02_pools.py --sim experiments/sim/sim02.json
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim02_common import Sim02, record_design_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/sim02.json")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace derived inputs that exist (default: refuse)")
    return parser.parse_args()


def split(members: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    """Per pool, the positions in ``members`` of its out-of-PUMA records: a
    random half each (A takes the odd one out)."""
    outside = np.flatnonzero(~members["in_puma"].to_numpy(dtype=bool))
    order = np.random.default_rng(seed).permutation(outside.size)
    half = (outside.size + 1) // 2
    return {"A": np.sort(outside[order[:half]]), "B": np.sort(outside[order[half:]])}


def pool_problem(source, heldout, d: np.ndarray, records: np.ndarray, epsilon: float,
                 extra: dict):
    """The source restricted to the PUMA's own records and ``records`` (positions in
    its ``unit_members``), support weights recomputed, and its held-out tables."""
    from pmedm_vb.assemble.build import support_report, unit_frame
    from pmedm_vb.assemble.design import design_weights, support_weights
    from pmedm_vb.assemble.heldout import HeldOut

    members = source.unit_members
    own = np.flatnonzero(members["in_puma"].to_numpy(dtype=bool))
    chosen = np.sort(np.concatenate([own, records]))
    selected = members.iloc[chosen].reset_index(drop=True)
    d_sel = d[chosen]
    in_puma = selected["in_puma"].to_numpy(dtype=bool)
    weight = support_weights(pd.Series(d_sel), in_puma, epsilon)
    old_row = selected["row"].to_numpy()
    live = old_row >= 0
    keep = np.unique(old_row[live & (weight > 0)])
    renumber = np.full(source.n_units, -1)
    renumber[keep] = np.arange(keep.size)
    row = np.where(live, renumber[np.where(live, old_row, 0)], -1)
    new = pd.DataFrame({"SERIALNO": selected["SERIALNO"].to_numpy(), "row": row,
                        "weight": weight, "in_puma": in_puma, "design_weight": d_sel})
    gq = source.units["is_group_quarters"].to_numpy(dtype=bool)
    is_gq = np.where(live, gq[np.where(live, old_row, 0)], False)
    X_T, X_B = sp.csr_matrix(source.X_T)[keep], sp.csr_matrix(source.X_B)[keep]
    units = unit_frame(new, is_gq, keep.size)
    support, columns = support_report(new, d_sel, X_T, X_B, list(source.tract_constraints),
                                      list(source.bg_constraints), epsilon)
    out = dataclasses.replace(
        source, q=design_weights(units["weight"], n_zones=source.n_zones), X_T=X_T, X_B=X_B,
        units=units, unit_members=new, support={**support, **extra}, support_columns=columns)
    out.validate()
    held = HeldOut(X={level: sp.csr_matrix(X)[keep] for level, X in heldout.X.items()},
                   Y=heldout.Y, v=heldout.v, names=heldout.names)
    return out, held


def main() -> None:
    args = parse_args()
    sim = Sim02(args.sim)
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    path = sim.statewide_inputs()
    source = PMEDMInputs.load(path)
    if source.unit_members is None:
        raise SystemExit(f"{path}: not a statewide support")
    if not HeldOut.exists(path):
        raise SystemExit(f"{path}/heldout: missing (run_map.py heldout --epsilon "
                         f"{sim.epsilon:g})")
    epsilon = float(source.support["epsilon"])
    if epsilon != sim.epsilon:
        raise SystemExit(f"{path}: epsilon {epsilon:g}, the simulation's is {sim.epsilon:g}")
    targets = {pool: sim.pool_inputs(pool) for pool in ("A", "B")}
    existing = [p for p in targets.values() if (p / "manifest.json").exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing[0]} exists; --overwrite to replace it")
    heldout = HeldOut.load(path)
    d = record_design_weights(source)
    seed = int(sim.settings["pools"]["seed"])
    pools = split(source.unit_members, seed)
    own = int(source.unit_members["in_puma"].sum())
    print(f"{path}: {own:,} own records, {sum(p.size for p in pools.values()):,} from elsewhere, "
          f"{source.n_units:,} rows; N {source.N:,.0f}; epsilon {epsilon:g}", flush=True)
    for pool, records in pools.items():
        extra = {"pool": pool, "pool_seed": seed, "pool_records": int(records.size),
                 "derived_from": str(path)}
        out, held = pool_problem(source, heldout, d, records, epsilon, extra)
        out.save(targets[pool])
        held.save(targets[pool])
        s = out.support
        print(f"pool {pool}: {records.size:,} records (design weight {d[records].sum():,.0f}); "
              f"{out.n_units:,} rows, {s['rows_added']:,} without a PUMA record "
              f"({100 * s['prior_share_added']:.2f}% of the prior); n {out.n:,}, N {out.N:,.0f}"
              f"\n  wrote {targets[pool]}", flush=True)


if __name__ == "__main__":
    main()
