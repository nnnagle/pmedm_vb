"""Figure A's data: what single draws of the population look like in a few block groups.

For one experiment, PUMA and alpha, and the methods in ``METHODS``, draws the
weights ``W`` exactly as the scoring does (``compare_methods.method_draws``,
with the arguments ``score_experiment.cell_calls`` gives it) and, in four
block groups, keeps each draw's household shares: each household record's
share of the block group's households, as in compare_methods.py's n_eff (block
groups with more than ``--hh-min`` published occupied households, B25003's
owner and renter cells).

The four block groups (``case``), each taken once, in this order:

- ``ipf``: the IPF fit's smallest n_eff;
- ``hmc``: the smallest median n_eff over ``--select-draws`` HMC reference draws;
- ``vb``: the smallest median n_eff over ``--select-draws`` VB Gaussian draws;
- ``typical``: the block group whose median n_eff over the HMC reference draws
  is the median over the eligible block groups.

If one is already taken, the next in line is used. Each case shows the same
``--draws`` draws per method (IPF: its one fit), chosen at random among the
selection draws. Writes one row per (case, method, draw, record) for the
records over ``--min-share`` of the block group's households, and one row
(record -1) for all the others together, with each draw's n_eff, and
``highlight``: the rank (1 to ``--top``) of the case's records with the largest
mean share over the shown draws, averaged over the methods, empty for the rest::

    PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
    $CONDA_PREFIX/bin/python experiments/draw_examples.py \\
        --root $RUNS/paper_v3 --experiment exp01_baseline --puma 4701504 --alpha 0.01 \\
        --out $RUNS/paper_v3/tables_compare/draw_examples.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_methods as cm  # noqa: E402
from score_experiment import cell_calls  # noqa: E402

from pmedm_vb.assemble.allcells import table_leaves  # noqa: E402
from pmedm_vb.assemble.hierarchy import Hierarchy  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator  # noqa: E402

METHODS = ("ipf", "ipf_multinomial", "vb_gaussian", "hmc_ref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiment", default="exp01_baseline")
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--select-draws", type=int, default=400,
                        help="VB and HMC draws that choose the block groups")
    parser.add_argument("--draws", type=int, default=12, help="draws shown per method")
    parser.add_argument("--top", type=int, default=5, help="records highlighted per case")
    parser.add_argument("--min-share", type=float, default=0.002,
                        help="records under this share are written together as record -1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def method_args(spec: dict, exp: Path, puma: str, alpha: float) -> dict[str, argparse.Namespace]:
    """compare_methods.py's parsed arguments for each of ``METHODS``, as scored."""
    calls, missing = cell_calls(spec, exp, puma, alpha, Path("unused"), 1)
    found = {}
    argv = sys.argv
    try:
        for call in calls:
            tokens = call.split()
            method = tokens[tokens.index("--method") + 1]
            if method in METHODS:
                sys.argv = ["compare_methods.py", *tokens]
                found[method] = cm.parse_args()
    finally:
        sys.argv = argv
    absent = [m for m in METHODS if m not in found]
    if absent:
        raise SystemExit(f"no fits for {', '.join(absent)}; missing: {'; '.join(missing)}")
    return found


def weights(source: dict, inputs: PMEDMInputs):
    """Each draw's ``W``, ``(zones, units)``, as compare_methods.run_draws scores it."""
    if "W_draws" in source:
        yield from source["W_draws"]
    elif "W" in source:
        yield np.asarray(source["W"], dtype=float)
    else:
        op = ConstraintOperator(inputs)
        for d in range(source["lam"].shape[1]):
            yield inputs.N * np.exp(cm.log_p_of(inputs, op, source["lam"][:, d]))


def household_shares(W: np.ndarray, zones: np.ndarray, households: np.ndarray) -> np.ndarray:
    W_hh = W[np.ix_(zones, households)]
    return W_hh / W_hh.sum(axis=1, keepdims=True)


def n_eff(s: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(s).sum(axis=-1)


def pick_cases(neff: dict[str, np.ndarray]) -> dict[str, int]:
    """The cases' positions among the eligible block groups (see the module docstring)."""
    taken: list[int] = []

    def first_free(order) -> int:
        position = next(int(k) for k in order if int(k) not in taken)
        taken.append(position)
        return position

    hmc = np.median(neff["hmc_ref"], axis=0)
    vb = np.median(neff["vb_gaussian"], axis=0)
    cases = {"ipf": first_free(np.argsort(neff["ipf"][0], kind="stable"))}
    cases["hmc"] = first_free(np.argsort(hmc, kind="stable"))
    cases["vb"] = first_free(np.argsort(vb, kind="stable"))
    cases["typical"] = first_free(np.argsort(np.abs(hmc - np.median(hmc)), kind="stable"))
    return {case: cases[case] for case in ("typical", "hmc", "ipf", "vb")}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    exp = args.root / args.experiment
    spec = json.loads((exp / "experiment.json").read_text())
    margs = method_args(spec, exp, args.puma, args.alpha)
    first = margs["hmc_ref"]
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / first.area / args.puma)
    hierarchy = Hierarchy.build(inputs, first.hierarchy, alpha=args.alpha,
                                taper=cm.taper_spec(first.taper),
                                variance_floor=cm.floor_spec(first.variance_floor))
    households = np.flatnonzero(~inputs.units["is_group_quarters"].to_numpy(bool))
    published = inputs.Y_B[:, table_leaves(inputs.bg_constraints, "B25003")].sum(axis=1)
    zones = np.flatnonzero(published > args.hh_min)
    if zones.size < 4:
        raise SystemExit(f"only {zones.size} block groups over {args.hh_min:g} households")

    # Every method's household shares in every eligible block group: all the
    # selection draws' n_eff, and the shown draws' shares.
    neff, shares = {}, {}
    for method in METHODS:
        margs[method].draws = {"ipf": 1, "ipf_multinomial": args.draws}.get(method,
                                                                         args.select_draws)
        source, _ = cm.method_draws(margs[method], inputs, rng, hierarchy)
        n = source["lam"].shape[1] if "lam" in source else margs[method].draws
        shown = set(rng.choice(n, min(args.draws, n), replace=False).tolist())
        values, kept = [], []
        for d, W in enumerate(weights(source, inputs)):
            s = household_shares(W, zones, households)
            values.append(n_eff(s))
            if d in shown:
                kept.append(s)
        neff[method], shares[method] = np.array(values), np.array(kept)
        print(f"{method}: {len(values)} draws, {len(kept)} kept", flush=True)

    rows = []
    for case, position in pick_cases(neff).items():
        zone = int(zones[position])
        geoid = str(inputs.zones.iloc[zone].iloc[0])
        by_method = {m: shares[m][:, position, :] for m in METHODS}
        pooled = np.mean([s.mean(axis=0) for s in by_method.values()], axis=0)
        highlight = {int(j): rank + 1 for rank, j in enumerate(np.argsort(-pooled)[:args.top])}
        print(f"{case}: block group {geoid} (row {zone}), {published[zone]:,.0f} households; "
              f"n_eff IPF {neff['ipf'][0, position]:.2f}, HMC reference median "
              f"{np.median(neff['hmc_ref'][:, position]):.2f}, VB Gaussian median "
              f"{np.median(neff['vb_gaussian'][:, position]):.2f}")
        base = dict(case=case, zone=zone, block_group=geoid, households=float(published[zone]))
        for method, s in by_method.items():
            for d, draw in enumerate(s):
                info = dict(base, method=method, draw=d + 1, n_eff=float(n_eff(draw)))
                big = np.flatnonzero(draw >= args.min_share)
                for j in big:
                    rows.append(dict(info, record=int(households[j]), share=float(draw[j]),
                                     highlight=highlight.get(int(j))))
                rows.append(dict(info, record=-1, share=float(1.0 - draw[big].sum()),
                                 highlight=None))
    frame = pd.DataFrame(rows)
    frame["highlight"] = frame["highlight"].astype("Int64")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame):,} rows to {args.out}")


if __name__ == "__main__":
    main()
