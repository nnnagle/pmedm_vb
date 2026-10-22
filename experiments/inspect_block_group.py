"""Why is one block group's population concentrated on a few household records?

For one experiment, PUMA and alpha, finds the block group with the smallest
effective number of household records (n_eff = 1 / sum s^2 over the records'
shares s of its households, as compare_methods.py; block groups with more than
``--hh-min`` published occupied households, B25003's owner and renter cells) in
the IPF fit, or takes ``--zone``, and reports, for the IPF fit and for draws of
the HMC reference:

- the block group: its GEOID, tract, published households and population, its
  n_eff in the IPF fit, across the HMC draws, and under the base weights alone
  (each record's design weight spread evenly, before any constraint), and how
  many household records may be placed there;
- its published leaf cells, block group and tract, with SE: the targets the
  allocation has to meet, the zeros and small counts first;
- the records with the largest shares, in the IPF fit and on average over the
  HMC draws: each share, how often the record is the block group's largest in
  the HMC draws, its design weight and the published categories it contributes
  to (its row of X, the people or units it adds to each).

With ``--least-diverse N`` it reports instead on the N HMC reference draws (of
the ``--scored-draws`` the scoring thins the trace to, so the same draws) whose
smallest block-group D/H is smallest -- D the expected number of distinct
household records among the block group's H published households drawn by the
shares, as compare_methods.py -- and, for each, on the block group giving it:

- its D/H in the draw against its D/H over all the draws;
- the draw's fit there against the other draws': the root mean square z,
  (allocated - published) / SE, over the block group's and its tract's
  constraint leaf cells and held-out leaf cells, with the draw's worst held-out
  cells; copying a few records many times is unrealistic in a way the data can
  show when it spoils the held-out tables the fit never saw;
- the records with the largest shares in the draw: each one's PUMS SERIALNO,
  share and copies (share x H), its share over all the draws, its published
  categories and, with ``--pums``, its household's PUMS attributes, which no
  table constrains (``PUMS_HOUSING``, and each person's age and sex).

Writes the report to ``--out`` and prints it. Reads the experiment's saved fits,
so it needs ``PMEDM_VB_DATA`` pointing at the experiment's data::

    PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \\
    $CONDA_PREFIX/bin/python experiments/inspect_block_group.py \\
        --root $RUNS/paper_v3 --experiment exp01_baseline --puma 4701504 --alpha 0.01 \\
        --out $RUNS/paper_v3/block_group_4701504.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_methods as cm  # noqa: E402
from run_experiment import area_slug, fit_name, rake_output, study_area  # noqa: E402

from pmedm_vb.assemble.allcells import is_subtotal, table_leaves  # noqa: E402
from pmedm_vb.assemble.heldout import HeldOut  # noqa: E402
from pmedm_vb.assemble.inputs import PMEDMInputs  # noqa: E402
from pmedm_vb.config import processed_dir  # noqa: E402
from pmedm_vb.solvers.base import ConstraintOperator, weights_from_lambda  # noqa: E402

#: Household attributes shown with --pums, where the vintage has them: persons,
#: household type, tenure, units in structure, bedrooms, vehicles, household
#: income (not ADJINC-adjusted) and when moved in.
PUMS_HOUSING = ("NP", "HHT", "TEN", "BLD", "BDSP", "VEH", "HINCP", "MV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--experiment", default="exp01_baseline")
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--zone", type=int, default=None,
                        help="a block group's row (default: the IPF fit's smallest n_eff)")
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--draws", type=int, default=400, help="HMC reference draws used")
    parser.add_argument("--top", type=int, default=8, help="records listed")
    parser.add_argument("--least-diverse", type=int, default=0,
                        help="report on the N HMC reference draws with the smallest D/H instead")
    parser.add_argument("--scored-draws", type=int, default=4000,
                        help="--least-diverse: the draws the scoring thinned the trace to")
    parser.add_argument("--pums", action="store_true",
                        help="--least-diverse: the records' PUMS household attributes")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def shares(W: np.ndarray, households: np.ndarray) -> np.ndarray:
    """Each household record's share of every block group's households."""
    W_hh = W[:, households]
    return W_hh / W_hh.sum(axis=1, keepdims=True)


def n_eff(s: np.ndarray) -> np.ndarray:
    return 1.0 / np.square(s).sum(axis=-1)


def hmc_draws(path: Path, draws: int) -> np.ndarray:
    """``(m, <= draws)`` evenly spaced multipliers, today's layout, from a trace."""
    with np.load(path) as saved:
        kept = saved["lam"]
    flat = kept.reshape(-1, kept.shape[-1])
    pick = np.unique(np.linspace(0, len(flat) - 1, min(draws, len(flat))).astype(int))
    return flat[pick].T


def describe_record(unit: int, inputs: PMEDMInputs, x_b: sp.csr_matrix) -> str:
    """The published leaf categories a record adds to, with how many."""
    row = x_b[unit].toarray().ravel()
    parts = [f"{inputs.bg_constraints[k]}={row[k]:g}" for k in np.flatnonzero(row)
             if not is_subtotal(inputs.bg_constraints[k])]
    return "; ".join(parts)


def leaf_cells(names) -> np.ndarray:
    return np.array([k for k, name in enumerate(names) if not is_subtotal(name)], dtype=int)


def value_labels(area, variables) -> dict[str, dict[str, str]]:
    """Each variable's single-code value labels, from the PUMS data dictionary."""
    from pmedm_vb.data import pums

    frame = pums.data_dictionary(area)
    frame = frame[(frame["record"] == "VAL") & frame["variable"].isin(variables)
                  & (frame["lo"] == frame["hi"])]
    out: dict[str, dict[str, str]] = {}
    for row in frame.itertuples():
        out.setdefault(row.variable, {})[str(row.lo).lstrip("0") or "0"] = row.label
    return out


def code_of(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).lstrip("0") or "0"


def pums_attributes(area, serials: list[str]) -> dict[str, str]:
    """Each household's PUMS_HOUSING attributes, and its persons' ages and sexes."""
    from pmedm_vb.data import pums

    declared = set(pums.variables(area)["variable"])
    housing_vars = [v for v in PUMS_HOUSING if v in declared]
    person_vars = [v for v in ("AGEP", "SEX") if v in declared]
    housing = pums.load_pums(area, housing_vars, record_type="housing", replicate_weights=False)
    housing = housing[housing["SERIALNO"].isin(serials)].set_index("SERIALNO")
    persons = pums.load_pums(area, person_vars, record_type="person", replicate_weights=False)
    persons = persons[persons["SERIALNO"].isin(serials)]
    labels = value_labels(area, housing_vars + ["SEX"])
    out = {}
    for serial in serials:
        parts = []
        if serial in housing.index:
            row = housing.loc[serial]
            for v in housing_vars:
                code = code_of(row[v])
                parts.append(f"{v}={labels.get(v, {}).get(code, code)[:40]}")
        members = persons[persons["SERIALNO"] == serial].sort_values("SPORDER")
        if len(members) and "AGEP" in members:
            sexes = labels.get("SEX", {})
            parts.append("persons " + ", ".join(
                f"{code_of(m.AGEP)}{sexes.get(code_of(m.SEX), '?')[:1] if 'SEX' in members else ''}"
                for m in members.itertuples()))
        out[serial] = "; ".join(parts) if parts else "not found in the PUMS files"
    return out


def least_diverse_report(args, spec, exp, inputs, households, published_hh) -> list[str]:
    """The ``--least-diverse`` report (see the module docstring)."""
    hh = np.flatnonzero(households)
    zones = np.flatnonzero(published_hh > args.hh_min)
    H = published_hh[zones]
    trace = exp / args.puma / "hmc_ref" / f"{fit_name(spec, args.puma, args.alpha)}_trace.npz"
    with np.load(trace) as saved:
        lam = cm.thinned(saved["lam"], args.scored_draws)
    op = ConstraintOperator(inputs)
    n = lam.shape[1]

    def weights(d: int) -> np.ndarray:
        return inputs.N * np.exp(cm.log_p_of(inputs, op, lam[:, d]))

    def shares(W: np.ndarray, rows) -> np.ndarray:
        w = W[np.ix_(rows, hh)]
        return w / w.sum(axis=1, keepdims=True)

    dh = np.empty((n, zones.size))
    for d in range(n):
        dh[d] = cm.expected_distinct(shares(weights(d), zones), H) / H
    picks = np.argsort(dh.min(axis=1), kind="stable")[:args.least_diverse]

    # The cell sets of each pick's block group and tract: (name, level, columns
    # of X, published, SE, the areas summed).
    heldout = HeldOut.load(processed_dir() / "inputs" / area_slug(spec["area"]) / args.puma)
    variance = inputs.floored_variances(None)
    tracts = inputs.zone_tracts()
    n_t, n_b = inputs.Y_T.shape[0], inputs.Y_B.shape[0]

    def cell_sets(zone: int) -> list[tuple]:
        t = int(tracts[zone])
        in_tract = np.flatnonzero(tracts == t)
        bg, tr = leaf_cells(inputs.bg_constraints), leaf_cells(inputs.tract_constraints)
        sets = [("block group, constraints", inputs.bg_constraints, inputs.X_B, bg,
                 inputs.Y_B[zone, bg], np.sqrt(variance[inputs.Y_T.size + bg * n_b + zone]), [zone]),
                ("tract, constraints", inputs.tract_constraints, inputs.X_T, tr,
                 inputs.Y_T[t, tr], np.sqrt(variance[tr * n_t + t]), in_tract)]
        for level, area, rows in (("block group", zone, [zone]), ("tract", t, in_tract)):
            names = heldout.names[level]
            keep = leaf_cells(names)
            sets.append((f"{level}, held out", names, heldout.X[level], keep,
                         heldout.Y[level][area, keep], np.sqrt(heldout.v[level][area, keep]), rows))
        return sets

    serial = inputs.units["SERIALNO"].astype(str).to_numpy()
    lines = [f"{args.experiment}, PUMA {args.puma}, alpha {args.alpha:g}: the {len(picks)} HMC "
             f"reference draws (of {n}) with the smallest block-group D/H, over the "
             f"{zones.size} block groups with more than {args.hh_min:g} households", ""]
    for rank, d in enumerate(picks, start=1):
        position = int(np.argmin(dh[d]))
        zone = int(zones[position])
        sets = cell_sets(zone)
        # Every draw's allocation in this block group and its tract's cells.
        s_all = np.empty((n, hh.size))
        outcomes = [np.empty((n, len(cols))) for _, _, _, cols, _, _, _ in sets]
        X_sets = [sp.csr_matrix(X)[:, cols].T.tocsr() for _, _, X, cols, _, _, _ in sets]
        for e in range(n):
            W = weights(e)
            s_all[e] = shares(W, [zone])[0]
            for i, (_, _, _, _, _, _, rows) in enumerate(sets):
                outcomes[i][e] = X_sets[i] @ W[rows].sum(axis=0)
        own = dh[:, position]
        lines += [f"Draw {d} (#{rank}): block group {inputs.zones.iloc[zone].iloc[0]} (row {zone}), "
                  f"{published_hh[zone]:,.0f} households",
                  f"  D/H {own[d]:.3f} (~{own[d] * published_hh[zone]:.0f} distinct records); "
                  f"over all draws median {np.median(own):.3f}, 5% {np.quantile(own, 0.05):.3f}, "
                  f"this draw's rank {int((own < own[d]).sum()) + 1} of {n}; the block group's "
                  f"median D/H ranks {int((np.median(dh, axis=0) < np.median(own)).sum()) + 1} "
                  f"of {zones.size} (1 = least diverse)",
                  "  Fit, root mean square z over the cells with an SE (this draw; all draws' "
                  "5%, 50%, 95%; this draw's percentile):"]
        worst = []
        for (name, names, _, cols, Y, se, _), values in zip(sets, outcomes):
            ok = se > 0
            z = (values[:, ok] - Y[ok]) / se[ok]
            rms = np.sqrt(np.mean(np.square(z), axis=1))
            lines.append(f"    {name:<26} {rms[d]:6.2f};  {np.quantile(rms, 0.05):6.2f} "
                         f"{np.median(rms):6.2f} {np.quantile(rms, 0.95):6.2f};  "
                         f"{(rms < rms[d]).mean():.0%}  ({int(ok.sum())} cells)")
            if "held out" in name:
                median = np.median(values[:, ok], axis=0)
                for j in np.argsort(-np.abs(z[d]))[:3]:
                    k = np.flatnonzero(ok)[j]
                    worst.append(f"    {name.split(',')[0]:<11} {names[cols[k]]:<60} published "
                                 f"{Y[k]:7.1f} (SE {se[k]:5.1f}); this draw {values[d, k]:7.1f}, "
                                 f"z {z[d, j]:+5.1f}; all draws' median {median[j]:7.1f}")
        lines += ["  This draw's worst held-out cells:", *worst,
                  "  Records with the largest shares in this draw (share; copies, share x H; "
                  "share over all draws, median and 95%):"]
        top = np.argsort(-s_all[d])[:args.top]
        attributes = (pums_attributes(study_area(spec["area"]), [serial[hh[j]] for j in top])
                      if args.pums else {})
        x_b = sp.csr_matrix(inputs.X_B)
        for j in top:
            unit = int(hh[j])
            lines += [f"    record {unit}, SERIALNO {serial[unit]}: {s_all[d, j]:.3f}; "
                      f"{s_all[d, j] * published_hh[zone]:.0f} copies; "
                      f"{np.median(s_all[:, j]):.3f}, {np.quantile(s_all[:, j], 0.95):.3f}",
                      f"      tables: {describe_record(unit, inputs, x_b)}"]
            if args.pums:
                lines.append(f"      PUMS: {attributes[serial[unit]]}")
        lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    exp = args.root / args.experiment
    spec = json.loads((exp / "experiment.json").read_text())
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / area_slug(spec["area"]) / args.puma)
    households = ~inputs.units["is_group_quarters"].to_numpy(bool)
    hh_index = np.flatnonzero(households)
    tenure = table_leaves(inputs.bg_constraints, "B25003")
    published_hh = inputs.Y_B[:, tenure].sum(axis=1)
    eligible = published_hh > args.hh_min
    if args.least_diverse:
        report = "\n".join(least_diverse_report(args, spec, exp, inputs, households,
                                                 published_hh))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(report)
        return

    with np.load(rake_output(spec, exp, args.puma, "ipf")) as saved:
        W_ipf = np.asarray(saved["W"], dtype=float)
    s_ipf = shares(W_ipf, households)
    neff_ipf = n_eff(s_ipf)
    zone = (args.zone if args.zone is not None
            else int(np.flatnonzero(eligible)[np.argmin(neff_ipf[eligible])]))

    lam = hmc_draws(exp / args.puma / "hmc_ref" / f"{fit_name(spec, args.puma, args.alpha)}_trace.npz",
                    args.draws)
    s_hmc = np.empty((lam.shape[1], hh_index.size))
    for d in range(lam.shape[1]):
        W = inputs.N * weights_from_lambda(inputs, lam[:, d])
        s_hmc[d] = shares(W[[zone]], households)[0]
    neff_hmc = n_eff(s_hmc)
    s_base = shares(inputs.q[[zone]], households)[0]

    tract = int(inputs.zone_tracts()[zone])
    zone_id = inputs.zones.iloc[zone].to_dict() if hasattr(inputs.zones, "iloc") else zone
    tract_id = inputs.tracts.iloc[tract].to_dict() if hasattr(inputs.tracts, "iloc") else tract
    population = table_leaves(inputs.bg_constraints, "B01001")
    lines = [f"{args.experiment}, PUMA {args.puma}, alpha {args.alpha:g}: block group row {zone}",
             f"  block group {zone_id}, tract {tract_id}",
             f"  published occupied households {published_hh[zone]:,.0f} (B25003 owner + renter); "
             f"population {inputs.Y_B[zone, population].sum():,.0f} (B01001 leaves)",
             f"  household records that may be placed here: "
             f"{int((inputs.q[zone, households] > 0).sum()):,} of {hh_index.size:,}",
             f"  n_eff: IPF fit {neff_ipf[zone]:.2f}; HMC reference median "
             f"{np.median(neff_hmc):.2f} (90%: {np.quantile(neff_hmc, 0.05):.2f} to "
             f"{np.quantile(neff_hmc, 0.95):.2f}, {neff_hmc.size} draws); base weights "
             f"{n_eff(s_base):.1f}",
             f"  the IPF fit's smallest n_eff over the {int(eligible.sum())} block groups over "
             f"{args.hh_min:g} households: this one" if args.zone is None else "",
             ""]

    # Published leaf cells: the block group's and its tract's, zeros and small first.
    n_tracts = inputs.Y_T.shape[0]
    variance = inputs.floored_variances(None)
    rows = []
    for level, names, Y, offset, area, n_areas in (
            ("block group", inputs.bg_constraints, inputs.Y_B, inputs.Y_T.size, zone,
             inputs.Y_B.shape[0]),
            ("tract", inputs.tract_constraints, inputs.Y_T, 0, tract, n_tracts)):
        for k, name in enumerate(names):
            if is_subtotal(name):
                continue
            se = float(np.sqrt(variance[offset + k * n_areas + area]))
            rows.append(dict(level=level, cell=name, published=float(Y[area, k]), se=se))
    cells = pd.DataFrame(rows).sort_values(["level", "published"])
    lines += ["Published leaf cells (smallest first):",
              cells.to_string(index=False, float_format=lambda v: f"{v:,.1f}"), ""]

    # The records with the largest shares.
    x_b = sp.csr_matrix(inputs.X_B)
    mean_hmc = s_hmc.mean(axis=0)
    top_hmc = np.bincount(s_hmc.argmax(axis=1), minlength=hh_index.size) / s_hmc.shape[0]
    order = np.unique(np.concatenate([np.argsort(-s_ipf[zone])[:args.top],
                                      np.argsort(-mean_hmc)[:args.top]]))
    order = order[np.argsort(-(s_ipf[zone, order] + mean_hmc[order]))]
    lines.append("Records with the largest shares of the block group's households:")
    for j in order:
        unit = int(hh_index[j])
        lines += [f"  record {unit} (design weight {inputs.units['weight'].iloc[unit]:,.1f}): "
                  f"IPF share {s_ipf[zone, j]:.3f}; HMC mean share {mean_hmc[j]:.3f} "
                  f"(90%: {np.quantile(s_hmc[:, j], 0.05):.3f} to "
                  f"{np.quantile(s_hmc[:, j], 0.95):.3f}), largest in {top_hmc[j]:.0%} of draws; "
                  f"base share {s_base[j]:.4f}",
                  f"    {describe_record(unit, inputs, x_b)}"]
    report = "\n".join(line for line in lines if line is not None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
