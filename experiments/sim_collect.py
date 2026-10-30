"""Tabulate a simulation's truths, surveys and fits (``experiments/sim/<name>.json``;
``sim_common.py``), for scoring afterwards.

Every published line of the source PUMA -- its constraint tables, all of them,
and its held-out tables -- at block group and tract, for each truth:

``estimates/kind=<kind>/truth_id=<i>/part.parquet``
    One row per (draw, level, geoid, line): ``tract`` (a block group's tract;
    a tract's own GEOID), ``table``; ``truth`` (the true count), ``survey``
    (the survey's estimate), ``se`` (the square root of ``c`` times the line's
    published variance), ``bernoulli_se`` (the survey's exact standard error
    given the truth), ``ipf`` (the IPF fit, the same on every draw), and the
    ``draw``-th of ``draws`` draws of ``ipf_multinomial``, ``vb_gaussian``,
    ``vb_gaussian_trunc`` (the same fit's draws with the main experiments'
    rejection rule, ``compare_methods.truncated_draws``, from the source
    spec's ``vb_trunc_share`` and ``vb_trunc_distinct``; a random stream of
    its own, so the other methods' draws do not depend on it) and
    ``hmc_short`` (one from each of HMC short's superchains). VB's and HMC's
    draws are ``W = N p``, never exactly zero; ``vb_gaussian_int``,
    ``vb_gaussian_trunc_int`` and ``hmc_short_int`` are the same draws made populations as the truths and
    IPF multinomial are, each by one multinomial draw of ``round(N)`` units
    from its ``p``, so that a zero truth can be matched. Draw ``k`` of one
    method has nothing to do with draw ``k`` of another: score each column
    against ``truth``, never one method's row against another's.
    ``arrow::open_dataset()`` reads the folder as one table, with ``kind`` and
    ``truth_id`` from the paths.
``cells.parquet``
    One row per (level, line): ``table``, ``subtotal``, ``parent`` (the
    constraint tables'), ``role`` -- ``constraint``, ``dropped_line`` (a
    constraint table's line left out, ``drop_deepest``), ``other_table`` (a
    source constraint table the simulation does not fit) or ``heldout``.
``draws.parquet``
    One row per (kind, truth_id, method, draw), the truth itself among the
    methods: per block group with more than ``--hh-min`` households in the
    truth, ``D / H`` (expected distinct household records among the truth's
    ``H`` households) and ``n_eff = 1 / sum s^2`` over the household records'
    shares, their smallest and median over block groups, and the largest cell's
    share of the total.

And, for the text's numbers: ``truncation.csv`` (per synthetic PUMA, the
truncated VB's draws tried and kept, and those rejected by each rule),
``cells.csv`` (``cells.parquet`` as CSV),
``timing.csv`` (per synthetic PUMA, the seconds of its IPF fit, its MAP + VB
fit and its HMC short run, warmup apart) and ``settings.csv`` (the simulation's
and its fits' specs, flattened to ``sim.<path>`` and ``fits.<path>``, and the
calibration, ``calibration.<path>``).

A method whose fit is missing is left empty (NaN) and reported::

    sbatch --account=acf-utk0011 experiments/run_python.sbatch sim_collect.py \\
        --sim experiments/sim/sim01.json --root $R
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402
from provenance import SIM_COLLECT_SCRIPTS, write_stamp  # noqa: E402
from sim_common import Simulation, Tabulator, constraint_columns  # noqa: E402

METHODS = ("ipf_multinomial", "vb_gaussian", "vb_gaussian_trunc", "hmc_short", "vb_gaussian_int",
           "vb_gaussian_trunc_int", "hmc_short_int")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim", type=Path, required=True, help="experiments/sim/<name>.json")
    parser.add_argument("--root", type=Path, required=True, help="the experiments' folder")
    parser.add_argument("--pumas", nargs="+", default=None,
                        help="only these synthetic PUMAs (default: every one)")
    parser.add_argument("--hh-min", type=float, default=100)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <root>/<fits name>/simulation/output")
    return parser.parse_args()


# -- the fits' draws ------------------------------------------------------------


def ipf_fit(sim: Simulation, puma: str) -> np.ndarray | None:
    from pmedm_vb.rake import rake_name

    path = (sim.fits_folder / puma / "ipf"
            / f"{rake_name(puma, 'ipf', sim.fits['model'].get('rollup'))}.npz")
    if not path.exists():
        return None
    with np.load(path) as saved:
        return np.asarray(saved["W"], dtype=float)


def multinomial_draws(W: np.ndarray, rng, draws: int) -> list[np.ndarray]:
    total = int(round(W.sum()))
    p = np.clip(W.ravel() / W.sum(), 0.0, None)
    p /= p.sum()
    return [rng.multinomial(total, p).reshape(W.shape).astype(float) for _ in range(draws)]


def populations(inputs, lam: np.ndarray) -> list[np.ndarray]:
    """``W = N p`` of each column of ``lam`` (multipliers in the inputs' layout)."""
    from compare_methods import log_p_of
    from pmedm_vb.solvers.base import ConstraintOperator

    op = ConstraintOperator(inputs)
    return [inputs.N * np.exp(log_p_of(inputs, op, lam[:, d])) for d in range(lam.shape[1])]


#: As compare_methods.py's ``--trunc-max-factor``: give up after this many times
#: ``draws`` tries.
TRUNC_MAX_FACTOR = 20


def truncation(sim: Simulation, hh_min: float) -> argparse.Namespace:
    """The truncated VB's rejection rule, the source spec's (``vb_trunc_share``,
    ``vb_trunc_distinct``), as compare_methods.py takes it."""
    missing = [k for k in ("vb_trunc_share", "vb_trunc_distinct") if k not in sim.source]
    if missing:
        raise SystemExit(f"the source spec sets no {', '.join(missing)}: the truncated VB's rule")
    return argparse.Namespace(draws=sim.draws, hh_min=hh_min, trunc_max_factor=TRUNC_MAX_FACTOR,
                              trunc_share=float(sim.source["vb_trunc_share"]),
                              trunc_distinct=float(sim.source["vb_trunc_distinct"]))


def vb_draws(sim: Simulation, puma: str, inputs, rng, trunc_rng,
             rule: argparse.Namespace) -> tuple[list | None, list | None, dict]:
    """VB Gaussian's draws, and the same fit's truncated draws with their
    acceptance counts (``compare_methods.truncated_draws``)."""
    from compare_methods import check_level, floor_spec, taper_spec, truncated_draws, with_lambda
    from laplace_diagnostic import load_vb
    from pmedm_vb.assemble.hierarchy import Hierarchy

    path = sim.fits_folder / puma / "vb_gaussian" / f"{rx.fit_name(sim.fits, puma, sim.alpha)}.npz"
    if not path.exists():
        return None, None, {}
    level = rx.level(sim.fits)
    with np.load(path) as saved:
        check_level(path, saved, level)
    q, _, _ = load_vb(path)
    model = sim.fits["model"]
    h = Hierarchy.build(inputs, level, alpha=sim.alpha, taper=taper_spec(model["taper"]),
                        variance_floor=floor_spec(str(model["variance_floor"])))
    plain = populations(inputs, with_lambda(h, q.sample(rng, sim.draws), q)["lam"])
    xi, counts = truncated_draws(rule, inputs, trunc_rng, h, q)
    return plain, populations(inputs, with_lambda(h, xi, q)["lam"]), counts


def hmc_draws(sim: Simulation, puma: str, inputs) -> list[np.ndarray] | None:
    """One draw from each superchain of HMC short: the last iteration of the
    first chain of each (chain ``c`` is in superchain ``c // M``)."""
    path = (sim.fits_folder / puma / "hmc_short"
            / f"{rx.fit_name(sim.fits, puma, sim.alpha)}_trace.npz")
    if not path.exists():
        return None
    with np.load(path) as saved:
        kept, k = saved["lam"], int(saved["superchains"])
    chains = kept.shape[1]
    if k != sim.draws:
        raise SystemExit(f"{path}: {k} superchains, but the simulation takes {sim.draws} draws")
    return populations(inputs, kept[-1, np.arange(k) * (chains // k), :].T)


def settings(sim: Simulation, hh_min: float) -> pd.DataFrame:
    """The simulation's settings and calibration, flattened (``paper_tables.flatten``)."""
    import json

    from paper_tables import flatten

    rows = flatten(sim.settings, "sim") + flatten(
        {k: v for k, v in sim.fits.items() if not k.startswith("_")}, "fits")
    rule = truncation(sim, hh_min)
    rows += [("truncation.share", rule.trunc_share), ("truncation.distinct", rule.trunc_distinct),
             ("truncation.hh_min", rule.hh_min)]
    calibration = sim.folder / "calibration.json"
    if calibration.exists():
        rows += flatten(json.loads(calibration.read_text()), "calibration")
    return pd.DataFrame(rows, columns=["key", "value"])


def timing(sim: Simulation) -> pd.DataFrame:
    """Per synthetic PUMA, the seconds each fit took (NaN where it is missing)."""
    from pmedm_vb.rake import rake_name

    rows = []
    name = {p: rx.fit_name(sim.fits, p, sim.alpha) for p in sim.sim_pumas()}
    for puma in sim.sim_pumas():
        row = dict(puma=puma, kind=next(k for k in sim.kinds() if puma in
                                        [sim.sim_puma(k, i) for i in
                                         range(1, sim.settings["truths"][k] + 1)]),
                   truth_id=int(puma[-2:]))
        paths = {"ipf": (sim.fits_folder / puma / "ipf"
                         / f"{rake_name(puma, 'ipf', sim.fits['model'].get('rollup'))}.npz",
                         ["seconds"]),
                 "vb": (sim.fits_folder / puma / "vb_gaussian" / f"{name[puma]}.npz",
                        ["seconds", "map_seconds"]),
                 "hmc": (sim.fits_folder / puma / "hmc_short" / f"{name[puma]}_trace.npz",
                         ["seconds", "warmup_seconds"])}
        for method, (path, keys) in paths.items():
            values = {}
            if path.exists():
                with np.load(path) as saved:
                    values = {k: float(saved[k]) for k in keys if k in saved.files}
            for k in keys:
                row[f"{method}_{k}"] = values.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# -- tables ---------------------------------------------------------------------


def cell_frame(tab: Tabulator, inputs, columns: dict, sim: Simulation) -> pd.DataFrame:
    """``cells.parquet``: one row per (level, line)."""
    from pmedm_vb.assemble.allcells import is_subtotal

    fitted = {"tract": {inputs.tract_constraints[k] for k in columns["tract"]},
              "block group": {inputs.bg_constraints[k] for k in columns["block group"]}}
    parents = {}
    for table, entry in inputs.trees.items():
        lines = entry["lines"]
        for line, parent in zip(lines, entry["parents"]):
            parents[f"{table}.{line}"] = None if parent < 0 else lines[parent]
    rows = []
    for level, names in tab.names.items():
        constrained = set(sim.settings["constraints"][level])
        for name in names:
            table = name.split(".")[0]
            role = ("heldout" if name in tab.heldout[level]
                    else "constraint" if name in fitted[level]
                    else "dropped_line" if table in constrained else "other_table")
            rows.append(dict(level=level, table=table, line=name.split(".", 1)[1],
                             subtotal=is_subtotal(name), parent=parents.get(name), role=role))
    return pd.DataFrame(rows)


def keys(tab: Tabulator, inputs) -> dict[str, pd.DataFrame]:
    """Per level, the (geoid, tract, table, line) of each cell, row-major over
    (area, line) as the tabulations ravel."""
    bg = inputs.block_groups["block_group_geoid"].astype(str).to_numpy()
    tracts = inputs.tracts["tract_geoid"].astype(str).to_numpy()
    areas = {"block group": (bg, tracts[inputs.zone_tracts()]), "tract": (tracts, tracts)}
    out = {}
    for level, names in tab.names.items():
        geoid, tract = areas[level]
        n, k = len(geoid), len(names)
        out[level] = pd.DataFrame({
            "level": level,
            "geoid": np.repeat(geoid, k), "tract": np.repeat(tract, k),
            "table": np.tile([name.split(".")[0] for name in names], n),
            "line": np.tile([name.split(".", 1)[1] for name in names], n)})
    return out


def usability(W: np.ndarray, H: np.ndarray, households: np.ndarray, hh_min: float) -> dict:
    """Per-draw D/H, n_eff and largest share, over block groups with more than
    ``hh_min`` households (``H``, the truth's)."""
    from compare_methods import expected_distinct

    zones = H > hh_min
    if not zones.any():
        return dict(dh_min=np.nan, dh_median=np.nan, neff_min=np.nan, neff_median=np.nan,
                    max_share=float(W.max() / W.sum()), block_groups=0)
    w = W[np.ix_(zones, households)]
    s = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    dh = expected_distinct(s, H[zones]) / H[zones]
    neff = 1.0 / np.square(s).sum(axis=1)
    return dict(dh_min=float(dh.min()), dh_median=float(np.median(dh)),
                neff_min=float(neff.min()), neff_median=float(np.median(neff)),
                max_share=float(W.max() / W.sum()), block_groups=int(zones.sum()))


def main() -> None:
    args = parse_args()
    sim = Simulation(args.sim, args.root)
    from pmedm_vb.assemble.allcells import table_leaves
    from pmedm_vb.assemble.heldout import HeldOut
    from pmedm_vb.assemble.inputs import PMEDMInputs

    out = args.out or sim.folder / "output"
    out.mkdir(parents=True, exist_ok=True)
    source = PMEDMInputs.load(sim.source_inputs())
    tab = Tabulator(source, HeldOut.load(sim.source_inputs()))
    columns = constraint_columns(sim, source)
    cells = cell_frame(tab, source, columns, sim)
    cells.to_parquet(out / "cells.parquet", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    settings(sim, args.hh_min).to_csv(out / "settings.csv", index=False)
    timing(sim).to_csv(out / "timing.csv", index=False)
    key = keys(tab, source)
    tenure = table_leaves(source.bg_constraints, "B25003")
    households = ~source.units["is_group_quarters"].to_numpy(bool)
    usable, missing, accepted = [], [], []
    rule = truncation(sim, args.hh_min)
    pumas = args.pumas or sim.sim_pumas()
    for index, puma in enumerate(pumas):
        with np.load(sim.truth_file(puma)) as saved:
            T, S = saved["T"].astype(float), saved["S"].astype(float)
            c, kind, f = float(saved["c"]), str(saved["kind"]), float(saved["fraction"])
        truth_id = int(puma[-2:])
        inputs = PMEDMInputs.load(sim.sim_inputs(puma))
        stream = [sim.settings["seed"], 1 + sim.sim_pumas().index(puma)]
        rng = np.random.default_rng(stream)
        trunc_rng = np.random.default_rng(stream + [2])  # its own: the rest as before
        ipf = ipf_fit(sim, puma)
        multinomial = None if ipf is None else multinomial_draws(ipf, rng, sim.draws)
        vb, vb_trunc, counts = vb_draws(sim, puma, inputs, rng, trunc_rng, rule)
        draws = {"ipf_multinomial": multinomial, "vb_gaussian": vb, "vb_gaussian_trunc": vb_trunc,
                 "hmc_short": hmc_draws(sim, puma, inputs)}
        missing += [f"{puma} {m}" for m, d in [("ipf", ipf), *draws.items()] if d is None]
        if vb_trunc is not None and len(vb_trunc) < sim.draws:
            missing.append(f"{puma} vb_gaussian_trunc ({len(vb_trunc)} of {sim.draws} kept)")
        if counts:
            accepted.append(dict(puma=puma, kind=kind, truth_id=truth_id, **counts))
        for m in ("vb_gaussian", "hmc_short"):  # integer populations, as the truths are
            draws[f"{m}_int"] = (None if draws[m] is None
                                 else [multinomial_draws(W, rng, 1)[0] for W in draws[m]])
        draws["vb_gaussian_trunc_int"] = (None if vb_trunc is None else
                                          [multinomial_draws(W, trunc_rng, 1)[0] for W in vb_trunc])

        truth, survey = tab(T), tab(S / f)
        bern = {level: np.sqrt(v * (1 - f) / f) for level, v in tab(T, squared=True).items()}
        fitted = None if ipf is None else tab(ipf)
        tabulated = {m: None if d is None else [tab(W) for W in d] for m, d in draws.items()}
        frames = []
        for level in ("block group", "tract"):
            base = key[level].assign(
                truth=truth[level].ravel(), survey=survey[level].ravel(),
                se=np.sqrt(c * tab.v[level]).ravel(), bernoulli_se=bern[level].ravel(),
                ipf=np.nan if fitted is None else fitted[level].ravel())
            for d in range(sim.draws):
                frame = base.assign(draw=d + 1)
                for m in METHODS:
                    have = tabulated[m] is not None and d < len(tabulated[m])
                    frame[m] = tabulated[m][d][level].ravel() if have else np.nan
                frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        for column in ("level", "geoid", "tract", "table", "line"):
            frame[column] = frame[column].astype("category")
        for column in ("truth", "survey", "se", "bernoulli_se", "ipf", *METHODS):
            frame[column] = frame[column].astype(np.float32)
        frame = frame[["draw", "level", "geoid", "tract", "table", "line", "truth", "survey", "se",
                       "bernoulli_se", "ipf", *METHODS]]
        folder = out / "estimates" / f"kind={kind}" / f"truth_id={truth_id:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(folder / "part.parquet", index=False)

        H = T @ np.asarray(source.X_B[:, tenure].sum(axis=1)).ravel()
        base = dict(kind=kind, truth_id=truth_id)
        usable.append(dict(base, method="truth", draw=1,
                           **usability(T, H, households, args.hh_min)))
        if ipf is not None:
            usable.append(dict(base, method="ipf", draw=1,
                               **usability(ipf, H, households, args.hh_min)))
        for m, d in draws.items():
            for i, W in enumerate(d or []):
                usable.append(dict(base, method=m, draw=i + 1,
                                   **usability(W, H, households, args.hh_min)))
        print(f"[{index + 1}/{len(pumas)}] {puma}: {len(frame):,} rows", flush=True)
    pd.DataFrame(accepted).to_csv(out / "truncation.csv", index=False)
    pd.DataFrame(usable).to_parquet(out / "draws.parquet", index=False)
    if missing:
        print(f"missing fits, left empty: {', '.join(missing)}")
    elif args.pumas is None:
        # Last, and only when complete, so that run_sim.py takes it as current.
        write_stamp(out / "draws.parquet", SIM_COLLECT_SCRIPTS, sim=sim.settings["name"],
                    hh_min=args.hh_min)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
