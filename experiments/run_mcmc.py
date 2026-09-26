"""A reference posterior by HMC for one PUMA, and its report.

Samples ``pi(lambda) ~ exp(-n f(lambda))`` with :mod:`pmedm_vb.solvers.mcmc`,
whitened by the Gaussian part of a fitted VB family, then reports the sampler's
health, convergence, and how concentrated the posterior's weights get -- the
same question ``weight_draws.py`` asks of VB. Two steps::

    # GPU node (see run_mcmc.sbatch)
    $CONDA_PREFIX/bin/python experiments/run_mcmc.py sample --puma 4701502 --alpha 1.0 \\
        --variance-floor zero --vb-run $RUNS/<vb jobid> --out $RUNS/<jobid>/mcmc
    # anywhere, afterwards
    $CONDA_PREFIX/bin/python experiments/run_mcmc.py report --puma 4701502 --alpha 1.0 \\
        --variance-floor zero --vb-run $RUNS/<vb jobid> --out $RUNS/<jobid>/mcmc

**sample** starts every chain at a draw from the VB Gaussian -- screened so
that no chain starts with over ``--init-max-share`` of ``p`` on one cell, since
a chain started on a wall never moves (see
:func:`~pmedm_vb.solvers.mcmc.starting_points`) -- runs ``--warmup``
iterations adapting the step size, then ``--samples`` more at a fixed one. Every
iteration records, per chain, ``log pi``, the acceptance probability, whether it
diverged, the energy, and the largest (zone, unit) cell's share of ``p`` and
its index; every ``--thin``-th sampling iteration also keeps ``lambda``. Every
``--checkpoint`` iterations the state and records are written to
``<out>/<name>_state.npz`` and ``<name>_trace.npz``, and a job started with
an ``--out`` that holds them resumes where they stopped -- so a run that hits
its time limit is resubmitted, not restarted. The settings must match. The
trace also carries ``seconds``, the sampler's wall time summed over every job
that contributed, and ``warmup_seconds``, the part spent in warmup, so that a
resubmitted run still reports its total.

**report** writes ``<out>/<name>_report.txt``:

1. Sampler: step size, leapfrog steps, acceptance, divergences, E-BFMI per chain,
   and any chain that is stuck (mean acceptance under 5%).
2. Convergence: rank-normalised split R-hat and bulk/tail ESS (Vehtari et al.
   2021, *Bayesian Analysis* 16) for ``log pi`` and the largest-cell share, and
   over every coordinate of the kept ``lambda`` draws, with the ten worst.
3. Concentration: quantiles of the largest cell's share of ``N`` over the
   sampling iterations, the share of iterations over the MAP's largest cell,
   1% and 10% of ``N``, and the records responsible. Comparable to section 1
   of ``weight_draws.py``, except that successive iterations are correlated:
   the ESS of the share in section 2 says how many independent draws they are
   worth.

``--variance-floor`` must match the VB run, since it sets ``Sigma`` in ``f``.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import logger, record_run

from laplace_diagnostic import constraint_table, load_vb

#: Largest-cell thresholds as shares of N, as in weight_draws.py.
SHARES = (0.01, 0.10)
#: Per-iteration records, each (iterations, chains).
TRACE_KEYS = ("log_pi", "accept_prob", "divergent", "energy", "max_share", "max_cell")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["sample", "report"])
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", choices=["tract", "none"], default="tract")
    parser.add_argument("--hierarchy", choices=["none", "tract", "puma"], default="none",
                        help="the VB run's hierarchy level (pmedm_vb.assemble.hierarchy); "
                             "names gain _h<level>")
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument(
        "--variance-floor", default="none",
        help="'none', 'zero' or a number, as in run_map.py; must match the VB run",
    )
    parser.add_argument("--vb-run", type=Path, required=True,
                        help="VB results directory whose fit whitens the sampler")
    parser.add_argument("--out", type=Path, required=True, help="MCMC results directory")
    parser.add_argument("--chains", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--thin", type=int, default=10, help="keep lambda every this many iterations")
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--trajectory", type=float, default=3.0,
                        help="nominal integration time; 3.0 mixed rare directions far better than 1.5")
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--max-leapfrog", type=int, default=1000)
    parser.add_argument("--init-max-share", type=float, default=0.01,
                        help="largest share of p on one cell a starting point may have")
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    return parser.parse_args()


def floor_spec(text: str) -> str | float | None:
    if text == "none":
        return None
    if text == "zero":
        return "zero"
    return float(text)


def fit_name(args: argparse.Namespace) -> str:
    suffix = "" if args.hierarchy == "none" else f"_h{args.hierarchy}"
    return f"{args.puma}_{args.taper}_a{args.alpha:g}{suffix}"


def load_problem(args: argparse.Namespace):
    from pmedm_vb.assemble.hierarchy import Hierarchy

    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    taper = None if args.taper == "none" else args.taper
    hierarchy = Hierarchy.build(inputs, args.hierarchy)
    sigma = hierarchy.sigma(inputs, args.alpha, taper, floor_spec(args.variance_floor))
    vb_path = args.vb_run / f"{fit_name(args)}.npz"
    q, _, _ = load_vb(vb_path)
    with np.load(vb_path) as saved:
        map_lam = saved["map_lam"]
        saved_level = str(saved["hierarchy"]) if "hierarchy" in saved.files else "none"
    if saved_level != args.hierarchy:
        raise SystemExit(f"{vb_path} was fitted with hierarchy={saved_level}, not {args.hierarchy}")
    return inputs, sigma, q, map_lam, hierarchy


# -- sample ------------------------------------------------------------------


def draw_arrays(draws: np.ndarray, hierarchy) -> dict[str, np.ndarray]:
    """Kept draws for the trace: ``lam`` always in today's layout (the
    multipliers the data see), plus ``xi`` under a hierarchy, for densities."""
    if hierarchy.is_trivial:
        return {"lam": draws}
    kept, chains, size = draws.shape
    lam = hierarchy.lambda_data(draws.reshape(-1, size).T).T.reshape(kept, chains, hierarchy.m)
    return {"lam": lam, "xi": draws}


def sample(args: argparse.Namespace) -> None:
    import torch

    from pmedm_vb.solvers.mcmc import HMC, HMCSettings, starting_points, whitening_matrix
    from pmedm_vb.solvers.vb import _DualTarget

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    name = fit_name(args)
    state_path = args.out / f"{name}_state.npz"
    trace_path = args.out / f"{name}_trace.npz"

    inputs, sigma, q, _, hierarchy = load_problem(args)
    target = _DualTarget(inputs, sigma, device, hierarchy)
    A = whitening_matrix(q)
    rng = np.random.default_rng(args.seed)
    x0 = starting_points(target, q.mean, A, args.chains, rng, args.init_max_share, device)
    settings = HMCSettings(
        step_size=args.step_size, trajectory=args.trajectory,
        target_accept=args.target_accept, max_leapfrog=args.max_leapfrog,
    )
    sampler = HMC(target, inputs.n, q.mean, A, x0, settings, seed=args.seed, device=device)

    trace = {key: [] for key in TRACE_KEYS}
    step_sizes, leapfrogs, draws = [], [], []
    earlier_seconds, warmup_seconds = 0.0, float("nan")
    if state_path.exists() and trace_path.exists():
        with np.load(state_path) as state:
            sampler.load_state_dict(dict(state))
        with np.load(trace_path) as saved:
            trace = {key: list(saved[key]) for key in TRACE_KEYS}
            step_sizes, leapfrogs = list(saved["step_size"]), list(saved["n_leapfrog"])
            key = "lam" if hierarchy.is_trivial else "xi"
            draws = list(saved[key]) if key in saved.files else []
            earlier_seconds = float(saved["seconds"]) if "seconds" in saved.files else 0.0
            if "warmup_seconds" in saved.files:
                warmup_seconds = float(saved["warmup_seconds"])
        if sampler.chains != args.chains:
            raise SystemExit(f"{state_path} has {sampler.chains} chains, not {args.chains}")
        logger.info("resuming %s at iteration %d", name, sampler.iteration)

    total = args.warmup + args.samples
    logger.info(
        "HMC %s on %s: %s multipliers, %d chains, %d warmup + %d samples, n = %d",
        name, device, f"{q.size:,}", args.chains, args.warmup, args.samples, inputs.n,
    )
    started = time.perf_counter()
    start_iteration = sampler.iteration

    def save() -> None:
        np.savez(state_path, **sampler.state_dict())
        np.savez(
            trace_path,
            **{key: np.asarray(value) for key, value in trace.items()},
            step_size=np.asarray(step_sizes), n_leapfrog=np.asarray(leapfrogs),
            warmup=args.warmup, thin=args.thin, n_zones=inputs.n_zones, n_units=inputs.n_units,
            seconds=earlier_seconds + time.perf_counter() - started,
            warmup_seconds=warmup_seconds,
            **(draw_arrays(np.asarray(draws), hierarchy) if draws else {}),
        )

    while sampler.iteration < total:
        warming = sampler.iteration < args.warmup
        stats = sampler.step(adapt=warming)
        if sampler.iteration == args.warmup:
            sampler.end_warmup()
            warmup_seconds = earlier_seconds + time.perf_counter() - started
            logger.info("warmup done: step size fixed at %.4g", sampler.step_size)
        share, cell = sampler.largest_cell()
        for key in TRACE_KEYS[:4]:
            trace[key].append(stats[key])
        trace["max_share"].append(share)
        trace["max_cell"].append(cell)
        step_sizes.append(stats["step_size"])
        leapfrogs.append(stats["n_leapfrog"])
        sampling_index = sampler.iteration - args.warmup
        if sampling_index > 0 and sampling_index % args.thin == 0:
            draws.append(sampler.lam(sampler.x).detach().cpu().numpy())

        if sampler.iteration % args.log_every == 0:
            recent = slice(-args.log_every, None)
            elapsed = time.perf_counter() - started
            per = elapsed / (sampler.iteration - start_iteration)
            logger.info(
                "  iter %5d%s  step %.4g  leapfrog %4d  accept %.2f  divergent %3d  "
                "log pi median %.1f  max share median %.3g%%  %.1fs/iter",
                sampler.iteration, " (warmup)" if warming else "", stats["step_size"],
                stats["n_leapfrog"], np.mean(trace["accept_prob"][recent]),
                int(np.sum(trace["divergent"][recent])), np.median(stats["log_pi"]),
                100 * np.median(share), per,
            )
        if sampler.iteration % args.checkpoint == 0 or sampler.iteration == total:
            save()
    logger.info("HMC %s done: %.0fs this job; results in %s", name,
                time.perf_counter() - started, args.out)


# -- report ------------------------------------------------------------------


def quantile_line(x: np.ndarray, scale: float = 1.0, fmt: str = "{:.3f}") -> str:
    qs = np.quantile(x, [0.5, 0.9, 0.99, 1.0]) * scale
    return "  ".join(f"{label} {fmt.format(v)}" for label, v in zip(["median", "p90", "p99", "max"], qs))


def report(args: argparse.Namespace) -> None:
    import arviz as az

    name = fit_name(args)
    with np.load(args.out / f"{name}_trace.npz") as saved:
        trace = {key: saved[key] for key in saved.files}
    warmup = int(trace["warmup"])
    done = trace["log_pi"].shape[0]
    if done <= warmup:
        raise SystemExit(f"{name}: {done} iterations, all warmup; nothing to report yet")
    inputs, _, _, map_lam, _ = load_problem(args)
    post = slice(warmup, None)

    def chains_first(key: str) -> np.ndarray:
        return trace[key][post].T  # (chains, draws)

    lines = [f"HMC {name}, variance floor {args.variance_floor}, whitened by {args.vb_run}",
             f"{inputs.n_zones} zones x {inputs.n_units:,} units, N = {inputs.N:,.0f}; "
             f"{trace['log_pi'].shape[1]} chains, {warmup} warmup + {done - warmup} sampling "
             f"iterations"]

    divergent = chains_first("divergent")
    chain_accept = chains_first("accept_prob").mean(1)
    stuck = np.flatnonzero(chain_accept < 0.05)
    lines += [
        "",
        "1. Sampler",
        f"   step size {float(trace['step_size'][-1]):.4g}; leapfrog steps per iteration "
        f"median {np.median(trace['n_leapfrog'][post]):.0f}, max {trace['n_leapfrog'][post].max()}",
        f"   mean acceptance probability {trace['accept_prob'][post].mean():.3f}",
        f"   divergent transitions {int(divergent.sum()):,} of {divergent.size:,} "
        f"({100 * divergent.mean():.2f}%); per chain min {divergent.sum(1).min()} "
        f"max {divergent.sum(1).max()}",
        f"   chain mean acceptance min {chain_accept.min():.3f}, median {np.median(chain_accept):.3f}; "
        + (f"STUCK (under 5%): chains {', '.join(map(str, stuck))} -- everything below "
           f"includes them" if stuck.size else "no chain stuck"),
        "   E-BFMI per chain (below 0.3 is a warning sign): min {:.2f}, median {:.2f}".format(
            *np.quantile(az.bfmi(chains_first("energy")), [0.0, 0.5])),
    ]

    lines += ["", "2. Convergence (rank-normalised split R-hat; bulk and tail ESS)"]
    for key, label in (("log_pi", "log pi"), ("max_share", "largest-cell share")):
        values = chains_first(key)
        lines.append(
            f"   {label:<20} R-hat {float(az.rhat(values)):.3f}  "
            f"ESS bulk {float(az.ess(values, method='bulk')):,.0f}  "
            f"tail {float(az.ess(values, method='tail')):,.0f}"
        )
    if "lam" in trace and trace["lam"].shape[0] >= 4:
        lam = np.transpose(trace["lam"], (1, 0, 2))  # (chains, kept draws, m)
        # A hierarchy fixes some coordinates exactly (a lone block group's
        # deviation from its tract is 0): no diagnostics for those.
        varying = lam.reshape(-1, lam.shape[2]).std(axis=0) > 0
        dataset = az.convert_to_dataset({"lam": lam[:, :, varying]})
        rhat = np.full(lam.shape[2], np.nan)
        bulk, tail = rhat.copy(), rhat.copy()
        rhat[varying] = az.rhat(dataset)["lam"].values
        bulk[varying] = az.ess(dataset, method="bulk")["lam"].values
        tail[varying] = az.ess(dataset, method="tail")["lam"].values
        fixed = int((~varying).sum())
        rhat_v, bulk_v, tail_v = rhat[varying], bulk[varying], tail[varying]
        lines += [
            f"   lambda, {lam.shape[2]:,} coordinates from {lam.shape[1]} kept draws per chain"
            + (f" ({fixed:,} fixed by the hierarchy, left out):" if fixed else ":"),
            f"     R-hat   {quantile_line(rhat_v)}",
            f"     R-hat over 1.01: {int((rhat_v > 1.01).sum()):,}; over 1.1: {int((rhat_v > 1.1).sum()):,}",
            f"     ESS bulk min {bulk_v.min():,.0f}, median {np.median(bulk_v):,.0f}; "
            f"tail min {tail_v.min():,.0f}, median {np.median(tail_v):,.0f}",
            "     ten worst by R-hat:",
        ]
        table = constraint_table(inputs).assign(rhat=rhat, ess_bulk=bulk, ess_tail=tail,
                                               map_lam=map_lam)
        worst = table.sort_values("rhat", ascending=False).head(10)
        lines += ["     " + row for row in worst.to_string(index=False).splitlines()]
    else:
        lines.append("   lambda: too few kept draws for per-coordinate diagnostics")

    share = trace["max_share"][post].ravel()
    cell = trace["max_cell"][post].ravel()
    from pmedm_vb.solvers.base import weights_from_lambda

    map_max = float(weights_from_lambda(inputs, map_lam).max())
    lines += [
        "",
        "3. Each iteration's largest cell (one record in one block group)",
        f"   share of N: {quantile_line(share, 100, '{:.3f}%')}",
    ]
    units = inputs.units.iloc[:, 0].to_numpy()
    thresholds = [("MAP max", map_max, f"{map_max * inputs.N:,.1f}")] + [
        (f"{s:.0%} of N", s, f"{s * inputs.N:,.1f}") for s in SHARES
    ]
    for label, level, people in thresholds:
        over = share > level
        records = np.unique(cell[over] % inputs.n_units)
        lines.append(
            f"   iterations with a cell over {label} ({people}): {int(over.sum()):,} of "
            f"{share.size:,} ({100 * over.mean():.2f}%); records ever over it: {records.size:,}"
        )
    over = share > SHARES[0]
    if over.any():
        unit = cell[over] % inputs.n_units
        counts = pd.Series(unit).value_counts()
        top = pd.DataFrame({
            "unit": units[counts.index],
            "iterations > 1% of N": counts.to_numpy(),
            "max share of N": [share[over][unit == u].max() for u in counts.index],
        }).head(20)
        lines += ["   records over 1% of N, by iterations:"]
        lines += ["     " + row for row in top.to_string(index=False).splitlines()]

    text = "\n".join(lines) + "\n"
    path = args.out / f"{name}_report.txt"
    path.write_text(text)
    print(text)
    logger.info("report written to %s", path)


def main() -> None:
    args = parse_args()
    if args.step == "sample":
        sample(args)
    else:
        with contextlib.suppress(BrokenPipeError):
            report(args)


if __name__ == "__main__":
    main()
