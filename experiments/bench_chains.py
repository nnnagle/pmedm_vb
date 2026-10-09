"""How the cost of one HMC iteration grows with the number of chains, on one GPU.

Decides how many chains a many-short-chains run (nested R-hat, Margossian et
al. 2025, *Bayesian Analysis* 20) can afford. More chains are free only while
the GPU has spare capacity; past that, time per iteration grows in proportion
to the chains, and extra chains only multiply the warmup. Each gradient
multiplies every chain's state by the dense whitening matrix ``A``
(:mod:`~pmedm_vb.solvers.mcmc`), ``m^2`` doubles, which may already fill the
GPU at 32 chains; the target ``f`` holds one ``(n_zones, n_units)`` logit array
per chain, which bounds the chains by memory.

For each PUMA and each chain count, on the same problem ``run_mcmc.py`` samples
(the skewed VB fit's whitening), timed after two untimed repeats, median of
``--reps``:

- ``whiten_ms``: ``lambda = mean + x A'`` alone;
- ``target_ms``: ``n f(lambda)`` and its gradient in ``lambda``, without ``A``;
- ``gradient_ms``: one leapfrog's work, ``n f(mean + x A')`` and its gradient
  in ``x``;
- ``iteration_ms``: one :meth:`HMC.step` capped at ``--leapfrog`` steps, plus
  the largest-cell evaluation ``run_mcmc.py`` makes every iteration;
- ``peak_mb``: the largest GPU memory allocated during the iteration.

Per chain and per leapfrog step, ``iteration_ms / (chains x leapfrog)`` is the
figure to compare. A chain count that runs out of memory is recorded and ends
that PUMA's sweep. The states are standard normal draws, unscreened: the cost
does not depend on where the chains are. Writes ``<out>/bench_chains.csv``::

    $PY experiments/bench_chains.py --pumas 4701501 4701502 4701503 4701504 --alpha 0.01 \\
        --taper none --variance-floor zero --hierarchy puma \\
        --vb-root $PAPER/exp01_baseline --out $PAPER/bench_chains
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import logger, record_run

from run_mcmc import fit_name, load_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pumas", nargs="+", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", choices=["tract", "none"], default="none")
    parser.add_argument("--hierarchy", choices=["none", "tract", "puma", "nullspace"], default="puma")
    parser.add_argument("--rollup", default=None, help="as in run_mcmc.py")
    parser.add_argument("--variance-floor", default="zero", help="as in run_mcmc.py")
    parser.add_argument("--area", default="knox-2024-5yr", help="assembled-inputs directory name")
    parser.add_argument("--vb-root", type=Path, required=True,
                        help="experiment folder; the whitening fit is <vb-root>/<puma>/vb_skewed")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chains", type=int, nargs="+",
                        default=[32, 64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--leapfrog", type=int, default=10, help="leapfrog steps per timed iteration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from pmedm_vb.assemble.hierarchy import level_name

    args.hierarchy = level_name(args.hierarchy, None, args.rollup)
    return args


def timed(fn, device: str, reps: int) -> float:
    """Median milliseconds of ``fn()`` over ``reps`` calls, after two untimed ones."""
    import torch

    def sync():
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    for _ in range(2):
        fn()
    times = []
    for _ in range(reps):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times))


def bench_chains(args, target, n, mean, A, x0, settings) -> dict:
    """The timings for one chain count; everything it puts on the GPU is freed on return."""
    import torch

    from pmedm_vb.solvers.mcmc import HMC

    cuda = args.device.startswith("cuda")
    sampler = HMC(target, n, mean, A, x0, settings, seed=args.seed, device=args.device)
    x = sampler.x

    def target_only():
        lam = sampler.lam(x).detach().requires_grad_(True)
        U = n * target(lam)
        torch.autograd.grad(U.sum(), lam)

    def iteration():
        sampler.step(adapt=False)
        sampler.largest_cell()

    row = {}
    with torch.no_grad():
        row["whiten_ms"] = timed(lambda: sampler.lam(x), args.device, args.reps)
    row["target_ms"] = timed(target_only, args.device, args.reps)
    row["gradient_ms"] = timed(lambda: sampler.potential_and_gradient(x), args.device, args.reps)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    row["iteration_ms"] = timed(iteration, args.device, max(1, args.reps // 2))
    row["peak_mb"] = torch.cuda.max_memory_allocated() / 2**20 if cuda else float("nan")
    return row


def bench_puma(args: argparse.Namespace, puma: str) -> list[dict]:
    import torch

    from pmedm_vb.solvers.mcmc import HMCSettings, whitening_matrix
    from pmedm_vb.solvers.vb import _DualTarget

    problem = argparse.Namespace(**vars(args), puma=puma, vb_run=args.vb_root / puma / "vb_skewed")
    inputs, sigma, q, _, hierarchy = load_problem(problem)
    target = _DualTarget(inputs, sigma, args.device, hierarchy)
    A = whitening_matrix(q)
    m = q.size
    logger.info("%s (%s): m = %s, %d zones x %d units", puma, fit_name(problem), f"{m:,}",
                inputs.n_zones, inputs.n_units)
    # A step size this small hits the cap, so every timed iteration takes --leapfrog steps.
    settings = HMCSettings(step_size=1e-6, trajectory=1.0, max_leapfrog=args.leapfrog)
    rng = np.random.default_rng(args.seed)

    rows = []
    for chains in sorted(args.chains):
        row = {"puma": puma, "m": m, "n_zones": inputs.n_zones, "n_units": inputs.n_units,
               "chains": chains, "leapfrog": args.leapfrog}
        try:
            row.update(bench_chains(args, target, inputs.n, q.mean, A,
                                    rng.standard_normal((chains, m)), settings))
            row["us_per_chain_leapfrog"] = 1000 * row["iteration_ms"] / (chains * args.leapfrog)
            row["status"] = "ok"
        except torch.cuda.OutOfMemoryError:
            row["status"] = "out of memory"
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        rows.append(row)
        logger.info("%s, %5d chains: %s", puma, chains,
                    row["status"] if row["status"] != "ok" else
                    f"whiten {row['whiten_ms']:.2f} ms, target {row['target_ms']:.2f} ms, "
                    f"gradient {row['gradient_ms']:.2f} ms, iteration {row['iteration_ms']:.1f} ms, "
                    f"{row['us_per_chain_leapfrog']:.2f} us per chain-leapfrog, "
                    f"peak {row['peak_mb']:,.0f} MB")
        if row["status"] != "ok":
            break
    del A, target
    return rows


def main() -> None:
    import torch

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU %s, %.1f GB", props.name, props.total_memory / 2**30)
    rows = []
    for puma in args.pumas:
        rows += bench_puma(args, puma)
        pd.DataFrame(rows).to_csv(args.out / "bench_chains.csv", index=False)
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print(f"\nwrote {args.out / 'bench_chains.csv'}")


if __name__ == "__main__":
    main()
