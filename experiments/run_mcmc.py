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

``--scale-from`` adds a fixed diagonal scale to the whitening, ``lambda = mu +
A diag(s) x``, with ``s`` the ``sd`` array of an ``.npz`` (each whitened
coordinate's posterior sd, as ``whitened_mixing.py`` writes them). It is the
diagonal mass matrix HMC would learn, given in advance: an oracle that shows
what learning one could gain. The chains start at the same points as without
it (each start divided by ``s``), so only the sampler's geometry changes.

``--diag-from`` takes the same scale from an earlier run's trace instead of an
``.npz``: each whitened coordinate's sd over its kept draws, measured in *this*
run's whitening (``--vb-run``), for every coordinate -- the diagonal mass
matrix on top of the VB whitening, ``Sigma = A S^2 A'``, given in advance.
With ``--vb-run`` a Gaussian-only fit, that is the oracle for whitening by
Gaussian VB plus a learned diagonal. ``--diag-stat steep`` scales each
coordinate by the narrower of its two half-widths, ``min(q50 - q16, q84 - q50)``,
instead of its sd: the same for a symmetric coordinate, but a lopsided one is
scaled only to its steep side, which is what bounds the step size.

``--retry-factor a`` turns on delayed rejection
(:mod:`~pmedm_vb.solvers.mcmc`): a divergent transition is retried with step
``eps / a``; the trace records which chains retried, whether the retry was
accepted and the steps it cost, and the report summarises them.

``--locate K`` records, at every divergent transition, the ``K`` whitened
coordinates with the largest ``|gradient|`` at the trajectory's end; the
report lists the coordinates that come first most often.

``--adapt-diag`` learns that diagonal in warmup instead, in Stan's windows
(:func:`~pmedm_vb.solvers.mcmc.stan_windows`: ``--warmup-init`` iterations
adapting only the step size, then windows of ``--warmup-base``, twice that,
..., then ``--warmup-term``): at each window's end every whitened coordinate is
scaled by its sd over the window, pooled across chains. The state carries the
scale, so a resumed run continues with it.

**Many short chains** (``--superchains K``): ``--chains`` split into ``K``
superchains of ``M = chains / K`` that start at one point each -- ``K`` screened
draws from the VB Gaussian, times ``--init-inflate`` so that they are
overdispersed, shrunk from there by 0.95 at a time (not below 1) until enough
pass the ``--init-max-share`` screen
(:func:`~pmedm_vb.solvers.mcmc.overdispersed_starts`) -- the design of nested ``R-hat`` (Margossian et al. 2024,
*Bayesian Analysis*). Warmup is adaptive: ``--warmup-init`` step-size-only
iterations, then repeatedly a metric window (``--warmup-base``, doubling each
time), ``--warmup-term`` step-size iterations, and one iteration at the fixed
kernel, whose draw gives nested ``R-hat`` (one draw per chain) for ``log pi``,
the largest cell's share and the 99th percentile of ``|lambda|``; warmup ends
at the first check where all three are under ``sqrt(1 + 1/M + tau)``, ``tau``
from ``--nested-tau`` or, by default, the level ``K`` superchains can resolve
(:func:`~pmedm_vb.solvers.mcmc.nested_tau`, a ``--nested-alpha`` false-alarm
rate split over the three), or at ``--warmup`` iterations, flagged as not
converged. That draw is the first of ``ceil(--draws / chains)`` sampling
iterations, all kept; sampling is extended by as many again (up to
``--max-extend`` times) while a monitored quantity's bulk or tail ESS is under
``--ess-target``. No checkpoints: a run that stops is rerun.

``--block-from`` goes one step further, from an earlier run's trace: every
coordinate is scaled by its sd in that trace's kept draws, as above, except
the coordinates with sd over ``--block-sd``, which are mapped by the Cholesky
factor ``L`` of their covariance there instead, ``lambda = mu + A M x`` with
``M = diag(s)`` but ``L`` on the wide block. Their correlation (up to 0.9 in
the reference runs) leaves directions much narrower than 1 after a diagonal
scale, which may be what halved the step size with ``--scale-from``. The
starts are again unchanged (``M x0`` is the unscaled start).
"""

from __future__ import annotations

import argparse
import contextlib
import math
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
#: Per-iteration delayed-rejection records (--retry-factor), each (iterations, chains).
RETRY_KEYS = ("retried", "accepted2")
#: Per-divergence records (--locate): iteration, chain, and the top coordinates and |gradient|s.
LOCATED_KEYS = ("div_iter", "div_chain", "div_coords", "div_grad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["sample", "report"])
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", choices=["tract", "none"], default="tract")
    parser.add_argument("--hierarchy", choices=["none", "tract", "puma", "nullspace"], default="none",
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
    parser.add_argument("--scale-from", type=Path, default=None,
                        help="an .npz whose 'sd' array (one per whitened coordinate) scales the "
                             "whitening's columns: lambda = mean + A diag(sd) x")
    parser.add_argument("--diag-from", type=Path, default=None,
                        help="an earlier run's _trace.npz: scale every whitened coordinate (in "
                             "this run's whitening) by its sd there; excludes --scale-from and "
                             "--block-from")
    parser.add_argument("--diag-stat", choices=["sd", "steep"], default="sd",
                        help="--diag-from's scale: each coordinate's sd, or the narrower of its "
                             "two half-widths min(q50 - q16, q84 - q50)")
    parser.add_argument("--retry-factor", type=float, default=0.0,
                        help="delayed rejection: retry divergent transitions with step size "
                             "divided by this (0: off); e.g. 4")
    parser.add_argument("--locate", type=int, default=0,
                        help="at each divergence, record this many whitened coordinates with "
                             "the largest |gradient| at the trajectory's end")
    parser.add_argument("--block-from", type=Path, default=None,
                        help="an earlier run's _trace.npz: scale each whitened coordinate by its "
                             "sd there, and the wide ones (--block-sd) by the Cholesky factor of "
                             "their covariance; excludes --scale-from")
    parser.add_argument("--block-sd", type=float, default=2.0,
                        help="whitened sd above which --block-from treats a coordinate as wide")
    parser.add_argument("--init-max-share", type=float, default=0.01,
                        help="largest share of p on one cell a starting point may have")
    parser.add_argument("--init-inflate", type=float, default=1.0,
                        help="starting points are VB Gaussian draws times this (overdispersed > 1); "
                             "with --superchains, the most it is shrunk from while starts fail the screen")
    parser.add_argument("--adapt-diag", action="store_true",
                        help="learn a diagonal on top of the whitening in warmup windows; "
                             "excludes --scale-from, --diag-from and --block-from")
    parser.add_argument("--warmup-init", type=int, default=75,
                        help="warmup iterations adapting only the step size before the first window")
    parser.add_argument("--warmup-base", type=int, default=25, help="the first metric window's length")
    parser.add_argument("--warmup-term", type=int, default=50,
                        help="step-size-only iterations after the (each, with --superchains) last window")
    parser.add_argument("--superchains", type=int, default=0,
                        help="many short chains: split --chains into this many superchains and end "
                             "warmup by nested R-hat (implies --adapt-diag; --warmup is its cap)")
    parser.add_argument("--draws", type=int, default=4000,
                        help="--superchains: total kept draws, ceil(draws / chains) per chain")
    parser.add_argument("--ess-target", type=float, default=400,
                        help="--superchains: extend sampling while a monitor's bulk or tail ESS is under this")
    parser.add_argument("--max-extend", type=int, default=3,
                        help="--superchains: at most this many extensions")
    parser.add_argument("--nested-tau", type=float, default=None,
                        help="--superchains: tolerated nonstationary variance (default: what K "
                             "superchains can resolve at --nested-alpha)")
    parser.add_argument("--nested-alpha", type=float, default=0.05,
                        help="--superchains: false-alarm rate at stationarity for the default tau, "
                             "split over the monitored quantities")
    parser.add_argument("--checkpoint", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    parser.add_argument("--collapse", type=float, default=None,
                        help="collapse each area's zero cells into one and merge small positive "
                             "cells until each holds more than this many (pmedm_vb.assemble."
                             "collapse); the hierarchy level becomes <level>-c<threshold>")
    parser.add_argument("--rollup", default=None,
                        help="the structure-aware roll-up (pmedm_vb.assemble.rollup) with this "
                             "threshold, PERSONS or PERSONShHOUSEHOLDS (e.g. 15h10); the "
                             "hierarchy level becomes <level>-r<spec>")
    parser.add_argument("--share-cap", type=float, default=None,
                        help="the soft cap on each cell's share of the population, w / N "
                             "(pmedm_vb.assemble.sharecap), e.g. 0.005; the hierarchy level gains "
                             "+share<share>x<strength>")
    parser.add_argument("--cap-strength", type=float, default=None,
                        help="the cap's strength tau (required with --share-cap)")
    parser.add_argument("--prior-sd", type=float, default=None,
                        help="the Gaussian prior on the multipliers, zeta ~ N(0, s^2) "
                             "(pmedm_vb.assemble.prior), e.g. 3; the hierarchy level gains "
                             "+prior<sd>")
    parser.add_argument("--tweedie-weight", type=float, default=None,
                        help="a Tweedie error for the rows published as zero, with amounts of "
                             "this mean, the average weight w (pmedm_vb.assemble.tweedie), e.g. 18; "
                             "the hierarchy level gains +tweedie<w>")
    args = parser.parse_args()
    given = [name for name in ("scale_from", "block_from", "diag_from")
             if getattr(args, name, None) is not None]
    if getattr(args, "superchains", 0):
        args.adapt_diag = True
        if args.chains % args.superchains:
            parser.error(f"--chains {args.chains} does not split into --superchains {args.superchains}")
    if getattr(args, "adapt_diag", False):
        given.append("adapt_diag")
    if len(given) > 1:
        parser.error("--scale-from, --block-from, --diag-from and --adapt-diag are alternatives; give one")
    from pmedm_vb.assemble.hierarchy import level_name
    from pmedm_vb.assemble.prior import prior_level
    from pmedm_vb.assemble.sharecap import cap_level
    from pmedm_vb.assemble.tweedie import tweedie_level

    if getattr(args, "hierarchy", None) is not None:
        args.hierarchy = tweedie_level(
            prior_level(cap_level(level_name(args.hierarchy, args.collapse, args.rollup),
                                  args.share_cap, args.cap_strength), args.prior_sd),
            args.tweedie_weight)
    return args


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
    floor = floor_spec(args.variance_floor)
    hierarchy = Hierarchy.build(inputs, args.hierarchy, alpha=args.alpha, taper=taper,
                                variance_floor=floor)
    sigma = hierarchy.sigma(inputs, args.alpha, taper, floor)
    vb_path = args.vb_run / f"{fit_name(args)}.npz"
    q, _, _ = load_vb(vb_path)
    with np.load(vb_path) as saved:
        map_lam = saved["map_lam"]
        saved_level = str(saved["hierarchy"]) if "hierarchy" in saved.files else "none"
    if saved_level != args.hierarchy:
        raise SystemExit(f"{vb_path} was fitted with hierarchy={saved_level}, not {args.hierarchy}")
    return inputs, sigma, q, map_lam, hierarchy


# -- sample ------------------------------------------------------------------


def whitening_scale(path: Path | None, size: int) -> np.ndarray | None:
    """The diagonal scale ``s`` from ``path``'s ``sd`` array; 1 where an sd is
    not positive (a coordinate the hierarchy fixes)."""
    if path is None:
        return None
    with np.load(path) as saved:
        sd = np.asarray(saved["sd"], dtype=float)
    if sd.shape != (size,):
        raise SystemExit(f"{path}: 'sd' has shape {sd.shape}, the whitening has {size} coordinates")
    return np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)


def whitened_draws(path: Path, q) -> np.ndarray:
    """An earlier run's kept draws at ``path``, ``(draws, m)``, in ``q``'s
    whitened coordinates ``x = G' (xi - mu)``."""
    with np.load(path) as saved:
        key = "xi" if "xi" in saved.files else "lam"
        draws = saved[key]
    m = draws.shape[2]
    if m != q.size:
        raise SystemExit(f"{path}: draws have {m} coordinates, the whitening has {q.size}")
    return q.whiten((draws.reshape(-1, m) - q.mean).T).T


def block_metric(path: Path, q, wide_sd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(s, wide, L)`` from an earlier run's trace at ``path``: each whitened
    coordinate's sd over its kept draws (1 where it does not vary), the
    coordinates with sd over ``wide_sd``, and the Cholesky factor of their
    covariance, all in ``q``'s whitening."""
    x = whitened_draws(path, q)
    kept_chains = x.shape[0]
    sd = x.std(axis=0)
    wide = sd > wide_sd
    cov = np.cov(x[:, wide], rowvar=False).reshape(int(wide.sum()), -1)
    L = np.linalg.cholesky(cov) if wide.any() else np.zeros((0, 0))
    corr = cov / np.outer(sd[wide], sd[wide]) if wide.any() else cov
    eig = np.linalg.eigvalsh(corr) if wide.any() else np.array([np.nan])
    logger.info("block metric from %s (%d draws): %d coordinates with sd over %g; their "
                "correlation's eigenvalues min %.3g, max %.3g", path, kept_chains,
                int(wide.sum()), wide_sd, eig[0], eig[-1])
    return np.where(sd > 0, sd, 1.0), wide, L


def metric_name(args: argparse.Namespace) -> str:
    """What the whitening is scaled by, for the trace and the report ('' for nothing)."""
    if getattr(args, "superchains", 0):
        return (f"nested {args.superchains} x {args.chains // args.superchains}, diagonal learned "
                f"in warmup ({args.warmup_init}+{args.warmup_base}x2^k+{args.warmup_term})")
    if getattr(args, "adapt_diag", False):
        return (f"diagonal learned in warmup ({args.warmup_init}+{args.warmup_base}x2^k+"
                f"{args.warmup_term})")
    if getattr(args, "block_from", None) is not None:
        return f"block {args.block_from} sd>{args.block_sd:g}"
    if getattr(args, "diag_from", None) is not None:
        return f"diag {args.diag_from}" + ("" if args.diag_stat == "sd" else f" {args.diag_stat}")
    return str(args.scale_from or "")


def draw_arrays(draws: np.ndarray, hierarchy) -> dict[str, np.ndarray]:
    """Kept draws for the trace: ``lam`` always in today's layout (the
    multipliers the data see), plus ``xi`` under a hierarchy, for densities."""
    if hierarchy.is_trivial:
        return {"lam": draws}
    kept, chains, size = draws.shape
    lam = hierarchy.lambda_data(draws.reshape(-1, size).T).T.reshape(kept, chains, -1)
    return {"lam": lam, "xi": draws}


def sample(args: argparse.Namespace) -> None:
    import torch

    from pmedm_vb.solvers.mcmc import HMC, HMCSettings, stan_windows, starting_points, whitening_matrix
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
    x0 = starting_points(target, q.mean, A, args.chains, rng, args.init_max_share, device,
                         inflate=args.init_inflate)
    scale = whitening_scale(args.scale_from, q.size)
    if scale is not None:
        A = A * scale            # columns: lambda = mean + A diag(s) x
        x0 = x0 / scale          # the same starting points as without the scale
        logger.info("whitening scaled by %s: s median %.3g, max %.3g", args.scale_from,
                    np.median(scale), scale.max())
    if args.diag_from is not None:
        x = whitened_draws(args.diag_from, q)
        if args.diag_stat == "sd":
            sd = x.std(axis=0)
        else:
            q16, q50, q84 = np.quantile(x, [0.16, 0.5, 0.84], axis=0)
            sd = np.minimum(q50 - q16, q84 - q50)
        del x
        scale = np.where(sd > 0, sd, 1.0)
        A = A * scale            # Sigma = A S^2 A', every coordinate scaled
        x0 = x0 / scale
        logger.info("diagonal (%s) from %s: s p10 %.3g, median %.3g, p90 %.3g, max %.3g",
                    args.diag_stat, args.diag_from, *np.quantile(scale, [0.1, 0.5, 0.9, 1.0]))
    if args.block_from is not None:
        from scipy.linalg import solve_triangular

        s, wide, L = block_metric(args.block_from, q, args.block_sd)
        scaled = A * s                               # lambda = mean + A M x, M = diag(s) ...
        scaled[:, wide] = A[:, wide] @ L             # ... with L on the wide block
        x0 = x0 / s
        x0[:, wide] = solve_triangular(L, (x0[:, wide] * s[wide]).T, lower=True).T
        A = scaled
    settings = HMCSettings(
        step_size=args.step_size, trajectory=args.trajectory,
        target_accept=args.target_accept, max_leapfrog=args.max_leapfrog, locate=args.locate,
        retry_factor=args.retry_factor,
    )
    sampler = HMC(target, inputs.n, q.mean, A, x0, settings, seed=args.seed, device=device)
    windows = (stan_windows(args.warmup, args.warmup_init, args.warmup_term, args.warmup_base)
               if args.adapt_diag else [])
    if args.adapt_diag:
        logger.info("diagonal learned in %d windows: %s", len(windows), windows)

    trace = {key: [] for key in TRACE_KEYS}
    step_sizes, leapfrogs, draws = [], [], []
    located = {key: [] for key in LOCATED_KEYS}   # one entry per divergent transition
    retries = {key: [] for key in RETRY_KEYS}
    extra_leapfrogs = []
    earlier_seconds, warmup_seconds = 0.0, float("nan")
    if state_path.exists() and trace_path.exists():
        with np.load(state_path) as state:
            sampler.load_state_dict(dict(state))
        with np.load(trace_path) as saved:
            trace = {key: list(saved[key]) for key in TRACE_KEYS}
            step_sizes, leapfrogs = list(saved["step_size"]), list(saved["n_leapfrog"])
            saved_retry = float(saved["retry_factor"]) if "retry_factor" in saved.files else 0.0
            if saved_retry != args.retry_factor:
                raise SystemExit(f"{trace_path} was sampled with --retry-factor {saved_retry:g}, "
                                 f"not {args.retry_factor:g}")
            if args.retry_factor > 1:
                retries = {key: list(saved[key]) for key in RETRY_KEYS}
                extra_leapfrogs = list(saved["extra_leapfrog"])
            key = "lam" if hierarchy.is_trivial else "xi"
            draws = list(saved[key]) if key in saved.files else []
            earlier_seconds = float(saved["seconds"]) if "seconds" in saved.files else 0.0
            if "warmup_seconds" in saved.files:
                warmup_seconds = float(saved["warmup_seconds"])
            located = {key: list(saved[key]) if key in saved.files else [] for key in LOCATED_KEYS}
            saved_metric = next((str(saved[k]) for k in ("metric", "scale_from")
                                 if k in saved.files), "")
        if saved_metric != metric_name(args):
            raise SystemExit(f"{trace_path} was sampled with metric {saved_metric or 'none'!r}, "
                             f"not {metric_name(args) or 'none'!r}")
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
            retry_factor=args.retry_factor,
            **({**{key: np.asarray(value) for key, value in retries.items()},
                "extra_leapfrog": np.asarray(extra_leapfrogs)} if args.retry_factor > 1 else {}),
            **({key: np.asarray(value) for key, value in located.items()}
               if located["div_iter"] else {}),
            warmup=args.warmup, thin=args.thin, n_zones=inputs.n_zones, n_units=inputs.n_units,
            seconds=earlier_seconds + time.perf_counter() - started,
            warmup_seconds=warmup_seconds, metric=metric_name(args),
            scale=sampler.scale.cpu().numpy(),
            **(draw_arrays(np.asarray(draws), hierarchy) if draws else {}),
        )

    while sampler.iteration < total:
        warming = sampler.iteration < args.warmup
        stats = sampler.step(adapt=warming)
        done = sampler.iteration                     # the step just taken had index done - 1
        for start, end in windows:
            if start <= done - 1 < end:
                sampler.accumulate()
                if done == end:
                    new = sampler.adapt_scale()
                    logger.info("  window %d-%d: diagonal p10 %.3g, median %.3g, p90 %.3g, max %.3g",
                                start, end, *np.quantile(new, [0.1, 0.5, 0.9, 1.0]))
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
        if "div_chain" in stats:
            located["div_iter"].extend([sampler.iteration] * len(stats["div_chain"]))
            located["div_chain"].extend(stats["div_chain"])
            located["div_coords"].extend(stats["div_coords"])
            located["div_grad"].extend(stats["div_grad"])
        leapfrogs.append(stats["n_leapfrog"])
        if args.retry_factor > 1:
            for key in RETRY_KEYS:
                retries[key].append(stats[key])
            extra_leapfrogs.append(stats["extra_leapfrog"])
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


#: The quantities nested R-hat and the ESS check monitor in a --superchains run.
MONITORS = ("log_pi", "max_share", "lam_q99")


def sample_nested(args: argparse.Namespace) -> None:
    """Many short chains: superchains, adaptive warmup ended by nested R-hat,
    then just enough sampling (see the module docstring)."""
    import torch

    from pmedm_vb.compare import import_arviz
    from pmedm_vb.solvers.mcmc import (HMC, HMCSettings, nested_rhat, nested_tau, nested_threshold,
                                       overdispersed_starts, whitening_matrix)
    from pmedm_vb.solvers.vb import _DualTarget

    az = import_arviz()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    name = fit_name(args)
    trace_path = args.out / f"{name}_trace.npz"
    if trace_path.exists():
        with np.load(trace_path) as saved:
            if "nested_done" in saved.files:
                logger.info("%s is done; nothing to sample", trace_path)
                return

    k, chains = args.superchains, args.chains
    m_per = chains // k
    inputs, sigma, q, _, hierarchy = load_problem(args)
    target = _DualTarget(inputs, sigma, device, hierarchy)
    A = whitening_matrix(q)
    rng = np.random.default_rng(args.seed)
    starts, inflation, passed = overdispersed_starts(target, q.mean, A, k, rng, args.init_max_share,
                                                     device, inflate=args.init_inflate)
    logger.info("starts: VB Gaussian draws x%.3g (asked x%g; %.0f%% of that batch passed the "
                "%.1f%% screen)", inflation, args.init_inflate, 100 * passed, 100 * args.init_max_share)
    x0 = np.repeat(starts, m_per, axis=0)             # chain c is in superchain c // M
    settings = HMCSettings(
        step_size=args.step_size, trajectory=args.trajectory,
        target_accept=args.target_accept, max_leapfrog=args.max_leapfrog, locate=args.locate,
    )
    sampler = HMC(target, inputs.n, q.mean, A, x0, settings, seed=args.seed, device=device)
    tau = (args.nested_tau if args.nested_tau is not None
           else nested_tau(k, m_per, args.nested_alpha, len(MONITORS)))
    threshold = nested_threshold(m_per, tau)
    logger.info("HMC %s on %s: %s multipliers, %d superchains x %d chains, starts x%.3g; nested "
                "R-hat threshold %.4f (tau %.4g); warmup at most %d, %d draws wanted", name, device,
                f"{q.size:,}", k, m_per, inflation, threshold, tau, args.warmup, args.draws)

    trace = {key: [] for key in TRACE_KEYS}
    step_sizes, leapfrogs, draws, monitors = [], [], [], []
    located = {key: [] for key in LOCATED_KEYS}
    checks = []                                      # (iteration, rhat per monitor, max over x, count over)
    started = time.perf_counter()

    def monitor_values() -> np.ndarray:
        with torch.no_grad():
            lam = sampler.lam(sampler.x)
            q99 = torch.quantile(lam.abs(), 0.99, dim=1)
        share, _ = sampler.largest_cell()
        return np.stack([(-sampler.U).cpu().numpy(), share, q99.cpu().numpy()], axis=1)

    def advance(adapt: bool, window: bool = False, keep: bool = False) -> None:
        stats = sampler.step(adapt=adapt)
        if window:
            sampler.accumulate()
        share, cell = sampler.largest_cell()
        for key in TRACE_KEYS[:4]:
            trace[key].append(stats[key])
        trace["max_share"].append(share)
        trace["max_cell"].append(cell)
        step_sizes.append(stats["step_size"])
        leapfrogs.append(stats["n_leapfrog"])
        if "div_chain" in stats:
            located["div_iter"].extend([sampler.iteration] * len(stats["div_chain"]))
            located["div_chain"].extend(stats["div_chain"])
            located["div_coords"].extend(stats["div_coords"])
            located["div_grad"].extend(stats["div_grad"])
        if keep:
            draws.append(sampler.lam(sampler.x).detach().cpu().numpy())
            monitors.append(monitor_values())
        if sampler.iteration % args.log_every == 0:
            logger.info("  iter %5d  step %.4g  leapfrog %4d  accept %.2f  divergent %3d  "
                        "log pi median %.1f", sampler.iteration, stats["step_size"],
                        stats["n_leapfrog"], float(np.mean(stats["accept_prob"])),
                        int(np.sum(stats["divergent"])), float(np.median(stats["log_pi"])))

    # Warmup: step size alone, then window / step size / check until nested R-hat passes.
    for _ in range(args.warmup_init):
        advance(adapt=True)
    size, converged = args.warmup_base, False
    while True:
        for _ in range(size):
            advance(adapt=True, window=True)
        new = sampler.adapt_scale()
        for _ in range(args.warmup_term):
            advance(adapt=True)
        sampler.end_warmup()
        advance(adapt=False, keep=True)              # the check: one draw at the fixed kernel
        rhat = nested_rhat(monitors[-1][None], k)
        with torch.no_grad():
            u = (sampler.x * sampler.scale).cpu().numpy()
        rhat_x = nested_rhat(u[None], k)
        checks.append((sampler.iteration, rhat, float(rhat_x.max()), int((rhat_x > threshold).sum())))
        logger.info("  check at %d (window %d; diagonal median %.3g, max %.3g; step %.4g): nested "
                    "R-hat %s (threshold %.4f); whitened coordinates max %.4f, %d over",
                    sampler.iteration, size, np.median(new), new.max(), sampler.step_size,
                    ", ".join(f"{label} {v:.4f}" for label, v in zip(MONITORS, rhat)), threshold,
                    rhat_x.max(), int((rhat_x > threshold).sum()))
        if bool((rhat <= threshold).all()):
            converged = True
            break
        size *= 2
        if sampler.iteration + size + args.warmup_term + 1 > args.warmup:
            logger.warning("warmup cap %d reached without nested R-hat passing; sampling anyway",
                           args.warmup)
            break
        draws.pop()
        monitors.pop()                               # a failed check's draw is warmup
        sampler.restart_step_size()
    warmup = sampler.iteration - 1
    warmup_seconds = time.perf_counter() - started

    # Sampling: the check's draw and enough more for --draws; extended while ESS is short.
    per_chain = math.ceil(args.draws / chains)
    for _ in range(per_chain - 1):
        advance(adapt=False, keep=True)
    extensions = 0
    while True:
        values = np.asarray(monitors)                 # (draws, chains, monitors)
        bulk = [float(az.ess(values[:, :, j].T, method="bulk")) for j in range(len(MONITORS))]
        tail = [float(az.ess(values[:, :, j].T, method="tail")) for j in range(len(MONITORS))]
        logger.info("  sampling %d per chain: ESS bulk %s, tail %s", values.shape[0],
                    ", ".join(f"{v:,.0f}" for v in bulk), ", ".join(f"{v:,.0f}" for v in tail))
        if min(bulk + tail) >= args.ess_target or extensions >= args.max_extend:
            break
        extensions += 1
        for _ in range(per_chain):
            advance(adapt=False, keep=True)

    np.savez(
        trace_path,
        **{key: np.asarray(value) for key, value in trace.items()},
        step_size=np.asarray(step_sizes), n_leapfrog=np.asarray(leapfrogs), retry_factor=0.0,
        **({key: np.asarray(value) for key, value in located.items()} if located["div_iter"] else {}),
        warmup=warmup, thin=1, n_zones=inputs.n_zones, n_units=inputs.n_units,
        seconds=time.perf_counter() - started, warmup_seconds=warmup_seconds,
        metric=metric_name(args), scale=sampler.scale.cpu().numpy(),
        superchains=k, nested_tau=tau, nested_threshold=threshold, nested_converged=converged,
        init_inflation=inflation, init_passed=passed,
        check_iter=np.asarray([c[0] for c in checks]), check_rhat=np.asarray([c[1] for c in checks]),
        check_rhat_x_max=np.asarray([c[2] for c in checks]),
        check_x_over=np.asarray([c[3] for c in checks]), monitors=np.asarray(MONITORS),
        monitor_draws=np.asarray(monitors), ess_bulk=np.asarray(bulk), ess_tail=np.asarray(tail),
        extensions=extensions, nested_done=True,
        **draw_arrays(np.asarray(draws), hierarchy),
    )
    logger.info("HMC %s done: warmup %d (%s), %d sampling iterations, %.0fs; results in %s", name,
                warmup, "nested R-hat passed" if converged else "NOT converged",
                np.asarray(monitors).shape[0], time.perf_counter() - started, args.out)


# -- report ------------------------------------------------------------------


def quantile_line(x: np.ndarray, scale: float = 1.0, fmt: str = "{:.3f}") -> str:
    qs = np.quantile(x, [0.5, 0.9, 0.99, 1.0]) * scale
    return "  ".join(f"{label} {fmt.format(v)}" for label, v in zip(["median", "p90", "p99", "max"], qs))


def locate_lines(trace: dict, warmup: int, top: int = 10) -> list[str]:
    """Where the divergences happened (``--locate``): the whitened coordinates
    that carry the largest ``|gradient|`` at a divergent trajectory's end most
    often, in warmup and sampling together and in sampling alone."""
    coords, grads, iters = trace["div_coords"], trace["div_grad"], trace["div_iter"]
    lines = [f"   divergences located ({coords.shape[0]:,}, {coords.shape[1]} coordinates each; "
             "whitened coordinate: times first / times in the top "
             f"{coords.shape[1]} / median |gradient| when first):"]
    for label, mask in (("all", np.ones(iters.size, bool)), ("sampling", iters > warmup)):
        if not mask.any():
            lines.append(f"     {label}: none")
            continue
        first = coords[mask, 0]
        values, counts = np.unique(first, return_counts=True)
        order = np.argsort(-counts)[:top]
        anywhere = np.bincount(coords[mask].ravel(), minlength=int(values.max()) + 1)
        parts = [f"{int(values[i])}: {counts[i]}/{anywhere[values[i]]}/"
                 f"{np.median(grads[mask, 0][first == values[i]]):.3g}" for i in order]
        share = counts[order].sum() / mask.sum()
        lines.append(f"     {label} ({int(mask.sum()):,}; these {len(order)} are first in "
                     f"{share:.0%}): " + ", ".join(parts))
    return lines


def report(args: argparse.Namespace) -> None:
    from pmedm_vb.compare import import_arviz

    az = import_arviz()

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

    lines = [f"HMC {name}, variance floor {args.variance_floor}, whitened by {args.vb_run}"
             + (f", scaled by {metric_name(args)}" if metric_name(args) else ""),
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
    if "retried" in trace:
        retried, taken = chains_first("retried"), chains_first("accepted2")
        extra = trace["extra_leapfrog"][post].sum()
        lines.append(
            f"   delayed rejection (step / {float(trace['retry_factor']):g}): {int(retried.sum()):,} "
            f"retries ({100 * retried.mean():.2f}% of transitions), "
            f"{100 * taken.sum() / max(1, retried.sum()):.1f}% of them accepted; retries and their "
            f"ghosts took {int(extra):,} extra leapfrog steps per chain "
            f"({100 * extra / trace['n_leapfrog'][post].sum():.0f}% of the first stage's)")
    if "div_coords" in trace:
        lines += locate_lines(trace, warmup)

    if "superchains" in trace:
        k = int(trace["superchains"])
        names = [str(v) for v in trace["monitors"]]
        lines += [
            "",
            f"1b. Many short chains: {k} superchains x {trace['log_pi'].shape[1] // k} chains; nested "
            f"R-hat threshold {float(trace['nested_threshold']):.4f} (tau {float(trace['nested_tau']):.4g}); "
            + (f"starts x{float(trace['init_inflation']):.3g} "
               f"({100 * float(trace['init_passed']):.0f}% passed the screen); "
               if "init_inflation" in trace else "")
            + f"warmup {warmup} iterations, "
            + ("ended by nested R-hat" if bool(trace["nested_converged"]) else "NOT CONVERGED (cap reached)")
            + f"; {done - warmup} sampling iterations ({int(trace['extensions'])} extensions)",
            "   checks (iteration: nested R-hat per monitor; whitened coordinates max, number over):",
        ]
        for it, rh, mx, over in zip(trace["check_iter"], trace["check_rhat"], trace["check_rhat_x_max"],
                                    trace["check_x_over"]):
            lines.append(f"     {int(it):5d}: " + ", ".join(f"{n} {v:.4f}" for n, v in zip(names, rh))
                         + f"; x max {float(mx):.4f}, {int(over):,} over")
        lines.append("   ESS over the sampling draws (bulk / tail): " + ", ".join(
            f"{n} {b:,.0f} / {t:,.0f}" for n, b, t in zip(names, trace["ess_bulk"], trace["ess_tail"])))

    lines += ["", "2. Convergence (rank-normalised split R-hat; bulk and tail ESS)"]
    if "superchains" in trace:
        lines.append(f"   (with {done - warmup} draws per chain split R-hat is not a convergence test -- "
                     "it needs long chains (Margossian et al. 2024, figure 1); nested R-hat in 1b is. "
                     "The ESS counts every chain.)")
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
        (sample_nested if args.superchains else sample)(args)
    else:
        with contextlib.suppress(BrokenPipeError):
            report(args)


if __name__ == "__main__":
    main()
