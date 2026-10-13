"""Does delayed-rejection HMC sample the right distribution? A check on a known target.

The target in ``d`` dimensions has potential

.. math::

    U(x) = \\tfrac12 |x|^2 + c \\sum_j e^{k\\, w_j' x},

``w_1 ... w_J`` orthonormal: a standard normal with an exponential wall along
each ``w_j`` at ``t = w_j' x`` about ``--wall-at`` (``c = e^{-k t_wall}``) --
the shape of the PMEDM walls, a log-concave density whose curvature jumps
from 1 to about ``k^2`` there. The coordinates factorise, so each ``t_j`` has
density proportional to ``exp(-t^2/2 - c e^{k t})``, whose moments are computed
exactly by quadrature, and any direction orthogonal to the walls is standard
normal.

Four runs of :class:`~pmedm_vb.solvers.mcmc.HMC` (identity whitening):

- ``hmc``: plain HMC, step size adapted in warmup;
- ``dr``: delayed rejection (``--retry-factor``), step size adapted;
- ``hmc fixed``, ``dr fixed``: the step size fixed at ``--fixed-step`` (too
  large for the walls) and the divergence threshold at ``--fixed-divergence``
  nats, so that ``dr fixed`` retries often -- the hardest test of the
  delayed-rejection acceptance and its ghost rule.

Every chain starts at an exact draw (the wall coordinates by inverse CDF on
the quadrature grid, the rest standard normal), so the check tests that the
samplers leave the target invariant, apart from how fast they converge; a
chain started far behind a wall would never move, which looks like bias.

For each run and each wall ``j``: ``E[t]``, ``E[t^2]`` and ``P(t > q)`` with
``q`` the exact 90th percentile, and for one direction orthogonal to the walls
``E[z]`` and ``E[z^2]``; each against its exact value, with ``z`` = (estimate -
exact) / Monte Carlo SE, the SE from the ESS of the mean (arviz). A correct
sampler gives ``|z|`` mostly under 2 and none far over 4.

    $PY experiments/check_drhmc.py --device cuda      # or cpu; a few minutes
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--walls", type=int, default=3)
    parser.add_argument("--k", type=float, default=8.0, help="wall steepness")
    parser.add_argument("--wall-at", type=float, default=1.5)
    parser.add_argument("--chains", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--trajectory", type=float, default=3.0)
    parser.add_argument("--retry-factor", type=float, default=4.0)
    parser.add_argument("--fixed-step", type=float, default=0.6)
    parser.add_argument("--fixed-divergence", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="also write the table here (.txt)")
    return parser.parse_args()


def wall_grid(c: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """The quadrature grid and the normalised density of ``t`` on it."""
    t = np.linspace(-12.0, 12.0, 400_001)
    log_d = -0.5 * t * t - c * np.exp(np.minimum(k * t, 700.0))
    d = np.exp(log_d - log_d.max())
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return t, d / trapezoid(d, t)


def exact_draws(c: float, k: float, size: tuple, rng: np.random.Generator) -> np.ndarray:
    """Draws of ``t`` by inverse CDF on the quadrature grid."""
    t, d = wall_grid(c, k)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))])
    return np.interp(rng.uniform(size=size) * cdf[-1], cdf, t)


def exact_moments(c: float, k: float) -> dict[str, float]:
    """``E[t]``, ``E[t^2]``, the 90th percentile ``q`` and ``P(t > q)`` for the
    density proportional to ``exp(-t^2/2 - c e^{kt})``, by quadrature."""
    t = np.linspace(-12.0, 12.0, 400_001)
    log_d = -0.5 * t * t - c * np.exp(np.minimum(k * t, 700.0))
    d = np.exp(log_d - log_d.max())
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    d /= trapezoid(d, t)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t))])
    q = float(np.interp(0.9, cdf, t))
    return {"mean": float(trapezoid(t * d, t)), "second": float(trapezoid(t * t * d, t)),
            "q90": q, "tail": 0.1}


class WallTarget:
    """``U(x)`` above, for a batch of rows of ``x``."""

    def __init__(self, W, c: float, k: float) -> None:
        self.W, self.c, self.k = W, c, k

    def __call__(self, x):
        import torch

        return 0.5 * (x * x).sum(1) + self.c * torch.exp(self.k * (x @ self.W)).sum(1)


def run(args, name: str, retry: float, fixed: bool, W, u, c: float) -> tuple[dict, np.ndarray]:
    import torch

    from pmedm_vb.solvers.mcmc import HMC, HMCSettings
    from pmedm_vb.solvers.vb import DTYPE

    settings = HMCSettings(
        step_size=args.fixed_step if fixed else 0.3, trajectory=args.trajectory,
        divergence=args.fixed_divergence if fixed else 1000.0, retry_factor=retry)
    rng = np.random.default_rng(args.seed)
    z = rng.standard_normal((args.chains, args.dim))
    x0 = z - (z @ W) @ W.T + exact_draws(c, args.k, (args.chains, W.shape[1]), rng) @ W.T
    target = WallTarget(torch.as_tensor(W, dtype=DTYPE, device=args.device), c, args.k)
    sampler = HMC(target, 1, np.zeros(args.dim), np.eye(args.dim), x0, settings,
                  seed=args.seed, device=args.device)
    project = torch.as_tensor(np.column_stack([W, u]), dtype=DTYPE, device=args.device)

    kept, retried, taken, divergent, leapfrogs = [], 0, 0, 0, 0
    start = time.perf_counter()
    for i in range(args.warmup + args.samples):
        warming = i < args.warmup
        stats = sampler.step(adapt=warming and not fixed)
        if i + 1 == args.warmup and not fixed:
            sampler.end_warmup()
        if not warming:
            kept.append((sampler.x @ project).cpu().numpy())
            retried += int(stats["retried"].sum())
            taken += int(stats["accepted2"].sum())
            divergent += int(stats["divergent"].sum())
            leapfrogs += stats["n_leapfrog"] + stats["extra_leapfrog"]
    transitions = args.samples * args.chains
    info = {"run": name, "step_size": sampler.step_size, "divergent_%": 100 * divergent / transitions,
            "retried_%": 100 * retried / transitions,
            "retry_accepted_%": 100 * taken / max(1, retried),
            "leapfrog_per_iter": leapfrogs / args.samples, "seconds": time.perf_counter() - start}
    return info, np.asarray(kept)                               # (samples, chains, walls + 1)


def compare(az, name: str, proj: np.ndarray, exact: dict, walls: int) -> list[dict]:
    rows = []

    def add(label: str, values: np.ndarray, truth: float) -> None:
        chains_first = values.T                                  # (chains, samples)
        est = float(values.mean())
        ess = float(az.ess(chains_first, method="mean"))
        se = float(values.std()) / np.sqrt(ess)
        rows.append({"run": name, "statistic": label, "exact": truth, "estimate": est,
                     "mc_se": se, "ess": ess, "z": (est - truth) / se if se > 0 else np.nan})

    for j in range(walls):
        t = proj[:, :, j]
        add(f"wall {j}: E[t]", t, exact["mean"])
        add(f"wall {j}: E[t^2]", t * t, exact["second"])
        add(f"wall {j}: P(t > q90)", (t > exact["q90"]).astype(float), exact["tail"])
    z = proj[:, :, walls]
    add("orthogonal: E[z]", z, 0.0)
    add("orthogonal: E[z^2]", z * z, 1.0)
    return rows


def main() -> None:
    import pandas as pd

    from pmedm_vb.compare import import_arviz

    az = import_arviz()
    args = parse_args()
    rng = np.random.default_rng(args.seed + 1)
    basis, _ = np.linalg.qr(rng.standard_normal((args.dim, args.walls + 1)))
    W, u = basis[:, : args.walls], basis[:, args.walls]
    c = float(np.exp(-args.k * args.wall_at))
    exact = exact_moments(c, args.k)

    runs = [("hmc", 0.0, False), ("dr", args.retry_factor, False),
            ("hmc fixed", 0.0, True), ("dr fixed", args.retry_factor, True)]
    infos, rows = [], []
    for name, retry, fixed in runs:
        info, proj = run(args, name, retry, fixed, W, u, c)
        infos.append(info)
        rows += compare(az, name, proj, exact, args.walls)
        print(f"{name}: done in {info['seconds']:.0f}s", flush=True)

    table, runs_table = pd.DataFrame(rows), pd.DataFrame(infos)
    worst = table.groupby("run")["z"].apply(lambda z: float(np.abs(z).max()))
    lines = [f"Delayed-rejection check: d = {args.dim}, {args.walls} walls, k = {args.k:g}, wall at "
             f"t = {args.wall_at:g} (exact E[t] {exact['mean']:.4f}, E[t^2] {exact['second']:.4f}, "
             f"q90 {exact['q90']:.4f}); {args.chains} chains, {args.warmup} warmup + "
             f"{args.samples} samples",
             "", runs_table.to_string(index=False, float_format=lambda v: f"{v:,.4g}"), "",
             table.to_string(index=False, float_format=lambda v: f"{v:,.4g}"), "",
             "largest |z| per run: " + ", ".join(f"{k} {v:.2f}" for k, v in worst.items()),
             "A correct sampler gives |z| mostly under 2 and none far over 4."]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
