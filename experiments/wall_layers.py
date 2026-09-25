"""Prototype: bend a fitted VB family's tail back along the walls it crosses.

A VB draw goes wrong when one (block group, record) cell takes a large share
of ``p``: its logit ``log q - a' lambda`` rises because ``s = -a' delta`` is
large, ``delta = lambda - lambda*`` and ``a`` the cell's column of the
constraint operator (its *wall normal*; ``wall_attribution.py``). The normals
are known exactly, so the family can be told where the walls are. Each wall
``j`` gets one layer acting on ``delta`` after the fitted family,

.. math::

    \\delta \\leftarrow \\delta + c_j \\frac{a_j}{a_j^\\top a_j}
        \\operatorname{softplus}(s_j - \\tau_j), \\qquad s_j = -a_j^\\top \\delta,

which leaves ``s_j`` alone well below ``tau_j`` and beyond it slows its growth
to slope ``1 - c_j``: a one-sided compression along that normal only. With
``0 < c_j < 1`` the layer is invertible and its log-Jacobian is
``log(1 - c_j sigmoid(s_j - tau_j))`` (matrix determinant lemma). It is a
planar flow (Rezende & Mohamed 2015, ICML) whose direction is fixed by the
model rather than learned.

**Prototype scope.** The fitted family (``run_map.py vb``) is frozen and only
``c_j, tau_j`` are fitted, by the same reparameterised ELBO. The walls are the
cells that exceed ``--threshold`` of ``N`` in draws from the base family; with
``--rounds 2`` the fitted layers are drawn from again and any new walls get
layers too. Fitting the base and the layers jointly is left for later.

Reports, for the base family and the layered one on ``--eval-draws`` fresh
draws: the share of draws with a cell over the threshold, the largest share's
quantiles, the ELBO, PSIS k-hat (Yao et al. 2018), and -- with ``--reference``,
an HMC run's directory -- the 90% width and sd ratios of every block group
constrained cell against HMC, over all cells and over rare ones (block group
published count at most ``--rare-count``). Writes ``<out>/<name>_walls.txt``
and ``_layers.npz`` (normals, ``c``, ``tau``)::

    $CONDA_PREFIX/bin/python experiments/wall_layers.py --puma 4701502 --alpha 1.0 \\
        --variance-floor zero --vb-run $RUNS/<sumdiff jobid> \\
        --reference $RUNS/hmc/4701502_a1.0_ref --out $RUNS/wall_layers
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

from pmedm_vb import compare
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir
from pmedm_vb.progress import logger, record_run
from pmedm_vb.solvers.base import ConstraintOperator
from pmedm_vb.solvers.vb import DTYPE, _DualTarget

from laplace_diagnostic import batched_f, load_vb
from wall_attribution import wall_normal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--puma", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--taper", default="tract")
    parser.add_argument("--area", default="knox-2024-5yr")
    parser.add_argument("--variance-floor", default="zero")
    parser.add_argument("--vb-run", type=Path, required=True, help="fitted VB family (run_map.py vb)")
    parser.add_argument("--reference", type=Path, default=None, help="HMC run_mcmc.py --out directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="share of N that makes a wall")
    parser.add_argument("--find-draws", type=int, default=4000, help="draws searched for walls")
    parser.add_argument("--max-walls", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--draws", type=int, default=64, help="draws per gradient step")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--eval-draws", type=int, default=4000)
    parser.add_argument("--rare-count", type=float, default=6.0)
    parser.add_argument("--min-width", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def floor_spec(text: str):
    return None if text == "none" else "zero" if text == "zero" else float(text)


class WallLayers(torch.nn.Module):
    """One compression layer per wall normal, applied in order to ``delta``."""

    def __init__(self, normals: np.ndarray, s0: np.ndarray) -> None:
        super().__init__()
        a = torch.as_tensor(normals, dtype=DTYPE)
        self.register_buffer("a", a)
        self.register_buffer("u", a / (a * a).sum(1, keepdim=True))
        # c starts small (sigmoid(-4) ~ 0.018) and tau at the wall's s where the
        # base family's upper tail begins, so the layers start near the identity.
        self.raw_c = torch.nn.Parameter(torch.full((len(a),), -4.0, dtype=DTYPE))
        self.tau = torch.nn.Parameter(torch.as_tensor(s0, dtype=DTYPE))

    @property
    def c(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_c)

    def forward(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(batch, m)`` -> transformed ``delta`` and each row's log-Jacobian."""
        c = self.c
        log_det = torch.zeros(delta.shape[0], dtype=DTYPE)
        for j in range(self.a.shape[0]):
            s = -(delta @ self.a[j])
            z = s - self.tau[j]
            delta = delta + c[j] * torch.nn.functional.softplus(z)[:, None] * self.u[j][None]
            log_det = log_det + torch.log1p(-c[j] * torch.sigmoid(z))
        return delta, log_det


def find_walls(inputs, op, draws: np.ndarray, threshold: float, limit: int) -> list[tuple[int, int]]:
    """Distinct (zone, unit) cells over ``threshold`` of ``N`` in any column of ``draws``,
    most frequent first."""
    with np.errstate(divide="ignore"):
        log_q = np.log(inputs.q)
    counts: dict[tuple[int, int], int] = {}
    for d in range(draws.shape[1]):
        logits = log_q - op.adjoint(draws[:, d])
        log_p = logits - logsumexp(logits)
        for zone, unit in zip(*np.nonzero(log_p > np.log(threshold))):
            counts[(int(zone), int(unit))] = counts.get((int(zone), int(unit)), 0) + 1
    return [cell for cell, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]


def evaluate(label, inputs, op, target, q, lam_star, layers, n_draws, rng, reference, args) -> dict:
    """Event rate, largest-share quantiles, ELBO, k-hat and (optionally) outcome ratios."""
    base = q.sample(rng, n_draws)
    log_q_base = q.log_density(base)
    if layers is None:
        lam, log_det = base, np.zeros(n_draws)
    else:
        with torch.no_grad():
            delta, ld = layers(torch.as_tensor((base - lam_star[:, None]).T, dtype=DTYPE))
        lam, log_det = lam_star[:, None] + delta.numpy().T, ld.numpy()
    log_q = log_q_base - log_det
    log_pi = -inputs.n * batched_f(target, lam, 100)
    share, _, _ = compare.largest_cells(inputs, lam)
    row = {
        "family": label,
        "draws_over_threshold": float((share > args.threshold).mean()),
        "draws_over_10pct": float((share > 0.10).mean()),
        "max_share_p50": float(np.median(share)), "max_share_p99": float(np.quantile(share, 0.99)),
        "max_share_max": float(share.max()),
        "elbo": float(np.mean(log_pi - log_q)),
        "elbo_se": float(np.std(log_pi - log_q, ddof=1) / np.sqrt(n_draws)),
        "psis_khat": compare.psis_khat(log_pi - log_q),
    }
    if reference is not None:
        matrix, meta = compare.outcome_matrix(inputs, crosstabs=())
        zones = inputs.zones.iloc[:, 0].to_numpy()
        mine = compare.summarise(compare.outcome_draws(inputs, lam, matrix), meta, zones)
        both = compare.compare_outcomes(reference["summary"], mine)
        wide = both[(both.q95_ref - both.q05_ref) >= args.min_width]
        published = pd.Series(inputs.Y_B.ravel(order="C"),
                              index=pd.MultiIndex.from_product([zones, inputs.bg_constraints]))
        key = pd.MultiIndex.from_arrays([both.zone, both.outcome])
        rare = both[published.reindex(key).to_numpy() <= args.rare_count]
        for name, frame in (("all", wide), ("rare", rare)):
            for col in ("width_ratio", "sd_ratio"):
                x = frame[col].dropna()
                if len(x):
                    row.update({f"{name}_{col}_p{int(100 * p)}": float(np.quantile(x, p))
                                for p in (0.01, 0.1, 0.5)})
        row["abs_median_diff_p99"] = float(both.median_diff.abs().quantile(0.99))
    return row


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    name = f"{args.puma}_{args.taper}_a{args.alpha:g}"

    inputs = PMEDMInputs.load(processed_dir() / "inputs" / args.area / args.puma)
    taper = None if args.taper == "none" else args.taper
    sigma = inputs.sigma(args.alpha, taper, floor_spec(args.variance_floor))
    target = _DualTarget(inputs, sigma, "cpu")
    op = ConstraintOperator(inputs)
    q, _, _ = load_vb(args.vb_run / f"{name}.npz")
    with np.load(args.vb_run / f"{name}.npz") as saved:
        lam_star = saved["map_lam"]

    reference = None
    if args.reference is not None:
        with np.load(args.reference / f"{name}_trace.npz") as saved:
            kept = saved["lam"]
        flat = kept.reshape(-1, kept.shape[-1])
        ref_lam = flat[np.unique(np.linspace(0, len(flat) - 1, min(args.eval_draws, len(flat))).astype(int))].T
        matrix, meta = compare.outcome_matrix(inputs, crosstabs=())
        reference = {"summary": compare.summarise(compare.outcome_draws(inputs, ref_lam, matrix), meta,
                                                  inputs.zones.iloc[:, 0].to_numpy())}

    lines = [f"Wall layers on {args.vb_run} for {name}; threshold {args.threshold:.0%} of N"]
    rows = [evaluate("base", inputs, op, target, q, lam_star, None, args.eval_draws, rng, reference, args)]

    cells: list[tuple[int, int]] = []
    layers = None
    for round_ in range(1, args.rounds + 1):
        search = q.sample(rng, args.find_draws)
        if layers is not None:
            with torch.no_grad():
                delta, _ = layers(torch.as_tensor((search - lam_star[:, None]).T, dtype=DTYPE))
            search = lam_star[:, None] + delta.numpy().T
        new = [c for c in find_walls(inputs, op, search, args.threshold, args.max_walls) if c not in cells]
        lines.append(f"round {round_}: {len(new)} new wall cells in {args.find_draws:,} draws")
        if not new:
            break
        cells += new[: max(0, args.max_walls - len(cells))]
        normals = np.stack([wall_normal(inputs, z, u) for z, u in cells])
        # tau starts at each wall's 99th percentile of s under the base family.
        base_delta = q.sample(rng, 2000) - lam_star[:, None]
        s0 = np.quantile(-(normals @ base_delta), 0.99, axis=1)
        previous = layers
        layers = WallLayers(normals, s0)
        if previous is not None:  # keep what round 1 learned
            k = previous.a.shape[0]
            with torch.no_grad():
                layers.raw_c[:k] = previous.raw_c
                layers.tau[:k] = previous.tau

        optimiser = torch.optim.Adam(layers.parameters(), lr=args.learning_rate)
        started = time.perf_counter()
        trace = []
        for step in range(args.steps):
            base = q.sample(rng, args.draws)
            delta0 = torch.as_tensor((base - lam_star[:, None]).T, dtype=DTYPE)
            delta, log_det = layers(delta0)
            lam = torch.as_tensor(lam_star, dtype=DTYPE)[None] + delta
            # log q_base does not depend on the layers, so it drops out of the gradient.
            elbo = (-inputs.n * target(lam) + log_det).mean()
            optimiser.zero_grad()
            (-elbo).backward()
            optimiser.step()
            trace.append(elbo.item())
            if (step + 1) % 100 == 0:
                logger.info("round %d step %d: ELBO part %.2f (last 100 mean), c median %.3f max %.3f",
                            round_, step + 1, np.mean(trace[-100:]), float(layers.c.median()),
                            float(layers.c.max()))
        lines.append(f"   {len(cells)} layers fitted in {time.perf_counter() - started:.0f}s; "
                     f"c median {float(layers.c.median()):.3f}, max {float(layers.c.max()):.3f}")

    if layers is not None:
        rows.append(evaluate(f"layers ({len(cells)})", inputs, op, target, q, lam_star, layers,
                             args.eval_draws, rng, reference, args))
        np.savez(args.out / f"{name}_layers.npz", normals=layers.a.numpy(), c=layers.c.detach().numpy(),
                 tau=layers.tau.detach().numpy(),
                 cells=np.array(cells), zones=inputs.zones.iloc[:, 0].to_numpy()[[z for z, _ in cells]],
                 units=inputs.units.iloc[:, 0].to_numpy()[[u for _, u in cells]])

    table = pd.DataFrame(rows).set_index("family").T
    lines += ["", table.to_string(float_format=lambda v: f"{v:.4g}")]
    text = "\n".join(lines) + "\n"
    (args.out / f"{name}_walls.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
