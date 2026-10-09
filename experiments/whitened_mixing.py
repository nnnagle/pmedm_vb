"""The posterior in HMC's whitened coordinates, and how fast HMC mixes there.

Read from the HMC traces the paper's experiments already hold; no sampling.
HMC samples ``x`` with ``xi = mu + G'^{-1} x`` (:mod:`~pmedm_vb.solvers.mcmc`),
``mu`` and ``G`` the skewed VB fit's median and Gaussian-part precision factor,
so each kept draw is mapped back by ``x = G' (xi - mu)``. Were the VB Gaussian
the posterior, ``x`` would be standard normal. Two questions, before the nested
R-hat runs:

1. **Shape** (decides a learned diagonal scale on the whitening): per
   coordinate of ``x``, the posterior mean and sd over every kept draw of every
   chain; quantiles over coordinates, and how many sds are over 2 or 3 or
   under 0.5. One step size and trajectory length suit every coordinate only
   if the sds are close to 1.
2. **Mixing** (decides the chain count and the trajectory length): ``tau``,
   sampling iterations per effective draw, ``chains x iterations / ESS``,
   and ESS per 1,000 gradient evaluations per chain (each chain's leapfrog
   steps, the cost that scales with chains on a saturated GPU).

   - ``log pi`` and the largest cell's share, recorded every iteration: bulk
     ESS, as in ``run_mcmc.py``'s report. For a Gaussian, ``log pi`` is
     ``-|x|^2 / 2`` up to a constant, so it shows how fast second moments mix.
   - ``x`` and ``x^2`` per coordinate, from the kept (thinned) draws: ESS of
     the mean (``method="mean"``, not rank-normalised), so ``x^2`` measures the
     second moment. Thinning caps what can be seen: where ESS per kept draw
     is near 1, ``tau`` is at most the thinning interval, not equal to it.

Writes ``<out>/whitened_mixing.csv`` (one row per experiment, PUMA, alpha and
run), ``whitened_mixing.txt`` (the same, readable), and per run
``<experiment>_<fit>_<kind>.npz`` with each coordinate's mean, sd and ESS::

    $PY experiments/whitened_mixing.py --root $PAPER \\
        --experiments exp01_baseline exp02_nullspace exp03_rollup --alpha 0.01 \\
        --out $PAPER/whitened_mixing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pmedm_vb.progress import logger, record_run

from laplace_diagnostic import load_vb
from run_experiment import fit_name

#: Quantiles over coordinates reported for the sds and for ESS per kept draw.
SD_QUANTILES = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
ESS_QUANTILES = (0.0, 0.01, 0.1, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' results root")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--kinds", nargs="+", choices=["short", "ref"], default=["short", "ref"])
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def label(q: float) -> str:
    return {0.0: "min", 1.0: "max"}.get(q, f"p{round(100 * q):02d}")


def one_run(az, q, trace: dict, chains_first) -> tuple[dict, dict]:
    """The summary row and the per-coordinate arrays for one HMC trace."""
    warmup = int(trace["warmup"])
    thin = int(trace["thin"])
    iterations, chains = trace["log_pi"][warmup:].shape
    gradients = int(trace["n_leapfrog"][warmup:].sum())   # per chain: the step size is shared
    key = "xi" if "xi" in trace else "lam"
    kept, _, m = trace[key].shape
    draws = trace[key].reshape(-1, m)                      # (kept x chains, m), kept-major
    x = q.whiten((draws - q.mean).T).T.reshape(kept, chains, m)

    row = {"chains": chains, "warmup": warmup, "iterations": iterations, "thin": thin,
           "kept": kept, "m": m, "step_size": float(trace["step_size"][-1]),
           "leapfrog_mean": gradients / iterations,
           "accept_mean": float(trace["accept_prob"][warmup:].mean()),
           "divergent": int(trace["divergent"][warmup:].sum())}

    flat = x.reshape(-1, m)
    sd, mean = flat.std(axis=0), flat.mean(axis=0)
    varying = sd > 0
    row["fixed"] = int((~varying).sum())
    for qq in SD_QUANTILES:
        row[f"sd_{label(qq)}"] = float(np.quantile(sd[varying], qq))
    row["sd_over_2"] = int((sd > 2).sum())
    row["sd_over_3"] = int((sd > 3).sum())
    row["sd_under_0.5"] = int((varying & (sd < 0.5)).sum())
    row["abs_mean_p50"] = float(np.median(np.abs(mean[varying])))
    row["abs_mean_max"] = float(np.abs(mean[varying]).max())

    def rate(ess: float | np.ndarray) -> dict:
        return {"tau": chains * iterations / ess, "per_1000_grad": 1000 * ess / (chains * gradients)}

    for name in ("log_pi", "max_share"):
        ess = float(az.ess(chains_first(name), method="bulk"))
        row[f"{name}_ess"] = ess
        row.update({f"{name}_{k}": v for k, v in rate(ess).items()})

    per_coordinate = {"mean": mean, "sd": sd}
    for name, values in (("x", x), ("x2", x * x)):
        ess = np.full(m, np.nan)
        dataset = az.convert_to_dataset({name: np.transpose(values[:, :, varying], (1, 0, 2))})
        ess[varying] = az.ess(dataset, method="mean")[name].values
        per_coordinate[f"ess_{name}"] = ess
        per_kept = ess[varying] / (chains * kept)
        for qq in ESS_QUANTILES:
            row[f"{name}_ess_per_kept_{label(qq)}"] = float(np.quantile(per_kept, qq))
        for at, qq in (("worst", 0.0), ("median", 0.5)):
            value = float(np.quantile(ess[varying], qq))
            row.update({f"{name}_{k}_{at}": v for k, v in rate(value).items()})
    return row, per_coordinate


def main() -> None:
    from pmedm_vb.compare import import_arviz

    az = import_arviz()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)

    rows = []
    for experiment in args.experiments:
        exp = args.root / experiment
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            name = fit_name(spec, puma, args.alpha)
            vb_path = exp / puma / "vb_skewed" / f"{name}.npz"
            if not vb_path.exists():
                logger.info("%s %s: no skewed VB fit (%s); skipped", experiment, puma, vb_path)
                continue
            q, _, _ = load_vb(vb_path)
            for kind in args.kinds:
                trace_path = exp / puma / f"hmc_{kind}" / f"{name}_trace.npz"
                if not trace_path.exists():
                    logger.info("%s %s %s: no trace (%s); skipped", experiment, puma, kind, trace_path)
                    continue
                with np.load(trace_path) as saved:
                    trace = {key: saved[key] for key in saved.files}
                warmup = int(trace["warmup"])
                if trace["log_pi"].shape[0] <= warmup:
                    logger.info("%s %s %s: still in warmup; skipped", experiment, puma, kind)
                    continue

                def chains_first(key: str) -> np.ndarray:
                    return trace[key][warmup:].T  # (chains, iterations)

                logger.info("%s %s %s: %s", experiment, puma, kind, trace_path)
                row, per_coordinate = one_run(az, q, trace, chains_first)
                rows.append({"experiment": experiment, "puma": puma, "alpha": args.alpha,
                             "kind": kind, **row})
                np.savez(args.out / f"{experiment}_{name}_{kind}.npz", **per_coordinate)
                pd.DataFrame(rows).to_csv(args.out / "whitened_mixing.csv", index=False)
    if not rows:
        raise SystemExit("no traces found")

    table = pd.DataFrame(rows)
    ident = ["experiment", "puma", "kind"]
    sections = {
        "Sampler": ident + ["chains", "iterations", "thin", "m", "step_size", "leapfrog_mean",
                            "accept_mean", "divergent"],
        "Shape: sd of each whitened coordinate (N(0, I) would give 1)":
            ident + [c for c in table.columns if c.startswith("sd_")] + ["abs_mean_p50", "abs_mean_max"],
        "Mixing per iteration: tau = iterations per effective draw; ESS per 1,000 gradients per chain":
            ident + [c for c in table.columns if c.startswith(("log_pi_", "max_share_"))],
        "Mixing of x and x^2 (kept draws): ESS per kept draw over coordinates; tau and "
        "ESS per 1,000 gradients at the worst and median coordinate":
            ident + [c for c in table.columns if c.startswith(("x_", "x2_"))],
    }
    lines = [f"Whitened coordinates and mixing, alpha {args.alpha:g}, from {args.root}", ""]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for title, columns in sections.items():
            lines += [title, table[columns].to_string(index=False, float_format=lambda v: f"{v:,.3g}"), ""]
    lines.append("ESS per kept draw near 1 means thinning hides the per-iteration mixing: "
                 "tau is then at most the thinning interval.")
    text = "\n".join(lines)
    (args.out / "whitened_mixing.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out / 'whitened_mixing.csv'} and whitened_mixing.txt")


if __name__ == "__main__":
    main()
