"""Are the posterior's wide directions lopsided, and does the skewed VB family know?

Read from the reference HMC traces and the skewed VB fits the experiments
already hold; no sampling. Tests the hypothesis that the step size halved
under a diagonal scale (``run_mcmc.py --scale-from``) because the wide
whitened coordinates fall steeply on one side and trail on the other -- the
Poisson log-rate shape ``vb.py``'s *Skewed family* was built for -- and asks
whether that family's sinh-arcsinh layers already point the right way, which
would make HMC through the family's whole nonlinear map (NeuTra) a candidate.

Shape is measured three ways per coordinate: the moment skewness; Kelly's
quantile skewness ``(q90 + q10 - 2 q50) / (q90 - q10)``, robust to the tails;
and the tail ratio ``(q99 - q50) / (q50 - q01)``, reported as ``max(r, 1/r)``
(1 for a symmetric shape; 5 means one side reaches five times as far as the
other). Flatness is measured two ways: the excess kurtosis (0 for a normal,
-1.2 for a uniform) and the quantile ratio ``(q99 - q01) / (q90 - q10)``
(1.82 for a normal, 1.23 for a uniform; a flat-topped shape with steep walls
on both sides falls between). Draws are autocorrelated, so a noise level is
set by simulation from standard normal samples as large as the band's median
ESS: the median ``|Kelly|``, and the 5th-95th percentile of the quantile
ratio for one coordinate (a band's median, over many coordinates, varies
much less).

1. **Whitened coordinates** ``x = G' (xi - mu)`` (HMC's space; ``mu``, ``G``
   the skewed fit's median and Gaussian-part factor), by band of posterior sd:
   median ``|moment skew|``, ``|Kelly|`` and tail ratio, against the noise level.
2. **The skewed family in the same space**: ``--vb-draws`` draws of the
   skewed family, whitened the same way, for the wide coordinates (sd over
   ``--wide-sd``): its ``|Kelly|``, and how often its sign matches the
   posterior's where the posterior's is beyond twice the noise level, and
   the rank correlation of the two.
3. **The skewed family where its layers act**, the sampling coordinates
   ``xi`` themselves: the coordinates whose posterior sd exceeds the
   Gaussian part's marginal sd by over ``--wide-sd`` times; the posterior's
   sd against the Gaussian part's and the skewed family's, and the same skew
   comparisons as section 2.

4. **The Gaussian-only fit in the same whitened space** (``vb_gaussian``,
   fitted like the skewed fit's first stage; the second stage re-optimises the
   Gaussian part together with the skew layers): ``--vb-draws`` of its draws,
   whitened by the skewed fit. Their sd by band of posterior sd, and on the
   wide coordinates with the posterior's sd over theirs and their mean offset.
   An sd clearly over 1 on the wide coordinates means the plain Gaussian was
   wider there, so the skewed stage narrowed those directions; about 1 means
   they were as narrow before the skew was fitted.

Writes ``<out>/skew_check.csv`` (one row per experiment and PUMA),
``skew_check.txt`` and per fit ``<experiment>_<fit>_skew.npz`` (each
coordinate's sd and Kelly skewness, posterior and VB, in both spaces)::

    $PY experiments/skew_check.py --root $PAPER --alpha 0.01 --pumas 4701501 \\
        --out $PAPER/oracle_scale/skew_check
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

#: Bands of whitened posterior sd, as in whitened_mixing.py.
SD_BANDS = ((0.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 10.0), (10.0, np.inf))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True, help="the experiments' results root")
    parser.add_argument("--experiments", nargs="+",
                        default=["exp01_baseline", "exp02_nullspace", "exp03_rollup"])
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--pumas", nargs="+", default=None, help="default: the experiment's")
    parser.add_argument("--kind", default="ref", help="the HMC run read, <puma>/hmc_<kind>")
    parser.add_argument("--wide-sd", type=float, default=2.0,
                        help="sections 2-3: a whitened sd, or a ratio of sds, above which a "
                             "coordinate counts as wide")
    parser.add_argument("--vb-draws", type=int, default=8000)
    parser.add_argument("--null-reps", type=int, default=200,
                        help="normal samples per band for the noise level")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def shape(z: np.ndarray) -> dict[str, np.ndarray]:
    """Per column of ``z`` (``(draws, k)``): sd, moment skewness, Kelly's
    skewness, the tail ratio ``max(r, 1/r)``, excess kurtosis and the quantile
    ratio ``(q99 - q01) / (q90 - q10)``."""
    sd = z.std(axis=0)
    centred = z - z.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        moment = (centred ** 3).mean(axis=0) / sd ** 3
        kurt = (centred ** 4).mean(axis=0) / sd ** 4 - 3.0
        q01, q10, q50, q90, q99 = np.quantile(z, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0)
        kelly = (q90 + q10 - 2 * q50) / (q90 - q10)
        ratio = (q99 - q50) / (q50 - q01)
        qkurt = (q99 - q01) / (q90 - q10)
    return {"sd": sd, "moment": moment, "kelly": kelly, "tail": np.maximum(ratio, 1 / ratio),
            "kurt": kurt, "qkurt": qkurt}


def noise(n: int, reps: int, rng: np.random.Generator) -> dict[str, float]:
    """For standard normal samples of size ``n``: the median ``|Kelly|`` and the
    5th and 95th percentiles of the quantile ratio."""
    sim = shape(rng.standard_normal((max(int(n), 20), reps)))
    lo, hi = np.quantile(sim["qkurt"], [0.05, 0.95])
    return {"kelly": float(np.median(np.abs(sim["kelly"]))), "qkurt_p05": float(lo),
            "qkurt_p95": float(hi)}


def noise_kelly(n: int, reps: int, rng: np.random.Generator) -> float:
    return noise(n, reps, rng)["kelly"]


def agreement(post: np.ndarray, vb: np.ndarray, threshold: float) -> dict[str, float]:
    """How the skewed family's Kelly skewness matches the posterior's."""
    from scipy.stats import spearmanr

    clear = np.abs(post) > threshold
    out = {"n_clear": int(clear.sum()),
           "sign_agree": float(np.mean(np.sign(post[clear]) == np.sign(vb[clear])))
           if clear.any() else np.nan,
           "post_abs_kelly": float(np.median(np.abs(post))),
           "vb_abs_kelly": float(np.median(np.abs(vb)))}
    out["rank_corr"] = float(spearmanr(post, vb)[0]) if post.size > 2 else np.nan
    return out


def gaussian_only(q, q_gauss, post_sd: np.ndarray, wide_sd: float, draws: int,
                  rng: np.random.Generator) -> tuple[dict, np.ndarray]:
    """Section 4: the Gaussian-only fit's draws in the skewed fit's whitened space."""
    if q_gauss.size != q.size:
        raise SystemExit(f"the Gaussian fit has {q_gauss.size} coordinates, the skewed {q.size}")
    x_g = q.whiten(q_gauss.sample(rng, draws) - q.mean[:, None])           # (m, draws)
    sd_g, mean_g = x_g.std(axis=1), x_g.mean(axis=1)
    varying = post_sd > 0
    row = {}
    for low, high in SD_BANDS:
        band = varying & (post_sd >= low) & (post_sd < high)
        prefix = f"g_sd_{low:g}_{high:g}" if np.isfinite(high) else f"g_sd_over_{low:g}"
        if band.any():
            row[f"{prefix}_gauss_sd"] = float(np.median(sd_g[band]))
    wide = varying & (post_sd > wide_sd)
    if wide.any():
        row["g_wide_gauss_sd"] = float(np.median(sd_g[wide]))
        row["g_wide_gauss_sd_p90"] = float(np.quantile(sd_g[wide], 0.9))
        row["g_wide_post_over_gauss"] = float(np.median(post_sd[wide] / sd_g[wide]))
        row["g_wide_gauss_abs_mean"] = float(np.median(np.abs(mean_g[wide])))
    return row, sd_g


def one_fit(az, q, trace: dict, args: argparse.Namespace, rng: np.random.Generator) -> tuple[dict, dict]:
    key = "xi" if "xi" in trace else "lam"
    kept, chains, m = trace[key].shape
    xi = trace[key].reshape(-1, m)                                    # posterior, (draws, m)
    x = q.whiten((xi - q.mean).T).T
    post_x = shape(x)
    varying = post_x["sd"] > 0

    ess = np.full(m, np.nan)
    cube = np.transpose(x.reshape(kept, chains, m)[:, :, varying], (1, 0, 2))
    ess[varying] = az.ess(az.convert_to_dataset({"x": cube}), method="bulk")["x"].values
    row = {"draws": kept * chains, "m": m}

    for low, high in SD_BANDS:
        band = varying & (post_x["sd"] >= low) & (post_x["sd"] < high)
        prefix = f"sd_{low:g}_{high:g}" if np.isfinite(high) else f"sd_over_{low:g}"
        row[f"{prefix}_n"] = int(band.sum())
        if not band.any():
            continue
        row[f"{prefix}_abs_moment"] = float(np.median(np.abs(post_x["moment"][band])))
        row[f"{prefix}_abs_kelly"] = float(np.median(np.abs(post_x["kelly"][band])))
        row[f"{prefix}_tail"] = float(np.median(post_x["tail"][band]))
        row[f"{prefix}_kurt"] = float(np.median(post_x["kurt"][band]))
        row[f"{prefix}_qkurt"] = float(np.median(post_x["qkurt"][band]))
        level = noise(np.median(ess[band]), args.null_reps, rng)
        row[f"{prefix}_noise_kelly"] = level["kelly"]
        row[f"{prefix}_noise_qkurt_p05"] = level["qkurt_p05"]
        row[f"{prefix}_noise_qkurt_p95"] = level["qkurt_p95"]

    # The skewed family's draws, and its Gaussian part's from the same eps.
    eps = rng.standard_normal((m, args.vb_draws))
    gauss = q.gaussian_part(eps)                                      # (m, draws), offsets
    vb_xi = (q.skew_forward(gauss)[0] if q.is_skewed else gauss) + q.mean[:, None]
    vb_x = q.whiten(vb_xi - q.mean[:, None]).T
    vb_shape_x = shape(vb_x)

    wide = varying & (post_x["sd"] > args.wide_sd)
    row["wide_x_n"] = int(wide.sum())
    if wide.any():
        noise_level = noise_kelly(np.median(ess[wide]), args.null_reps, rng)
        row["wide_x_noise_kelly"] = noise_level
        row.update({f"wide_x_{k}": v for k, v in
                    agreement(post_x["kelly"][wide], vb_shape_x["kelly"][wide], 2 * noise_level).items()})
        row["wide_x_vb_sd_median"] = float(np.median(vb_shape_x["sd"][wide]))
        row["wide_x_post_tail"] = float(np.median(post_x["tail"][wide]))
        row["wide_x_vb_tail"] = float(np.median(vb_shape_x["tail"][wide]))
        row["wide_x_post_qkurt"] = float(np.median(post_x["qkurt"][wide]))
        row["wide_x_vb_qkurt"] = float(np.median(vb_shape_x["qkurt"][wide]))

    post_xi = shape(xi)
    vb_xi_shape = shape(vb_xi.T)
    gauss_sd = gauss.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_gauss = post_xi["sd"] / gauss_sd
        ratio_skew = post_xi["sd"] / vb_xi_shape["sd"]
    wide_xi = (post_xi["sd"] > 0) & (gauss_sd > 0) & (ratio_gauss > args.wide_sd)
    row["wide_xi_n"] = int(wide_xi.sum())
    if wide_xi.any():
        cube = np.transpose(xi.reshape(kept, chains, m)[:, :, wide_xi], (1, 0, 2))
        ess_xi = az.ess(az.convert_to_dataset({"xi": cube}), method="bulk")["xi"].values
        noise_level = noise_kelly(np.median(ess_xi), args.null_reps, rng)
        row["wide_xi_noise_kelly"] = noise_level
        row["wide_xi_sd_over_gauss"] = float(np.median(ratio_gauss[wide_xi]))
        row["wide_xi_sd_over_skewed"] = float(np.median(ratio_skew[wide_xi]))
        row.update({f"wide_xi_{k}": v for k, v in
                    agreement(post_xi["kelly"][wide_xi], vb_xi_shape["kelly"][wide_xi], 2 * noise_level).items()})
        row["wide_xi_post_tail"] = float(np.median(post_xi["tail"][wide_xi]))
        row["wide_xi_vb_tail"] = float(np.median(vb_xi_shape["tail"][wide_xi]))

    per_coordinate = {"x_sd": post_x["sd"], "x_kelly": post_x["kelly"], "x_tail": post_x["tail"],
                      "x_kurt": post_x["kurt"], "x_qkurt": post_x["qkurt"],
                      "x_ess": ess, "x_vb_sd": vb_shape_x["sd"], "x_vb_kelly": vb_shape_x["kelly"],
                      "xi_sd": post_xi["sd"], "xi_kelly": post_xi["kelly"],
                      "xi_gauss_sd": gauss_sd, "xi_vb_sd": vb_xi_shape["sd"],
                      "xi_vb_kelly": vb_xi_shape["kelly"]}
    return row, per_coordinate


def main() -> None:
    from pmedm_vb.compare import import_arviz

    az = import_arviz()
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record_run(args.out, args)
    rng = np.random.default_rng(args.seed)

    rows = []
    for experiment in args.experiments:
        exp = args.root / experiment
        spec = json.loads((exp / "experiment.json").read_text())
        for puma in args.pumas or spec["pumas"]:
            name = fit_name(spec, puma, args.alpha)
            vb_path = exp / puma / "vb_skewed" / f"{name}.npz"
            trace_path = exp / puma / f"hmc_{args.kind}" / f"{name}_trace.npz"
            if not (vb_path.exists() and trace_path.exists()):
                logger.info("%s %s: missing %s; skipped", experiment, puma,
                            vb_path if not vb_path.exists() else trace_path)
                continue
            q, _, _ = load_vb(vb_path)
            with np.load(trace_path) as saved:
                trace = {k: saved[k] for k in ("xi", "lam") if k in saved.files}
            logger.info("%s %s: %s", experiment, puma, trace_path)
            row, per_coordinate = one_fit(az, q, trace, args, rng)
            del trace
            gauss_path = exp / puma / "vb_gaussian" / f"{name}.npz"
            if gauss_path.exists():
                q_gauss, _, _ = load_vb(gauss_path)
                extra, sd_g = gaussian_only(q, q_gauss, per_coordinate["x_sd"], args.wide_sd,
                                            args.vb_draws, rng)
                row.update(extra)
                per_coordinate["x_gauss_only_sd"] = sd_g
            else:
                logger.info("%s %s: no Gaussian-only fit (%s); section 4 skipped",
                            experiment, puma, gauss_path)
            rows.append({"experiment": experiment, "puma": puma, "alpha": args.alpha,
                         "kind": args.kind, **row})
            np.savez(args.out / f"{experiment}_{name}_skew.npz", **per_coordinate)
            pd.DataFrame(rows).to_csv(args.out / "skew_check.csv", index=False)
    if not rows:
        raise SystemExit("no traces found")

    table = pd.DataFrame(rows)
    ident = ["experiment", "puma"]
    sections = {
        "1. Posterior shape of the whitened coordinates by band of sd: median |moment skew|, "
        "|Kelly|, tail ratio max(r, 1/r), excess kurtosis and quantile ratio (normal 1.82, "
        "uniform 1.23); noise from normal samples of the band's median ESS":
            ident + [c for c in table.columns if c.startswith("sd_")],
        f"2. Wide whitened coordinates (sd over {args.wide_sd:g}): the skewed family against the "
        "posterior. sign_agree is over coordinates whose posterior |Kelly| exceeds twice the noise":
            ident + [c for c in table.columns if c.startswith("wide_x_")],
        f"3. Sampling coordinates whose posterior sd exceeds the Gaussian part's by over "
        f"{args.wide_sd:g} times: sd ratios (posterior over Gaussian part, over skewed family) "
        "and the same skew comparison":
            ident + [c for c in table.columns if c.startswith("wide_xi_")],
    }
    gauss_columns = [c for c in table.columns if c.startswith("g_")]
    if gauss_columns:
        sections["4. The Gaussian-only fit, whitened by the skewed fit: its median sd by band of "
                 "posterior sd (1 = as wide as the skewed fit's Gaussian part), and on the wide "
                 "coordinates its sd (median, p90), the posterior's sd over it and its mean offset"] = (
            ident + gauss_columns)
    lines = [f"Shape of the wide directions, alpha {args.alpha:g}, {args.kind} runs, from {args.root}", ""]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        for title, columns in sections.items():
            lines += [title, table[columns].to_string(index=False, float_format=lambda v: f"{v:,.3g}"), ""]
    text = "\n".join(lines)
    (args.out / "skew_check.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {args.out / 'skew_check.csv'} and skew_check.txt")


if __name__ == "__main__":
    main()
