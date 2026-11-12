"""Fitting the encoder-free baseline: a free position per merged row, fitted
jointly with the linear decoder (:mod:`pmedm_vb.embed.model`).

This is close to a design-weighted exponential-family PCA (Collins, Dasgupta &
Schapire 2001; the multinomial case is GLM-PCA, Townes et al. 2019), with the
agreed loss: design weights, rare-category weights ``c_k``, one multinomial per
target over its universe, and a ridge penalty.

**Held-out entries.** Withheld from the loss, with a fixed seed, by merged row
(the unit of the free positions: every unit merged into a row shares the row's
held-out targets), in one of two ways recorded in the targets' metadata:

* ``"entry"`` (configuration v1, tables as targets): a random 10% of the
  (row, target) entries with a nonzero universe count, independently.
* ``"attribute"`` (v2, attribute targets): each applicable target of a row is
  a *seed* with probability ``q``, and a seed is held out together with every
  target depending on it (its ``masks`` list, from
  :func:`~pmedm_vb.embed.attribute_targets.mask_closure`); a dependent masked
  as a seed masks only itself. ``q`` is set by bisection, on the same random
  draws, so that 10% of applicable entries are held out overall. A held-out
  attribute then cannot be read back through an attribute computed from it.

Check 1
(:mod:`pmedm_vb.embed.check`) scores reconstruction on those entries only,
because in-sample reconstruction with free positions improves with every
extra dimension and every smaller penalty, whatever the truth.

**Initialisation.** The intercepts and group-quarters offsets start at the
exact optimum of the intercept-only model (with ``W = 0`` every unit of a type
shares its logits, and the weighted multinomial optimum is the ``c``-weighted
category share). The positions start in the leading ``d``-dimensional
subspace of that model's Pearson residuals -- a truncated SVD, with rows
weighted by the square root of their fit weight -- scaled to unit variance,
with the decoder slopes starting small (0.1 times the singular vectors). The
SVD gives the optimiser a sensible subspace; the scale is left to it.

**Optimiser.** Full-batch L-BFGS with a strong-Wolfe line search over all
parameters at once, run in chunks of :data:`CHUNK` iterations; it stops when a
chunk lowers the loss by less than ``tol`` relative, or at ``max_iter``. The
problem is smooth and fits in one GPU's memory, which is the case L-BFGS suits;
stochastic optimisers would need a learning-rate schedule and many more passes
for no gain here.

**After fitting.** The learned coordinates are centred and whitened with the
design weights, the decoder transformed so predictions are unchanged
(:func:`~pmedm_vb.embed.model.whiten`). Then a **calibrated decoder** is refitted
with the positions held fixed and ``c_k = 1``: the rare-category weights tilt
the fitted decoder's probabilities towards rare categories (with free
intercepts, the weighted optimum is proportional to ``c_k`` times the share),
so they shape the positions but must not be used to judge them. Check 1 uses
the calibrated decoder, design-weighted, for every fit -- the unweighted
diagnostic included -- so every fit is judged on the same population scale.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from pmedm_vb.embed.model import Decoder, category_weights, data_term, ridge_term, whiten
from pmedm_vb.embed.units import TargetArrays
from pmedm_vb.progress import logger

#: L-BFGS iterations per chunk; the loss is logged once per chunk.
CHUNK = 25


def universe_counts(arrays: TargetArrays) -> np.ndarray:
    """``(rows, targets)`` universe count: the row sum within each target."""
    return np.add.reduceat(arrays.counts, arrays.groups[:, 0], axis=1)


def heldout_mask(universe: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Exactly ``round(fraction × eligible)`` (row, target) entries with a
    nonzero universe count, chosen uniformly at random with ``seed``."""
    eligible = np.flatnonzero(universe.ravel() > 0)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=int(round(fraction * len(eligible))), replace=False)
    mask = np.zeros(universe.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(universe.shape)


def attribute_heldout_mask(universe: np.ndarray, masks: np.ndarray, fraction: float,
                           seed: int) -> tuple[np.ndarray, float]:
    """Attribute masking (v2): ``masks[s, g]`` is True when holding out target
    ``s`` also holds out ``g``. Returns the (row, target) mask and the seed
    probability ``q`` giving ``fraction`` of applicable entries held out."""
    applicable = universe > 0
    draws = np.random.default_rng(seed).random(applicable.shape)
    spread = masks.astype(np.float32)

    def held(q: float) -> np.ndarray:
        seeds = ((draws < q) & applicable).astype(np.float32)
        return ((seeds @ spread) > 0) & applicable

    total = applicable.sum()
    low, high = 0.0, fraction
    for _ in range(40):
        q = (low + high) / 2
        if held(q).sum() / total < fraction:
            low = q
        else:
            high = q
    q = (low + high) / 2
    return held(q), q


def masking_matrix(arrays: TargetArrays) -> np.ndarray:
    """``(targets, targets)``: row ``s`` marks the targets held out with ``s``."""
    ids = [t["id"] for t in arrays.meta["targets"]]
    index = {t: i for i, t in enumerate(ids)}
    out = np.eye(len(ids), dtype=bool)
    for i, t in enumerate(arrays.meta["targets"]):
        for j in t.get("masks", [t["id"]]):
            out[i, index[j]] = True
    return out


def column_group(groups: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(groups)), groups[:, 1] - groups[:, 0])


def intercepts(counts: np.ndarray, row_weight: np.ndarray, gq: np.ndarray,
               groups: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact optimum of the intercept-only model: ``b`` for households and an
    offset per group-quarters type, from ``c``-weighted category shares of the
    (already masked) ``counts``. A target with no units of a type in its
    universe leaves that type's offset at zero."""
    K = counts.shape[1]
    household = gq.sum(1) == 0
    types = [household, *(gq[:, j] == 1 for j in range(gq.shape[1]))]
    logs = []
    for rows in types:
        totals = (row_weight * rows) @ counts.astype(np.float64) * c
        log_share = np.zeros(K)
        defined = np.zeros(K, dtype=bool)
        for start, end in groups:
            block = totals[start:end]
            if block.sum() > 0:
                log_share[start:end] = np.log(np.maximum(block / block.sum(), 1e-8))
                defined[start:end] = True
        logs.append((log_share, defined))
    b = logs[0][0]
    O = np.zeros((K, gq.shape[1]))
    for j, (log_share, defined) in enumerate(logs[1:]):
        O[:, j] = np.where(defined & logs[0][1], log_share - b, np.where(defined, log_share, 0.0))
    return b, O


def _lbfgs(params, closure, objective, *, max_iter: int, tol: float, label: str) -> list[dict]:
    """L-BFGS in chunks of :data:`CHUNK` iterations. ``objective()`` evaluates
    the loss without gradients; it is logged after each chunk (``step`` itself
    returns the loss at the *start* of the chunk, which would lag by one)."""
    optimiser = torch.optim.LBFGS(params, lr=1.0, max_iter=CHUNK, history_size=20,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-10, tolerance_change=1e-14)
    trajectory, done, previous, start = [], 0, None, time.perf_counter()
    while done < max_iter:
        optimiser.step(closure)
        with torch.no_grad():
            loss = float(objective())
        done += CHUNK
        state = optimiser.state[optimiser._params[0]]
        trajectory.append({"iteration": int(state.get("n_iter", done)), "loss": loss,
                           "evaluations": int(state.get("func_evals", 0)),
                           "seconds": round(time.perf_counter() - start, 2)})
        logger.info("%s: iteration %d, loss %.6f", label, trajectory[-1]["iteration"], loss)
        if previous is not None and (previous - loss) <= tol * abs(previous):
            break
        previous = loss
    return trajectory


def fit_baseline(
    arrays: TargetArrays,
    *,
    d: int,
    gamma: float,
    ridge: float,
    weighted: bool = True,
    heldout_fraction: float = 0.1,
    seed: int = 1,
    device: str | None = None,
    max_iter: int = 3000,
    tol: float = 1e-6,
    calibrate_iter: int = 500,
) -> dict:
    """Fit positions and decoder; return numpy arrays and diagnostics."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    universe = universe_counts(arrays)
    masking = arrays.meta.get("masking", "entry")
    seed_probability = None
    if masking == "attribute":
        held, seed_probability = attribute_heldout_mask(
            universe, masking_matrix(arrays), heldout_fraction, seed)
    else:
        held = heldout_mask(universe, heldout_fraction, seed)
    train = (universe > 0) & ~held
    col = column_group(arrays.groups)
    train_counts = arrays.counts * train[:, col]
    fit_weight = arrays.weight if weighted else arrays.n_units
    design = arrays.weight
    empty = arrays.counts.sum(0) == 0
    c = category_weights(arrays.counts, fit_weight, arrays.groups, gamma, train=train)
    c[empty] = 0.0

    b0, O0 = intercepts(train_counts, fit_weight, arrays.gq, arrays.groups, c)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)  # noqa: E731
    counts_t, gq_t = t(train_counts), t(arrays.gq)
    w_t, c_t = t(fit_weight), t(c)
    total = float(fit_weight.sum())

    decoder = Decoder(arrays.counts.shape[1], d, arrays.gq.shape[1], arrays.groups, empty).to(device)
    with torch.no_grad():
        decoder.b.copy_(t(b0))
        decoder.O.copy_(t(O0))
        # Pearson residuals of the intercept-only model on the training entries.
        p0 = decoder.log_probs(torch.zeros(len(fit_weight), d, device=device), gq_t).exp()
        m = t(np.repeat(universe * train, arrays.groups[:, 1] - arrays.groups[:, 0], axis=1))
        resid = (counts_t - m * p0) / torch.sqrt(m * p0 + 1e-8)
        scale = torch.sqrt(w_t / w_t.mean())[:, None]
        _, _, V = torch.svd_lowrank(resid * scale, q=min(d + 8, resid.shape[1]), niter=4)
        z0 = resid @ V[:, :d]
        del resid, m, p0
        sd = torch.sqrt((w_t[:, None] * (z0 - (w_t @ z0) / total) ** 2).sum(0) / total)
        z0 = (z0 - (w_t @ z0) / total) / torch.clamp(sd, min=1e-8)
        decoder.W.copy_(0.1 * V[:, :d])
    z = torch.nn.Parameter(z0)

    def objective():
        return data_term(decoder.log_probs(z, gq_t), counts_t, w_t, c_t, total) \
            + ridge_term(z, decoder.W, w_t, total, ridge)

    def closure():
        for p in (z, *decoder.parameters()):
            if p.grad is not None:
                p.grad.zero_()
        loss = objective()
        loss.backward()
        return loss

    with torch.no_grad():
        initial = float(objective())
    trajectory = [{"iteration": 0, "loss": initial, "evaluations": 1, "seconds": 0.0}]
    trajectory += _lbfgs([z, *decoder.parameters()], closure, objective, max_iter=max_iter,
                         tol=tol, label=f"fit d={d} gamma={gamma} ridge={ridge:g}")
    with torch.no_grad():
        data = float(data_term(decoder.log_probs(z, gq_t), counts_t, w_t, c_t, total))
        penalty = float(ridge_term(z, decoder.W, w_t, total, ridge))

    z_np = z.detach().cpu().double().numpy()
    W_np = decoder.W.detach().cpu().double().numpy()
    b_np = decoder.b.detach().cpu().double().numpy()
    O_np = decoder.O.detach().cpu().double().numpy()
    norm = whiten(z_np, W_np, b_np, design)
    before = z_np @ W_np.T + b_np
    after = norm["z"] @ norm["W"].T + norm["b"]
    whitening_error = float(np.abs(before - after).max())
    del before, after

    # Calibrated decoder: positions fixed, c = 1, design-weighted.
    ones = np.where(empty, 0.0, 1.0)
    cal = Decoder(arrays.counts.shape[1], d, arrays.gq.shape[1], arrays.groups, empty).to(device)
    with torch.no_grad():
        cal.W.copy_(t(norm["W"]))
        cal.b.copy_(t(norm["b"]))
        cal.O.copy_(t(O_np))
    z_fixed, d_t, ones_t = t(norm["z"]), t(design), t(ones)
    design_total = float(design.sum())

    def cal_objective():
        return data_term(cal.log_probs(z_fixed, gq_t), counts_t, d_t, ones_t, design_total) \
            + ridge * (cal.W * cal.W).sum()

    def cal_closure():
        for p in cal.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = cal_objective()
        loss.backward()
        return loss

    cal_trajectory = _lbfgs(list(cal.parameters()), cal_closure, cal_objective,
                            max_iter=calibrate_iter, tol=tol, label="calibrate decoder")
    null_b, null_O = intercepts(train_counts, design, arrays.gq, arrays.groups, ones)

    return {
        "settings": {"d": d, "gamma": gamma, "ridge": ridge, "weighted": weighted,
                     "heldout_fraction": heldout_fraction, "masking": masking,
                     "heldout_seed_probability": seed_probability,
                     "heldout_share": float(held.sum() / max((universe > 0).sum(), 1)),
                     "seed": seed, "device": device,
                     "max_iter": max_iter, "tol": tol, "calibrate_iter": calibrate_iter},
        "z": norm["z"], "W": norm["W"], "b": norm["b"], "O": O_np,
        "W_cal": cal.W.detach().cpu().double().numpy(),
        "b_cal": cal.b.detach().cpu().double().numpy(),
        "O_cal": cal.O.detach().cpu().double().numpy(),
        "b_null": null_b, "O_null": null_O,
        "c": c, "empty": empty, "heldout": held,
        "mean": norm["mean"], "sqrt_cov": norm["sqrt_cov"], "eigenvalues": norm["eigenvalues"],
        "trajectory": trajectory, "calibration": cal_trajectory,
        "final": {"loss": trajectory[-1]["loss"], "data": data, "penalty": penalty,
                  "whitening_max_logit_change": whitening_error},
    }


def git_commit() -> str | None:
    import subprocess
    from pathlib import Path

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10, cwd=Path(__file__).resolve().parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_fit(directory, arrays: TargetArrays, fit: dict, checks: dict, config_path) -> dict:
    """Write one fit's outputs to ``directory``; ``result.json``, written last,
    marks it complete.

    * ``phi_units.parquet``: every unit (by ``SERIALNO``) with its merged row,
      weight, unit type, the two group-quarters indicators and the whitened
      coordinates ``z1``..``zd``.
    * ``decoder.npz``: fitted decoder (``W``, ``b``, ``O``; on the whitened
      coordinates), calibrated decoder (``W_cal``...), null intercepts,
      ``c``, the empty-category mask, the whitening (``mean``, ``sqrt_cov``,
      ``eigenvalues``) and the merged-row coordinates ``z``.
    * ``heldout.npz``: the (row, target) held-out mask.
    * ``loss.csv``, ``calibration.csv``: optimiser trajectories.
    * ``check1_targets.csv``, ``check1_categories.csv``, ``check1_rare.csv``,
      ``check1_gq_targets.csv``.
    * ``config.json`` (a copy), ``settings.json``, ``version.json``.
    """
    import hashlib
    import json
    import shutil
    from pathlib import Path

    import pandas as pd

    from pmedm_vb.config import ensure_dir
    from pmedm_vb.embed.check import overall
    from pmedm_vb.progress import save_npz, write_text

    directory = ensure_dir(Path(directory))
    config_path = Path(config_path)
    shutil.copyfile(config_path, directory / "config.json")
    version = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()[:12],
        "targets_key": arrays.meta["key"],
        "git_commit": git_commit(),
    }
    write_text(directory / "version.json", json.dumps(version, indent=1))
    write_text(directory / "settings.json", json.dumps(fit["settings"], indent=1))

    d = fit["z"].shape[1]
    units = arrays.units.copy()
    rows = units["row"].to_numpy()
    units["gq_institutional"] = arrays.gq[rows, 0]
    units["gq_noninstitutional"] = arrays.gq[rows, 1]
    for j in range(d):
        units[f"z{j + 1}"] = fit["z"][rows, j]
    part = directory / ".phi_units.parquet.part"
    units.reset_index().to_parquet(part, index=False)
    part.replace(directory / "phi_units.parquet")

    save_npz(directory / "decoder.npz", **{k: fit[k] for k in (
        "z", "W", "b", "O", "W_cal", "b_cal", "O_cal", "b_null", "O_null", "c", "empty",
        "mean", "sqrt_cov", "eigenvalues")})
    save_npz(directory / "heldout.npz", mask=fit["heldout"], seed=np.array(fit["settings"]["seed"]))
    write_text(directory / "loss.csv", pd.DataFrame(fit["trajectory"]).to_csv(index=False))
    write_text(directory / "calibration.csv", pd.DataFrame(fit["calibration"]).to_csv(index=False))
    for name, frame in checks.items():
        write_text(directory / f"check1_{name}.csv", frame.to_csv(index=False))

    result = {
        **fit["settings"], **fit["final"], **version,
        "iterations": fit["trajectory"][-1]["iteration"],
        "seconds": fit["trajectory"][-1]["seconds"],
        "heldout_mean_deviance_explained": overall(checks["targets"]),
        "eigenvalues": [float(v) for v in fit["eigenvalues"]],
    }
    write_text(directory / "result.json", json.dumps(result, indent=1))
    return result
