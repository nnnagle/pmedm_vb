"""Step 5, phase 2 (and 2b, 2c): the pilot of the realized-population sampler
(``PLAN_latent_step5.md`` version 3, §4 items 2, 2b and 2c, §5 second bullet;
:mod:`pmedm_vb.latent.sampler_numba`; model ``MODEL_latent_v5_draft.md``
§7, the cut form).

    R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs
    PYTHONPATH=src python experiments/latent/step5_pilot.py \\
        --config experiments/embed/stage1_v2.json --regions experiments/latent/regions_tn.json \\
        --run $R/embed/stage1_v2/d16_g0.5_r0.01 --step1 $R/latent/step1_sigma \\
        --step3 $R/latent/step3 --step3b $R/latent/step3b --step4 $R/latent/step4 \\
        --step4c $R/latent/step4c --out $R/latent/step5_pilot --threads 16
    # phase 2b, the rerun with the version 2 moves, beside the first pilot:
    PYTHONPATH=src python experiments/latent/step5_pilot.py ... (as above) \\
        --sweeps 5000 --burn 1000 --mix 0.3,0.2,0.2,0.3 --kernel-basis kern30 \\
        --pile-moves 1N --out $R/latent/step5_pilot2 --compare-root $R/latent/step5_pilot
    # phase 2c, the third pilot: two arms, beside the second pilot
    PYTHONPATH=src python experiments/latent/step5_pilot.py ... (as above) \\
        --sweeps 5000 --burn 1000 --mix 0.3,0.2,0.2,0.3 --kernel-basis kern30 \\
        --pile-moves 1N --pile-first occupied --pile-second kernel --draw-start warm \\
        --out $R/latent/step5_pilot3_kernel --compare-root $R/latent/step5_pilot2
    #   (and the same with --pile-second r_hat --out $R/latent/step5_pilot3_rhat)
    # the dry run, no project data needed:
    PYTHONPATH=src python experiments/latent/step5_pilot.py --synthetic --out /tmp/pilot \\
        --sweeps 300 --burn 100 --threads 4

**Unit** (per ``--region``, default ``knoxville_core`` (4 zones) and
``nashville_core`` (6 zones, the most of any region); replicate
``--replicate`` (0) of simulation S, at step 4c's chosen ``(τ, c)`` for S,
``--step4c``'s ``choice.json``), built as ``step4e_cut.fit_unit`` builds it:
S's data (step 4c's ``Replicate``), the totals module
(:func:`~pmedm_vb.latent.free_shares.cut_totals`), the engine, the MAP of
``U`` given the module's mean (:func:`~pmedm_vb.latent.free_shares.fit_cut_U`),
``p`` = the engine's probabilities at ``Û``, ``κ = c n_A``, ``Y``, and ``Σ``
= S's ``BlockSigma`` (zones independent: the block backend). The sampler's
zone totals are the module's mean rounded to integers summing to ``N_A``
exactly (:func:`round_totals`: floors, then one more unit to the zones with
the largest fractional parts; derived here, the largest-remainder rule).
The shares are those integers over ``N_A``, and the realized MAP ``r̂``
(proposal (b)) and stage 2's moments are computed at them (``Û`` is the MAP
at the unrounded mean; the change is under one unit per zone). Atoms with
``p = 0`` in every zone and parameter draw are dropped
(:func:`~pmedm_vb.latent.sampler.restrict_support`).

**Functionals** traced: step 4e's (fitted cells, B1, B2, rare groups, joint,
the zone total), at most ``--max-per-class`` (100; 0 for all) columns per
class, evenly spaced in the class's order; every rare group. Version 2 adds
the class ``hhsize`` (all 8 columns traced): households (occupied housing
units; group-quarters units are not households and are left out) with 1,
2, …, 6 and 7 or more persons, and persons in households of 7 or more.
Persons per unit = the unit's total over ``B01001``'s cells (sex by age,
whose categories partition persons: the table total and the two sex
subtotals are waived), i.e. the sum of the unit's row of the check data's
``B01001`` counts; where the units carry PUMS ``NP``, the two are compared
over the households and the number that differ is recorded in the unit's
info (``hhsize_np_mismatch``).

**Version 2 moves** (``PLAN_latent_step5.md`` §3a;
:mod:`pmedm_vb.latent.sampler_numba`'s module notes): ``--mix`` takes four
weights ``a,b,c,e`` (three give ``e = 0``, version 1's mixture; default
``0.3,0.2,0.2,0.3``); the kernel of (e) and of the pile moves is the kernel
basis ``--kernel-basis`` (``kern30``, the default, or ``kern100``): its
stored centres and width ``ℓ`` (from step 1's ``inputs/bases``, else step
3b's ``inputs/bases``, else rebuilt by step 1's recipe, which is
deterministic) and the atoms' positions ``z`` (``load_atoms``, the stage 1
run's), restricted to the sampler's atoms; ``--kernel-trunc ε`` truncates
each centre's table to atoms with ``K_jk ≥ ε max_j K_jk`` (default 0:
none); ``--kernel-weights`` ``p`` (default) or ``r_hat``. When the kernel
basis is also the model's ``g`` (``--basis``), its standardized columns are
rebuilt from these centres and compared with the ``g`` in use
(``kernel_check_max_abs`` in the info). ``--pile-moves``: pile moves per
zone per sweep, an integer or a multiple of ``N_b`` written ``<f>N``
(default ``1N``, ``N_b`` pile moves per zone per sweep. Chosen on the
synthetic dry run, not on S: with 0.1N, 0.5N, 1N and 2N the median bulk ESS
per second of B1, joint and hhsize rose up to 1N and was flat from 1N to 2N,
and per sweep it kept rising; at about the pilot's size, on made-up data, a
pile move cost about 2.3 µs of CPU against 1.8–2.2 µs for a single-unit
move, so 1N about doubles a sweep's cost). Reported beside the acceptance: the kernel
tables' size and the pile statistics (pairs tried, null pairs, ``n = 0``,
splits drawn and changed, mean ``|Δ|`` and ``n``, the ``n`` histogram).
``--compare-root``: a first pilot's output root; where
``<root>/<region>/pilot.json`` exists its mixing table (median bulk ESS
per sweep and max ``R̂`` by class) is printed beside the new one.

**Version 3** (``PLAN_latent_step5.md`` §3b, §4 item 2c, §5's version 3
rule; :mod:`pmedm_vb.latent.sampler_numba`'s module notes).

* ``--pile-first``: ``r_hat`` (default, version 2's pile pairs, so that the
  second pilot's command reproduces it) or ``occupied`` (the occupied-record
  Metropolis–Hastings pile move); ``--pile-second``: ``kernel`` (default) or
  ``r_hat`` (the ablation; needs ``--pile-first occupied``). With
  ``--pile-second r_hat`` and kernel weight 0 no kernel tables are built.
* ``--draw-start``: ``cold`` (default; version 2's ``Multinomial(N_b,
  r̂_b)``) or ``warm``: parameter draw ``d``'s chain starts from MAP chain
  ``d mod --chains``'s state after its last sweep (so after burn-in), its
  zone totals adjusted to the draw's (:func:`warm_starts`): units on records
  off the draw's support are removed (none when the draw's support is the
  MAP's; the number is reported), a zone that loses units loses
  them uniformly at random among its units (a multivariate hypergeometric
  draw from its counts), a zone that gains units gets them from the draw's
  ``r̂_b`` restricted to its support. The adjustment uses its own random
  stream, so the cold runs' random choices are unchanged. Starts do not
  change the target, only the burn-in needed. **Re-burn**
  (:func:`reburn`): per draw chain, the first sweep at which its log target
  crosses the median of its *own* second half (the draws have different
  targets, so neither a pooled median nor ``R̂`` across them applies), and the
  start's distance from that median in second-half standard deviations. It
  measures the log target's settling only; a slow functional could take
  longer to forget the start (not checked by this statistic).
* **The mixing rule, version 3** (:func:`rule_v3`; §5, fixed before
  results): with 50 parameter draws per unit and 1,800 s on one 16-core
  node, the projected effective draws per unit for the median functional of
  every class are at least 100 (bulk and tail), and ``R̂ < 1.01`` for at
  least 95% of each class's functionals and below 1.05 for all (at the MAP
  chains). Projection (derived here): per class the median ESS per
  chain-sweep ``e`` at the MAP chains (bulk, tail: ESS over all chains' kept
  sweeps divided by their number); the cold burn-in ``B_c`` (once: the MAP
  chains' suggested burn-in, else ``--burn``), the re-burn ``B_w`` (the
  largest over the draw chains); then each of the 50 draw chains gets
  ``(1800 - B_c t_c)/t_d`` sweeps, of which ``K`` after the re-burn, and the
  unit's effective draws are ``50 K e``. **Cost model** (an assumption): a
  slot-sweep (one sweep of one zone of one chain) costs a fixed
  ``c = t min(P, S)/S`` thread-seconds, ``t`` the measured median seconds
  per sweep (moves and refresh) with ``S`` slots on ``P`` threads, and ``S'``
  slots on the node take ``t' = c S'/min(16, S')`` s per sweep (``t_c`` for
  the cold chains' ``chains × Z`` slots, ``t_d`` for the draws' ``50 Z``).
  Basis: the block backend runs every slot in parallel with dynamic
  scheduling of one slot per chunk, so with at least 16 slots all threads
  stay busy and wall time is total work over threads. Not modelled:
  unequal zone sizes at the end of a sweep (the last slots), memory
  bandwidth shared by more slots, the refresh's serial numpy part, the
  set-up (MAP fit, the draws' ``r̂`` refits, kernel tables; reported, not
  in the budget), and memory (the kernel tables are about 16 K M bytes per
  zone and draw: run the draws in batches of at least 16 slots, which leaves
  the cost unchanged under the model). It also assumes the draw chains mix
  as the MAP chains do (the draws' own median ESS per sweep, a one-chain
  estimate, gives the column ``proj_bulk_draws_ess`` as a check). The
  verdict is printed as PASS only with ``--draw-start warm`` (the rule's
  setting). Version 2's rule is still printed, labelled as replaced.
* ``--compare-root`` may be repeated (e.g. the first and the second
  pilot's roots): for each, the per-class table of median bulk and tail ESS
  per sweep, bulk ESS per second, max ``R̂`` and the share below 1.01, new
  beside previous with the bulk ratios, and the two runs' pile settings and
  shares of null pile pairs.

**At the MAP parameters:** ``--chains`` (4) chains from dispersed starts,
chain ``k`` drawn ``Multinomial(N_b, q_k)`` with ``q`` cycling through
``r̂_b``, ``p_b``, ``½(r̂_b + p_b)`` and ``p_b^{1/2}`` (normalized; flatter
than ``p``), each restricted to the zone's support. ``--burn`` + ``--sweeps``
sweeps (a sweep is ``N_b`` moves in zone ``b``, every zone of every chain in
parallel), every sweep recorded: the log target (up to a constant), the zone
totals check (the largest ``|Σ_j C_bj - N_b|``, which must be 0) and the
functionals. Diagnostics on the sweeps after ``--burn``: rank-normalized
split ``R̂`` and bulk and tail ESS (arviz), per class the median and minimum
ESS per sweep (ESS over all chains' kept sweeps divided by their number,
``chains × sweeps``: the plan's "effective sample size per draw", with one
draw per sweep) and ESS per second of wall time (all chains run together);
seconds per sweep, acceptance per proposal. Functionals constant over the
kept draws (the zone total, and cells no occupied atom reaches) are counted
and left out. **Burn-in** from the log target's trace: for each chain the
first sweep at which it crosses the median of the pooled second half of the
run, and ``R̂`` of the log target over ``[t, end)`` for ``t`` on a grid; the
suggested burn-in is the smallest grid ``t`` after every chain's crossing
with that ``R̂ < 1.01`` (a heuristic, derived here). **Decision rule** (§5,
second bullet, version 2's; replaced from phase 2c by version 3's, above):
median ESS per sweep ≥ 0.1 in every class and ``R̂ < 1.01`` (the largest
over the non-constant functionals reported beside the share below 1.01).

**Agreement with stage 2** (§5's third-bullet thresholds, applied here at
the MAP parameters): the pooled chains' means and variances against
:func:`~pmedm_vb.latent.free_shares.free_share_conditional_moments` at the
MAP parameters, for functionals whose stage 2 mean is at least 50: the share
with ``|mean difference| ≤ 0.1`` stage 2 sd, with variance ratio in
``[0.8, 1.25]``, and with both (pass: both ≥ 90%), with the Monte Carlo
errors (mean: ``sd/√ESS_bulk``; variance ratio: ``ratio·√(2/ESS_bulk)``, a
Gaussian approximation, derived here) so that a gap can be told from noise.

**Parameter draws** (``--param-draws``, 4): draws of the cut posterior as
:func:`~pmedm_vb.latent.free_shares.cut_posterior_samples` makes them
(totals from the module, rounded as above; ``U`` from its Laplace
approximation), one chain each, all run together as one batch, with their
own ``r̂`` (``--draw-rhat refit``, the default) or the MAP's (``map``), each
started from ``Multinomial(N_b, r̂_b)`` or warm (``--draw-start``, above):
ESS per sweep per class for each draw, to see whether mixing differs from
the MAP parameters, and the re-burn.

**Torch** (``--torch-chains k``, default 0): ``k`` chains of the torch
sampler at the MAP parameters, ``--torch-steps`` timed steps after a warm-up,
for seconds per sweep (torch's sweep: ``max N_b`` steps, each moving every
zone) and moves per second, beside numba's. ``--engine torch`` runs the main
chains with the torch sampler instead (slow; traces computed from its counts).

**Outputs** (``<out>/<region>/``): ``map_traces.npz`` (functionals ``(T,
chains, Z, F)`` float32, log target ``(T, chains)``, seconds and totals check
per sweep, the classes and names), ``map_log_target.csv``,
``map_functionals.csv`` (per zone and functional: means, variances, ``R̂``,
ESS, stage 2's moments), ``draws_traces.npz``, ``draws_functionals.csv``,
``summary.txt`` and ``pilot.json`` (written last; a region with it is
skipped, so a timed-out job can be resubmitted unchanged). Regions run one
after another, each with all ``--threads`` (``--workers`` is not used).

**Synthetic** (``--synthetic``): the same pipeline on made-up units with the
real ones' structure (a ``Σ`` block per zone, leave-zone-out supports, one
category per table per atom, rare groups, κ small against ``N_b``), two
"regions" of 4 and 6 zones; ``r̂`` from the same conditional PMEDM; stage 2's
moments by the same Gaussian conditioning written densely
(:func:`dense_conditional_moments`). Version 2: positions ``z`` (8-D, a
random linear map of the atom's fitted-table row plus noise, so that atoms
near in ``z`` have similar profiles), kernel centres by the same weighted
k-means as a kernel basis (``kern30``/``kern100`` sizes), and household
sizes (1–9 persons) for the ``hhsize`` class, all from their own random
streams (the rest of the unit is as before). For checking the driver
without the project data; its numbers say nothing about simulation S.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pmedm_vb.latent.sampler import (
    BlockCovariance,
    ProposalMix,
    SamplerInputs,
    restrict_support,
)
from pmedm_vb.latent.sampler_numba import PILE_N_BINS, KernelSpec, ProposalMix4

#: Default regions: Knoxville core (4 zones) and Nashville core (6, the most).
DEFAULT_REGIONS = ("knoxville_core", "nashville_core")
#: Tag of the pilot's seeds, apart from steps 4c–4e.
PILOT_TAG = 5002
#: The plan's thresholds (§5).
ESS_PER_SWEEP_MIN = 0.1
RHAT_MAX = 1.01
AGREE_MEAN_SD = 0.1
AGREE_VAR = (0.8, 1.25)
AGREE_SHARE = 0.9
AGREE_MIN_MEAN = 50.0
#: Household sizes of the ``hhsize`` class (the last one is "7 or more").
HHSIZE_MAX = 7
#: The mixing rule, version 3 (§5): budget per regional unit, parameter
#: draws per unit, the node's cores, effective draws per unit for the median
#: functional of every class (bulk and tail), and the R-hat shares.
BUDGET_SECONDS = 1800.0
BUDGET_DRAWS = 50
NODE_THREADS = 16
ESS_UNIT_MIN = 100.0
RHAT_SHARE_MIN = 0.95
RHAT_ALL_MAX = 1.05


# --------------------------------------------------------------------------
# The unit
# --------------------------------------------------------------------------


@dataclass
class PilotUnit:
    """One region's sampler inputs at the MAP parameters, the parameter
    draws' inputs, the traced functionals and stage 2's moments (all on the
    restricted support)."""

    name: str
    T0: sp.csr_matrix                    # (M, C)
    Y: np.ndarray                        # (C Z,) stacked
    N: np.ndarray                        # (Z,) integer totals
    p: np.ndarray                        # (Z, M)
    r_hat: np.ndarray                    # (Z, M)
    kappa: float
    cov: BlockCovariance
    G: sp.csr_matrix                     # (M, F)
    cls: np.ndarray                      # (F,)
    names: np.ndarray                    # (F,)
    stage2_mean: np.ndarray              # (Z, F)
    stage2_var: np.ndarray               # (Z, F)
    draws: list = field(default_factory=list)   # [{"N", "p", "r_hat"}]
    info: dict = field(default_factory=dict)
    z: np.ndarray | None = None          # (M, d) atom positions (version 2's kernel)
    kernel: dict = field(default_factory=dict)  # {"centres", "ell", "basis", "source"}
    keep: np.ndarray | None = None       # (M,) the support's atoms, indices into all atoms

    def kernel_spec(self, trunc: float = 0.0, weights: str = "p") -> KernelSpec:
        return KernelSpec(self.z, self.kernel["centres"], self.kernel["ell"], trunc=trunc,
                          weights=weights)

    @property
    def Z(self) -> int:
        return self.N.size

    def inputs(self, n: int) -> SamplerInputs:
        return SamplerInputs(self.T0, self.Y, np.repeat(self.N[None, :], n, 0), self.p,
                             self.r_hat, self.kappa)

    def draw_inputs(self) -> SamplerInputs:
        N = np.stack([d["N"] for d in self.draws])
        return SamplerInputs(self.T0, self.Y, N, np.stack([d["p"] for d in self.draws]),
                             np.stack([d["r_hat"] for d in self.draws]), self.kappa)


def round_totals(x: np.ndarray, N_A: float) -> np.ndarray:
    """Integers summing to ``N_A`` exactly: floors, then one more unit to the
    ``N_A - Σ floor`` zones with the largest fractional parts (ties: the
    lower index)."""
    x = np.asarray(x, float)
    total = int(round(float(N_A)))
    if abs(total - float(N_A)) > 1e-6:
        raise ValueError("N_A is not an integer")
    base = np.floor(x).astype(np.int64)
    short = total - int(base.sum())
    if not 0 <= short <= x.size:
        raise ValueError(f"the totals sum to {x.sum():.3f}, not N_A = {total}")
    order = np.argsort(-(x - base), kind="stable")
    base[order[:short]] += 1
    if np.any(base < 1):
        raise ValueError("a rounded zone total is below 1")
    return base


def select_columns(cls: np.ndarray, max_per_class: int) -> np.ndarray:
    """At most ``max_per_class`` columns per class (0: all), evenly spaced in
    the class's order; every rare group, the total and the household sizes."""
    keep = []
    for c in pd.unique(cls):
        idx = np.flatnonzero(cls == c)
        if max_per_class > 0 and idx.size > max_per_class and c not in ("rare", "total",
                                                                          "hhsize"):
            idx = idx[np.unique(np.linspace(0, idx.size - 1, max_per_class).round().astype(int))]
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def household_size_columns(persons: np.ndarray, household: np.ndarray
                           ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """The ``hhsize`` class (module notes): ``(M, 8)`` columns, households
    with 1, …, 6, 7+ persons and persons in households of 7+; their classes
    and names."""
    persons = np.asarray(persons, float)
    M = persons.size
    size = np.minimum(np.rint(persons).astype(np.int64), HHSIZE_MAX)
    ok = np.asarray(household, bool) & (size >= 1)
    rows = np.flatnonzero(ok)
    G = sp.csr_matrix((np.ones(rows.size), (rows, size[rows] - 1)), shape=(M, HHSIZE_MAX + 1))
    big = np.flatnonzero(ok & (size == HHSIZE_MAX))
    G = (G + sp.csr_matrix((persons[big], (big, np.full(big.size, HHSIZE_MAX))),
                           shape=(M, HHSIZE_MAX + 1))).tocsr()
    names = [f"hhsize:{k}" for k in range(1, HHSIZE_MAX)] + [f"hhsize:{HHSIZE_MAX}plus",
                                                             f"hhsize:persons_in_{HHSIZE_MAX}plus"]
    return G, np.array(["hhsize"] * (HHSIZE_MAX + 1)), np.array(names)


def real_household_sizes(data) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, dict]:
    """The ``hhsize`` columns of the check data (module notes) and a record:
    persons from ``B01001``, households = not group quarters, ``NP``
    compared where present."""
    persons = np.asarray(data.counts["B01001"].sum(axis=1)).ravel()
    household = ~data.units["is_group_quarters"].to_numpy(dtype=bool)
    info = {"hhsize_persons_from": "B01001 row sums", "hhsize_households": int(household.sum()),
            "hhsize_households_without_persons": int((household & (persons < 1)).sum()),
            "hhsize_households_7plus": int((household & (persons >= HHSIZE_MAX)).sum())}
    if "NP" in data.units.columns:
        np_ = pd.to_numeric(data.units["NP"], errors="coerce").to_numpy(dtype=float)
        info["hhsize_np_mismatch"] = int((household & (np_ != persons)).sum())
    else:
        info["hhsize_np_mismatch"] = "NP not among the units' columns"
    G, cls, names = household_size_columns(persons, household)
    return G, cls, names, info


def kernel_centres(name: str, args, atoms, G_used: np.ndarray | None = None) -> dict:
    """Centres and ``ℓ`` of kernel basis ``name`` (module notes): step 1's or
    step 3b's saved files, else rebuilt by step 1's recipe. With ``G_used``
    (the model's ``g``, when it is this basis) its standardized columns are
    rebuilt from these centres and compared (a failed check is recorded, not
    raised)."""
    from pmedm_vb.latent.atoms import GQ_COLUMNS
    from pmedm_vb.latent.basis import kernel_features
    from pmedm_vb.latent.deviation_basis import build_kernel_basis

    def check(G_here):
        if G_used is None:
            return None
        if G_here.shape != G_used.shape:
            return f"shape differs: {G_here.shape} vs {G_used.shape}"
        return float(np.abs(G_here - G_used).max())

    for d in (Path(args.step1) / "inputs" / "bases", Path(args.step3b) / "inputs" / "bases"):
        if (d / f"{name}.npz").exists() and (d / f"{name}.json").exists():
            saved = np.load(d / f"{name}.npz")
            centres = saved["centres"]
            ell = float(json.loads((d / f"{name}.json").read_text())["ell"])
            out = {"centres": centres, "ell": ell, "basis": name, "source": str(d),
                   "check_max_abs": None}
            if G_used is not None:
                # the standardized columns, as area.load_basis rebuilds kern100/kern200
                # (it does not take kern30)
                try:
                    F = np.column_stack([atoms.gq, kernel_features(atoms.z, centres, ell)])
                    names = [*GQ_COLUMNS, *(f"k{j + 1}" for j in range(len(centres)))]
                    pos = {c: j for j, c in enumerate(names)}
                    F = F[:, [pos[str(c)] for c in saved["columns"]]]
                    out["check_max_abs"] = check((F - saved["mean"]) / saved["scale"])
                except Exception as err:                     # a check, not a stop
                    out["check_max_abs"] = f"check failed: {err!r}"
            return out
    basis = build_kernel_basis(name, atoms)
    return {"centres": basis.params["centres"], "ell": float(basis.params["ell"]), "basis": name,
            "source": "rebuilt (step 1's recipe)", "check_max_abs": check(basis.H)}


def restrict_unit(name, T0, Y, N, p, r_hat, kappa, cov, G, cls, names, mean_g, var_g,
                  draws, info, z=None, kernel=None) -> PilotUnit:
    """Drop atoms with ``p = 0`` in every zone of the MAP and every draw."""
    stack = np.concatenate([p[None]] + [d["p"][None] for d in draws])
    keep = restrict_support(stack, T0)["keep"]
    info = {**info, "atoms": int(p.shape[1]), "atoms_on_support": int(keep.size)}
    return PilotUnit(
        name=name, T0=sp.csr_matrix(T0)[keep], Y=np.asarray(Y, float), N=np.asarray(N),
        p=np.ascontiguousarray(p[:, keep]), r_hat=np.ascontiguousarray(r_hat[:, keep]),
        kappa=float(kappa), cov=cov,
        G=sp.csr_matrix(G)[keep], cls=np.asarray(cls), names=np.asarray(names),
        stage2_mean=mean_g, stage2_var=var_g,
        draws=[{"N": d["N"], "p": np.ascontiguousarray(d["p"][:, keep]),
                "r_hat": np.ascontiguousarray(d["r_hat"][:, keep])} for d in draws],
        info=info, z=None if z is None else np.ascontiguousarray(np.asarray(z)[keep]),
        kernel=kernel or {}, keep=keep)


def build_real_unit(region_id: str, args, *, tag: int = PILOT_TAG, replicate: int | None = None,
                    parts: dict | None = None) -> PilotUnit:
    """Simulation S's unit, as ``step4e_cut.fit_unit`` builds it (module notes).
    ``tag``: the parameter draws' seed tag (the pilot's by default);
    ``replicate``: default ``args.replicate``; ``parts``: if a dictionary, it
    is filled with the unit's intermediate objects (the replicate, engine,
    fits, the realized MAP's result on all atoms, the totals module and the
    set-up's timings) for a caller that writes step 4e's files (step 5's
    regional run); the unit returned is the same either way."""
    import step4c_joint as s4c
    import step4e_cut as s4e
    from scipy.linalg import cholesky, solve_triangular

    from pmedm_vb.latent.free_shares import (
        cut_information,
        cut_totals,
        draw_cut_totals,
        fit_cut_U,
        free_share_conditional_moments,
        free_share_engine,
        shares_to_psi,
        solve_conditional_fixed_shares,
    )
    from pmedm_vb.latent.overdispersed import dirmult_phi
    from pmedm_vb.latent.posterior import Functionals
    from pmedm_vb.latent.region_reference import zone_log_bases

    s = s4c._SHARED
    pick = s["choice"][f"S/{region_id}"]
    c, m = pick["c"], pick["tau_multiple"]
    k = args.replicate if replicate is None else replicate
    t0 = time.perf_counter()
    rep = s4c.Replicate("S", region_id, k)
    kappa = c * rep.n_A
    tau = m * s["tau_ref_g"]
    x, Sx = s4e.direct_unit_totals(rep)
    mean, tcov = cut_totals(x, Sx, rep.N_A)
    if not np.all(mean > 0):
        raise ValueError(f"{region_id} r{k}: the totals module gave a non-positive total")
    N_int = round_totals(mean, rep.N_A)
    shares = mean / rep.N_A
    psi = shares_to_psi(shares)
    engine = free_share_engine(rep.area, s["H"], rep.ref["theta"], s["G"],
                               zone_log_bases(rep.ref["logb"], rep.ref["zone"], rep.Z, s4c.MODE),
                               s["pairs"], rep.N_A, kappa)
    fit, starts = fit_cut_U(engine, rep.theta0, tau, psi, gtol=args.map_gtol,
                            max_iter=args.map_max_iter)
    seconds_map = time.perf_counter() - t0
    phi = dirmult_phi(rep.N_A, kappa)
    T0 = s["X"]["A"]
    Y = rep.pA.inputs.targets()

    def realized(p_mz, N):
        res = solve_conditional_fixed_shares(T0, p_mz, N / N.sum(), rep.N_A, rep.N_A / phi, Y,
                                             rep.sigma_dense, tol=args.cond_tol,
                                             max_iter=args.cond_max_iter)
        return res.r, res

    p = engine.probabilities(rep.theta0, fit.U)                           # (M, Z)
    t_r = time.perf_counter()
    r_hat, res = realized(p, N_int)
    seconds_realized = time.perf_counter() - t_r
    # stage 2 at the MAP parameters, the traced columns only
    t_s2 = time.perf_counter()
    fun = s["fun"]
    cols = select_columns(fun.cls, args.max_per_class)
    fsub = Functionals(G=fun.G[:, cols].tocsc(), cls=fun.cls[cols], names=fun.names[cols])
    psi_int = shares_to_psi(N_int / N_int.sum())
    ev = engine.evaluate(rep.theta0, fit.U, psi_int, gradient=False)
    mean_g, var_g, negative = free_share_conditional_moments(engine, ev, fsub, T0)
    seconds_stage2 = time.perf_counter() - t_s2

    # parameter draws of the cut posterior (as cut_posterior_samples)
    rng = np.random.default_rng([args.seed, s4c.SIM_INDEX["S"], rep.index, k, 2,
                                 int(round(m * 1000)), int(round(kappa)), tag])
    draws = []
    seconds_rhat = 0.0
    t_d = time.perf_counter()
    if args.param_draws > 0:
        ev0 = engine.evaluate(rep.theta0, fit.U, fit.psi, gradient=False)
        Rf = cholesky(cut_information(engine, ev0, fit.tau), lower=False)
        xi = rng.standard_normal((fit.U.size, args.param_draws))
        Us = (fit.U.ravel()[:, None] + solve_triangular(Rf, xi, lower=False)).T
        totals = draw_cut_totals(mean, tcov, args.param_draws, rng)
        for d in range(args.param_draws):
            Nd = round_totals(totals[d] + (rep.N_A - totals[d].sum()) / rep.Z, rep.N_A)
            pd_ = engine.probabilities(rep.theta0, Us[d].reshape(engine.Z, engine.K))
            t_r = time.perf_counter()
            rd = realized(pd_, Nd)[0] if args.draw_rhat == "refit" else r_hat
            seconds_rhat += time.perf_counter() - t_r
            draws.append({"N": Nd, "p": pd_.T, "r_hat": rd})
    seconds_draws = time.perf_counter() - t_d
    cov = BlockCovariance.from_factors(rep.sigma, rep.Z)
    info = {"simulation": "S", "region": region_id, "replicate": k, "c": c, "tau_multiple": m,
            "tau": tau, "kappa": kappa, "phi": phi, "N_A": rep.N_A, "n_A": rep.n_A,
            "Z": rep.Z, "cells_per_zone": int(T0.shape[1]),
            "totals_mean": mean.tolist(), "totals_int": N_int.tolist(),
            "totals_sd": np.sqrt(np.clip(np.diag(tcov), 0, None)).tolist(),
            "map_objective": fit.objective, "map_converged": bool(fit.converged),
            "map_grad_max": fit.grad_max, "map_start": fit.start,
            "realized_converged": bool(res.converged), "realized_gap": float(res.gap),
            "stage2_negative_variances": int(negative), "seconds_map": seconds_map,
            "seconds_unit": time.perf_counter() - t0, "draw_rhat": args.draw_rhat,
            "functionals_traced": int(cols.size), "functionals_all": int(fun.F),
            **s["hhsize_info"],
            "kernel_basis": s["kernel"]["basis"], "kernel_source": s["kernel"]["source"],
            "kernel_ell": s["kernel"]["ell"],
            "kernel_check_max_abs": s["kernel"]["check_max_abs"]}
    if parts is not None:
        parts.update(rep=rep, engine=engine, fit=fit, starts=starts, res=res, p=p, N_int=N_int,
                     totals_mean=mean, totals_cov=tcov, x=x, Sx=Sx, shares=shares, cols=cols,
                     seconds={"map": seconds_map, "realized": seconds_realized,
                              "stage2": seconds_stage2, "draws": seconds_draws,
                              "draws_rhat": seconds_rhat})
    return restrict_unit(region_id, T0, Y, N_int, p.T, r_hat, kappa, cov, fsub.G, fsub.cls,
                         fsub.names, mean_g, var_g, draws, info, z=s["z"], kernel=s["kernel"])


# --------------------------------------------------------------------------
# The synthetic unit (dry runs)
# --------------------------------------------------------------------------


def dense_conditional_moments(T0, Y, N, p, kappa, Sigma, G) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2's Gaussian conditioning given the totals (the formulas of
    :func:`~pmedm_vb.latent.free_shares.free_share_conditional_moments`,
    with ``S = Σ + Σ_b c_b D_b V_b D_b'`` built densely): ``(Z, F)`` means
    and variances of ``G'C_b`` given ``Y``."""
    T = sp.csr_matrix(T0).toarray()
    G = sp.csr_matrix(G).toarray()
    Z = p.shape[0]
    C = T.shape[1]
    N = np.asarray(N, float)
    s = N / N.sum()
    a = kappa * s
    c = N * (N + a) / (1.0 + a) if np.isfinite(kappa) else N.copy()
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    S = np.array(Sigma, float, copy=True)
    m = np.zeros(C * Z)
    mu = p @ T                                                            # (Z, C)
    for b in range(Z):
        V = T.T @ (p[b][:, None] * T) - np.outer(mu[b], mu[b])
        S[np.ix_(rows[b], rows[b])] += c[b] * V
        m[rows[b]] = N[b] * mu[b]
    Sinv = np.linalg.inv(0.5 * (S + S.T))
    r = Sinv @ (np.asarray(Y, float) - m)
    mean, var = np.zeros((Z, G.shape[1])), np.zeros((Z, G.shape[1]))
    for b in range(Z):
        Eg = G.T @ p[b]
        E2 = (G * G).T @ p[b]
        kg = T.T @ (p[b][:, None] * G) - np.outer(mu[b], Eg)              # (C, F)
        Gbb = Sinv[np.ix_(rows[b], rows[b])]
        mean[b] = N[b] * Eg + c[b] * (kg.T @ r[rows[b]])
        var[b] = np.maximum(c[b] * (E2 - Eg ** 2)
                            - c[b] ** 2 * np.einsum("ij,ij->j", kg, Gbb @ kg), 0.0)
    return mean, var


def build_synthetic_unit(name: str, Z: int, args, seed: int, *, kappa_c: float = 0.25,
                         parts: dict | None = None) -> PilotUnit:
    """A made-up unit with the real one's structure (module notes).
    ``kappa_c``: ``κ = kappa_c n_A`` (the pilot's 0.25 by default);
    ``parts``: if a dictionary, it is filled with every functional (not only
    the traced ones: ``G_all``, ``cls_all``, ``names_all``), the made-up
    truth, ``Σ`` and the traced columns; the unit returned is the same
    either way."""
    from pmedm_vb.latent.free_shares import solve_conditional_fixed_shares
    from pmedm_vb.latent.overdispersed import dirmult_phi

    t0 = time.perf_counter()
    rng = np.random.default_rng([seed, Z, PILOT_TAG])
    M = args.synthetic_atoms
    n_cat = 5                                             # categories per table
    home = rng.integers(0, Z, M)                          # each atom's own zone (LZO)

    def onehot_tables(k, weights=None):
        cols = []
        for t in range(k):
            cat = rng.integers(0, n_cat, M)
            v = np.ones(M) if weights is None else weights
            cols.append(sp.csr_matrix((v, (np.arange(M), cat)), shape=(M, n_cat)))
        return sp.hstack(cols).tocsr()

    persons = rng.integers(1, 5, M).astype(float)
    T0 = sp.hstack([onehot_tables(4), onehot_tables(4, persons)]).tocsr()  # (M, 40)
    C = T0.shape[1]
    B1, B2 = onehot_tables(6), onehot_tables(4)
    t1, t2 = rng.integers(0, n_cat, M), rng.integers(0, n_cat, M)
    joint = sp.csr_matrix((np.ones(M), (np.arange(M), t1 * n_cat + t2)), shape=(M, n_cat ** 2))
    rare = sp.csr_matrix((rng.random((M, 15)) < 0.004).astype(float))
    total = sp.csr_matrix(np.ones((M, 1)))
    # version 2: household sizes and positions, from their own streams
    rng_h = np.random.default_rng([seed, Z, PILOT_TAG, 7])
    hh_size = persons + rng_h.geometric(0.6, M) - 1              # 7+ for about 1.6%
    household = rng_h.random(M) > 0.02
    Ghh, cls_hh, names_hh = household_size_columns(hh_size, household)
    Gfull = sp.hstack([T0, B1, B2, rare, joint, Ghh, total]).tocsr()
    cls = np.array(["fitted"] * C + ["B1"] * B1.shape[1] + ["B2"] * B2.shape[1]
                   + ["rare"] * 15 + ["joint"] * joint.shape[1] + list(cls_hh) + ["total"])
    names = np.array([f"{c}:{k}" for c, k in zip(cls, range(cls.size))])
    names[cls == "hhsize"] = names_hh
    kernel, z = synthetic_kernel(T0, args.kernel_basis, [seed, Z, PILOT_TAG, 8])
    cols = select_columns(cls, args.max_per_class)

    N = rng.integers(args.synthetic_N // 2, args.synthetic_N + 1, Z)
    N_A = int(N.sum())
    n_A = N_A // 8
    kappa = kappa_c * n_A
    logq = rng.normal(0.0, 1.0, M)
    p = np.exp(logq[None, :] + rng.normal(0.0, 0.7, (Z, M)))
    p[home[None, :] == np.arange(Z)[:, None]] = 0.0       # leave-zone-out supports
    p /= p.sum(1, keepdims=True)
    shares = N / N_A
    truth = np.stack([rng.multinomial(N[b], rng.dirichlet(kappa * shares[b] * p[b] + 1e-300))
                      for b in range(Z)])
    fitted = truth @ T0.toarray()                          # (Z, C)
    rows = np.arange(C)[None, :] * Z + np.arange(Z)[:, None]
    blocks = []
    for b in range(Z):
        f = fitted[b]
        Bf = rng.normal(0.0, 1.0, (C, 2)) * np.sqrt(0.5 * f.mean())
        blocks.append(np.diag(7.0 * np.maximum(f, 5.0)) + Bf @ Bf.T + 4.0 * np.eye(C))
    cov = BlockCovariance(np.array(blocks))
    Sigma = cov.dense()
    Y = np.zeros(C * Z)
    for b in range(Z):
        Y[rows[b]] = fitted[b] + np.linalg.cholesky(blocks[b]) @ rng.standard_normal(C)
    phi = dirmult_phi(N_A, kappa)

    def realized(pz, Nz):
        res = solve_conditional_fixed_shares(T0, pz.T, Nz / Nz.sum(), N_A, N_A / phi, Y, Sigma,
                                             tol=args.cond_tol, max_iter=args.cond_max_iter)
        return res.r, res

    r_hat, res = realized(p, N)
    mean_g, var_g = dense_conditional_moments(T0, Y, N, p, kappa, Sigma, Gfull[:, cols])
    draws = []
    for d in range(args.param_draws):
        x = N + rng.normal(0.0, 0.01 * N.mean(), Z)
        Nd = round_totals(x + (N_A - x.sum()) / Z, N_A)
        pd_ = p * np.exp(rng.normal(0.0, 0.05, (Z, M)))
        pd_ /= pd_.sum(1, keepdims=True)
        rd = realized(pd_, Nd)[0] if args.draw_rhat == "refit" else r_hat
        draws.append({"N": Nd, "p": pd_, "r_hat": rd})
    info = {"simulation": "synthetic", "region": name, "Z": Z, "N_A": N_A, "n_A": n_A,
            "kappa": kappa, "phi": phi, "cells_per_zone": C, "totals_int": N.tolist(),
            "realized_converged": bool(res.converged), "realized_gap": float(res.gap),
            "seconds_unit": time.perf_counter() - t0, "draw_rhat": args.draw_rhat,
            "functionals_traced": int(cols.size), "functionals_all": int(cls.size),
            "kernel_basis": kernel["basis"], "kernel_source": kernel["source"],
            "kernel_ell": kernel["ell"]}
    if parts is not None:
        parts.update(G_all=Gfull, cls_all=cls, names_all=names, truth=truth, Sigma=Sigma,
                     cols=cols, res=res)
    return restrict_unit(name, T0, Y, N, p, r_hat, kappa, cov, Gfull[:, cols], cls[cols],
                         names[cols], mean_g, var_g, draws, info, z=z, kernel=kernel)


def synthetic_kernel(T0: sp.csr_matrix, basis: str, seed) -> tuple[dict, np.ndarray]:
    """Positions ``z`` (8-D: a random linear map of the atom's row of ``T0``
    plus noise, columns standardized) and kernel centres by a kernel basis's
    recipe (unweighted k-means, ``ℓ`` the median distance to the nearest
    centre) for the synthetic units (module notes)."""
    from pmedm_vb.latent.basis import nearest_distance, weighted_kmeans
    from pmedm_vb.latent.deviation_basis import DEVIATION_BASES

    rng = np.random.default_rng(seed)
    M, C = T0.shape
    z = np.asarray(T0 @ rng.normal(0.0, 1.0, (C, 8))) + 0.5 * rng.normal(0.0, 1.0, (M, 8))
    z = (z - z.mean(0)) / z.std(0)
    centres, _ = weighted_kmeans(z, np.ones(M), DEVIATION_BASES[basis], seed=1)
    ell = float(np.median(nearest_distance(z, centres)))
    return {"centres": centres, "ell": ell, "basis": basis,
            "source": "synthetic (k-means on made-up z)", "check_max_abs": None}, z


# --------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------


def dispersed_starts(unit: PilotUnit, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(n, Z, M)`` start counts (module notes)."""
    from pmedm_vb.latent.sampler_numba import multinomial_counts

    support = unit.p > 0
    rh = np.where(support, unit.r_hat, 0.0)
    rh = np.where(rh.sum(1, keepdims=True) > 0, rh, unit.p)
    weights = [rh, unit.p, 0.5 * rh / rh.sum(1, keepdims=True) + 0.5 * unit.p,
               np.sqrt(unit.p)]
    out = np.empty((n, unit.Z, unit.p.shape[1]), dtype=np.int64)
    for k in range(n):
        out[k] = multinomial_counts(weights[k % len(weights)], unit.N[None, :], rng)[0]
    return out


def start_names(n: int) -> list[str]:
    names = ["r_hat", "p", "mix", "sqrt_p"]
    return [names[k % len(names)] for k in range(n)]


def parse_pile_moves(spec: str) -> tuple[str, float]:
    """``--pile-moves``: ``"<k>"`` (k per zone per sweep) or ``"<f>N"``
    (``round(f N_b)`` per zone per sweep)."""
    spec = str(spec).strip()
    if spec.upper().endswith("N"):
        f = float(spec[:-1]) if spec[:-1] else 1.0
        if f < 0:
            raise ValueError("--pile-moves must be non-negative")
        return "fraction", f
    k = int(spec)
    if k < 0:
        raise ValueError("--pile-moves must be non-negative")
    return "count", float(k)


def pile_counts(spec: tuple[str, float], N: np.ndarray) -> np.ndarray:
    """Pile moves per zone per sweep for totals ``N`` (any shape)."""
    kind, v = spec
    N = np.asarray(N)
    if kind == "fraction":
        return np.rint(v * N).astype(np.int64)
    return np.full(N.shape, int(v), dtype=np.int64)


def run_chains(engine: str, inputs: SamplerInputs, cov, mix: ProposalMix4, *, seed: int,
               start_counts: np.ndarray | None, sweeps: int, G: sp.csr_matrix,
               threads: int | None, kernel: KernelSpec | None = None,
               piles: np.ndarray | None = None, pile_first: str = "r_hat",
               pile_second: str = "kernel") -> dict:
    """Every sweep recorded: functionals, log target, totals check, seconds.
    ``kernel``, ``piles`` (pile moves per zone per sweep, ``(n, Z)``): the
    version 2 moves (numba only); ``pile_first``, ``pile_second``: version
    3's choice of the pile pair (numba only). ``final_counts``: the chains'
    counts after the last sweep (``(n, Z, M)``; the warm starts' source)."""
    has_piles = piles is not None and np.max(piles, initial=0) > 0
    if engine == "numba":
        from pmedm_vb.latent.sampler_numba import NumbaSampler

        t0 = time.perf_counter()
        sm = NumbaSampler(inputs, cov, mix, seed=seed, start_counts=start_counts,
                          threads=threads, kernel=kernel,
                          pile_moves=piles if has_piles else 0, pile_first=pile_first,
                          pile_second=pile_second)
        setup = time.perf_counter() - t0
        res = sm.run(sweeps, thin=1, G=G, trace=True)
        return {"functionals": res.functionals.astype(np.float32), "log_target": res.log_target,
                "totals_error": res.totals_error, "seconds": res.seconds_per_sweep,
                "acceptance": res.acceptance, "drift": res.drift,
                "moves_per_sweep": res.steps_per_sweep, "threads": res.threads,
                "seconds_setup": setup, "seconds_trace": res.seconds_functionals,
                "piles": res.piles, "piles_per_sweep": res.piles_per_sweep,
                "kernel": sm.kernel_info(), "final_counts": sm.counts_numpy()}
    from pmedm_vb.latent.sampler import RealizedSampler
    from pmedm_vb.latent.sampler_numba import log_target_numpy

    if mix.kernel > 0 or has_piles or pile_first != "r_hat" or pile_second != "kernel":
        raise ValueError("the torch engine has no kernel proposal or pile moves: use "
                         "--mix with three weights and --pile-moves 0")
    mix = ProposalMix(mix.occupancy, mix.realized, mix.prior)
    t0 = time.perf_counter()
    sm = RealizedSampler(inputs, cov, mix, seed=seed, start_counts=start_counts)
    setup = time.perf_counter() - t0
    n, Z, M, _ = inputs.shape
    fun, lt, te, secs, drift, t_trace = [], [], [], [], 0.0, 0.0
    for _ in range(sweeps):
        s, d = sm.run_sweeps(1)
        secs.append(s[0])
        drift = max(drift, d)
        t1 = time.perf_counter()
        counts = sm.counts_numpy()
        fun.append((sp.csr_matrix(counts.reshape(n * Z, M)) @ G).toarray()
                   .reshape(n, Z, -1).astype(np.float32))
        lt.append(log_target_numpy(inputs, cov, counts))
        te.append(-1 if counts.min() < 0 else int(np.abs(counts.sum(2) - inputs.N).max()))
        t_trace += time.perf_counter() - t1
    return {"functionals": np.stack(fun), "log_target": np.stack(lt),
            "totals_error": np.asarray(te), "seconds": np.asarray(secs),
            "acceptance": sm.acceptance(), "drift": drift,
            # torch: a block step moves every zone once, a low-rank step one zone
            "moves_per_sweep": sm.steps_per_sweep * (Z if cov.kind == "block" else 1),
            "threads": None, "seconds_setup": setup, "seconds_trace": t_trace,
            "piles": None, "piles_per_sweep": 0, "kernel": None,
            "final_counts": sm.counts_numpy()}


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def _arviz():
    import logging
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import arviz as az
    logging.getLogger("arviz").setLevel(logging.ERROR)    # one-chain shape warnings
    return az


def mixing(x: np.ndarray) -> dict:
    """``x``: ``(chains, draws, ...)``. Rank-normalized split R̂, bulk and tail
    ESS (arviz), pooled means and variances; NaN where ``x`` is constant."""
    import warnings

    az = _arviz()
    x = np.asarray(x, float)
    ds = az.convert_to_dataset({"x": x})
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="arviz")
    var = x.reshape(-1, *x.shape[2:]).var(axis=0, ddof=1)
    const = ~(var > 1e-12 * np.maximum(np.abs(x.mean((0, 1))), 1.0) ** 2)
    out = {"mean": x.mean((0, 1)), "var": var, "constant": const}
    for key, kw in (("rhat", {"method": "rank"}), ("ess_bulk", {"method": "bulk"}),
                    ("ess_tail", {"method": "tail"})):
        f = az.rhat if key == "rhat" else az.ess
        v = np.asarray(f(ds, **kw)["x"].values, float)
        out[key] = np.where(const, np.nan, v)
    return out


def class_table(cls: np.ndarray, mix_: dict, draws_total: int, seconds_kept: float) -> pd.DataFrame:
    """Per class: median and minimum ESS per sweep (bulk, tail), ESS per
    second, R̂ (max, share < 1.01), constants."""
    rows = []
    Zc = np.broadcast_to(cls[None, :], mix_["rhat"].shape)
    for c in list(pd.unique(cls)) + ["all"]:
        sel = np.ones(Zc.shape, bool) if c == "all" else Zc == c
        live = sel & ~mix_["constant"]
        eb, et, rh = mix_["ess_bulk"][live], mix_["ess_tail"][live], mix_["rhat"][live]
        row = {"cls": c, "n": int(sel.sum()), "constant": int((sel & mix_["constant"]).sum())}
        if live.any():
            row.update({
                "ess_bulk_per_sweep_median": float(np.median(eb)) / draws_total,
                "ess_bulk_per_sweep_min": float(np.min(eb)) / draws_total,
                "ess_tail_per_sweep_median": float(np.median(et)) / draws_total,
                "ess_tail_per_sweep_min": float(np.min(et)) / draws_total,
                "ess_bulk_per_second_median": float(np.median(eb)) / max(seconds_kept, 1e-300),
                "ess_tail_per_second_median": float(np.median(et)) / max(seconds_kept, 1e-300),
                "rhat_max": float(np.max(rh)), "rhat_share_below": float(np.mean(rh < RHAT_MAX))})
        rows.append(row)
    return pd.DataFrame(rows)


def burn_in(lt: np.ndarray) -> dict:
    """``lt``: ``(T, chains)`` log target. Each chain's first crossing of the
    pooled second half's median, and R̂ of the log target over ``[t, T)``
    (module notes)."""
    T, n = lt.shape
    med = float(np.median(lt[T // 2:]))
    crossing = []
    for k in range(n):
        side = np.sign(lt[:, k] - med)
        changed = np.flatnonzero(side != side[0])
        crossing.append(int(changed[0]) if changed.size else T)
    grid = sorted({int(t) for t in np.linspace(0, T // 2, 11)})
    rh = {}
    for t in grid:
        if T - t >= 8 and n >= 2:
            rh[t] = float(mixing(lt[t:].T[:, :, None])["rhat"][0])
    ok = [t for t, r in rh.items() if t >= max(crossing) and np.isfinite(r) and r < RHAT_MAX]
    return {"median_second_half": med, "first_crossing": crossing, "rhat_from": rh,
            "suggested_burn": min(ok) if ok else None}


def agreement(mean_s, var_s, ess, mean_g, var_g, cls) -> tuple[pd.DataFrame, dict]:
    """Stage 2 against the chains (module notes), functionals with stage 2
    mean ≥ 50 and positive variance."""
    Zc = np.broadcast_to(cls[None, :], mean_g.shape)
    sel = (mean_g >= AGREE_MIN_MEAN) & (var_g > 0) & np.isfinite(ess)
    sd_g = np.sqrt(np.where(var_g > 0, var_g, np.nan))
    dm = (mean_s - mean_g) / sd_g
    ratio = var_s / np.where(var_g > 0, var_g, np.nan)
    se_dm = np.sqrt(var_s / np.maximum(ess, 1.0)) / sd_g
    se_ratio = ratio * np.sqrt(2.0 / np.maximum(ess, 1.0))
    ok_m = np.abs(dm) <= AGREE_MEAN_SD
    ok_v = (ratio >= AGREE_VAR[0]) & (ratio <= AGREE_VAR[1])
    rows = []
    for c in list(pd.unique(cls)) + ["all"]:
        s_ = sel & ((Zc == c) if c != "all" else True)
        if not s_.any():
            continue
        rows.append({"cls": c, "n": int(s_.sum()),
                     "share_mean_ok": float(ok_m[s_].mean()), "share_var_ok": float(ok_v[s_].mean()),
                     "share_both_ok": float((ok_m & ok_v)[s_].mean()),
                     "median_abs_dm_sd": float(np.median(np.abs(dm[s_]))),
                     "max_abs_dm_sd": float(np.max(np.abs(dm[s_]))),
                     "median_mc_se_dm_sd": float(np.median(se_dm[s_])),
                     "var_ratio_median": float(np.median(ratio[s_])),
                     "var_ratio_min": float(np.min(ratio[s_])),
                     "var_ratio_max": float(np.max(ratio[s_])),
                     "median_mc_se_ratio": float(np.median(se_ratio[s_]))})
    tab = pd.DataFrame(rows)
    allrow = tab[tab["cls"] == "all"]
    verdict = {"n": int(sel.sum()),
               "pass": bool(len(allrow) and allrow["share_both_ok"].iloc[0] >= AGREE_SHARE)}
    return tab, verdict


def acceptance_text(acc: dict) -> str:
    return ", ".join(f"{k} {acc[k]['rate']:.3f} (non-null {acc[k]['rate_non_null']:.3f})"
                     for k in acc if acc[k]["proposed"])


def pile_text(piles: dict | None) -> str:
    """One line of pile statistics (module notes)."""
    if not piles:
        return "pile moves: none"
    hist = piles["n_histogram"]
    tot = max(sum(hist.values()), 1)
    return (f"pile moves: {piles['tried']} tried, {piles['null_pair']} null pairs (j = i or an "
            f"empty table), {piles['n_zero']} with n = 0, {piles['off_support']} off the "
            f"support, {piles['drawn']} splits drawn, {piles['changed']} changed "
            f"({piles['share_changed']:.3f}); mean |Delta| {piles['mean_abs_delta']:.3f}, mean n "
            f"{piles['mean_n']:.3f}; n histogram (share of non-null pairs) "
            + ", ".join(f"{b}: {hist[b] / tot:.3f}" for b in PILE_N_BINS))


def kernel_text(info: dict | None) -> str:
    if not info:
        return "kernel tables: none"
    return (f"kernel tables: {info['centres']} centres, {info['tables']} weight rows "
            f"(w = {info['weights']}), truncation {info['trunc']:g}: median "
            f"{info['kept_per_centre_median']:.0f} atoms per centre (min "
            f"{info['kept_per_centre_min']}, {info['kept_share']:.3f} of all), "
            f"{info['empty_tables']} empty tables, {info['megabytes']:.1f} MB")


def compare_table(roots, region: str, tab: pd.DataFrame, piles: dict | None = None
                  ) -> tuple[list, list | None]:
    """Lines comparing the mixing table with earlier pilots' (module notes),
    one block per root in ``roots`` (a path, a list of paths or ``None``):
    per class the median bulk and tail ESS per sweep, the median bulk ESS per
    second, the max R-hat and the share below 1.01, new and earlier ("prev"),
    and the new/prev ratios of the bulk ESS (per second: all chains' kept
    sweeps' wall time, so it includes the moves' cost); then the two runs'
    pile settings and share of null pile pairs."""
    if roots is None:
        return [], None
    if isinstance(roots, (str, Path)):
        roots = [roots]
    lines, records = [], []
    cols = ["ess_bulk_per_sweep_median", "ess_tail_per_sweep_median",
            "ess_bulk_per_second_median", "rhat_max", "rhat_share_below"]
    for root in roots:
        path = Path(root) / region / "pilot.json"
        if not path.exists():
            lines.append(f"  (compare: no {path})")
            continue
        old = json.loads(path.read_text())
        prev = pd.DataFrame(old["mixing"])
        for c in cols:
            if c not in prev:
                prev[c] = np.nan
        both = (tab.set_index("cls")[cols].join(prev.set_index("cls")[cols], how="outer",
                                                lsuffix="_new", rsuffix="_prev"))
        both["ratio_per_sweep"] = both["ess_bulk_per_sweep_median_new"] / both[
            "ess_bulk_per_sweep_median_prev"]
        both["ratio_per_second"] = both["ess_bulk_per_second_median_new"] / both[
            "ess_bulk_per_second_median_prev"]
        both = both.rename(columns=lambda c: c.replace("ess_bulk_per_", "bulk_")
                           .replace("ess_tail_per_", "tail_").replace("_median", "")
                           .replace("rhat_share_below", "rhat_lt1.01"))
        secs_old = old.get("seconds_per_sweep_median")
        op = old.get("piles") or {}
        mode_old = op.get("mode", f"{old.get('pile_first', 'r_hat')}/"
                                  f"{old.get('pile_second', 'kernel')}" if op else "none")
        null_old = (_null_share(op) if op else float("nan"))
        lines += [f"  beside {path} (mix {old.get('mix')}, pile moves {old.get('pile_moves')} "
                  f"{mode_old}, {secs_old if secs_old is None else round(secs_old, 3)} s per "
                  f"sweep, {old.get('sweeps')} kept sweeps):",
                  _indent(both.to_string(float_format=lambda v: f"{v:.4g}")),
                  f"    share of null pile pairs: new {_null_share(piles):.3f}, prev "
                  f"{null_old:.3f}"]
        records.append({"path": str(path), "table": both.reset_index().to_dict(orient="records"),
                        "seconds_per_sweep_prev": secs_old, "pile_mode_prev": mode_old,
                        "null_share_prev": null_old})
    return lines, records


def _null_share(piles: dict | None) -> float:
    """Share of pile pairs tried that were null (``j = i`` or an empty
    table, ``n = 0``, off the support)."""
    if not piles or not piles.get("tried"):
        return float("nan")
    return (piles["null_pair"] + piles["n_zero"] + piles["off_support"]) / piles["tried"]


# --------------------------------------------------------------------------
# Version 3: warm-started parameter draws and the mixing rule on a budget
# --------------------------------------------------------------------------


def warm_starts(map_counts: np.ndarray, unit: PilotUnit, rng: np.random.Generator
                ) -> tuple[np.ndarray, dict]:
    """Start counts ``(draws, Z, M)`` for the parameter draws' chains from the
    MAP chains' last state (module notes): draw ``d`` from MAP chain
    ``d mod chains``; per zone, units on records off the draw's support are
    removed first (none where the supports agree); then a zone with too many
    units loses the excess uniformly at random among its units (a
    multivariate hypergeometric draw from its counts), and a zone with too
    few gains the deficit drawn ``Multinomial(·, r̂_b)``, the draw's ``r̂_b``
    restricted to its support (uniform on the support if ``r̂_b`` misses
    it)."""
    n_map, Z, M = map_counts.shape
    out = np.empty((len(unit.draws), Z, M), dtype=np.int64)
    info = {"source_chain": [], "added": [], "removed": [], "removed_off_support": []}
    for d, dr in enumerate(unit.draws):
        k = d % n_map
        c = np.asarray(map_counts[k], dtype=np.int64).copy()
        support = np.asarray(dr["p"]) > 0
        add, rem, off = [], [], []
        for b in range(Z):
            cb = c[b]
            off.append(int(cb[~support[b]].sum()))
            cb[~support[b]] = 0
            target, cur = int(dr["N"][b]), int(cb.sum())
            if cur > target:
                cb -= rng.multivariate_hypergeometric(cb, cur - target, method="marginals")
            elif cur < target:
                q = np.where(support[b], np.asarray(dr["r_hat"][b], float), 0.0)
                if not q.sum() > 0:
                    q = support[b].astype(float)
                cb += rng.multinomial(target - cur, q / q.sum())
            add.append(max(target - cur, 0))
            rem.append(max(cur - target, 0))
        if not np.array_equal(c.sum(1), np.asarray(dr["N"])) or c.min() < 0:
            raise RuntimeError("warm start: the adjusted totals do not match the draw's")
        out[d] = c
        info["source_chain"].append(k)
        info["added"].append(add)
        info["removed"].append(rem)
        info["removed_off_support"].append(off)
    return out, info


def reburn(lt: np.ndarray) -> dict:
    """``lt``: ``(T, chains)`` log target of chains with different targets
    (one per parameter draw). Per chain: the first sweep at which it crosses
    the median of its *own* second half (the pooled median of
    :func:`burn_in` would mix targets), and how far the start is from that
    median in units of the second half's sd (``start_z``)."""
    T, n = lt.shape
    crossing, start_z = [], []
    for k in range(n):
        tail = lt[T // 2:, k]
        med = float(np.median(tail))
        side = np.sign(lt[:, k] - med)
        changed = np.flatnonzero(side != side[0])
        crossing.append(int(changed[0]) if changed.size else T)
        sd = float(np.std(tail))
        start_z.append(float((lt[0, k] - med) / sd) if sd > 0 else float("nan"))
    return {"first_crossing": crossing, "start_z": start_z, "max": int(max(crossing)),
            "median": float(np.median(crossing))}


def rule_v3(tab: pd.DataFrame, *, seconds_per_sweep: float, slots: int, threads: int,
            chains: int, Z: int, burn_cold: int, burn_warm: int,
            draws_ess: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """The plan's §5 mixing rule, version 3 (module notes): projected
    effective draws per regional unit under the budget, and the R-hat shares.

    Cost model (an assumption, stated in the module notes): a slot-sweep (one
    sweep of one zone of one chain) costs ``c = t · min(P, S)/S``
    thread-seconds, ``t`` the measured seconds per sweep with ``S`` slots on
    ``P`` threads, and ``S'`` slots on the node's 16 threads take
    ``c · S'/min(16, S')`` seconds per sweep."""
    c_slot = seconds_per_sweep * min(threads, slots) / slots
    s_cold = chains * Z
    t_cold = c_slot * s_cold / min(NODE_THREADS, s_cold)
    s_draw = BUDGET_DRAWS * Z
    t_draw = c_slot * s_draw / min(NODE_THREADS, s_draw)
    left = BUDGET_SECONDS - burn_cold * t_cold
    per_draw = left / t_draw
    kept = per_draw - burn_warm
    live = tab[(tab["cls"] != "all") & tab["ess_bulk_per_sweep_median"].notna()].copy()
    k_eff = max(kept, 0.0)
    live["proj_bulk"] = BUDGET_DRAWS * k_eff * live["ess_bulk_per_sweep_median"]
    live["proj_tail"] = BUDGET_DRAWS * k_eff * live["ess_tail_per_sweep_median"]
    if draws_ess:
        live["proj_bulk_draws_ess"] = [BUDGET_DRAWS * k_eff * draws_ess.get(c, np.nan)
                                       for c in live["cls"]]
    live["rhat_ok"] = ((live["rhat_share_below"] >= RHAT_SHARE_MIN)
                       & (live["rhat_max"] < RHAT_ALL_MAX))
    live["ess_ok"] = (live["proj_bulk"] >= ESS_UNIT_MIN) & (live["proj_tail"] >= ESS_UNIT_MIN)
    ok = bool(len(live) and kept > 0 and live["ess_ok"].all() and live["rhat_ok"].all())
    cost = {"seconds_per_sweep_measured": seconds_per_sweep, "slots_measured": slots,
            "threads_measured": threads, "thread_seconds_per_slot_sweep": c_slot,
            "seconds_per_sweep_cold": t_cold, "slots_cold": s_cold,
            "seconds_per_sweep_draws": t_draw, "slots_draws": s_draw,
            "burn_cold": burn_cold, "seconds_burn_cold": burn_cold * t_cold,
            "burn_warm": burn_warm, "sweeps_per_draw": per_draw, "kept_per_draw": kept,
            "budget_seconds": BUDGET_SECONDS, "draws": BUDGET_DRAWS,
            "node_threads": NODE_THREADS, "pass": ok}
    cols = ["cls", "n", "ess_bulk_per_sweep_median", "ess_tail_per_sweep_median", "proj_bulk",
            "proj_tail", *(["proj_bulk_draws_ess"] if draws_ess else []), "rhat_share_below",
            "rhat_max", "ess_ok", "rhat_ok"]
    return live[cols].reset_index(drop=True), cost


# --------------------------------------------------------------------------
# One region
# --------------------------------------------------------------------------


def pilot_region(unit: PilotUnit, args, out: Path) -> dict:
    from pmedm_vb.progress import logger, save_npz, write_text

    out.mkdir(parents=True, exist_ok=True)
    mix = ProposalMix4(*args.mix)
    piles_spec = parse_pile_moves(args.pile_moves)
    kspec = None
    if mix.kernel > 0 or (pile_counts(piles_spec, unit.N).max() > 0
                          and args.pile_second == "kernel"):
        kspec = unit.kernel_spec(args.kernel_trunc, args.kernel_weights)
    pile_mode = {"pile_first": args.pile_first, "pile_second": args.pile_second}
    n = args.chains
    T = args.burn + args.sweeps
    seed_base = [args.seed if args.seed is not None else 0, PILOT_TAG,
                 sum(map(ord, unit.name))]
    rng = np.random.default_rng(seed_base)
    lines = [f"step 5 pilot: {unit.name} ({unit.info.get('simulation')}), engine {args.engine}",
             "unit: " + json.dumps({k: v for k, v in unit.info.items()
                                    if not isinstance(v, list)}, default=float),
             f"zone totals N_b = {unit.N.tolist()} (sum {int(unit.N.sum())}); atoms on the "
             f"support {unit.T0.shape[0]}, cells per zone {unit.T0.shape[1]}, largest row nnz "
             f"{int(np.diff(unit.T0.indptr).max())}; kappa {unit.kappa:.1f}; functionals traced "
             f"{unit.G.shape[1]} per zone; proposal mix a/b/c/e "
             f"{[round(float(w), 3) for w in mix.weights()]}; pile moves per zone per sweep "
             f"{pile_counts(piles_spec, unit.N).tolist()} (--pile-moves {args.pile_moves}; "
             f"first record {args.pile_first}, second {args.pile_second}); parameter-draw "
             f"chains {args.draw_start}-started; kernel basis {unit.kernel.get('basis')} (ell {unit.kernel.get('ell', float('nan')):.4g}, "
             f"{unit.kernel.get('source')})"]

    # -- MAP parameters, dispersed starts -----------------------------------
    inputs = unit.inputs(n)
    starts = dispersed_starts(unit, n, rng)
    t0 = time.perf_counter()
    ch = run_chains(args.engine, inputs, unit.cov, mix, seed=int(rng.integers(2 ** 31)),
                    start_counts=starts, sweeps=T, G=unit.G, threads=args.threads,
                    kernel=kspec, piles=pile_counts(piles_spec, inputs.N), **pile_mode)
    wall = time.perf_counter() - t0
    fun, lt = ch["functionals"], ch["log_target"]
    kept = slice(args.burn, T)
    mx = mixing(np.moveaxis(fun[kept], 0, 1))                        # (Z, F)
    draws_total = n * (T - args.burn)
    secs = ch["seconds"]
    seconds_kept = float(secs[kept].sum())
    tab = class_table(unit.cls, mx, draws_total, seconds_kept)
    lt_mix = mixing(lt[kept].T[:, :, None])
    bi = burn_in(lt)
    compare_lines, compare = compare_table(args.compare_root, unit.name, tab, ch["piles"])
    agree_tab, agree = agreement(mx["mean"], mx["var"], mx["ess_bulk"], unit.stage2_mean,
                                 unit.stage2_var, unit.cls)
    live = tab[(tab["cls"] != "all") & tab["ess_bulk_per_sweep_median"].notna()]
    rh_all = tab.loc[tab["cls"] == "all", "rhat_max"].iloc[0] if "rhat_max" in tab else np.nan
    mixing_pass = bool(len(live) and (live["ess_bulk_per_sweep_median"] >= ESS_PER_SWEEP_MIN).all()
                       and np.isfinite(rh_all) and rh_all < RHAT_MAX)
    Zi, Fi = np.meshgrid(np.arange(unit.Z), np.arange(unit.G.shape[1]), indexing="ij")
    frame = pd.DataFrame({"zone": Zi.ravel(), "cls": unit.cls[Fi.ravel()],
                          "functional": unit.names[Fi.ravel()],
                          "mean": mx["mean"].ravel(), "var": mx["var"].ravel(),
                          "rhat": mx["rhat"].ravel(), "ess_bulk": mx["ess_bulk"].ravel(),
                          "ess_tail": mx["ess_tail"].ravel(), "constant": mx["constant"].ravel(),
                          "stage2_mean": unit.stage2_mean.ravel(),
                          "stage2_var": unit.stage2_var.ravel()})
    write_text(out / "map_functionals.csv", frame.to_csv(index=False))
    write_text(out / "map_log_target.csv", pd.DataFrame(
        {"sweep": np.arange(1, T + 1), "seconds": secs, "totals_error": ch["totals_error"],
         **{f"chain{k}_{nm}": lt[:, k] for k, nm in enumerate(start_names(n))}}
    ).to_csv(index=False))
    save_npz(out / "map_traces.npz", functionals=fun, log_target=lt, seconds=secs,
             totals_error=ch["totals_error"], cls=unit.cls, names=unit.names, N=unit.N,
             starts=np.array(start_names(n)), burn=args.burn)
    lines += ["", f"== MAP parameters: {n} chains (starts {start_names(n)}), {T} sweeps "
              f"({args.burn} burn-in), {ch['moves_per_sweep']} moves per chain per sweep, "
              f"threads {ch['threads']}",
              f"  seconds per sweep: median {np.median(secs):.3f}, mean {secs.mean():.3f} "
              f"(moves + refresh); traces {ch['seconds_trace']:.1f} s in all; set-up "
              f"{ch['seconds_setup']:.1f} s; wall {wall:.0f} s",
              f"  single-unit moves per second (all chains): "
              f"{n * ch['moves_per_sweep'] / np.median(secs):.3g}; pile moves per chain per sweep "
              f"{ch['piles_per_sweep']}",
              f"  acceptance: {acceptance_text(ch['acceptance'])}",
              f"  {pile_text(ch['piles'])}",
              f"  {kernel_text(ch['kernel'])}",
              f"  zone totals check: max |sum C_b - N_b| over sweeps = "
              f"{int(np.max(ch['totals_error']))} (0 = holds); drift {ch['drift']:.1e}",
              f"  log target: R-hat {lt_mix['rhat'][0]:.4f}, bulk ESS {lt_mix['ess_bulk'][0]:.0f} "
              f"({lt_mix['ess_bulk'][0] / draws_total:.3f} per sweep)",
              f"  burn-in: first crossing of the second half's median per chain "
              f"{bi['first_crossing']}; R-hat of the log target from sweep t: "
              + ", ".join(f"{t}: {r:.3f}" for t, r in bi["rhat_from"].items())
              + f"; suggested burn-in {bi['suggested_burn']} (used {args.burn})",
              "  mixing by class (ESS per sweep = ESS / (chains x kept sweeps)):",
              _indent(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}")),
              f"  DECISION (plan §5, version 2's rule; replaced by version 3's below from phase "
              f"2c): median bulk ESS per sweep >= {ESS_PER_SWEEP_MIN} in every "
              f"class and max R-hat < {RHAT_MAX}: {'PASS' if mixing_pass else 'FAIL'}",
              *compare_lines,
              "", f"== agreement with stage 2 at the MAP parameters (stage 2 mean >= "
              f"{AGREE_MIN_MEAN:g}; |dm| <= {AGREE_MEAN_SD} sd, variance ratio in {AGREE_VAR}; "
              f"MC se from bulk ESS):",
              _indent(agree_tab.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
              if len(agree_tab) else "  (no functional with stage 2 mean >= 50)",
              f"  {AGREE_SHARE:.0%} of the {agree['n']} with both: "
              f"{'PASS' if agree['pass'] else 'FAIL'}"]

    # -- parameter draws ----------------------------------------------------
    draws_summary = None
    if unit.draws:
        dinp = unit.draw_inputs()
        nd = len(unit.draws)
        Td = args.burn + (args.draw_sweeps or args.sweeps)
        t1 = time.perf_counter()
        dstart, winfo = None, None
        if args.draw_start == "warm":
            # its own stream, so the cold runs' random choices are those of pilot 2
            dstart, winfo = warm_starts(ch["final_counts"], unit,
                                        np.random.default_rng([*seed_base, 3]))
        dch = run_chains(args.engine, dinp, unit.cov, mix, seed=int(rng.integers(2 ** 31)),
                         start_counts=dstart, sweeps=Td, G=unit.G, threads=args.threads,
                         kernel=kspec, piles=pile_counts(piles_spec, dinp.N), **pile_mode)
        rb = reburn(dch["log_target"])
        dk = slice(args.burn, Td)
        per = []
        rows = []
        for d in range(nd):
            m1 = mixing(np.moveaxis(dch["functionals"][dk, d:d + 1], 0, 1))
            t_ = class_table(unit.cls, m1, Td - args.burn, float(dch["seconds"][dk].sum()))
            t_.insert(0, "draw", d)
            per.append(t_)
            rows.append(pd.DataFrame({"draw": d, "zone": Zi.ravel(), "cls": unit.cls[Fi.ravel()],
                                      "functional": unit.names[Fi.ravel()],
                                      "mean": m1["mean"].ravel(), "var": m1["var"].ravel(),
                                      "rhat_split": m1["rhat"].ravel(),
                                      "ess_bulk": m1["ess_bulk"].ravel(),
                                      "ess_tail": m1["ess_tail"].ravel()}))
        per = pd.concat(per, ignore_index=True)
        write_text(out / "draws_functionals.csv", pd.concat(rows).to_csv(index=False))
        save_npz(out / "draws_traces.npz", functionals=dch["functionals"],
                 log_target=dch["log_target"], seconds=dch["seconds"],
                 totals_error=dch["totals_error"], N=np.stack([d["N"] for d in unit.draws]),
                 cls=unit.cls, names=unit.names, burn=args.burn)
        med = per[per["cls"] != "all"].pivot(index="draw", columns="cls",
                                             values="ess_bulk_per_sweep_median")
        draws_summary = {"draws": nd, "sweeps": Td, "acceptance": dch["acceptance"],
                         "piles": dch["piles"], "kernel": dch["kernel"],
                         "seconds_per_sweep": float(np.median(dch["seconds"])),
                         "ess_bulk_per_sweep_median_by_draw": med.round(4).to_dict(orient="index"),
                         "totals_error_max": int(np.max(dch["totals_error"])),
                         "drift": dch["drift"], "wall": time.perf_counter() - t1,
                         "start": args.draw_start, "warm": winfo, "reburn": rb}
        start_text = ("Multinomial(N_b, r_hat_b)" if args.draw_start == "cold" else
                      "warm: MAP chain d mod chains after its last sweep, totals adjusted")
        lines += ["", f"== {nd} parameter draws of the cut posterior, one chain each (start "
                  f"{start_text}; r_hat {unit.info.get('draw_rhat')}), {Td} sweeps "
                  f"({args.burn} burn-in); median {np.median(dch['seconds']):.3f} s per sweep for "
                  "all draws together",
                  f"  totals per draw: {[d['N'].tolist() for d in unit.draws]}",
                  *([f"  warm starts: source MAP chain {winfo['source_chain']}; units added "
                     f"(from r_hat_b) {winfo['added']}, removed (at random) {winfo['removed']}, "
                     f"removed off the draw's support {winfo['removed_off_support']}"]
                    if winfo else []),
                  f"  re-burn ({args.draw_start} start): first crossing of each chain's own "
                  f"second-half median {rb['first_crossing']} (max {rb['max']}, median "
                  f"{rb['median']:g}); start's distance from it {[round(z, 1) for z in rb['start_z']]}"
                  " second-half sd",
                  f"  acceptance: {acceptance_text(dch['acceptance'])}; totals check "
                  f"{draws_summary['totals_error_max']}; drift {dch['drift']:.1e}",
                  f"  {pile_text(dch['piles'])}",
                  f"  {kernel_text(dch['kernel'])}",
                  "  median bulk ESS per sweep by class (rows: draws; MAP chains for reference "
                  "below):",
                  _indent(med.to_string(float_format=lambda v: f"{v:.4f}")),
                  _indent(live.set_index("cls")["ess_bulk_per_sweep_median"].to_frame("MAP")
                          .T.to_string(float_format=lambda v: f"{v:.4f}"))]

    # -- the mixing rule, version 3 (§5) ---------------------------------------
    burn_cold = bi["suggested_burn"]
    cold_note = "the MAP chains' suggested burn-in"
    if burn_cold is None:
        burn_cold, cold_note = args.burn, "no suggested burn-in: the burn-in used (fallback)"
    if draws_summary is not None:
        burn_warm = draws_summary["reburn"]["max"]
        warm_note = (f"the largest re-burn over the {draws_summary['draws']} "
                     f"{args.draw_start}-started draw chains")
        dmed = per[(per["cls"] != "all")].groupby("cls")["ess_bulk_per_sweep_median"].median()
        draws_ess = dmed.to_dict()
    else:
        burn_warm, warm_note, draws_ess = burn_cold, "no parameter draws: the cold burn-in", None
    rule_tab, rule = rule_v3(tab, seconds_per_sweep=float(np.median(secs)), slots=n * unit.Z,
                             threads=int(ch["threads"] or args.threads or 1), chains=n,
                             Z=unit.Z, burn_cold=int(burn_cold), burn_warm=int(burn_warm),
                             draws_ess=draws_ess)
    rule.update({"burn_cold_source": cold_note, "burn_warm_source": warm_note,
                 "draw_start": args.draw_start})
    v3_pass = rule["pass"] and args.draw_start == "warm"
    lines += ["", f"== mixing rule, version 3 (plan §5): {BUDGET_DRAWS} parameter draws per unit, "
              f"{BUDGET_SECONDS:.0f} s on one {NODE_THREADS}-core node",
              f"  cost model (an assumption): a slot-sweep (one zone of one chain) costs "
              f"c = t min(P, S)/S = {rule['thread_seconds_per_slot_sweep']:.4g} thread-seconds "
              f"(t = {rule['seconds_per_sweep_measured']:.4g} s per sweep measured with S = "
              f"{rule['slots_measured']} slots on P = {rule['threads_measured']} threads); S' "
              f"slots on {NODE_THREADS} threads take c S'/min({NODE_THREADS}, S') s per sweep",
              f"  cold burn-in once: {rule['burn_cold']} sweeps ({cold_note}) of {n} chains "
              f"({rule['slots_cold']} slots, {rule['seconds_per_sweep_cold']:.4g} s per sweep): "
              f"{rule['seconds_burn_cold']:.0f} s",
              f"  {BUDGET_DRAWS} draw chains ({rule['slots_draws']} slots, "
              f"{rule['seconds_per_sweep_draws']:.4g} s per sweep; run in batches of at least "
              f"{NODE_THREADS} slots the cost is the same under this model): "
              f"{rule['sweeps_per_draw']:.0f} sweeps each, re-burn {rule['burn_warm']} "
              f"({warm_note}), so {rule['kept_per_draw']:.0f} kept sweeps per draw",
              f"  set-up not in the budget above: the unit's set-up took "
              f"{unit.info.get('seconds_unit', float('nan')):.0f} s here "
              f"({len(unit.draws)} parameter draws' r_hat)",
              "  projected effective draws per unit (median functional of each class = "
              f"{BUDGET_DRAWS} x kept sweeps x ESS per chain-sweep at the MAP chains; "
              "proj_bulk_draws_ess: the same from the draw chains' own median ESS per sweep):",
              _indent(rule_tab.to_string(index=False, float_format=lambda v: f"{v:.4g}")),
              f"  DECISION (plan §5, version 3): projected bulk and tail >= {ESS_UNIT_MIN:g} for "
              f"every class's median functional, R-hat < {RHAT_MAX} for >= {RHAT_SHARE_MIN:.0%} "
              f"of each class's functionals and < {RHAT_ALL_MAX} for all: "
              + ("PASS" if v3_pass else "FAIL")
              + ("" if args.draw_start == "warm" else
                 " (the draws were cold-started: the rule assumes warm starts; rerun with "
                 "--draw-start warm)")]

    # -- torch comparison -----------------------------------------------------
    torch_summary = None
    if args.torch_chains > 0:
        from pmedm_vb.latent.sampler import RealizedSampler

        k = args.torch_chains
        # torch has (a)-(c) only: their weights, renormalized
        sm = RealizedSampler(unit.inputs(k), unit.cov,
                             ProposalMix(mix.occupancy, mix.realized, mix.prior), seed=1)
        sm.time_steps(10)
        per_step = sm.time_steps(args.torch_steps)
        tsweep = per_step * sm.steps_per_sweep
        torch_summary = {"chains": k, "seconds_per_step": per_step,
                         "steps_per_sweep": sm.steps_per_sweep, "seconds_per_sweep": tsweep,
                         "moves_per_second": k * unit.Z / per_step}
        nmps = n * ch["moves_per_sweep"] / float(np.median(secs))
        lines += ["", f"== torch, {k} chains: {1e3 * per_step:.2f} ms per step (every zone of "
                  f"every chain moves once), {tsweep:.1f} s per torch sweep ({sm.steps_per_sweep} "
                  f"steps); {k * unit.Z / per_step:.3g} moves/s vs numba's {nmps:.3g} with {n} "
                  "chains (torch: proposals (a)-(c) only, no pile moves)"]

    summary = {
        "region": unit.name, "engine": args.engine, "info": unit.info, "chains": n,
        "sweeps": args.sweeps, "burn": args.burn, "mix": list(mix.weights()),
        "pile_moves": args.pile_moves, "piles_per_sweep": ch["piles_per_sweep"],
        "pile_first": args.pile_first, "pile_second": args.pile_second,
        "draw_start": args.draw_start, "rule_v3": rule,
        "rule_v3_table": rule_tab.to_dict(orient="records"), "rule_v3_pass": v3_pass,
        "piles": ch["piles"], "kernel": ch["kernel"], "kernel_trunc": args.kernel_trunc,
        "kernel_weights": args.kernel_weights, "kernel_basis": unit.kernel.get("basis"),
        "compare": compare,
        "moves_per_sweep": ch["moves_per_sweep"], "threads": ch["threads"],
        "seconds_per_sweep_median": float(np.median(secs)), "wall": wall,
        "acceptance": ch["acceptance"], "drift": ch["drift"],
        "totals_error_max": int(np.max(ch["totals_error"])),
        "log_target": {"rhat": float(lt_mix["rhat"][0]), "ess_bulk": float(lt_mix["ess_bulk"][0])},
        "burn_in": bi, "mixing": tab.to_dict(orient="records"), "mixing_pass": mixing_pass,
        "agreement": agree_tab.to_dict(orient="records"), "agreement_pass": agree["pass"],
        "draws": draws_summary, "torch": torch_summary}
    text = "\n".join(lines) + "\n"
    write_text(out / "summary.txt", text)
    write_text(out / "pilot.json", json.dumps(summary, indent=1, default=_json_default))
    print(text, flush=True)
    logger.info("pilot %s: mixing (v2 rule) %s, mixing (v3 rule) %s, agreement %s", unit.name,
                "PASS" if mixing_pass else "FAIL", "PASS" if v3_pass else "FAIL",
                "PASS" if agree["pass"] else "FAIL")
    return summary


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def _json_default(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    import step4d_free_shares as s4d

    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    s4d.common_arguments(parser)
    parser.set_defaults(region=list(DEFAULT_REGIONS))
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=2000, help="kept sweeps (after --burn)")
    parser.add_argument("--burn", type=int, default=500, help="burn-in sweeps (also traced)")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--engine", default="numba", choices=("numba", "torch"))
    parser.add_argument("--mix", type=lambda x: [float(v) for v in x.split(",")],
                        default=[0.3, 0.2, 0.2, 0.3],
                        help="proposal weights a,b,c,e (three: e = 0, version 1's mixture)")
    parser.add_argument("--kernel-basis", default="kern30", choices=("kern30", "kern100"),
                        help="kernel of proposal (e) and of the pile moves")
    parser.add_argument("--kernel-trunc", type=float, default=0.0,
                        help="relative truncation of each centre's table (0: none)")
    parser.add_argument("--kernel-weights", default="p", choices=("p", "r_hat"))
    parser.add_argument("--pile-moves", default="1N",
                        help="pile moves per zone per sweep: an integer, or <f>N for f N_b")
    parser.add_argument("--compare-root", type=Path, action="append", default=None,
                        help="an earlier pilot's output root, to print its mixing table beside "
                             "(repeatable)")
    parser.add_argument("--pile-first", default="r_hat", choices=("r_hat", "occupied"),
                        help="pile pairs' first record: from r_hat (version 2, the default) or "
                             "a uniformly picked unit's (version 3)")
    parser.add_argument("--pile-second", default="kernel", choices=("kernel", "r_hat"),
                        help="pile pairs' second record: the kernel route or r_hat (the "
                             "ablation; with --pile-first occupied)")
    parser.add_argument("--draw-start", default="cold", choices=("cold", "warm"),
                        help="parameter-draw chains: Multinomial(N_b, r_hat_b) (cold, the "
                             "default) or the MAP chains' last state with adjusted totals (warm)")
    parser.add_argument("--max-per-class", type=int, default=100)
    parser.add_argument("--param-draws", type=int, default=4)
    parser.add_argument("--draw-sweeps", type=int, default=None,
                        help="kept sweeps per parameter draw (default --sweeps)")
    parser.add_argument("--draw-rhat", default="refit", choices=("refit", "map"))
    parser.add_argument("--torch-chains", type=int, default=0)
    parser.add_argument("--torch-steps", type=int, default=200)
    parser.add_argument("--synthetic", action="store_true",
                        help="made-up units, no project data (a dry run)")
    parser.add_argument("--synthetic-atoms", type=int, default=3000)
    parser.add_argument("--synthetic-N", type=int, default=6000)
    if "--synthetic" in argv:                          # the data arguments are not needed
        for action in parser._actions:
            if action.required and action.dest != "out":
                action.required = False
    args = parser.parse_args(argv)
    if len(args.mix) == 3:
        args.mix = [*args.mix, 0.0]
    if len(args.mix) != 4:
        parser.error("--mix takes four weights a,b,c,e (or three, with e = 0)")
    try:
        parse_pile_moves(args.pile_moves)
    except ValueError as err:
        parser.error(f"--pile-moves: {err}")
    if args.pile_first == "r_hat" and args.pile_second != "kernel":
        parser.error("--pile-second r_hat needs --pile-first occupied")
    return args


def prepare_version2(s: dict, args) -> None:
    """Version 2's shared inputs (module notes): the ``hhsize`` columns
    appended to the functionals, the atoms' positions and the kernel."""
    from pmedm_vb.latent.atoms import load_atoms
    from pmedm_vb.latent.posterior import Functionals
    from pmedm_vb.progress import logger

    Ghh, cls_hh, names_hh, info = real_household_sizes(s["data"])
    fun = s["fun"]
    s["fun"] = Functionals(G=sp.hstack([fun.G, Ghh]).tocsc(), cls=np.append(fun.cls, cls_hh),
                           names=np.append(fun.names, names_hh))
    s["hhsize_info"] = info
    atoms = load_atoms(s["data"], args.run)
    s["z"] = np.ascontiguousarray(atoms.z)
    s["kernel"] = kernel_centres(args.kernel_basis, args, atoms,
                                 G_used=s["G"] if args.kernel_basis == args.basis else None)
    del atoms
    logger.info("version 2: hhsize %s; kernel %s (%d centres, ell %.4g, %s; check %s)",
                info, args.kernel_basis, len(s["kernel"]["centres"]), s["kernel"]["ell"],
                s["kernel"]["source"], s["kernel"]["check_max_abs"])


def main(argv=None) -> None:
    args = parse_args(argv)
    from pmedm_vb.progress import logger

    if args.engine == "numba":
        import numba

        if args.threads:
            numba.set_num_threads(min(args.threads, numba.config.NUMBA_NUM_THREADS))
        logger.info("numba %s, %d threads (NUMBA_NUM_THREADS %d)", numba.__version__,
                    numba.get_num_threads(), numba.config.NUMBA_NUM_THREADS)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        if args.seed is None:
            args.seed = 0
        todo = [(f"synthetic_z{Z}", Z) for Z in (4, 6)]
        for name, Z in todo:
            out = args.out / name
            if (out / "pilot.json").exists():
                logger.info("%s: done", name)
                continue
            unit = build_synthetic_unit(name, Z, args, args.seed)
            pilot_region(unit, args, out)
        return
    import step4c_joint as s4c
    import step4d_free_shares as s4d

    pending = [r for r in args.region if not (args.out / r / "pilot.json").exists()]
    for r in set(args.region) - set(pending):
        logger.info("%s: done", r)
    if not pending:
        return
    args.region = pending
    _regions, chosen, choice = s4d.setup(args, {
        "step": "5 pilot", "pilot_tag": PILOT_TAG, "replicate": args.replicate,
        "sweeps": args.sweeps, "burn": args.burn, "chains": args.chains, "engine": args.engine,
        "mix": args.mix, "max_per_class": args.max_per_class, "param_draws": args.param_draws,
        "draw_rhat": args.draw_rhat, "kernel_basis": args.kernel_basis,
        "kernel_trunc": args.kernel_trunc, "kernel_weights": args.kernel_weights,
        "pile_moves": args.pile_moves, "pile_first": args.pile_first,
        "pile_second": args.pile_second, "draw_start": args.draw_start})
    s4c._SHARED["choice"] = choice
    prepare_version2(s4c._SHARED, args)
    for rid in chosen:
        unit = build_real_unit(rid, args)
        pilot_region(unit, args, args.out / rid)


if __name__ == "__main__":
    main()
