# Handoff: end-to-end comparison of the methods

Project: nnnagle/pmedm_vb -- Penalized MaxEnt Dasymetric Modeling as Variational
Bayes. Research code for a paper comparing VB with the MAP/Laplace formulation.
Compute runs on ISAAC (UTK HPC); the user runs jobs there and pastes output
back. You cannot reach ISAAC -- give commands to run, and push code for the
user to pull.

## Read first, in order

1. `README.md` -- Specs, ToDo, Open research questions, **Every session** (the
   ISAAC login steps).
2. `solvers_synopsis.md` -- all of it, especially findings 9-12, "Proposed
   method" and "Left to do". This is the record of what was found and why.
3. `pmedm_derivation.md` -- the model and its dual.
4. Code: `src/pmedm_vb/solvers/` (`base.py`, `map_dual.py`, `vb.py`,
   `mcmc.py`), `src/pmedm_vb/compare.py`, `src/pmedm_vb/simulate.py` (a
   stub), and `experiments/` (`run_map.py`, `run_mcmc.py`,
   `compare_vb_hmc.py`, `weight_draws.py`, `laplace_diagnostic.py` and their
   `.sbatch` wrappers).

## Where things stand

- **MAP** converges on all four Knox PUMAs (4701501-4701504) for alpha from 1
  to 0.01, in seconds.
- **VB** converges in three families -- `gaussian`, `skewed` (per-coordinate
  sinh-arcsinh) and `sumdiff` (skew also on tract + block group sums and
  differences). Gaussian and skewed have been fitted at alpha 1 and 0.1,
  sumdiff at alpha 1 only. **No VB fit has been run at alpha 0.01.**
- **HMC** (`solvers/mcmc.py`, whitened by a VB fit's Gaussian part) gives a
  reference posterior. It has been run on **one PUMA (4701502) at alpha 1
  only**. There the posterior never put over 0.62% of N on one cell; VB put
  over 1% of N on one cell in 4.6-8.6% of draws, and its rare-group 90%
  intervals were as little as a tenth of HMC's width (findings 9, 10, 12).
- **VB skew + a short HMC run** (200 warmup + 1,000 samples, trajectory 3.0)
  reproduced the reference on 4701502 in a few minutes on a V100 (finding
  11). It is the proposed simulator, not yet checked on other PUMAs or alphas.
- Caveats carried forward: the reference used so far (run 6273419) stopped
  after ~2,300 of 10,000 sampling iterations; the short runs shared its seed
  and starting points; the job id of the one finished full-length run was not
  recorded. A clean reference is part of this task.

## Settled -- do not relitigate

The unit is the household or GQ person; one problem per PUMA; alpha in (0, 1],
alpha = 1 is classic diagonal PMEDM; tract taper is the default; n is the
sample size (record count) wherever it appears. HMC jobs run on the
`campus-gpu` partition and QoS, which gives a V100: elsewhere they can land on
a T4, about 30x slower for this float64 code. The recent VB and HMC runs all
used `--variance-floor zero`; confirm with the user whether the comparison
keeps it.

## Clean slate: rerun everything

Every result in the comparison is produced fresh in this task, even where an
earlier run exists: assembly, MAP, all three VB families, the short HMC runs
and the HMC references, for every (PUMA, alpha). The earlier runs (below and
in the synopsis) are history -- use them to cross-check that the fresh
results agree with what was found, never as inputs. So:

- One commit for the whole grid. Record it; `run_args.jsonl` in every results
  folder records each job's commit, settings and job id.
- Rebuild the assembled inputs first (`run_map.py assemble --rebuild`).
- Refit MAP and VB for every family at every alpha, with the same settings
  across the grid (64 draws per step was the last setting used; confirm).
- HMC on `campus-gpu` (V100) only, a distinct `--seed` per run, and the short
  runs whitened by the fresh VB fits, not the old ones. Reference runs must
  finish: set `--time` from a timing test, and resubmit with `--out` if one
  stops (the sampler checkpoints).
- Keep a table of job ids -> (step, PUMA, alpha, family, seed) as jobs are
  submitted, in the repository, so no result is untraceable.

## The task

An end-to-end comparison of seven methods, crossed with alpha in {1, 0.1,
0.01}, on the four Knox PUMAs, evaluated on **timing** and on **usability for
inference tasks and for simulation tasks**.

| # | Method | Code that exists | Code still to write |
|---|---|---|---|
| 0 | Raking (IPF) | nothing | everything; feasibility is an open question (below) |
| 1 | MAP | `run_map.py solve` | Laplace draws for simulation (`StructuredGaussian.laplace` exists; nothing saves or scores it) |
| 2 | VB Gaussian | `run_map.py vb --family gaussian` | -- |
| 3 | VB skewed | `--family skewed` | -- |
| 4 | VB sumdiff | `--family sumdiff` | -- |
| 5 | VB skew + short HMC | `run_mcmc.py --warmup 200 --samples 1000` | -- |
| 6 | HMC reference | `run_mcmc.py`, full length | -- |

Every row is run afresh for every (PUMA, alpha); none has been run at alpha
0.01 before, and only 4701502 at alpha 1 has an HMC reference so far.

**Raking may not be feasible, and finding out is part of the job.** The
constraints are published estimates at two levels that do not agree -- a
tract estimate is not the sum of its block groups' -- and many small cells are
published as 0 or with large SEs, so classic IPF (Deming & Stephan 1940) has
no exact solution to converge to. Options to put to the user: rake to one
level only, rake to a reconciled set of margins, or a relaxed raking with a
convergence tolerance. Raking has no Sigma, so its result does not depend on
alpha, and it gives one weight matrix, so like MAP its only simulation is
resampling from that matrix.

**Alpha 0.01 is new ground.** Sigma then keeps 99% of the replicate
correlation, and its residual diagonal is small. How VB and HMC behave there
is unknown -- mixing, step size and run time may all change. The saddlepoint
evidence cannot choose alpha (synopsis, "Other notes"), so alpha is a factor
in the comparison, not something the comparison selects.

## Evaluation -- a starting proposal; agree the definitions with the user before coding

- **Timing.** Wall time per (method, PUMA, alpha) on stated hardware (CPU
  cores, or a V100), split into setup, fit and drawing; for samplers also time
  per 1,000 effective draws (ESS). Sources: `seconds` in `summary.csv`, the
  `done:` line of HMC logs, `run_args.jsonl` in every results folder.
- **Inference tasks** -- estimates and their uncertainty from one fit:
  - constrained block group cells, against the published values and SEs;
  - unconstrained cross-tabulations (race x household income is built in;
    `compare.outcome_matrix` takes others);
  - possibly held-out published tables, i.e. a table left out of the
    constraints and predicted, which gives a truth that does not depend on
    HMC. The README lists held-out prediction as a candidate for choosing
    alpha. Which tables, and whether to do this at all, is the user's call.
  - Against the HMC reference: `compare_vb_hmc.py` section 4 already gives
    median and mean differences in reference sds, sd ratios and 90% width
    ratios (cells with reference width >= 5 people).
- **Simulation tasks** -- synthetic populations drawn from each method:
  - plausibility: the share of draws putting over 1% or 10% of N on one
    (block group, record) cell (`weight_draws.py`, `run_mcmc.py report`);
  - agreement with the reference distribution: `compare_vb_hmc.py` sections
    1-3 and 5 (PSIS k-hat, whitened coordinates, wall cells, tract/block
    group pairs);
  - `simulate.py` is a stub (`simulate_population`, `expected_counts` raise
    `NotImplementedError`). Drawing integer populations -- and whether
    population-level multinomial noise is part of the comparison -- needs
    implementing and agreeing.
- **Deliverable.** Probably one driver over the grid plus a summary table per
  (method, alpha), aggregated over PUMAs; its form is for the user to decide.

## Gaps to plan for

- `compare_vb_hmc.py` scores a saved VB family against an HMC trace. MAP and
  raking give a single weight matrix, and Laplace is not saved at all; they
  need a path into the same comparison.
- `run_map.py` fits every PUMA in one sweep; `run_mcmc.py` does one (PUMA,
  alpha) per job. 4701501 is the largest problem (8,577 multipliers; the dense
  whitening matrix is ~590 MB).
- `run_map.sbatch` runs on the `short` partition, capped at one hour; VB at
  64 draws per step took 11-19 minutes per fit on the largest problems, so
  plan the sweep (or a longer partition) around that.
- The code added since synopsis finding 8 (statewide support, HMC,
  comparison, `sumdiff`) is checked by scratch scripts only, not by tests in
  `tests/`. The user asked for no test scripts unless requested; ask before
  relying on it for the paper.

## Earlier runs -- history only, for cross-checking

Under `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/` (details in the
synopsis's "Runs referenced"). Do not use these as inputs to the comparison:

- `6272357` -- VB skewed, 64 draws per step, alpha 1, all four PUMAs; the
  whitening for the HMC runs so far.
- `6275353` -- VB sumdiff, 64 draws per step, alpha 1 (4701502 scored
  against HMC; check its `summary.csv` for which PUMAs it covers).
- `6271480` -- VB skewed, 8 draws per step, alpha 1 and 0.1.
- `6273419` -- HMC, 4701502, alpha 1, truncated; reference so far.
- `6275306`, `6275393` -- short HMC, 4701502, alpha 1 (T4, V100).

## Working rules

Clarify before generating code; no test scripts, example scripts or
documentation unless the user asks; do not state unverified claims as fact --
say when you are unsure, or look it up. The user is experienced in R and newer
to Python. Develop on the branch the session assigns; commit and push there;
no PR unless asked. Record each job's id and settings as results arrive
(`run_args.jsonl` does the settings); this session lost track of one.

**First step:** read the files above, then put the evaluation definitions,
the raking question, the alpha 0.01 plan and the run plan for the clean
grid (settings, seeds, partitions, time limits) to the user before writing
code or submitting jobs.
