# Handoff: a latent-space PMEDM — stage 1, an embedding of the statewide PUMS

Project: nnnagle/pmedm_vb — Penalized MaxEnt Dasymetric Modeling with
MAP/VB/HMC solvers. Compute runs on ISAAC (UTK HPC). The user runs jobs there
and pastes output back; you cannot reach ISAAC. Give commands, and push code
for the user to pull.

**Branch:** `stage1-embedding`, started from `tn-statewide-gpu`, which holds
everything below (statewide support, GPU bundles, the queue driver, the τ
runs, sim04). `main` does not.

## Working with this user

- Codes mainly in Python but is newer to it; a very experienced R programmer.
- **Clarify before writing code.** Propose a plan and ask about real choices.
- **No test scripts, example scripts or documentation unless asked.**
- **No unverified claims as fact.** Say when something is derived, assumed, or
  only tested on stand-ins. Look literature up before citing it; mark
  from-memory citations as such.
- Prefers implementation done by background agents, with a concise report
  back: files, commit, what was tested and how, ISAAC commands in order.
- Commit messages end with the session's attribution lines (the harness gives
  them). Never push to another branch, and no pull requests unless asked.

## ISAAC

- **Repository:** `/lustre/isaac24/proj/UTK0533/pmedm_vb`.
- **Results root:** `export R=/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/paper`
  (Python reads it only when exported).
- **Data:** `PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data`.
- **Environment:** `/lustre/isaac24/proj/UTK0533/envs/pmedm_vb`, with
  `unset PYTHONPATH` (the login profile's PYTHONPATH shadows packages). The
  `PMEDM_VB_REPO`, `PMEDM_VB_RUNS` and `PMEDM_VB_ENV` variables point the
  sbatch scripts at UTK0533.
- **Account:** `isaac-utk0533`.
- **Partitions:**

  | partition | hardware | user limits | time limit | use |
  |---|---|---|---|---|
  | `campus`, `short` | CPU (QoS of the same name) | | | CPU jobs |
  | `campus-gpu` | V100s, `--gpus=v100s:1` | 6 submitted / 3 running | 24 h | GPU work |
  | `ai-tenn-debug` | H100, `--gpus=1` | 2 submitted / 1 running | 3 h | GPU work |
  | `ai-tenn` | H100 | | | avoid: a multi-day queue |

- **CPU jobs:**
  `C="--account=isaac-utk0533 --partition=campus --qos=campus --output=$PMEDM_VB_RUNS/slurm-%j.log"`,
  then `sbatch $C experiments/run_python.sbatch <script.py> <args>`.
- **GPU experiments:** `experiments/run_queue.py <experiment JSONs> --root $R --start`.
  - It submits a small self-rescheduling driver job on `short`, which fills
    ai-tenn-debug, then campus-gpu, and resubmits timed-out work. HMC resumes
    from checkpoints.
  - `--status` and `--stop` take the same list. Only one driver per `--root`.
  - Use the `=` form for options whose values start with `--`.

## Where the modelling stands

**The model in use** is the paper baseline: Gaussian error, alpha 0.01,
hierarchy `puma`, taper none, variance floor zero, **multiplier prior sd 3**
(the τ term).

> **τ is not a code default.** It is on only when an experiment sets
> `multiplier_prior_sd`. The first Tennessee runs (tn01, tn05 except
> `tn05_long_prior3`, tn07) and sim02's truths were run without it by mistake.
> Every new experiment must set it explicitly. A guard in run_experiment.py
> that refuses experiments without an explicit `multiplier_prior_sd` was
> offered, but has not been built.

**Tried and dropped,** on the four test PUMAs 4703000, 4702300, 4702900 and
4701900, with prior 3:

| variant | what it does | outcome |
|---|---|---|
| decorrelation (`+decor2`) | drops the replicate correlations of cells with k̂ < 2 | slightly worse on held-out tables in every group and under both scores; worse against sim02's truths |
| Poisson rows (`+poisson2`; `assemble/poisson.py`) | count-model rows for k̂ < 2 cells and zeros, sampled with Stan's bounded transform | much worse: over-allocates rare groups into true-zero block groups, and race × poverty is badly hit |
| Tweedie zero term | Tweedie error on published zeros | HMC did not sample |
| K0 term | count-shaped error on every row | HMC did not sample |

**Why Poisson failed** (code review: no bug found):
- **Zero rows are weak.** Each zero row's multiplier is capped at
  b = N/(n·w). With entropy weighted by n, a zero can only tilt a household
  down by about e^{−0.5 × persons}.
- **Sparse categories lose their PUMA-total anchor.** The PUMA row covers only
  the remaining Gaussian cells.
- **The uniform base weights then win** and spread rare groups everywhere.

The honest weak-zero likelihood needs a prior that knows rare groups cluster.
**That is one motivation for the latent-space model.**

**Statewide support (ε = 0.01) against PUMA-only, both with prior 3:**
- **Real held-out tables:** slightly worse overall, slightly better on race ×
  poverty.
- **sim02:** about equal on local truths and better on borrowed ones (CRPS and
  income-within-race TVD).
- **Coverage:** worse, but partly an artefact. A truth of exactly 0 is covered
  only by draws of exactly 0, which local fits get free where the support
  lacks a category. A fix splitting coverage by zero and positive truths is
  **on hold** at the user's request.
- **Cost:** 2–5× the HMC time.
- Undecided for production.

**Simulation caveats.** sim02's and sim04's truths are draws from PMEDM
posteriors, so they inherit the generating model's zero handling and
clustering. sim02's are also τ-less.

**Running now:** sim04 (`experiments/sim/sim04.json`).
- **Truths:** from tn08 (Gaussian, prior 3, statewide) and tn10 (Poisson,
  prior 3, statewide) on the four PUMAs, 5 each.
- **Fits:** both models fitted to all 40 truths.
- **State:** the build was submitted; fits via run_queue; then
  `sim02_score.py --sim experiments/sim/sim04.json --root $R --fits gauss poisson --base gauss`.

## The new direction: a two-stage latent-space PMEDM

- **Stage 1 (this handoff):** an aspatial embedding of the statewide PUMS,
  giving each unit a fixed φ_i of dimension d.
- **Stage 2 (later):** a PMEDM-type small-area model,
  p_ib ∝ q_ib · exp(−x_i′λ_P − θ_b′h(φ_i)), with h linear to start (see
  the stage 1 section for h and the group-quarters indicators).
  - Per-cell multipliers λ_P at PUMA level only, where cells are well
    measured.
  - A short vector θ_b per block group carrying all small-area variation
    through φ.
  - Similar households then move together, which supplies the information
    sharing zeros lack.
  - Far fewer parameters, and a cheap CAR spatial prior on θ_b.
  - **Probably a proper hierarchical likelihood** (base weights q, priors on θ,
    the tables' Gaussian likelihood) rather than today's dual
    exp(−n·f(λ)). That makes the entropy weight a prior scale. This is a
    design decision, not settled.
- **Order agreed:** build it as an **estimator** first, judged on real held-out
  tables, sampling health and sim04, against today's model. Its posterior
  predictive then becomes the simulation **generator**, calibrated so that
  simulated surveys match the real tables' zero rates, k̂ distribution and
  presence ratios (block groups containing a group against block groups where
  it was sampled).
- **Milestones:**
  1. stage 1;
  2. stage 2 with independent θ, PUMA-only, on 4701502;
  3. CAR prior, four PUMAs, held-out comparison;
  4. statewide support;
  5. as generator (sim05), crossed with sim04's generators.

## Stage 1: design settled with the user (no code yet)

**Purpose.** A fixed position φ_i for every statewide PUMS unit (households
and group-quarters persons, as `assemble/households.py` defines them). Stage 2
reweights the PUMS records by a tilt over φ; it never creates households
outside the PUMS.

### Framing (settled)

- **Records stay the support.** Stage 1 delivers coordinates only. A decoder
  is used to train φ and for diagnostics, never in stage 2.
- **Global household types, area-specific distributions over them.** Types
  may be discrete or continuous. Stage 2 is a tilt of the weighted records:
  p_ib ∝ q_i · exp(−x_i′λ_P − θ_b′h(φ_i)). The choice of h (linear,
  quadratic, type indicators) is compared in the checks.
- **The assumption stage 2 rests on:** x ⊥ b | φ (within a PUMA, units at the
  same position are exchangeable across block groups). Stage 1 makes it cheap
  by making φ predict the units' attributes sharply.
- **Within-unit associations** (couples' matching, sizes) are inherited from
  the records; block groups vary them only through the mix of positions.
- **φ is unstructured.** The sorting literature (socio-economic status, life
  stage, race/ethnicity; Shevky & Bell 1955, factorial ecology) and
  spatially strong outcomes (transport, housing stock) guide **which
  attributes go in**, nothing else. No named blocks, no block-sparse decoder.
- **No pass-through indicators** (S3: no), apart from the unit-type
  indicators below.
- **Group quarters (S4):** one shared space for all units, with group
  quarters as their own unit types, not one-person households.
  φ_i = [GQ-institutional indicator, GQ-noninstitutional indicator (TYPEHUGQ
  2/3), learned part]. Indicators are fixed 0/1. Decoder slopes are shared by
  all units; the indicators give each unit type its own intercepts.
  Household targets apply to households only. Stage 2 gets a coefficient on
  each indicator.
- **No PUMA information in fitting.** PUMA is used only for validation.

### Loss (settled)

L = Σ_h w_h Σ_g Σ_k c_k ℓ_gk(y_hg | W_g φ_h + b_g) / Σ_h w_h + ridge

- **Linear decoder** with natural links:
  - person tables: multinomial over the unit's members in the table's
    universe (e.g. workers for B08301), probabilities softmax(W_g φ + b_g);
    a unit with no members in the universe contributes nothing;
  - household tables and attributes: categorical cross-entropy;
  - continuous attributes: binned, as the tables bin them.
- **Prediction only through φ** (the decoder sees nothing else).
- **Design-weighted (S5: weighted).** w_h = WGTP for households, PWGTP for
  group-quarters persons. An unweighted fit of the baseline is run as a
  diagnostic (Solon, Haider & Wooldridge 2015; Pfeffermann 1993).
- **Rare categories:** c_k = π_k^(−γ), π_k the category's statewide weighted
  share, normalised to mean 1 within each table; γ ∈ {0, 0.5, 1} compared on
  the baseline. Attributes outside tables weigh the same as tables (β = 1).
- **Ridge** λ(Σ_h w_h‖φ_h‖² + ‖W‖²), λ tuned on the baseline (encoder weight
  decay in the deep model).
- **No VAE / KL term.**
- **After fitting:** centre and whiten the learned part of φ (design-weighted
  mean 0, covariance I), so stage 2 can use i.i.d. priors on θ_b. The GQ
  indicators stay 0/1.

### Models (settled)

- **Baseline:** encoder-free. φ_h are free parameters fitted with W
  (close to a design-weighted exponential-family PCA). Merge identical
  records before fitting.
- **Challenger:** a deep set encoder (attention over a household's members,
  plus household-level tokens; Lee et al. 2019 Set Transformer, Zaheer et
  al. 2017 Deep Sets), same loss and decoder. Attribute tokens are randomly
  masked in training (a "masked" token distinct from "not applicable"),
  sometimes a whole person. φ for stage 2 is computed from the unmasked unit.
- Latent classes (one-hot φ) remain possible as another φ under the same loss.

### Targets and inputs (settled)

A JSON configuration lists **targets** (what φ must predict) and **inputs**
(what the encoder reads) separately; most items are both. Entries are tables
(categories and universes from the builders in `constraints.py`) or PUMS
attributes with a categorisation. The baseline has no inputs.

All items below are targets and inputs.

- **Constraint tables:** B01001, B03002, B08301, B19001, B25003, C24010 (block
  group and tract); B26001, B17024, B23001 (tract).
- **Held-out tables (targets too; PUMS attributes, not published values):**
  B25044, C17002, C24030, B12001, B15002, B25040, B17010A–I.
- **Vintage:** 2020–2024 (the repo's study areas use year 2024): the
  2020–2024 PUMS and data dictionary, and 2024 5-year tables.
- **Added tables** (built in step 1, `embed/tables.py`; BG = block group and
  tract, T = tract only):
  - housing stock (occupied-unit versions, crossed with tenure, because the
    all-unit tables B25024/B25034/B25017/B25041 count vacant units the model
    doesn't carry): B25032 units in structure (BG), B25036 year built (BG),
    B25020 rooms (BG), B25042 bedrooms (BG); also B25014 tenure by occupants
    per room (BG), B25075 value (BG), B25063 gross rent (BG);
  - transport: B08303 travel time (BG), B08134 mode by travel time (BG, full
    detail via JWRIP);
  - composition: B11001 household type (BG), B11016 type by size (BG),
    B11005 children under 18 by type (BG), B25007 tenure by age of
    householder (BG), B25009 tenure by size (BG), B19037 age of householder by
    income (BG), B09019 relationship (T);
  - socio-economic: B23025 employment status (BG), B25070 rent burden (BG),
    B25091 owner cost burden (BG), B27010 health insurance (BG), B14001
    school enrolment (T), B18101 disability (T);
  - language, nativity, mobility: C16002 household language (BG), B05002
    nativity and citizenship (T), B07003 mobility by sex (T; replaces B07201,
    whose universe is metro-area residents only; the "moved within the US"
    lines are collapsed because splitting them needs MIGPUMA).
- **Attributes without a usable table:** MULTG (multigenerational; B11017 is
  national and state only), CPLT (couple type), HUPAC (presence and age of
  children), MV (time since moving in; B25038's calendar-year bands can't be
  reproduced from MV, so B25038 was dropped).
- **Omitted by the user:** veteran status, internet.
- **Excluded:** MIGPUMA, POWPUMA (geography), ancestry, detailed race codes.
- **Notes from building:** JWMNP 888 is treated as missing (none in
  Tennessee); both rooms/bedrooms and both B08303/B08134 were kept and can be
  pruned. Judgement calls (civilian noninstitutionalized, mortgage status,
  cost-burden bands, the missing "Medicaid only" 65+ line in B27010) are
  documented in `embed/tables.py`. The user accepted these departures.
- **Configuration:** `experiments/embed/stage1_v1.json` (49 tables, 4
  attributes, all targets and inputs).

### Data

- Stage 1 fitting uses **no published values**; tables give categories and
  universes only. The PUMS PUMA codes use the 2020 definitions (data
  dictionary).
- **Published tables outside the VRE come from the Census API** (new code; the
  repo has never used it). Estimates and MOEs only; SE = MOE/1.645. The VRE
  route stays as it is for Σ in stage 2.
  - The API requires a key: `CENSUS_API_KEY`, stored on ISAAC in
    `~/.config/census/env` (sourced from `~/.bashrc`). Never written to the
    cache, repo or logs.
  - **API calls run only in prefetch, on the login node or dtn1/dtn2,** never
    on a compute node. Responses are cached under explicit names (vintage,
    span, geography, state, table) in the raw-data directory; batch jobs set
    `PMEDM_VB_OFFLINE` as today.
  - First use: PUMA-level tables (summary level 795) for scoring the PUMA
    check.

### Checks

1. **Reconstruction** (deviance explained per category from φ), rare
   categories reported separately; across d, γ, λ.
2. **Neighbour sanity** and 3. **stability** (bootstrap refits, subspace
   comparison) as originally proposed; not yet discussed in detail.
4. **Leave-one-PUMA-out check (settled; validation only):**
   - one stage 1 fit serves all PUMAs (it never sees PUMA codes);
   - for PUMA P: support = the other PUMAs' records; targets = P's constraint
     margins computed from P's own PUMS records, **unperturbed**, with
     replicate-weight SEs (SE² = 4/80 Σ_r (X_r − X)²);
   - fit by maximum entropy three tilts: x only (today's PMEDM, the
     baseline), φ only, and both;
   - score the predicted unit distribution against P's actual weighted
     records on unconstrained joint distributions (race × poverty, tenure ×
     vehicles, income × race, couples' matching), rare-group counts and
     held-out categories (TVD, KL);
   - rare groups also scored against **published PUMA-level tables** (API)
     in SE units, and pooled across PUMAs.
5. **Projection test at block group** (as before): 4701502, then the four
   test PUMAs, θ_b per block group with λ_P at PUMA level, residuals in SE
   units against today's model.
6. **Choice of d:** the smallest that passes checks 1, 4 and 5 for the rare
   categories.

### Still open

- h forms for checks 4–5 (linear only, or linear and quadratic).
- d values (proposed {4, 8, 16, 32}).
- Implementation: PyTorch (already in the ISAAC environment); CPU or GPU;
  the number of statewide units (to confirm on ISAAC).

### Expected files (proposal)

- `src/pmedm_vb/data/api.py`: ACS detailed tables from the API, cached,
  offline-safe; called from `prefetch`.
- Builders for the added tables and attributes (in `constraints.py` or a new
  module), each declaring categories and universe.
- `src/pmedm_vb/embed/`: unit target and input arrays from a configuration;
  the loss; the encoder-free fit; the deep set encoder; normalisation;
  saving φ with a version tag.
- `experiments/embed_pums.py`, `experiments/embed/*.json`,
  `experiments/puma_check.py`, later `experiments/projection_test.py`.

**Not in stage 1:** stage 2 inference; spatial priors; households outside the
support; use as a generator.

## Code to read first

- `src/pmedm_vb/assemble/households.py`: the unit frame (households and
  group-quarters persons, weights).
- `src/pmedm_vb/assemble/constraints.py`, `allcells.py`: table builders, X, the
  table sets.
- `src/pmedm_vb/assemble/build.py`: `build_puma`, the statewide support with
  ε and merged rows; `heldout.py` for held-out tables.
- `src/pmedm_vb/assemble/inputs.py`: `PMEDMInputs` (q, X_T, X_B, A_T, A_B,
  Y, sigma).
- `src/pmedm_vb/solvers/base.py`: `weights_from_lambda`, `ConstraintOperator`.
  `map_dual.py` is a template for the projection test's per-block-group fits.
- `experiments/run_experiment.py`, `run_queue.py`: experiments and the driver.
- `experiments/compare_variants.py`, `heldout_scores.py`, `cell_cases.py`:
  evaluation.
- `experiments/sim02_*.py`, `experiments/sim/sim04.json`: the simulations.
- `pmedm_derivation.md`, `OPEN_QUESTIONS.md` (#4 Tweedie, #5 K0),
  `HANDOFF_experiments.md`: background.

## Other open items (not this branch's job)

- Statewide or PUMA-only support for production.
- The 58-PUMA production run with prior 3 (`tn01_gpu` was τ-less).
- The τ guard in run_experiment.py.
- Coverage split by zero and positive truths in `sim02_score.py` (on hold).
- Under-coverage of unconstrained joint quantities; tempering (κ) is on hold.
- The multi-experiment pilot work list (deferred).
