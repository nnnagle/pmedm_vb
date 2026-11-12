# Handoff: a latent-space PMEDM — stage 1, an embedding of the statewide PUMS

Project: nnnagle/pmedm_vb — Penalized MaxEnt Dasymetric Modeling with
MAP/VB/HMC solvers. Compute runs on ISAAC (UTK HPC). The user runs jobs there
and pastes output back; you cannot reach ISAAC. Give commands, and push code
for the user to pull.

**Branch:** start a new branch from `tn-statewide-gpu`, which holds everything
below (statewide support, GPU bundles, the queue driver, the τ runs, sim04).
`main` does not.

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
  p_ib ∝ q_ib · exp(−x_i′λ_P − φ_i′θ_b).
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

## Stage 1 plan (proposal for the user's review; no code yet)

**Purpose.** A fixed φ_i for every statewide PUMS unit. It must:
1. **keep the table categories recoverable,** rare groups included;
2. **make "close" mean "likely to co-locate".** This is an **assumption**:
   stage 1 is aspatial and learns only attribute similarity, so the
   projection test checks it;
3. **be one fixed vector per record,** its uncertainty ignored in version 1;
4. **cover every unit the statewide support uses** (households and
   group-quarters persons, as `assemble/households.py` defines them), with
   design weights (WGTP for households, PWGTP for group-quarters persons).

**Features [decision S2]:**
- **Table categories:** each unit's counts in every category.
  - Constraint tables: B01001, B03002, B19001, B25003, B26001.
  - Held-out tables: B25044, B25040, B17010A–I, C17002, B12001, B15002, and
    C24030 if the PUMS dictionary is available.
  - Held-out attributes are PUMS attributes, not published values, so there's
    no leak.
- **Composition:**
  - household size; children, adults and people 65+;
  - householder's age, sex and race/ethnicity;
  - spouse or partner present;
  - multigenerational household;
  - group-quarters type.
- **Encoding:** counts as shares plus log size; one-hot for categorical
  attributes.

**Model family [decision S1]:**
- **A. Latent classes.** φ = the type probabilities. Literature to verify
  before citing: Dunson & Xing 2009 JASA; Hu, Reiter & Wang 2018 Bayesian
  Analysis for households with members; Manrique-Vallier & Reiter 2014 JCGS
  for structural zeros.
- **B. Weighted linear factor model.** Exponential-family PCA or multiple
  correspondence analysis; φ = the scores.
- **C. Autoencoder or VAE.**
- **Recommended:** B as the baseline, A as the comparison, C only if both blur
  the rare groups.
- **[decision S3] Pass-through indicators:** append exact indicators for
  categories that must not blur (householder race/ethnicity, group quarters).
  Then φ = [indicators, latent factors]. Recommended.

**Checks:**
1. **Reconstruction:** the design-weighted deviance explained of each
   category x_ik by φ, as d or K varies. Rare categories reported separately
   (AIAN, NH/PI, other race, Asian, group quarters, oldest ages).
2. **Neighbour sanity:** nearest neighbours in φ share size, life stage and
   race well above chance.
3. **Stability:** refit on bootstrap resamples and compare subspaces or
   matched classes.
4. **Projection test (decisive):**
   - On 4701502, then the four test PUMAs, fit θ_b per block group by maximum
     entropy, with λ_P at PUMA level and no spatial prior.
   - Report block-group and tract residuals in SE units against today's
     per-cell model.
   - Pass: most cells within about 1 SE at a modest d, rare groups about as
     good as today. If only a large d fits, the low-rank idea or the features
     need rethinking.
5. **Choice of d (or K):** the smallest that passes checks 1 and 4 for the rare
   categories.

**Steps:**
1. Unit-feature table for all statewide units, reusing the unit frame and
   constraint builders and adding the composition features; keyed by SERIALNO
   (plus person number for group-quarters units), with design weight.
2. Fit B (and A) over d ∈ {4, 8, 16, 32} and K ∈ {8, 16, 32, 64}, with and
   without pass-through indicators.
3. Run checks 1–3.
4. Run the projection test on the best few.
5. Freeze φ, the model and the settings under the data directory, with a
   version tag that stage 2 records.

**Expected files:**
- `src/pmedm_vb/embed/` (or `assemble/embedding.py`);
- `experiments/embed_pums.py`;
- `experiments/projection_test.py`.

**Compute:** small; CPU. The number of units needs confirming on ISAAC; it is
roughly the statewide support's units.

**Open decisions for the user:**
- **S1:** model family.
- **S2:** features.
- **S3:** pass-through indicators.
- **S4:** group-quarters persons as units, or a separate model.
- **S5:** design-weighted or unweighted fits.

**Not in stage 1:** anything spatial; stage 2 inference; decoding new
households; use as a generator.

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
