# Handoff: a latent-space PMEDM — stage 2, the small-area estimator

Project, user, ISAAC and branch conventions: see `HANDOFF_embedding.md`
("Working with this user", "ISAAC"). This file records what stage 1 found and
the stage 2 design as settled with the user. No stage 2 code exists yet.

**ISAAC clone for this work:** `/lustre/isaac24/proj/UTK0533/pmedm_vb_embed`
(branch `stage1-embedding`). The `pmedm_vb` environment's editable install
points at the other clone, so every shell and job sets
`PYTHONPATH=$REPO/src` after `conda activate`. API calls (prefetch) run on the
login node or dtn1/dtn2 with `CENSUS_API_KEY` from `~/.config/census/env`.
A separate `envs/pmedm_maps` environment has geopandas for maps.

## What stage 1 established

Embedding: attribute-level targets (`experiments/embed/stage1_v2.json`),
design-weighted, γ = 0.5, c_k capped at 10, ridge 10⁻², 3,000 L-BFGS
iterations (running longer slightly lowers held-out scores). Runs:
`/lustre/isaac24/proj/UTK0533/pmedm_vb_runs/embed/stage1_v2/d{8,16}_g0.5_r0.01`.

- **Check 1 (held-out reconstruction, masked by attribute):** mean deviance
  explained 0.20 at d = 8 and 16; d = 32 overfits at this ridge. Household
  composition is captured best; travel time, year built, value and rent
  barely. Race is encoded strongly when observed (train 0.77) but not
  inferable from other attributes.
- **Check 3 (leave-one-PUMA-out, S1, 58 PUMAs):**
  - linear tilts on φ are too rigid; quadratic or decoder-probability tilts
    are needed;
  - their oracles match the x tilt (PMEDM form) on TVD and nearly on
    log-odds-ratio error, so the representation is adequate;
  - fitted to tables with only a ridge prior they overfit badly; the PMEDM's
    entropy penalty α·n·KL(p‖q) fixes most of it. Best: d = 16, decoder h,
    α ≈ 0.1 — TVD 0.061 (x 0.060), log-odds-ratio error 0.419 (x 0.404, no
    tilt 0.437), better than x on householder race × income;
  - **risk: rare groups are shrunk towards the base** (pooled AIAN 0.82,
    NH/PI 0.72 of truth at α = 0.1); between block groups this would spread
    rare groups too evenly;
  - under the entropy penalty an x+φ tilt over one area collapses to the x
    tilt, so φ adds value only where x's multipliers are shared across zones
    (stage 2's design).
- **PUMA neighbour diagnostic:** z-space recovers the large contrasts the
  tables show (Spearman 0.89–0.93); at d = 8 z distance tracks the PUMAs'
  unconstrained joint structure as well as the tables do (0.91–0.92 vs 0.90)
  and much better than an SVD of the tables at the same d (0.74–0.77).
- Model description for the paper: Claude Doc
  https://claude.ai/artifact/Uq6xJrRKwDXRQFY7qomj4T (not yet updated for
  these results).

## Stage 2 design (settled)

For unit i and zone b within an area (block groups within a PUMA, or PUMAs
within a region):

  η_ib = x_i′λ_A + h(φ_i)′θ_b,   p_ib = q_ib exp(−η_ib) / Σ q exp(−η)

Objective (MAP, the PMEDM primal restricted to this family):

  ½ r′(Σ + τ²I)⁻¹ r + α·n·KL(p‖q) + Σ_b ‖θ_b‖² / 2τ_θ²,   r = N Ã′p − Y

- **h (D1):** the stage 1 decoder's predicted category probabilities for
  age × sex, race/ethnicity, income, tenure, household type, household size
  and householder race (one category dropped per attribute), plus the two
  group-quarters indicators; d = 16. Standardised over the support.
- **Identifiability (D2):** weighted sum-to-zero, Σ_b N_b θ_b = 0; λ_A
  carries the area's level, θ_b only differences between zones.
- **Σ (D3):** the full replicate covariance as in the paper baseline, plus
  the baseline's multiplier prior in its PMEDM form, Σ → Σ + τ²I with
  τ = N/(n·s), s = 3 (`assemble/prior.py`). No separate ridge on λ_A (it
  would count that prior twice). Inverse via the low-rank-plus-diagonal
  structure (Woodbury).
- **Race composition (D4):** a variant appending each household's
  race/ethnicity member shares to h (from the records; no stage 1 refit).
- **Support (D5):** each area's own records first; statewide support later.
- **Uncertainty (D6):** none for now (no Laplace approximation); judge point
  estimates on reconstruction and accuracy against truth.
- **α:** tuned on the regional run over {0.03, 0.1, 0.3}; τ_θ = 3 on
  standardised features; independent θ_b first, CAR at milestone 3.
- **Fitting:** damped Gauss–Newton with exact gradient (as built for the
  check), L-BFGS fallback.

## Validation

1. **Regions of PUMAs, with known truth (first run).** Areas = regions of
   3–6 contiguous PUMAs (D7: metro regions — Memphis, Nashville, Knoxville,
   Chattanooga, Clarksville, Tri-Cities — and rural groupings by adjacency,
   from the PUMA–county crosswalk; list for the user's review before running).
   Zones = PUMAs. Y = each PUMA's S1 tables from its own PUMS records, SEs
   from replicate weights. Compare today's PMEDM (per-cell multipliers per
   zone, existing dual solver on these inputs), the latent model, and the
   latent model with race composition. Score against truth: per-zone joint
   structure (TVD, log-odds-ratio error), per-zone rare-group totals,
   allocation accuracy (share of each record's weight in its true PUMA),
   misfit in SE units. Choose α and whether race composition is needed.
2. **Block groups.** PUMA 4701502, then the four test PUMAs; inputs from
   `build_puma` plus h(φ); compare with the paper-baseline PMEDM on the real
   held-out tables (`heldout.py`), residuals in SE units, rare groups.

## Code layout (planned)

`src/pmedm_vb/latent/` (inputs, model, fit, regions, scoring) and
`experiments/latent/`. Reuse `PMEDMInputs`, `Sigma`, `build_puma`, held-out
scoring and the stage 1 φ by import; `assemble/` and `solvers/` unchanged.

## Milestones

- 2a: estimator + regional run with truth (α, race composition).
- 2b: block groups, 4701502, against today's PMEDM.
- 3: CAR prior on θ_b, four test PUMAs, held-out comparison.
- 4: statewide support.

## Open items carried from stage 1

- d = 32 check run cancelled; ignored for now.
- SVD baseline and PUMA-trained reference embedding not run in the check.
- Noise floor for log-odds-ratio error not estimated.
- Age leak in check 1 documented, not fixed.
- Model description doc needs the check results and the stage 2 design.
