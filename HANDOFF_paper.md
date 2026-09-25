# Handoff: writing the paper's introduction and methods

Project: nnnagle/pmedm_vb -- Penalized Maximum Entropy Dasymetric Modeling
(PMEDM) as Variational Bayes. Research code for one paper: it presents VB for
PMEDM and compares it, as a tool for inference and for simulation, with the
original MAP/Laplace formulation and with raking. This task is **writing**, not
coding: a draft introduction and methods section. The results are still being
computed (`HANDOFF.md` is the running task that produces them).

## Working rules (the author's)

- The author is an experienced R programmer, newer to Python, and the paper's
  author. Clarify before drafting; propose an outline before prose.
- **Do not state anything unverified as fact.** Only claims verifiable in the
  published literature, or in this repository's code and records. If unsure,
  say so, or look it up first. This matters most for citations: the
  references listed below are the ones the code already cites; any other
  reference must be checked before it goes in, and none may be recalled from
  memory and cited as if checked.
- No test scripts, example scripts or documentation unless asked.
- Develop on the branch the session assigns; commit and push there; no PR
  unless asked.

## Ask the author first

- Where the draft lives and in what format (Markdown in the repo, LaTeX,
  Word, a shared document), and the target journal or audience, which sets
  length and how much derivation goes in the main text.
- The paper's framing in their words: the question it answers and the
  contribution it claims. The records below support a framing but do not fix
  one.
- Notation. `pmedm_derivation.md` and the code use `p, q, X, lambda, Sigma,
  n, N, W`; confirm the rest before building on it. **Settled by the author:
  the paper's weights sum to `N`.** They are sample weights in the survey
  sense -- a record's weight in a block group is the number of population
  units it represents there, so weighted sums of records are population
  counts. That is the code's `W = N p` and the `w` of Nagle et al. (2013).
  `pmedm_derivation.md` writes it `w'`, and uses lowercase `w = n p` (the
  expected *sample* count) only to build the multinomial likelihood; in the
  paper's notation that step becomes `n p = (n/N) W`, which is where the
  `n/N` on the entropy term comes from.

## Read first, in order

1. `README.md` -- specs, the study area, *Open research questions*.
2. `pmedm_derivation.md` -- the likelihood, the primal, the dual, gradient
   and Hessian. The methods section's model subsection comes from here.
3. `src/pmedm_vb/solvers/vb.py` module docstring -- the posterior reading of
   the dual, the VB families (Gaussian, skewed, sum-difference), the ELBO,
   parameterisation and stopping. The clearest statement of the method.
4. `src/pmedm_vb/assemble/sigma.py` docstring -- `Sigma(alpha)` from the
   replicate weights, and why `alpha` shrinks correlations but not variances.
5. `src/pmedm_vb/solvers/map_dual.py` docstring -- MAP by damped Newton on
   the dual, the saddlepoint evidence and why it cannot choose `alpha`.
6. `src/pmedm_vb/solvers/mcmc.py` docstring -- the HMC reference and its
   whitening.
7. `src/pmedm_vb/rake.py` docstring -- the two raking baselines.
8. `solvers_synopsis.md` -- findings 1-12: why VB alone is not a simulator
   and why VB + short HMC is the proposed method. Background for the
   introduction's motivation; the numbers there are from earlier runs and are
   being redone (see *Results*, below).
9. `census_data_sources.md` and `src/pmedm_vb/assemble/constraints.py` --
   data sources, variance formulas, and every constraint table's
   PUMS correspondence.
10. `HANDOFF.md` -- the task that produces the results; the evaluation
    design below is the version agreed with the author since it was written.

## Material for the methods section

Each item names where the authoritative statement is. Paraphrase from there;
check against the code rather than this summary where they could differ.

**Study area and data.** Knox County, Tennessee: exactly four whole PUMAs,
4701501-4701504, 121 tracts, 301 block groups (`README.md`). ACS 2020-2024
5-year summary tables from the Variance Replicate Estimate tables (estimate,
MOE and 80 replicates in one file), and the 2020-2024 PUMS
(`census_data_sources.md`). One PMEDM problem per PUMA; the unit is a
household or a group-quarters person, weighted by `WGTP` or `PWGTP`
(`assemble/households.py` docstring). Problem sizes run from 5,304 to 8,577
multipliers (runs 6277110, 6277100); 4701501 has 3,565 units and
N = 67,388 (runs 6276606, 6276611).

**Constraints** (`assemble/constraints.py`, `default_tables`): six tables at
both tract and block group -- B01001 sex by age, B03002 Hispanic origin by
race, B08301 means of transportation, B19001 household income, B25003 tenure,
C24010 sex by occupation -- and three at tract only: B26001 group quarters,
B17024 poverty by age, B23001 employment status. Categories are collapsed to
published boundaries; collapsing is exact because the replicates are summed
with the cells (module docstring). Both levels are constrained deliberately:
a tract estimate is not the sum of its block groups' (`default_tables`
docstring).

**Variances and `Sigma`.** Successive-differences replicate variance, `4/80`
times the sum of squared replicate deviations (Fay & Train 1995 is cited in
the code; check the citation). Published zeros take a modelled zero-cell
variance (`census_data_sources.md` section 3). `Sigma(alpha) = D^(1/2)
[(1 - alpha) R + alpha I] D^(1/2)`: published variances on the diagonal at
every alpha, replicate correlations shrunk toward zero; `alpha = 1` is
classic diagonal PMEDM; replicate covariance is kept only within a tract
("tract taper", the default). All runs floor each cell's variance at its
area's zero-count variance (`variance_floor = "zero"`,
`PMEDMInputs.floored_variances` docstring gives the reason).

**The model.** The penalised MaxEnt primal and its dual,
`f(lambda) = Y'lambda/N + log q'exp(-X lambda) + (n / 2N^2) lambda' Sigma lambda`
(`pmedm_derivation.md`). The posterior reading, `pi(lambda) ~ exp(-n f)`:
up to constants the log posterior of the MaxEnt family's natural parameter
under the prior `lambda ~ N(0, (N/n)^2 Sigma^-1)` (`vb.py` docstring). Its
mode is the MAP solution, so MAP, Laplace, VB and HMC all address one
posterior. Which Laplace approximation is the baseline -- over `p` or over
`lambda` -- is listed as open in `README.md`; the comparison uses the one over
`lambda`, `N(lambda*, (nH)^-1)`, which is the one VB starts from.

**Methods compared** (the grid is every method x PUMA x alpha in
{1, 0.1, 0.01}):

| Method | Draws of W | Where described |
|---|---|---|
| Raking, hard IPF to block group margins | one W | `rake.py` |
| Unbalanced Sinkhorn, KL-penalised margins, both levels | one W | `rake.py` |
| MAP + Laplace over lambda | 4,000 | `map_dual.py`, `StructuredGaussian.laplace` |
| VB, Gaussian / skewed / sum-difference | 4,000 each | `vb.py` |
| VB skewed + short HMC (200 warmup + 1,000) | thinned to 4,000 | `mcmc.py`, `run_mcmc.py` |
| HMC reference (2,000 + 10,000) | thinned to 4,000 | `mcmc.py` |

Settings: VB 64 draws per step; HMC 32 chains, trajectory 3.0, whitened by
the same cell's skewed VB fit, distinct seeds per run; raking has no `Sigma`
and so no alpha. Two raking points for the text: hard IPF has no exact
solution here (published margins at the two levels, and different tables'
totals, disagree), so it is run to a tolerance and reported where it stalls;
Sinkhorn's penalty weights `c_k = Y_k / sigma_k^2` and its zero-target rule
(published SE / 1000) are this project's choices, not the literature's, and
should be written as such (`rake.py` docstring).

**Evaluation** (`experiments/compare_methods.py` docstring is authoritative):

- *Timing*: wall seconds per fit on stated hardware (CPU fits on 18 cores;
  HMC on one V100), split into MAP, fit and drawing; for samplers, seconds
  per 1,000 bulk ESS.
- *Inference*: for constrained cells and **held-out tables** -- published
  tables never in the fit -- each method's predicted counts against the
  published estimate and MOE: z, interval half-width over MOE, coverage, tail
  shares; and against the HMC reference: median difference, sd and interval
  width ratios. Held-out tables: B25044 tenure by vehicles, C17002
  income-to-poverty ratio, C24030 sex by industry, B12001 marital status,
  B15002 educational attainment ("related" to the constraints), and B25040
  heating fuel ("less related"). B25034 (year built) was dropped: its
  universe includes vacant units, which the model does not carry.
- *Simulation*: how often a draw puts implausible weight on one (block group,
  record) cell, by four rules (ratio to the prior `p/q`, share of the block
  group, above the reference's maximum, share of N), and joint diagnostics
  (PSIS k-hat, whitened coordinates, tract/block group pairs). Weights only:
  integer synthetic populations are not part of the comparison.
- Alpha is a factor in the comparison, not something it selects: the
  saddlepoint evidence rises without bound as alpha -> 0 (`map_dual.py`).

## Results -- not yet available

Do not write results, and do not quote numbers from `solvers_synopsis.md` as
the paper's: every result is being recomputed on one clean grid (`runs.csv`
lists the jobs). The earlier findings motivate the method and can frame the
introduction as what prompted the work, but the paper's numbers will come
from the new grid. The first finished cell (4701501, alpha 0.01, runs
6276605-6276611) behaves as findings 9-11 describe; treat that as
provisional.

## References the code already cites (verify each before use)

Nagle et al. (2013) -- the PMEDM formulation, cited in `pmedm_derivation.md`;
Deming & Stephan (1940) -- IPF; Csiszar (1975, *Annals of Probability* 3) --
I-divergence projections; Chizat, Peyre, Schmitzer & Vialard (2018,
*Mathematics of Computation* 87) -- unbalanced scaling; Fay & Train (1995) --
successive-differences replicates; Jones & Pewsey (2009, *Biometrika* 96) --
sinh-arcsinh; Kucukelbir et al. (2017, *JMLR* 18) -- ADVI; Hoffman & Gelman
(2014, *JMLR* 15) -- dual averaging; Hoffman et al. (2019, arXiv:1903.03704)
-- NeuTra; Yao et al. (2018, ICML) -- PSIS diagnostic for VI; Vehtari et al.
(2021, *Bayesian Analysis* 16) -- rank-normalised R-hat and ESS; Schafer &
Strimmer (2005) -- shrinkage covariance, cited only as an untried option for
choosing alpha. Full bibliographic details are not recorded in the repository
and must be looked up.
