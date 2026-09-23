# Solvers: status and findings (README ToDo #4)

The MAP and VB solvers work as algorithms: they converge, and their objectives
behave as the theory says. The VB fit is **not yet usable as a simulator**:
about 1 VB draw in 6 to 10 puts an implausible share of a PUMA's housing units on a
single household record in a single block group, and the posterior itself
rejects those draws. This file records what was built, what was found, and the
open decision. Numbers are from Knox County, TN (4 PUMAs, ACS 2024 5-year),
tract taper unless stated.

## What exists

- `src/pmedm_vb/solvers/base.py` -- the constraint operator (Kronecker product
  applied through its factors), `p(lambda)`, the dual objective and gradient.
- `src/pmedm_vb/solvers/map_dual.py` -- `solve_map`: damped Newton on the dual
  with the Hessian as per-tract blocks plus a signed low-rank term
  (`DualHessian`), Newton-decrement stopping. Also the saddlepoint log
  evidence and `laplace_precision`.
- `src/pmedm_vb/solvers/vb.py` -- `solve_vb`: VB over `lambda` with target
  `pi(lambda) ~ exp(-n f(lambda))`. Family `N(mu, (G G')^-1)`,
  `G = blockdiag(L_t)(I + W V')`, started at Laplace; ELBO by reparameterised
  Monte Carlo in torch with Adam, per-group learning rates, window-based
  learning-rate halving and Polyak averaging. `family="skewed"` adds a
  per-coordinate sinh-arcsinh transform (Jones & Pewsey 2009) fitted in a
  second stage. `posterior_weights` turns draws into weights.
- `src/pmedm_vb/assemble/inputs.py` -- `variance_floor=None | "zero" | number`;
  `"zero"` floors every cell at its area's zero-count variance.
- `experiments/run_map.py` + `run_map.sbatch` -- steps `prefetch` (login
  node), `assemble`, `solve` (MAP), `vb`; per-task logs; resumable sweeps.
- `experiments/laplace_diagnostic.py` + `.sbatch` -- per fit: (1) per-draw
  excess over the Laplace quadratic for Laplace and VB draws, (2) coordinate
  scan, (3) importance-sampled normaliser, (4) the worst VB draws decomposed by
  (zone, household) cell and by constraint.
- `experiments/weight_draws.py` + `.sbatch` -- draws weights from a VB fit and
  reports how concentrated they get, `log pi - log q` by concentration class,
  and the households responsible.
- `tests/` -- 71 tests on synthetic problems (`tests/synthetic.py`).

## What works

- **MAP** converges on every Knox PUMA for alpha from 1 down to 0.01, both
  tapers, in about 5 s per problem on a compute node.
- **Taper:** the tract taper is the default; the global/tract comparison
  favoured it.
- **VB** (Gaussian, then skewed) converges on every PUMA at alpha 1 and 0.1 in
  2-6 minutes, and its ELBO exceeds the Laplace ELBO by 24,000-91,000 nats
  (run 6271480).

## What does not yet: the chain of findings

1. **Laplace over `lambda` fails badly.** Laplace draws sit a median
   23,000-72,000 nats below the Laplace quadratic's prediction (diagnostic
   part 1, alpha = 1).
2. **Cause: a one-sided posterior along rare cells.** Lowering a multiplier
   multiplies a household's weight by `exp(loading x delta)`, where the loading
   is how many of its members fall in the cell. Where a rare category is
   carried by a few households, some in bulk (six members), the posterior is
   Gaussian-like up to a boundary and then collapses: a soft truncation.
   Diagnostic part 4 splits each bad draw's excess exactly (residual ~1e-11)
   and finds one cell -- one household in one block group -- taking 10-70% of
   `p(lambda)`. At alpha = 1 its logit shift comes mostly from one rare
   attribute at the block group (45-89%, median 68%) plus the same attribute at
   the tract; at alpha = 0.1 it is spread wider (38-62%, median 41%), including
   whole-household bundles.
3. **Variance floor.** `"zero"` removed the extra blow-up at small alpha but
   little at alpha = 1: the curvature comes from how rare the attribute is in
   the sample, not from its SE.
4. **Skewed family.** Cut VB's excess tail (diagnostic part 1, alpha = 1) from
   p99 1,614-3,765 to 52-464, and raised the ELBO over the within-run Gaussian
   by 180-640 nats. A tail remained: max 338-4,459.
5. **Weight draws** (run 6271480, 4,000 draws per fit, all 4 PUMAs, alpha 1
   and 0.1): the largest single cell holds over 1% of N (60-95% of an average
   block group's units) in 10-17% of draws, and over 10% of N in 0.65-1.9%; the
   worst draw puts 72-100% of N on one record. `log pi - log q`, relative to
   its median over all draws: median -47 to -75 for the 1-10% class, -550 to
   -1,400 for the >= 10% class (p90 still -290 to -450), against +2 to +8 for
   the rest. **These draws are an artifact of q; the posterior rejects them.**
   The households responsible are large (mean 3.9-4.8 members against 1.9-2.3
   overall) and carry a rare category in bulk -- race (nh_other_race x10,
   nh_nhpi x8, nh_asian x7, nh_aian x4) or commute mode (bicycle, taxicab,
   other means).
6. **Training saw them.** In the last 500 steps, 23-41% of steps dip more than
   100 nats below the median ELBO and 4-12% more than 1,000; the worst steps
   imply single draws hundreds of thousands of nats low.

Consequences:

- Posterior summaries of W from the VB draws as they stand are not usable.
- Importance reweighting cannot repair them: ESS is 1-2 of 4,000.
- The ELBO means average heavy-tailed terms, so their standard errors (and
  the "3-20 se" skewed-over-Gaussian gains) are less reliable than they look.

Why VB keeps the tail is **not established**. Two candidates, neither tested:
KL(q || pi) weights the tail only by its frequency (about 1.5% x ~1,000 nats is
~15 nats of ELBO), so the family's optimum may carry it; and Adam's running
squared-gradient average, dominated by these spikes, may shrink the steps of
exactly the parameters that would remove it.

## The open decision

Research-direction choices, none started:

- **A family built around the walls** -- transform or constrain along the
  (block group, household) directions where a cell's logit can blow up.
- **A different parameterisation or divergence** that penalises q's mass where
  pi is tiny more than KL(q || pi) does. Needs a literature check first.
- **A reference posterior by MCMC** on `pi(lambda)` for one PUMA, to measure
  VB and Laplace against. Gradients are cheap; mixing in 5,000-8,500
  dimensions with these walls is uncertain.
- **Changing the support** -- add statewide PUMS households at a small design
  weight, so rare categories are not carried by a few large households alone.
  Keeps the dual dimension; multiplies the rows; `n` must stay the sample
  size, not the row count; results would depend on the small weight.

## Other notes

- **Evidence cannot choose alpha.** The saddlepoint log evidence rises like
  `(k/2) log(1/alpha)` as alpha -> 0 (see the `map_dual` docstring).
- **Duplicate rows.** 24-34% of units per PUMA share their X row with another.
  Merging identical rows and summing their design weights leaves `f(lambda)`
  exactly unchanged, provided `n` stays the record count -- a computational
  saving only, worth doing if the row count grows (statewide support).
- **`weight_draws.py` reports per record**, so concentration on a household
  *type* with duplicates is understated.

## Runs referenced

Under `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/`:

- `6271250` -- VB, Gaussian family, `--variance-floor zero`.
- `6271480` -- VB, skewed family, `--variance-floor zero`, alpha 1 and 0.1;
  `diagnostics/` and `weight_draws/` hold the reports quoted above.
