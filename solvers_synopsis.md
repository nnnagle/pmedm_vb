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
- `src/pmedm_vb/assemble/build.py` -- `build_puma(..., epsilon=E)`: optional
  statewide support. The PUMA's records keep `(1 - E) d_puma`, every state
  record adds `E d_state N / sum(d_state)`, identical rows of X (households
  and GQ kept apart) are merged, and `n`, `N` stay the PUMA's own. Off by
  default; `run_map.py --epsilon`. See finding 8 for why it did not help.
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
7. **More draws per step halve the tail but do not remove it.** The skewed
   fits of run 6271480 repeated with 64 draws per step instead of 8, all
   else equal (weight draws as in 5):

   | PUMA | alpha | > 1% of N, 8 -> 64 draws | > 10% of N, 8 -> 64 draws |
   |---|---|---|---|
   | 4701501 | 1 | 13.2% -> 7.5% | 1.50% -> 0.38% |
   | 4701501 | 0.1 | 17.1% -> 12.9% | 1.52% -> 0.68% |
   | 4701502 | 1 | 10.1% -> 7.7% | 0.65% -> 0.47% |
   | 4701502 | 0.1 | 14.7% -> 12.6% | 0.90% -> 0.60% |
   | 4701503 | 1 | 9.9% -> 6.2% | 1.18% -> 0.50% |
   | 4701503 | 0.1 | 15.8% -> 10.1% | 1.88% -> 0.65% |
   | 4701504 | 1 | 9.9% -> 5.8% | 1.52% -> 0.33% |
   | 4701504 | 0.1 | 14.1% -> 10.8% | 1.75% -> 0.62% |

   The remaining tail is still rejected by the posterior (`log pi - log q`
   medians -44 to -71 for the 1-10% class, -400 to -920 for >= 10%; ESS 1-3
   of 4,000), and the households are the same kind. So gradient noise is part
   of the cause, not all of it. The comparison is not of draws alone: with
   smaller window standard errors the stopping rule also ran longer. ELBO
   gains over Laplace were unchanged (24,560-91,511 nats); the skewed stage
   added 106-215 nats over the Gaussian.
8. **A statewide support makes the tail worse.** With `epsilon = 0.01` the
   155,047 Tennessee records collapse to 69,291 rows, of which a PUMA's own
   2,329-3,565 records form 1,757-2,361; 0.70-0.75% of the prior lands on
   the ~67,000 new rows (run 6272529, MAP; 6272652, VB with 8 draws per step).
   MAP converges in 7-9 Newton steps, 50-98 s; a VB fit takes 11-19 min.
   - Most rare categories gain carriers at or above the PUMA's highest
     loading (4701501: nh_aian 1 -> 106 rows, nh_asian 1 -> 173,
     other_means 1 -> 103), but the prior share on the largest carrier barely
     moves, since the new rows hold ~1% of the prior. Two walls stay
     unshared: taxicab x3 (2021HU1143286, 4701501) and nh_other_race x10
     (2023HU0854456, 4701504). `B08301.motorcycle` and `B03002.nh_nhpi` had no
     carrier in 4701501 at all before; now 82 and 89.
   - Against the 8-draw PUMA-only fits of run 6271480, draws with a row over
     1% of N rose on every fit (4701501: 13.2 -> 40.1% at alpha 1, 17.1 ->
     53.9% at 0.1; the others 10-16% -> 12-23%), and over 10% of N from
     0.65-1.9% to 2.4-5.7%. `weight_draws.py` now counts merged rows, but the
     MAP's largest row is 38-59 people, about 0.1% of N, so this is not
     the merging.
   - The posterior rejects these draws more strongly than before (4701503 at
     alpha 1 and 4701504 at both: `log pi - log q` median -980 to -1,380 and
     p10 -7,200 to -11,900 for the >= 10% class), and the bulk widens (< 1% class: p10 -43 to
     -75, p90 +50 to +94, against p10 -18 to -33, p90 +19 to +38 in the
     64-draw PUMA-only fits of finding 7).
   - **The tail moves onto the new rows.** In 4701504 at alpha 1, 13 of the 15
     largest-weight rows seen hold no PUMA record, each a single record with
     a MAP weight of 0.001-0.02 people, and reach 77-99.5% of N in their
     worst draw: e.g. six Hispanic men over 65 below poverty (x6 on four
     cells at once), bicycle x3, nh_aian x7. Widening the support adds
     walls: every record whose loading is concentrated on a few cells is a
     direction along which a modest shift in those multipliers multiplies its
     weight by `exp(loading x shift)`.

Consequences:

- Posterior summaries of W from the VB draws as they stand are not usable.
- Importance reweighting cannot repair them: ESS is 1-2 of 4,000.
- The ELBO means average heavy-tailed terms, so their standard errors (and
  the "3-20 se" skewed-over-Gaussian gains) are less reliable than they look.

Why VB keeps the tail is **partly established**. Gradient noise contributes
(finding 7: 64 draws per step halve the > 10% class), but the tail survives
it. The other candidate, untested: KL(q || pi) weights the tail only by its
frequency (about 1.5% x ~1,000 nats is ~15 nats of ELBO), so the family's
optimum may carry it. Findings 5 and 8 give the walls' shape: each is a
record's row `x_i` of X, and a draw `delta` goes wrong when the largest
`x_i' delta` over records makes that record dominate `p(lambda)`. A wall
involves several coordinates at once, which a per-coordinate skew cannot
follow.

## The open decision

**Decided: an MCMC reference posterior for one PUMA, PUMA-only support,** to
see how the posterior itself treats these directions before designing a
family around them. The statewide support (last option below) was tried and
made the tail worse (finding 8).

The options as first recorded:

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
  exactly unchanged, provided `n` stays the record count. Implemented for the
  statewide support (`design.collapse_units`); not applied to the PUMA-only
  problem.
- **`weight_draws.py` reports per record**, so concentration on a household
  *type* with duplicates is understated.

## Runs referenced

Under `/lustre/isaac24/proj/UTK0496/pmedm_vb_runs/`:

- `6271250` -- VB, Gaussian family, `--variance-floor zero`.
- `6271480` -- VB, skewed family, `--variance-floor zero`, alpha 1 and 0.1;
  `diagnostics/` and `weight_draws/` hold the reports quoted above.
- The 64-draw repeat of `6271480` (finding 7); its job id was not recorded.
- `6272529` -- MAP, `--epsilon 0.01 --variance-floor zero`, alpha 1 and 0.1.
- `6272652` -- VB on the same inputs, skewed, 8 draws per step, with
  `weight_draws/` (finding 8). Inputs under
  `$PMEDM_VB_DATA/processed/inputs/knox-2024-5yr-state-e0.01`, with
  `support_summary.csv`.
