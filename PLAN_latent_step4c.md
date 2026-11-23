# Plan, step 4c: choosing $\tau$ and $\kappa$ together (draft)

> **Status: fixed.** Review of step 4b (`STEP4_latent.qmd`), for
> `MODEL_latent.qmd` version 4 (no model change). Added to `PLAN_latent.md`
> in its version 4. Change it only when the plan itself changes, and record
> each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-23 | approved as drafted; S-cal optional, at S's chosen $(\tau, c)$; the $\pi^0$ test (§5 item 5) is not part of this step |
> | 2 | 2026-11-23 | the $\pi^0$ test added as an optional arm (P0) |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Why

Step 4b's K2 arm calibrated four of six classes on simulation S. Two things
remain:

* **$\tau$ was pinned at the top of its grid** ($0.5\,\tau_{\mathrm{ref}}$
  in 110 of 110 cases on S), because $\tau$ was chosen by B1 misfit before
  $c$ was chosen by B1 log density.
* **B2 and zone totals** fell short (0.82 at 90%).

The zone deviations ($\tau$) and the Dirichlet overdispersion ($\kappa$)
both add variance to a zone's tables, but in different places (derived
here):

* $u_b$ adds a low-rank, smooth term along the 32 directions $g$ can reach,
  and moves the mean;
* the Dirichlet inflates the multinomial in every direction and leaves the
  mean alone.

So excess spread in directions $g$ cannot reach identifies $\kappa$, and
excess within $g$'s span beyond $\kappa$'s share identifies $\tau$. A
criterion that rewards correct spread can tell the two apart, which choosing
them one after the other cannot.

## 2. Design

As step 4b's K2 arm: regional run, leave-zone-out, $g$ = `kern30`, model
version 4, simulations S and G, $R = 10$, the same seeds and realized
truths, the same functionals and calibration measures.

**Joint choice of $(\tau, c)$**, per simulation and region, on the grid

* $\tau \in \{0.25, 0.5, 1, 2\}\times\tau^g_{\mathrm{ref}}$ (extended
  upward from step 4b's edge);
* $c \in \{0.125, 0.25, 0.5\}$ (centred on step 4b's choices),

by the B1 posterior predictive log density summed over the first 3
replicates. The other 7 replicates run only the chosen pair.

* **Cost control:** the selection phase uses 50 posterior draws (the
  log density needs a mean and variance per cell, not tail quantiles); the
  final fits use 200. Step 4b's fits are reused where the $(\tau, c)$ pair
  and replicate match.
* **Reported beside the choice:**
  - the Laplace evidence for each pair (does adding $\kappa$ repair its
    preference for the largest $\tau$?);
  - the truth-best pair (lowest mean log-odds-ratio error), as a check;
  - a flag if the chosen $\tau$ is at the new top edge ($2\times$).

### Optional simulation S-cal: a post-stratified survey

An additional simulation, off by default and run only when requested. It
does not replace S or G; S stays primary.

* **Sample:** the same 1-in-8 Bernoulli sample of the realized population
  as S (same seeds).
* **Calibration:** the sample weights are calibrated to region-level
  controls taken from the realized population, the analogue of the ACS's
  county-level weighting areas:
  - total households;
  - group-quarters persons;
  - persons by sex × broad age group.

  By raking (iterative proportional fitting) to those margins, with a
  generalized-regression fallback if raking fails to converge (reported).
* **Tables:** every fitted and held-out cell from the calibrated weights.
* **Covariance:** the linearization estimator for a calibration estimator:
  the Bernoulli-sampling formula of S applied to the residuals of each
  unit's contributions after regression on the calibration variables (to
  verify against the literature before building, as recalled from Deville
  and Särndal and Särndal et al.). Zone totals' errors become negatively
  correlated, and the controlled totals are nearly exact, as in ACS
  estimates.
* **Fitting** uses that covariance (plus $\omega^2 I$), as with S.
* **No selection of its own:** S-cal uses, for each region, the
  $(\tau, c)$ pair that S chose, and runs all 10 replicates at that pair
  with 200 draws, fitted from scratch. With the tuning held fixed,
  differences in coverage between S and S-cal come from the calibration.

S-cal tests whether the totals' shortfall, and the rest of the calibration,
carry over to a survey whose weights are post-stratified like the ACS's.

### Optional arm P0: prior shares from an independent source

An additional arm, run on request, on simulation S only. It tests whether
using the survey's zone totals as both the prior shares $\pi^0$ and data
narrows the posterior of zone totals.

* **Prior shares:** $\pi^0_b = \hat N^{\mathrm{ind}}_b / \sum_{b'} \hat
  N^{\mathrm{ind}}_{b'}$, where $\hat N^{\mathrm{ind}}_b$ is the zone's
  unit total from a second, independent 1-in-8 Bernoulli sample of the same
  realized population (its own seeds). It has the same precision as the
  survey's own totals but is independent of $Y$, like a count from another
  source.
* **Everything else as S:** the same $Y$ and covariance, and, as with S-cal,
  the $(\tau, c)$ pair S chose for each region; all 10 replicates with 200
  draws, fitted from scratch.
* **Reading it:** if zone totals reach the band under P0 while they miss
  under S, using the survey's totals twice is the cause, and real
  applications should take $\pi^0$ from another source (for example
  decennial counts; model fork 8).

## 3. Decision rules (fixed before results)

Simulation S primary, G diagnostic; classes pooled over replicates, zones
and cells.

* **Fitted cells, B1, joint cells, rare groups, zone totals:** step 4's
  bands, $[0.85, 0.95]$ at 90% and $[0.42, 0.58]$ at 50%.
* **B2 (tables never used in fitting or tuning):** a wider band,
  $[0.80, 0.95]$ at 90% and $[0.40, 0.60]$ at 50%. Predicting tables the
  model never saw, including attributes its type space barely encodes,
  should not be held to the same standard as tables it was fitted or tuned
  on.
* **Calibrated:** every class within its band. Then model version 4 with
  $(\tau, c)$ chosen by B1 is the posterior; go to step 6.
* **Otherwise:** stop and review, reporting which classes fail.
* **S-cal and P0, when run,** are reported with the same bands but do not
  enter the decision.

Under these bands step 4b's K2 would pass B2 (0.82) and still fail on zone
totals (0.82).

## 4. Compute

Selection: 12 pairs × 3 replicates × 26 (simulation, region) units with 50
draws; final: 7 × 26 units with 200 draws. In step 4b a unit took about 18
minutes, most of it the 200 draws. Estimated 150–180 CPU-hours, about 10–12
hours on one 16-core node, less with step 4b's fits reused. S-cal, when
run, adds about 40 CPU-hours (10 replicates × 13 regions at S's chosen
pair, no selection phase). P0, when run, adds about the same.

## 5. Resolved at approval

1. The grid: $\tau$ to $2\times$, $c \in \{0.125, 0.25, 0.5\}$.
2. Joint choice by B1 log density over 3 replicates, with 50 draws in the
   selection phase.
3. The wider band for B2 only; zone totals kept at the standard band.
4. S-cal as an optional simulation, with controls on households,
   group-quarters persons and persons by sex × broad age at region level,
   at S's chosen $(\tau, c)$; reported, not decided on.
5. Added in version 2 as the optional arm P0: an arm with prior shares $\pi^0$ from an independent source
   (truth plus independent error), to test whether using the survey's zone
   totals as both prior and data narrows the totals' posterior.
