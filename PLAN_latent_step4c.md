# Plan, step 4c: choosing $\tau$ and $\kappa$ together (draft)

> **Status: draft for review.** Review of step 4b (`STEP4_latent.qmd`), for
> `MODEL_latent.qmd` version 4 (no model change). Once approved it is fixed,
> with a change log, and `PLAN_latent.md` gets an entry adding it.
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

Under these bands step 4b's K2 would pass B2 (0.82) and still fail on zone
totals (0.82).

## 4. Compute

Selection: 12 pairs × 3 replicates × 26 (simulation, region) units with 50
draws; final: 7 × 26 units with 200 draws. In step 4b a unit took about 18
minutes, most of it the 200 draws. Estimated 150–180 CPU-hours, about 10–12
hours on one 16-core node, less with step 4b's fits reused.

## 5. Open for review

1. The grid: $\tau$ to $2\times$, $c \in \{0.125, 0.25, 0.5\}$.
2. Joint choice by B1 log density over 3 replicates, with 50 draws in the
   selection phase.
3. The wider band for B2 only; zone totals kept at the standard band.
