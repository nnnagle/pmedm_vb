# Plan, step 3b: the basis of the zone deviations (review of step 3)

> **Status: draft for review.** Step 3 failed its decision rule
> (`PLAN_latent_step3.md` §6). This is the review it calls for: a test of
> whether the zone deviations help with a basis of a different scale, before
> deciding whether they stay in the model. Model: `MODEL_latent.qmd`
> version 2, with the change in §2 (version 3 on approval). Once approved it
> is fixed, with a change log; `PLAN_latent.md` gets a version 2 entry
> adding this step.
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Question and what step 3 showed

Step 3 (leave-zone-out, all 58 zones):

* The type-reference support helped: `pmedm_ref` beat `pmedm_own` by 9–17%
  on joint structure and rare groups.
* The zone deviations $u_b$, on the reference's own basis (`kern100`),
  worsened joint structure at every $\tau$, more as $\tau$ grew; at
  $0.5\,\tau_{\mathrm{ref}}$ (the grid's lower edge) they gave the best
  transfer to unfitted tables.
* The Laplace evidence chose the largest $\tau$ in every unit; truth chose
  the smallest in every unit.

**Question:** is `kern100` the wrong scale for zone differences, too coarse
(go finer) or too flexible (go coarser), and can $\tau$ be chosen without
truth?

## 2. Model change (version 3, on approval)

The zone deviations get their own basis $g$, separate from the reference's
$h$:

$$
p_{bi} \propto q_i\,\gamma^{d(P,i)}
  \exp\{\theta_P' h(z_i) + u_b' g(z_i)\},
\qquad u_b \sim \mathcal N(0, \tau^2 I_{K_g}),
$$

with $h$ = `kern100` as fixed in step 1. Version 2 is the case $g = h$.

## 3. Design

* **Deviation bases $g$:** `kern30`, `kern100` (step 3's, for continuity)
  and `kern200`. Kernels on the whitened $z$ at design-weighted k-means
  centres (as in step 1), with the two group-quarters indicators; each
  basis's own width $\ell$ (median distance to the nearest centre).
* **A scale for $\tau$ per basis.** $\tau_{\mathrm{ref}}$ was the spread of
  $\hat\theta_P$ across PUMAs on `kern100`; it does not carry over to
  another basis. For each $g$, fit PUMA tilts on $g$ from the PUMAs' own
  records (step 1's pseudo-likelihood, $\sigma_\theta = 1$, $\gamma = 0.5$)
  and set $\tau^g_{\mathrm{ref}}$ from their spread the same way.
* **$\tau$ grid,** extended below step 3's edge:
  $\{0.125, 0.25, 0.5, 1\}\times\tau^g_{\mathrm{ref}}$.
* **Mode:** leave-zone-out only (step 3's primary).
* **Reused from step 3, not rerun:** region references, `pmedm_own`,
  `pmedm_ref`, `latent_u0` (none depends on $g$).
* **MAP:** as step 3 v3's selected set (`all4`, plain L-BFGS), the lowest
  exact objective over the starts.

## 4. Choosing $\tau$ without truth

The evidence failed in step 3, so it is reported but not used to decide.
Instead, **held-out published tables** choose $\tau$:

* Split transfer set B (`latent.regions.transfer_sets`) into two halves by
  table, B1 and B2, fixed before the run.
* For each region and basis, choose the $\tau$ with the lowest misfit of
  the realized MAP on B1 (in SE units).
* Score transfer on B2 only, so the choice and the score do not share
  tables.

The truth-best $\tau$ (lowest mean log-odds-ratio error) is reported beside
it, as a check on the rule.

## 5. Scoring

Step 3's scoring (`latent.scoring.score_region`), leave-zone-out, all zones:
TVD, log-odds-ratio error, rare-group spread, misfit on fitted cells,
transfer misfit on B2. The baselines are re-scored on B2 so transfer is
comparable.

## 6. Decision rules (fixed before results)

For each basis at its B1-chosen $\tau$, against `latent_u0` (the same model
without deviations) and `pmedm_ref` (the best step 3 variant):

* **The deviations stay,** with the best such basis, if `latent` is no
  more than 2% worse than `latent_u0` on log-odds-ratio error and on TVD,
  and at least 5% better on transfer (B2).
* **Otherwise they are dropped:** the model keeps the type-reference
  support and the realized-population level, without $u_b$, and step 4
  proceeds on that.
* **Reported, not decided:** each basis at each $\tau$; the comparison with
  `pmedm_ref`; the evidence-chosen and truth-best $\tau$.

## 7. Compute

CPU on ISAAC. `kern30` is cheaper than step 3 per evaluation, `kern200`
about twice as costly; smaller $\tau$ means fewer local optima and fewer
evaluations (step 3's diagnostic). Leave-zone-out only halves step 3's
cost. To be estimated on the smoke run; within one 24-hour job, or two.

## 8. Open for review

1. The bases: `kern30`, `kern100`, `kern200`.
2. $\tau^g_{\mathrm{ref}}$ from each basis's PUMA tilts, and the grid
   $\{0.125, 0.25, 0.5, 1\}\times\tau^g_{\mathrm{ref}}$.
3. Choosing $\tau$ by held-out tables (B1), scoring transfer on B2.
4. The decision rule in §6 (2% / 5%).
5. Leave-zone-out only.
