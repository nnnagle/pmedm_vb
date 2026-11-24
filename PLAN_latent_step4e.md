# Plan, step 4e: the cut form of free shares (draft)

> **Status: draft, for review.** Step 4e of `PLAN_latent.md` (version 6),
> for the draft model version 5 in its cut form (`MODEL_latent_v5_draft.md`
> §7, draft 2). The code was built with this draft, at the user's request.
> Once approved, change it only when the plan itself changes, and record
> each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-24 | draft |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Why

Step 4d's free shares kept every composition class calibrated but left
zone totals short (0.843 at 90%). Its review (`STEP4_latent.qmd`) found the
totals' intervals the right width and their centre pulled by the model: a
model part of 0.29% RMS, 91% fixed by zone across replicates. A totals
module from the unit-type cells and the known $N_A$ alone, checked on the
existing outputs, was calibrated (0.888 at 90%, 0.485 at 50%). This step
tests the cut form, in which that module sets the totals and the composition
is fitted given them, through the full pipeline.

## 2. Design

**One arm, FScut, on simulation S,** as step 4d's FS in everything but the
totals: S's data exactly as step 4c built them, step 4c's $(\tau, c)$ for S
in each region, 10 replicates, 200 draws, fitted from scratch.

* **Totals module:** each zone's direct unit total (its fitted
  `unit_type:*` cells summed) with its variance from S's $\hat\Sigma$ (the
  cells' block summed plus their zero-cell floors; zones sampled
  independently), conditioned on $\sum_b N_b = N_A$. The same estimator as
  the diagnostic's "conditioned on $N_A$".
* **$U$ given the totals:** the MAP with the shares fixed at the module's
  mean (two starts at $U = 0$, Gauss–Newton-preconditioned and plain
  L-BFGS; the lower), and its Laplace approximation. $\hat U$'s dependence
  on the totals within their uncertainty (about 1%) is ignored.
* **Realized MAP:** the conditional PMEDM with the zone totals fixed at the
  module's mean (one normalizer per zone in the dual; derived here).
* **Posterior:** per draw, totals from the module, $U$ from its Laplace
  approximation, each zone's functionals conditioned on $Y$ given its total.
* **Outputs:** step 4d's unit files, with the module's totals, the raw
  direct totals and the truth.

**Checks of the code** (toy problem, 3 zones, 5 cells, 80 atoms): the
module's draws sum to $N_A$ exactly and have its mean and covariance; the
gradient of the $U$ objective against finite differences ($2\times10^{-8}$);
its information against a numerical Jacobian ($10^{-8}$); both starts reach
the same MAP; the fixed-total realized MAP hits the totals exactly, with
duality gap $5\times10^{-13}$, and no feasible perturbation lowers its
objective; the posterior's totals have the module's mean and sd.

## 3. Decision rules (fixed before results)

FScut judged with step 4c's bands ($[0.85, 0.95]$ and $[0.42, 0.58]$; B2
$[0.80, 0.95]$ and $[0.40, 0.60]$), classes pooled over replicates, zones
and cells; FS, S and P0 reported beside it.

* **Every class within its band:** model version 5 in its cut form is
  adopted: the draft is merged into `MODEL_latent.qmd` as version 5, and the
  next step is the real application (step 6), starting with its open point:
  block-group unit totals when group quarters are published only by tract.
* **Zone totals within the band but another class not:** stop and review,
  naming the class (fixing the totals would then be reaching the
  composition).
* **Zone totals outside:** stop and review; the module alone was calibrated
  on these data, so the first suspect is the code path, not the model.

Expected (not part of the rule): zone totals near the module's 0.888 and
0.485; other classes as in steps 4c and 4d.

## 4. Compute

130 units, each a MAP over $U$ alone (two starts), one realized-MAP solve
and 200 draws: about step 4d's 39 CPU-hours or less, 3–4 hours on one
16-core node.

## 5. To resolve at approval

1. FScut at S's chosen pairs, as FS.
2. The totals module as in the diagnostic (unit-type cells, $N_A$, zones
   independent in S).
3. The decision rules above.
4. Merging the model draft only if FScut passes.
