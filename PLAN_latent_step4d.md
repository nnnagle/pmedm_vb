# Plan, step 4d: free zone shares (draft)

> **Status: draft, for review.** Step 4d of `PLAN_latent.md` (version 5),
> for the draft model version 5 (`MODEL_latent_v5_draft.md`). The code was
> built with this draft, at the user's request, so that the run can follow
> approval directly. Once approved, change it only when the plan itself
> changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-24 | draft |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Why

Step 4c (`STEP4_latent.qmd`) left model version 4 calibrated on simulation S
in every class but zone totals (0.824 at 90%), and arm P0 identified the
cause: the prior shares $\pi^0$ are built from the same survey totals that
are in $Y$, so the totals are used twice. With prior shares as noisy but
independent of $Y$, the totals reach the band with unchanged widths.

There is no independent source of zone shares for block groups, so the
draft model version 5 removes $\pi^0$: the shares are parameters with a
weak $\mathrm{Dirichlet}(\tfrac12)$ prior, and the tables alone determine
the totals. This step tests it.

## 2. Design

**One arm, FS, on simulation S.** Everything as step 4c's S, except the
realized level:

* the same regions, replicates (10), seeds, realized truths, $Y$ and
  $\hat\Sigma$ (built by step 4c's own `Replicate`), leave-zone-out,
  $g$ = `kern30`, the same functionals including the zone totals;
* **$(\tau, c)$: step 4c's choice for S in each region,** with no new
  selection. The shares change only the prior on the zone totals; B1, the
  held-out set the pair was chosen on, scores composition. Using S's pair
  also keeps the comparison with S and P0 about the shares alone.
* **Model version 5** (`src/pmedm_vb/latent/free_shares.py`):
  - shares $s = \mathrm{softmax}(\psi)$, last coordinate fixed, prior
    $\mathrm{Dirichlet}(\tfrac12)$; the survey's shares are a starting value
    only;
  - collapsed likelihood: $m = N_A\sum_b s_b D_bT'p_b$,
    $S = \Sigma + \sum_b \varphi_b N_A s_b\, D_b V_b D_b'$ with
    $\varphi_b = s_b(N_A + \kappa)/(1 + \kappa s_b)$; no global rank-one
    term;
  - MAP of $(U, \psi)$: two starts at $U = 0$ (from the survey's shares and
    from equal shares), L-BFGS preconditioned by the Gauss–Newton Hessian;
    the lower objective;
  - posterior: Laplace draws of $(U, \psi)$ with the Gauss–Newton
    information plus the priors' Hessians; for each draw, each zone's
    functionals conditioned on $Y$ given the shares (a zone's total is then
    $N_A s_b$, so its uncertainty is all in $\psi$); 200 draws;
  - realized MAP: the conditional PMEDM with the shares free, by
    alternating the existing solver with the shares update, accelerated by
    SQUAREM with a monotone safeguard, to 0.01 units per zone.
* **Outputs** as step 4c's units, plus per unit the MAP's, the survey's
  and the realized MAP's zone totals against the truth, and the totals'
  posterior sd.

**Checks of the code** (done on a toy problem, 3 zones, 5 cells, 80
atoms): the gradient against finite differences (agreement to $10^{-8}$),
the log-likelihood against a dense computation (exact), the conditional
means and variances against dense Gaussian conditioning over pairs
(machine precision), the Jacobian and Gauss–Newton information against
numerical versions, and the accelerated realized MAP against a long plain
alternation (within 0.002 units).

## 3. Decision rules (fixed before results)

FS is judged with step 4c's bands: $[0.85, 0.95]$ at 90% and
$[0.42, 0.58]$ at 50%, B2 $[0.80, 0.95]$ and $[0.40, 0.60]$; classes pooled
over replicates, zones and cells. S and P0 (from step 4c) are reported
beside it.

* **Every class within its band:** model version 5 is adopted; the draft is
  merged into `MODEL_latent.qmd` as version 5, and the next step is step 6
  (the real application).
* **Zone totals within the band but another class not:** stop and review,
  naming the class (the shares' uncertainty would then be reaching the
  composition).
* **Zone totals still outside:** stop and review; first suspects the
  Laplace approximation in $\psi$ and the information about the totals in
  $Y$.

Expected (not part of the rule): zone totals' 90% coverage near the direct
estimates' (about 0.90), intervals wider than S's; other classes as in step
4c.

## 4. Compute

130 units (13 regions × 10 replicates), each a MAP with two starts, a
realized MAP of a few solver calls and 200 draws. Step 4c's P0 arm, the
same number of units with one MAP and one solve, took about 40 CPU-hours.
Estimated 40–80 CPU-hours: one 16-core node, 3–5 hours. Units interleave
regions, so a partial run covers every region.

## 5. To resolve at approval

1. FS at S's chosen pairs, no re-selection of $(\tau, c)$.
2. The shares' prior $\mathrm{Dirichlet}(\tfrac12)$.
3. The decision rules above.
4. Whether to merge the model draft into `MODEL_latent.qmd` before the run
   or only if FS passes (proposed: only if it passes).
