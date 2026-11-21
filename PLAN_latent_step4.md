# Plan, step 4: the posterior and its calibration

> **Status: fixed.** Lower-level plan for step 4 of `PLAN_latent.md`
> (version 2), for the model in `MODEL_latent.qmd` (version 3) with the
> choices fixed in step 3b (`STEP3b_latent.qmd`). Change it only when the
> plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-21 | approved as drafted, with two simulations (G and S); §9 items accepted as proposed |
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Scope and traceability

Step 4 of the parent plan: **C2 and D2 coverage**. It decides whether the
stage 2 posterior (Laplace for the parameters, Gaussian conditioning for the
realized population) is calibrated.

| parent item | where in this plan |
|---|---|
| C2 stage 2: Laplace for the parameters, Gaussian conditioning for $C$ | §4.2, §4.3 |
| D2 coverage, with $Y$ drawn with survey error | §3, §4.4 |
| "decides": calibration; step 5 only if it fails for rare groups | §6 |

Out of scope: samplers (step 5), block groups (step 6).

## 2. The generative model being checked

From `MODEL_latent.qmd` version 3. For each zone, the inferential target is
the **realized population** $C_b$ of $N_b$ units, and any table of it,
$g'C_b$. The published table is a survey estimate of the realized table:

$$
Y = \mathcal A(C) + \epsilon, \qquad \epsilon \sim \mathcal N(0, \Sigma_{\mathrm{survey}}).
$$

In fitting, the model's covariance is $\Sigma = \Sigma_{\mathrm{survey}} +
\omega^2 I$; the $\omega^2 I$ term is the model gap, not survey error.

## 3. Data: the regional run with survey error

In the regional run, each zone's tables have so far equalled its truth
exactly: no survey error. To check calibration, the tables are drawn with
survey error, in two ways.

* **Truth (both simulations):** each zone's own records, design-weighted (as
  in steps 3 and 3b): the realized population, record $i$ standing for
  $w_i$ identical units. Its tables $\mathcal A(C^{\mathrm{true}})$.
* **Leave-zone-out support** in both. The region's records and $\theta_A$
  are not perturbed (the PUMS is the same in every replicate). $N_b$ is the
  truth's unit total (known, as the model assumes).
* **$R = 10$ replicates per simulation,** fixed seeds. The held-out tables
  B1 and B2 are drawn the same way as the fitted ones, since B1 plays
  published tables in choosing $\tau$.

### Simulation G: Gaussian survey error

$Y^{(r)} = \mathcal A(C^{\mathrm{true}}) + \epsilon^{(r)}$,
$\epsilon^{(r)} \sim \mathcal N(0, \Sigma_{\mathrm{survey}})$, with the
region's replicate covariance (untapered, $\alpha = 0.01$). Fitting uses the
same $\Sigma_{\mathrm{survey}}$. The model's survey assumption holds
exactly, so this isolates the posterior approximation.

### Simulation S: a 1-in-8 sample of the true population

* **Sample:** each of the $w_i$ units behind record $i$ is in the sample
  independently with probability $\pi = 1/8$ (Bernoulli sampling), so record
  $i$ contributes $n_i \sim \mathrm{Binomial}(w_i, \pi)$ sampled units.
  Households are sampled whole, with their members, so person cells of one
  household move together.
* **Survey estimate:** the Horvitz–Thompson estimate of every cell,
  $\hat Y = \pi^{-1}\sum_i n_i\, t(x_i)$.
* **Estimated covariance** (the usual unbiased estimator under Bernoulli
  sampling):
  $\hat\Sigma = \frac{1 - \pi}{\pi^2}\sum_i n_i\, t(x_i)\,t(x_i)'$,
  across all cells of a zone; zones are sampled independently.
* **Zero cells:** a cell with $\hat Y = 0$ gets the variance of one sampled
  unit, $(1 - \pi)/\pi^2$, as a floor, in the spirit of the standard PMEDM's
  zero-count floor.
* **Fitting** uses $\hat\Sigma$ in place of $\Sigma_{\mathrm{survey}}$ (plus
  $\omega^2 I$), as a real application uses estimated MOEs.

The survey error is then discrete and skewed for small cells, and $\Sigma$
is itself estimated: closer to the real ACS than simulation G.

## 4. Pieces

### 4.1 MAP per replicate

As in step 3b with $g$ = `kern30`: the $U$ MAP (lowest exact objective over
the `all4` starts) at each $\tau \in \{0.0625, 0.125, 0.25, 0.5\}\times
\tau^{g}_{\mathrm{ref}}$, $\tau$ chosen by B1 misfit. The grid extends one
step below 3b's lower edge, where B1 and truth mostly chose. If the B1
choice sits at $0.0625\times$ in most regions, that is reported, not acted
on.

### 4.2 Laplace for $U$

At the MAP, $U \sim \mathcal N(\hat U, H^{-1})$ with $H$ the Gauss–Newton
matrix (the tables' mean-part information plus $\tau^{-2}I$). With `kern30`
this is 32 coordinates per zone, at most 192 per region. $S$ draws
($S = 200$). $\theta_A$ and $\tau$ are held at their fitted values (an
approximation, recorded).

### 4.3 Gaussian conditioning for the realized population

For each draw of $U$, the zone superpopulations $p_b$, and for any
functional $g'C_b$ the conditional mean and variance given $Y$
(`MODEL_latent.qmd`, stage 2, derived there):

$$
\mathbb E[g'C_b \mid \cdot, Y] = N_b\,g'p_b + k_b'S^{-1}(Y - m),
\quad
\mathrm{Var}[g'C_b \mid \cdot, Y] = N_b\,g'(\mathrm{diag}\,p_b - p_bp_b')g - k_b'S^{-1}k_b.
$$

* One Gaussian draw per $U$ draw gives a posterior sample of each
  functional; intervals are its empirical quantiles.
* **Functionals:**
  - fitted cells (each zone's S1 cells);
  - unfitted cells: transfer sets B1 and B2;
  - rare-group totals (the regional scoring's 15 groups);
  - joint-structure cells: the cells of the two-way scoring tables.
* Intervals whose lower end is below zero are counted and reported (the
  Gaussian's known weak spot at small counts).

### 4.4 Calibration

Against truth, pooled over replicates, zones and cells, by functional class:

* **coverage** of 50% and 90% intervals;
* **standardized errors** $z = (\text{truth} - \text{mean})/\text{sd}$: their
  RMS (1 if calibrated) and a PIT histogram;
* **width:** median interval width against the published SE
  ($\sqrt{\mathrm{diag}\,\Sigma_{\mathrm{survey}}}$) for fitted cells, and
  against truth's size for the others;
* **reference:** the direct survey estimate $Y \pm 1.645\,\mathrm{SE}$ for
  fitted cells, which covers at about 90% by construction;
* the variation of coverage across replicates, since cells within a
  replicate are correlated.

All of the above for each simulation separately.

The MAP's point accuracy under noisy $Y$ is reported beside step 3b's
noise-free results.

## 5. Compute

CPU on ISAAC. Step 3b's `kern30` MAPs took about 4 CPU-hours for one data
set over a 4-value $\tau$ grid; ten replicates of two simulations give
about 80 CPU-hours. The Laplace draws and conditioning are small (192
coordinates per region, closed-form conditioning). Estimated 6–10 hours on
one 16-core node, to be checked on the smoke run.

## 6. Decision rules (fixed before results)

Thresholds are proposals to confirm at review.

* **Primary: simulation S** (the realistic one). Simulation G is the
  diagnostic: if S fails where G passes, the failure is in the survey
  error's form or in estimating $\Sigma$, not in the posterior
  approximation.
* **Calibrated** if, for each functional class except rare groups, pooled
  coverage is within $[0.85, 0.95]$ at 90% and $[0.42, 0.58]$ at 50%.
* **Rare groups** are judged separately with the same bands:
  - pass: stage 2 is the posterior; go to step 6;
  - fail while the other classes pass: step 5 (samplers) as the parent plan
    says;
  - fail in other classes too: stop and review the posterior approximation
    (Laplace, $\tau$ fixed, $\theta_A$ fixed) before any sampler.
* **Reported, not decided:** RMS $z$, PIT, widths, the share of intervals
  below zero, per-replicate variation, MAP accuracy under noise.

## 7. Code and constraints

* **New modules:** `src/pmedm_vb/latent/posterior.py` (§4.2, §4.3),
  `calibration.py` (§4.4).
* **Experiment:** `experiments/latent/step4_posterior.py`, `.sbatch`,
  `collect_step4.py`.
* **Unchanged:** `assemble/`, `solvers/`, existing `data/` modules, `embed/`,
  the step 1–3b code paths and outputs. Imported, not changed.
* **Batch:** runs offline.
* **Verification** (scratchpad, not committed):
  - the conditioning formulas against a dense Gaussian computation on a
    small case;
  - the Laplace covariance against the inverse of a finite-differenced
    Hessian of the fixed-$U$ objective at the MAP (mean part);
  - coverage on a fully simulated case where the model is true (draw $U$
    from the prior, $C$ from the multinomial, $Y$ from the survey model, then
    fit): coverage should be near nominal there, separating errors in the
    code from failures of the model;
  - an end-to-end smoke run.

## 8. Outputs handed to step 5 or 6

* The decision under §6.
* Posterior samples' summaries per zone and functional; coverage tables.
* `summary.txt`.

## 9. Resolved at approval (as proposed)

1. Two simulations, G (Gaussian from the replicate covariance) and S (a
   1-in-8 Bernoulli sample of the true population, Horvitz–Thompson
   estimates, estimated covariance, a one-unit floor for zero cells),
   $R = 10$ replicates each; S primary.
2. The $\tau$ grid with one step below 3b's edge.
3. $\theta_A$ and $\tau$ held fixed in the posterior (stage 2 as specified).
4. The coverage bands in §6 and the split of rare groups.
5. The simulated-truth check in §7.
