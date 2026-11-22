# Plan, step 4b: an overdispersed realized population (draft)

> **Status: draft for review.** A contingency plan prepared while step 4
> (`PLAN_latent_step4.md`, version 2) runs. It applies if step 4 finds the
> posterior over-confident, as its smoke run suggests. Model:
> `MODEL_latent.qmd` version 3, with the change in §2 (version 4 on
> approval). Once approved it is fixed, with a change log, and
> `PLAN_latent.md` gets an entry adding it.
>
> "Derived here" marks derivations made for this project, not taken from
> the literature.

## 1. Why

Step 4's checks show the stage 2 posterior is calibrated when the model is
true. On the smoke run against a realized truth from the zones' own records,
90% intervals covered only 13–28%. The posterior mean was closer to the
truth than the survey estimate, but the intervals were about 5 times too
narrow.

The likely cause: the model treats a zone's realized population as $N_b$
independent draws from $p_b$, a member of the model's family. A real zone's
distribution is not in that family (leave-zone-out ensures it in the test).
For $N_b$ in the tens of thousands, the multinomial variance around $p_b$ is
far smaller than the zone's systematic departure from the family. The
model-gap term $\omega^2 I$ covers that departure as noise on $Y$ only,
never as uncertainty in $C$.

## 2. Model change (version 4, on approval)

The zone's composition scatters around the family before its people are
drawn:

$$
\tilde p_b \mid p_b \sim \mathrm{Dirichlet}(\kappa_b\, p_b),
\qquad
C_b \mid \tilde p_b \sim \mathrm{Multinomial}(N_b, \tilde p_b),
$$

that is, $C_b \sim \mathrm{DirMult}(N_b, \kappa_b p_b)$. Version 3 is the
limit $\kappa_b \to \infty$.

**Moments** (standard Dirichlet-multinomial results):

$$
\mathbb E\, C_b = N_b p_b,
\qquad
\mathrm{Cov}\, C_b = \varphi_b\, N_b(\mathrm{diag}\,p_b - p_bp_b'),
\qquad
\varphi_b = \frac{N_b + \kappa_b}{1 + \kappa_b}.
$$

**What changes elsewhere** (derived here):

* **Collapsed likelihood:** the realization term in $S$ is multiplied by
  $\varphi_b$: $S = \Sigma + \sum_b \varphi_b N_b D_bT'(\mathrm{diag}\,p_b -
  p_bp_b')TD_b'$. The tables become less informative about $U$, so the zone
  deviations shrink more.
* **Gaussian conditioning (stage 2):** the same formulas, with
  $N_b(\mathrm{diag}\,p_b - p_bp_b')$ replaced by
  $\varphi_b N_b(\mathrm{diag}\,p_b - p_bp_b')$ in both $\mathrm{Cov}\,C_b$
  and $k_b$.
* **Conditional PMEDM (MAP of the realized population):** to second order,
  the prior on $r_b = C_b/N_b$ has covariance $\varphi_b(\mathrm{diag}\,p_b
  - p_bp_b')/N_b$, which matches an entropy weight $N_b/\varphi_b \approx
  \kappa_b$ for $\kappa_b \ll N_b$. The conditional PMEDM uses weight
  $N_b/\varphi_b$ in place of $N_b$.

**Relation to the standard PMEDM** (derived here, second-order argument).
The PMEDM's entropy weight $n$ over (unit, zone) pairs splits into
$n\,\pi_b$ per zone, about the zone's expected sample size
$n_b = N_b\, n_A/N_A$. So the PMEDM's choice of $n$ rather than $N_b$
corresponds to $\kappa_b \approx n_b$. That gives $\varphi_b \approx N_b/n_b$,
the mean design weight (about 19), so standard deviations about 4.4 times
wider: close to the "about 5 times too narrow" seen on smoke. With
$\kappa_b = n_b$, the MAP of the realized population is step 3's `latent_n`
variant.

## 3. Choosing $\kappa_b$

Two arms:

* **K1 (primary): $\kappa_b = n_b = N_b\, n_A/N_A$.** No fitting; the
  PMEDM-equivalent overdispersion.
* **K2: $\kappa_b = c\, n_b$, with $c$ chosen per region without truth**
  from $c \in \{0.25, 0.5, 1, 2, 4\}$, by the posterior predictive log
  density of the held-out tables B1. B1's predictive distribution is the
  posterior of its true tables plus their survey error. A log density
  rewards correct width as well as correct centre, which a misfit does not.

$\tau$ is still chosen by B1 misfit, as in step 3b, and the $U$ MAP is
refitted under each arm's $S$.

## 4. Design

As step 4 version 2: regional run, leave-zone-out, $g$ = `kern30`,
simulations G and S, $R = 10$ replicates, the same seeds and realized
truths, the same functionals and calibration measures. Step 4's run is the
$\kappa = \infty$ arm and is not repeated.

Also reported: the MAP's point accuracy under each arm (the zone deviations
change when $S$ does) beside step 4's.

## 5. Decision rules (fixed before results)

* Each arm is judged by step 4's §6 bands on simulation S (G as the
  diagnostic).
* **If K1 is calibrated,** it is adopted: model version 4 with
  $\kappa_b = n_b$.
* **If only K2 is calibrated,** K2 is adopted, with $c$ chosen by B1.
* **If neither is,** stop and review.

## 6. Compute

Each arm refits the $U$ MAP for every replicate, so about step 4's cost per
arm. K2's five values of $c$ multiply the MAP and posterior cost by up to
five; to be estimated on smoke, possibly by choosing $c$ on fewer
replicates.

## 7. Open for review

1. The Dirichlet-multinomial realized level (model version 4).
2. K1 as primary ($\kappa_b = n_b$), K2 as the tuned alternative.
3. Choosing $c$ by B1 predictive log density.
4. Running 4b only if step 4 fails §6.
