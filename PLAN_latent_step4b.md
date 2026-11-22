# Plan, step 4b: an overdispersed realized population (draft)

> **Status: fixed.** Step 4b of `PLAN_latent.md` (version 3), for
> `MODEL_latent.qmd` version 4. It runs if step 4 (`PLAN_latent_step4.md`,
> version 2) finds the posterior over-confident, as its smoke run suggests.
> Change it only when the plan itself changes, and record each change below.
>
> | version | date | change |
> |---|---|---|
> | 1 | 2026-11-22 | approved; prior shares $\pi^0_b = \hat N_b/N_A$ from estimated zone totals, as in the PMEDM's $q_{ib} = d_i N_b/N$ |
> | 2 | 2026-11-22 | §3: K2's grid moved down to $c \in \{0.0625, 0.125, 0.25, 0.5, 1\}$. Step 4 found errors 6–12 times the posterior sd (coverage 0.14–0.34 at 90%), and K2 chose the grid's lower edge (0.25) on smoke; values above 1 only narrow the intervals |
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

## 2. Model change (version 4)

**Totals.** As in the original PMEDM, the area total $N_A$ (a PUMA, or a
region in the test) is known: it is a hard control in ACS post-stratification
or large enough for its error to be negligible. The zone totals $N_b$ are
unknown.

**Realized population over (zone, atom) pairs.** With prior zone shares
$\pi^0_b = \hat N_b/N_A$ from estimated zone totals, as in the PMEDM's design
prior $q_{ib} = d_i N_b/N$, and the zone superpopulations $p_b$, let
$q_{bi} = \pi^0_b\, p_{bi}$. Then

$$
\tilde q \mid q \sim \mathrm{Dirichlet}(\kappa\, q),
\qquad
C \mid \tilde q \sim \mathrm{Multinomial}(N_A, \tilde q),
$$

that is, $C \sim \mathrm{DirMult}(N_A, \kappa q)$ over all (zone, atom) pairs
of the area. Version 3 is the limit $\kappa \to \infty$ with the zone totals
fixed.

* **Zone totals** $N_b = \sum_i C_{bi}$ are random, with $\sum_b N_b = N_A$.
  By the Dirichlet's aggregation property, $(N_b)_b \sim
  \mathrm{DirMult}(N_A, \kappa\pi^0)$, and given the $N_b$, each zone's
  population is $C_b \sim \mathrm{DirMult}(N_b, \kappa\pi^0_b\, p_b)$: the
  zone's own overdispersion is $\kappa_b = \kappa\pi^0_b$.
* **Moments** (standard Dirichlet-multinomial results):
  $$
  \mathbb E\, C = N_A\, q,
  \qquad
  \mathrm{Cov}\, C = \varphi\, N_A(\mathrm{diag}\,q - qq'),
  \qquad
  \varphi = \frac{N_A + \kappa}{1 + \kappa}.
  $$
  The covariance couples zones through $-qq'$ (their totals sum to
  $N_A$).
* **The Gamma–Poisson view** (standard): independent counts
  $C_{bi} \sim \mathrm{Poisson}(\lambda_{bi})$ with
  $\lambda_{bi} \sim \mathrm{Gamma}(\kappa q_{bi}, \beta)$, conditioned on
  their total $N_A$, are exactly this Dirichlet-multinomial.

**What changes elsewhere** (derived here):

* **Collapsed likelihood:** $m = N_A\sum_b \pi^0_b D_bT'p_b$, and the
  realization term in $S$ becomes $\varphi N_A\,\mathcal D T'(\mathrm{diag}\,q
  - qq')T\mathcal D'$ over pairs. The cross-zone part $-qq'$ is rank one and
  is handled by one more Woodbury correction.
* **Gaussian conditioning (stage 2):** the same formulas with
  $\mathrm{Cov}\,C$ above.
* **Conditional PMEDM (MAP of the realized population):** to second order,
  the PMEDM primal over pairs with base $q$ and entropy weight
  $N_A/\varphi \approx \kappa$; the zone shares float, which is the existing
  solver's form (parent plan, finding 3).

**Relation to the standard PMEDM** (derived here, second-order argument).
The PMEDM's primal is $n\,\mathrm{KL}(r \Vert q)$ over (unit, zone) pairs,
with $q$ the design distribution and the zone shares free. That is the MAP of
this model with $\kappa = n$ and $p_b$ = the design distribution. So:

* the standard PMEDM is version 4 with $\kappa = n$ and no type tilt;
* the model with tilts is the PMEDM with a learned base $q$;
* $\kappa = n$ gives $\varphi \approx N_A/n$, the mean design weight (about
  19): standard deviations about 4.4 times wider than with independent
  units, close to the "about 5 times too narrow" seen on smoke.

## 3. Choosing $\kappa$

Two arms:

* **K1 (primary): $\kappa = n_A$,** the area's number of records: the
  PMEDM-equivalent overdispersion. No fitting.
* **K2: $\kappa = c\, n_A$,** with $c$ chosen per region without truth from
  $c \in \{0.0625, 0.125, 0.25, 0.5, 1\}$, by the posterior predictive log density of
  the held-out tables B1 (the posterior of B1's true tables plus their
  survey error). A log density rewards correct width as well as correct
  centre, which a misfit does not.

In both, $\tau$ is still chosen by B1 misfit, and the $U$ MAP is refitted
under each arm's $S$.

**Prior shares in the test.** $\hat N_b$ is each replicate's survey estimate
of the zone's unit total (households plus group-quarters persons, from the
fitted unit-type cells: the Horvitz–Thompson estimate in S, truth plus error
in G), scaled so that $\sum_b \hat N_b = N_A$. The true zone totals are not
used.

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
  $\kappa = n_A$.
* **If only K2 is calibrated,** K2 is adopted, with $c$ chosen by B1.
* **If neither is,** stop and review.

## 6. Compute

Each arm refits the $U$ MAP for every replicate, so about step 4's cost per
arm. K2's five values of $c$ multiply the MAP and posterior cost by up to
five; to be estimated on smoke, possibly by choosing $c$ on fewer
replicates.

## 7. Resolved at approval

1. The Dirichlet-multinomial realized level over (zone, atom) pairs, with
   $N_A$ known and the zone totals random (model version 4).
2. K1 as primary ($\kappa = n_A$), K2 as the tuned alternative;
   $\pi^0_b = \hat N_b/N_A$ from each replicate's survey estimates.
3. Choosing $c$ by B1 predictive log density.
4. Running 4b only if step 4 fails §6.
