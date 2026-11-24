# Model version 5 (draft): free block shares

> **Status: draft, for review.** A proposed change to `MODEL_latent.qmd`
> (version 4). Nothing here is fixed. On approval it is merged into
> `MODEL_latent.qmd` as version 5, with an entry in its change table, and
> this file is removed. Notation follows `MODEL_latent.qmd`: PUMA $P$,
> blocks $b$ (zones in the regional run), atoms $i$, $N_P$ units, $n_P$
> records.
>
> "Derived here" marks derivations made for this project, not taken from
> the literature. All of them need checking before the model is fixed.
>
> | draft | date | change |
> |---|---|---|
> | 1 | 2026-11-24 | free shares (§1–§6), tested in step 4d |
> | 2 | 2026-11-24 | §7, the cut form, after step 4d: totals from the unit-type cells and $N_P$ alone, composition given the totals; to be tested in step 4e |

## 1. Why

Step 4c (`STEP4_latent.qmd`) found version 4 calibrated on simulation S in
every class but block (zone) totals, and identified the cause with arm P0.
The prior shares $\pi^0_b = \hat N_b/N_P$ are built from the survey's
estimates of the block totals, and the same estimates are in $Y$. The
posterior treats prior and data as independent, so it stays centred on the
survey's error and is too narrow. Prior shares as noisy as the survey's
but independent of $Y$ (P0) bring the totals into the band with unchanged
widths.

No independent source of block shares is in hand, and post-stratification
does not provide one: ACS benchmarks are at county level, and a block
group's total is the benchmark times a survey-estimated share (step 4c,
S-cal and the zero-MOE checks). So the survey's block totals should be used
once, in $Y$, and the prior on the shares should carry no information
about them.

## 2. The change

Version 4 centres the realized population on $q_{bi} = \pi^0_b p_{bi}$
with $\pi^0$ fixed from the survey. Version 5 replaces $\pi^0_b$ by the
block's own realized share. Writing $N_b(C) = \sum_i C_{bi}$:

$$
q_{bi}(C) = \frac{N_b(C)}{N_P}\, p_{bi}(\theta_P, u_b),
\qquad
\Pr(C \mid \theta_P, U) \propto
\mathrm{DirMult}\big(C;\ N_P,\ \kappa\, q(C)\big).
$$

The definition is circular, but $C$ ranges over a finite set, so it is a
proper distribution; only its normalizing constant differs from the
Dirichlet-multinomial's.

### It factorizes exactly into totals and composition

Derived here. By the Dirichlet-multinomial's aggregation property, for any
fixed concentration $\kappa q$ the density splits into the block totals'
$\mathrm{DirMult}(N;\, N_P, \kappa\pi)$ times, for each block,
$\mathrm{DirMult}(C_b;\, N_b, \kappa\pi_b p_b)$. With
$\pi_b = s_b \equiv N_b/N_P$ inserted, the block factors are normalized
given $N$, so

$$
\Pr(C) \;\propto\;
\underbrace{\mathrm{DirMult}\big(N;\ N_P,\ \kappa s\big)}_{\text{block totals}}
\;\prod_b \underbrace{\mathrm{DirMult}\big(C_b;\ N_b,\ \kappa s_b\, p_b\big)}_{\text{composition given the total}} .
$$

* **Composition** is version 4's, with $\pi^0_b$ replaced by the realized
  share: concentration $\kappa_b = \kappa s_b$ ("self-weighted"), so each
  block's overdispersion factor is
  $(N_b + \kappa_b)/(1 + \kappa_b) \approx N_P/\kappa \approx \varphi$, the
  same for every block, as in version 4.
* **Block totals** carry the density of a Dirichlet-multinomial evaluated
  at its own mean. For $N_P \gg \kappa$ that is close to the Dirichlet
  density $\mathrm{Dir}(s \mid \kappa s)$, and with every $\kappa s_b$
  large (Stirling's approximation)
  $$
  \log \mathrm{Dir}(s \mid \kappa s) \approx \text{const} - \tfrac12 \sum_b \log s_b ,
  $$
  the kernel of a $\mathrm{Dirichlet}(\tfrac12, \dots, \tfrac12)$: almost
  flat. The $\kappa$ terms cancel.

### Specification

The circular form motivates the model; the specification uses the
factorized form, with the block totals' prior stated directly:

$$
\begin{aligned}
s &\sim \mathrm{Dirichlet}(a\,\mathbf 1_B),
  \quad N_b = N_P\, s_b
  && \text{block shares: weak prior, } a = \tfrac12\\
C_b \mid N_b, \theta_P, u_b &\sim \mathrm{DirMult}\big(N_b,\ \kappa\, s_b\, p_b(\theta_P, u_b)\big)
  && \text{composition, independent across blocks given the totals}
\end{aligned}
$$

with everything else as in version 4: $\theta_P$, $\tau$, $u_b$, $h$, $g$,
$\gamma$, $\ell_P$, $\kappa = c\,n_P$, and
$Y \mid C \sim \mathcal N(\mathcal A(C), \Sigma)$ with
$\Sigma = \Sigma_{\mathrm{survey}} + \omega^2 I$.

* $\pi^0$ leaves the model. A starting value for computation (survey
  totals, or equal shares) is only a starting value.
* $N_P$ stays known; $\sum_b N_b = N_P$.
* $s$ is continuous and $N_b$ an integer; the difference is ignored, as
  everywhere else at these sizes.
* It is a Dirichlet tree with two levels: a near-flat root (total
  concentration $aB$) and the blocks' composition below it (concentration
  $\kappa s_b$). Version 4 is the same tree with the root's concentration
  tied to $\kappa$ and centred on $\pi^0$.

### Where the block totals are identified

Derived here. Only by $Y$, through the Gaussian table likelihood, and by
$\sum_b N_b = N_P$. Adding one unit to block $b$ with the block's own
composition moves the expected tables by $a_b = D_b T' p_b$, the table
profile of an average unit. The information about $N_b$ is about
$a_b' S^{-1} a_b$ (over contrasts between blocks, given the constraint),
with $S$ as in §3.

* The unit-type cells (households, group-quarters persons) identify it most
  directly; person tables contribute through mean household size.
* Repeated totals in different tables are one survey estimate (identical
  replicates); $\Sigma_{\mathrm{survey}}$'s replicate part counts them once.
  The $\alpha$ diagonal and $\omega^2 I$ let the copies average a little.
* Expected precision: about the survey's own for each block total, so
  intervals for the totals near the direct estimate's. Nothing from outside
  $Y$ narrows them, as $\pi^0$ did.

## 3. What changes in inference

Derived here throughout.

### Parameters: the shares join $(\theta_P, U)$

The shares become parameters of the collapsed likelihood, written as
$s = \mathrm{softmax}(\psi)$ (one coordinate fixed), with the
$\mathrm{Dirichlet}(a)$ prior ($\log\pi(\psi) = a\sum_b \log s_b$ + const,
Jacobian included). Given the shares, blocks are independent, so

$$
m = N_P \sum_b s_b\, D_b T' p_b,
\qquad
S = \Sigma + \varphi N_P \sum_b s_b\, D_b T'\big(\mathrm{diag}\,p_b - p_b p_b'\big) T D_b' .
$$

Against version 4: $\pi^0_b$ becomes the parameter $s_b$, and the global
rank-one term $-vv'$ becomes one rank-one term per block (the totals are
fixed given $s$). The engine's structure (block-diagonal by tract plus low
rank) is unchanged. The MAP and the Laplace approximation run over
$(\theta_P, U, \psi)$; $\psi$ adds $B - 1$ parameters.

### Realized population, MAP (stage 1, step 2)

To second order the MAP of $r = C/N_P$ is

$$
\min_r\; W \sum_b \pi_b\, \mathrm{KL}(r_b \Vert p_b) + \tfrac12 e'\Sigma^{-1}e,
\qquad \pi_b = \textstyle\sum_i r_{bi},
\quad W = N_P/\varphi .
$$

* **Convex in $r$.** $\sum_b \pi_b \mathrm{KL}(r_b\Vert p_b) =
  \sum_b \mathrm{KL}(r_{b\cdot} \Vert \pi_b p_b)$, and KL is jointly convex
  in its arguments, both of which are linear in $r$. The Dirichlet(½)
  prior on the shares adds $\tfrac12\sum_b \log \pi_b$, of order one
  against terms of order $N_P$; it is dropped here.
* **Solved with the existing solver.** Since
  $\mathrm{KL}(r \Vert \rho\otimes p) = \mathrm{KL}(\pi\Vert\rho) +
  \sum_b \pi_b \mathrm{KL}(r_b\Vert p_b)$, the objective is
  $\min_\rho$ of version 4's with base $q = \rho\otimes p$. Alternate:
  solve the conditional PMEDM with base $\rho\otimes p$ (the existing
  dual), then set $\rho$ to the solution's block shares. The objective is
  jointly convex in $(r, \rho)$, so the alternation converges to the global
  minimum.
* This is the "offset from the estimated population" iterated to its fixed
  point. As a point estimate it is right; its uncertainty is carried by
  the shares' Laplace approximation, not by the fixed point.

### Posterior (stage 2)

* $(\theta_P, U, \psi) \sim \mathcal N(\text{MAP}, H^{-1})$: the
  uncertainty in the block totals is here.
* Given a draw, the totals are fixed and each block's composition is
  conditioned on $Y$ with version 4's formulas restricted to the block:
  $\mathbb E\, C_b = N_b p_b$, $\mathrm{Cov}\, C_b = \varphi N_b
  (\mathrm{diag}\,p_b - p_b p_b')$, blocks independent before conditioning.
* Mixing over draws gives every functional's interval, totals included.

### Stage 3

The samplers, still to be revised for version 4, would move individuals
between (block, atom) pairs; under version 5 the prior ratio is the
factorized density above.

## 4. Relation to version 4 and the PMEDM

Derived here.

* **Version 4** has nearly the same composition level (concentration
  $\kappa\pi^0_b$ rather than $\kappa s_b$) and the totals' prior
  $\mathrm{DirMult}(N_P, \kappa\pi^0)$: centred on the survey's shares,
  with the same $\kappa$ that sets the composition's spread.
* **The standard PMEDM** pulls the shares toward $\pi^0$ with the entropy
  weight $n$, through $\mathrm{KL}(\pi \Vert \pi^0)$. Version 5 drops that
  term: the PMEDM with free shares.
* **The intercept form** (a block intercept with prior variance $\tau_0^2$
  on the log shares, offset by $\log\pi^0_b$) becomes essentially the
  same model in the limit $\tau_0 \to \infty$.

## 5. Open points

1. **The root's concentration $a$.** $\tfrac12$ matches the circular form;
   $1$ is uniform on the simplex. With $N_P$ large the choice should hardly
   matter; to confirm.
2. **Small blocks.** Stirling's step needs every $\kappa s_b$ large. It
   motivates the prior only; the specification states the prior directly,
   so nothing rests on it.
3. **Group quarters at block-group level.** `B26001` is tract-only in the
   table set, so at block-group level group-quarters persons are identified
   only through person totals minus household persons. Under free shares
   their split among a tract's block groups is weakly identified, and the
   posterior should be wide there. To check in the real application.
4. **Independent sources.** If counts independent of $Y$ become available
   (fork 8, for example the decennial census), they enter as a prior on
   $s$ with their own precision, in place of the Dirichlet($a$).
5. **Laplace for $\psi$.** Shares are well identified at these sizes, so
   the Gaussian approximation should be adequate; it is checked by the
   regional run's coverage of the totals.

## 6. Test (to be planned)

A regional-run arm "free shares" beside S and P0 at S's chosen $(\tau, c)$,
judged with step 4c's bands. Expected: zone totals' coverage near the
direct estimates' (about 0.90), other classes as in step 4c. To be written
as a step plan once this draft is approved.

## 7. The cut form (draft 2, after step 4d)

**What step 4d found** (`STEP4_latent.qmd`). Free shares removed the double
use and kept every composition class calibrated, but zone totals stayed
short (0.843 at 90%). Their intervals had the right width; their centre
carried a pull from the model that is 91% fixed by zone across replicates.
With the shares free, every table informs a block's total, and a block's
composition departs from the model in a fixed way. A totals module using
only the unit-type cells and $N_P$ was calibrated (0.888, 0.485).

**Definition.** Two modules, with the feedback from the second to the
first cut:

1. **Totals.** $x_b$, the block's direct unit total (the sum of its
   unit-type cells: households and group-quarters persons), and its survey
   covariance $\Sigma_x = A_u \Sigma_{\mathrm{survey}} A_u'$ ($A_u$ sums each
   block's unit-type cells; no model-gap term, since no model enters). With
   $x \sim \mathcal N(N, \Sigma_x)$, a flat prior on the shares and
   $\sum_b N_b = N_P$ known:
   $$
   N \mid x \sim \mathcal N\Big(x + \Sigma_x \mathbf 1\,\frac{N_P - \mathbf 1'x}{\mathbf 1'\Sigma_x\mathbf 1},\;
   \Sigma_x - \frac{\Sigma_x \mathbf 1\mathbf 1'\Sigma_x}{\mathbf 1'\Sigma_x\mathbf 1}\Big)
   $$
   (standard Gaussian conditioning on a linear constraint). With blocks
   sampled independently (simulation S) $\Sigma_x$ is diagonal; in the real
   application the replicate covariance couples blocks and the full
   $\Sigma_x$ is used.
2. **Composition given the totals:** version 5's composition level (§2)
   with $N_b$ fixed, and the table likelihood over all tables, unit-type
   cells included (they still inform, for example, the split between
   households and group quarters within a block).

The **cut posterior** is $p(N \mid x)\, p(U, C \mid N, Y)$: the composition
module does not update the totals. It is not the joint posterior of one
model; it is the standard remedy when one module is misspecified and should
not contaminate another (cut distributions in Bayesian graphical models,
Plummer 2015, as recalled; to verify). Here the misspecified module is the
composition, and the diagnostics show it contaminating the totals.

**Inference** (derived here):

* totals: draws from the Gaussian above;
* $U$: MAP and Laplace with the totals fixed at the module's mean (Û's
  dependence on $N$ within the totals' uncertainty, about 1%, is ignored);
* stage 2: for each draw, totals from the module, $U$ from its Laplace
  approximation, and each block's composition conditioned on $Y$ given its
  total, as in §3;
* realized MAP: the conditional PMEDM with the block totals fixed at the
  module's mean. Its dual has one normalizer per block,
  $f(\lambda) = \lambda'Y + W\sum_b s_b \log\sum_i p_{bi}\,
  e^{-(N_P/W)\,t_i'\lambda_b} + \tfrac12\lambda'\Sigma\lambda$, and a Hessian
  block-diagonal by block apart from $\Sigma$.

**Open for the real application.** In the table set, group quarters are
published at tract level only (`B26001`), so block-group unit totals are not
directly available: households are, group-quarters persons are not. A
block-group totals module needs either a two-level form (tract
group-quarters totals, split among the tract's block groups by the
composition module, a partial cut) or another source for block-group
group quarters.

