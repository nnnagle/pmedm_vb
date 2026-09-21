# PMEDM — The Nitty Gritty Derivations

## PMEDM as a Likelihood Problem

Assume that microdata represent a histogram. Then the problem is to use the histogram to estimate the unknown probability density.

Suppose that each person is selected into the sample with probability $q_{ij}$, that each person is independently sampled, and that $w_{ij} = n p_{ij}$ is the expected sample count of people in each histogram bin. Note that $w_{ij}$ here differs from Nagle et al. (2013), where $w_{ij} = N p_{ij}$ (the expected *population* in each bin); here it is the expected *sample* in each bin.

The multinomial density of the histogram bins is

$$
\frac{n!}{\prod_{ij} w_{ij}!} \prod_{ij} q_{ij}^{w_{ij}}
$$

Multiply this by the Gaussian density of the tract-level and block-group-level errors (assuming independence, which isn't quite right, but that's a fight for another day):

$$
\prod_{j \in J_T} \prod_{k \in K_T} \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp\!\left(-\frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
\prod_{j \in J_B} \prod_{k \in K_B} \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp\!\left(-\frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

We thus maximize the joint likelihood

$$
\frac{n!}{\prod_{ij} w_{ij}!} \prod_{ij} q_{ij}^{w_{ij}}
\prod_{j \in J_T} \prod_{k \in K_T} \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp\!\left(-\frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
\prod_{j \in J_B} \prod_{k \in K_B} \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp\!\left(-\frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

subject to the constraints that

$$
\sum_{j'} A_{j,j'} w_{ij} X_{ik} = Y_{jk} + e_{jk}
$$

for all census tracts ($j \in J_T$) and block groups ($j \in J_B$).

## From Likelihood to Log-Likelihood

Taking the log of the likelihood:

$$
\log(n!) - \sum_{ij} \log w_{ij}! + \sum_{ij} w_{ij} \log(q_{ij}) + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Assume Stirling's approximation, $\log(n!) \sim n\log n - n$, is appropriate. Then the log-likelihood is

$$
(n\log n - n) - \sum_{ij}\left(w_{ij}\log w_{ij} - w_{ij}\right) + \sum_{ij} w_{ij}\log(q_{ij}) + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Rearranging:

$$
n\log n - n - \sum_{ij} w_{ij}\log w_{ij} + \sum_{ij} w_{ij} + \sum_{ij} w_{ij}\log(q_{ij}) + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Substitute $w_{ij} = n p_{ij}$:

$$
n\log n - n - n\sum_{ij} p_{ij}\big(\log n + \log p_{ij}\big) + n\sum_{ij} p_{ij} + n\sum_{ij} p_{ij}\log(q_{ij}) + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Rearrange:

$$
(n\log n)\left(1 - \sum_{ij} p_{ij}\right) - n\left(1 - \sum_{ij} p_{ij}\right) - n\sum_{ij} p_{ij}\log\frac{p_{ij}}{q_{ij}} + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Since $\sum_{ij} p_{ij} = 1$, a whole bunch of terms cancel:

$$
-n\sum_{ij} p_{ij}\log\frac{p_{ij}}{q_{ij}} + \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \left(-\log(2\pi) - \log(\sigma_{jk}) - \frac{e_{jk}^2}{2\sigma_{jk}^2}\right)
$$

Maximizing this is equivalent to maximizing:

$$
-n\sum_{ij} p_{ij}\log\frac{p_{ij}}{q_{ij}} - \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \frac{e_{jk}^2}{2\sigma_{jk}^2}
$$

Or, using design weights $d_{ij} = N q_{ij}$ and population weights $w'_{ij} = N p_{ij}$:

$$
-\frac{n}{N}\sum_{ij} w'_{ij}\log\frac{w'_{ij}}{d_{ij}} - \sum_{\ell \in (B,T)} \sum_{j \in J_\ell} \sum_{k \in K_\ell} \frac{e_{jk}^2}{2\sigma_{jk}^2}
$$

which is the form given in Nagle et al. (2013).

## Some Matrix Notation

In simple matrix notation, the pynophylactic constraints for tracts and block groups are

$$
Y_T = A_T w' X_T + e_T, \qquad Y_B = A_B w' X_B + e_B,
$$

where $e_T$ and $e_B$ are error terms with variance $V_T$ and $V_B$.

(See http://rpubs.com/nnnagle/PMEDM_1 for a visual demonstration of these matrices.)

Rewrite these as

$$
\mathrm{vec}(Y_T) = \mathrm{vec}(A_T w' X_T) = (X_T' \otimes A_T)\,\mathrm{vec}(w')
$$

and similarly

$$
\mathrm{vec}(Y_B) = \mathrm{vec}(A_B w' X_B) = (X_B' \otimes A_B)\,\mathrm{vec}(w')
$$

This means that

$$
\begin{bmatrix} \mathrm{vec}(Y_T) \\ \mathrm{vec}(Y_B) \end{bmatrix}
=
\begin{bmatrix} (X_T' \otimes A_T) \\ (X_B' \otimes A_B) \end{bmatrix}
\mathrm{vec}(w')
$$

or (overloading the tilde notation)

$$
\tilde{Y} = \tilde{X}'\tilde{w}
$$

Similarly, the maximum-likelihood objective may be rewritten as

$$
-n\,\mathrm{vec}(p)' \log\big(\mathrm{vec}(p)\big) - 0.5\,\mathrm{vec}(e)'\,\mathrm{diag}\!\big(\mathrm{vec}(\sigma^{-2})\big)\,\mathrm{vec}(e)
$$

or, redefining everything as a vector/matrix:

$$
-n\,p'\log p - 0.5\, e' \Sigma^{-1} e
$$

where $\Sigma$ is the variance–covariance matrix.

## The Primal Problem

### Formulation

The Max Entropy Problem is

$$
\max:\quad \mathcal{L} \sim -n\,\tilde{p}'\log\tilde{p} - 0.5\,\tilde{e}'\Sigma^{-1}\tilde{e}
$$

subject to

$$
\sum_{ij} p_{ij} = 1 \qquad \text{and} \qquad \tilde{X}'\tilde{p} = \tilde{Y}/n + \tilde{e}/n
$$

### The Lagrangian

(Dropping the tildes from here on — everything is implicitly "tilde'd.") Solving the constrained problem is equivalent to solving the unconstrained problem

$$
\mathcal{L} = n\,p'\log(p/q) - n\lambda'\big(X'p - Y/N - e/N\big) - 0.5\, e'\Sigma^{-1}e - n\mu\big(\mathbf{1}'p - 1\big)
$$

### The Gradient

$$
\frac{d\mathcal{L}}{dp} = -n\log p - n + n\log q - nX\lambda - n\mu
$$

$$
\frac{d\mathcal{L}}{d\lambda} = -nX'p + nY/N + ne/N
$$

$$
\frac{d\mathcal{L}}{de} = n\lambda/N - \Sigma^{-1}e
$$

### The Solution

Solve for $p$:

$$
\log p - \log q = -X\lambda - 1 - \mu
$$

$$
p/q = \exp(-X\lambda)\exp(-1-\mu)
$$

$$
p = q \odot \frac{\exp(-X\lambda)}{q'\exp(-X\lambda)}
$$

Solve for $e$:

$$
e = \Sigma\lambda\, n/N
$$

### Forming the Dual Problem

Substitute $p(\lambda)$ and $e(\lambda)$ into the objective, and minimize rather than maximize:

$$
n^{-1} M(\lambda, p(\lambda)) = -p(\lambda)'\log\!\left(q \odot \frac{\exp(-X\lambda)}{q'\exp(-X\lambda)}\right) - 0.5\,\frac{n}{N^2}\lambda'\Sigma\lambda
$$

Breaking apart the fraction in the logarithm:

$$
p(\lambda)'X\lambda + p(\lambda)'\mathbf{1}\,\log\big(q'\exp(-X\lambda)\big) - 0.5\,\frac{n}{N^2}\lambda'\Sigma\lambda
$$

Substitute $p'X = Y'/N + e'/N$ and $p'\mathbf{1} = 1$:

$$
\left(\frac{Y'}{N} + \frac{e'}{N}\right)\lambda + \log\big(q'\exp(-X\lambda)\big) - 0.5\,\frac{n}{N^2}\lambda'\Sigma\lambda
$$

Substitute $e' = \lambda'\Sigma n/N$:

$$
\left(\frac{Y'}{N} + \frac{n}{N^2}\lambda'\Sigma\right)\lambda + \log\big(q'\exp(-X\lambda)\big) - 0.5\,\frac{n}{N^2}\lambda'\Sigma\lambda
$$

Collecting terms, this is the **dual objective**:

$$
n^{-1} M(\lambda) = \frac{Y'\lambda}{N} + \log\big(q'\exp(-X\lambda)\big) + 0.5\,\frac{n}{N^2}\lambda'\Sigma\lambda
$$

The inputs are $q$, $X$, $Y/N$, and $n\Sigma/N^2$.

### The Dual Gradient

The differential:

$$
n^{-1} dM = \left(\frac{Y'}{N} + \frac{n}{N^2}\lambda'\Sigma\right)d\lambda + \frac{1}{q'\exp(-X\lambda)}\big(q \odot \exp(-X\lambda)\big)'(-X)\,d\lambda
$$

$$
n^{-1}\frac{dM}{d\lambda} = \left(\frac{Y}{N} + \frac{n}{N^2}\Sigma\lambda\right) - X'p
$$

This is one of the nice things about MaxEnt: its gradient is easy to calculate.

### The Dual Hessian

Calculate the differential of the gradient:

$$
n^{-1}\frac{d(dM)}{d\lambda} = \frac{n}{N^2}\Sigma\,d\lambda - X'\,dp
$$

$$
dp = -\frac{q \odot \exp(-X\lambda)}{\big(q'\exp(-X\lambda)\big)^2}\big(q \odot \exp(-X\lambda)\big)'(-X\,d\lambda) + \frac{q \odot \exp(-X\lambda)}{q'\exp(-X\lambda)}(-X\,d\lambda)
$$

which means

$$
n^{-1}\frac{d^2 M}{d\lambda\, d\lambda'} = -(X'p)(p'X) + X'\mathrm{diag}(p)X + \frac{n}{N^2}\Sigma
$$

This is a bit of a pain:

- $\dfrac{n}{N^2}\Sigma$ is diagonal (trivial)
- $X'\mathrm{diag}(p)X$ is sparse (easy)
- $-(X'p)(p'X)$ is rank-one and dense (the annoying part)

In linear-algebra terms, this is a sparse matrix with a rank-1 downdate.

In its full glory, it is:

$$
\begin{bmatrix} (X_T' \otimes A_T) \\ (X_B' \otimes A_B) \end{bmatrix}
\mathrm{diag}\big(\mathrm{vec}(p')\big)
\begin{bmatrix} (X_T' \otimes A_T) \\ (X_B' \otimes A_B) \end{bmatrix}'
+ \frac{n}{N^2}\Sigma
-
\begin{bmatrix} (X_T' \otimes A_T) \\ (X_B' \otimes A_B) \end{bmatrix}
\mathrm{vec}(p')\,\mathrm{vec}(p')'
\begin{bmatrix} (X_T' \otimes A_T) \\ (X_B' \otimes A_B) \end{bmatrix}'
$$
