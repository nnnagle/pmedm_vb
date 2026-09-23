"""Solvers for the PMEDM problem (README ToDo #4).

Two are planned, sharing :mod:`~pmedm_vb.solvers.base`:

* :mod:`~pmedm_vb.solvers.map_dual` -- the MAP / Laplace estimate: damped
  Newton on the dual objective, replacing the trust-region solver of the
  original R/Rcpp code. numpy and scipy only.
* :mod:`~pmedm_vb.solvers.vb` -- the variational Bayes formulation the paper is
  about: a structured Gaussian over the multipliers, started at the Laplace
  approximation of the MAP fit and improved by stochastic gradient in torch.

Once VB is implemented, importing it will pull in torch; the MAP path never
does. :mod:`pmedm_vb` itself never does either, so code that only assembles
data stays cheap to import.
"""
