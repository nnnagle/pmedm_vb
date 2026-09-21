"""Solvers for the PMEDM problem (README ToDo #3).

Two are planned, sharing :mod:`~pmedm_vb.solvers.base`:

* :mod:`~pmedm_vb.solvers.map_dual` -- the MAP / Laplace estimate, a port of the
  existing Gauss-Newton solver on the dual objective.
* :mod:`~pmedm_vb.solvers.vb` -- the variational Bayes formulation the paper is
  about, optionally seeded from a MAP solution.

Once these are implemented, importing this package will pull in torch.
:mod:`pmedm_vb` itself never does, so code that only assembles data stays
cheap to import.
"""
