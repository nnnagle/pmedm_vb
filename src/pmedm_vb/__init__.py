"""Penalized Maximum Entropy Dasymetric Modeling as a variational Bayes problem.

Kept deliberately light so that ``import pmedm_vb`` stays cheap: the only
import is :mod:`pmedm_vb.progress`, which is standard library alone, and torch
is pulled in only when a submodule of :mod:`pmedm_vb.solvers` is imported.
"""

from pmedm_vb.progress import set_verbosity

__version__ = "0.1.0"

__all__ = ["set_verbosity", "__version__"]
