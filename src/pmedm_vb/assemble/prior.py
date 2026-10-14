"""A Gaussian prior on the multipliers: ``zeta ~ N(0, s^2)``, each one independently.

``zeta`` is what multiplies ``Sigma`` in the dual: the multipliers of the
hierarchy's extended constraints, the PUMA rows' included
(:mod:`pmedm_vb.assemble.hierarchy`), or ``lambda`` itself without a
hierarchy. Under ``exp(-n f)`` the prior adds ``zeta' zeta / (2 n s^2)`` to
``f``, which is exactly ``Sigma -> Sigma + tau^2 I`` with ``tau = N / (n s)``
(*Anything above the published variance* in :mod:`pmedm_vb.assemble.sigma`),
so it is carried as :attr:`Sigma.prior_var <pmedm_vb.assemble.sigma.Sigma.prior_var>`
and every solver -- MAP and its Hessian, VB, HMC -- sees it with no code of
its own. Read either way: each multiplier, a per-member log ratio, is held
within a few ``s`` of zero; or every published row gains independent error
of sd ``N / (n s)`` people (6.3 at ``s = 3`` for a PUMA whose average weight
``N / n`` is 18.9).

The null-space coordinates are built from ``Sigma`` without the prior, so a
level samples in the same coordinates with and without one.

It travels with the hierarchy level: ``"puma+prior3"`` is the ``puma`` level
with ``s = 3``; it may sit with a share cap, ``"puma+share0.005x100+prior3"``.
"""

from __future__ import annotations

import dataclasses

SEPARATOR = "+prior"


def split_prior(level: str) -> tuple[str, float | None]:
    """``"puma+prior3"`` -> ``("puma", 3.0)``; no prior -> ``(level, None)``.
    The suffix may be followed by another (``"+share..."``), which is kept."""
    if SEPARATOR not in level:
        return level, None
    head, _, rest = level.partition(SEPARATOR)
    spec, plus, tail = rest.partition("+")
    try:
        sd = float(spec)
    except ValueError:
        raise ValueError(f"a multiplier prior is +prior<sd>, e.g. +prior3; got {level!r}") from None
    if not sd > 0:
        raise ValueError(f"the prior sd must be positive, got {level!r}")
    return head + (plus + tail if plus else ""), sd


def prior_level(level: str, sd: float | None) -> str:
    """The level string with a prior appended, for a CLI's ``--prior-sd``."""
    if sd is None:
        return level
    if not sd > 0:
        raise ValueError(f"--prior-sd must be positive, got {sd}")
    return f"{level}{SEPARATOR}{sd:g}"


def prior_variance(sd: float | None, n: int, N: float) -> float:
    """``tau^2 = (N / (n s))^2``, the diagonal the prior adds to ``Sigma``; 0 without one."""
    return 0.0 if sd is None else (N / (n * sd)) ** 2


def with_prior(sigma, prior_var: float):
    """``sigma`` with ``prior_var`` on its diagonal (unchanged when it is 0)."""
    return sigma if not prior_var else dataclasses.replace(sigma, prior_var=prior_var)
