"""A soft cap on each cell's share of the population: a prior against walls.

A unit's weight in a zone as a share of the PUMA's ``N`` sampling units is
``w / N = p``; its log,

    l = log p = log q - X lambda - log Z,

is what the cap acts on. It adds, for every unit-zone with ``q > 0``,

    phi = (tau / 2) * sum (l - b)_+^2,     b = log(share),

and multiplies the posterior by ``exp(-phi)``: flat while every cell is below
``share`` of ``N``, a Gaussian tail beyond it, one-sided, with a fixed strength
``tau`` (not scaled by the sample size). Read as a half-normal prior on the log
excess, its standard deviation is ``1 / sqrt(tau)``: at ``tau = 100`` a cell 11%
over the cap is one sd out, at ``tau = 1000`` one 3% over. Since the dual
objective ``f`` enters the posterior as ``exp(-n f)``, the solvers add
``phi / n`` to ``f``.

Its gradient is exact: ``dl/dlambda = -x + u`` (``u = X'p``), so with
``g = tau (l - b)_+`` and ``G = sum g``,

    grad phi = -X'g + G u.

The MAP Hessian leaves the cap out: it is inactive at the fits' modes when
``share`` sits above the MAP's largest cell, and Newton with the exact gradient
and a line search still converges if it is not. VB and HMC differentiate it
exactly in torch.

It travels with the hierarchy level: ``"puma+share0.005x100"`` is the ``puma``
level with ``share = 0.005`` (0.5% of ``N``) and ``tau = 100``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SEPARATOR = "+share"


@dataclass(frozen=True)
class ShareCap:
    share: float
    strength: float

    @classmethod
    def parse(cls, text: str) -> "ShareCap":
        """``"0.005x100"``: the share of ``N``, then ``x`` and the strength."""
        share, _, strength = text.partition("x")
        if not strength:
            raise ValueError(f"a share cap is SHARExSTRENGTH, e.g. 0.005x100; got {text!r}")
        cap = cls(float(share), float(strength))
        if not 0 < cap.share < 1 or cap.strength <= 0:
            raise ValueError(f"need 0 < share < 1 and strength > 0, got {text!r}")
        return cap

    def __str__(self) -> str:
        return f"{self.share:g}x{self.strength:g}"

    @property
    def bound(self) -> float:
        return math.log(self.share)

    def terms(self, logits: np.ndarray, total: float) -> tuple[float, np.ndarray]:
        """``(phi, g)`` from the log weights ``logits = log q - X lambda`` and their
        log-sum-exp; ``g = tau (l - b)_+`` per unit-zone (0 off the support, where
        ``logits`` is ``-inf``)."""
        over = np.maximum(logits - total - self.bound, 0.0)
        return 0.5 * self.strength * float(np.square(over).sum()), self.strength * over


def split_cap(level: str) -> tuple[str, ShareCap | None]:
    """``"puma+share0.005x100"`` -> ``("puma", ShareCap(0.005, 100))``; no cap ->
    ``(level, None)``."""
    if SEPARATOR not in level:
        return level, None
    base, spec = level.split(SEPARATOR, 1)
    return base, ShareCap.parse(spec)


def cap_level(level: str, share: float | None, strength: float | None) -> str:
    """The level string with a cap appended, for a CLI's ``--share-cap`` and ``--cap-strength``."""
    if share is None:
        if strength is not None:
            raise ValueError("--cap-strength needs --share-cap")
        return level
    if strength is None:
        raise ValueError("--share-cap needs --cap-strength")
    return f"{level}{SEPARATOR}{ShareCap(share, strength)}"
