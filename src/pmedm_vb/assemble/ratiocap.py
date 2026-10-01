"""A soft cap on the weight ratios: a prior against walls.

A unit's weight ratio in a zone is ``w / (N q)`` (``q`` normalised to sum to
1); its log,

    l = log p - log q = -X lambda - log Z + log sum(q),

depends on ``lambda`` only through the weights ``p``. The cap adds, for every
unit-zone with ``q > 0``,

    phi = (tau / 2) * sum (l - b)_+^2,     b = log(ratio),

and multiplies the posterior by ``exp(-phi)``: flat while every ratio is below
``ratio``, a Gaussian tail beyond it, one-sided, with a fixed strength ``tau``
(not scaled by the sample size). Since the dual objective ``f`` enters the
posterior as ``exp(-n f)``, the solvers add ``phi / n`` to ``f``.

Its gradient is exact: ``dl/dlambda = -x + u`` (``u = X'p``), so with
``g = tau (l - b)_+`` and ``G = sum g``,

    grad phi = -X'g + G u.

The MAP Hessian leaves the cap out: it is inactive at the fits' modes when
``ratio`` sits above what the MAP uses (``experiments/ratio_diagnostic.py``),
and Newton with the exact gradient and a line search still converges if it is
not. VB and HMC differentiate it exactly in torch.

It travels with the hierarchy level: ``"puma+cap5000x1"`` is the ``puma``
level with ``ratio = 5000`` and ``tau = 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SEPARATOR = "+cap"


@dataclass(frozen=True)
class RatioCap:
    ratio: float
    strength: float

    @classmethod
    def parse(cls, text: str) -> "RatioCap":
        """``"5000x1"``: the ratio, then ``x`` and the strength."""
        ratio, _, strength = text.partition("x")
        if not strength:
            raise ValueError(f"a ratio cap is RATIOxSTRENGTH, e.g. 5000x1; got {text!r}")
        cap = cls(float(ratio), float(strength))
        if cap.ratio <= 1 or cap.strength <= 0:
            raise ValueError(f"need ratio > 1 and strength > 0, got {text!r}")
        return cap

    def __str__(self) -> str:
        return f"{self.ratio:g}x{self.strength:g}"

    @property
    def bound(self) -> float:
        return math.log(self.ratio)

    def terms(self, logits: np.ndarray, total: float, log_q: np.ndarray, log_q_sum: float
              ) -> tuple[float, np.ndarray]:
        """``(phi, g)`` from the log weights ``logits = log q - X lambda`` and their
        log-sum-exp; ``g = tau (l - b)_+`` per unit-zone (0 off the support)."""
        support = np.isfinite(log_q)
        with np.errstate(invalid="ignore"):
            log_ratio = np.where(support, logits - log_q - total + log_q_sum, -np.inf)
        over = np.maximum(log_ratio - self.bound, 0.0)
        return 0.5 * self.strength * float(np.square(over).sum()), self.strength * over


def split_cap(level: str) -> tuple[str, RatioCap | None]:
    """``"puma+cap5000x1"`` -> ``("puma", RatioCap(5000, 1))``; no cap -> ``(level, None)``."""
    if SEPARATOR not in level:
        return level, None
    base, spec = level.split(SEPARATOR, 1)
    return base, RatioCap.parse(spec)


def cap_level(level: str, ratio: float | None, strength: float | None) -> str:
    """The level string with a cap appended, for a CLI's ``--ratio-cap`` and ``--cap-strength``."""
    if ratio is None:
        if strength is not None:
            raise ValueError("--cap-strength needs --ratio-cap")
        return level
    return f"{level}{SEPARATOR}{RatioCap(ratio, 1.0 if strength is None else strength)}"
