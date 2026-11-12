"""Shared pieces of the borrowing simulation (``experiments/sim/sim02.json``).

The question: does a statewide support (``model.epsilon``, here 0.01) recover
a known population better or worse than a fit on the PUMA's own PUMS sample,
above all its race x income joint structure and its rare groups? The steps:

1. ``sim02_pools.py`` splits the real statewide inputs' out-of-PUMA records
   at random into pools A and B and writes two derived statewide inputs, the
   PUMA's own records plus one pool each (areas ``tnpoola`` and ``tnpoolb``);
2. ``run_experiment.py experiments/tn/tn04_poola_e0.01.json`` fits the PUMA
   on pool A; its HMC draws are the *borrowed* truths, as the PUMA-only
   experiment's (``tn01_gpu``) are the *local* ones;
3. ``sim02_build.py`` draws the truths, a 1-in-``1/f`` survey of each, a
   simulated PUMS sample of each survey, and writes two synthetic problems per
   truth: a *local* fit on the simulated PUMS alone and a *statewide* fit on
   the simulated PUMS plus pool B (``sim02_fits_local.json`` and
   ``sim02_fits_state.json``, fitted by ``run_experiment.py`` unchanged);
4. ``sim02_score.py`` scores both fits' HMC draws against each truth.

Optionally (the ``bootstrap`` block), ``sim02_bootstrap.py`` writes ``R``
Bayesian-bootstrap replicates of some truths' local-fit inputs, each a PUMA
``<sim puma>r<j>`` of the local fits' area, fitted by ``run_experiment.py``
with ``bootstrap.fits`` (the local fit's model and HMC), and ``sim02_score.py``
scores their pooled draws as a third fit, ``boot``.

Pool B never holds a borrowed truth's out-of-PUMA records (those are pool A's),
so the statewide fit cannot borrow the truth itself, only records like it.

Everything the simulation writes besides the inputs is under
``<root>/<name>/simulation`` (``truths/``, ``calibration.json``, ``scores/``):
the simulation's own folder, since its two fits' experiment folders
(``<root>/<fits name>``) are run_experiment.py's.

**Other simulations on the same code** (``experiments/sim/sim04.json``). The
build and the scores take any simulation JSON of this form, with three
generalisations; sim02.json uses none of them and is built and scored as before:

- ``pumas`` (a list) in place of ``puma``: the truths of each kind are drawn
  for every PUMA from the kind's source experiment run on that PUMA, named
  ``<puma><letter><i>`` (kind by kind, then PUMA by PUMA in ``pumas``' order,
  then ``i``); every per-PUMA quantity (the source, its calibration ``c``, the
  PUMS target, the fits' out-of-PUMA records) is the PUMA's own;
- ``fits`` names any fits JSONs: one with ``model.epsilon`` unset fits the
  *local* problem (the simulated PUMS alone), one with ``model.epsilon`` set
  (the simulation's) the *statewide* one; fits that share an area share the
  inputs, which the build writes once per area;
- without a ``pools`` block, the fits' out-of-PUMA records are those of the
  real statewide inputs (``statewide``) of the PUMA, all of them. Then a truth
  drawn from a statewide source may hold out-of-PUMA records that are also in
  the fits' support (the leakage that sim02's pools A and B prevent); its
  share of ``N`` on such records is recorded (calibration.json and the scores'
  ``borrowed.csv``, ``truth_borrowed``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_experiment as rx  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FITS = ("local", "statewide")
#: The local fit's settings that the bootstrap fits must share: the same model,
#: fit and HMC on other inputs.
BOOT_SHARED = ("area", "alphas", "data_dir", "heldout", "model", "rake", "vb_families",
               "vb_device", "hmc")


class Sim02:
    """The settings of a simulation JSON (``experiments/sim/sim02.json``, or one
    with the generalisations of the module docstring, ``sim04.json``) and what
    follows from them.

    ``pumas`` is the list of real PUMAs the truths are drawn for; ``puma`` is
    the one PUMA of a single-PUMA simulation (``None`` with several). Methods
    that take ``puma`` default to that one PUMA, so a single-PUMA caller need
    not name it.
    """

    def __init__(self, path: Path, root: Path | None = None) -> None:
        self.path = Path(path)
        self.settings = json.loads(self.path.read_text())
        self.root = None if root is None else Path(root)
        s = self.settings
        self.name = s["name"]
        if ("puma" in s) == ("pumas" in s):
            raise SystemExit(f"{self.path}: give 'puma' (one PUMA) or 'pumas' (a list), not both")
        self.pumas = [s["puma"]] if "puma" in s else [str(p) for p in s["pumas"]]
        if not self.pumas or len(set(self.pumas)) != len(self.pumas):
            raise SystemExit(f"{self.path}: pumas must be distinct and at least one")
        self.puma = self.pumas[0] if len(self.pumas) == 1 else None
        self.alpha = float(s["alpha"])
        self.epsilon = float(s["epsilon"])
        self.fraction = float(s["sampling_fraction"])
        self.draws = int(s["draws"])
        self.statewide = rx.load_experiment(REPO / s["statewide"])
        self.truth_specs = {kind: rx.load_experiment(REPO / entry["spec"])
                            for kind, entry in s["truths"].items()}
        self.fits = {fit: rx.load_experiment(REPO / path) for fit, path in s["fits"].items()}
        #: The ``bootstrap`` block (``None`` without one); its fits are ``fits["boot"]``.
        self.bootstrap = s.get("bootstrap")
        if self.bootstrap is not None:
            self.fits["boot"] = rx.load_experiment(REPO / self.bootstrap["fits"])
        self._check()
        os.environ["PMEDM_VB_DATA"] = str(rx.data_dir(next(iter(self.fits.values()))))

    def _check(self) -> None:
        s, problems = self.settings, []
        expected = self.sim_pumas()
        dirs = {str(rx.data_dir(spec)) for spec in
                [self.statewide, *self.truth_specs.values(), *self.fits.values()]}
        if len(dirs) > 1:
            problems.append(f"the experiments name different data_dir values: {sorted(dirs)}")
        if float(self.statewide["model"].get("epsilon") or -1) != self.epsilon:
            problems.append(f"{s['statewide']}: model.epsilon must be {self.epsilon:g}")
        if not s["fits"] or "boot" in s["fits"]:
            problems.append("fits: at least one, and 'boot' is the bootstrap block's")
        for fit in s["fits"]:
            spec = self.fits[fit]
            if spec["pumas"] != expected:
                problems.append(f"{s['fits'][fit]}: pumas must be {expected[0]} ... "
                                f"{expected[-1]} ({len(expected)}), as {self.path} names them")
            if [float(a) for a in spec["alphas"]] != [self.alpha]:
                problems.append(f"{s['fits'][fit]}: alphas must be [{self.alpha:g}]")
            epsilon = spec["model"].get("epsilon")
            if set(s["fits"]) == set(FITS):
                want = self.epsilon if fit == "statewide" else None
                if (epsilon is None) != (want is None) or (want is not None
                                                           and float(epsilon) != want):
                    problems.append(f"{s['fits'][fit]}: model.epsilon must be "
                                    f"{'unset' if want is None else f'{want:g}'}")
            elif epsilon is not None and float(epsilon) != self.epsilon:
                problems.append(f"{s['fits'][fit]}: model.epsilon must be unset (the local "
                                f"problem) or {self.epsilon:g} (the statewide one)")
            if "short" not in (spec["hmc"] or {}).get("kinds", []):
                problems.append(f"{s['fits'][fit]}: the scores read HMC short (hmc.kinds)")
        if len({spec["area"]["name"] for spec in self.fits.values()}) > 1:
            problems.append("the fits JSONs must share area.name")
        for kind, entry in s["truths"].items():
            spec = self.truth_specs[kind]
            if self.alpha not in [float(a) for a in spec["alphas"]]:
                problems.append(f"{entry['spec']}: alpha {self.alpha:g} is not among its alphas")
            absent = [p for p in self.pumas if p not in rx.grid(spec)["pumas"]]
            if absent:
                problems.append(f"{entry['spec']}: PUMAs {absent} are not in its grid")
            if entry.get("pool") == "A":
                if self.pools is None:
                    problems.append(f"truths.{kind}: pool A, but there is no pools block")
                elif self.truth_inputs(kind, self.pumas[0]) != self.pool_inputs("A",
                                                                                self.pumas[0]):
                    problems.append(f"{entry['spec']}: its inputs "
                                    f"({self.truth_inputs(kind, self.pumas[0])}) are not "
                                    f"pool A's ({self.pool_inputs('A', self.pumas[0])}); its "
                                    f"area.name must be {s['pools']['areas']['A']!r}")
        if self.bootstrap is not None:
            problems += self._check_bootstrap()
        if problems:
            raise SystemExit(f"{self.path}:\n  " + "\n  ".join(problems))

    def _check_bootstrap(self) -> list[str]:
        """The ``bootstrap`` block against the truths and its fits JSON."""
        b, problems = self.bootstrap, []
        for key in ("truths", "replicates", "seed", "fits"):
            if key not in b:
                problems.append(f"bootstrap: missing '{key}'")
        if problems:
            return problems
        unknown = [t for t in b["truths"] if t not in self.sim_pumas()]
        if unknown:
            problems.append(f"bootstrap.truths: {unknown} are not among the truths "
                            f"({self.sim_pumas()[0]} ... {self.sim_pumas()[-1]})")
        if len(set(b["truths"])) != len(b["truths"]):
            problems.append("bootstrap.truths: a truth is named twice")
        if int(b["replicates"]) < 1:
            problems.append("bootstrap.replicates: at least 1")
        if self.draws % int(b["replicates"]):
            problems.append(f"bootstrap.replicates ({b['replicates']}) must divide draws "
                            f"({self.draws}), so the pooled fit has as many draws as the others")
        spec, local = self.fits["boot"], self.fits["local"]
        expected = self.boot_pumas()
        if spec["pumas"] != expected:
            problems.append(f"{b['fits']}: pumas must be {expected[0]} ... {expected[-1]} "
                            f"({len(expected)}): bootstrap.truths x replicates, in that order")
        for key in BOOT_SHARED:
            if spec.get(key) != local.get(key):
                problems.append(f"{b['fits']}: {key} must be {self.settings['fits']['local']}'s "
                                "(the bootstrap fits are the local fit on other weights)")
        if spec["name"] in {f["name"] for fit, f in self.fits.items() if fit != "boot"}:
            problems.append(f"{b['fits']}: its name must differ from the other fits'")
        return problems

    # -- names and paths ------------------------------------------------------

    @property
    def pools(self) -> dict | None:
        """The ``pools`` block (sim02's pools A and B), ``None`` without one."""
        return self.settings.get("pools")

    @property
    def pool_label(self) -> str:
        """What the fits' out-of-PUMA records are called in reports: ``pool B``,
        or ``out-of-PUMA`` when they are the statewide inputs' own."""
        return "pool B" if self.pools is not None else "out-of-PUMA"

    def _puma(self, puma: str | None) -> str:
        """``puma``, or the simulation's one PUMA when ``None``."""
        if puma is not None:
            return puma
        if self.puma is None:
            raise ValueError(f"{self.path}: several PUMAs, so name one")
        return self.puma

    def kinds(self) -> list[str]:
        return list(self.settings["truths"])

    def count(self, kind: str) -> int:
        """Truths of ``kind`` per PUMA."""
        return int(self.settings["truths"][kind]["count"])

    def sim_puma(self, kind: str, i: int, puma: str | None = None) -> str:
        """``<puma><letter><i>``, e.g. ``4701502b07``: the ``i``-th truth (from 1) of
        ``kind`` drawn for ``puma``."""
        return f"{self._puma(puma)}{self.settings['truths'][kind]['letter']}{i:02d}"

    def sim_pumas(self, kind: str | None = None) -> list[str]:
        """Every truth, kind by kind, then PUMA by PUMA, then by number: the fits'
        ``pumas``."""
        kinds = self.kinds() if kind is None else [kind]
        return [self.sim_puma(k, i, p) for k in kinds for p in self.pumas
                for i in range(1, self.count(k) + 1)]

    def truths(self, kind: str, puma: str) -> list[tuple[int, str]]:
        """``(i, sim puma)`` of every truth of ``kind`` drawn for ``puma``."""
        return [(i, self.sim_puma(kind, i, puma)) for i in range(1, self.count(kind) + 1)]

    def statewide_inputs(self, puma: str | None = None) -> Path:
        """The real statewide inputs: what the pools are split from, or, without a
        ``pools`` block, the fits' out-of-PUMA records."""
        return rx.inputs_dir(self.statewide) / self._puma(puma)

    def pool_spec(self, pool: str) -> dict:
        """The statewide spec with the pool's area name: what run_experiment.py
        and run_map.py would call its inputs."""
        area = {**self.statewide["area"], "name": self.settings["pools"]["areas"][pool]}
        return {**self.statewide, "area": area}

    def pool_inputs(self, pool: str, puma: str | None = None) -> Path:
        """``<data>/processed/inputs/<pool area>-<year>-<span>yr-state-e<epsilon>/<puma>``."""
        return rx.inputs_dir(self.pool_spec(pool)) / self._puma(puma)

    def fit_pool_inputs(self, puma: str | None = None) -> Path:
        """The inputs whose out-of-PUMA records join the statewide fits: pool B's,
        or the real statewide inputs' without a ``pools`` block."""
        if self.pools is not None:
            return self.pool_inputs("B", puma)
        return self.statewide_inputs(puma)

    def truth_inputs(self, kind: str, puma: str | None = None) -> Path:
        return rx.inputs_dir(self.truth_specs[kind]) / self._puma(puma)

    def truth_trace(self, kind: str, puma: str | None = None) -> Path:
        """The truth source's HMC trace (``truths.<kind>.run``: ``short`` or ``ref``)."""
        spec, puma = self.truth_specs[kind], self._puma(puma)
        run = self.settings["truths"][kind].get("run", "short")
        return (self._root() / spec["name"] / puma / f"hmc_{run}"
                / f"{rx.fit_name(spec, puma, self.alpha)}_trace.npz")

    def is_statewide(self, fit: str) -> bool:
        """Whether the fit's support is the statewide one (``model.epsilon`` set)."""
        return self.fits[fit]["model"].get("epsilon") is not None

    def input_fits(self) -> dict[str, list[str]]:
        """The fits (not ``boot``) by the support they fit, ``local`` or
        ``statewide``: the inputs the build writes."""
        out = {"local": [], "statewide": []}
        for fit in self.settings["fits"]:
            out["statewide" if self.is_statewide(fit) else "local"].append(fit)
        return out

    def boot_truths(self) -> list[str]:
        """The truths with bootstrap replicates (none without a ``bootstrap`` block)."""
        return list(self.bootstrap["truths"]) if self.bootstrap else []

    def replicates(self) -> int:
        return int(self.bootstrap["replicates"]) if self.bootstrap else 0

    def replicate_puma(self, sim_puma: str, j: int) -> str:
        """``<sim puma>r<j>``, e.g. ``4701502l01r3``: the ``j``-th replicate (from 1)."""
        return f"{sim_puma}r{j}"

    def replicate_pumas(self, sim_puma: str) -> list[str]:
        return [self.replicate_puma(sim_puma, j) for j in range(1, self.replicates() + 1)]

    def boot_pumas(self) -> list[str]:
        """Every replicate, truth by truth: the bootstrap fits JSON's ``pumas``."""
        return [p for t in self.boot_truths() for p in self.replicate_pumas(t)]

    def truth_index(self, sim_puma: str) -> tuple[int, int]:
        """``(k, i)``: the truth's kind's position in ``truths`` and its number."""
        for k, kind in enumerate(self.kinds()):
            for puma in self.pumas:
                for i, name in self.truths(kind, puma):
                    if name == sim_puma:
                        return k, i
        raise KeyError(sim_puma)

    def fit_inputs(self, fit: str, sim_puma: str) -> Path:
        return rx.inputs_dir(self.fits[fit]) / sim_puma

    def fit_trace(self, fit: str, sim_puma: str) -> Path:
        """The fit's HMC short trace; for ``boot``, ``sim_puma`` is a replicate's."""
        spec = self.fits[fit]
        return (self._root() / spec["name"] / sim_puma / "hmc_short"
                / f"{rx.fit_name(spec, sim_puma, self.alpha)}_trace.npz")

    def _root(self) -> Path:
        if self.root is None:
            raise SystemExit("--root is needed")
        return self.root

    @property
    def folder(self) -> Path:
        """``<root>/<name>/simulation``: the truths, the calibration and the scores."""
        return self._root() / self.name / "simulation"

    def truth_file(self, sim_puma: str) -> Path:
        return self.folder / "truths" / f"{sim_puma}.npz"


# -- the statewide support's records ---------------------------------------------


def record_design_weights(inputs) -> np.ndarray:
    """Each record's own design weight ``d`` in a statewide support's ``unit_members``.

    ``unit_members.weight`` holds the support weight
    (:func:`pmedm_vb.assemble.design.support_weights`): ``(1 - E) d + s d`` for
    the PUMA's records and ``s d`` for the rest, with ``s = E N / sum(d)``.
    Since the PUMA's ``d`` sum to ``N``, the PUMA's support weights sum to
    ``(1 - E + s) N``, which gives ``s`` and so every ``d``. A
    ``design_weight`` column (written by ``sim02_pools.py``) is read directly.
    """
    members = inputs.unit_members
    if members is None:
        raise ValueError(f"PUMA {inputs.puma}: not a statewide support (no unit_members)")
    if "design_weight" in members:
        return members["design_weight"].to_numpy(dtype=float)
    epsilon = float(inputs.support["epsilon"])
    if epsilon <= 0:
        raise ValueError("epsilon 0: the records from elsewhere carry no weight to invert")
    w = members["weight"].to_numpy(dtype=float)
    own = members["in_puma"].to_numpy(dtype=bool)
    s = w[own].sum() / inputs.N - (1 - epsilon)
    d = np.where(own, w / (1 - epsilon + s), w / s)
    if not np.isclose(epsilon * inputs.N / s, d.sum(), rtol=1e-6):
        raise ValueError(f"PUMA {inputs.puma}: the support weights are not support_weights' "
                         f"(E N / s = {epsilon * inputs.N / s:,.1f}, sum d = {d.sum():,.1f})")
    return d
