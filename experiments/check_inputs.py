"""Check the assembled inputs against facts that hold if the assembly is right.

Beyond the shape checks of ``PMEDMInputs.validate``, for every assembled PUMA
under ``$PMEDM_VB_DATA/processed/inputs/knox-2024-5yr``:

- ``B01001`` is constrained as published (46 categories) at tract and block group;
- every unit holds at least one person in ``B01001``'s categories, and every
  group-quarters unit exactly one;
- published estimates nest: each tract's estimate, and each of its replicates,
  equals the sum over its block groups, for every category at both levels;
- ``B01001`` and ``B03002``, both total population, agree in every block group;

and prints when the inputs, held-out tables and table trees were written, and
the PUMS-weighted persons beside the published total (for information: they
are not expected to agree exactly). Exits 1 if any check fails::

    $CONDA_PREFIX/bin/python experiments/check_inputs.py
"""

import os, sys, time, json
import numpy as np
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.constraints import age_sex
from pmedm_vb.config import processed_dir

root = processed_dir() / "inputs" / "knox-2024-5yr"
want = [f"B01001.{c.name}" for c in age_sex().categories]
stamp = lambda p: time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))) if os.path.exists(p) else "MISSING"
bad = 0
def check(ok, msg):
    global bad
    bad += not ok
    print(f"   {'ok  ' if ok else 'FAIL'} {msg}")

for puma in sorted(p.name for p in root.iterdir() if p.is_dir()):
    path = root / puma
    inp = PMEDMInputs.load(path)
    print(f"\n{puma}: {inp.n_units:,} units, {inp.n_zones} block groups, {inp.A_T.shape[0]} tracts, "
          f"{inp.n_constraints:,} constraints, N = {inp.N:,.0f}")
    print(f"   written: inputs {stamp(path / 'manifest.json')}, held-out {stamp(path / 'heldout' / 'names.json')}, "
          f"trees {stamp(path / 'table_trees.json')}")
    inp.validate(); check(True, "validate()")
    for level, names in (("tract", inp.tract_constraints), ("block group", inp.bg_constraints)):
        got = [n for n in names if n.startswith("B01001.")]
        check(got == want, f"B01001 at {level}: {len(got)} categories (expected 46, as published)")

    # Every unit holds at least one person, each counted once across B01001's categories.
    cols = [k for k, n in enumerate(inp.bg_constraints) if n.startswith("B01001.")]
    size = np.asarray(inp.X_B.tocsc()[:, cols].sum(axis=1)).ravel()
    gq = inp.units["is_group_quarters"].to_numpy(bool)
    check(size.min() >= 1, f"every unit has >= 1 person (sizes {size.min():.0f}-{size.max():.0f})")
    check(np.all(size[gq] == 1), f"every GQ unit is one person ({gq.sum()} GQ units)")
    pums = float(inp.units["weight"].to_numpy() @ size)
    pub = float(inp.Y_B[:, cols].sum())
    print(f"   info persons: PUMS weighted {pums:,.0f}, published B01001 over block groups {pub:,.0f} "
          f"(ratio {pums / pub:.3f}; not expected to be exact)")

    # Published estimates nest exactly: each tract = the sum of its block groups, estimates and replicates.
    n_t, n_b = inp.Y_T.shape[0], inp.Y_B.shape[0]
    L = inp.sigma_l
    worst_y, worst_l, both = 0.0, 0.0, 0
    for kt, name in enumerate(inp.tract_constraints):
        if name not in inp.bg_constraints:
            continue
        kb = inp.bg_constraints.index(name)
        both += 1
        worst_y = max(worst_y, np.abs(inp.A_T @ inp.Y_B[:, kb] - inp.Y_T[:, kt]).max())
        if L is not None:
            lt = L[kt * n_t:(kt + 1) * n_t]
            lb = L[inp.Y_T.size + kb * n_b: inp.Y_T.size + (kb + 1) * n_b]
            worst_l = max(worst_l, np.abs(inp.A_T @ lb - lt).max())
    check(worst_y < 1e-6, f"tract estimates = sum of their block groups, {both} categories at both levels "
                          f"(max diff {worst_y:.3g})")
    if L is not None:
        check(worst_l < 1e-6, f"tract replicates = sum of their block groups' (max diff {worst_l:.3g})")

    # Two tables of total population agree in every block group.
    race = [k for k, n in enumerate(inp.bg_constraints) if n.startswith("B03002.")]
    if race:
        diff = np.abs(inp.Y_B[:, cols].sum(1) - inp.Y_B[:, race].sum(1)).max()
        check(diff < 1e-6, f"B01001 and B03002 totals agree in every block group (max diff {diff:.3g})")

print(f"\n{'all checks passed' if not bad else f'{bad} check(s) FAILED'}")
sys.exit(1 if bad else 0)
