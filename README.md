# pmedm_vb
Penalized Max Ent Dasymetric Modeling problem as a variational bayes

The derivation of the original max ent problem is in pmedm_derivation.md.
The Census endpoints the downloaders use -- layouts, file schemas, coverage and
the variance formulas -- are documented in census_data_sources.md.
The original Maximum a posteriori problem was solved using Gauss Newton optimization on the dual function with custom rcpp code to evaluate the Hessian. This is a port to PyTorch for solving the Variational Bayes Problem (as well as the original MAP/Laplacian formulation for comparison).

This is not production code. This is research code to write a paper. The paper will present the VB method, and compare simulation accuracy usability with the original (MAP/Laplacian/Penalized MaxEnt) method.

## Specs:
- include code to download the PUMS
- include code to download summary files at block group/tract/PUMS level.
- include code to download summary file covariance tables (optional).
- Assemble input data
- Solver(s) (user choose either VB or MAP/Laplacian) in PyTorch.
    VB might use previous Laplacian solution as seed, or some other default not requiring prior evaluation 
- Simulator

## ToDo:
- [x] Determine package folder structure
- [x] write downloaders (`src/pmedm_vb/data/`)
- [x] write assembly (`src/pmedm_vb/assemble/`) -- attribute matrices,
      aggregation operators, targets, and the `Sigma` representation the
      solvers consume. One problem per PUMA: `assemble.build.build_all()`
- [x] write solvers (`src/pmedm_vb/solvers/`) -- MAP (`map_dual.solve_map`)
      and VB (`vb.solve_vb`, Gaussian or skewed family). Both work as
      algorithms; the VB fit is not yet usable as a simulator. Status,
      findings and the open decision: `solvers_synopsis.md`
- [ ] write experiments (`experiments/`)

## Open research questions

- **Choosing `alpha`.** The saddlepoint evidence (`MAPResult.log_evidence`)
  rises without bound as `alpha -> 0`, because tract totals are exactly the sums
  of their block groups in the model and so the tract-versus-block-group
  contrasts carry no sampling variance; the solver module docstring has the
  argument and the numbers. Candidates, none tried: an analytic shrinkage
  intensity estimated from the replicates alone (Schafer & Strimmer 2005,
  whose derivation assumes independent samples, which SDR replicates are
  not); the evidence restricted to the directions the sample informs; and
  held-out-table prediction.
- **Which Laplace approximation is the baseline.** Over `p` on the simplex, or
  over `lambda` (the MaxEnt family). They agree on the constrained totals and
  differ everywhere else, and the VB family -- a Gaussian over `lambda` --
  corresponds to the second. `map_dual.laplace_precision` returns the dual
  Hessian, from which either follows.
- **The posterior over `lambda` is one-sided for rare cells.** On Knox the
  Laplace approximation `N(lambda*, (nH)^-1)` fails badly: its median draw sits
  about 44,000 nats below the posterior (PUMA 4701501, alpha = 1), and VB beats
  it by 27,000-400,000. `experiments/laplace_diagnostic.py` traces this to
  cells published as 1, 2 or 6 with SEs of 1.4-2.6: the fitted mass on those
  attributes is tiny, so the curvature is small and the Laplace sd large, but
  lowering the multiplier multiplies those units' weight exponentially. Flat on
  one side, a wall on the other; VB, being Gaussian, can only shrink it. Two
  responses: a family that can be one-sided in those coordinates, or treating
  such tight small-count SEs as too small -- `variance_floor="zero"` floors
  every cell at its area's zero-count variance, to test the second. The floor
  removed the extra blow-up at small alpha but little at alpha = 1: the
  curvature comes from how rare the attribute is in the sample, not from its
  SE. So `solve_vb(..., family="skewed")` adds a per-coordinate sinh-arcsinh
  skew on top of the converged Gaussian, to test the first. It fits the bulk
  but keeps a tail; see the next question.
- **Making VB usable as a simulator.** In 10-17% of VB draws, more than 1%
  of a PUMA's units sit on one household record in one block group, and the
  posterior rejects those draws (`log pi - log q` hundreds to thousands of
  nats below typical). More draws per step halve the tail; a statewide
  support makes it worse. Next: an MCMC reference posterior for one PUMA.
  `solvers_synopsis.md` has the evidence and the options.
- **Group quarters.** A PUMS shortfall the fit cannot place; see the open
  questions in `census_data_sources.md`.

## Choosing constraint tables

A PMEDM run is defined as much by *which* published tables constrain it as by
the solver. Two questions decide that, and `pmedm_vb.data` answers both without
downloading anything large.

**1. What is published, and where?** Coverage thins as geography gets finer, and
block group is the binding constraint. `coverage()` returns one row per table
with a boolean per summary level:

```python
from pmedm_vb.config import StudyArea
from pmedm_vb.data.variance import coverage

area = StudyArea(name="knox", state="47", year=2024, counties=("093",))
cov = coverage(area)

cov[["tract", "block_group"]].sum()     # 2020-2024: 133 and 73
cov[cov.block_group]                    # what a block-group run may use
```

It reads the directory index once per level. Built table by table from
`is_available()` the same answer costs one request per table per geography --
268 for this vintage -- so `is_available()` is for asking about *a* table, and
`coverage()` for asking about all of them. `table_list()` gives the master list
with titles; `published_tables()` gives one level's IDs alone.

Two things to know when reading the result. Match `[BC]`, not `B`: `C02003`,
`C15010`, `C17002`, `C24010` and `C24030` all reach block group, and a `B`-only
habit hides them. And `B16001` is published at *neither* level despite appearing
in the master list, which is why the master count exceeds the tract count.

**2. Can PUMS reproduce its cells?** A table published at block group is still
unusable if the microdata cannot rebuild its categories break for break. That is
checked against the data dictionary, which is a small CSV -- not the
hundred-megabyte data zips:

```python
from pmedm_vb.data.pums import variables, variable_labels

variables(area)                          # every declared column, with its label
variable_labels(area, "JWTRNS")          # the codes one cell must be written against
```

`variables()` answers "does this vintage carry the column I think it does",
which is worth asking first because PUMS renames columns between vintages and a
missing one otherwise surfaces as a `KeyError` after the download.
`variable_labels()` gives the value codes and their published meanings --
the correspondence between a table cell and a set of PUMS codes is where silent
misfit comes from, so it is read rather than assumed. A continuous column
(`AGEP`, `HINCP`) declares no values and raises; `data_dictionary()` returns the
whole file if you would rather query it yourself.

**Granularity is a separate choice from table choice.** Because the replicate
files carry all 80 replicates rather than only a margin of error, collapsing
published cells is exact: summing cells is a linear map on the estimates, and
the same map applied to the replicate deviations gives the collapsed
constraints' covariance including the correlations between the merged cells. So
`B01001`'s 23 age bands per sex can become five without approximation --
which matters at block group, where fine bands are mostly zeros and every zero
cell falls back on the modelled `w·k` variance.

Everything above is cached under `$PMEDM_VB_DATA/raw/` on first call and reused
after, so re-running while deciding costs nothing. Pass `force=True` to refetch.

## Verification

Two scripts, plus a check that runs implicitly. All need census.gov, which is
unreachable from sandboxed environments.

```
tools/verify_vintage.sh                 # the data layer, against a vintage
tools/check_mapping.py [PUMA]           # every constraint definition at once
```

`verify_vintage.sh` confirms the endpoints are shaped as the downloaders expect
and that recomputed MOEs match Census's published ones to rounding -- the
strongest check available, since the files publish their own answer.

`check_mapping.py` compares weighted PUMS totals against published estimates
for every constraint, standardised by the published standard error. Its
docstring says how to read the result; briefly, a whole table off-centre is a
definition error and scattered large `|z|` is sampling.

Assembling a problem runs the third check implicitly -- `PMEDMInputs.validate()`
per PUMA, after `ConstraintTable.validate()` has checked every declared cell
against the published cell list:

```python
from pmedm_vb.assemble.build import build_all
problems = build_all(area, default_tables(area))
```

**Knox County is exactly four whole PUMAs** -- 4701501 to 4701504, 121 tracts,
301 block groups, none reaching into a neighbouring county -- so the whole-PUMA
expansion is a no-op here. `geography.puma_coverage(area)` reports this for any
study area and should be run before adopting a new one: a PUMA crossing the
boundary makes a per-PUMA run ill-posed, since its PUMS records represent all
of it.

## Every session

The steps under Setup below are one-time. Each new login needs only:

```
module purge
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data

python -c "import sys; print(sys.executable)"   # must be inside the env
```

**Do not `module load anaconda3`.** Source conda's shell hook from the module's
installation path directly, as above. Loading the module prepends its own `bin`
to `PATH`, and `conda activate` does not win that race -- the prompt and
`CONDA_PREFIX` change while `python` still resolves to the module's interpreter
(Python 3.8 or 3.12 depending on which module, rather than this environment's
3.11). Not loading it means there is nothing to lose the race to. Sourcing the
hook by absolute path gives you the `conda` command without that side effect.

As a `~/.bashrc` function, so it is one word:

```
pmedm() {
    module purge
    source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
    conda activate /lustre/isaac24/proj/UTK0496/envs/pmedm_vb
    export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data
    mkdir -p "$PMEDM_VB_DATA"
    case "$(python -c 'import sys; print(sys.executable)')" in
        "$CONDA_PREFIX"/*) echo "pmedm_vb: $(python -V) OK  data=$PMEDM_VB_DATA" ;;
        *) echo "SHADOWED: python is $(command -v python), not $CONDA_PREFIX/bin/python" ;;
    esac
}
```

The guard compares the interpreter's **path** against `$CONDA_PREFIX`, not its
version. An earlier version checked the version, and could not catch the case
it existed for: the anaconda3 2024.06 module ships Python 3.11.11, the same
minor version as this environment, so `python -V` reads correct while `python`
is the module's binary. `CONDA_PREFIX` and the prompt both look right too. The
symptom is `ModuleNotFoundError: No module named 'pmedm_vb'` from a shell that
appears fully activated.

`$CONDA_PREFIX/bin/python` always works and is worth using for real runs
regardless.

## Setup

Conda owns the interpreter and the compiled dependencies; `pyproject.toml` owns
the package. On ISAAC, run this from inside the clone:

```
PROJ=$(dirname "$PWD")                    # the project allocation holding this repo

module purge                              # drop the anaconda auto-loaded at login
source /sw/isaac/applications/anaconda3/2024.06/rhel8_cascadelake_binary/anaconda3-2024.06/etc/profile.d/conda.sh
conda --version                           # expect >= 23.10, i.e. the libmamba solver

export CONDA_PKGS_DIRS=$PROJ/conda_pkgs   # keep the tarball cache out of $HOME
conda env create -f environment.yml --prefix $PROJ/envs/pmedm_vb
conda activate $PROJ/envs/pmedm_vb

python -c "import sys; print(sys.executable)"   # MUST be inside $CONDA_PREFIX
$CONDA_PREFIX/bin/python -m pip install -e . --no-deps
```

**Why the install goes through `$CONDA_PREFIX/bin/python`.** It is immune to
`PATH` ordering, which is worth having in the one step that is hard to redo.
If a loaded anaconda module ever gets ahead of the environment on `PATH`, the
symptom is obscure rather than obvious: the module's python carries an old pip
(the 2021.05 module ships pip 21.0.1), which is below the pip 21.3 that PEP 660
editable installs of a pyproject-only project require, so `pip install -e .`
demands a `setup.py` that this project correctly does not have. The
`sys.executable` check on the preceding line catches that before it happens.

Not loading the anaconda module at all -- see **Every session** above -- is what
keeps `PATH` clean in the first place. In Slurm scripts, where `PATH` is less
predictable than in a login shell, prefer `$CONDA_PREFIX/bin/python script.py`
regardless.

Three things that all have the same cause -- nothing large may live in a quota'd
home directory:

- `--prefix` puts the environment in project space. Without it conda writes to
  `~/.conda/envs`, and a PyTorch environment runs to several GB.
- `CONDA_PKGS_DIRS` moves the *package cache*, which `--prefix` does not. It
  defaults to `~/.conda/pkgs` and holds the downloaded tarballs.
- `PMEDM_VB_DATA` sets the download cache root, and defaults to `./data`:

```
export PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_data
mkdir -p "$PMEDM_VB_DATA"
```

`config.py` resolves this path but deliberately does not create it, so the
`mkdir` is needed the first time. Note this is *scratch*: check ISAAC's purge
policy before treating anything cached here as durable. Re-downloading is only
an inconvenience, but a silently emptied cache is a confusing one.

The module version matters because conda's solver changed: libmamba became the
default in conda 23.10.0 (Nov 2023), so the 2021.05 module is on the old classic
solver, which handles `pytorch` + `conda-forge` + a pinned Python badly. If your
shell already shows `(base)` from the default module, `module swap` or start a
fresh shell rather than loading a second anaconda over the first.
