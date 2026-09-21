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
- [ ] write assembly (`src/pmedm_vb/assemble/`) -- attribute matrices,
      aggregation operators, targets, and the `Sigma` representation the
      solvers consume
- [ ] write solvers (`src/pmedm_vb/solvers/`)
- [ ] write experiments (`experiments/`)

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
    echo "pmedm_vb: $(python -V), $(python -c 'import sys; print(sys.executable)')"
}
```

The echo is a guard: anything other than 3.11 from inside the env prefix means
something has shadowed it again, and `$CONDA_PREFIX/bin/python` is the fallback
that always works.

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
