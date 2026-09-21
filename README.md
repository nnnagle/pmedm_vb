# pmedm_vb
Penalized Max Ent Dasymetric Modeling problem as a variational bayes

The derivation of the original max ent problem is in pmedm_derivation.md.
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
- [ ] write downloaders (`src/pmedm_vb/data/`)
- [ ] write solvers (`src/pmedm_vb/solvers/`)
- [ ] write experiments (`experiments/`)

## Setup

Conda owns the interpreter and the compiled dependencies; `pyproject.toml` owns
the package. On ISAAC, run this from inside the clone:

```
PROJ=$(dirname "$PWD")                    # the project allocation holding this repo

module purge                              # drop the 2021.05 loaded at login
module load anaconda3/2024.06             # NOT bare `anaconda3` -- that is 2021.05
source $(conda info --base)/etc/profile.d/conda.sh
conda --version                           # expect >= 23.10, i.e. the libmamba solver

export CONDA_PKGS_DIRS=$PROJ/conda_pkgs   # keep the tarball cache out of $HOME
conda env create -f environment.yml --prefix $PROJ/envs/pmedm_vb
conda activate $PROJ/envs/pmedm_vb

python -c "import sys; print(sys.executable)"   # MUST be inside $CONDA_PREFIX
$CONDA_PREFIX/bin/python -m pip install -e . --no-deps
```

**On the `python` you get -- address the interpreter by path, not by name.**
On ISAAC a loaded anaconda module keeps its own `bin` ahead of the environment
on `PATH`, and `conda activate` does not win that race. Observed directly:
after `module purge`, `module load anaconda3/2024.06`, sourcing conda's shell
hook and activating the prefix, `CONDA_PREFIX` and the shell prompt were both
correct while `sys.executable` still pointed at the module's base interpreter.
The `source .../conda.sh` line above is kept because it is harmless and correct
practice, but it is *not* sufficient here.

The consequence is a confusing failure rather than an obvious one: the module's
python carries an old pip (2021.05 ships pip 21.0.1), which is below the pip
21.3 that PEP 660 editable installs of a pyproject-only project require, so
`pip install -e .` demands a `setup.py` that this project correctly does not
have.

So: run the `sys.executable` check after activating, and if it reports anything
outside `$CONDA_PREFIX`, do not try to repair `PATH` -- just invoke
`$CONDA_PREFIX/bin/python` directly, as the block above does. The same applies
in Slurm scripts, where `PATH` is even less predictable than in a login shell.
Prefer `$CONDA_PREFIX/bin/python script.py` over `python script.py` throughout.

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
