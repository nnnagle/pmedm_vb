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

module load anaconda3/2024.06             # NOT bare `anaconda3` -- that is 2021.05
conda --version                           # expect >= 23.10, i.e. the libmamba solver

export CONDA_PKGS_DIRS=$PROJ/conda_pkgs   # keep the tarball cache out of $HOME
conda env create -f environment.yml --prefix $PROJ/envs/pmedm_vb
conda activate $PROJ/envs/pmedm_vb

python -m pip --version                   # must report a pip inside the prefix
python -m pip install -e . --no-deps
```

Use `python -m pip`, not bare `pip`. `environment.yml` installs pip into the
environment, but `python -m pip` additionally guarantees the install binds to
the interpreter that is actually active rather than to whatever pip `PATH`
happens to find first -- which on a module-based cluster is often a read-only
base installation with a pip too old for editable installs.

Three things that all have the same cause -- nothing large may live in a quota'd
home directory:

- `--prefix` puts the environment in project space. Without it conda writes to
  `~/.conda/envs`, and a PyTorch environment runs to several GB.
- `CONDA_PKGS_DIRS` moves the *package cache*, which `--prefix` does not. It
  defaults to `~/.conda/pkgs` and holds the downloaded tarballs.
- `PMEDM_VB_DATA` sets the download cache root, and defaults to `./data`:

```
export PMEDM_VB_DATA=<scratch space>/pmedm_vb_data
```

`<scratch space>` is a placeholder. Take the real path from the OIT storage
documentation for ISAAC rather than guessing -- an earlier version of this file
had a made-up path here.

The module version matters because conda's solver changed: libmamba became the
default in conda 23.10.0 (Nov 2023), so the 2021.05 module is on the old classic
solver, which handles `pytorch` + `conda-forge` + a pinned Python badly. If your
shell already shows `(base)` from the default module, `module swap` or start a
fresh shell rather than loading a second anaconda over the first.
