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
the package. On ISAAC:

```
module load anaconda3
conda env create -f environment.yml
conda activate pmedm_vb
pip install -e . --no-deps
export PMEDM_VB_DATA=/lustre/isaac/scratch/$USER/pmedm_vb_data
```

`PMEDM_VB_DATA` sets the download cache root; it defaults to `./data`, which is
wrong on a cluster.
