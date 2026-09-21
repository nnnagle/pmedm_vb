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
Determine package folder structure
write downloaders
write solvers
write experiments
