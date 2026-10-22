"""Do the MAP and shrunk Laplace perturbations pass HMC's start screens?

For c from 0 (the MAP) to 1 (Laplace draws), the share of 400 starts
lambda = MAP + c (Laplace draw - MAP) whose largest cell holds at most 1% of p
(starting_points's screen) and that lie inside a Tweedie term's bound, with
the median largest share. Untapered, no variance floor, as the baseline.

    sbatch --account=acf-utk0011 \
        --export=ALL,PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \
        experiments/run_python.sbatch start_check.py knoxall-2024-5yr 4701502 0.01 \
        puma-rpruned+prior3+tweedie18
"""
import sys
import numpy as np, torch
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.assemble.hierarchy import Hierarchy
from pmedm_vb.config import processed_dir
from pmedm_vb.solvers.map_dual import solve_map
from pmedm_vb.solvers.vb import StructuredGaussian, _DualTarget, DTYPE
from pmedm_vb.solvers.mcmc import largest_share, whitening_matrix

area, puma, alpha, level = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
draws, max_share = 400, 0.01
inputs = PMEDMInputs.load(processed_dir() / "inputs" / area / puma)
r = solve_map(inputs, alpha=alpha, taper=None, hierarchy=level)
h = Hierarchy.build(inputs, level, alpha=alpha, taper=None, variance_floor=None)
sigma = h.sigma(inputs, alpha, None, None)
q = StructuredGaussian.laplace(inputs, r)
target = _DualTarget(inputs, sigma, "cpu", h)
A = torch.as_tensor(whitening_matrix(q), dtype=DTYPE)
mean = torch.as_tensor(q.mean, dtype=DTYPE)
rng = np.random.default_rng(0)
x = torch.as_tensor(rng.standard_normal((draws, mean.numel())), dtype=DTYPE)
print(f"{puma} {level} alpha {alpha}: {mean.numel()} multipliers, {draws} draws per c, "
      f"screen: largest share <= {max_share:.0%} and inside the bound")
print(f"{'c':>6} {'pass':>6} {'share ok':>9} {'feasible':>9} {'median largest share':>21}")
for c in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    lam = mean + c * x @ A.T
    share, _ = largest_share(target, lam)
    ok_s, ok_f = share <= max_share, target.feasible(lam)
    print(f"{c:6.2f} {(ok_s & ok_f).float().mean():6.1%} {ok_s.float().mean():9.1%} "
          f"{ok_f.float().mean():9.1%} {share.median():21.4f}")
