"""How much of Sigma is negative: the off-diagonal correlations of Sigma on the
published rows (untapered, no prior), by the published size of the two cells and
whether they share a tract. Rows with no replicate spread are left out.

    sbatch --export=ALL,PMEDM_VB_DATA=/lustre/isaac24/scratch/$USER/pmedm_vb_paper_data \
        experiments/run_python.sbatch sigma_sign.py knoxall-2024-5yr 0.01 \
        4701501 4701502 4701503 4701504
"""
import sys
import numpy as np
import pandas as pd
from pmedm_vb.assemble.inputs import PMEDMInputs
from pmedm_vb.config import processed_dir

area, alpha, pumas = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
CUTS = [1, 25, 100]                          # published value: 0 | 1-24 | 25-99 | 100+
CLASSES = ["zero", "1-24", "25-99", "100+"]
BINS = np.linspace(-1, 1, 201)

rows = []
for puma in pumas:
    inputs = PMEDMInputs.load(processed_dir() / "inputs" / area / puma)
    sigma = inputs.sigma(alpha, None, None)            # untapered, as the baseline
    b, sd = sigma.b, np.sqrt(sigma.diagonal())
    y, tract = inputs.targets(), inputs.constraint_tracts()
    keep = np.flatnonzero(np.square(b).sum(1) > 0)     # rows with replicate spread
    print(f"{puma}: {y.size} rows, {y.size - keep.size} with no replicate spread (left out)")
    b, sd, cls, tract = b[keep], sd[keep], np.searchsorted(CUTS, y[keep], side="right"), tract[keep]
    k = len(CLASSES)
    hist = np.zeros((k, k, 2, BINS.size - 1))
    for i0 in range(0, keep.size, 2000):
        i1 = min(i0 + 2000, keep.size)
        r = (b[i0:i1] @ b.T) / np.outer(sd[i0:i1], sd)
        ii, jj = np.nonzero(np.arange(i0, i1)[:, None] < np.arange(keep.size)[None, :])
        rho = r[ii, jj]
        a, c = cls[i0 + ii], cls[jj]
        lo, hi = np.minimum(a, c), np.maximum(a, c)
        same = (tract[i0 + ii] == tract[jj]).astype(int)
        bin_ = np.clip(np.searchsorted(BINS, rho, side="right") - 1, 0, BINS.size - 2)
        code = ((lo * k + hi) * 2 + same) * (BINS.size - 1) + bin_
        hist += np.bincount(code, minlength=hist.size).reshape(hist.shape)
    mid = (BINS[:-1] + BINS[1:]) / 2
    for a in range(k):
        for c in range(a, k):
            for same in (1, 0):
                h = hist[a, c, same]
                if not h.sum():
                    continue
                rows.append(dict(puma=puma, cells=f"{CLASSES[a]} x {CLASSES[c]}",
                                 tract="same" if same else "different", pairs=int(h.sum()),
                                 neg=h[mid < 0].sum() / h.sum(),
                                 neg_01=h[mid < -0.1].sum() / h.sum(),
                                 pos_01=h[mid > 0.1].sum() / h.sum(),
                                 neg_mass=(np.abs(mid) * h)[mid < 0].sum() / (np.abs(mid) * h).sum()))
out = pd.DataFrame(rows)
pooled = (out.assign(w=out.pairs).groupby(["cells", "tract"], sort=False)
          .apply(lambda g: pd.Series({"pairs": g.pairs.sum(),
                                      **{c: np.average(g[c], weights=g.pairs)
                                         for c in ["neg", "neg_01", "pos_01", "neg_mass"]}}))
          .reset_index())
pd.set_option("display.width", 200)
print(f"\nalpha {alpha}, correlations of Sigma off the diagonal, pooled over PUMAs")
print("neg: share < 0; neg_01 / pos_01: share below -0.1 / above 0.1; "
      "neg_mass: negative share of sum |rho|")
print(pooled.round(3).to_string(index=False))
tot = out.pairs.sum()
print(f"\nall pairs: {tot:,}; share < 0: {np.average(out.neg, weights=out.pairs):.3f}; "
      f"share < -0.1: {np.average(out.neg_01, weights=out.pairs):.3f}; "
      f"share > 0.1: {np.average(out.pos_01, weights=out.pairs):.3f}")
