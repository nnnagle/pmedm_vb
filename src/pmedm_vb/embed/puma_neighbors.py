"""PUMA neighbour diagnostic: do PUMAs with similar tables have similar people in z-space?

A coarse check of the embedding's geometry, meant for large contrasts only
(urban cores against suburbs against rural PUMAs, one metro against another)
and not for fine neighbour structure: with 58 PUMAs, individual neighbour lists
would be noise. Nothing is fitted; it reads saved positions and the PUMS.

Four distances between every pair of PUMAs, each from the PUMA's own
design-weighted units:

* **tables** -- Hellinger distance between the PUMAs' within-table shares,
  averaged over the leave-one-PUMA-out check's S1 tables
  (:data:`~pmedm_vb.embed.puma_check.S0` plus
  :data:`~pmedm_vb.embed.puma_check.S1_ADD`), each table weighted equally.
  Hellinger is bounded, symmetric, defined at zero cells, and does not let one
  table with many cells dominate.
* **z** -- distance between the PUMAs' weighted distributions of learned
  positions ``z1..zd`` (the group-quarters indicators are left out: the unit
  type is already one of the S1 tables). Two versions: the **energy
  distance** (square root of the V-statistic between weighted subsamples, so it
  is a true distance between the two empirical samples), and the **Gaussian
  2-Wasserstein** distance between the weighted means and covariances, as a
  robustness check that uses no sampling.
* **svd** -- the same two distances on a baseline embedding of the same d: a
  design-weighted truncated SVD (principal components) of the units' S1
  constraint-cell counts, columns centred and scaled to unit weighted sd (a
  correlation-matrix PCA, so cells with many members do not dominate), scores
  whitened to unit variance like z. It separates "any low-dimensional summary
  of the constraint cells" from "the learned embedding".
* **joint** -- the reference: Hellinger distance on the PUMAs' unconstrained
  joint structure, averaged over the check's scoring crosses (race x poverty
  stacked, householder race x income, householder race x tenure, tenure x
  income, household type x income). This is what neither tables nor
  positions are told directly.

Summaries (and nothing finer):

1. **Alignment**: Spearman rank correlation between the table distance and each
   other distance over all PUMA pairs, with a Mantel permutation p-value
   (PUMA labels of one matrix permuted jointly over rows and columns).
2. **Groupings**: average-linkage hierarchical clustering of each distance into
   k groups, agreement with the table-space and joint-space groupings by the
   adjusted Rand index. Average linkage because the distances are not all
   Euclidean (Ward's criterion assumes they are).
3. **Added value**: Spearman correlation of each distance (tables, z, svd) with
   the joint-structure distance, with Mantel p-values: whether z tracks
   unconstrained joint structure at least as well as the tables do.

Cluster labels are aligned across spaces to the table-space labels by maximum
overlap (the Hungarian assignment), so a map can colour them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform
from scipy.stats import rankdata

from pmedm_vb.embed.puma_check import S0, S1_ADD, CheckData
from pmedm_vb.progress import stage

#: The scoring crosses whose joint structure is the reference distance.
JOINT_TABLES = ("B17010A-G", "hh_race8xB19001", "hh_race8xB25003", "B25003xB19001",
                "household_typexB19001")

#: The S1 constraint set of the leave-one-PUMA-out check.
S1_TABLES = (*S0, *S1_ADD)

#: Linkage for the hierarchical clustering.
LINKAGE = "average"


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------


def puma_index(data: CheckData) -> tuple[list[str], np.ndarray]:
    """The PUMAs in order, and each unit's position in that order."""
    pumas = data.pumas
    code = pd.Index(pumas).get_indexer(data.puma)
    return pumas, code


def puma_totals(counts: sp.csr_matrix, weight: np.ndarray, code: np.ndarray,
                n_puma: int) -> np.ndarray:
    """PUMAs x cells: design-weighted counts of each PUMA's own units."""
    member = sp.csr_matrix((weight, (code, np.arange(len(code)))),
                           shape=(n_puma, len(code)))
    return np.asarray((member @ counts).todense(), dtype=np.float64)


def hellinger(shares: np.ndarray) -> np.ndarray:
    """PUMAs x PUMAs Hellinger distance between rows of within-table shares."""
    root = np.sqrt(shares)
    affinity = np.clip(root @ root.T, 0.0, 1.0)
    out = np.sqrt(1.0 - affinity)
    np.fill_diagonal(out, 0.0)
    return out


def table_distance(data: CheckData, tables, code: np.ndarray, n_puma: int) -> np.ndarray:
    """Mean Hellinger distance over ``tables``, each table counting once."""
    total = np.zeros((n_puma, n_puma))
    for name in tables:
        cells = puma_totals(data.counts[name], data.weight, code, n_puma)
        sums = cells.sum(axis=1, keepdims=True)
        if (sums <= 0).any():
            raise ValueError(f"{name}: a PUMA has no units in the table's universe")
        total += hellinger(cells / sums)
    return total / len(tables)


def weighted_samples(weight: np.ndarray, code: np.ndarray, n_puma: int, size: int,
                     seed: int) -> list[np.ndarray]:
    """For each PUMA, ``size`` unit indices drawn with replacement with
    probability proportional to the design weight (one draw, reused by every
    space so that their energy distances compare like with like)."""
    rng = np.random.default_rng(seed)
    out = []
    for j in range(n_puma):
        rows = np.flatnonzero(code == j)
        prob = weight[rows] / weight[rows].sum()
        out.append(rng.choice(rows, size=size, replace=True, p=prob))
    return out


def energy_distance(z: np.ndarray, samples: list[np.ndarray]) -> np.ndarray:
    """PUMAs x PUMAs energy distance between the weighted samples' positions:
    sqrt(2 E|X-Y| - E|X-X'| - E|Y-Y'|), all as V-statistics (diagonal pairs
    included), which keeps it a true, nonnegative distance between the two
    empirical samples."""
    points = [z[s] for s in samples]
    n = len(points)
    within = np.array([cdist(p, p).mean() for p in points])
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            between = cdist(points[a], points[b]).mean()
            value = max(2.0 * between - within[a] - within[b], 0.0)
            out[a, b] = out[b, a] = np.sqrt(value)
    return out


def _psd_sqrt(m: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((m + m.T) / 2.0)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T


def gaussian_w2(z: np.ndarray, weight: np.ndarray, code: np.ndarray, n_puma: int) -> np.ndarray:
    """PUMAs x PUMAs 2-Wasserstein distance between Gaussians with each PUMA's
    design-weighted mean and covariance of the positions."""
    means, covs, roots = [], [], []
    for j in range(n_puma):
        rows = code == j
        w = weight[rows] / weight[rows].sum()
        mu = w @ z[rows]
        centred = z[rows] - mu
        cov = (centred * w[:, None]).T @ centred
        means.append(mu)
        covs.append(cov)
        roots.append(_psd_sqrt(cov))
    out = np.zeros((n_puma, n_puma))
    for a in range(n_puma):
        for b in range(a + 1, n_puma):
            cross = _psd_sqrt(roots[b] @ covs[a] @ roots[b])
            value = (np.sum((means[a] - means[b]) ** 2)
                     + np.trace(covs[a]) + np.trace(covs[b]) - 2.0 * np.trace(cross))
            out[a, b] = out[b, a] = np.sqrt(max(value, 0.0))
    return out


def svd_positions(data: CheckData, tables, d: int) -> np.ndarray:
    """Baseline positions: a design-weighted principal-component analysis of
    the units' constraint-cell counts in ``tables``. Columns are centred and
    scaled to unit weighted sd (cells with zero variance dropped); the first d
    component scores are whitened to unit weighted variance, like z."""
    counts = sp.hstack([data.counts[t] for t in tables]).tocsc().astype(np.float64)
    w = data.weight / data.weight.sum()
    dense = counts.toarray()
    mean = w @ dense
    sd = np.sqrt(np.clip(w @ (dense - mean) ** 2, 0.0, None))
    keep = sd > 0
    standard = (dense[:, keep] - mean[keep]) / sd[keep]
    gram = (standard * w[:, None]).T @ standard
    vals, vecs = np.linalg.eigh(gram)
    order = np.argsort(vals)[::-1][:d]
    scores = standard @ vecs[:, order]
    return scores / np.sqrt(vals[order])


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def mantel(a: np.ndarray, b: np.ndarray, permutations: int, seed: int) -> tuple[float, float]:
    """Spearman correlation of two distance matrices over all pairs, and the
    one-sided permutation p-value of a correlation at least that large, the
    PUMA labels of ``b`` permuted jointly over rows and columns."""
    ra = rankdata(_upper(a))
    ra = (ra - ra.mean()) / ra.std()

    def rho(m):
        rb = rankdata(_upper(m))
        return float(np.mean(ra * (rb - rb.mean()) / rb.std()))

    observed = rho(b)
    rng = np.random.default_rng(seed)
    n = len(b)
    hits = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        hits += rho(b[np.ix_(perm, perm)]) >= observed
    return observed, (hits + 1) / (permutations + 1)


def cluster(distance: np.ndarray, k: int) -> np.ndarray:
    """Average-linkage hierarchical clustering into k groups, labels 1..k."""
    tree = linkage(squareform(distance, checks=False), method=LINKAGE)
    return fcluster(tree, t=k, criterion="maxclust")


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index of two labelings (Hubert and Arabie)."""
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    table = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(table, (ia, ib), 1)

    def pairs(x):
        return float(np.sum(x * (x - 1) / 2.0))

    n = len(a)
    index = pairs(table)
    rows, cols = pairs(table.sum(axis=1)), pairs(table.sum(axis=0))
    expected = rows * cols / (n * (n - 1) / 2.0)
    maximum = (rows + cols) / 2.0
    return 1.0 if maximum == expected else (index - expected) / (maximum - expected)


def align(labels: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Relabel ``labels`` to agree with ``reference`` as far as possible (the
    assignment of maximum overlap); groups left unmatched get the next free
    numbers."""
    groups, ref_groups = np.unique(labels), np.unique(reference)
    overlap = np.array([[np.sum((labels == g) & (reference == r)) for r in ref_groups]
                        for g in groups])
    row, col = linear_sum_assignment(-overlap)
    mapping = {groups[i]: ref_groups[j] for i, j in zip(row, col)}
    spare = iter(range(int(max(ref_groups.max(), groups.max())) + 1, 10_000))
    for g in groups:
        mapping.setdefault(g, next(spare))
    return np.array([mapping[g] for g in labels])


# --------------------------------------------------------------------------
# The diagnostic
# --------------------------------------------------------------------------


@dataclass
class Result:
    pumas: list[str]
    distances: dict[str, np.ndarray] = field(default_factory=dict)
    mantel: pd.DataFrame | None = None
    ari: pd.DataFrame | None = None
    joint: pd.DataFrame | None = None
    clusters: pd.DataFrame | None = None


def run_diagnostic(data: CheckData, runs: dict[str, np.ndarray], *, ks=(4, 5, 6),
                   sample: int = 500, permutations: int = 999, seed: int = 1) -> Result:
    """Distances, alignment, groupings and joint-structure correlations.

    ``runs`` maps a label to each unit's learned positions (units x d, aligned
    to ``data.units``). An SVD baseline is added for every distinct d.
    """
    pumas, code = puma_index(data)
    n = len(pumas)
    result = Result(pumas=pumas)
    with stage(f"table distance ({len(S1_TABLES)} S1 tables, {n} PUMAs)"):
        result.distances["tables"] = table_distance(data, S1_TABLES, code, n)
    with stage(f"joint-structure distance ({len(JOINT_TABLES)} crosses)"):
        result.distances["joint"] = table_distance(data, JOINT_TABLES, code, n)

    samples = weighted_samples(data.weight, code, n, sample, seed)
    spaces = dict(runs)
    for d in sorted({z.shape[1] for z in runs.values()}):
        with stage(f"SVD baseline, d = {d}"):
            spaces[f"svd{d}"] = svd_positions(data, S1_TABLES, d)
    for label, z in spaces.items():
        with stage(f"distances in {label} (d = {z.shape[1]})"):
            result.distances[f"{label}:energy"] = energy_distance(z, samples)
            result.distances[f"{label}:w2"] = gaussian_w2(z, data.weight, code, n)

    others = [s for s in result.distances if s not in ("tables", "joint")]
    with stage(f"Mantel tests ({permutations} permutations)"):
        rows = []
        for s in [*others, "joint"]:
            r, p = mantel(result.distances["tables"], result.distances[s], permutations, seed)
            rows.append({"space": s, "spearman_vs_tables": r, "p": p})
        result.mantel = pd.DataFrame(rows)
        rows = []
        for s in ["tables", *others]:
            r, p = mantel(result.distances["joint"], result.distances[s], permutations, seed)
            rows.append({"space": s, "spearman_vs_joint": r, "p": p})
        result.joint = pd.DataFrame(rows)

    labels, ari = [], []
    for k in ks:
        reference = cluster(result.distances["tables"], k)
        joint_ref = cluster(result.distances["joint"], k)
        for s, dist in result.distances.items():
            raw = reference if s == "tables" else cluster(dist, k)
            aligned = align(raw, reference)
            labels.append(pd.DataFrame({"puma": pumas, "space": s, "k": k,
                                        "label": raw, "aligned": aligned}))
            ari.append({"space": s, "k": k, "ari_vs_tables": adjusted_rand(raw, reference),
                        "ari_vs_joint": adjusted_rand(raw, joint_ref)})
    result.clusters = pd.concat(labels, ignore_index=True)
    result.ari = pd.DataFrame(ari)
    return result
