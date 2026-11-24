"""Pool step 4e's cut arm and apply its decision rule
(``PLAN_latent_step4e.md`` version 1, §3).

    python experiments/latent/collect_step4e.py --root $PMEDM_VB_RUNS/latent/step4e \
        --step4d $PMEDM_VB_RUNS/latent/step4d --step4c $PMEDM_VB_RUNS/latent/step4c

Reads ``step4e_cut.py``'s units (``<root>/FScut/<region>/r<k>/``) and, for
comparison, step 4d's FS and step 4c's S and P0, and writes the same files
as ``collect_step4d.py`` under ``<root>`` (:func:`collect_step4d.report`):
calibration of every class with FScut primary under step 4c's bands, the
zone totals side by side, the totals' point errors (``direct`` is the raw
Horvitz–Thompson total, ``map`` the totals module's mean), PIT histograms,
point accuracy and fit diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from collect_step4d import report, step4c_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path, help="step 4e output root")
    parser.add_argument("--step4d", required=True, type=Path, help="step 4d output root")
    parser.add_argument("--step4c", required=True, type=Path, help="step 4c output root")
    args = parser.parse_args()
    others, n_expected = step4c_groups(args.step4c)
    groups = {"FScut": sorted(args.root.glob("FScut/*/r*")),
              "FS": sorted(args.step4d.glob("FS/*/r*")), **others}
    report(args.root, "FScut", groups, n_expected,
           "Step 4e: the cut form of model version 5 (draft) on simulation S",
           "calibrated: model version 5 in its cut form, with step 4c's (tau, c), is the "
           "posterior; merge the draft into MODEL_latent.qmd and go to step 6", "step 4e")


if __name__ == "__main__":
    main()
