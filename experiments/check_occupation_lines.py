"""Check where Census puts two occupations among ``C24010``'s published lines.

The 2018 Census Occupation Code List groups occupation codes only down to the
SOC minor groups, not to every ``C24010`` line, so two placements in
:data:`pmedm_vb.assemble.allcells.SOC_LINES` are read off SOC and need
confirming:

- **dental hygienists** (OCCP 3310, SOC 29-1292) -- with the health
  technologists and technicians (29-2), among whose Census codes 3310 sits (the
  placement used, since a first run of this check), or with the health
  diagnosing and treating practitioners and other technical occupations (SOC
  29-1 and 29-9), where SOC puts them;
- **other healthcare practitioners and technical occupations** (OCCP 3550, SOC
  29-9000) -- with the practitioners, as that line's title says (the placement
  used), or with the technologists, after whose codes 3550 comes;
- **supervisors of transportation and material moving workers** (OCCP 9005,
  SOC 53-1000) -- transportation, as the code list groups them (9005-9430; the
  placement used), or material moving.

For each, the statewide PUMS total of the two lines, weighted by ``PWGTP``
over the civilian employed (``ESR`` 1, 2), is set against the published lines
summed over every tract in the state, under both placements. Moving an
occupation shifts its weighted count from one line to the other, so the
placement Census uses should leave both lines closer to their published
totals. PUMS and the summary tables do not agree exactly -- the PUMS is a
subsample of the survey with its own weights -- so each line's two standard
errors are printed (the PUMS one from its 80 replicate weights, the published
one from the tracts' summed replicates), and a placement is only told apart
when the occupation's count is large against them.

    $CONDA_PREFIX/bin/python experiments/check_occupation_lines.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pmedm_vb.assemble.allcells import soc_lines
from pmedm_vb.config import StudyArea
from pmedm_vb.data.pums import load_pums
from pmedm_vb.data.variance import SDR_FACTOR, fetch_replicates

#: Male-block ``C24010`` orders of the lines compared; female = male + 36.
LINES = {"health_practitioners": 17, "health_technologists": 18,
         "transportation": 36, "material_moving": 37}

#: The two occupations, the line each is placed in, and the alternative.
QUESTIONS = (
    ("dental hygienists", ("291292",), "health_technologists", "health_practitioners"),
    ("other healthcare practitioners and technical", ("299000",), "health_practitioners",
     "health_technologists"),
    ("transportation and material moving supervisors", ("531000",), "transportation",
     "material_moving"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", default="47")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--span", type=int, default=5)
    return parser.parse_args()


def published(area: StudyArea) -> pd.DataFrame:
    """Each compared line summed over the state's tracts, with its SE."""
    frame = fetch_replicates(area, ["C24010"], geography="tract")
    reps = [f"rep_{r}" for r in range(1, 81)]
    totals = frame.groupby(level="cell")[["estimate", *reps]].sum()
    se = np.sqrt(SDR_FACTOR * np.square(totals[reps].sub(totals["estimate"], axis=0)).sum(axis=1))
    return pd.DataFrame({"published": totals["estimate"], "published_se": se})


def main() -> None:
    args = parse_args()
    area = StudyArea(name="state", state=args.state, year=args.year, span=args.span)
    persons = load_pums(area, ["SEX", "ESR", "SOCP"])
    persons = persons[persons["ESR"].isin(("1", "2"))]
    weights = ["PWGTP", *(f"PWGTP{r}" for r in range(1, 81))]
    persons[weights] = persons[weights].apply(pd.to_numeric)
    pub = published(area)
    line_of = {c: (soc_lines(str(c)) or [None])[0] for c in persons["SOCP"].dropna().unique()}
    persons["line"] = persons["SOCP"].map(line_of)

    for name, codes, used, alternative in QUESTIONS:
        moved = persons["SOCP"].isin(codes)
        print(f"\n{name} (SOCP {', '.join(codes)}): placed in {used}; "
              f"alternative {alternative}")
        for sex, code, offset in (("male", "1", 0), ("female", "2", 36)):
            sel = persons["SEX"] == code
            count = persons.loc[sel & moved, "PWGTP"].sum()
            print(f"  {sex}: weighted count {count:,.0f}")
            for line in (used, alternative):
                base = sel & (persons["line"] == line) & ~moved
                cell = f"C24010_{LINES[line] + offset:03d}"
                row = pub.loc[cell]
                for placement, add in (("as placed", line == used), ("alternative", line != used)):
                    mask = base | (moved & sel) if add else base
                    tot = persons.loc[mask, weights].sum()
                    pums_se = np.sqrt(SDR_FACTOR * np.square(tot.iloc[1:] - tot.iloc[0]).sum())
                    gap = tot.iloc[0] - row["published"]
                    print(f"    {line:22s} {placement:11s} PUMS {tot.iloc[0]:>10,.0f} "
                          f"(SE {pums_se:,.0f})  published {row['published']:>10,.0f} "
                          f"(SE {row['published_se']:,.0f})  PUMS - published {gap:>+9,.0f}")


if __name__ == "__main__":
    main()
