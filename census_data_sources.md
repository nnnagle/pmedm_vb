# Census data sources for PMEDM

What the Census endpoints actually look like, and how we know.

Every fact here was confirmed on **2026-09-21** against a live directory
listing, file header, or published document — not recalled or inferred. It
exists because the endpoints are not self-describing: directory layouts differ
between products, casing is inconsistent, coverage varies by geography, and the
variance formulas live in a PDF.

The vintage probed in detail was **2019–2023 (2023 ACS 5-year)**. The
**2020–2024** vintage was then confirmed to have the same shape and is the one
to use — see [Vintage](#vintage) below. Replicate tables are published for 2014
through 2024. See [Open questions](#open-questions) for what remains unchecked.

This document is **descriptive**. `src/pmedm_vb/data/` holds the authoritative
constants and is what runs; if the two disagree, this file is stale.

---

## 1. PUMS microdata

Base: `https://www2.census.gov/programs-surveys/acs/data/pums`

| | |
|---|---|
| directory | `{base}/{year}/{span}-Year/` |
| casing | **`5-Year`, capital Y** — differs from the variance tree |
| layout | flat; one zip per state, no subdirectories |
| person file | `csv_p{st}.zip` |
| housing file | `csv_h{st}.zip` |
| `{st}` | lowercase postal abbreviation, e.g. `csv_ptn.zip` for Tennessee |

**Data dictionary** —
`https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_{start}-{end}.csv`

Machine-readable, two row forms:

```
NAME,<var>,<type>,<width>,"description"
VAL,<var>,<type>,<width>,<lo>,<hi>,<label>
```

Two columns matter more than the rest:

- **`PUMA`** — in the 2020–2024 dictionary this is a single 5-character column
  on the **2020 Census definition**, combined with `ST` for a unique code.
  There is no `PUMA10`/`PUMA20` split to reconcile. This is what allows the
  crosswalk in §4 to work from the 2020 file alone.
- **`SERIALNO`** — character, e.g. `2020GQ0000001`. Never read it as numeric.

`ST` and `PUMA` are numeric-looking and lose leading zeros if a reader infers
their type. Read as text, then zero-pad defensively.

---

## 2. Variance Replicate Estimate Tables

Base: `https://www2.census.gov/programs-surveys/acs/replicate_estimates`

| | |
|---|---|
| data | `{base}/{year}/data/{span}-year/{sumlevel}/{TBLID}_{stfips}.csv.zip` |
| documentation | `{base}/{year}/documentation/{span}-year/{filename}` |
| casing | **`5-year`, lowercase** — differs from the PUMS tree |
| zip member | nested under the level, e.g. `150/B01003_47.csv` |

These files carry the estimate, the MOE **and** the 80 replicates together, so
one download supplies both `Y` and `Sigma`. The API does not serve replicates.

### Summary levels

`140` = census tract, `150` = block group. Also published: `010` US, `040`
state, `050` county, `060` county subdivision, `160` place, `250` AIANNH,
`310` CBSA, `500` congressional district, `860` ZCTA.

### File suffix

Two-digit **state FIPS**. Codes `03`, `07` and `14` are absent, which is what
identifies the suffix as FIPS rather than a sequence — those three are
unassigned.

### Column layout

```
TBLID, GEOID, NAME, ORDER, TITLE, ESTIMATE, MOE, CME, SE, Var_Rep1 … Var_Rep80
```

- One row per **(geography, cell)**.
- The **first two rows of every file** are the table title and its universe.
  They carry a `TBLID` but a blank `GEOID`, and must be dropped.
- `GEOID` is a Census GEO_ID: `1500000US470010201001` at block group,
  `1400000US47001020100` at tract. Split on `US`.
- `ORDER` is the 1-based cell index. `TBLID` plus a zero-padded `ORDER`
  reproduces Census variable naming (`B25003_001`), which makes a cell key
  recognisable against published documentation.
- `CME` is the MOE as a display string (`+/-251`) — redundant with `MOE`.

### Coverage

| scope | 2019–2023 | 2020–2024 |
|---|---|---|
| master list (`VRE_TABLE_LIST_{year}.csv`, columns `TBLID,TITLE`) | 132 | 134 |
| tract (`140`) | 131 | 133 |
| block group (`150`) | **73** | **73** |

Two tables were added at tract between the vintages. Block group is unchanged,
which is the count that actually constrains a PMEDM run.

Coverage is strictly nested — the 73 at block group are a subset of the master
list. Block group is the binding constraint for PMEDM.

**Absent from the program entirely:** household size. No `B11016`, `B25009` or
`B08201` at any geography. A household-size constraint would have to come from
the detailed-table API with an MOE-derived diagonal.

**In the program but not at block group** (59 tables — still usable as *tract*
constraints): `B05003` nativity · `B14001` school enrolment · `B18101`
disability · `B09001` · `B23001` · `B27001`–`B27003` · `B17001`/`B17024` ·
`B16001`/`C16001` · `B25048`/`B25052`/`B25106` · `B11010` · `B22001` SNAP ·
`B26001` group quarters · the `B17010A`–`I` race iterations · the `*PR` Puerto
Rico variants.

**Watch the prefix.** C-tables are published too (`C02003`, `C15010`, `C17002`,
`C24010`, `C24030` are all at block group). A `B`-only pattern silently hides
them — this caused a wrong coverage claim during probing.

---

## 3. Variance formulas

Source: `2019-2023_Variance_Replicate_Table_Documentation.pdf`, in the
`documentation/5-year/` directory.

### Successive-differences replication

$$\mathrm{Var}(\hat X) = \frac{4}{80}\sum_{r=1}^{80}(\hat X_r - \hat X)^2
\qquad \mathrm{MOE} = 1.645\sqrt{\mathrm{Var}}$$

- Deviations are taken against the **published full-sample estimate**, not the
  replicate mean. The two differ.
- The `4/80` is an artifact of using SDR with 80 replicates
  (Fay & Train 1995).
- `1.645` is the 90 percent normal deviate, the level ACS publishes MOEs at.

### Cross-cell covariance

$$\mathrm{Cov}(a,b) = \frac{4}{80}\sum_{r=1}^{80}(a_r-\hat a)(b_r-\hat b)$$

The documentation licenses formula (1) for **any** derived statistic, which is
what justifies this bilinear form. The cross-cell case is not itself written
out in the document — that extension is ours.

> **Rank.** This is a sum of 80 outer products, so `Sigma` has rank ≤ 80 no
> matter how many cells there are. A block-group run has thousands of cells, so
> a dense `Sigma` is both large and singular, and `Sigma⁻¹` does not exist.
> Carry the `(n, 80)` deviations and let the solver exploit the low-rank
> structure. This is a live question for the VB formulation, not a detail.

### Zero counts

A zero count has all 80 replicates equal to the estimate, so its SDR variance
is exactly zero — and at block group these are common, not exceptional. Census
models it instead:

$$\mathrm{MOE}_0 = 1.645\sqrt{w\cdot k} \quad\Longrightarrow\quad
\mathrm{Var}_0 = w\cdot k$$

Neither `w` nor `k` varies by cell, so one variance covers every zero cell in a
geography.

> The documentation states that average weights and k-values are derived from
> **internal files not available to the public**, so a recomputed MOE may not
> reproduce the published one exactly. Expected, not a bug.

---

## 4. Model parameters

### Average weight `w`

File: `VRE_AVERAGE_WEIGHT_{year}.csv` (documentation directory)

Columns `GEOGRAPHY,STATE,AVERAGE_WEIGHT`; 53 rows — US, 50 states, DC, PR.
`STATE` is FIPS, with `US` for the national row. The values are
vintage-specific: Tennessee (47) is 17 for 2019–2023 and 18 for 2020–2024,
so the file must be read for the vintage in use rather than cached across
vintages.

Block groups and tracts nest within states, so the national row is never the
right one for our purposes.

### k-value

Source: Table APP2, Appendix A
(`2019-2023_Appendix_A_Average_Weights_and_k-Values.xlsx`, or its PDF twin).
Keyed on the geography's **total population**, from `B01003`.

| total population | k |
|---|---|
| 4,999 or less | 4 |
| 5,000 – 9,999 | 8 |
| 10,000 – 19,999 | 10 |
| 20,000 – 29,999 | 14 |
| 30,000 – 49,999 | 18 |
| 50,000 or more | 22 |

Worked examples from the documentation, both reproduced by the implementation:
pop 25,000 → k = 14; pop 75,659 → k = 22.

**Consequence:** `B01003` is fetched on any run using the zero-count model,
whether or not it is a constraint. And since block groups run roughly
600–3,000 people, nearly all land in the k = 4 bracket — a Tennessee block
group's zero cells get `Var = w × 4` — 68 for 2019–2023, 72 for 2020–2024.

---

## 5. Geography

File:
`https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt`

Columns `STATEFP,COUNTYFP,TRACTCE,PUMA5CE`; 85,452 national rows. All four are
zero-padded fixed-width codes — read as text.

| derived | construction | width |
|---|---|---|
| tract GEOID | `STATEFP + COUNTYFP + TRACTCE` | 2+3+6 = 11 |
| PUMA GEOID | `STATEFP + PUMA5CE` | 2+5 = 7, matching PUMS `ST + PUMA` |

**Block group nesting is string slicing.** A 12-character block group GEOID
contains its parent tract GEOID as its first 11 characters. No spatial join.

**Block group universe** comes from the `B01003` replicate file at level `150`
— exactly the set of block groups constraints are published for, which is the
set a run can actually use.

Together these mean **no GIS dependency**: both relationships PMEDM needs are
satisfied without geometry, so no TIGER extract and no spatial library. A
geometry source is only needed if the simulator wants to map its output.

`rel2020/` also holds `aiannh/`, `blkgrp/`, `cbsa/`, `cd-sld/`, `cousub/`,
`place/`, `puma520/`, `t10t20/`, `tract/`, `ua/`, `zcta520/`.

---

## Vintage: use 2020–2024

Verified 2026-09-21 that the 2024 tree matches the 2023 one in every respect
the code depends on:

| check | result |
|---|---|
| summary-level directories | identical set; `140` and `150` present |
| documentation filenames | `VRE_TABLE_LIST_2024.csv`, `VRE_AVERAGE_WEIGHT_2024.csv`, `2020-2024_Appendix_A_Average_Weights_and_k-Values.{pdf,xlsx}`, `2020-2024_Variance_Replicate_Table_Documentation.pdf` |
| block-group coverage | 73 tables, same as 2023 |
| PUMS | `pums/2024/` holds both `1-Year/` and `5-Year/` |

Every one of those paths is what `pmedm_vb.data` constructs from `area.year`
and `area.span`, so selecting the vintage is `StudyArea(..., year=2024)` and
nothing else.

**Why this vintage rather than 2023.** The 2020–2024 PUMS dictionary codes a
single `PUMA` column on the 2020 Census definition (§1), so the period does not
span the PUMA redraw and the `rel2020` crosswalk applies to all five years
without reconciliation. The 2019–2023 period does span it, and its dictionary
was never checked.

PUMS and the replicate tables must come from the *same* vintage — the PUMA
coding and the published geographies have to agree.

---

## Open questions

| question | why it matters |
|---|---|
| **2019–2023 PUMA coding** | Its PUMS dictionary was never checked, and that period spans the redraw. Only matters if a run needs that vintage; 2020–2024 avoids the question. |
| **Table-by-geography list** | `VRE_Table_and_Geo_List_{year}.xlsx` holds per-geography coverage but was not parsed, to avoid an Excel reader dependency. `is_available()` asks the server with a HEAD request instead. |
| **Suppressed values** | Whether `ESTIMATE` or the replicates ever carry non-numeric suppression markers was not observed. The parser coerces, so such a value becomes `NaN` rather than failing loudly. |
| **PUMS zip members** | The PUMS zips were never opened — too large to probe casually. `load_pums` reads every `.csv` member and concatenates, which is correct whether a state ships one file or several. |

## Reproducing a probe

`tools/verify_vintage.sh` replays this sequence against any vintage. Run it
before adopting a new one:

```bash
tools/verify_vintage.sh              # defaults to 2024, 5-year, TN, Knox County
tools/verify_vintage.sh 2025 5 47 093
PYTHON=$CONDA_PREFIX/bin/python tools/verify_vintage.sh   # on a cluster
```

It checks, in order: reachability; that the vintage is published for both
products; that `140` and `150` exist; that the documentation files the code
builds by name are present; table coverage at both levels; that the state's
PUMS zips exist; that the crosswalk columns are unchanged and the PUMS
dictionary still carries a single `PUMA` column; and finally an end-to-end
run that recomputes MOEs and compares them against the published ones. It
exits non-zero if anything a downloader depends on is missing.

The last of those is the real accuracy check. The files publish their own
`MOE`, so the SDR implementation can be checked against Census's own answer.
Published MOEs are integers and recomputed ones are continuous, so agreement
means **max |difference| ≤ 0.5 with median ≈ 0.25** — the signature of
`published == round(ours)`, and a bound that a transposed or misread replicate
column breaks immediately. The 2023 vintage gives max 0.500, median 0.248.

Two things the script encodes that are easy to get wrong by hand:

- **Match `[BC]`, not `B`.** A `B`-only pattern silently hides `C02003`,
  `C15010`, `C17002`, `C24010` and `C24030`, all published at block group.
  This produced a wrong coverage claim during the original probing.
- **The crosswalk file is CRLF.** `head -1` on it yields a trailing `\r`, so a
  naive string comparison against the expected header fails on something
  invisible. pandas handles CRLF natively, so this only ever bites shell
  checks, never `geography.py`.
- **Never probe a PUMS zip with a plain `GET`.** They are hundreds of
  megabytes. The script uses a one-byte range request, which also avoids
  assuming the server honours `HEAD` — an assumption never tested against this
  tree. `variance.is_available()` asks the same way for the same reason.

For a listing by hand, with the page furniture stripped:

```bash
L() { curl -sS --max-time 60 "$1" \
      | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"$//' \
      | grep -vE '^(https?:|/|\?C=)' | sort -u; }

L https://www2.census.gov/programs-surveys/acs/replicate_estimates/2024/data/5-year/150/
```

census.gov is unreachable from some sandboxed environments (the egress proxy
returns 403 to `CONNECT`), so this may need running from a host with direct
access. The script detects that in its first check and stops rather than
reporting misleading failures.
