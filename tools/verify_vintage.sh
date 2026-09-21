#!/usr/bin/env bash
#
# Verify an ACS vintage before adopting it.
#
# This is the sequence that established the 2019-2023 layout and then confirmed
# 2020-2024 was a drop-in, kept runnable so the next vintage costs one command
# rather than a round of probing. Each section prints what it establishes; the
# script exits non-zero if anything a downloader depends on is missing or if a
# recomputed MOE disagrees with the published one.
#
# Must run somewhere census.gov is reachable. Sandboxed environments are often
# blocked by an egress proxy (403 to CONNECT), in which case section 0 fails
# immediately and the rest is meaningless.
#
#   tools/verify_vintage.sh [YEAR] [SPAN] [STATE_FIPS] [COUNTY_FIPS]
#   tools/verify_vintage.sh 2024 5 47 093
#
# Sections 1-6 need only curl. Section 7 needs pmedm_vb importable; set
# PYTHON=... to override the interpreter, which on a cluster should be
# $CONDA_PREFIX/bin/python rather than whatever PATH resolves.

set -uo pipefail

YEAR="${1:-2024}"
SPAN="${2:-5}"
STATE="${3:-47}"
COUNTY="${4:-093}"
PYTHON="${PYTHON:-${CONDA_PREFIX:-}/bin/python}"

VRE="https://www2.census.gov/programs-surveys/acs/replicate_estimates"
PUMS="https://www2.census.gov/programs-surveys/acs/data/pums"
REL="https://www2.census.gov/geo/docs/maps-data/data/rel2020"
START=$((YEAR - SPAN + 1))

failures=0
note() { printf '\n\033[1m%s\033[0m\n' "$*"; }
pass() { printf '  ok    %s\n' "$*"; }
fail() { printf '  FAIL  %s\n' "$*"; failures=$((failures + 1)); }

# Directory listing with the census.gov page furniture stripped. The relative
# entries are what matter; everything absolute is nav, CSS or social links.
list() {
    curl -sS --max-time 90 "$1" \
        | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"$//' \
        | grep -vE '^(https?:|/|\?C=)' | sort -u
}

# One-byte range request: works for a multi-hundred-megabyte PUMS zip, and
# does not assume the server honours HEAD. 206 is the normal answer, 200 means
# the range was ignored and the body started coming -- both mean it is there.
exists() { curl -sS -o /dev/null --max-time 60 -r 0-0 -w '%{http_code}' "$1"; }
found() { [ "$1" = "200" ] || [ "$1" = "206" ]; }

note "0. Reachability"
if found "$(exists "$VRE/")"; then
    pass "census.gov reachable"
else
    fail "census.gov unreachable -- egress blocked, or the site is down"
    echo "nothing below is meaningful; stopping." >&2
    exit 1
fi

note "1. Vintage is published"
for label in "replicate tables:$VRE/$YEAR/" "PUMS:$PUMS/$YEAR/"; do
    code=$(exists "${label#*:}")
    found "$code" && pass "${label%%:*} ($code)" || fail "${label%%:*} ($code)"
done

note "2. Replicate summary levels"
levels=$(list "$VRE/$YEAR/data/$SPAN-year/")
echo "$levels" | tr '\n' ' ' | sed 's/^/  /;s/$/\n/'
for level in 140/ 150/; do
    echo "$levels" | grep -qx "$level" \
        && pass "$level present" \
        || fail "$level absent -- 140 is tract, 150 is block group; both are needed"
done

note "3. Documentation files the code constructs by name"
docs=$(list "$VRE/$YEAR/documentation/$SPAN-year/")
echo "$docs" | sed 's/^/  /'
for f in "VRE_TABLE_LIST_$YEAR.csv" "VRE_AVERAGE_WEIGHT_$YEAR.csv"; do
    echo "$docs" | grep -qx "$f" && pass "$f" || fail "$f missing"
done
# The k-value table is read by eye, not by the code, so its name only has to be
# findable -- it has varied between spaces and underscores across vintages.
echo "$docs" | grep -qi "appendix_a\|appendix%20a" \
    && pass "Appendix A (k-values) present" \
    || fail "Appendix A not found -- k-value brackets cannot be confirmed"

note "4. Table coverage"
for level in 140 150; do
    n=$(list "$VRE/$YEAR/data/$SPAN-year/$level/" \
        | grep -oE '^[BC][0-9]{5}[A-Z]*' | sort -u | wc -l)
    printf '  summary level %s: %s tables\n' "$level" "$n"
    [ "$n" -gt 0 ] && pass "level $level populated" || fail "level $level empty"
done
master=$(curl -sS --max-time 60 "$VRE/$YEAR/documentation/$SPAN-year/VRE_TABLE_LIST_$YEAR.csv" \
         | tail -n +2 | wc -l)
printf '  master list: %s tables\n' "$master"
echo "  (2019-2023 for reference: 132 master, 131 at tract, 73 at block group)"
echo "  NB: match [BC], not B alone -- a B-only pattern hides C02003, C15010,"
echo "      C17002, C24010 and C24030, all published at block group."

note "5. PUMS files for state $STATE"
abbrev=$("$PYTHON" -c "
import sys
sys.path.insert(0, 'src')
from pmedm_vb.config import STATE_ABBREV
print(STATE_ABBREV['$STATE'].lower())
" 2>/dev/null) || abbrev=""
if [ -z "$abbrev" ]; then
    fail "could not resolve state abbreviation (is pmedm_vb importable?)"
else
    for kind in p h; do
        url="$PUMS/$YEAR/$SPAN-Year/csv_${kind}${abbrev}.zip"
        code=$(exists "$url")
        found "$code" && pass "csv_${kind}${abbrev}.zip" \
                      || fail "csv_${kind}${abbrev}.zip ($code)"
    done
    echo "  NB: '$SPAN-Year' capitalised here, '$SPAN-year' lowercase in the replicate tree."
fi

note "6. PUMA crosswalk and PUMS PUMA coding"
header=$(curl -sS --max-time 60 "$REL/2020_Census_Tract_to_2020_PUMA.txt" | head -1)
printf '  %s\n' "$header"
[ "$header" = "STATEFP,COUNTYFP,TRACTCE,PUMA5CE" ] \
    && pass "crosswalk columns unchanged" \
    || fail "crosswalk columns changed -- geography.py assumes these four"
puma=$(curl -sS --max-time 90 \
       "https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_${START}-${YEAR}.csv" \
       | grep -i '^NAME,PUMA' | head -2)
printf '  %s\n' "$puma"
if [ "$(printf '%s' "$puma" | grep -c .)" -eq 0 ]; then
    fail "no PUMA entry found in the ${START}-${YEAR} data dictionary"
elif printf '%s' "$puma" | grep -qi 'PUMA10\|PUMA20'; then
    fail "split PUMA10/PUMA20 coding -- this vintage spans the redraw and needs
        the matching earlier crosswalk as well as rel2020"
else
    pass "single PUMA column; rel2020 alone is sufficient"
fi

note "7. End-to-end accuracy check"
if [ ! -x "$PYTHON" ]; then
    fail "interpreter not found at '$PYTHON'; set PYTHON=..."
else
    "$PYTHON" - "$YEAR" "$SPAN" "$STATE" "$COUNTY" <<'PY'
import sys
import numpy as np
from pmedm_vb.config import StudyArea
from pmedm_vb.data import variance as v, geography

year, span, state, county = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
area = StudyArea(name="verification", state=state, year=year,
                 counties=(county,), span=span)

# Crosswalk: exercises the relationship file, GEOID slicing, and the block
# group universe taken from the B01003 replicate file.
zones = geography.puma_crosswalk(area)
print(f"  block groups: {len(zones)}  PUMAs: {zones['puma_geoid'].nunique()}")
assert zones["puma_geoid"].notna().all()
assert zones["block_group_geoid"].str.len().eq(12).all()
assert zones["tract_geoid"].str.len().eq(11).all()
print("  ok    crosswalk resolves, GEOID widths correct")

# Accuracy: the files publish their own MOE, so the SDR implementation can be
# checked against Census's answer. Published MOEs are integers and ours are
# continuous, so agreement means |difference| <= 0.5 -- the signature of
# published == round(ours). A transposed or misread replicate column breaks
# that bound immediately.
frame = v.fetch_replicates(area, ["B25003"], geography="block group")
var = v.variances(frame, area=area, geography="block group", policy="model")

reps = frame[list(v.REPLICATE_COLUMNS)].to_numpy(float)
zero = (reps == frame["estimate"].to_numpy(float)[:, None]).all(axis=1)
diff = (v.Z_90 * np.sqrt(var) - frame["moe"])[~zero].abs()

print(f"  cells: {len(frame)}  zero-variance: {int(zero.sum())}")
print(f"  recomputed vs published MOE -- max {diff.max():.3f}, median {diff.median():.3f}")
print(f"  average weight: {v.average_weight(area)}  "
      f"modelled zero-cell variance: {sorted(set(var[zero].round(1)))[:3]}")

# Internal consistency: B25003 is total, owner, renter -- the parts must sum to
# the total, which only holds if cell ordering and GEOID parsing are both right.
wide = frame["estimate"].unstack("cell")
if {"B25003_001", "B25003_002", "B25003_003"} <= set(wide.columns):
    assert (wide["B25003_002"] + wide["B25003_003"] == wide["B25003_001"]).all()
    print("  ok    tenure parts sum to the table total")

if diff.max() > 0.51:
    print(f"  FAIL  MOE disagreement of {diff.max():.3f} exceeds rounding")
    sys.exit(1)
print("  ok    MOEs agree to rounding")
PY
    [ $? -eq 0 ] && pass "vintage $YEAR verified end to end" \
                 || fail "end-to-end check failed"
fi

note "Result"
if [ "$failures" -eq 0 ]; then
    echo "  vintage $YEAR ($START-$YEAR, $SPAN-year) is usable: StudyArea(..., year=$YEAR, span=$SPAN)"
    exit 0
fi
echo "  $failures check(s) failed -- see above before adopting this vintage"
exit 1
