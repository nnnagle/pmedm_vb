"""Pool the leave-one-PUMA-out results under one output root.

    python experiments/embed/collect_puma_check.py --root $PMEDM_VB_RUNS/embed/puma_check

Reads every ``<root>/<label>/<puma>/{scores,misfit,rare}.csv`` and writes, under
``--root``, ``summary_overall.csv`` (means over PUMAs by source, set and tilt:
TVD, KL, log-odds-ratio errors, constraint misfit), ``summary_tables.csv`` (the
same per scoring table) and ``summary_rare.csv`` (rare groups summed over
PUMAs against truth and, where published PUMA tables were available, the z of
the pooled prediction and the RMS of per-PUMA z). Prints the overall table,
then, for one constraint set (``--set``, default S1), every tilt side by side:
TVD and log-odds-ratio error overall and per scoring table, and the rare
groups' pooled prediction over truth. Tilts of different feature maps h carry
their suffix (``phi_quad``, ``oracle_dec``...), so all h line up as columns.

Then the penalty comparison for the same set: per feature map h, the oracle
beside the fitted tilts at every penalty setting (α, τ) present -- overall
TVD, log-odds-ratio error (mean absolute and precision-weighted RMS),
constraint misfit (RMS z, share within 1 SE), KL(p ‖ q) from the support's
weights, the log-odds-ratio error of the race crosses, and the rare groups'
pooled prediction over truth. ``none`` and the ``x`` tilts head the table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pmedm_vb.embed.puma_check import (
    FEATURE_MAPS, H_TILTS, TILTS, parse_tilt_label, summarise,
)

#: Scoring tables whose log-odds-ratio error the penalty comparison shows.
RACE_CROSSES = ("hh_race8xB19001", "hh_race8xB25003", "B17010A-G")

#: Rare groups shown in the penalty comparison: (table, category).
RARE_SHOWN = (("B03002", "nh_aian"), ("B03002", "nh_asian"), ("B03002", "nh_nhpi"),
              ("B03002", "nh_other_race"), ("B17010C", "below_poverty"),
              ("B17010E", "below_poverty"))


def penalty_tables(summary: dict, set_name: str) -> list[tuple[str, pd.DataFrame]]:
    """Per (source, h): ``none``, every ``x``, the oracle and the fitted
    ``phi``/``both`` tilts at every (α, τ), with fit and score columns."""
    overall = summary["overall"]
    overall = overall[overall["set"] == set_name].copy()
    if overall.empty:
        return []
    parsed = overall["tilt"].map(parse_tilt_label).apply(pd.Series)
    overall = pd.concat([overall, parsed.rename(columns={"tilt": "base"})], axis=1)
    keys = ["source", "tilt"] if "source" in overall else ["tilt"]
    tables = summary["tables"]
    tables = tables[(tables["set"] == set_name) & tables["table"].isin(RACE_CROSSES)]
    if not tables.empty and "lor_mae" in tables:
        race = tables.pivot_table(index=keys, columns="table", values="lor_mae")
        race.columns = [f"lor:{c}" for c in race.columns]
        overall = overall.merge(race.reset_index(), on=keys, how="left")
    rare = summary["rare"]
    rare = rare[rare["set"] == set_name]
    shown = pd.Series(list(zip(rare["table"], rare["category"])), index=rare.index).isin(RARE_SHOWN)
    rare = rare[shown]
    if not rare.empty:
        where = [("" if t == "B03002" else f"@{t}") for t in rare["table"]]
        wide = rare.assign(col=["p/t:" + c + w for c, w in zip(rare["category"], where)]) \
            .pivot_table(index=keys, columns="col", values="pred_over_truth")
        overall = overall.merge(wide.reset_index(), on=keys, how="left")
    columns = [c for c in ("tvd", "lor_mae", "lor_wrms", "rms_z", "share_within_1se",
                           "kl_to_base") if c in overall]
    columns += sorted(c for c in overall if c.startswith("lor:"))
    columns += sorted(c for c in overall if c.startswith("p/t:"))
    out = []
    sources = overall["source"].unique() if "source" in overall else [None]
    for source in sources:
        frame = overall if source is None else overall[overall["source"] == source]
        head = frame[frame["base"].isin(["none", "x"])].sort_values(["base", "alpha", "tau"])
        for h in FEATURE_MAPS:
            body = frame[(frame["h"] == h) & frame["base"].isin(["oracle", "phi", "both"])]
            if body.empty:
                continue
            order = {"oracle": 0, "phi": 1, "both": 2}
            body = body.assign(_o=body["base"].map(order)).sort_values(["_o", "alpha", "tau"])
            table = pd.concat([head, body])
            table = table.set_index(["base", "alpha", "tau"])[columns]
            out.append((f"{source or ''} h={h}".strip(), table.round(3)))
    return out
from pmedm_vb.progress import write_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--set", default="S1", help="constraint set for the side-by-side tables")
    args = parser.parse_args()

    frames = {}
    for name in ("scores", "misfit", "rare"):
        paths = sorted(args.root.glob(f"*/*/{name}.csv"))
        if not paths:
            raise SystemExit(f"no {name}.csv under {args.root}")
        frames[name] = pd.concat([pd.read_csv(p, dtype={"puma": str}) for p in paths],
                                 ignore_index=True)
    summary = summarise(frames["scores"], frames["misfit"], frames["rare"])
    for name, frame in summary.items():
        write_text(args.root / f"summary_{name}.csv", frame.to_csv(index=False))
    n = frames["misfit"].groupby("source")["puma"].nunique()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary["overall"].round(4).to_string(index=False))
    order = [*(t for t in TILTS if t not in H_TILTS),
             *(t + suffix for suffix in FEATURE_MAPS.values() for t in H_TILTS)]

    def side_by_side(frame, index, value):
        frame = frame[frame["set"] == args.set]
        if frame.empty or value not in frame:
            return None
        wide = frame.pivot_table(index=index, columns="tilt", values=value)
        return wide[[t for t in order if t in wide.columns]].round(3)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for label, frame, index in (("overall", summary["overall"], "source"),
                                    ("per table", summary["tables"], ["source", "table"])):
            for value in ("tvd", "lor_mae"):
                wide = side_by_side(frame, index, value)
                if wide is not None:
                    print(f"\n{args.set} {label}: {value}\n{wide.to_string()}")
        rare = summary["rare"]
        wide = side_by_side(rare, ["source", "table", "category"], "pred_over_truth")
        if wide is not None:
            print(f"\n{args.set} rare groups: pooled prediction / truth\n{wide.to_string()}")
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        for title, table in penalty_tables(summary, args.set):
            print(f"\n{args.set} penalty comparison, {title} "
                  f"(alpha = entropy weight, tau = prior sd; NaN for none/oracle)\n"
                  f"{table.to_string()}")
    print(f"\nPUMAs per source: {n.to_dict()}; tables in {args.root}/summary_*.csv")


if __name__ == "__main__":
    main()
