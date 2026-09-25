"""
run_01_build_panel.py
Reads every CPHS wave file under RAW_DIR, aggregates to district x month,
writes the panel plus two diagnostic tables.

Run once. Takes ~10-25 min for 144 files depending on your disk.

    python run_01_build_panel.py
"""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

import argparse

import config as C
from panel import build_panel, deflate, smooth, screen_units, detect_level_breaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-nominal", action="store_true",
                    help="build the panel in NOMINAL rupees, with no CPI deflator")
    args = ap.parse_args()

    # Running nominal has to be a decision, not an oversight. Cumulative
    # inflation over 2014-2025 is 70-80%, and state-specific inflation
    # differences leak straight into the estimate; SCM only differences out the
    # COMMON component. Refusing by default means a nominal headline can never
    # be produced by forgetting to set CPI_CSV.
    if C.CPI_CSV is None and not args.allow_nominal:
        sys.exit(
            "\nREFUSING TO BUILD: CPI_CSV is None, so this would run in nominal\n"
            "rupees and the headline estimate would not be trustworthy.\n\n"
            "  Fix it   : set CPI_CSV in config.py (see fetch_cpi.py)\n"
            "  Override : python run_01_build_panel.py --allow-nominal\n")

    for d in (C.OUT_DIR, C.FIG_DIR, C.RESULT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    panel, audit = build_panel(
        raw_dir=C.RAW_DIR,
        file_glob=C.FILE_GLOB,
        usecols=C.USECOLS,
        income_sources=C.INCOME_SOURCES,
        weight_col=C.WEIGHT_COL,
        winsor=(C.WINSOR_LO, C.WINSOR_HI),
    )

    panel = deflate(panel, C.CPI_CSV)
    panel = smooth(panel, col="ln_inc_real", window=C.SMOOTH_WINDOW)

    # Flag the COVID telephone-interview window as a measurement break.
    panel["covid_mode"] = (
        (panel["MONTH_DT"] >= pd.Timestamp(C.COVID_START))
        & (panel["MONTH_DT"] <= pd.Timestamp(C.COVID_END))
    )

    panel.to_parquet(C.PANEL_PARQUET, index=False)
    audit.to_csv(C.AUDIT_CSV, index=False)

    screen = screen_units(panel, C.MIN_HH_PER_CELL, C.MIN_COVERAGE_FRAC)
    screen.to_csv(C.SCREEN_CSV, index=False)

    # ------------------------------------------------------------ report --
    print("\n" + "=" * 66)
    print(f"Panel      : {len(panel):,} unit-months")
    print(f"Units      : {panel['unit'].nunique():,}")
    print(f"Months     : {panel['MONTH_DT'].nunique()} "
          f"({panel['MONTH_DT'].min():%b %Y} to {panel['MONTH_DT'].max():%b %Y})")
    deflated = bool(panel["_deflated"].iloc[0])
    print(f"Deflated   : {deflated}")
    if not deflated:
        print("  *** NOMINAL RUPEES -- headline effect is NOT inflation-adjusted ***")
    print(f"Eligible donors (n_hh >= {C.MIN_HH_PER_CELL}, full coverage): "
          f"{int(screen['eligible'].sum()):,}")

    print("\nNon-response rate by year (watch 2020 -- the telephone switch):")
    a = audit.copy()
    a["year"] = pd.to_datetime(a["month"]).dt.year
    print(a.groupby("year")[["nonresponse_rate", "n_districts"]]
           .mean().round(3).to_string())

    for name, spec in C.TREATMENTS.items():
        if spec["unit"] not in set(panel["unit"]):
            print(f"\n  WARNING: {name} ({spec['unit']}) is not in the panel. "
                  f"If CPI_CSV is set, it may have been dropped for incomplete "
                  f"CPI coverage.")

    # ---- structural-break screen ----------------------------------------
    brk = detect_level_breaks(panel, col="ln_inc_real")
    if len(brk):
        brk.to_csv(C.OUT_DIR / "level_breaks.csv", index=False)
        print("\nLargest level shifts (possible CPHS sample replacement):")
        for _, r in brk.head(8).iterrows():
            flag = "  <-- SUSPECT" if r["suspect_break"] else ""
            print(f"  {r['max_step']:+.3f} log pts at {r['step_month']:%b %Y}  "
                  f"({r['step_ratio']:5.1f}x typical monthly move)  "
                  f"{r['unit']}{flag}")
        tre = brk[brk["unit"].isin({s["unit"] for s in C.TREATMENTS.values()})]
        for _, r in tre.iterrows():
            if r["suspect_break"]:
                print(f"\n  *** TREATED UNIT {r['unit']} has a suspect level "
                      f"break at {r['step_month']:%b %Y}.\n"
                      f"      Any effect estimated across that date may be a "
                      f"sample change, not an effect. ***")

    print("\nTreated units:")
    for name, spec in C.TREATMENTS.items():
        row = screen[screen["unit"] == spec["unit"]]
        if len(row):
            r = row.iloc[0]
            ok = "OK " if r["eligible"] else "FAIL"
            print(f"  [{ok}] {name:9s} coverage={r['coverage']:.2f} "
                  f"min_n={int(r['min_n_hh']):4d} "
                  f"median_n={int(r['median_n_hh']):4d} "
                  f"median_SE={r['median_se']:.4f}")
        else:
            print(f"  [MISS] {name:9s} unit not found: {spec['unit']!r}")
            near = screen[screen["district"].str.contains(
                name.split()[0], case=False, na=False)]
            if len(near):
                print("         did you mean:",
                      ", ".join(near["unit"].head(3)))

    print("=" * 66)
    print(f"\nWrote {C.PANEL_PARQUET}")
    print(f"      {C.AUDIT_CSV}")
    print(f"      {C.SCREEN_CSV}")


if __name__ == "__main__":
    main()
