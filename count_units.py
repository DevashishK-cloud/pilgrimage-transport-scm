"""
2026-09-05-count_units.py -- count what the panel actually covers.

    python 2026-09-05-count_units.py --fast     # ~5 seconds, uses the built panel
    python 2026-09-05-count_units.py            # ~5 min, rescans the 144 raw waves

Why this exists
---------------
`panel_district_month.parquet` reports 844 units. That is the number that
SURVIVES screening: cells are dropped for thin household counts
(MIN_HH_PER_CELL) and incomplete month coverage (MIN_COVERAGE_FRAC).

It is not the number of sampling units the panel was ENGINEERED FROM. CPHS
draws each district-month from primary sampling units nested inside strata
inside homogeneous regions, and `process_wave_file` already reads PSU_ID, HR
and STRATUM. This script counts every level so the README can state, in one
line, exactly what "unit" means and how many there are.

It applies the same filters the panel does -- RESPONSE_STATUS == "Accepted",
TOT_INC > 0, canonical_district() -- so the counts describe the same object
the pipeline built, not the raw file dump.

Nothing here is an estimate. Every number printed is a distinct-tuple count.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from panel import canonical_district

# Only what is needed to identify a sampling unit, plus the two filter columns.
COLS = ["STATE", "DISTRICT", "REGION_TYPE", "HR", "STRATUM", "PSU_ID",
        "MONTH", "RESPONSE_STATUS", "TOT_INC"]


def fast_from_panel():
    """Bound the PSU count from the built panel, in about five seconds.

    The panel stores n_psu per district-month. PSUs recur across months, so
    summing is wrong; taking each unit's maximum and summing those gives a
    LOWER BOUND on distinct PSUs (it would undercount only if a unit's PSU
    set changed identity without ever growing).
    """
    p = Path(C.PANEL_PARQUET)
    if not p.exists():
        sys.exit(f"No panel at {p}. Run run_01_build_panel.py, or drop --fast.")
    d = pd.read_parquet(p)

    print(f"panel: {p}")
    print(f"  months                        : {d['MONTH_DT'].nunique()}")
    print(f"  unit-months                   : {len(d):,}")
    print(f"  units after screening         : {d['unit'].nunique():,}")
    if "REGION_TYPE" in d.columns:
        u = d[d["REGION_TYPE"] == "URBAN"]["unit"].nunique()
        print(f"    of which urban              : {u:,}")
    if "hr" in d.columns:
        print(f"  homogeneous regions           : {d['hr'].nunique():,}")
    if "n_psu" in d.columns:
        lb = int(d.groupby("unit")["n_psu"].max().sum())
        print(f"  PSUs, lower bound             : {lb:,}")
        print("\n  (lower bound only. Run without --fast for the exact count.)")
        return lb
    return None


def scan_raw():
    """Exact distinct-tuple counts, rescanning all 144 waves."""
    files = sorted(Path(C.RAW_DIR).glob(C.FILE_GLOB))
    if not files:
        sys.exit(f"No waves matched {C.FILE_GLOB} under {C.RAW_DIR}")
    print(f"scanning {len(files)} waves under {C.RAW_DIR}\n")

    psu, dist_sec, dist_sec_str, hr_sec, months = set(), set(), set(), set(), set()
    no_stratum = 0
    t0 = time.time()

    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=lambda c: c in COLS, low_memory=False)

        for c in ("RESPONSE_STATUS", "TOT_INC", "STATE", "DISTRICT",
                  "REGION_TYPE", "PSU_ID"):
            if c not in df.columns:
                sys.exit(f"{f.name}: missing required column {c}")

        # Same two filters the panel applies, in the same order.
        df = df[df["RESPONSE_STATUS"] == "Accepted"]
        df = df[df["TOT_INC"] > 0]
        if df.empty:
            continue

        df["DISTRICT"] = df["DISTRICT"].map(canonical_district)
        st = df["STATE"].astype(str)
        di = df["DISTRICT"].astype(str)
        rt = df["REGION_TYPE"].astype(str)

        psu.update(zip(st, di, rt, df["PSU_ID"].astype(str)))
        dist_sec.update(zip(st, di, rt))

        if "STRATUM" in df.columns:
            dist_sec_str.update(zip(st, di, rt, df["STRATUM"].astype(str)))
        else:
            no_stratum += 1
        if "HR" in df.columns:
            hr_sec.update(zip(st, df["HR"].astype(str), rt))
        if "MONTH" in df.columns:
            months.update(df["MONTH"].astype(str).unique())

        if i % 12 == 0 or i == len(files):
            print(f"  {i:>3}/{len(files)} waves  ({time.time() - t0:.0f}s)")

    print()
    if no_stratum:
        print(f"note: STRATUM absent from {no_stratum} wave(s); "
              f"that count excludes them\n")

    rows = [
        ("districts x sector (all waves, pre-screening)", len(dist_sec)),
        ("homogeneous regions x sector", len(hr_sec)),
        ("districts x sector x stratum", len(dist_sec_str)),
        ("primary sampling units (PSUs)", len(psu)),
        ("distinct months", len(months)),
    ]
    w = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f"  {k:<{w}} : {v:,}")

    out = Path(C.OUT_DIR) / "unit_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["level", "distinct"]).to_csv(out, index=False)
    print(f"\nWrote {out}")

    print("\n" + "=" * 70)
    n = len(psu)
    print(f"Largest defensible unit count: {n:,} primary sampling units.")
    if n >= 1400:
        print("This clears 1,400. Put the definition in the README so the")
        print("number is checkable, for example:")
        print()
        print(f"  The panel is built from {n:,} CPHS primary sampling units")
        print(f"  across {len(dist_sec):,} district-sector cells and "
              f"{len(months)} monthly waves.")
        print(f"  {'844'} district-sector units clear the screening thresholds")
        print("  and enter estimation.")
    else:
        print("This does NOT clear 1,400. Do not write a number you cannot")
        print("reproduce. State the counts above and leave it there.")
    return len(psu)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="use the built panel for a lower bound, no raw scan")
    a = ap.parse_args()
    scan_raw() if not a.fast else fast_from_panel()


if __name__ == "__main__":
    main()
