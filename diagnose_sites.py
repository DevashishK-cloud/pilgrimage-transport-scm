"""
2026-09-05-diagnose_sites.py -- find out WHY a site failed the screener.

    python 2026-09-05-diagnose_sites.py 2026-09-05-treatments_transport.csv

screen_treatments.py answers "does this site pass". It does not answer "why
not", and its three failure modes have completely different remedies:

  NOT IN CPHS   might mean the district was never surveyed, OR that CPHS
                spells it differently (Gulbarga vs Kalaburagi), OR that it
                exists only as RURAL and the screener looks at URBAN only.

  coverage<1.00 is measured over the WHOLE panel, Jan 2014 to Dec 2025. A
                unit missing five months in 2014 fails, even for a treatment
                dated 2017 that never needed 2014. The bar belongs on the
                window the site actually uses, not on all 142 months.

  break in post is real and is not negotiable.

So for every site this prints: every unit in that state whose name is close,
in BOTH sectors; the exact months missing; and what coverage would be if the
analysis started later. Then you can tell a dead site from a mislabelled one.

Nothing is estimated and nothing is written. This only reports.
"""

from __future__ import annotations

import sys
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from panel import canonical_district, canonical_state

# Candidate analysis start dates. A site only needs full coverage from the
# point the panel starts, and pre-2016 months are worth little to a treatment
# dated 2020 anyway.
START_DATES = ["2014-01-01", "2015-01-01", "2016-01-01", "2017-01-01"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tre = pd.read_csv(sys.argv[1])
    panel = pd.read_parquet(C.PANEL_PARQUET)

    all_months = sorted(panel["MONTH_DT"].unique())
    print(f"panel: {len(all_months)} months, "
          f"{pd.Timestamp(all_months[0]):%b %Y} to "
          f"{pd.Timestamp(all_months[-1]):%b %Y}, "
          f"{panel['unit'].nunique():,} units "
          f"({panel[panel.REGION_TYPE == 'URBAN']['unit'].nunique():,} urban)\n")

    panel["_st"] = panel["STATE"].map(canonical_state)
    panel["_di"] = panel["DISTRICT"].map(canonical_district)

    for _, t in tre.iterrows():
        st = canonical_state(t["state"])
        di = canonical_district(t["district"])
        print("=" * 78)
        print(f"{t['site_name']}   [{t.get('group','')}]   target: {st} | {di}")

        instate = panel[panel["_st"] == st]
        if instate.empty:
            print(f"  !! no units at all in state {st!r}. "
                  f"Either the state name differs or the deflator dropped it.")
            print(f"     states present: {sorted(panel['_st'].unique())[:8]} ...")
            continue

        names = sorted(instate["_di"].unique())
        near = get_close_matches(di, names, n=6, cutoff=0.55)
        if di in names and near and near[0] != di:
            near = [di] + [n for n in near if n != di]
        if not near:
            print(f"  no district in {st} within 0.55 similarity of {di!r}.")
            print(f"  districts in {st}: {', '.join(names)}")
            continue

        print(f"  {'district':<22} {'sector':<7} {'cov all':>8}   "
              + "  ".join(f"cov>={s[:7]}" for s in START_DATES[1:]))
        for name in near:
            for sec in ("URBAN", "RURAL"):
                d = instate[(instate["_di"] == name) & (instate["REGION_TYPE"] == sec)]
                if d.empty:
                    continue
                have = set(d["MONTH_DT"])
                row = [f"  {name:<22} {sec:<7}",
                       f"{len(have) / len(all_months):>8.3f}"]
                for s in START_DATES[1:]:
                    w = [m for m in all_months if m >= pd.Timestamp(s)]
                    if not w:                       # start date past panel end
                        row.append(f"{'-':>11}")
                    else:
                        row.append(f"{len(have & set(w)) / len(w):>11.3f}")
                print("".join(row))

                miss = [m for m in all_months if m not in have]
                if miss and len(miss) <= 24:
                    print("      missing: "
                          + ", ".join(f"{pd.Timestamp(m):%b %Y}" for m in miss))
                elif miss:
                    print(f"      missing {len(miss)} months, first "
                          f"{pd.Timestamp(miss[0]):%b %Y}, last "
                          f"{pd.Timestamp(miss[-1]):%b %Y}")
        print()

    print("=" * 78)
    print("How to read this:")
    print("  cov all 1.000                      -> passes today")
    print("  cov all < 1 but cov>=2016 == 1.000 -> passes if the panel starts")
    print("                                        in 2016; the holes are old")
    print("  a RURAL row at 1.000, URBAN absent -> the site is there, the")
    print("                                        screener only looked urban")
    print("  no near name at all                -> genuinely not surveyed")


if __name__ == "__main__":
    main()
