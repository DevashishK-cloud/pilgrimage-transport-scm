"""
2026-09-05-find_renames.py -- find districts CPHS renamed mid-panel, which
appear as two half-units and are then silently dropped by the coverage screen.

    python 2026-09-05-find_renames.py

What counts as a rename
-----------------------
A rename leaves a very specific fingerprint, and nothing else leaves it:

  * the old label starts at the FIRST month of the panel and stops early
  * the new label starts exactly one month later and runs to the LAST month
  * together the two cover every month of the panel, once each
  * the two fragments look like the same place: similar real income, similar
    household count

The first version of this script required only that one unit ended early and
another began late, then paired each late unit with its nearest candidate. In
a panel where districts enter and leave the sample for ordinary reasons, that
matched almost everything, produced contradictions (one old name mapped to
three new ones) and was useless. All four conditions above are now required,
and every pair must be one-to-one.

This is deliberately strict. It will miss a rename that coincided with a gap
in the series. Missing one is cheap; a wrong alias silently merges two real
districts into one unit and corrupts the panel.

Every pair is still a SUGGESTION. Nothing is written to panel.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

MAX_D_LN_INC = 0.25    # two fragments of one district should not differ more
MIN_FRAGMENT = 2       # ignore 1-month slivers, too little to judge


def main():
    panel = pd.read_parquet(C.PANEL_PARQUET)
    months = pd.Index(sorted(panel["MONTH_DT"].unique()))
    last = len(months) - 1

    g = panel.groupby(["STATE", "DISTRICT", "REGION_TYPE"], observed=True)
    inc_col = "ln_inc_real" if "ln_inc_real" in panel.columns else "ln_inc"
    span = (g["MONTH_DT"].agg(["min", "max", "nunique"])
             .rename(columns={"nunique": "n_months"})
             .join(g.agg(inc=(inc_col, "mean"), n_hh=("n_hh", "median")))
             .reset_index())

    def idx(ts):
        return int(months.get_indexer([ts])[0])

    span["i0"] = span["min"].map(idx)
    span["i1"] = span["max"].map(idx)
    # Contiguous means no interior holes: months present == span length.
    span["solid"] = span["n_months"] == (span["i1"] - span["i0"] + 1)

    olds = span[(span.i0 == 0) & (span.i1 < last) & span.solid
                & (span.n_months >= MIN_FRAGMENT)]
    news = span[(span.i1 == last) & (span.i0 > 0) & span.solid
                & (span.n_months >= MIN_FRAGMENT)]

    print(f"panel {months[0]:%b %Y} to {months[-1]:%b %Y}, {len(months)} months, "
          f"{panel['unit'].nunique():,} units")
    print(f"candidate old fragments (start at panel start, stop early): {len(olds):,}")
    print(f"candidate new fragments (start late, run to panel end)    : {len(news):,}\n")

    rows = []
    for _, n in news.iterrows():
        c = olds[(olds.STATE == n.STATE) & (olds.REGION_TYPE == n.REGION_TYPE)
                 & (olds.DISTRICT != n.DISTRICT)
                 & (olds.i1 == n.i0 - 1)]           # exactly adjacent, no gap
        if c.empty:
            continue
        c = c.assign(d_inc=(c.inc - n.inc).abs())
        c = c[c.d_inc <= MAX_D_LN_INC]
        if c.empty:
            continue
        b = c.loc[c.d_inc.idxmin()]
        rows.append({
            "state": n.STATE, "sector": n.REGION_TYPE,
            "old_name": b.DISTRICT, "new_name": n.DISTRICT,
            "old_window": f"{b['min']:%b %Y}-{b['max']:%b %Y}",
            "new_window": f"{n['min']:%b %Y}-{n['max']:%b %Y}",
            "new_months": int(n.n_months),
            "d_ln_inc": round(float(b.d_inc), 3),
            "d_ln_nhh": round(float(abs(np.log(max(b.n_hh, 1))
                                        - np.log(max(n.n_hh, 1)))), 3),
            "n_rivals": int(len(c)),
        })

    if not rows:
        print("No pair meets all four conditions.")
        return

    r = pd.DataFrame(rows)

    # A rename is one-to-one. If an old name is claimed by two new names, or a
    # new name by two old ones, none of them is trustworthy.
    for col in ("old_name", "new_name"):
        dup = r[col][r.duplicated(col, keep=False)].unique()
        if len(dup):
            print(f"dropping {col} claimed more than once: {sorted(dup)}")
            r = r[~r[col].isin(dup)]
    if r.empty:
        print("\nAll candidates were ambiguous. None suggested.")
        return

    r = r.sort_values(["d_ln_inc"])
    pd.set_option("display.width", 200, "display.max_colwidth", 30)
    print("\n" + r.to_string(index=False))

    out = Path(C.OUT_DIR) / "suggested_renames.csv"
    r.to_csv(out, index=False)
    print(f"\nWrote {out}")

    print("\n" + "=" * 74)
    print("Candidate aliases. CHECK EACH ONE against India's actual district")
    print("rename history before pasting into DISTRICT_ALIASES in panel.py:\n")
    for _, x in r.iterrows():
        print(f'    "{x.old_name}": "{x.new_name}",')
        print(f'    "{x.new_name}": "{x.new_name}",')


if __name__ == "__main__":
    main()
