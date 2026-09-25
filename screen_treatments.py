"""
screen_treatments.py -- decide which candidate sites can actually be
estimated, before any estimation is attempted.

    python screen_treatments.py treatments.csv

Reads a treatment list, matches each site's district to the CPHS panel, and
reports for every site whether it clears the bars that earlier work on this
panel established the hard way:

  * the district exists in CPHS at all           (half of the pilgrimage
                                                  districts checked did not)
  * coverage == 1.00                             (a hole in the series and the
                                                  estimation matrix drops the
                                                  unit -- this is what killed
                                                  several candidate sites)
  * enough clean months BEFORE the sanction date (dating from completion puts
                                                  the construction period
                                                  inside the pre-window and
                                                  manufactures an effect)
  * enough months AFTER completion
  * no suspect level break in the post window    (a CPHS sample refresh once
                                                  produced a "+54.7%, p=0.028"
                                                  out of nothing)

Run this FIRST. If it says no, nothing downstream is worth doing.
"""

from __future__ import annotations

import sys
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from panel import canonical_district, canonical_state, detect_level_breaks

# Thresholds, fixed in advance. See the scoping document, section 9.
MIN_PRE_MONTHS = 24
MIN_POST_MONTHS = 24
MIN_HH = 60


def load_panel():
    if not Path(C.PANEL_PARQUET).exists():
        sys.exit(f"No panel at {C.PANEL_PARQUET}. Run run_01_build_panel.py first.")
    return pd.read_parquet(C.PANEL_PARQUET)


def match_unit(state, district, units_by_key, all_units):
    """
    Map a Ministry site's state+district onto a CPHS unit label.

    Ministry documents, Census and CPHS disagree constantly on district names,
    and India split and renamed districts throughout the period. Exact match
    first, then the project's own alias table, then fuzzy -- and anything that
    only matches fuzzily is reported for a human to confirm rather than
    silently accepted.
    """
    st, di = canonical_state(state), canonical_district(district)
    key = f"{st}|{di}"
    if key in units_by_key:
        return units_by_key[key], "exact"
    cand = [u for u in all_units if u.split("|")[0].strip() == st]
    names = [u.split("|")[1].strip() for u in cand]
    near = get_close_matches(di, names, n=1, cutoff=0.82)
    if near:
        return next(u for u in cand if u.split("|")[1].strip() == near[0]), f"fuzzy->{near[0]}"
    return None, "no match"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tre = pd.read_csv(sys.argv[1])
    need = {"site_name", "state", "district", "tradition",
            "sanction_date", "completion_date"}
    missing = need - set(tre.columns)
    if missing:
        sys.exit(f"treatment file missing column(s): {sorted(missing)}")

    panel = load_panel()
    urban = panel[panel["REGION_TYPE"] == "URBAN"]
    all_units = sorted(urban["unit"].unique())
    units_by_key = {f"{u.split('|')[0].strip()}|{u.split('|')[1].strip()}": u
                    for u in all_units}
    brk = detect_level_breaks(panel, col="ln_inc_real").set_index("unit")
    pmin, pmax = panel["MONTH_DT"].min(), panel["MONTH_DT"].max()

    rows = []
    for _, t in tre.iterrows():
        unit, how = match_unit(t["state"], t["district"], units_by_key, all_units)
        rec = {"site": t["site_name"], "tradition": t.get("tradition", ""),
               "district": t["district"], "match": how, "unit": unit}

        if unit is None:
            rec.update(verdict="NOT IN CPHS", reason="district absent from panel")
            rows.append(rec); continue

        d = urban[urban["unit"] == unit]
        n_months = panel["MONTH_DT"].nunique()
        cov = d["MONTH_DT"].nunique() / n_months
        rec["coverage"] = round(cov, 3)
        rec["min_n"] = int(d["n_hh"].min())
        rec["median_n"] = int(d["n_hh"].median())

        # Panel dates are month starts. A completion date of 22 Jan 2024
        # compared against 2024-01-01 silently drops January, so round both
        # anchors down to the month before any comparison.
        anchor = pd.to_datetime(t.get("sanction_date"), errors="coerce")
        done = pd.to_datetime(t.get("completion_date"), errors="coerce")
        anchor = anchor.to_period("M").to_timestamp() if pd.notna(anchor) else anchor
        done = done.to_period("M").to_timestamp() if pd.notna(done) else done
        if pd.isna(anchor) and pd.isna(done):
            rec.update(verdict="NO DATE", reason="need sanction or completion date")
            rows.append(rec); continue
        if pd.isna(anchor):
            anchor = done          # fall back, but flag it
            rec["date_note"] = "no sanction date - pre-window may be contaminated"

        pre = int(((d["MONTH_DT"] >= pmin) & (d["MONTH_DT"] < anchor)).sum())
        post = int((d["MONTH_DT"] >= (done if pd.notna(done) else anchor)).sum())
        rec["pre_months"], rec["post_months"] = pre, post

        b = brk.loc[unit] if unit in brk.index else None
        rec["break"] = round(float(b["max_step"]), 3) if b is not None else np.nan
        break_in_post = bool(
            b is not None and b["suspect_break"]
            and pd.notna(b["step_month"])
            and b["step_month"] >= (done if pd.notna(done) else anchor))

        fails, warns = [], []
        if cov < 1.0:
            fails.append(f"coverage {cov:.2f} < 1.00")
        # min_n is the DONOR screen, not an estimability bar. A treated unit
        # only needs complete coverage -- a unit with min_n 28 (four COVID
        # months) still estimates perfectly well. A thin month raises the noise
        # floor, which the estimator already reports, so warn rather than fail.
        if rec["min_n"] < MIN_HH:
            warns.append(f"min_n {rec['min_n']} (thin month; raises noise floor)")
        if pre < MIN_PRE_MONTHS:
            fails.append(f"only {pre} pre months")
        if post < MIN_POST_MONTHS:
            fails.append(f"only {post} post months")
        if break_in_post:
            fails.append(f"suspect level break at {b['step_month']:%b %Y} in post window")

        rec["verdict"] = "OK" if not fails else "FAIL"
        rec["reason"] = "; ".join(fails) or ("warn: " + "; ".join(warns) if warns else "")
        rows.append(rec)

    r = pd.DataFrame(rows)
    cols = ["site", "tradition", "district", "match", "coverage", "min_n",
            "pre_months", "post_months", "break", "verdict", "reason"]
    r = r.reindex(columns=[c for c in cols if c in r.columns])

    pd.set_option("display.width", 200, "display.max_colwidth", 46)
    print(r.to_string(index=False))

    ok = r[r["verdict"] == "OK"]
    print("\n" + "=" * 78)
    print(f"estimable: {len(ok)} of {len(r)} sites")
    if len(ok):
        by = ok["tradition"].value_counts()
        print("by tradition: " + ", ".join(f"{k} {v}" for k, v in by.items()))

    # Section 9 go/no-go, fixed in advance.
    enough_sites = len(ok) >= 20
    trads = ok["tradition"].value_counts() if len(ok) else pd.Series(dtype=int)
    enough_trads = int((trads >= 3).sum()) >= 3
    print("\nGO / NO-GO (scoping doc section 9)")
    print(f"  >= 20 estimable sites          : {'PASS' if enough_sites else 'FAIL'} ({len(ok)})")
    print(f"  >= 3 traditions with >= 3 sites: {'PASS' if enough_trads else 'FAIL'} "
          f"({int((trads >= 3).sum())})")
    if enough_sites and enough_trads:
        print("\n  -> GO. Proceed to estimation.")
    else:
        print("\n  -> NO-GO as a multi-site study. The honest fallback is a")
        print("     small-N paper on the sites that do clear, which is what")
        print("     the predecessor study already is.")

    fuzzy = r[r["match"].astype(str).str.startswith("fuzzy")]
    if len(fuzzy):
        print(f"\n{len(fuzzy)} district name(s) matched only approximately -- CONFIRM BY HAND:")
        for _, f in fuzzy.iterrows():
            print(f"  {f['district']}  ->  {f['match']}")

    out = Path(C.OUT_DIR) / "treatment_screen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
