"""
2026-09-05-screen_treatments2.py -- screen treatment sites, considering BOTH
sectors and allowing a later analysis start.

    python 2026-09-05-screen_treatments2.py 2026-09-05-treatments_transport.csv
    python 2026-09-05-screen_treatments2.py <file> --start 2016-01

Two changes from screen_treatments.py, both forced by the transport screen:

1. It looked only at URBAN units. Gurdaspur exists in CPHS as RURAL at
   coverage 1.000 and was reported "NOT IN CPHS". For a site like the
   Kartarpur crossing at Dera Baba Nanak, rural is the CORRECT sector, not a
   fallback. Both sectors are now checked and the one that scores better is
   used, with the sector reported.

2. Coverage was measured over the whole panel, Jan 2014 to Dec 2025. A site
   treated in 2018 was failed for holes in 2014 that it never needed.
   --start moves the analysis window; coverage and the pre-window are then
   measured from there.

Everything else -- the coverage bar, the break test, the minimum window
lengths, the fuzzy-match warning -- is unchanged from the original.

The go/no-go block from the old scoping document is NOT reproduced. It asked
for 20 sites and three traditions, which belonged to the abandoned multi-faith
design and only confuses the output now.
"""

from __future__ import annotations

import argparse
import sys
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from panel import canonical_district, canonical_state, detect_level_breaks

MIN_PRE_MONTHS = 24
MIN_POST_MONTHS = 24
MIN_HH = 60


def match_unit(state, district, by_key, all_units):
    """Map a site's state+district onto CPHS labels, returning both sectors."""
    st, di = canonical_state(state), canonical_district(district)
    hits, how = {}, "exact"
    for sec in ("URBAN", "RURAL"):
        u = by_key.get(f"{st}|{di}|{sec}")
        if u:
            hits[sec] = u
    if hits:
        return hits, how

    cand = [u for u in all_units if u.split("|")[0].strip() == st]
    names = sorted({u.split("|")[1].strip() for u in cand})
    near = get_close_matches(di, names, n=1, cutoff=0.82)
    if not near:
        return {}, "no match"
    for sec in ("URBAN", "RURAL"):
        u = by_key.get(f"{st}|{near[0]}|{sec}")
        if u:
            hits[sec] = u
    return hits, f"fuzzy->{near[0]}"


def assess(d, unit, sec, brk, months, anchor, done):
    """Score one candidate unit-sector. Returns a record dict."""
    have = set(d["MONTH_DT"])
    cov = len(have) / len(months)
    rec = {"sector": sec, "coverage": round(cov, 3),
           "min_n": int(d["n_hh"].min()), "median_n": int(d["n_hh"].median())}

    rec["pre_months"] = int(sum(1 for m in months if m in have and m < anchor))
    rec["post_months"] = int(sum(1 for m in months if m in have and m >= done))

    b = brk.loc[unit] if unit in brk.index else None
    rec["break"] = round(float(b["max_step"]), 3) if b is not None else np.nan
    break_in_post = bool(b is not None and b["suspect_break"]
                         and pd.notna(b["step_month"]) and b["step_month"] >= done)

    fails, warns = [], []
    if cov < 1.0:
        fails.append(f"coverage {cov:.3f} < 1.00")
    if rec["min_n"] < MIN_HH:
        warns.append(f"min_n {rec['min_n']} (thin month; raises noise floor)")
    if rec["pre_months"] < MIN_PRE_MONTHS:
        fails.append(f"only {rec['pre_months']} pre months")
    if rec["post_months"] < MIN_POST_MONTHS:
        fails.append(f"only {rec['post_months']} post months")
    if break_in_post:
        fails.append(f"suspect break {b['step_month']:%b %Y} in post window")

    rec["verdict"] = "OK" if not fails else "FAIL"
    rec["reason"] = "; ".join(fails) or ("warn: " + "; ".join(warns) if warns else "")
    rec["_rank"] = (0 if not fails else 1, len(fails), -cov)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("treatments")
    ap.add_argument("--start", default=None,
                    help="analysis start, YYYY-MM. Coverage and the pre-window "
                         "are measured from here instead of the panel start.")
    a = ap.parse_args()

    tre = pd.read_csv(a.treatments)
    need = {"site_name", "state", "district", "tradition",
            "sanction_date", "completion_date"}
    missing = need - set(tre.columns)
    if missing:
        sys.exit(f"treatment file missing column(s): {sorted(missing)}")

    if not Path(C.PANEL_PARQUET).exists():
        sys.exit(f"No panel at {C.PANEL_PARQUET}. Run run_01_build_panel.py first.")
    panel = pd.read_parquet(C.PANEL_PARQUET)

    months = sorted(panel["MONTH_DT"].unique())
    if a.start:
        cut = pd.Timestamp(a.start)
        months = [m for m in months if m >= cut]
        panel = panel[panel["MONTH_DT"] >= cut].copy()
        if len(months) < MIN_PRE_MONTHS + MIN_POST_MONTHS:
            sys.exit(f"--start {a.start} leaves only {len(months)} months.")
    print(f"panel window: {pd.Timestamp(months[0]):%b %Y} to "
          f"{pd.Timestamp(months[-1]):%b %Y} ({len(months)} months)\n")

    all_units = sorted(panel["unit"].unique())
    by_key = {"|".join(p.strip() for p in u.split("|")): u for u in all_units}
    brk = detect_level_breaks(panel, col="ln_inc_real").set_index("unit")

    rows = []
    for _, t in tre.iterrows():
        base = {"site": t["site_name"], "group": t.get("group", ""),
                "tradition": t.get("tradition", ""), "district": t["district"]}
        hits, how = match_unit(t["state"], t["district"], by_key, all_units)
        base["match"] = how
        if not hits:
            rows.append({**base, "verdict": "NOT IN CPHS",
                         "reason": "district absent from panel"})
            continue

        anchor = pd.to_datetime(t.get("sanction_date"), errors="coerce")
        done = pd.to_datetime(t.get("completion_date"), errors="coerce")
        anchor = anchor.to_period("M").to_timestamp() if pd.notna(anchor) else anchor
        done = done.to_period("M").to_timestamp() if pd.notna(done) else done
        if pd.isna(anchor) and pd.isna(done):
            rows.append({**base, "verdict": "NO DATE",
                         "reason": "need sanction or completion date"})
            continue
        if pd.isna(anchor):
            anchor = done
        if pd.isna(done):
            done = anchor

        cands = [assess(panel[panel["unit"] == u], u, sec, brk, months, anchor, done)
                 for sec, u in hits.items()]
        best = min(cands, key=lambda r: r["_rank"])
        best.pop("_rank")
        rows.append({**base, **best})

    r = pd.DataFrame(rows)
    cols = ["site", "group", "tradition", "district", "match", "sector",
            "coverage", "min_n", "pre_months", "post_months", "break",
            "verdict", "reason"]
    r = r.reindex(columns=[c for c in cols if c in r.columns])
    pd.set_option("display.width", 250, "display.max_colwidth", 44)
    print(r.to_string(index=False))

    ok = r[r["verdict"] == "OK"]
    print("\n" + "=" * 90)
    print(f"estimable: {len(ok)} of {len(r)}")
    if "group" in r.columns and len(ok):
        for gname, sub in ok.groupby("group"):
            print(f"  {gname:<24} {len(sub)}  ({', '.join(sub['site'])})")

    fuzzy = r[r["match"].astype(str).str.startswith("fuzzy")]
    if len(fuzzy):
        print(f"\n{len(fuzzy)} district name(s) matched only approximately "
              f"-- CONFIRM BY HAND:")
        for _, f in fuzzy.iterrows():
            print(f"  {f['district']}  ->  {f['match']}")

    out = Path(C.OUT_DIR) / "treatment_screen2.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
