"""
test_deflate.py -- regression tests for the CPI deflator.

Run:  python test_deflate.py

The CPI values here are SYNTHETIC, generated in-process from a fixed seed.
They are a test fixture for the join-and-rebase logic and nothing else. They
are never written next to the real data and must never be used as a deflator.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from panel import canonical_state, deflate


def _panel(states=("Uttar Pradesh", "Odisha", "Jammu & Kashmir"), n_months=6):
    months = pd.date_range("2014-01-01", periods=n_months, freq="MS")
    rows = []
    for st in states:
        for m in months:
            rows.append({"STATE": st, "DISTRICT": f"{st[:3]}town",
                         "REGION_TYPE": "URBAN", "MONTH_DT": m,
                         "ln_inc": 10.0, "mean_inc": 22026.4657})
    p = pd.DataFrame(rows)
    p["unit"] = p["STATE"] + " | " + p["DISTRICT"] + " | " + p["REGION_TYPE"]
    return p


def _cpi(states, n_months=6, seed=0, shuffle=True):
    """SYNTHETIC index values -- fixture only, not real CPI."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2014-01-01", periods=n_months, freq="MS")
    rows = []
    for st in states:
        level = 100.0
        for m in months:
            level *= 1.0 + rng.uniform(0.002, 0.008)
            rows.append({"STATE": st, "REGION_TYPE": "URBAN",
                         "MONTH_DT": m, "CPI": round(level, 4)})
    df = pd.DataFrame(rows)
    if shuffle:                       # the bug this guards: unsorted input
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _write(df, tmp="/tmp/_synthetic_cpi_TESTONLY.csv"):
    df.to_csv(tmp, index=False)
    return tmp


def test_rebase_is_month_anchored_not_row_order():
    """
    The old code used groupby(...).transform("first"), which took whichever
    row appeared first in the FILE. On unsorted input every state got a
    different base period. Deflators must be anchored to one calendar month.
    """
    states = ["Uttar Pradesh", "Odisha", "Jammu & Kashmir"]
    p = _panel(states)
    cpi = _cpi(states)
    out = deflate(p.copy(), _write(cpi), verbose=False)

    base = out["MONTH_DT"].min()
    at_base = out.loc[out["MONTH_DT"] == base, "defl"]
    assert np.allclose(at_base, 1.0), \
        f"deflator at base month should be exactly 1.0, got {at_base.tolist()}"

    # And the answer must not depend on row order in the file.
    out2 = deflate(p.copy(), _write(_cpi(states, shuffle=False)), verbose=False)
    assert np.allclose(out["defl"].to_numpy(), out2["defl"].to_numpy()), \
        "deflator changed when the CPI file was re-ordered"
    print("PASS  rebase anchored to base month, order-independent")


def test_state_aliases_join():
    """MoSPI spellings must fold onto CPHS spellings."""
    assert canonical_state("Jammu and Kashmir") == "Jammu & Kashmir"
    assert canonical_state("Orissa") == "Odisha"
    assert canonical_state("NCT of Delhi") == "Delhi"
    assert canonical_state("Uttaranchal") == "Uttarakhand"
    assert canonical_state("  Tamil   Nadu ") == "Tamil Nadu"

    states = ["Uttar Pradesh", "Odisha", "Jammu & Kashmir"]
    p = _panel(states)
    cpi = _cpi(["Uttar Pradesh", "Orissa", "Jammu and Kashmir"])
    out = deflate(p.copy(), _write(cpi), verbose=False)
    assert out["unit"].nunique() == 3, \
        f"alias folding failed; kept {out['unit'].nunique()}/3 units"
    assert out["defl"].notna().all()
    print("PASS  state aliases fold across CPHS/MoSPI spellings")


def test_uncovered_units_are_dropped_not_left_nominal():
    """
    The dangerous old behaviour: unmatched rows were filled with defl=1.0, so
    some units were real and others nominal inside one panel. SCM compares the
    treated unit directly against donors, so that mixture manufactures a trend.
    """
    states = ["Uttar Pradesh", "Odisha", "Jammu & Kashmir"]
    p = _panel(states)
    cpi = _cpi(["Uttar Pradesh", "Odisha"])          # J&K missing entirely
    out = deflate(p.copy(), _write(cpi), verbose=False)

    kept = set(out["STATE"])
    assert "Jammu & Kashmir" not in kept, \
        "unit with no CPI coverage survived; it would be silently nominal"
    assert kept == {"Uttar Pradesh", "Odisha"}
    assert out["defl"].notna().all(), "a NaN deflator survived"
    print("PASS  units without CPI coverage are dropped, not left nominal")


def test_partial_month_coverage_drops_the_whole_unit():
    """One missing month is enough to disqualify a unit."""
    states = ["Uttar Pradesh", "Odisha"]
    p = _panel(states)
    cpi = _cpi(states)
    cpi = cpi[~((cpi["STATE"] == "Odisha") &
                (cpi["MONTH_DT"] == pd.Timestamp("2014-04-01")))]
    out = deflate(p.copy(), _write(cpi), verbose=False)
    assert set(out["STATE"]) == {"Uttar Pradesh"}, \
        "a unit with a hole in its CPI series was kept"
    print("PASS  partial coverage disqualifies the whole unit")


def test_missing_base_month_raises():
    """A CPI file that starts after the panel cannot silently half-work."""
    states = ["Uttar Pradesh"]
    p = _panel(states)
    cpi = _cpi(states)
    cpi = cpi[cpi["MONTH_DT"] > pd.Timestamp("2014-01-01")]
    try:
        deflate(p.copy(), _write(cpi), verbose=False)
    except ValueError as e:
        assert "base month" in str(e)
        print("PASS  missing base month raises instead of guessing")
        return
    raise AssertionError("expected ValueError for missing base month")


def test_schema_error_names_the_columns():
    p = _panel(["Uttar Pradesh"])
    bad = pd.DataFrame({"state": ["Uttar Pradesh"], "sector": ["URBAN"],
                        "month": ["2014-01-01"], "index": [100.0]})
    try:
        deflate(p.copy(), _write(bad), verbose=False)
    except (ValueError, KeyError) as e:
        assert "STATE" in str(e)
        print("PASS  wrong schema fails loudly and names what it wanted")
        return
    raise AssertionError("expected an error for a malformed CPI file")


def test_real_income_is_actually_deflated():
    """ln_inc_real must fall below ln_inc once prices rise above base."""
    states = ["Uttar Pradesh"]
    p = _panel(states)
    out = deflate(p.copy(), _write(_cpi(states)), verbose=False)
    last = out.sort_values("MONTH_DT").iloc[-1]
    assert last["defl"] > 1.0
    assert last["ln_inc_real"] < last["ln_inc"], \
        "real income was not reduced by a rising price level"
    assert np.isclose(last["ln_inc_real"],
                      last["ln_inc"] - np.log(last["defl"]))
    assert np.isclose(last["mean_inc_real"], last["mean_inc"] / last["defl"])
    print("PASS  real series is nominal minus log price level")


def test_nominal_passthrough_is_flagged():
    p = _panel(["Uttar Pradesh"])
    out = deflate(p.copy(), None, verbose=False)
    assert out["_deflated"].eq(False).all()
    assert np.allclose(out["ln_inc_real"], out["ln_inc"])
    assert "mean_inc_real" in out.columns
    print("PASS  nominal passthrough sets _deflated=False and keeps columns")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} deflator tests passed")
