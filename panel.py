"""
src/panel.py — turn 144 monthly CPHS wave files into a district x month panel.

The whole raw set is ~8 GB of CSV. This module never holds more than one
month in memory: each file is read, aggregated, and discarded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# District renames that break panel continuity. Every alias maps to ONE label.
# India renames districts, and CPHS switches labels mid-panel when it does.
# Miss one and the old name ends, a new one begins, and the estimator reads a
# phantom structural break. Worse, both halves then fail the coverage screen
# and vanish from the donor pool without a word.
DISTRICT_ALIASES = {
    "Faizabad": "Ayodhya",
    "Ayodhya": "Ayodhya",
    "Allahabad": "Prayagraj",
    "Prayagraj": "Prayagraj",
    "Ahmadabad": "Ahmedabad",
    "Ahmedabad": "Ahmedabad",
    "Bangalore": "Bengaluru",
    "Bengaluru": "Bengaluru",
    "Gurgaon": "Gurugram",
    "Gurugram": "Gurugram",
    "Mysore": "Mysuru",
    "Mysuru": "Mysuru",
    # Ahmednagar -> Ahilyanagar (renamed 2024). CPHS switched labels in the
    # Aug 2025 wave, so the panel carried "Ahmadnagar" Jan 2014-Jul 2025 and
    # "Ahilyanagar" Aug-Dec 2025 as two separate units. Both then failed the
    # 100%-coverage screen and were dropped. Three spellings are needed:
    # CPHS writes Ahmadnagar, the treatment file writes Ahmednagar.
    "Ahmadnagar": "Ahilyanagar",
    "Ahmednagar": "Ahilyanagar",
    "Ahilyanagar": "Ahilyanagar",
}


# ------------------------------------------------------------- helpers ----

def weighted_mean(v, w) -> float:
    v = np.asarray(v, float); w = np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float((v[ok] * w[ok]).sum() / w[ok].sum())


def weighted_quantile(v, w, q) -> float:
    v = np.asarray(v, float); w = np.asarray(w, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    v, w = v[ok], w[ok]
    o = np.argsort(v); v, w = v[o], w[o]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, v))


def canonical_district(name):
    if pd.isna(name):
        return name
    s = str(name).strip()
    return DISTRICT_ALIASES.get(s, s)


def _weighted_percentiles(df, keys, value_col, weight_col, qs):
    """
    Weighted percentiles for every group at once, with no Python-level loop
    over groups.

    Sort within group by value, walk the cumulative weight share, and take the
    first observation whose share reaches q. Note that CPHS assigns weights at
    the stratum level, so within a district-month cell most households share
    an identical weight and this is very close to the unweighted percentile --
    but it stays correct where a cell spans two strata.
    """
    d = df[keys + [value_col, weight_col]].copy()
    d = d.sort_values(keys + [value_col], kind="mergesort")

    g = d.groupby(keys, dropna=False, observed=True)[weight_col]
    cw = g.cumsum() - 0.5 * d[weight_col]
    tot = g.transform("sum")
    d["_share"] = (cw / tot.replace(0, np.nan)).astype(float)

    out = None
    for q in qs:
        d["_hit"] = d["_share"] >= q
        # idxmax on a boolean gives the first True within each group
        idx = d.groupby(keys, dropna=False, observed=True)["_hit"].idxmax()
        col = f"inc_p{int(q * 100)}"
        part = d.loc[idx, keys + [value_col]].rename(columns={value_col: col})
        out = part if out is None else out.merge(part, on=keys, how="outer")
    return out.reset_index(drop=True)


# --------------------------------------------------------- one raw file ----

def process_wave_file(path, usecols, income_sources, weight_col,
                      winsor=(0.01, 0.99)):
    """Read one monthly CPHS file -> (district-month aggregate, audit row)."""
    df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    n_raw = len(df)

    # Fail loudly and by name if a wave is missing a column we depend on.
    # CPHS schemas drift between waves; a bare KeyError 200 files in is useless.
    required = ["RESPONSE_STATUS", "TOT_INC", "DISTRICT", "STATE",
                "REGION_TYPE", "MONTH", "HH_ID", "PSU_ID", "HR", weight_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{path.name}: missing required column(s) {missing}. "
                       f"Present: {sorted(df.columns)}")

    # --- 1. Non-response. CPHS codes missing income as -99, NOT as blank.
    #        Skipping this step silently destroys every mean you compute.
    df = df[df["RESPONSE_STATUS"] == "Accepted"]
    n_acc = len(df)
    df = df[df["TOT_INC"] > 0].copy()
    n_pos = len(df)
    if n_pos == 0:
        warnings.warn(f"{path.name}: no usable rows")
        return None, None

    # --- 2. Geography + time
    df["DISTRICT"] = df["DISTRICT"].map(canonical_district)
    df["MONTH_DT"] = pd.to_datetime(df["MONTH"], format="%b %Y", errors="coerce")

    # --- 3. Winsorise the income tail, then log
    lo, hi = df["TOT_INC"].quantile(winsor[0]), df["TOT_INC"].quantile(winsor[1])
    df["INC_W"] = df["TOT_INC"].clip(lo, hi)
    df["LN_INC"] = np.log(df["INC_W"])
    df["_w"] = df[weight_col].fillna(0.0)

    # --- 4. Collapse to STATE x DISTRICT x REGION_TYPE x MONTH
    #        Fully vectorised: a per-group Python loop over ~1,400 cells x 144
    #        files is the difference between 35 minutes and 4.
    keys = ["STATE", "DISTRICT", "REGION_TYPE", "MONTH_DT"]
    df["_w_ln"] = df["_w"] * df["LN_INC"]
    df["_w_inc"] = df["_w"] * df["INC_W"]
    for src in income_sources:
        if src in df.columns:
            df[f"_pos_{src}"] = df[src].clip(lower=0)
            df[f"_w_{src}"] = df["_w"] * df[f"_pos_{src}"]

    gb = df.groupby(keys, dropna=False, observed=True)
    agg = gb.agg(
        n_hh=("HH_ID", "size"),
        n_psu=("PSU_ID", "nunique"),
        hr=("HR", "first"),
        _w_sum=("_w", "sum"),
        _w_ln_sum=("_w_ln", "sum"),
        _w_inc_sum=("_w_inc", "sum"),
        _ln_sd=("LN_INC", "std"),
        _tot_inc=("TOT_INC", "sum"),
    ).reset_index()

    agg["ln_inc"] = agg["_w_ln_sum"] / agg["_w_sum"].replace(0, np.nan)
    # Weighted mean income in rupees. ln_inc is the mean of logs, which is not
    # the log of the mean; the source decomposition needs an actual level.
    agg["mean_inc"] = agg["_w_inc_sum"] / agg["_w_sum"].replace(0, np.nan)
    agg["ln_inc_se"] = agg["_ln_sd"] / np.sqrt(agg["n_hh"])

    for src in income_sources:
        if src in df.columns:
            s = gb.agg(**{f"_ws_{src}": (f"_w_{src}", "sum"),
                          f"_ps_{src}": (f"_pos_{src}", "sum")}).reset_index()
            agg = agg.merge(s, on=keys, how="left")
            agg[f"mean_{src}"] = agg[f"_ws_{src}"] / agg["_w_sum"].replace(0, np.nan)
            agg[f"share_{src}"] = agg[f"_ps_{src}"] / agg["_tot_inc"].replace(0, np.nan)

    # Weighted percentiles, vectorised via cumulative weight shares.
    q_tbl = _weighted_percentiles(df, keys, "INC_W", "_w", [0.25, 0.50, 0.75])
    agg = agg.merge(q_tbl, on=keys, how="left")

    # Composition controls -- guard against the CPHS sample itself shifting
    # underneath you (an occupation-mix change can masquerade as an effect).
    if "OCCUPATION_GROUP" in df.columns:
        occ = (df.assign(_one=1)
                 .pivot_table(index=keys, columns="OCCUPATION_GROUP",
                              values="_one", aggfunc="sum", fill_value=0))
        occ = occ.div(occ.sum(axis=1), axis=0).reset_index()
        rename = {"Wage Labourers": "sh_wage_lab",
                  "White-collar Employees": "sh_white_collar",
                  "Entrepreneurs": "sh_entrepreneur"}
        occ = occ[keys + [c for c in rename if c in occ.columns]].rename(columns=rename)
        agg = agg.merge(occ, on=keys, how="left")

    agg = agg.drop(columns=[c for c in agg.columns if c.startswith("_")])
    audit = {
        "file": path.name,
        "month": df["MONTH_DT"].iloc[0],
        "rows_raw": n_raw,
        "rows_accepted": n_acc,
        "rows_used": n_pos,
        "nonresponse_rate": 1 - n_acc / n_raw if n_raw else np.nan,
        "n_districts": agg[["STATE", "DISTRICT"]].drop_duplicates().shape[0],
        "n_psu": df["PSU_ID"].nunique(),
        "median_hh_per_urban_cell": float(
            agg.loc[agg.REGION_TYPE == "URBAN", "n_hh"].median()),
    }
    return agg, audit


# ------------------------------------------------------------ the panel ----

def build_panel(raw_dir, file_glob, usecols, income_sources, weight_col,
                winsor=(0.01, 0.99), verbose=True):
    files = sorted(raw_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matched {file_glob!r} under {raw_dir}. "
            "Check RAW_DIR in config.py."
        )
    if verbose:
        print(f"Found {len(files)} wave files "
              f"({files[0].name} ... {files[-1].name})")

    parts, audits = [], []
    for i, f in enumerate(files, 1):
        agg, audit = process_wave_file(f, usecols, income_sources,
                                       weight_col, winsor)
        if agg is not None:
            parts.append(agg); audits.append(audit)
        if verbose and (i % 12 == 0 or i == len(files)):
            print(f"  {i:>3}/{len(files)}  {f.name}")

    panel = pd.concat(parts, ignore_index=True)
    panel["unit"] = (panel["STATE"] + " | " + panel["DISTRICT"]
                     + " | " + panel["REGION_TYPE"])
    panel = panel.sort_values(["unit", "MONTH_DT"]).reset_index(drop=True)
    return panel, pd.DataFrame(audits).sort_values("month")


# ------------------------------------------------------- post-processing ----

# State-name normalisation for the CPHS <-> CPI join. CPHS says
# "Jammu & Kashmir", MoSPI says "Jammu and Kashmir"; CPHS says "Odisha", older
# CPI vintages say "Orissa". An unnormalised merge silently drops those states,
# and the old code then filled them with defl=1.0 -- mixing real and nominal
# series inside one panel, which is worse than running fully nominal.
STATE_ALIASES = {
    "JAMMU & KASHMIR": "Jammu & Kashmir",
    "JAMMU & KASHMIR & LADAKH": "Jammu & Kashmir",
    "JAMMU & KASHMIR AND LADAKH": "Jammu & Kashmir",
    "ORISSA": "Odisha",
    "ODISHA": "Odisha",
    "UTTARANCHAL": "Uttarakhand",
    "UTTARAKHAND": "Uttarakhand",
    "PONDICHERRY": "Puducherry",
    "PUDUCHERRY": "Puducherry",
    "NCT OF DELHI": "Delhi",
    "DELHI (NCT)": "Delhi",
    "DELHI": "Delhi",
    "CHATTISGARH": "Chhattisgarh",
    "CHHATTISGARH": "Chhattisgarh",
    "TELENGANA": "Telangana",
    "TELANGANA": "Telangana",
    "ANDHRA PRADESH (UNDIVIDED)": "Andhra Pradesh",
    "TAMILNADU": "Tamil Nadu",
}


def canonical_state(name):
    """Fold spelling variants so CPHS and CPI state labels meet."""
    if pd.isna(name):
        return name
    s = " ".join(str(name).strip().split())
    key = s.upper().replace(" AND ", " & ")
    if key in STATE_ALIASES:
        return STATE_ALIASES[key]
    return s


def deflate(panel, cpi_csv, base_month=None, verbose=True):
    """
    Convert nominal to real income using state x sector x month CPI.

    Expected CPI schema (one row per state-sector-month):
        STATE, REGION_TYPE, MONTH_DT, CPI     (any base year)

    Source: MoSPI CPI, https://cpi.mospi.gov.in -> Rural/Urban indices by state,
    or CMIE Economic Outlook, Statistics -> Inflation -> CPI Base Year 2012.

    Three things the previous version got wrong, all of which bias quietly:

    1. It re-based with groupby(...).transform("first"), which takes the first
       row in FILE order, not the earliest month. On an unsorted CSV every
       state got a different base period and the deflators were incomparable.
       We now re-base every state-sector on one explicit calendar month.
    2. It did not normalise state names, so any spelling mismatch failed to
       merge.
    3. Worst: unmatched rows were filled with defl=1.0, leaving some units real
       and others nominal inside the same panel. Synthetic control compares the
       treated unit against donors directly, so a deflated treated unit against
       partly-nominal donors manufactures a trend. Units without complete CPI
       coverage are now DROPPED, loudly, rather than silently left nominal.
    """
    if cpi_csv is None:
        panel["ln_inc_real"] = panel["ln_inc"]
        panel["mean_inc_real"] = panel["mean_inc"]
        panel["_deflated"] = False
        return panel

    # Read first, validate second. Passing parse_dates=["MONTH_DT"] up front
    # makes pandas raise its own opaque error on a malformed file, before we
    # can say which columns we actually wanted.
    cpi = pd.read_csv(cpi_csv)
    need = {"STATE", "REGION_TYPE", "MONTH_DT", "CPI"}
    missing = need - set(cpi.columns)
    if missing:
        raise ValueError(f"{cpi_csv}: CPI file missing column(s) {sorted(missing)}. "
                         f"Expected {sorted(need)}; found {sorted(cpi.columns)}.")
    cpi["MONTH_DT"] = pd.to_datetime(cpi["MONTH_DT"], errors="coerce")

    cpi["STATE"] = cpi["STATE"].map(canonical_state)
    cpi["REGION_TYPE"] = cpi["REGION_TYPE"].astype(str).str.strip().str.upper()
    cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce")
    cpi = (cpi.dropna(subset=["STATE", "REGION_TYPE", "MONTH_DT", "CPI"])
              .drop_duplicates(["STATE", "REGION_TYPE", "MONTH_DT"])
              .sort_values(["STATE", "REGION_TYPE", "MONTH_DT"]))

    # One common calendar base for every state-sector.
    base = pd.Timestamp(base_month) if base_month else panel["MONTH_DT"].min()
    base_vals = (cpi.loc[cpi["MONTH_DT"] == base]
                    .set_index(["STATE", "REGION_TYPE"])["CPI"])
    if base_vals.empty:
        raise ValueError(
            f"No CPI rows for the base month {base:%b %Y}. The CPI file must "
            f"cover the panel's first month. CPI spans "
            f"{cpi['MONTH_DT'].min():%b %Y} to {cpi['MONTH_DT'].max():%b %Y}.")

    cpi = cpi.join(base_vals.rename("_base"), on=["STATE", "REGION_TYPE"])
    cpi["defl"] = cpi["CPI"] / cpi["_base"]

    # A month for which NO state has CPI is a publication gap in the source,
    # not a per-unit coverage failure. India suspended price collection during
    # the 2020 lockdown and MoSPI published no CPI for April and May 2020.
    # Treating those as missing coverage would drop every unit in the panel.
    # Dropping the months instead costs 2 of 144 and keeps the treated unit and
    # its donors on identical calendars, which is what the comparison needs.
    gap = sorted(set(panel["MONTH_DT"]) - set(cpi["MONTH_DT"]))
    if gap:
        shown = ", ".join(f"{m:%b %Y}" for m in gap[:6])
        if len(gap) > 6:
            shown += f", ... ({len(gap)} total)"
        print(f"  CPI publication gap, absent for every state: {shown}")
        print(f"  -> dropping those {len(gap)} month(s) from the panel.")
        panel = panel[~panel["MONTH_DT"].isin(gap)].copy()

    panel = panel.merge(cpi[["STATE", "REGION_TYPE", "MONTH_DT", "defl"]],
                        on=["STATE", "REGION_TYPE", "MONTH_DT"], how="left")

    # A unit is usable only if EVERY one of its months deflates.
    ok = panel.groupby("unit")["defl"].transform(lambda s: s.notna().all())
    dropped = sorted(panel.loc[~ok, "unit"].unique())
    n_before = panel["unit"].nunique()
    panel = panel[ok].copy()

    if verbose:
        print(f"Deflator: base {base:%b %Y}; "
              f"{n_before - len(dropped)}/{n_before} units fully covered.")
        if dropped:
            states = sorted({u.split("|")[0].strip() for u in dropped})
            print(f"  DROPPED {len(dropped)} unit(s) with incomplete CPI "
                  f"coverage, in: {', '.join(states)}")
    if panel.empty:
        raise ValueError("Every unit was dropped for want of CPI coverage. "
                         "Check the state names and month range in the CPI file.")

    panel["ln_inc_real"] = panel["ln_inc"] - np.log(panel["defl"])
    panel["mean_inc_real"] = panel["mean_inc"] / panel["defl"]
    panel["_deflated"] = True
    return panel


def smooth(panel, col="ln_inc_real", window=3):
    """Centred moving average within unit. Cuts monthly sampling noise."""
    out = panel.sort_values(["unit", "MONTH_DT"]).copy()
    out[f"{col}_sm"] = (out.groupby("unit")[col]
                           .transform(lambda s: s.rolling(window, center=True,
                                                          min_periods=1).mean()))
    return out


def screen_units(panel, min_hh, min_coverage_frac=1.0):
    """
    Which units can serve as donors?

    A unit qualifies only if it appears in (nearly) every month with an
    adequate sample. Units that blink in and out of CPHS produce phantom
    'treatment effects' that are really sampling churn.
    """
    n_months = panel["MONTH_DT"].nunique()
    s = (panel.groupby("unit")
              .agg(state=("STATE", "first"),
                   district=("DISTRICT", "first"),
                   region=("REGION_TYPE", "first"),
                   months=("MONTH_DT", "nunique"),
                   min_n_hh=("n_hh", "min"),
                   median_n_hh=("n_hh", "median"),
                   median_se=("ln_inc_se", "median"))
              .reset_index())
    s["coverage"] = s["months"] / n_months
    s["eligible"] = (s["min_n_hh"] >= min_hh) & (s["coverage"] >= min_coverage_frac)
    return s.sort_values("median_n_hh", ascending=False).reset_index(drop=True)


# Pre-treatment structural profile used to pick economically comparable
# donors. FIXED IN ADVANCE (2 Sep 2026) before any matched-pool estimate was
# computed, so that neither the variable list nor K could be chosen to produce
# a preferred answer.
STRUCTURAL_MATCH_VARS = [
    "sh_entrepreneur",
    "sh_wage_lab",
    "sh_white_collar",
    "share_INC_OF_HH_FRM_BIZ_PROFIT",
    "ln_inc_real",
    "log_n_hh",
]


def structural_neighbours(panel, treated_unit, pool, before, k=30,
                          variables=None, verbose=True):
    """
    Restrict a donor pool to the k units most like the treated unit in
    pre-treatment ECONOMIC STRUCTURE, not just in income path.

    Why this exists. The estimator matches on the pre-treatment outcome path
    and nothing else, so it will happily build a small district's twin out of
    metros carrying roughly twice its income and a tenth of its wage-labourer
    share. The income LEVEL then matches almost exactly while the structure
    does not, and the fitted counterfactual ends up with an entrepreneur share
    less than half the treated unit's. That matters whenever a finding concerns
    business profit or the top of the income distribution, because the
    counterfactual then under-represents the very group the result is about.

    This is deliberately a SCREEN, not a fitted covariate weighting. Adding
    covariates to the SCM objective requires choosing a V matrix, and that
    choice is the researcher degree of freedom Ferman, Pinto & Possebom (2020)
    show can be tuned to produce whatever result is wanted. Selecting the donor
    pool in advance on a fixed rule introduces no such freedom: it is the same
    kind of restriction as "urban only" or "exclude the treated state".

    Distances are Euclidean over z-scored pre-treatment means, standardised
    across the candidate pool so that no variable dominates by units.

    Returns (selected_units, distance_series).
    """
    vars_ = list(variables or STRUCTURAL_MATCH_VARS)
    cand = list(dict.fromkeys(list(pool) + [treated_unit]))

    d = panel[(panel["MONTH_DT"] < pd.Timestamp(before))
              & (panel["unit"].isin(cand))].copy()
    if d.empty:
        raise ValueError("no pre-treatment rows for the candidate pool")
    d["log_n_hh"] = np.log(d["n_hh"].clip(lower=1))

    have = [v for v in vars_ if v in d.columns]
    missing = [v for v in vars_ if v not in d.columns]
    prof = d.groupby("unit")[have].mean()

    # A unit that cannot be profiled cannot be matched; drop it rather than
    # dropping the variable, which would weaken the screen for everyone.
    bad = prof.index[prof.isna().any(axis=1)].tolist()
    prof = prof.drop(index=bad)
    if treated_unit not in prof.index:
        raise ValueError(f"{treated_unit} has no usable structural profile")

    sd = prof.std(ddof=0).replace(0, np.nan)
    z = (prof - prof.mean()) / sd
    z = z.dropna(axis=1, how="all")
    dist = np.sqrt(((z - z.loc[treated_unit]) ** 2).sum(axis=1)).drop(treated_unit)
    sel = list(dist.nsmallest(min(k, len(dist))).index)

    if verbose:
        print(f"  matched on {len(z.columns)} standardised pre-treatment "
              f"characteristics: {', '.join(z.columns)}")
        if missing:
            print(f"  (not available in panel: {', '.join(missing)})")
        if bad:
            print(f"  ({len(bad)} candidate(s) dropped for missing profile)")
        print(f"  kept {len(sel)} of {len(dist)} donors; "
              f"distance range {dist.nsmallest(len(sel)).min():.2f}-"
              f"{dist.nsmallest(len(sel)).max():.2f} "
              f"(furthest available {dist.max():.2f})")
    return sel, dist


def detect_level_breaks(panel, col="ln_inc_real", window=6, min_months=24):
    """
    Find discrete level shifts in a unit's outcome series.

    Why this exists. CPHS draws each district-month from a single primary
    sampling unit (n_psu == 1 for 248 of 294 urban units). When CMIE re-draws
    that block, the district's entire income series steps to a new level, and
    nothing else in the pipeline notices: sample size is unchanged, coverage is
    unchanged, non-response is unchanged.

    This is not hypothetical. One urban district in this panel stepped from a
    median household income of Rs 25,400 in Sep 2024 to Rs 52,000 in Oct 2024,
    then to Rs 80,500 by Jun 2025, while its wage-labourer share fell from 0.32
    to 0.06 -- a different set of households, not richer ones. Estimated
    naively, that break produced a "+54.7% effect, p = 0.028" out of nothing.

    For each unit and each candidate month, compares the mean of the following
    `window` months against the preceding `window`, and scores that step in
    units of the unit's own typical monthly movement. Returns one row per unit
    with the largest step found and when it occurred.
    """
    out = []
    for unit, g in panel.sort_values("MONTH_DT").groupby("unit"):
        y = g[col].to_numpy(float)
        dates = g["MONTH_DT"].to_numpy()
        if len(y) < max(min_months, 2 * window + 1):
            continue
        # typical monthly movement, robust to the break itself
        d = np.abs(np.diff(y))
        scale = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else np.nan
        best, best_i = 0.0, None
        for i in range(window, len(y) - window):
            step = abs(np.nanmean(y[i:i + window]) - np.nanmean(y[i - window:i]))
            if step > best:
                best, best_i = step, i
        out.append({
            "unit": unit,
            "max_step": best,
            "step_month": pd.Timestamp(dates[best_i]) if best_i is not None else pd.NaT,
            "typical_monthly_move": scale,
            "step_ratio": best / scale if scale and scale > 0 else np.inf,
        })
    r = pd.DataFrame(out)
    if r.empty:
        return r

    # Two conditions, because either alone misfires.
    #
    # Judged against PEERS, within region: rural cells are inherently far more
    # volatile (median largest step 0.61 log points vs 0.39 urban), so pooling
    # them lets rural churn crowd out real urban breaks. Comparing to peers
    # rather than an absolute cutoff also means a genuine macro shock, which
    # moves every unit together, does not trip the screen -- only a step that
    # is unusual for its cohort does.
    #
    # AND judged against the unit's OWN month-to-month movement: a large step
    # in a series that always jumps around is not informative; a large step in
    # an otherwise steady series is a discontinuity.
    r["region"] = r["unit"].str.split("|").str[2].str.strip()
    peer = r.groupby("region")["max_step"].transform(lambda x: x.quantile(0.90))
    r["suspect_break"] = (r["max_step"] >= peer) & (r["step_ratio"] >= 5.0)
    return r.sort_values("max_step", ascending=False).reset_index(drop=True)


def to_wide(panel, outcome, units=None):
    """Pivot to the units x months matrix the SCM estimator consumes."""
    d = panel if units is None else panel[panel["unit"].isin(units)]
    w = d.pivot_table(index="unit", columns="MONTH_DT", values=outcome)
    return w.dropna(axis=0, how="any").sort_index(axis=1)
