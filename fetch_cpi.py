"""
fetch_cpi.py -- build the state x sector x month CPI file the deflator needs.

Two sources, because either may be easier on a given day:

    python fetch_cpi.py --source mospi
        Pulls the General index from MoSPI's public eSankhyiki API. No login,
        no API key. Needs internet from the machine you run it on.

    python fetch_cpi.py --source cmie --infile "CPI_export.xlsx"
        Reshapes a CMIE Economic Outlook export. Use this if you already have
        EO access: Statistics -> Inflation -> Consumer Price Indices ->
        CPI Base Year: 2012 -> "CPI General Index, Groups and Sub-groups".
        Accepts wide (months across columns) or long layouts.

Either way it writes, next to your other outputs:

    cpi_state_sector_monthly.csv
        STATE,REGION_TYPE,MONTH_DT,CPI

Then set in config.py:

    CPI_CSV = OUT_DIR / "cpi_state_sector_monthly.csv"

WHAT YOU MUST GET, and why each matters:

  * STATE-level, not All-India. Synthetic control already differences out the
    inflation component common to every unit. Only the state-specific part
    changes an estimate, so an All-India series deflates to no purpose.
  * RURAL and URBAN separately. CPHS splits households this way and the panel
    joins on it. A Combined index silently mixes two different price paths.
  * GENERAL index (all groups), not Food or Fuel. You are deflating total
    household income.
  * MONTHLY. Annual figures interpolated to months invent smooth within-year
    price paths, which is worse than nominal because it looks deflated.
  * Index LEVELS (roughly 110-210 on 2012=100 over 2014-2025), not inflation
    rates. If your numbers are single digits you have the rate series.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import pandas as pd

import config as C
from panel import canonical_state

OUT_NAME = "cpi_state_sector_monthly.csv"

SECTORS = {"RURAL", "URBAN"}


# ------------------------------------------------------------------ MoSPI ----

MONTHS = {m.lower(): k for k, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _mospi_page(base_year, series, page, limit, extra=None):
    """
    One page of CPI rows, as a plain list of dicts.

    Two quirks of the client are handled here:
      * format="dict" returns a LIST of records, not the raw envelope, so
        there is no meta_data and no totalPages to page against. We page
        until a short page comes back instead.
      * it prints the entire JSON response to stdout on every call. Over
        hundreds of pages that buries the terminal, so stdout is swallowed
        for the duration of the request.
    """
    import esankhyiki
    f = {"base_year": base_year, "series": series,
         "limit": int(limit), "page": str(page)}
    if extra:
        f.update(extra)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = esankhyiki.get_data("CPI", f, format="dict")
    if isinstance(r, dict):
        r = r.get("data") or []
    return list(r or [])


def probe_mospi(base_year="2012", series="Current"):
    """Fetch one page and report what the API actually offers."""
    try:
        import esankhyiki
    except ImportError:
        sys.exit("pip install mospi-esankhyiki")
    print(f"Probing MoSPI (base {base_year}, {series}) ...")
    rows = _mospi_page(base_year, series, 1, 500)
    print(f"  page 1 returned {len(rows)} records")
    if not rows:
        sys.exit("  no rows returned")
    df = pd.DataFrame(rows)
    print(f"  columns: {list(df.columns)}")
    for col in ("state", "sector", "group", "subgroup", "year"):
        if col in df.columns:
            vals = sorted(str(v) for v in df[col].dropna().unique())
            print(f"  {col:9s} ({len(vals):3d} in this page): {vals[:12]}"
                  f"{' ...' if len(vals) > 12 else ''}")
    return df


def from_mospi(base_year="2012", series="Current", limit=1000, max_pages=None):
    """
    Pull the CPI General index, state x sector x month, from MoSPI.

    The endpoint returns every group and sub-group for every state, sector and
    month -- 326,442 rows on the 2012 base -- and the client does not page for
    you. We calibrate the real page size from the first response (the server
    may cap the requested limit) and then page until a short page arrives.
    """
    try:
        import esankhyiki  # noqa: F401
    except ImportError:
        sys.exit("pip install mospi-esankhyiki")

    print(f"MoSPI eSankhyiki: CPI base {base_year}, {series} series ...")
    first = _mospi_page(base_year, series, 1, limit)
    if not first:
        sys.exit("MoSPI returned no rows on page 1.")
    page_size = len(first)
    print(f"  page size {page_size} (asked for {limit}); "
          f"~{326442 // page_size + 1} pages expected, a few minutes")

    rows = list(first)
    page = 1
    cap = int(max_pages) if max_pages else 5000
    while page < cap:
        page += 1
        try:
            batch = _mospi_page(base_year, series, page, limit)
        except Exception as e:
            print(f"  page {page} failed ({e}); using what we have")
            break
        if not batch:
            break
        rows.extend(batch)
        if page % 20 == 0:
            print(f"  page {page}  ({len(rows):,} rows)")
        if len(batch) < page_size:      # last page
            break

    print(f"  fetched {len(rows):,} rows over {page} pages")
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("MoSPI returned no usable rows.")
    return _from_mospi_frame(df)


def _from_mospi_frame(df):
    """Keep the General index only, and reshape to the deflator's schema."""
    df = df.copy()

    # The General index is the all-groups total. It is labelled in the group or
    # subgroup column; everything else (Food, Fuel, Housing, Miscellaneous and
    # their children) is a component and must go.
    def _is_general(v):
        return bool(re.fullmatch(r"\s*general(\s*index)?(-overall)?\s*",
                                 str(v), flags=re.I))

    mask = pd.Series(False, index=df.index)
    for col in ("subgroup", "group"):
        if col in df.columns:
            mask |= df[col].map(_is_general)
    if not mask.any():
        groups = sorted({str(v) for c in ("group", "subgroup") if c in df.columns
                         for v in df[c].dropna().unique()})
        sys.exit("Could not find a 'General' row. Values seen:\n  "
                 + "\n  ".join(groups[:40]))
    df = df[mask]
    print(f"  {len(df):,} General-index rows")

    out = pd.DataFrame({
        "STATE": df["state"],
        "REGION_TYPE": df["sector"].astype(str).str.strip().str.upper(),
        "CPI": pd.to_numeric(df["index"], errors="coerce"),
    })
    mnum = df["month"].astype(str).str.strip().str.lower().map(MONTHS)
    out["MONTH_DT"] = pd.to_datetime(
        dict(year=pd.to_numeric(df["year"], errors="coerce"),
             month=mnum, day=1), errors="coerce")

    # All India carries no cross-state variation, and SCM weights sum to 1, so
    # a nationally uniform deflator cancels exactly out of the treated-minus-
    # synthetic gap. Dropping it is not a preference, it is arithmetic.
    n_ai = int(out["STATE"].astype(str).str.strip().str.lower()
                  .eq("all india").sum())
    out = out[~out["STATE"].astype(str).str.strip().str.lower().eq("all india")]
    if n_ai:
        print(f"  dropped {n_ai:,} All-India rows (no cross-state variation)")

    out["STATE"] = out["STATE"].map(canonical_state)
    out = out.dropna(subset=["STATE", "REGION_TYPE", "MONTH_DT", "CPI"])
    dropped = sorted(set(out["REGION_TYPE"]) - SECTORS)
    if dropped:
        print(f"  dropping non-Rural/Urban sectors: {dropped}")
    out = out[out["REGION_TYPE"].isin(SECTORS)]
    out = (out.drop_duplicates(["STATE", "REGION_TYPE", "MONTH_DT"])
              .sort_values(["STATE", "REGION_TYPE", "MONTH_DT"])
              .reset_index(drop=True))
    return out[["STATE", "REGION_TYPE", "MONTH_DT", "CPI"]]


# ------------------------------------------------------------------- CMIE ----

def from_cmie(infile):
    """
    Reshape a CMIE Economic Outlook export into the four-column schema.

    EO usually exports wide: one row per state (or per state-sector) and one
    column per month. Long exports work too. Sector may be its own column or
    baked into the row label ("Uttar Pradesh - Urban").
    """
    path = Path(infile)
    if not path.exists():
        sys.exit(f"No such file: {path}")
    df = (pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"}
          else pd.read_csv(path))
    print(f"  read {path.name}: {df.shape[0]} rows x {df.shape[1]} cols")

    cols = {c.strip().lower(): c for c in df.columns}
    has_long = any(k in cols for k in ("month", "month_dt", "date", "period"))

    if not has_long:
        # Wide: everything that parses as a date becomes a month column.
        id_cols, month_cols = [], []
        for c in df.columns:
            if pd.notna(pd.to_datetime(str(c), errors="coerce",
                                       dayfirst=False)):
                month_cols.append(c)
            else:
                id_cols.append(c)
        if not month_cols:
            sys.exit("Could not find month columns. Expected either a long "
                     "layout with a month/date column, or a wide one with "
                     "parseable dates as headers.")
        print(f"  wide layout: {len(id_cols)} id col(s), "
              f"{len(month_cols)} month col(s)")
        df = df.melt(id_vars=id_cols, value_vars=month_cols,
                     var_name="MONTH_DT", value_name="CPI")

    return _normalise(df)


# -------------------------------------------------------------- normalise ----

def _pick(df, *candidates):
    low = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def _normalise(df):
    """Map whatever came back onto STATE, REGION_TYPE, MONTH_DT, CPI."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    st = _pick(df, "state", "state_name", "statename", "region", "name")
    sec = _pick(df, "region_type", "sector", "sector_name", "sectorname")
    mon = _pick(df, "month_dt", "month", "date", "period")
    val = _pick(df, "cpi", "index", "value", "general", "general_index")

    if st is None or val is None:
        sys.exit(f"Could not identify state/value columns in {list(df.columns)}. "
                 "Rename them to STATE and CPI and re-run.")

    out = pd.DataFrame({
        "STATE": df[st],
        "MONTH_DT": pd.to_datetime(df[mon] if mon else df.get("MONTH_DT"),
                                   errors="coerce"),
        "CPI": pd.to_numeric(df[val], errors="coerce"),
    })

    if sec is not None:
        out["REGION_TYPE"] = df[sec].astype(str).str.strip().str.upper()
    else:
        # Sector baked into the label, e.g. "Uttar Pradesh - Urban".
        lab = out["STATE"].astype(str)
        out["REGION_TYPE"] = (
            lab.str.extract(r"(?i)\b(rural|urban|combined)\b")[0]
               .str.upper())
        out["STATE"] = (lab.str.replace(r"(?i)[\s\-:,]*\b(rural|urban|combined)"
                                        r"\b", "", regex=True).str.strip())

    out["STATE"] = out["STATE"].map(canonical_state)
    out = out.dropna(subset=["STATE", "REGION_TYPE", "MONTH_DT", "CPI"])

    dropped = out.loc[~out["REGION_TYPE"].isin(SECTORS), "REGION_TYPE"].unique()
    if len(dropped):
        print(f"  dropping non-Rural/Urban sectors: {sorted(dropped)}")
    out = out[out["REGION_TYPE"].isin(SECTORS)]

    # Month start, so it joins against the panel's MONTH_DT.
    out["MONTH_DT"] = out["MONTH_DT"].values.astype("datetime64[M]")
    out = (out.drop_duplicates(["STATE", "REGION_TYPE", "MONTH_DT"])
              .sort_values(["STATE", "REGION_TYPE", "MONTH_DT"])
              .reset_index(drop=True))
    return out[["STATE", "REGION_TYPE", "MONTH_DT", "CPI"]]


# ---------------------------------------------------------------- validate ----

def validate(cpi):
    """Refuse to write something that will quietly ruin the estimates."""
    problems, notes = [], []

    if cpi.empty:
        problems.append("no usable rows")
        return problems, notes

    med = cpi["CPI"].median()
    if med < 20:
        problems.append(
            f"median value is {med:.1f}. That looks like an INFLATION RATE, "
            "not an index level. You need index levels (~110-210 on 2012=100).")

    for sector in SECTORS:
        if sector not in set(cpi["REGION_TYPE"]):
            problems.append(f"no {sector} rows")

    span = cpi.groupby(["STATE", "REGION_TYPE"])["MONTH_DT"]
    n_months = span.nunique()
    ragged = n_months[n_months < n_months.max()]
    if len(ragged):
        notes.append(f"{len(ragged)} state-sector series are shorter than the "
                     f"longest ({n_months.max()} months). Units whose series "
                     f"has a hole get DROPPED from the panel, not deflated.")

    lo, hi = cpi["MONTH_DT"].min(), cpi["MONTH_DT"].max()
    notes.append(f"coverage {lo:%b %Y} to {hi:%b %Y}; "
                 f"{cpi['STATE'].nunique()} states; {len(cpi):,} rows")
    if hi < pd.Timestamp("2025-12-01"):
        notes.append(
            f"WARNING: series ends {hi:%b %Y}, before the panel's last month "
            "(Dec 2025).\n    MoSPI rebased CPI to 2024=100, so the 2012-base "
            "series may stop early.\n    Splicing the two bases lands a "
            "discontinuity inside the post-treatment\n    window -- resolve "
            "this before trusting the headline.")
    return problems, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["mospi", "cmie"], default="mospi")
    ap.add_argument("--infile", help="CMIE export (.xlsx/.csv), for --source cmie")
    ap.add_argument("--base-year", default="2012")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="fetch one page and print the API's actual shape")
    ap.add_argument("--limit", type=int, default=1000,
                    help="records per request (default 1000)")
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    if args.probe:
        probe_mospi(base_year=args.base_year)
        return

    if args.source == "cmie":
        if not args.infile:
            sys.exit("--source cmie needs --infile")
        cpi = from_cmie(args.infile)
    else:
        cpi = from_mospi(base_year=args.base_year, limit=args.limit,
                         max_pages=args.max_pages)

    problems, notes = validate(cpi)
    for n in notes:
        print(f"  note: {n}")
    if problems:
        print("\nNOT WRITING -- fix these first:")
        for p in problems:
            print(f"  * {p}")
        sys.exit(1)

    out = Path(args.out) if args.out else C.OUT_DIR / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    cpi.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(cpi):,} rows)")
    print(f'Now set in config.py:  CPI_CSV = OUT_DIR / "{OUT_NAME}"')
    print("Then rebuild:          python run_01_build_panel.py")


if __name__ == "__main__":
    main()
