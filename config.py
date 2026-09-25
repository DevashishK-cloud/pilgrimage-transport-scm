"""
config.py: every path, date and tuning knob for the project lives here.
Edit this file; do not hard-code paths anywhere else.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------- PATHS ----
# Raw CMIE waves. Expected layout:
#   CMIE_Income_raw/2014/household_income_20140131_MS_rev.csv
#   CMIE_Income_raw/2014/household_income_20140228_MS_rev.csv
#   ...
RAW_DIR = Path(os.environ.get(
    "CMIE_RAW_DIR",
    r"C:\Users\DEVASHISH\Documents\Cowork Playground\CMIE_Income_raw"))

# Everything this project writes goes here.
OUT_DIR = Path(os.environ.get(
    "CMIE_OUT_DIR", str(Path(__file__).resolve().parent / "output")))

FILE_GLOB = "**/household_income_*_MS_rev.csv"

PANEL_PARQUET = OUT_DIR / "panel_district_month.parquet"
AUDIT_CSV     = OUT_DIR / "audit_by_month.csv"
SCREEN_CSV    = OUT_DIR / "donor_screen.csv"
FIG_DIR       = OUT_DIR / "figures"
RESULT_DIR    = OUT_DIR / "results"

# Optional CPI deflator. See README for the exact schema.
# Set to None to run in nominal terms (not recommended for the headline result).
# State x sector x month CPI, General index, base 2012=100.
# Built by:  python fetch_cpi.py --source mospi
# Source: MoSPI eSankhyiki, https://api.mospi.gov.in/api/cpi/getCPIIndex
# Coverage as fetched 2 Sep 2026: Jan 2013 - Dec 2025, 36 states, 10,929 rows.
CPI_CSV = OUT_DIR / "cpi_state_sector_monthly.csv"

# ------------------------------------------------------------ TREATMENTS ----
# Treatment dates are the FIRST treated month.
#
# ACTIVE STUDY: pilgrimage transport infrastructure. The treated units are
# transport built to move pilgrims, dated to the first month of scheduled
# service rather than to the ribbon-cutting, which can precede it by weeks.
#
# On anticipation. A project that demolishes and rebuilds a town centre starts
# affecting incomes years before it opens, which puts construction inside the
# "pre-treatment" window and manufactures a rebound. That mechanism is
# displacement of local commerce, and it does not apply here: an airport
# terminal or a border check post is built on open land outside the town and
# clears no shops. So anticipation is left unset and the pre-window runs to
# the month scheduled service began. Run the 12-month-anticipation variant as
# a robustness check; if it moves the estimate, say so.
TREATMENTS = {
    "Shirdi": {
        "unit":  "Maharashtra | Ahilyanagar | URBAN",
        "date":  "2017-10-01",              # commercial ops from 1 Oct 2017
        "anticipation": None,               # see note above
        "exclude_states": ["Maharashtra"],  # spillover donut
    },
    # Same case, spillover donut removed. This is a CONFOUND TEST, not a
    # second result.
    #
    # Excluding the whole state is the right call where corridor demolition
    # plausibly moves activity into neighbouring districts. It is the wrong
    # call here: an airport terminal on the edge of town displaces
    # nobody, and dropping Maharashtra removes every district that shared the
    # 2018-19 western Maharashtra drought. With the donut in place the
    # counterfactual ended up 55% weighted on Barddhaman, West Bengal, and
    # returned -24% to -42% -- far too large for a regional airport, and
    # exactly what a state-level agrarian shock would look like.
    #
    # If the effect collapses with Maharashtra donors admitted, it was the
    # drought. If it survives, the airport story is worth taking seriously.
    "ShirdiOpenPool": {
        "unit":  "Maharashtra | Ahilyanagar | URBAN",
        "date":  "2017-10-01",
        "anticipation": None,
        "exclude_states": [],
    },
    # The STRONG confound test. ShirdiOpenPool merely ADMITTED Maharashtra
    # donors; the simplex then gave them 4.8% of the weight and the estimate
    # did not move (-0.274 vs -0.278), so that test decided nothing. This one
    # forces the counterfactual to be built ONLY from other Maharashtra
    # cities, which all shared the 2018-19 western Maharashtra drought.
    #
    # If Ahilyanagar still falls ~24% against its own neighbours, a
    # state-wide agrarian shock cannot explain it and the airport story is
    # worth taking seriously. If the gap collapses, it was the drought and
    # the transport study has no estimable case.
    #
    # Caveat, stated in advance: the Maharashtra-only pool is small, and a
    # simplex over few donors fits easily. Read this as a direction, not a
    # p-value. run_02_scm.py warns when the pool drops below five.
    "ShirdiMHOnly": {
        "unit":  "Maharashtra | Ahilyanagar | URBAN",
        "date":  "2017-10-01",
        "anticipation": None,
        "exclude_states": [],
        "only_states": ["Maharashtra"],
    },
    "Prayagraj": {
        "unit":  "Uttar Pradesh | Prayagraj | URBAN",
        "date":  "2018-12-01",              # new terminal opened 16 Dec 2018
        "anticipation": None,
        "exclude_states": ["Uttar Pradesh"],
    },
    "Kartarpur": {
        # The crossing is at Dera Baba Nanak, a border village in Gurdaspur,
        # not in Gurdaspur town. RURAL is the correct sector, not a fallback.
        "unit":  "Punjab | Gurdaspur | RURAL",
        "date":  "2019-11-01",              # corridor opened 9 Nov 2019
        "anticipation": None,
        "exclude_states": ["Punjab"],
    },
}

# Set to the confound test so `python run_02_scm.py` (no --all) runs only
# that one case. Shirdi's donut results are already in output/results/.
PRIMARY_CASE = "ShirdiMHOnly"

# Placebo-in-time needs a fake treatment date in a CLEAN pre-period: before
# any anticipation effect, and outside the COVID measurement break.
#
# A mechanical rule (t0 minus 3 years) once put a fake date inside the COVID
# window AND four months into a construction period, and returned a large
# wrong-signed effect. That looked like catastrophic failure but was really the
# test being pointed at months that were already treated and badly measured. A
# placebo date must sit where the unit is genuinely untreated and well
# measured, so each one below is chosen by hand.
#
# For the transport cases each fake date sits well before the real one and
# well before Mar 2020, so no placebo window straddles the COVID break.
PLACEBO_TIME_DATE = {
    "Shirdi":    "2016-01-01",  # 24 pre / 21 post, all before Oct 2017
    "ShirdiOpenPool": "2016-01-01",
    "ShirdiMHOnly":   "2016-01-01",
    "Prayagraj": "2016-06-01",  # 29 pre / 30 post, all before Dec 2018
    "Kartarpur": "2017-01-01",  # 36 pre / 34 post, ends before COVID
}

# ------------------------------------------------------------- COLUMNS ----
USECOLS = [
    "HH_ID", "STATE", "HR", "DISTRICT", "REGION_TYPE", "STRATUM", "PSU_ID",
    "MONTH", "RESPONSE_STATUS",
    # CMIE dropped the unadjusted HH_WGT_MS from the Apr-2023 wave onward.
    # R_HH_WGT_MS (non-response adjusted) is present in ALL 144 waves and is
    # the correct weight here anyway, since we keep only Accepted households.
    # Do not fall back per-file: that would change the weight definition in
    # Apr 2023, in the middle of the panel and close to treatment dates.
    "R_HH_WGT_MS", "R_HH_WGT_FOR_COUNTRY_MS",
    "SIZE_GROUP", "OCCUPATION_GROUP", "EDU_GROUP",
    "TOT_INC",
    # All 14 CPHS income components, not 5. See INCOME_GROUPS below.
    "INC_OF_ALL_MEMS_FRM_WAGES",
    "INC_OF_ALL_MEMS_FRM_PENSION",
    "INC_OF_ALL_MEMS_FRM_DIVIDEND",
    "INC_OF_ALL_MEMS_FRM_INTEREST",
    "INC_OF_ALL_MEMS_FRM_FD_PF_INS",
    "INC_OF_HH_FRM_RENT",
    "INC_OF_HH_FRM_SELF_PRODN",
    "INC_OF_HH_FRM_PVT_TRF",
    "INC_OF_HH_FRM_GOVT_TRF",
    "INC_OF_HH_FRM_IN_KIND_GOVT_TRF",
    "INC_OF_HH_FRM_NGO_TRF",
    "INC_OF_HH_FRM_BIZ_PROFIT",
    "INC_OF_HH_FRM_ASSET_SALE",
    "INC_OF_HH_FRM_GAMBLING",
]

INCOME_SOURCES = [
    "INC_OF_ALL_MEMS_FRM_WAGES",
    "INC_OF_ALL_MEMS_FRM_PENSION",
    "INC_OF_ALL_MEMS_FRM_DIVIDEND",
    "INC_OF_ALL_MEMS_FRM_INTEREST",
    "INC_OF_ALL_MEMS_FRM_FD_PF_INS",
    "INC_OF_HH_FRM_RENT",
    "INC_OF_HH_FRM_SELF_PRODN",
    "INC_OF_HH_FRM_PVT_TRF",
    "INC_OF_HH_FRM_GOVT_TRF",
    "INC_OF_HH_FRM_IN_KIND_GOVT_TRF",
    "INC_OF_HH_FRM_NGO_TRF",
    "INC_OF_HH_FRM_BIZ_PROFIT",
    "INC_OF_HH_FRM_ASSET_SALE",
    "INC_OF_HH_FRM_GAMBLING",
]

# The mechanism decomposition runs on these GROUPS, not on all 14 columns.
#
# The earlier version used five components and called the result a
# decomposition. It was not a partition: the five covered roughly 77% of the
# income effect, and the missing 23% went unlabelled.
#
# The serious omission was GOVERNMENT transfers. The README's confound test
# says an effect driven by transfers is "remittances and government payments,
# not local demand" -- but only PRIVATE transfers were measured, so the test
# could not detect the government half at all. For a state-led project in a
# politically salient district, state transfers are among the most plausible
# confounds there is. Grouping them makes that channel visible and keeps the
# decomposition to nine interpretable series instead of fourteen.
INCOME_GROUPS = {
    "Wages":               ["INC_OF_ALL_MEMS_FRM_WAGES"],
    "Business profit":     ["INC_OF_HH_FRM_BIZ_PROFIT"],
    "Rent":                ["INC_OF_HH_FRM_RENT"],
    "Self-production":     ["INC_OF_HH_FRM_SELF_PRODN"],
    "Private transfers":   ["INC_OF_HH_FRM_PVT_TRF"],
    "Government transfers": ["INC_OF_HH_FRM_GOVT_TRF",
                             "INC_OF_HH_FRM_IN_KIND_GOVT_TRF"],
    "Pension":             ["INC_OF_ALL_MEMS_FRM_PENSION"],
    "Financial income":    ["INC_OF_ALL_MEMS_FRM_DIVIDEND",
                            "INC_OF_ALL_MEMS_FRM_INTEREST",
                            "INC_OF_ALL_MEMS_FRM_FD_PF_INS",
                            "INC_OF_HH_FRM_ASSET_SALE"],
    "Other":               ["INC_OF_HH_FRM_NGO_TRF",
                            "INC_OF_HH_FRM_GAMBLING"],
}

SOURCE_LABELS = {c: c for c in INCOME_SOURCES}

WEIGHT_COL = "R_HH_WGT_MS"

# --------------------------------------------------------------- TUNING ----
WINSOR_LO, WINSOR_HI = 0.01, 0.99   # trim CPHS income tails
MIN_HH_PER_CELL = 60                # a district-month below this is unusable
MIN_COVERAGE_FRAC = 1.0             # donors must be present in 100% of months
SMOOTH_WINDOW = 3                   # centred moving average, months

# CPHS moved to telephone interviews during COVID. This is a MEASUREMENT
# break, not an economic one. Months in this window get flagged.
# ---- structurally matched donor pool (robustness check) -------------------
# Pre-specified 2 Sep 2026, before any matched-pool estimate was computed.
# The donor pool is screened to the K units closest to the treated unit on
# pre-treatment economic structure; the estimator itself is unchanged.
STRUCT_MATCH_K = 30

# Worker processes for the placebo refits (the dominant cost of a run).
# None = use every core. Set to 1 to debug serially.
N_JOBS = None

COVID_START = "2020-03-01"
COVID_END   = "2021-06-01"
