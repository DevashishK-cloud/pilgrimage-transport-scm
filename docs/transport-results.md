# Pilgrimage Transport Infrastructure and Household Income

### Synthetic control evidence from CPHS, 2014-2025

**Date:** 2026-09-05
**Status:** Three cases estimated. Two withdrawn on falsification evidence. One survives, with a thin counterfactual.

---

## Headline

> Across three pilgrimage transport projects, there is **no evidence that
> improved access to a religious site raised local household income**. Two of
> the three estimates fail their own placebo test and are withdrawn. The third,
> Shirdi, shows household income **21% to 35% below synthetic control on
> average**, but that average hides a gap that opens, peaks in 2022 and has
> closed by 2025. It rests on a counterfactual built from five donors, one of
> which carries half the weight, and a state-level confound that cannot be
> fully excluded.

The result is a null on the question as posed, plus a large negative
association in the one case that survives falsification, which this document
does not claim is causal.

---

## Question

Does public investment in transport built to move pilgrims raise income in the
district it serves?

The treated object is a government capital project with a dated opening. The
site selection was made by the Ministry of Civil Aviation, the Land Ports
Authority and state governments, not by the analyst. Faith enters only as a
description of what each project serves.

Transport is chosen over the funded sites themselves for a reason. Rebuilding
a pilgrimage site clears the shops around it, so any estimate confounds
construction disruption with visitor arrivals. An airport terminal or a border
check post goes up on open land outside the town, displaces nobody, and
delivers visitors directly. That separation is what makes transport the
cleaner treatment.

---

## Data

| | |
|---|---|
| Source | CMIE Consumer Pyramids Household Survey, 144 monthly waves |
| Span | January 2014 to December 2025 |
| Analysis panel | 842 district-sector units, 97,697 unit-months, 142 months |
| Sampling units the panel is built from | at least 4,725 CPHS PSUs, across 102 homogeneous regions |
| Urban units | 292 |
| Deflator | MoSPI CPI, General index, state x sector x month, base 2012=100 |

Two months, April and May 2020, are absent for every unit. MoSPI published no
CPI during the lockdown, and those months are dropped rather than interpolated
so that treated units and donors stay on identical calendars. Two units in
Puducherry are dropped for incomplete CPI coverage.

### A district rename that would have hidden the only estimable case

CPHS relabelled Ahmednagar as Ahilyanagar in the August 2025 wave. The panel
therefore carried `Ahmadnagar` from January 2014 to July 2025 and
`Ahilyanagar` from August to December 2025 as two separate units. Both then
failed the 100% coverage screen and were dropped.

Shirdi's airport is in that district. Without the alias the case is not
estimable at all, and the screener reports it as a coverage failure rather
than as a naming problem. `DISTRICT_ALIASES` now folds all three spellings to
one label. Merging them removed exactly two units from the panel, 844 to 842,
one urban and one rural, which is the arithmetic signature of a single
district rename.

This is the second instance of the same failure mode in this project, after
a documented rename already carried in the alias table, and the first one
found by looking rather than by knowing in advance.

---

## Treatment selection

Twelve candidate sites were screened before any estimation. The screen tests
whether a district exists in CPHS at all, whether its series is complete,
whether there are 24 clean months either side of the opening, and whether a
suspect level break falls inside the post window.

| | Screened | Estimable |
|---|---|---|
| Pilgrimage transport | 6 | **3** |
| Non-pilgrimage regional airports (intended control arm) | 4 | **0** |
| Ambiguous (Bareilly, itself a pilgrimage centre) | 1 | 0 |
| Excluded by design (see below) | 1 | n/a |

**The control arm does not exist.** Kannur, Jharsuguda, Kalaburagi and
Darbhanga all fail on coverage. The study therefore cannot separate "the
effect of an airport" from "the effect of a pilgrimage airport". That
comparison was the original motivation for the transport design and it is not
available on this survey.

Three pilgrimage sites survived and were estimated.

| Site | Unit | Opened |
|---|---|---|
| Shirdi Airport | Maharashtra \| Ahilyanagar \| URBAN | Oct 2017 |
| Prayagraj Airport terminal | Uttar Pradesh \| Prayagraj \| URBAN | Dec 2018 |
| Kartarpur Corridor ICP | Punjab \| Gurdaspur \| RURAL | Nov 2019 |

Dates are the first month of scheduled service, not the ribbon-cutting.
Kushinagar was inaugurated on 20 October 2021 but its first scheduled flight
was 26 November 2021; that distinction matters and it is applied throughout.

One further airport is excluded on purpose. It opened 23 days before a major
religious site in the same district, so the two treatments fall in the same
month and cannot be separated.

### Specification

- **Outcome:** log real mean household income, 3-month centred moving average
- **Estimator:** ridge-augmented SCM (Ben-Michael, Feller & Rothstein 2021),
  with simplex-constrained and de-meaned variants reported alongside
- **Penalty:** rolling-origin cross-validation on pre-treatment months only,
  held fixed across the treated unit and every placebo
- **Inference:** permutation over placebo units, two p-values reported
- **Anticipation:** none. A project that demolishes and rebuilds a town centre
  affects incomes years before it opens, which puts construction inside the
  pre-treatment window and manufactures a rebound. An airport clears no shops,
  so that contamination does not apply and the pre-window runs to the month
  scheduled service began.

---

## Results

### Two cases fail their own placebo test

The placebo-in-time test moves the treatment date into a window where nothing
happened. If the model still reports an effect, it is measuring a pre-existing
trend.

| Case | Real ATT | Placebo ATT | Ratio |
|---|---|---|---|
| Shirdi | -0.2784 | **-0.0509** | 5.5x |
| Prayagraj | -0.2067 | **-0.2123** | 1.0x |
| Kartarpur | -0.1022 | **-0.2127** | 0.5x |

**Prayagraj is withdrawn.** The model finds -21% in a clean window and -21%
after the terminal opened. Prayagraj was already drifting below its synthetic
control. The headline is trend, not treatment.

**Kartarpur should never have been estimated at all**, and the placebo test is
the second reason it is withdrawn rather than the first.

Its treated unit is `Punjab | Gurdaspur | RURAL`. `build_donor_pool` admits
only urban units, by design and by its own docstring: the pool compares cities
to cities. So a rural district was matched against 85 urban ones. That
mismatch, not the corridor, is the most likely explanation for its numbers:
pre-treatment RMSPE 0.1252 against roughly 0.03 for the two urban cases, an
empirical design effect of 6.3x against 1.2x and 1.8x, and a counterfactual
assembled from Jabalpur, Hazaribagh and Gwalior city.

The specification was invalid before any result was computed. It is reported
here rather than deleted because the estimate was run, and a withdrawn case
with a stated reason is more useful to a reader than a case that quietly
disappears.

Its placebo is also double its headline, and its pre-treatment fit is 0.43x the
noise floor, meaning the model is fitting inside the measurement error. Its
permutation p-value is 0.895. Any one of these disqualifies it.

Neither withdrawal depends on a p-value. Both are decided by a falsification
test the specification failed.

### Shirdi: the surviving case

| Specification | ATT | Notes |
|---|---|---|
| de-meaned | -0.244 | |
| simplex SCM | -0.259 | |
| drop-COVID (14 months removed) | -0.269 | |
| **augsynth, donor pool excluding Maharashtra** | **-0.278** | headline |
| augsynth, Maharashtra admitted | -0.274 | 89 donors |
| leave-one-out range | -0.250 to -0.288 | n = 5 |
| structurally matched pool (K = 30) | -0.424 | fit 1.68x noise |

Pre-treatment RMSPE 0.0297, or 0.90x the empirical noise floor. RMSPE ratio
11.04. p = 0.138 on the RMSPE ratio, p = 0.075 on the ATT. **No specification
reaches p < 0.05.**

The band across every specification is -0.24 to -0.42 log points, roughly
-21% to -35% in income.

### The gap is not permanent, and that matters

A single averaged ATT reads as though income fell and stayed down. It did not.
Mean gap by calendar year:

| Year | Gap (log points) | In income |
|---|---|---|
| 2018 | -0.224 | -20.1% |
| 2019 | -0.423 | -34.5% |
| 2020 | -0.374 | -31.2% |
| 2021 | -0.345 | -29.2% |
| 2022 | **-0.500** | **-39.3%** |
| 2023 | -0.292 | -25.4% |
| 2024 | -0.123 | -11.6% |
| 2025 | -0.030 | -3.0% |

Over the final six months the gap is -0.007 log points, which is zero for any
practical purpose. `output/figures/Shirdi_paths.png` shows this directly: the
two lines separate through 2018, stay apart for six years, and meet again at
the right-hand edge.

The headline -0.278 is the average of that profile, not a standing level.

**This weakens the causal reading rather than supporting it.** A new airport is
a permanent change to a district, so an effect caused by one should persist. A
gap that opens in 2018, deepens through the drought years and COVID, and then
closes as those pass looks far more like a transitory regional shock that
resolved. Read alongside the section below on the Maharashtra drought, the time
profile is the strongest single argument against attributing this to the
airport.

### Distribution

| Percentile | Effect | p |
|---|---|---|
| p25 | -11.0% | 0.512 |
| p50 | -25.7% | 0.100 |
| **p75** | **-42.6%** | **0.037** |

Monotonic in income, with the loss concentrated at the top. This is the same
pattern found wherever a large public project disrupts a town centre: the
households with the most to lose commercially lose the most.

---

## Why the Shirdi estimate is not a causal claim

### The magnitude is not plausible for the shock

A regional airport does not cut household income in its district by a quarter.
When a design returns a number that large from a shock that small, the number
is usually measuring something else. Ahmednagar is a drought-prone
agricultural district in western Maharashtra, and 2018-19 was a severe drought
across the region.

### Two attempts to test that, and why neither settles it

**Weak test: admit Maharashtra donors.** The estimate moved from -0.2784 to
-0.2737. But the simplex, offered ten Maharashtra districts, assigned them 4.8%
of the weight in total and still put 52% on Barddhaman in West Bengal. Making
donors eligible does not force the model to use them, so this test decided
little.

**Strong test: restrict donors to Maharashtra only.** This specification
collapses. Ten donors, all weight on Kolhapur alone, pre-treatment RMSPE 5.26x
the noise floor, and a placebo-in-time of **+0.1534**, wrong-signed and large.
The estimator itself warns that the treated unit lies outside the donor convex
hull. Nothing can be concluded from it.

### What the two failures agree on

No city in Maharashtra resembles pre-2017 Ahilyanagar. Offered them freely,
the optimiser declined; forced to use only them, it cannot fit.

That cuts against the drought explanation rather than for it. A state-wide
shock would require Ahilyanagar to have tracked its neighbours before 2017 and
diverged after. It never tracked them.

It also limits the headline. A district with no natural comparison group has a
thin counterfactual whichever pool is used, and half of Shirdi's rests on one
district 1,500 km away.

---

## Limitations

1. **No control arm.** Every non-pilgrimage regional airport failed the
   coverage screen. The pilgrimage component cannot be separated from the
   airport component.
2. **One primary sampling unit per district.** `n_psu = 1` for the treated
   units. A shock to one neighbourhood cannot be distinguished from a
   district-wide effect.
3. **Five donors, one at 52%.** The counterfactual is concentrated.
4. **The effect is not persistent.** The gap has closed by 2025, which is hard
   to reconcile with a permanent change to the district and easy to reconcile
   with a transitory shock.
5. **Convex hull.** Ahilyanagar is not spanned by the units available to
   reproduce it, most acutely within its own state.
6. **A state-level agrarian shock cannot be fully excluded**, for the reasons
   above.
7. **Treatment intensity is never measured.** AAI publishes monthly passenger
   traffic by airport. Until that is joined in, this study cannot say whether
   the visitors actually arrived, and "the visitors came and incomes did not
   move" is a far stronger sentence than "incomes did not move."
8. **Income composition is not interpretable here.** The channel shares sum to
   -3.13 percentage points instead of zero, which means the per-channel
   counterfactuals disagree. No single channel should be read.
9. **The donor pool is urban-only by construction.** `build_donor_pool` drops
   rural strata so that cities are compared to cities. A rural treated unit
   therefore cannot currently be estimated at all, which is what invalidated
   Kartarpur. Any future rural site needs a rural donor pool, which is a change
   to the pool builder and not a change to the estimator.

CPHS sampling is itself contested. See Drèze & Somanchi (2021) and Pais &
Rawal (2021).

---

## A robustness check that had never run

`run_02_scm.py` computed `t_break` in section 4 but first referenced it in
section 3c, roughly 230 lines earlier. Python treats it as a function-local, so
section 3c raised `UnboundLocalError` on every case, a bare `except` swallowed
it, and the run printed `matched-pool check failed` and continued.

The structurally matched donor pool check had therefore **never executed**, on
any case in this project. It is fixed,
and the check now runs; for Shirdi it returns -0.424 against a headline of
-0.278, which is a 52% disagreement on magnitude that the project did not
previously know about.

This is the second robustness check in this project found to be silently dead,
after `leave_one_out` was passed a duplicate keyword argument and returned an
empty dict inside a bare `except`. Both were invisible in the output. The
lesson is recorded here rather than quietly patched: a robustness check that
cannot fail loudly is not a robustness check.

---

## What would move this forward

**AAI monthly passenger traffic**, by airport, public and small. It converts a
null into a first stage and is the single highest-value addition.

**Night lights (VIIRS via SHRUG)** as a second outcome. They cover every
district, including the nine sites that failed the CPHS coverage screen, and
would restore the control arm this design lost.

**A rural donor pool.** `build_donor_pool` admits urban units only, which is
what invalidated the Kartarpur case. Adding a rural pool would make rural
treated units estimable and is a change to the pool builder rather than to the
estimator.
