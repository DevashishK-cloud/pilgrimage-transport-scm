"""
run_02_scm.py
The analysis. For each treated case:

  1. Headline SCM on log real household income
  2. Placebo-in-space plot + permutation p-values (two tests)
  3. Placebo-in-time and leave-one-out robustness
  4. Anticipation test -- does the gap open at the announcement or the opening?
  5. Income-source decomposition (the mechanism)
  6. Distributional SCM at p25 / p50 / p75 (who gained)

    python run_02_scm.py                 # primary case only
    python run_02_scm.py --all           # all three cases
"""

import argparse
import os
import sys
from pathlib import Path

# Pin BLAS to one thread BEFORE numpy loads. Two reasons: the placebo pool
# sets the same pin in its workers, so serial and parallel runs then follow an
# identical floating-point path and give identical p-values; and threaded BLAS
# is pure overhead on matrices this small.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from panel import to_wide, structural_neighbours
from scm import fit_scm, placebo_in_time, leave_one_out, trim_placebos

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

TREAT_C, SYNTH_C = "#c0392b", "#2c3e50"


# --------------------------------------------------------------- donors ----

def build_donor_pool(panel, screen, spec):
    """
    Eligible units, minus:
      * anything in the same state (spillover donut -- tourists, migrants and
        state policy cross district lines, so neighbours are contaminated)
      * rural strata (we compare cities to cities)
      * the other treated units in this study
    """
    eligible = set(screen.loc[screen["eligible"], "unit"])
    other_treated = {s["unit"] for k, s in C.TREATMENTS.items()}

    # only_states: the inverse of the donut. Restricts the pool to the named
    # states, which is how you test whether an estimate is really a
    # state-level shock. Removing the donut is a WEAK version of that test:
    # the simplex is free to ignore the newly admitted donors, and for Shirdi
    # it did, assigning Maharashtra just 4.8% of the weight and leaving the
    # estimate at -0.274 against -0.278. Forcing the counterfactual to be
    # built ONLY from the treated unit's own state is the strong version. If
    # the district still diverges from its own neighbours, no state-wide
    # shock explains it.
    only = spec.get("only_states")

    pool = []
    for u in eligible:
        state, district, region = [x.strip() for x in u.split("|")]
        if region != "URBAN":
            continue
        if only and state not in only:
            continue
        if state in spec["exclude_states"]:
            continue
        if u in other_treated:
            continue
        pool.append(u)
    return sorted(pool)


# ---------------------------------------------------------------- plots ----

def plot_paths(res, title, path, t0_label="Treatment"):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    d = res.dates
    ax.plot(d, res.y_treated, color=TREAT_C, lw=1.8, label=res.treated.split("|")[1].strip())
    ax.plot(d, res.y_synth, color=SYNTH_C, lw=1.5, ls="--", label="Synthetic control")
    ax.axvline(d[res.t0_index], color="k", lw=0.9, ls=":")
    ax.annotate(t0_label, xy=(d[res.t0_index], ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", fontsize=8)
    ax.axvspan(pd.Timestamp(C.COVID_START), pd.Timestamp(C.COVID_END),
               color="grey", alpha=0.12, lw=0)
    ax.set_ylabel("log real monthly household income")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_placebos(res, title, path, max_ratio=2.0):
    tr = trim_placebos(res, max_ratio)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    d = res.dates
    for g in tr.placebo_gaps.values():
        ax.plot(d, g, color="grey", alpha=0.28, lw=0.6)
    ax.plot(d, res.gap, color=TREAT_C, lw=2.0, label="Treated")
    ax.axvline(d[res.t0_index], color="k", lw=0.9, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("gap: actual - synthetic (log points)")
    ax.set_title(f"{title}\n{len(tr.placebo_gaps)} placebo units "
                 f"(pre-RMSPE <= {max_ratio}x treated)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_sources(rows, title, path, xlabel="ATT (log points); coloured = p < 0.10"):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labels = [r["label"] for r in rows]
    atts = [r["att"] for r in rows]
    ps = [r["p"] for r in rows]
    cols = [TREAT_C if p < 0.10 else "#95a5a6" for p in ps]
    y = np.arange(len(labels))
    ax.barh(y, atts, color=cols, height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------ one case ----

def run_case(name, panel, screen, method="augsynth", n_jobs=None):
    spec = C.TREATMENTS[name]
    t0 = pd.Timestamp(spec["date"])
    # The month the unit stops being untreated. With an anticipation date
    # (a court verdict, a demolition start) the pre-window must end there,
    # not at the opening. Without one, they coincide.
    #
    # This used to be assigned only in section 4, ~230 lines below its first
    # use in section 3c, so `t_break` was a local referenced before
    # assignment and the structurally-matched-pool check raised
    # UnboundLocalError on every case. The bare `except` printed
    # "matched-pool check failed" and the run carried on. Same failure mode
    # as the leave_one_out duplicate-kwarg bug: a robustness check that
    # looked present and never once executed.
    t_break = pd.Timestamp(spec["anticipation"]) if spec.get("anticipation") else t0
    has_anticipation = t_break < t0
    print("\n" + "=" * 66)
    print(f"CASE: {name}   treated {spec['unit']}   t0 = {t0:%b %Y}")
    print("=" * 66)

    pool = build_donor_pool(panel, screen, spec)
    if spec.get("only_states"):
        print(f"Donor pool: {len(pool)} urban units "
              f"(RESTRICTED to {', '.join(spec['only_states'])})")
    else:
        print(f"Donor pool: {len(pool)} urban units "
              f"(excluding {', '.join(spec['exclude_states']) or 'nothing'})")
    if len(pool) < 5:
        print(f"  !! only {len(pool)} donors. A simplex over so few units can "
              f"fit almost anything;\n     treat this specification as "
              f"indicative, not as inference.")

    wide = to_wide(panel, "ln_inc_real_sm", units=pool + [spec["unit"]])
    if spec["unit"] not in wide.index:
        print(f"  SKIP: {spec['unit']} dropped by to_wide (missing months).")
        return None

    # ---- 1. headline -----------------------------------------------------
    # noise_floor is the treated unit's own median sampling standard error.
    # It is purely diagnostic: it tells us whether the pre-treatment fit is
    # tighter than the outcome is actually measured, which would mean the
    # model is fitting CPHS sampling noise rather than economic structure.
    # Noise floor, measured EMPIRICALLY rather than from the reported
    # sampling SE.
    #
    # ln_inc_se is std/sqrt(n_hh), which assumes n_hh independent households.
    # They are not independent: CPHS draws each district-month from a single
    # primary sampling unit -- n_psu == 1 for all three treated units, and for
    # 248 of 294 urban units. Households inside one PSU share a labour market,
    # so the true standard error is inflated by a design effect the naive
    # formula cannot see. Measured on one treated unit: naive SE 0.033,
    # actual month-to-month scatter around trend 0.113, i.e. 3.4x.
    #
    # Using the naive figure would make the overfitting check 3.4x too
    # permissive -- it would pass fits that are entirely inside the noise.
    _tr = panel[panel["unit"] == spec["unit"]].sort_values("MONTH_DT")
    _pre = _tr[_tr["MONTH_DT"] < t0]
    _naive = float(_pre["ln_inc_se"].median())
    _trend = _pre["ln_inc"].rolling(12, center=True, min_periods=4).mean()
    _emp = float((_pre["ln_inc"] - _trend).std())
    noise_floor = max(_emp, _naive) if np.isfinite(_emp) else _naive
    print(f"Noise floor: naive SE {_naive:.4f}, empirical scatter {_emp:.4f} "
          f"({_emp/_naive:.1f}x design effect) -> using {noise_floor:.4f}")
    res = fit_scm(wide, spec["unit"], t0, donors=pool, method=method,
                  l2="cv", n_jobs=n_jobs, noise_floor=noise_floor)
    print("\n" + res.summary())
    if res.cv_scores:
        best = min(res.cv_scores, key=res.cv_scores.get)
        print(f"  CV over {len(res.cv_scores)} penalties on pre-treatment "
              f"months only; selected alpha={best:g}")

    plot_paths(res, f"{name}: household income vs synthetic control",
               C.FIG_DIR / f"{name}_paths.png")
    plot_placebos(res, f"{name}: placebo-in-space",
                  C.FIG_DIR / f"{name}_placebos.png")

    # ---- 2. robustness ---------------------------------------------------
    print("\n--- robustness ---")
    fake = pd.Timestamp(C.PLACEBO_TIME_DATE.get(name, t0 - pd.DateOffset(years=3)))
    # Truncate at the ANNOUNCEMENT, not the opening. A placebo whose post-window
    # runs to Dec 2023 contains the whole construction period, so it reports the
    # real disruption as a fake effect -- which is what -0.29 (Jan 2021) and
    # -0.22 (Jan 2017) were both measuring. The test is only meaningful if BOTH
    # its windows sit in genuinely untreated time.
    cut_at = pd.Timestamp(spec.get("anticipation") or t0)
    try:
        pt = placebo_in_time(wide, spec["unit"], fake, cut_at,
                             donors=pool, method=method, l2=res.l2)
        print(f"  placebo-in-time ({fake:%b %Y}, window ends {cut_at:%b %Y}): "
              f"ATT = {pt.att:+.4f}  (should be near zero; "
              f"{pt.t0_index} pre / {len(pt.dates)-pt.t0_index} post months)")
    except Exception as e:
        print(f"  placebo-in-time failed: {e}")

    loo = leave_one_out(wide, spec["unit"], t0, res, donors=pool,
                        method=method, l2=res.l2)
    if loo:
        v = np.array(list(loo.values()))
        print(f"  leave-one-out: n={len(v)}  range [{v.min():+.4f}, {v.max():+.4f}]"
              f"  (headline {res.att:+.4f})")

    # COVID measurement break, reported ALWAYS, not on request.
    # CPHS switched to telephone interviews Mar 2020 - Jun 2021. That is a
    # measurement break, not an economic one, and it can sit inside a unit's
    # pre-period: up to 16 of 120 pre-treatment months, several with fewer
    # households than the study's own quality threshold of 60.
    # Both specifications are printed so neither can be chosen after the fact.
    cov_mask = ((wide.columns >= pd.Timestamp(C.COVID_START)) &
                (wide.columns <= pd.Timestamp(C.COVID_END)))
    if cov_mask.any():
        try:
            rc = fit_scm(wide.loc[:, ~cov_mask], spec["unit"], t0, donors=pool,
                         method=method, l2=res.l2, run_placebos=False,
                         noise_floor=noise_floor)
            print(f"  drop-COVID ({int(cov_mask.sum())} months removed): "
                  f"ATT={rc.att:+.4f}  preRMSPE={rc.rmspe_pre:.4f}  "
                  f"(headline {res.att:+.4f})")
        except Exception as e:
            print(f"  drop-COVID refit failed: {e}")

    for m in ["scm", "demeaned", "augsynth"]:
        r = fit_scm(wide, spec["unit"], t0, donors=pool, method=m,
                    l2=res.l2, run_placebos=False, noise_floor=noise_floor)
        print(f"  method={m:9s} ATT={r.att:+.4f}  preRMSPE={r.rmspe_pre:.4f}"
              f"  ({r.overfit_ratio:.2f}x noise)")

    # ---- 3. anticipation -------------------------------------------------
    if spec.get("anticipation"):
        ta = pd.Timestamp(spec["anticipation"])
        try:
            ra = fit_scm(wide, spec["unit"], ta, donors=pool, method=method,
                         l2=res.l2, run_placebos=False)
            pre_open = ra.gap[ra.t0_index: res.t0_index].mean()
            print(f"\n--- anticipation ---")
            print(f"  Announcement {ta:%b %Y} -> opening {t0:%b %Y}")
            print(f"  mean gap over that window: {pre_open:+.4f} log points")
            print("  A large positive value means the effect began at the "
                  "announcement,\n  not the opening -- land and construction, "
                  "not tourism.")
            plot_paths(ra, f"{name}: dated from announcement ({ta:%b %Y})",
                       C.FIG_DIR / f"{name}_anticipation.png",
                       t0_label="Announcement")
        except Exception as e:
            print(f"  anticipation test failed: {e}")

    # ---- 3b. event study: is the "effect" a recovery? --------------------
    # SCM assumes the pre-period is untreated. Where a project demolishes and
    # rebuilds a town centre it is not: construction can begin years before the
    # opening, so a large share of the "pre-treatment" months are already
    # treated. Fitting the counterfactual across them anchors it to a
    # construction-depressed level,
    # and the post-opening rebound then reads as a positive treatment effect.
    #
    # This block re-fits using the ANNOUNCEMENT as the break, so the donor
    # weights are chosen only on genuinely untreated months, then reports the
    # gap in three windows. If the post-opening gap merely returns to the
    # pre-verdict gap, the headline is recovery, not gain.
    if spec.get("anticipation"):
        ta = pd.Timestamp(spec["anticipation"])
        print("\n--- event study: clean pre-period (fit ends at announcement) ---")
        try:
            rc = fit_scm(wide, spec["unit"], ta, donors=pool, method=method,
                         l2="cv", n_jobs=n_jobs, noise_floor=noise_floor)
            g = pd.Series(rc.gap, index=pd.to_datetime(rc.dates))
            w_pre = g.loc[:ta - pd.DateOffset(days=1)].mean()
            w_mid = g.loc[ta:t0 - pd.DateOffset(days=1)].mean()
            w_post = g.loc[t0:].mean()
            print(f"  fit on {rc.t0_index} untreated months "
                  f"(pre-RMSPE {rc.rmspe_pre:.4f}, {rc.overfit_ratio:.2f}x noise)")
            print(f"  gap before announcement      : {w_pre:+.4f}")
            print(f"  gap announcement -> opening  : {w_mid:+.4f}   "
                  f"(construction / disruption)")
            print(f"  gap after opening            : {w_post:+.4f}")
            print(f"  NET vs pre-verdict baseline  : {w_post - w_pre:+.4f} "
                  f"log points  ({100*(np.exp(w_post-w_pre)-1):+.1f}%)")
            net = abs(w_post - w_pre)
            if net < noise_floor:
                print(f"  -> net movement ({net:.4f}) is INSIDE the noise floor "
                      f"({noise_floor:.4f}).\n     The post-opening rise is "
                      f"recovery from the construction trough,\n     not a gain "
                      f"above the pre-verdict trajectory.")
            pd.DataFrame({"date": rc.dates, "gap": rc.gap}).to_csv(
                C.RESULT_DIR / f"{name}_eventstudy.csv", index=False)
        except Exception as e:
            print(f"  event study failed: {e}")

    # ---- 3c. structurally matched donor pool ------------------------------
    # Same estimator, same penalty selection, same inference. The ONLY change
    # is which donors are eligible: the pool is screened in advance to the
    # units most like the treated one in pre-treatment economic structure.
    # Reported alongside the headline, never instead of it.
    print("\n--- robustness: structurally matched donor pool "
          f"(K={C.STRUCT_MATCH_K}) ---")
    try:
        sel, dist = structural_neighbours(
            panel, spec["unit"], pool, before=t_break, k=C.STRUCT_MATCH_K)
        near = dist.nsmallest(5)
        print("  closest donors: " + "; ".join(
            f"{u.split('|')[1].strip()} ({d:.2f})" for u, d in near.items()))
        dropped = [u for u in pool if u not in set(sel)]
        big = [u.split("|")[1].strip() for u in dropped][:6]
        print(f"  dropped {len(dropped)} structurally distant donors"
              + (f", incl. {', '.join(big)}" if big else ""))

        wm = to_wide(panel, "ln_inc_real_sm", units=sel + [spec["unit"]])
        rm = fit_scm(wm, spec["unit"], t0, donors=sel, method=method,
                     l2="cv", n_jobs=n_jobs, noise_floor=noise_floor)
        print(f"  headline ATT = {rm.att:+.4f}  p(RMSPE) = "
              f"{rm.permutation_pvalue():.3f}  p(ATT) = {rm.att_pvalue():.3f}  "
              f"preRMSPE = {rm.rmspe_pre:.4f} ({rm.overfit_ratio:.2f}x noise)")
        print(f"    vs full pool  {res.att:+.4f}  p(RMSPE) = "
              f"{res.permutation_pvalue():.3f}  ({len(pool)} donors)")

        rme = fit_scm(wm, spec["unit"], t_break, donors=sel, method=method,
                      l2="cv", run_placebos=False, noise_floor=noise_floor)
        g = pd.Series(rme.gap, index=pd.to_datetime(rme.dates))
        base = g.loc[:t_break - pd.DateOffset(days=1)].mean()
        mid = g.loc[t_break:t0 - pd.DateOffset(days=1)].mean() - base
        post = g.loc[t0:].mean() - base
        print(f"  event study on matched pool: construction {mid:+.4f}, "
              f"post-opening {post:+.4f}, net {post:+.4f}")
        print(f"    vs full pool: construction see above, net compared below")
        pd.DataFrame({"date": rme.dates, "gap": rme.gap}).to_csv(
            C.RESULT_DIR / f"{name}_eventstudy_matched.csv", index=False)
        pd.DataFrame({"unit": dist.index, "distance": dist.values,
                      "selected": dist.index.isin(sel)}).sort_values(
            "distance").to_csv(C.RESULT_DIR / f"{name}_donor_distance.csv",
                               index=False)
    except Exception as e:
        print(f"  matched-pool check failed: {e}")

    # ---- 4. mechanism: income sources ------------------------------------
    # Estimated in REAL RUPEES, not log1p of a rupee level.
    #
    # The previous version used np.log1p(mean_<source>). Several of these
    # sources -- rent, self-production, private transfers -- sit at or near
    # zero rupees in most district-months, and log1p is violently unstable
    # there: it produced a "Business profit ATT" of +5.35 log points, i.e. a
    # literal 210x increase, and the numbers were not on a scale comparable
    # with the headline. More data does not fix that; the transform was wrong.
    #
    # Rupees per household-month are stable near zero, additive, and let the
    # parts be checked against the whole -- which is the actual question the
    # README asks: is the total effect coming from business profit and wages
    # (a genuine local demand story) or from private transfers (a confound)?
    defl = panel["defl"] if "defl" in panel.columns else 1.0
    pre_mask = panel["MONTH_DT"] < t0
    treated_pre_inc = float(panel.loc[
        pre_mask & (panel["unit"] == spec["unit"]), "mean_inc_real"].mean())

    # Every channel is fitted on the CLEAN pre-period and reported in the same
    # three windows as the headline event study.
    #
    # Fitting these at t0 = opening put four years of construction inside the
    # pre-window, so each channel measured recovery-from-trough exactly as the
    # headline did. That is what produced "wages +59%" and a 935% reconciliation
    # residual. Splitting construction from post-opening also answers the more
    # interesting question: which channels absorbed the disruption, and which
    # recovered?
    # t_break is defined at the top of run_case.

    def _scm_on(col_series, label):
        """Fit on the untreated pre-period; return (result, mid gap, post gap)."""
        p2 = panel.copy()
        p2["_y"] = col_series
        w2 = to_wide(p2, "_y", units=pool + [spec["unit"]])
        if spec["unit"] not in w2.index:
            raise ValueError("treated unit dropped by to_wide")
        r = fit_scm(w2, spec["unit"], t_break, donors=pool, method=method,
                    l2="cv", n_jobs=n_jobs)
        g = pd.Series(r.gap, index=pd.to_datetime(r.dates))
        base = g.loc[:t_break - pd.DateOffset(days=1)].mean()
        mid = g.loc[t_break:t0 - pd.DateOffset(days=1)].mean() - base
        post = g.loc[t0:].mean() - base
        return r, float(mid), float(post)

    # The rupee-level decomposition has been REMOVED.
    #
    # It was tried twice and failed twice. log1p(level) gave "business profit
    # +5.35 log points" (a 210x increase). Raw levels gave a total-income
    # construction effect of +Rs 22,532 when the log-based event study on the
    # same data and the same window showed a LOSS -- the sign was inverted on
    # the headline result -- with wages at +276% of pre-treatment income and a
    # reconciliation residual of 15,172%.
    #
    # The cause is not the transform. CPHS draws each district-month from one
    # PSU (n_psu == 1 for all three treated units), so mean income LEVELS are
    # dominated by a handful of households, and a simplex fit across 71 donors
    # whose levels differ by an order of magnitude extrapolates wildly. Shares
    # are bounded in [0,1] and do not have this failure mode.
    #
    # Removing it also halves the runtime.

    # ---- 4b. composition: did the income MIX change? ---------------------
    # Shares are bounded in [0, 1] and immune to the near-zero-level problem,
    # so they answer "did the structure of income change" cleanly even where
    # a rupee effect is noisy.
    print("\n--- income composition (percentage points of total income) ---")
    if has_anticipation:
        print(f"  {'channel':20s} {'construction':>12s} {'post-opening':>13s}")
    else:
        print(f"  {'channel':20s} {'post-opening':>13s}   "
              f"(no anticipation date: no construction window)")
    comp_rows = []
    for label, cols in C.INCOME_GROUPS.items():
        present = [f"share_{c}" for c in cols if f"share_{c}" in panel.columns]
        if not present:
            continue
        try:
            share = panel[present].clip(lower=0).sum(axis=1).clip(upper=1)
            r, mid, post = _scm_on(share, label)
            pv = r.att_pvalue()
            comp_rows.append({"label": label, "att": 100 * post,
                              "construction": 100 * mid, "p": pv})
            if has_anticipation:
                print(f"  {label:20s} {100*mid:+7.2f} pp {100*post:+9.2f} pp  "
                      f"p={pv:.3f}")
            else:
                print(f"  {label:20s} {100*post:+9.2f} pp  p={pv:.3f}")
        except Exception as e:
            print(f"  {label:20s} failed: {e}")
    if comp_rows:
        # Shares partition income, so the changes should very nearly cancel.
        # They are fitted separately, each with its own donor weights, so exact
        # zero is not guaranteed -- but a large sum means the per-channel
        # counterfactuals disagree and no single channel should be read alone.
        windows = ([("construction", "construction"), ("att", "post-opening")]
                   if has_anticipation else [("att", "post-opening")])
        for w, lab in windows:
            tot = sum(r[w] for r in comp_rows)
            flag = "  <-- channels not mutually consistent" if abs(tot) > 3 else ""
            print(f"  {'sum (should be ~0)':20s} {tot:+7.2f} pp  [{lab}]{flag}")
        plot_sources(comp_rows, f"{name}: change in income composition",
                     C.FIG_DIR / f"{name}_composition.png",
                     xlabel="ATT (percentage points of income); "
                            "coloured = p < 0.10")
        pd.DataFrame(comp_rows).to_csv(
            C.RESULT_DIR / f"{name}_composition.csv", index=False)

    # ---- 5. distribution: who gained -------------------------------------
    print("\n--- distributional SCM (who lost during construction, who gained after) ---")
    dist_rows = []
    for q, col in [("p25", "inc_p25"), ("p50", "inc_p50"), ("p75", "inc_p75")]:
        try:
            r, mid, post = _scm_on(np.log(panel[col].clip(lower=1) / defl), q)
            pv = r.att_pvalue()
            dist_rows.append({"label": q, "att": post, "construction": mid,
                              "p": pv})
            print(f"  {q:4s} construction={mid:+.4f} ({100*(np.exp(mid)-1):+.1f}%)"
                  f"   post-opening={post:+.4f} ({100*(np.exp(post)-1):+.1f}%)"
                  f"  p={pv:.3f}")
        except Exception as e:
            print(f"  {q:4s} failed: {e}")
    if dist_rows:
        plot_sources(dist_rows, f"{name}: effect by income percentile",
                     C.FIG_DIR / f"{name}_distribution.png",
                     xlabel="ATT (log points); coloured = p < 0.10")

    # ---- save ------------------------------------------------------------
    pd.DataFrame({
        "date": res.dates, "treated": res.y_treated,
        "synthetic": res.y_synth, "gap": res.gap,
        "post": np.arange(len(res.dates)) >= res.t0_index,
    }).to_csv(C.RESULT_DIR / f"{name}_series.csv", index=False)

    pd.DataFrame(res.top_weights(50), columns=["donor", "weight"]).to_csv(
        C.RESULT_DIR / f"{name}_weights.csv", index=False)

    return {
        "case": name, "att": res.att,
        "pct_effect": 100 * (np.exp(res.att) - 1),
        "p_rmspe": res.permutation_pvalue(), "p_att": res.att_pvalue(),
        "rmspe_pre": res.rmspe_pre, "n_donors": len(res.donors),
        "l2": res.l2, "overfit_ratio": res.overfit_ratio,
        "deflated": bool(panel["_deflated"].iloc[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run all treated cases")
    ap.add_argument("--method", default="augsynth",
                    choices=["scm", "demeaned", "augsynth"])
    ap.add_argument("--jobs", type=int, default=None,
                    help="worker processes for placebo refits (default: all cores)")
    args = ap.parse_args()
    n_jobs = args.jobs if args.jobs is not None else C.N_JOBS

    panel = pd.read_parquet(C.PANEL_PARQUET)
    screen = pd.read_csv(C.SCREEN_CSV)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)

    if not bool(panel["_deflated"].iloc[0]):
        print("\n" + "!" * 66)
        print("NOMINAL PANEL: no CPI deflator was applied. Every effect below")
        print("is in nominal rupees and is NOT an inflation-adjusted estimate.")
        print("!" * 66)

    cases = list(C.TREATMENTS) if args.all else [C.PRIMARY_CASE]
    out = [r for r in (run_case(c, panel, screen, args.method, n_jobs)
                       for c in cases) if r is not None]

    if out:
        summary = pd.DataFrame(out)
        summary.to_csv(C.RESULT_DIR / "summary.csv", index=False)
        print("\n" + "=" * 66)
        print(summary.round(4).to_string(index=False))
        print(f"\nFigures -> {C.FIG_DIR}")
        print(f"Results -> {C.RESULT_DIR}")


if __name__ == "__main__":
    main()
