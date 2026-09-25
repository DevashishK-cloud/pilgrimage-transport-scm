"""
src/scm.py — synthetic control, written from scratch.

Implements:
  * Abadie-Diamond-Hainmueller SCM: simplex-constrained weights fit on all
    pre-treatment outcome lags.
  * De-meaned SCM (Doudchenko-Imbens): allows an intercept shift, which
    matters when the treated unit sits outside the donor convex hull.
  * Ridge-augmented SCM (Ben-Michael, Feller & Rothstein): bias correction
    when pre-treatment fit is imperfect.
  * Placebo-in-space permutation inference with RMSPE ratios.
  * Placebo-in-time and leave-one-out robustness.

Using all pre-treatment lags as predictors with V = I avoids the nested
V-optimisation, which Ferman, Pinto & Possebom (2020) show is a researcher
degree of freedom that can be tuned to produce whatever result you want.
State this choice in your methodology section.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------ core solve ----

def solve_simplex_weights(y1: np.ndarray, Y0: np.ndarray,
                          l2: float = 0.0) -> np.ndarray:
    """
    min_w || y1 - Y0 w ||^2 + l2*||w||^2   s.t.  w >= 0,  sum(w) = 1

    y1 : (T_pre,)        treated unit's pre-treatment outcomes
    Y0 : (T_pre, J)      donor pre-treatment outcomes
    """
    T, J = Y0.shape

    def obj(w):
        r = y1 - Y0 @ w
        return float(r @ r + l2 * (w @ w))

    def grad(w):
        r = y1 - Y0 @ w
        return -2.0 * (Y0.T @ r) + 2.0 * l2 * w

    w0 = np.full(J, 1.0 / J)
    res = minimize(
        obj, w0, jac=grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * J,
        constraints=[{"type": "eq",
                      "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(J, 1.0 / J)


def ridge_augment(y1_pre, Y0_pre, Y0_post, w, lam):
    """
    Ridge-augmented SCM (Ben-Michael, Feller & Rothstein 2021).

        Y1t(0)_aug = sum_i w_i * Y_it  +  (x1 - X0 w)' eta_t

    where eta_t is the ridge regression of donors' period-t outcome on their
    pre-treatment outcome path. In words: plain SCM leaves some pre-treatment
    imbalance uncorrected; ridge estimates how that residual imbalance maps
    into post-period outcomes and subtracts the implied bias.

    y1_pre  : (T_pre,)          treated pre-treatment path
    Y0_pre  : (T_pre, J)        donor pre-treatment paths
    Y0_post : (T_post, J)       donor post-treatment paths
    w       : (J,)              SCM weights
    Returns : (T_post,)         additive correction
    """
    T_pre, J = Y0_pre.shape
    T_post = Y0_post.shape[0]

    # Centre donors period-by-period (ridge is not scale/location invariant).
    mu = Y0_pre.mean(axis=1, keepdims=True)          # (T_pre, 1)
    X0 = Y0_pre - mu                                 # (T_pre, J)
    x1 = y1_pre - mu.ravel()                         # (T_pre,)

    imbalance = x1 - X0 @ w                          # (T_pre,)

    # eta_t = (X0 X0' + lam I)^-1 X0 y0_t  for every post period t at once
    A = X0 @ X0.T + lam * np.eye(T_pre)              # (T_pre, T_pre)
    try:
        Eta = np.linalg.solve(A, X0 @ Y0_post.T)     # (T_pre, T_post)
    except np.linalg.LinAlgError:
        return np.zeros(T_post)

    return imbalance @ Eta                           # (T_post,)


def select_l2_cv(y1_pre, Y0_pre, alphas=None, n_folds=5, min_train=12):
    """
    Choose the simplex ridge penalty by rolling-origin cross-validation on
    PRE-TREATMENT data only.

    Why this exists. With J donors and T_pre periods, plain SCM (l2=0) is a
    least-squares fit of T_pre numbers using J weights. Whenever J > T_pre --
    which is the normal case here, ~250 urban donors against 120 pre-treatment
    months -- the simplex can reproduce the treated path almost exactly, and
    infinitely many weight vectors do. A pre-treatment RMSPE near zero is that
    happening. It is not a good fit; it is an unidentified one, and it makes
    the RMSPE-ratio p-value a division by noise.

    Cross-validation fixes the diagnosis directly: fit on early pre-treatment
    months, score on later pre-treatment months the fit has not seen. An
    unpenalised fit that merely memorises will score badly out of sample, so
    the procedure selects a penalty that generalises.

    Crucially this uses NO post-treatment outcomes, so it cannot be tuned
    toward a result -- the specification-search failure mode Ferman, Pinto &
    Possebom (2020) warn about. Report the selected value.

    Returns (l2, {alpha: cv_mse}).
    """
    T, J = Y0_pre.shape
    if alphas is None:
        alphas = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Scale-free: penalties are relative to the donor matrix's own magnitude,
    # so the same grid works on log income, shares and rupee levels alike.
    scale = float((Y0_pre ** 2).sum() / max(J, 1))
    if not np.isfinite(scale) or scale <= 0:
        return 0.0, {}

    h = max(3, T // 12)
    folds = []
    for k in range(n_folds):
        va_end = T - k * h
        va_start = va_end - h
        if va_start < min_train:
            break
        folds.append((va_start, va_end))
    if not folds:
        return 0.0, {}

    scores = {}
    for a in alphas:
        errs = []
        for s0, s1 in folds:
            try:
                w = solve_simplex_weights(y1_pre[:s0], Y0_pre[:s0], l2=a * scale)
            except Exception:
                continue
            r = y1_pre[s0:s1] - Y0_pre[s0:s1] @ w
            errs.append(float((r ** 2).mean()))
        if errs:
            scores[a] = float(np.mean(errs))
    if not scores:
        return 0.0, {}
    best_alpha = min(scores, key=scores.get)
    return best_alpha * scale, scores


def _fit_one(y_t, Y_d, t0_idx, method, l2, ridge_lambda):
    """
    Fit one treated-vs-donors problem. Module level (not a closure) so that
    the placebo loop can hand it to worker processes.
    Returns (weights, synthetic path).
    """
    y_pre, Y_pre = y_t[:t0_idx], Y_d[:t0_idx]
    if method == "demeaned":
        my, mY = y_pre.mean(), Y_pre.mean(axis=0)
        w = solve_simplex_weights(y_pre - my, Y_pre - mY, l2=l2)
        return w, (Y_d - mY) @ w + my
    w = solve_simplex_weights(y_pre, Y_pre, l2=l2)
    synth = Y_d @ w
    if method == "augsynth":
        corr = ridge_augment(y_pre, Y_pre, Y_d[t0_idx:], w, ridge_lambda)
        synth = synth.copy()
        synth[t0_idx:] += corr
    return w, synth


def _init_worker():
    """
    Pin BLAS to one thread inside each worker.

    Without this, N worker processes each spin up N BLAS threads and the pool
    is slower than running serially -- measured 129s serial vs 331s on two
    workers before the pin, and 59s after it. The matrices here are small
    enough that threaded BLAS buys nothing anyway.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass


def _placebo_task(args):
    """One placebo-in-space refit: donor j treated as if it were treated."""
    j, Y, t0_idx, method, l2, ridge_lambda = args
    others = [k for k in range(Y.shape[1]) if k != j]
    try:
        _, sj = _fit_one(Y[:, j], Y[:, others], t0_idx, method, l2, ridge_lambda)
        return j, Y[:, j] - sj
    except Exception:
        return j, None


# ------------------------------------------------------------- container ----

@dataclass
class SCMResult:
    treated: str
    donors: list
    weights: np.ndarray
    dates: np.ndarray
    y_treated: np.ndarray
    y_synth: np.ndarray
    t0_index: int
    method: str = "scm"
    placebo_gaps: dict = field(default_factory=dict)
    l2: float = 0.0
    cv_scores: dict = field(default_factory=dict)
    noise_floor: float = float("nan")   # treated unit's median sampling SE

    # -- derived quantities ------------------------------------------------
    @property
    def gap(self):
        return self.y_treated - self.y_synth

    @property
    def rmspe_pre(self):
        g = self.gap[: self.t0_index]
        return float(np.sqrt((g ** 2).mean()))

    @property
    def rmspe_post(self):
        g = self.gap[self.t0_index:]
        return float(np.sqrt((g ** 2).mean()))

    @property
    def rmspe_ratio(self):
        return self.rmspe_post / self.rmspe_pre if self.rmspe_pre > 0 else np.inf

    @property
    def att(self):
        """Average post-treatment gap. In logs this is ~ a proportional effect."""
        return float(self.gap[self.t0_index:].mean())

    @property
    def overfit_ratio(self):
        """
        Pre-treatment RMSPE as a multiple of the treated unit's own sampling
        standard error.

        The synthetic control cannot legitimately track the treated series more
        closely than that series is measured. CPHS gives ~200 households per
        district-month, so a district's monthly mean log income carries a
        standard error around 0.03-0.04. A pre-RMSPE well below that is the model
        fitting sampling noise, not economic structure, and every p-value built
        on the RMSPE ratio inherits the problem.
        """
        if not np.isfinite(self.noise_floor) or self.noise_floor <= 0:
            return float("nan")
        return self.rmspe_pre / self.noise_floor

    @property
    def overfit_warning(self):
        r = self.overfit_ratio
        if not np.isfinite(r):
            return None
        if r < 0.5:
            return (f"pre-RMSPE is {r:.2f}x the sampling SE -- the fit is inside "
                    f"the noise; treat RMSPE-ratio inference as unreliable")
        if r > 3.0:
            return (f"pre-RMSPE is {r:.1f}x the sampling SE -- poor pre-treatment "
                    f"fit; the treated unit may be outside the donor convex hull")
        return None

    def top_weights(self, n=10):
        idx = np.argsort(self.weights)[::-1][:n]
        return [(self.donors[i], float(self.weights[i]))
                for i in idx if self.weights[i] > 1e-4]

    def permutation_pvalue(self):
        """
        Share of donors whose placebo RMSPE ratio is at least as extreme as
        the treated unit's. This is your p-value: with one treated unit,
        conventional standard errors do not apply.
        """
        if not self.placebo_gaps:
            return np.nan
        ratios = []
        for g in self.placebo_gaps.values():
            pre = np.sqrt((g[: self.t0_index] ** 2).mean())
            post = np.sqrt((g[self.t0_index:] ** 2).mean())
            if pre > 0:
                ratios.append(post / pre)
        if not ratios:
            return np.nan
        ratios = np.array(ratios)
        return float((np.sum(ratios >= self.rmspe_ratio) + 1) / (len(ratios) + 1))

    def att_pvalue(self):
        """
        Two-sided permutation p-value based on the ATT itself rather than the
        RMSPE ratio. More directly interpretable, and a useful cross-check:
        report both. If they disagree sharply, your pre-treatment fit is
        doing the work rather than the post-treatment divergence.
        """
        if not self.placebo_gaps:
            return np.nan
        placebo_atts = np.array([float(g[self.t0_index:].mean())
                                 for g in self.placebo_gaps.values()])
        return float((np.sum(np.abs(placebo_atts) >= abs(self.att)) + 1)
                     / (len(placebo_atts) + 1))

    def summary(self):
        lines = [
            f"Treated unit      : {self.treated}",
            f"Method            : {self.method}",
            f"Donors in pool    : {len(self.donors)}",
            f"Pre-period RMSPE  : {self.rmspe_pre:.4f}",
            f"Ridge penalty l2  : {self.l2:.6g}" + (" (cross-validated)"
                                                    if self.cv_scores else ""),
            f"Post-period RMSPE : {self.rmspe_post:.4f}",
            f"RMSPE ratio       : {self.rmspe_ratio:.2f}",
            f"ATT (log points)  : {self.att:+.4f}  "
            f"(~{100*(np.exp(self.att)-1):+.1f}% in income)",
        ]
        p = self.permutation_pvalue()
        if np.isfinite(p):
            lines.append(f"p (RMSPE ratio)   : {p:.3f}  "
                         f"({len(self.placebo_gaps)} placebos)")
            lines.append(f"p (ATT, 2-sided)  : {self.att_pvalue():.3f}")
        if np.isfinite(self.overfit_ratio):
            lines.append(f"Pre-RMSPE / noise : {self.overfit_ratio:.2f}x "
                         f"(sampling SE {self.noise_floor:.4f})")
        w = self.overfit_warning
        if w:
            lines.append(f"  !! {w}")
        lines.append("Donor weights     :")
        for name, wt in self.top_weights():
            lines.append(f"    {wt:6.3f}  {name}")
        return "\n".join(lines)


# ------------------------------------------------------------- estimator ----

def fit_scm(wide, treated_unit, t0, donors=None, method="scm",
            l2=0.0, ridge_lambda=1.0, run_placebos=True,
            min_pre_periods=12, n_jobs=None, noise_floor=float("nan")):
    """
    wide         : DataFrame, index = unit, columns = dates (sorted)
    treated_unit : index label
    t0           : first treated period (anything pd.Timestamp-comparable)
    method       : "scm" | "demeaned" | "augsynth"
    l2           : ridge penalty on the weights. Pass "cv" to select it by
                   pre-treatment cross-validation (see select_l2_cv). The
                   selected value is then held FIXED across the treated unit
                   and every placebo, so permutation inference compares like
                   with like -- each unit gets an identical procedure.
    noise_floor  : the treated unit's median sampling SE, used only to report
                   whether the pre-treatment fit is tighter than the data are
                   measured. Purely diagnostic; affects no estimate.
    """
    import pandas as pd

    if treated_unit not in wide.index:
        raise KeyError(
            f"{treated_unit!r} not in the panel. "
            f"Closest matches: "
            f"{[u for u in wide.index if treated_unit.split('|')[1].strip() in u][:5]}"
        )

    dates = np.array(wide.columns)
    t0 = pd.Timestamp(t0)
    t0_idx = int(np.searchsorted(dates, np.datetime64(t0)))
    if t0_idx < min_pre_periods:
        raise ValueError(f"Only {t0_idx} pre-treatment periods; need "
                         f">= {min_pre_periods}.")

    pool = [u for u in wide.index if u != treated_unit]
    if donors is not None:
        pool = [u for u in pool if u in set(donors)]
    if len(pool) < 5:
        raise ValueError(f"Donor pool has only {len(pool)} units.")

    y = wide.loc[treated_unit].to_numpy(float)
    Y = wide.loc[pool].to_numpy(float).T          # (T, J)

    cv_scores = {}
    if isinstance(l2, str):
        if l2 != "cv":
            raise ValueError(f"l2 must be a number or 'cv', got {l2!r}")
        l2, cv_scores = select_l2_cv(y[:t0_idx], Y[:t0_idx])

    w, y_syn = _fit_one(y, Y, t0_idx, method, l2, ridge_lambda)
    res = SCMResult(treated=treated_unit, donors=pool, weights=w, dates=dates,
                    y_treated=y, y_synth=y_syn, t0_index=t0_idx, method=method,
                    l2=float(l2), cv_scores=cv_scores, noise_floor=noise_floor)

    # ---- placebo-in-space: refit treating each donor as if it were treated.
    # This is ~J independent SLSQP solves and is by far the dominant cost of a
    # run (J refits x 9 specifications x 3 cases). The fits do not talk to each
    # other, so they are farmed out to processes. Set n_jobs=1 to run serially.
    if run_placebos:
        tasks = [(j, Y, t0_idx, method, l2, ridge_lambda) for j in range(len(pool))]
        workers = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
        workers = max(1, min(workers, len(tasks)))
        if workers == 1:
            done = (_placebo_task(t) for t in tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=workers,
                                     initializer=_init_worker)
            try:
                done = list(ex.map(_placebo_task, tasks, chunksize=4))
            finally:
                ex.shutdown()
        for j, gap in done:
            if gap is not None:
                res.placebo_gaps[pool[j]] = gap
    return res


# ---------------------------------------------------------- robustness ----

def placebo_in_time(wide, treated_unit, fake_t0, real_t0, **kw):
    """
    Move the treatment date earlier into the pre-period. A real effect should
    NOT show up before the intervention. If it does, you have a pre-trend, not
    a treatment effect.
    """
    import pandas as pd
    cut = wide.loc[:, wide.columns < pd.Timestamp(real_t0)]
    return fit_scm(cut, treated_unit, fake_t0, run_placebos=False, **kw)


def leave_one_out(wide, treated_unit, t0, result, **kw):
    """
    Drop each positively-weighted donor in turn and refit. If the estimate
    collapses when one donor is removed, it is not robust -- report this.
    """
    out = {}
    # Callers pass donors=<pool> through **kw; that collides with the
    # per-iteration pool below and raises TypeError, which the bare except
    # swallowed -- leave-one-out silently returned {} instead of running.
    kw.pop("donors", None)
    contributors = [d for d, w in result.top_weights(n=50)]
    for d in contributors:
        pool = [u for u in result.donors if u != d]
        try:
            r = fit_scm(wide, treated_unit, t0, donors=pool,
                        run_placebos=False, **kw)
            out[d] = r.att
        except Exception:
            continue
    return out


def trim_placebos(result, max_ratio=5.0):
    """
    Standard practice: drop placebo units whose PRE-period RMSPE is far worse
    than the treated unit's. A donor the model could never fit tells you
    nothing about whether your effect is unusual.
    """
    keep = {}
    for u, g in result.placebo_gaps.items():
        pre = np.sqrt((g[: result.t0_index] ** 2).mean())
        if pre <= max_ratio * result.rmspe_pre:
            keep[u] = g
    trimmed = SCMResult(
        treated=result.treated, donors=result.donors, weights=result.weights,
        dates=result.dates, y_treated=result.y_treated, y_synth=result.y_synth,
        t0_index=result.t0_index, method=result.method, placebo_gaps=keep,
        l2=result.l2, cv_scores=result.cv_scores,
        noise_floor=result.noise_floor,
    )
    return trimmed
