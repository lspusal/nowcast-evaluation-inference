"""Econometric nowcasting benchmarks.

Implements docs/02_methodology.md §3 (M1): bridge equations, MIDAS, DFM.

All estimators consume ONLY an information set built by
``src/data/info_set.py`` (R1/R3) and follow the refit cadence of
``configs/experiment_*.yaml::refit_every_quarters.econ``.

Common interface
----------------
    class Model:
        name: str
        def fit(self, X: pd.DataFrame, y: pd.Series) -> None
        def predict(self, X: pd.DataFrame, target_quarter) -> Forecast

``X`` is the TRANSFORMED mixed-frequency panel exactly as
``information_set`` returned it (obs_date x series, ragged edge intact), and
``y`` is the quarterly target restricted to quarters publishable at ``t``
(``info_set.available_target``). ``predict`` takes the target quarter
explicitly: the stub's ``predict(X)`` left the predicted period implicit,
which is unsafe here because the same panel serves the backcast, nowcast and
forecast horizons of different quarters.

Deliberately, X is NOT pre-aggregated to quarterly by the caller: the bridge
wants quarterly averages, MIDAS wants the monthly lag structure, and the DFM
wants the raw ragged panel. Aggregating up front would throw away what two of
the three models need.

None of these estimators may look at ``X`` rows dated after the nowcast date
or at ``y`` quarters not yet published; both guarantees are the caller's
(``info_set``), and ``tests/test_econ_baselines.py`` re-checks them
end-to-end per model.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.info_set import to_quarterly

logger = logging.getLogger(__name__)

MAX_AR_LAGS = 2  # BIC-selected, per the module spec


@dataclass
class Forecast:
    mean: float
    q10: float | None = None
    q50: float | None = None
    q90: float | None = None


def _robust_scale(x: np.ndarray) -> float:
    """MAD-based scale estimate, normalised to be comparable to an SD.

    Used instead of the standard deviation wherever a threshold must survive
    a contaminated sample: the SD is inflated BY the outliers it is meant to
    detect, so an SD-scaled screen loosens exactly when it is most needed. The 1.4826 factor makes MAD consistent with
    the SD under normality.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return float("nan")
    mad = np.median(np.abs(x - np.median(x)))
    return float(1.4826 * mad)


def _mask_outliers(frame: pd.DataFrame, n_mad: float) -> tuple[pd.DataFrame, int]:
    """Blank observations further than ``n_mad`` robust deviations from median.

    They are set to NaN rather than winsorised or dropped, because
    ``DynamicFactorMQ`` handles missing data natively through the Kalman
    filter: the observation is simply not used to update the state, which is
    the standard treatment of the COVID quarters in the nowcasting literature
    (rather than letting them dominate the factor loadings).
    """
    out = frame.copy()
    n_masked = 0
    for col in out.columns:
        v = out[col]
        scale = _robust_scale(v.to_numpy())
        if not np.isfinite(scale) or scale <= 0:
            continue
        centre = float(np.nanmedian(v.to_numpy()))
        extreme = (v - centre).abs() > n_mad * scale
        n_masked += int(extreme.sum())
        out.loc[extreme, col] = np.nan
    return out, n_masked


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least squares with an intercept; returns (beta, residual variance)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    dof = max(len(y) - Xc.shape[1], 1)
    return beta, float(resid @ resid / dof)


def _bic(n: int, k: int, sigma2: float) -> float:
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return np.inf
    return n * np.log(sigma2) + k * np.log(n)


def _design(
    xq: pd.Series, y: pd.Series, n_ar: int
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Rows where the indicator and ``n_ar`` target lags are all observed."""
    frame = pd.DataFrame({"x": xq, "y": y})
    for lag in range(1, n_ar + 1):
        frame[f"y_lag{lag}"] = frame["y"].shift(lag)
    frame = frame.dropna()
    if frame.empty:
        return np.empty((0, 1 + n_ar)), np.empty(0), []
    cols = ["x"] + [f"y_lag{lag}" for lag in range(1, n_ar + 1)]
    return frame[cols].to_numpy(), frame["y"].to_numpy(), list(frame.index)


class NaiveMeanNowcaster:
    """Predict the expanding historical mean of published target growth.

    THE MISSING BENCHMARK. No trivial reference
    existed anywhere in this study, which is an omission no forecasting paper
    survives: without it, "model A beats model B by 30%" says nothing about
    whether either is useful. The first irreducibility calculation exposed it
    -- the target's own standard deviation (4.49) sits BETWEEN the DFM (4.03)
    and the TSFM+bridge (5.37), meaning one of the headline models does not
    beat predicting a constant.

    Uses only quarters publishable at t (R1), so it is a real-time naive
    forecast rather than a full-sample mean.
    """

    name = "naive_mean"

    def __init__(self, min_train_quarters: int = 8, **kw):
        self.min_train_quarters = min_train_quarters
        self.y_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.y_ = y.sort_index()

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        hist = y if y is not None else self.y_
        if hist is None:
            return Forecast(mean=float("nan"))
        hist = hist.loc[hist.index < pd.Timestamp(target_quarter)].dropna()
        if len(hist) < self.min_train_quarters:
            return Forecast(mean=float("nan"))
        return Forecast(mean=float(hist.mean()))


class NaiveAR1Nowcaster:
    """AR(1) on published target growth -- the standard econometric floor.

    A random walk is inappropriate for a stationary growth rate, so the
    conventional trivial benchmark here is an AR(1) fitted on whatever
    quarters are publishable at t (R1).
    """

    name = "naive_ar1"

    def __init__(self, min_train_quarters: int = 12, **kw):
        self.min_train_quarters = min_train_quarters
        self.y_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Nothing is estimated here on purpose: see predict().
        self.y_ = y.sort_index()

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        """Estimate AR(1) at predict time, on history strictly before q.

        Deliberately re-estimated here rather than in ``fit``: an AR(1) is
        essentially free, and fitting it in ``fit`` made beta depend on
        WHICH series the caller happened to pass, so a caller handing over a
        full-sample target would leak future quarters into the coefficient.
        In this pipeline ``available_target`` already filters by vintage so no
        leak occurred, but the benchmark whose whole job is to be the honest
        floor must not depend on its caller for that (defence in depth, R1).
        A test perturbing quarters at/after q caught exactly this.
        """
        hist = y if y is not None else self.y_
        if hist is None:
            return Forecast(mean=float("nan"))
        hist = hist.loc[hist.index < pd.Timestamp(target_quarter)].dropna()
        frame = pd.DataFrame({"y": hist, "lag": hist.shift(1)}).dropna()
        if len(frame) < self.min_train_quarters:
            return Forecast(mean=float("nan"))
        beta, _ = _ols(frame[["lag"]].to_numpy(), frame["y"].to_numpy())
        return Forecast(mean=float(beta @ np.array([1.0, hist.iloc[-1]])))


class BridgeNowcaster:
    """Per-indicator ARX bridge equations, averaged.

    For each indicator the monthly/daily series is averaged to quarterly over
    the observations actually published (so a partially-observed quarter is
    summarised from what is known -- the ragged edge carried to quarterly
    frequency), then

        y_q = a + b * x_q + sum_{j=1..p} c_j * y_{q-j},   p <= MAX_AR_LAGS

    is fitted by OLS with ``p`` chosen by BIC. Forecasts are averaged across
    indicators, which is the standard robust bridge aggregation: individual
    bridge equations are noisy and pooling them beats picking one.
    """

    name = "bridge"

    def __init__(self, min_train_quarters: int = 20):
        self.min_train_quarters = min_train_quarters
        self.equations: dict[str, dict] = {}
        self.y_: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.equations = {}
        self.y_ = y.sort_index()
        if len(self.y_) < self.min_train_quarters:
            logger.warning(
                "%s: only %d training quarters (< %d); no equations fitted",
                self.name,
                len(self.y_),
                self.min_train_quarters,
            )
            return

        xq_all = to_quarterly(X)
        for col in xq_all.columns:
            best = None
            for n_ar in range(MAX_AR_LAGS + 1):
                Xd, yd, _ = _design(xq_all[col], self.y_, n_ar)
                if len(yd) < self.min_train_quarters:
                    continue
                beta, sigma2 = _ols(Xd, yd)
                bic = _bic(len(yd), Xd.shape[1] + 1, sigma2)
                if best is None or bic < best["bic"]:
                    best = {"beta": beta, "n_ar": n_ar, "bic": bic, "sigma2": sigma2}
            if best is not None:
                self.equations[col] = best

        if not self.equations:
            logger.warning("%s: no indicator produced a usable equation", self.name)

    def predict(self, X: pd.DataFrame, target_quarter) -> Forecast:
        if not self.equations or self.y_ is None:
            return Forecast(mean=float("nan"))

        q = pd.Timestamp(target_quarter)
        xq_all = to_quarterly(X)
        preds = []
        for col, eq in self.equations.items():
            if q not in xq_all.index or pd.isna(xq_all.loc[q, col]):
                continue  # indicator has nothing published for this quarter yet
            row = [xq_all.loc[q, col]]
            ok = True
            for lag in range(1, eq["n_ar"] + 1):
                q_lag = (q.to_period("Q") - lag).start_time
                if q_lag not in self.y_.index or pd.isna(self.y_.loc[q_lag]):
                    ok = False
                    break
                row.append(self.y_.loc[q_lag])
            if not ok:
                continue
            preds.append(float(eq["beta"] @ np.concatenate([[1.0], row])))

        if not preds:
            return Forecast(mean=float("nan"))
        return Forecast(mean=float(np.mean(preds)))


def _almon_weights(theta: np.ndarray, n_lags: int) -> np.ndarray:
    """Normalised exponential Almon weights w(j) for j = 0..n_lags-1."""
    j = np.arange(n_lags, dtype=float)
    z = theta[0] * j + theta[1] * j**2
    z = z - z.max()  # stabilise before exp
    w = np.exp(z)
    total = w.sum()
    return w / total if total > 0 else np.full(n_lags, 1.0 / n_lags)


class MIDASNowcaster:
    """MIDAS with exponential Almon weights on the monthly lag structure.

        y_q = a + b * sum_{j=0..K-1} w(j; theta) * x_{m(q) - j}

    where ``m(q)`` is the last month of quarter ``q`` that is actually
    published, so the weighting adapts to the ragged edge instead of assuming
    a complete quarter. Fitted by non-linear least squares over
    ``(a, b, theta)``; the Almon parameterisation keeps the lag polynomial
    parsimonious, which is the entire point of MIDAS on quarterly samples.

    Indicators are restricted to the monthly block: daily series would make
    ``K`` months of lags span thousands of observations with no gain, and
    quarterly series have no within-quarter structure to exploit.
    """

    name = "midas"

    def __init__(
        self,
        n_lags: int = 6,
        min_train_quarters: int = 20,
        max_indicators: int = 8,
    ):
        self.n_lags = n_lags
        self.min_train_quarters = min_train_quarters
        self.max_indicators = max_indicators
        self.models: dict[str, dict] = {}
        self.y_: pd.Series | None = None

    @staticmethod
    def _lag_matrix(
        x: pd.Series, quarters: list[pd.Timestamp], n_lags: int
    ) -> np.ndarray:
        """(len(quarters) x n_lags) of the last observed values up to each quarter."""
        x = x.dropna().sort_index()
        rows = []
        for q in quarters:
            q_end = q.to_period("Q").end_time
            hist = x.loc[:q_end]
            vals = hist.to_numpy()[-n_lags:][::-1]  # most recent first
            if len(vals) < n_lags:
                vals = np.concatenate([vals, np.full(n_lags - len(vals), np.nan)])
            rows.append(vals)
        return np.asarray(rows, dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from scipy.optimize import least_squares

        self.models = {}
        self.y_ = y.sort_index()
        if len(self.y_) < self.min_train_quarters:
            logger.warning(
                "%s: too few training quarters (%d)", self.name, len(self.y_)
            )
            return

        # Monthly block only: between 20 and 45 days between observations.
        monthly = []
        for col in X.columns:
            idx = X[col].dropna().index
            if len(idx) < 24:
                continue
            gap = pd.Series(idx).diff().dt.days.median()
            if 20 <= gap <= 45:
                monthly.append(col)
        monthly = monthly[: self.max_indicators]
        if not monthly:
            logger.warning("%s: no monthly indicators available", self.name)
            return

        quarters = list(self.y_.index)
        yv = self.y_.to_numpy()
        for col in monthly:
            L = self._lag_matrix(X[col], quarters, self.n_lags)
            ok = ~np.isnan(L).any(axis=1)
            if ok.sum() < self.min_train_quarters:
                continue
            Lo, yo = L[ok], yv[ok]

            def residuals(p, Lo=Lo, yo=yo):
                a, b, t1, t2 = p
                w = _almon_weights(np.array([t1, t2]), self.n_lags)
                return a + b * (Lo @ w) - yo

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    sol = least_squares(
                        residuals,
                        x0=np.array([float(yo.mean()), 0.0, -0.1, -0.01]),
                        max_nfev=2000,
                    )
                except Exception as exc:  # noqa: BLE001  one bad indicator is not fatal
                    logger.debug("%s: %s failed to converge: %s", self.name, col, exc)
                    continue
            if not sol.success:
                continue
            self.models[col] = {"params": sol.x}

        if not self.models:
            logger.warning("%s: no indicator converged", self.name)

    def predict(self, X: pd.DataFrame, target_quarter) -> Forecast:
        if not self.models:
            return Forecast(mean=float("nan"))
        q = pd.Timestamp(target_quarter)
        preds = []
        for col, m in self.models.items():
            L = self._lag_matrix(X[col], [q], self.n_lags)
            if np.isnan(L).any():
                continue
            a, b, t1, t2 = m["params"]
            w = _almon_weights(np.array([t1, t2]), self.n_lags)
            preds.append(float(a + b * (L[0] @ w)))
        if not preds:
            return Forecast(mean=float("nan"))
        return Forecast(mean=float(np.mean(preds)))


class DFMNowcaster:
    """Dynamic factor model on the ragged panel (Bańbura & Modugno 2014).

    Uses ``statsmodels`` ``DynamicFactorMQ``, which is the Bańbura-Modugno /
    NY Fed specification: mixed monthly-quarterly frequencies, arbitrary
    missing data handled natively by the Kalman filter, parameters by EM. That
    is chosen over a hand-rolled EM deliberately -- the ragged edge is the
    object of study here (H3), so the filter that handles it must be a
    standard, independently-tested implementation rather than our own. See.

    The factor count is BIC-selected over ``factor_candidates``.
    """

    name = "dfm"

    def __init__(
        self,
        factor_candidates: tuple[int, ...] = (1, 2, 3),
        min_train_quarters: int = 20,
        em_maxiter: int = 100,
        start: str | None = None,
        min_column_coverage: float = 0.5,
        divergence_sd: float = 50.0,
        robust_outliers: bool = False,
        outlier_mad: float = 6.0,
    ):
        self.factor_candidates = factor_candidates
        self.min_train_quarters = min_train_quarters
        self.em_maxiter = em_maxiter
        self.start = start
        self.min_column_coverage = min_column_coverage
        self.divergence_sd = divergence_sd
        self.robust_outliers = robust_outliers
        self.outlier_mad = outlier_mad
        self.n_masked_: int = 0
        self.result_ = None
        self.candidates_: list = []
        self.band_: tuple | None = None
        self.n_factors_: int | None = None
        self.columns_: list[str] | None = None
        self.start_: pd.Timestamp | None = None
        self.y_: pd.Series | None = None

    def _monthly_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Monthly-frequency view of the panel (daily/weekly averaged up).

        The panel's index reaches back to whichever series starts earliest
        (PPIACO begins 1913), which leaves a long stretch where almost every
        column is missing. That costs EM iterations without adding
        identification, so the frame is trimmed to the first month where at
        least ``min_column_coverage`` of columns are observed -- data-driven
        rather than a hardcoded date. Trimming 1913-1985 took the panel from
        1,230 to 366 monthly rows and the EM fit from 5.5 s to 0.8 s.

        ``start`` overrides the automatic trim (the orchestrator passes
        ``sample.train_start``), keeping the choice config-driven.
        """
        if X.empty:
            return pd.DataFrame()
        m = X.groupby(X.index.to_period("M")).mean()
        m.index = m.index.to_timestamp()
        # Keep columns with enough coverage for the filter to identify.
        keep = [c for c in m.columns if m[c].notna().sum() >= 24]
        m = m[keep]
        if m.empty:
            return m

        if self.start is not None:
            start = pd.Timestamp(self.start)
        else:
            coverage = m.notna().mean(axis=1)
            enough = coverage[coverage >= self.min_column_coverage]
            start = enough.index[0] if len(enough) else m.index[0]
        self.start_ = start
        return m.loc[start:].asfreq("MS")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        self.result_ = None
        self.candidates_ = []
        self.band_ = None
        self.y_ = y.sort_index()
        if len(self.y_) < self.min_train_quarters:
            logger.warning(
                "%s: too few training quarters (%d)", self.name, len(self.y_)
            )
            return

        monthly = self._monthly_frame(X)
        if monthly.empty:
            logger.warning("%s: no monthly block to estimate on", self.name)
            return
        self.columns_ = list(monthly.columns)

        yq = self.y_.copy()
        yq.index = pd.PeriodIndex(yq.index, freq="Q").to_timestamp(how="start")
        yq = yq.rename("target").asfreq("QS")

        if self.robust_outliers:
            # Treat extreme observations as MISSING for estimation only. On a
            # short post-COVID sample the 2020 quarters are ~9% of the data
            # and +-40-60% in size, which destroys the factor structure of a
            # Gaussian DFM. This is the same
            # remedy the NY Fed adopted for Nowcast 2.0.
            monthly, n_m = _mask_outliers(monthly, self.outlier_mad)
            yq_masked, n_y = _mask_outliers(yq.to_frame(), self.outlier_mad)
            yq = yq_masked["target"]
            self.n_masked_ = n_m + n_y
            if self.n_masked_:
                logger.info(
                    "%s: masked %d extreme observations for estimation "
                    "(> %.1f robust SD)",
                    self.name,
                    self.n_masked_,
                    self.outlier_mad,
                )

        # Every converged specification is kept, ordered by BIC. The choice
        # among them is deferred to `predict`, because the instability that
        # matters only shows up when the filter is re-applied to the current
        # information set -- see the STABILITY note there.
        candidates = []
        for k in self.factor_candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    mod = DynamicFactorMQ(
                        monthly,
                        endog_quarterly=yq.to_frame(),
                        factors=k,
                        factor_orders=1,
                        idiosyncratic_ar1=False,
                        standardize=True,
                    )
                    res = mod.fit(maxiter=self.em_maxiter, disp=False)
                except Exception as exc:  # noqa: BLE001 - try the next factor count
                    logger.debug("%s: k=%d failed: %s", self.name, k, exc)
                    continue
            bic = getattr(res, "bic", np.inf)
            candidates.append((bic if np.isfinite(bic) else np.inf, k, res))

        if not candidates:
            logger.warning("%s: no factor specification converged", self.name)
            return
        candidates.sort(key=lambda c: c[0])
        self.candidates_ = candidates
        self.result_ = candidates[0][2]
        self.n_factors_ = candidates[0][1]
        # DIVERGENCE band, from the training target's own scale so it is
        # data-driven. Deliberately very wide: the screen must separate
        # NUMERICAL DIVERGENCE from a legitimately EXTREME nowcast, and those
        # differ by orders of magnitude, not by a few standard deviations.
        # The observed failures were ~1e6-1e7 while the largest real quarterly
        # outcome in the sample is -31.7 (2020Q2). A +/-5 SD band around the
        # training range would have rejected a -24.5 nowcast of that -31.7
        # quarter -- i.e. made the DFM abstain precisely during COVID, the
        # subsample M5 reports. 50 SD (~+/-125 here) passes COVID and still
        # rejects divergence by four orders of magnitude.
        # Robust centre/scale: an SD inflated by COVID would widen the very
        # band meant to catch COVID-driven blowups (§49).
        centre = float(np.nanmedian(self.y_.to_numpy()))
        scale = _robust_scale(self.y_.to_numpy())
        if not np.isfinite(scale) or scale <= 0:
            scale = float(self.y_.std())
        self.band_ = (
            centre - self.divergence_sd * scale,
            centre + self.divergence_sd * scale,
        )

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        """Nowcast quarter ``target_quarter`` from the CURRENT information set.

        The fitted parameters are held fixed but the Kalman filter is re-run on
        the information set as of this nowcast date (``result_.apply``). That
        re-filtering is the whole mechanism of a nowcasting DFM: between
        refits, new monthly releases must move the nowcast. Calling
        ``result_.predict`` directly instead would freeze the nowcast at the
        last refit, giving an identical number for every date within a quarter
        and flattening exactly the within-cycle error profile RQ2 measures.

        STABILITY. With ~80 quarterly observations and ~20 monthly series, EM
        can reach a degenerate optimum: the likelihood improves but the state
        is not identified, and the nowcast diverges once the filter is
        re-applied. Measured on 2012-01-21, k=3 had the BEST in-sample BIC
        (11563 vs 13510 for k=2) yet produced -2.57e6, while k=2 gave +3.01
        -- so choosing on BIC alone actively PREFERS the broken
        specification. That is how a 2009-2012 stretch of the first backtest
        came out at |yhat| > 1e7 and an RMSE of 4.2 million.

        The in-sample fit does NOT reveal this (its fitted values stay in
        range), so specifications are screened HERE, on the value actually
        produced: candidates are tried in BIC order and the first whose
        nowcast falls inside a generous band around the training target is
        returned. If none does, the model abstains (NaN) rather than emitting
        a diverging number -- and abstention is visible in the results rather
        than flattering the DFM's error metrics.
        """
        if self.result_ is None or not self.candidates_:
            return Forecast(mean=float("nan"))
        q = pd.Timestamp(target_quarter)

        monthly = self._monthly_frame(X)
        if monthly.empty or self.columns_ is None:
            return Forecast(mean=float("nan"))
        monthly = monthly.reindex(columns=self.columns_)

        yq = y if y is not None else self.y_
        if yq is None:
            return Forecast(mean=float("nan"))
        yq = yq.copy()
        yq.index = pd.PeriodIndex(yq.index, freq="Q").to_timestamp(how="start")
        yq = yq.rename("target").asfreq("QS")

        # DynamicFactorMQ carries the quarterly variable on the MONTHLY index,
        # aligned to the quarter's THIRD month, so that is where the nowcast
        # for quarter q lives. Reading the quarter's first month instead
        # returns a partial-quarter artifact (-0.39 vs +2.42 in testing).
        q_last_month = q.to_period("Q").end_time.to_period("M").to_timestamp()
        lo, hi = self.band_ if self.band_ else (-np.inf, np.inf)

        for bic, k, res_k in self.candidates_:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = res_k.apply(
                        monthly,
                        endog_quarterly=yq.to_frame(),
                        retain_standardization=True,
                    )
                    pred = res.predict(start=monthly.index[-1], end=q_last_month)
                except Exception as exc:  # noqa: BLE001 - try the next candidate
                    logger.debug("%s: k=%d prediction failed: %s", self.name, k, exc)
                    continue

            if "target" not in getattr(pred, "columns", []):
                continue
            series = pred["target"].dropna()
            if series.empty or q_last_month not in series.index:
                continue
            value = float(series.loc[q_last_month])
            if not np.isfinite(value) or not (lo <= value <= hi):
                logger.info(
                    "%s: rejected k=%d at %s, nowcast %.3g diverged, "
                    "outside [%.1f, %.1f]",
                    self.name,
                    k,
                    q.date(),
                    value,
                    lo,
                    hi,
                )
                continue
            self.n_factors_ = k
            return Forecast(mean=value)

        logger.warning(
            "%s: no stable specification for %s; abstaining", self.name, q.date()
        )
        return Forecast(mean=float("nan"))


class RobustDFMNowcaster(DFMNowcaster):
    """DFM with extreme observations treated as missing during estimation.

    Same specification as ``DFMNowcaster`` in every other respect, so the
    comparison between them isolates ONE thing: whether outlier handling is
    what makes a Gaussian factor model estimable on a short post-COVID
    sample. This is the remedy the NY Fed
    adopted for Nowcast 2.0 after suspending v1 in 2021.
    """

    name = "dfm_robust"

    def __init__(self, **kw):
        kw.setdefault("robust_outliers", True)
        super().__init__(**kw)


ECON_MODELS = {
    NaiveMeanNowcaster.name: NaiveMeanNowcaster,
    NaiveAR1Nowcaster.name: NaiveAR1Nowcaster,
    BridgeNowcaster.name: BridgeNowcaster,
    MIDASNowcaster.name: MIDASNowcaster,
    DFMNowcaster.name: DFMNowcaster,
    RobustDFMNowcaster.name: RobustDFMNowcaster,
}
