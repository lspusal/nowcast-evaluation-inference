"""Classical ML baselines: RandomForest, XGBoost, LSTM.

Implements docs/02_methodology.md §3 (M2). Protocol aligned with Hopp (2024)
for comparability.

CRITICAL (R1): imputation of the ragged panel is BACKWARD-ONLY
(LOCF or expanding mean computed on the information set itself). Any
forward-looking fill is a fatal bug caught by tests/test_no_lookahead.py.

Design notes specific to this milestone
---------------------------------------
* ``backward_impute`` DELEGATES to ``src.data.impute.backward_impute`` rather
  than reimplementing the rule. Two implementations of "only use the past"
  means two chances to leak and only one of them covered by the anti-leakage
  tests.
* Features are the quarterly aggregate of the transformed ragged panel, plus
  target AR lags. A partially observed quarter is summarised from the months
  actually published, so the ragged edge survives into the design matrix.
* The three models share ``_design_matrix`` so that any difference in their
  scores is a difference in the learner, not in the features it was given.
* LSTM early stopping validates on the LAST few quarters of the training
  window (a contiguous suffix), never a random split: a random split would
  let the model tune on quarters that come after the ones it trains on, which
  is a leak even though every quarter involved predates the nowcast date.
  The scaler is fitted on the training slice only, for the same reason.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.impute import backward_impute as _backward_impute
from src.data.info_set import to_quarterly
from src.models.econ_baselines import Forecast

logger = logging.getLogger(__name__)

N_AR_LAGS = 4  # quarters of target history offered to every ML model


def backward_impute(X: pd.DataFrame, method: str = "locf") -> pd.DataFrame:
    """Impute missing values using ONLY past information (R1).

    ``locf`` carries the last observed value forward; ``expanding_mean`` uses
    the mean of everything observed so far. Both are strictly backward, and
    a cell with no prior observation stays NaN in either case.
    """
    if method == "locf":
        return _backward_impute(X)
    if method == "expanding_mean":
        return X.fillna(X.expanding(min_periods=1).mean())
    raise ValueError(f"unsupported imputation method: {method!r}")


def _design_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    target_quarter: pd.Timestamp | None = None,
    impute_method: str = "locf",
    n_ar: int = N_AR_LAGS,
) -> tuple[pd.DataFrame, pd.Series]:
    """(features, aligned target) at quarterly frequency.

    Rows are quarters. Columns are the imputed quarterly indicator aggregates
    plus ``n_ar`` lags of the target. When ``target_quarter`` is given, its row
    is returned too (with a NaN target) so ``predict`` can use it.
    """
    xq = to_quarterly(X)
    if xq.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    xq = backward_impute(xq, impute_method)
    frame = xq.copy()
    frame["_y"] = y.reindex(frame.index)

    # AR features are POSITION-based, not calendar-based: for each row the
    # ``n_ar`` most recent target values PUBLISHED STRICTLY BEFORE that
    # quarter, whichever quarters those happen to be.
    #
    # Calendar lags (y_{q-1}, y_{q-2}, ...) look more natural but are wrong
    # here. The second release of q-1 is typically not out until well into
    # q+1, so demanding y_{q-1} makes the model abstain through most of the
    # publication cycle -- measured coverage 37/72 vs 72/72 on a one-year
    # smoke test. A real-time practitioner (and Hopp 2024's protocol) uses
    # the latest published GDP figure whatever quarter it is, so that is what
    # is supplied. The rule is identical for training and prediction rows and
    # only ever reads published values, so it stays strictly backward (R1).
    published = y.dropna().sort_index()
    ar_cols = {f"y_lag{k}": [] for k in range(1, n_ar + 1)}
    staleness = []
    for q in frame.index:
        prior = published.loc[published.index < q]
        recent = prior.to_numpy()[-n_ar:][::-1]  # most recent first
        for k in range(1, n_ar + 1):
            ar_cols[f"y_lag{k}"].append(recent[k - 1] if len(recent) >= k else np.nan)
        # How stale the freshest AR value is, in quarters. The model can use
        # this: a nowcast made with a 2-quarter-old GDP reading is a different
        # problem from one made with last quarter's.
        if len(prior):
            gap = (q.to_period("Q") - prior.index[-1].to_period("Q")).n
        else:
            gap = np.nan
        staleness.append(gap)

    for col, values in ar_cols.items():
        frame[col] = values
    frame["y_staleness_q"] = staleness

    feature_cols = [c for c in frame.columns if c != "_y"]
    # Drop rows where any AR feature is missing; indicator columns are already
    # imputed backward, so remaining NaNs there mean "no history yet".
    usable = frame[feature_cols].notna().all(axis=1)
    frame = frame[usable]
    return frame[feature_cols], frame["_y"]


class _SklearnStyleNowcaster:
    """Shared fit/predict plumbing for the tree ensembles."""

    name = "base"

    def __init__(
        self, impute_method: str = "locf", min_train_quarters: int = 20, **params
    ):
        self.params = params
        self.impute_method = impute_method
        self.min_train_quarters = min_train_quarters
        self.model_ = None
        self.columns_: list[str] | None = None

    def _make_model(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model_ = None
        feats, target = _design_matrix(X, y, impute_method=self.impute_method)
        train = target.notna()
        if int(train.sum()) < self.min_train_quarters:
            logger.warning(
                "%s: only %d usable training quarters", self.name, int(train.sum())
            )
            return
        self.columns_ = list(feats.columns)
        model = self._make_model()
        model.fit(feats[train].to_numpy(), target[train].to_numpy())
        self.model_ = model

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        if self.model_ is None or self.columns_ is None:
            return Forecast(mean=float("nan"))
        q = pd.Timestamp(target_quarter)
        feats, _ = _design_matrix(
            X,
            y if y is not None else pd.Series(dtype=float),
            impute_method=self.impute_method,
        )
        if q not in feats.index:
            return Forecast(mean=float("nan"))
        row = feats.reindex(columns=self.columns_).loc[[q]]
        if row.isna().any(axis=1).iloc[0]:
            return Forecast(mean=float("nan"))
        return Forecast(mean=float(self.model_.predict(row.to_numpy())[0]))


class RFNowcaster(_SklearnStyleNowcaster):
    name = "random_forest"

    def _make_model(self):
        from sklearn.ensemble import RandomForestRegressor

        params = {
            "n_estimators": self.params.get("n_estimators", 500),
            "max_depth": self.params.get("max_depth", None),
            "random_state": self.params.get("random_state", 42),
            "n_jobs": self.params.get("n_jobs", -1),
        }
        return RandomForestRegressor(**params)


class XGBNowcaster(_SklearnStyleNowcaster):
    name = "xgboost"

    def _make_model(self):
        from xgboost import XGBRegressor

        params = {
            "n_estimators": self.params.get("n_estimators", 800),
            "max_depth": self.params.get("max_depth", 4),
            "learning_rate": self.params.get("learning_rate", 0.05),
            "random_state": self.params.get("random_state", 42),
            "n_jobs": self.params.get("n_jobs", -1),
            "verbosity": 0,
        }
        return XGBRegressor(**params)


class LSTMNowcaster:
    """LSTM over rolling windows of quarterly features.

    Early stopping validates on a contiguous SUFFIX of the training window
    (the most recent ``val_quarters``), never a random split -- with time
    series a random split tunes on quarters that follow the ones it fits,
    which leaks even though all of them precede the nowcast date. The feature
    scaler is fitted on the training slice only.
    """

    name = "lstm"

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 1,
        epochs: int = 100,
        patience: int = 10,
        batch: int = 16,
        window: int = 8,
        val_quarters: int = 12,
        lr: float = 1e-3,
        seed: int = 42,
        impute_method: str = "locf",
        min_train_quarters: int = 40,
        device: str | None = None,
    ):
        self.hidden = hidden
        self.layers = layers
        self.epochs = epochs
        self.patience = patience
        self.batch = batch
        self.window = window
        self.val_quarters = val_quarters
        self.lr = lr
        self.seed = seed
        self.impute_method = impute_method
        self.min_train_quarters = min_train_quarters
        self.device = device
        self.model_ = None
        self.columns_: list[str] | None = None
        self.scaler_: tuple[np.ndarray, np.ndarray] | None = None
        self.best_epoch_: int | None = None

    @staticmethod
    def _windows(A: np.ndarray, window: int) -> np.ndarray:
        """(n-window+1, window, n_features) rolling windows."""
        n = len(A)
        if n < window:
            return np.empty((0, window, A.shape[1]))
        idx = np.arange(window)[None, :] + np.arange(n - window + 1)[:, None]
        return A[idx]

    def _resolve_device(self):
        import torch

        if self.device:
            return torch.device(self.device)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        import torch
        from torch import nn

        self.model_ = None
        feats, target = _design_matrix(X, y, impute_method=self.impute_method)
        mask = target.notna()
        feats, target = feats[mask], target[mask]
        if len(target) < self.min_train_quarters:
            logger.warning("%s: only %d usable quarters", self.name, len(target))
            return

        self.columns_ = list(feats.columns)
        # float32: MPS has no float64 support, and it is the standard
        # precision for neural-net training anyway.
        A = feats.to_numpy(dtype=np.float32)
        yv = target.to_numpy(dtype=np.float32)

        n_val = min(self.val_quarters, max(len(yv) // 5, 1))
        n_train = len(yv) - n_val
        if n_train <= self.window:
            logger.warning("%s: training slice too short after val split", self.name)
            return

        # Scaler on the TRAINING slice only.
        mu = A[:n_train].mean(axis=0).astype(np.float32)
        sd = A[:n_train].std(axis=0).astype(np.float32)
        sd[sd < 1e-12] = 1.0
        self.scaler_ = (mu, sd)
        As = (A - mu) / sd

        Wtr = self._windows(As[:n_train], self.window)
        ytr = yv[self.window - 1 : n_train]
        Wva = self._windows(As[: n_train + n_val], self.window)[-n_val:]
        yva = yv[-n_val:]
        if len(Wtr) == 0 or len(Wva) == 0:
            logger.warning("%s: not enough windows", self.name)
            return

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self._resolve_device()

        class _Net(nn.Module):
            def __init__(self, n_features, hidden, layers):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, hidden, num_layers=layers, batch_first=True
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        net = _Net(A.shape[1], self.hidden, self.layers).to(device).float()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xtr = torch.tensor(Wtr, device=device)
        Ytr = torch.tensor(ytr, device=device)
        Xva = torch.tensor(Wva, device=device)
        Yva = torch.tensor(yva, device=device)

        best_loss, best_state, bad = float("inf"), None, 0
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xtr), device=device)
            for i in range(0, len(Xtr), self.batch):
                sel = perm[i : i + self.batch]
                opt.zero_grad()
                loss = loss_fn(net(Xtr[sel]), Ytr[sel])
                loss.backward()
                opt.step()

            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(Xva), Yva).item())
            if val < best_loss - 1e-9:
                best_loss, bad = val, 0
                best_state = {
                    k: v.detach().clone() for k, v in net.state_dict().items()
                }
                self.best_epoch_ = epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model_ = net
        self._device_ = device

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        import torch

        if self.model_ is None or self.columns_ is None or self.scaler_ is None:
            return Forecast(mean=float("nan"))
        q = pd.Timestamp(target_quarter)
        feats, _ = _design_matrix(
            X,
            y if y is not None else pd.Series(dtype=float),
            impute_method=self.impute_method,
        )
        if q not in feats.index:
            return Forecast(mean=float("nan"))

        feats = feats.reindex(columns=self.columns_)
        upto = feats.loc[:q]
        if len(upto) < self.window or upto.tail(self.window).isna().any().any():
            return Forecast(mean=float("nan"))

        mu, sd = self.scaler_
        A = ((upto.tail(self.window).to_numpy(dtype=np.float32) - mu) / sd).astype(
            np.float32
        )
        with torch.no_grad():
            out = self.model_(torch.tensor(A[None, :, :], device=self._device_))
        value = float(out.item())
        return Forecast(mean=value if np.isfinite(value) else float("nan"))


ML_MODELS = {
    RFNowcaster.name: RFNowcaster,
    XGBNowcaster.name: XGBNowcaster,
    LSTMNowcaster.name: LSTMNowcaster,
}
