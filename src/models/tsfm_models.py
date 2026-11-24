"""Time series foundation model wrappers (M3 zero-shot, M4 fine-tuning).

Implements docs/02_methodology.md §3. Open-weights models only:
Chronos(-Bolt), Lag-Llama, TimesFM. TimeGPT is
optional and disabled by default (use_timegpt: false).

Uniform interface:
    class TSFMWrapper:
        name: str
        def predict(self, history: pd.Series, horizon: int) -> Forecast
        def predict_batch(self, histories, horizon) -> list[Forecast]

WHY THE STRATEGY MATTERS
---------------------------------------------------
A TSFM is a UNIVARIATE forecaster. Feeding it the quarterly GDP series and
extrapolating uses none of the 22 indicators, so it would lose to the DFM
for a reason that has nothing to do with foundation models -- it would
"confirm" H1 trivially. Two strategies are therefore implemented, and the
comparison between them is the actual experiment:

* ``aggregate_q`` -- univariate extrapolation of the quarterly target. This
  is the honest "can a general-purpose forecaster nowcast GDP out of the
  box" baseline. It is expected to be weak, and its weakness is informative
  only because the second strategy exists.

* ``ragged_edge`` -- the TSFM forecasts each INDICATOR's unpublished months,
  completing the ragged edge, and a bridge regression maps the completed
  quarterly panel to GDP. This is where a TSFM can plausibly add value in
  nowcasting, and it is the direct test of H3 ("the bottleneck is the
  information set, not the architecture"): if H3 holds, completing the
  ragged edge with a better extrapolator should move the needle far more
  than swapping the architecture does.

Both strategies are strictly zero-shot with respect to the TARGET (H1): the
foundation model never sees GDP growth in training or fine-tuning at M3. The
bridge regression in ``ragged_edge`` is fitted on published target values
only, exactly as the M1 bridge is.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.info_set import to_quarterly
from src.models.econ_baselines import Forecast, _ols

logger = logging.getLogger(__name__)

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
MIN_CONTEXT = 8  # a foundation model needs some history to condition on


# Loading a Chronos pipeline costs ~1s warm (and ~2min cold, downloading
# weights). A backtest instantiates a model per refit, so without this cache
# the same weights would be re-loaded dozens of times per run.
_PIPELINE_CACHE: dict[tuple, object] = {}


def _resolve_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ChronosWrapper:
    """Chronos/Chronos-Bolt wrapper (Amazon, open weights).

    Chronos is natively probabilistic, so quantiles come free. The pipeline
    is loaded lazily and cached on the instance: re-loading per nowcast date
    would dominate the runtime of a 1,400-date backtest.
    """

    def __init__(
        self,
        repo: str = "amazon/chronos-bolt-base",
        mode: str = "zero_shot",
        device: str | None = None,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    ):
        self.repo, self.mode = repo, mode
        self.name = f"chronos_{mode}"
        self.device = device
        self.quantiles = quantiles
        self._pipeline = None
        self.load_seconds_: float | None = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            import torch
            from chronos import BaseChronosPipeline

            device = _resolve_device(self.device)
            key = (self.repo, device)
            t0 = time.time()
            if key not in _PIPELINE_CACHE:
                _PIPELINE_CACHE[key] = BaseChronosPipeline.from_pretrained(
                    self.repo,
                    device_map=device,
                    dtype=torch.float32,  # MPS has no float64
                )
            self._pipeline = _PIPELINE_CACHE[key]
            self.load_seconds_ = time.time() - t0
            logger.info(
                "%s: loaded %s in %.1fs", self.name, self.repo, self.load_seconds_
            )
        return self._pipeline

    def predict_batch(self, histories: list[pd.Series], horizon: int) -> list[Forecast]:
        """Forecast many series at once -- the batched path is ~20x faster."""
        import torch

        usable, index_map = [], []
        for i, h in enumerate(histories):
            values = pd.Series(h).dropna().to_numpy(dtype="float32")
            if len(values) >= MIN_CONTEXT and np.isfinite(values).all():
                usable.append(torch.tensor(values))
                index_map.append(i)

        out: list[Forecast] = [Forecast(mean=float("nan"))] * len(histories)
        if not usable:
            return out

        # chronos renamed the first parameter from `context` to `inputs`
        # between 1.x and 2.x, so it is passed POSITIONALLY: pinning either
        # name would break the other version, and a replication package
        # should not be hostage to that rename.
        qs, _mean = self.pipeline.predict_quantiles(
            usable,
            prediction_length=horizon,
            quantile_levels=list(self.quantiles),
        )
        qs = qs.detach().cpu().numpy()  # (batch, horizon, n_quantiles)
        median_idx = list(self.quantiles).index(0.5) if 0.5 in self.quantiles else 0
        for slot, i in enumerate(index_map):
            last = qs[slot, horizon - 1, :]
            out[i] = Forecast(
                mean=float(last[median_idx]),
                q10=float(last[0]),
                q50=float(last[median_idx]),
                q90=float(last[-1]),
            )
        return out

    def predict(self, history: pd.Series, horizon: int) -> Forecast:
        return self.predict_batch([history], horizon)[0]


LAGLLAMA_CKPT = Path("data/raw/lagllama/lag-llama.ckpt")


class LagLlamaWrapper:
    """Lag-Llama wrapper (open-source checkpoint, GluonTS-based).

    The package installs from GitHub and the checkpoint from the HF hub
    rather than either coming from a package index, so both are checked at
    call time and the model abstains cleanly when absent -- an abstention
    shows up in ``coverage.csv`` instead of the model silently vanishing from
    the comparison.

    Lag-Llama is probabilistic by sampling, so quantiles come from the sample
    paths rather than a quantile head.
    """

    def __init__(
        self,
        checkpoint: str | Path = LAGLLAMA_CKPT,
        mode: str = "zero_shot",
        device: str | None = None,
        context_length: int = 256,
        # 30 rather than 100 sample paths: measured 5.6s vs 21.0s per batch
        # of 22 series, i.e. 2.2h vs 8.2h over the 1,403-date backtest, for a
        # median forecast within 1.6% of the 100-sample reference. Only the
        # median enters RMSE. The master plan explicitly permits reducing
        # scope when a run is this expensive, provided it is documented
        #.
        num_samples: int = 30,
        max_horizon: int = 8,
        **kw,
    ):
        self.checkpoint = Path(checkpoint)
        self.mode = mode
        self.name = f"lagllama_{mode}"
        self.device = device
        self.context_length = context_length
        self.num_samples = num_samples
        self.max_horizon = max_horizon
        self.available_ = None
        self.load_seconds_: float | None = None

    def _predictor(self, horizon: int):
        key = ("lagllama", str(self.checkpoint), horizon, self.context_length)
        if key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[key]

        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        device = _resolve_device(self.device)
        t0 = time.time()

        # PyTorch 2.6 flipped `torch.load(weights_only=)` to True by default,
        # and LagLlamaEstimator calls torch.load INTERNALLY on the checkpoint
        # path, so the flag cannot be passed through. This shim restores the
        # old default for the duration of construction only.
        #
        # Safe here because the checkpoint is the official one pulled from
        # `time-series-foundation-models/Lag-Llama` on the HF hub (see
        #), not arbitrary user input. Anything
        # broader in scope than this would not be justifiable.
        _orig_load = torch.load

        def _load_trusted(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)

        torch.load = _load_trusted
        try:
            ckpt = torch.load(self.checkpoint, map_location=device)
            args = ckpt["hyper_parameters"]["model_kwargs"]
            estimator = LagLlamaEstimator(
                ckpt_path=str(self.checkpoint),
                prediction_length=horizon,
                context_length=self.context_length,
                input_size=args["input_size"],
                n_layer=args["n_layer"],
                n_embd_per_head=args["n_embd_per_head"],
                n_head=args["n_head"],
                scaling=args["scaling"],
                time_feat=args["time_feat"],
                batch_size=32,
                num_parallel_samples=self.num_samples,
                device=torch.device(device),
            )
            lightning = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, lightning)
        finally:
            torch.load = _orig_load
        self.load_seconds_ = time.time() - t0
        logger.info("%s: loaded checkpoint in %.1fs", self.name, self.load_seconds_)
        _PIPELINE_CACHE[key] = predictor
        return predictor

    def predict_batch(self, histories: list[pd.Series], horizon: int) -> list[Forecast]:
        if self.available_ is None:
            ok = True
            try:
                import lag_llama  # noqa: F401
            except ImportError:
                logger.warning("%s: lag-llama not installed; abstaining", self.name)
                ok = False
            if ok and not self.checkpoint.exists():
                logger.warning(
                    "%s: checkpoint missing at %s; abstaining",
                    self.name,
                    self.checkpoint,
                )
                ok = False
            self.available_ = ok

        out: list[Forecast] = [Forecast(mean=float("nan"))] * len(histories)
        if not self.available_:
            return out

        from gluonts.dataset.pandas import PandasDataset

        horizon = min(horizon, self.max_horizon)
        frames, index_map = {}, []
        for i, h in enumerate(histories):
            v = pd.Series(h).dropna()
            if len(v) < MIN_CONTEXT or not np.isfinite(v.to_numpy()).all():
                continue
            v = v.tail(self.context_length).astype("float32")
            # GluonTS needs a regular index; the series' own spacing is
            # irrelevant to a lag-based model, so a synthetic monthly index is
            # used purely to satisfy the dataset contract.
            v.index = pd.period_range(end="2020-01", periods=len(v), freq="M")
            frames[f"s{i}"] = v
            index_map.append(i)
        if not frames:
            return out

        ds = PandasDataset(frames, target=None, freq="M")
        try:
            forecasts = list(self._predictor(horizon).predict(ds))
        except Exception as exc:  # noqa: BLE001 - abstain rather than abort
            logger.warning("%s: inference failed (%s); abstaining", self.name, exc)
            return out

        for slot, fc in enumerate(forecasts):
            if slot >= len(index_map):
                break
            samples = np.asarray(fc.samples)[:, horizon - 1]
            out[index_map[slot]] = Forecast(
                mean=float(np.median(samples)),
                q10=float(np.quantile(samples, 0.1)),
                q50=float(np.median(samples)),
                q90=float(np.quantile(samples, 0.9)),
            )
        return out

    def predict(self, history: pd.Series, horizon: int) -> Forecast:
        return self.predict_batch([history], horizon)[0]


class TimesFMWrapper:
    """TimesFM 2.5 wrapper (Google, open weights).

    Uses the 200M PyTorch checkpoint rather than the 500M one named in
    ``configs/experiment_us.yaml``: the master plan caps the compute budget at
    "adapters de modelos <= 200M parametros", so the 200M variant is the one
    that fits the stated budget. TimesFM needs
    Python >= 3.10, which is why it could not run before the environment was
    rebuilt on 3.11.

    Requires an explicit ``compile`` step declaring max context/horizon, so
    the model is built once and cached per (repo, max_horizon).
    """

    def __init__(
        self,
        repo: str = "google/timesfm-2.5-200m-pytorch",
        mode: str = "zero_shot",
        device: str | None = None,
        max_context: int = 512,
        max_horizon: int = 8,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        **kw,
    ):
        self.repo, self.mode = repo, mode
        self.name = f"timesfm_{mode}"
        self.device = device
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.quantiles = quantiles
        self.available_ = None
        self.load_seconds_: float | None = None

    @property
    def model(self):
        key = ("timesfm", self.repo, self.max_horizon)
        if key in _PIPELINE_CACHE:
            return _PIPELINE_CACHE[key]
        import timesfm

        t0 = time.time()
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo)
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
        )
        self.load_seconds_ = time.time() - t0
        logger.info("%s: loaded %s in %.1fs", self.name, self.repo, self.load_seconds_)
        _PIPELINE_CACHE[key] = model
        return model

    def predict_batch(self, histories: list[pd.Series], horizon: int) -> list[Forecast]:
        if self.available_ is None:
            try:
                import timesfm  # noqa: F401

                self.available_ = True
            except ImportError:
                logger.warning(
                    "%s: timesfm not installed (needs Python >= 3.10); abstaining",
                    self.name,
                )
                self.available_ = False
        out: list[Forecast] = [Forecast(mean=float("nan"))] * len(histories)
        if not self.available_:
            return out

        usable, index_map = [], []
        for i, h in enumerate(histories):
            v = pd.Series(h).dropna().to_numpy(dtype="float32")
            if len(v) >= MIN_CONTEXT and np.isfinite(v).all():
                usable.append(v[-self.max_context :])
                index_map.append(i)
        if not usable:
            return out

        horizon = min(horizon, self.max_horizon)
        point, quantiles = self.model.forecast(horizon=horizon, inputs=usable)
        point = np.asarray(point)
        quantiles = np.asarray(quantiles)
        for slot, i in enumerate(index_map):
            mean = float(point[slot, horizon - 1])
            # The quantile head returns 10 levels (mean + deciles); index
            # defensively so a layout change degrades to point-only rather
            # than silently mislabelling a quantile.
            q10 = q90 = None
            if quantiles.ndim == 3 and quantiles.shape[2] >= 10:
                q10 = float(quantiles[slot, horizon - 1, 1])
                q90 = float(quantiles[slot, horizon - 1, 9])
            out[i] = Forecast(mean=mean, q10=q10, q50=mean, q90=q90)
        return out

    def predict(self, history: pd.Series, horizon: int) -> Forecast:
        return self.predict_batch([history], horizon)[0]


TSFM_WRAPPERS = {
    "chronos": ChronosWrapper,
    "lagllama": LagLlamaWrapper,
    "timesfm": TimesFMWrapper,
}


class TSFMNowcaster:
    """Adapts a univariate TSFM to the GDP nowcasting task.

    Implements the two frequency strategies described in the module
    docstring. ``fit`` never shows the foundation model any target data --
    at M3 the model is frozen and zero-shot (H1). For ``ragged_edge`` the
    only estimated component is the bridge regression, fitted on published
    target values exactly as the M1 bridge is.
    """

    def __init__(
        self,
        backend: str = "chronos",
        strategy: str = "aggregate_q",
        repo: str | None = None,
        device: str | None = None,
        min_train_quarters: int = 20,
        max_context: int = 512,
        **kw,
    ):
        if strategy not in ("aggregate_q", "ragged_edge"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if backend not in TSFM_WRAPPERS:
            raise ValueError(f"unknown backend {backend!r}")
        self.backend_name = backend
        self.strategy = strategy
        self.min_train_quarters = min_train_quarters
        self.max_context = max_context
        kwargs = {"mode": "zero_shot", "device": device}
        if repo:
            kwargs["repo"] = repo
        self.wrapper = TSFM_WRAPPERS[backend](**kwargs)
        self.name = f"{backend}_{strategy}"
        self.y_: pd.Series | None = None
        self.bridge_: dict[str, dict] = {}
        self.predict_seconds_: float | None = None

    # -- fitting -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.y_ = y.sort_index()
        self.bridge_ = {}
        if self.strategy != "ragged_edge":
            return  # aggregate_q is fully zero-shot: nothing is estimated
        if len(self.y_) < self.min_train_quarters:
            logger.warning("%s: too few training quarters", self.name)
            return

        # Bridge from COMPLETE quarterly indicator averages to GDP. Fitted on
        # history only; the TSFM's role at predict time is to complete the
        # current quarter so this mapping can be applied to it.
        xq = to_quarterly(X)
        for col in xq.columns:
            frame = pd.DataFrame({"x": xq[col], "y": self.y_}).dropna()
            if len(frame) < self.min_train_quarters:
                continue
            beta, sigma2 = _ols(frame[["x"]].to_numpy(), frame["y"].to_numpy())
            self.bridge_[col] = {"beta": beta, "sigma2": sigma2}

    # -- prediction --------------------------------------------------------
    def _predict_aggregate_q(self, target_quarter: pd.Timestamp) -> Forecast:
        """Univariate extrapolation of the quarterly target itself."""
        if self.y_ is None or self.y_.empty:
            return Forecast(mean=float("nan"))
        history = self.y_.loc[self.y_.index < target_quarter].dropna()
        if len(history) < MIN_CONTEXT:
            return Forecast(mean=float("nan"))
        # Horizon = how many quarters past the last published one we are.
        gap = (target_quarter.to_period("Q") - history.index[-1].to_period("Q")).n
        if gap < 1:
            return Forecast(mean=float("nan"))
        return self.wrapper.predict(history.tail(self.max_context), horizon=gap)

    def _predict_ragged_edge(
        self, X: pd.DataFrame, target_quarter: pd.Timestamp
    ) -> Forecast:
        """Complete each indicator's missing months, then bridge to GDP."""
        if not self.bridge_ or self.y_ is None:
            # Was silent, which cost real debugging time: a full-fine-tuning
            # run produced 1,403 NaNs with an empty log because this branch
            # said nothing. Any path that abstains
            # must say why.
            logger.warning(
                "%s: abstaining at %s -- bridge has %d equations, y_ is %s",
                self.name,
                pd.Timestamp(target_quarter).date(),
                len(self.bridge_),
                "None" if self.y_ is None else f"{len(self.y_)} quarters",
            )
            return Forecast(mean=float("nan"))

        q_start = target_quarter
        q_end = target_quarter.to_period("Q").end_time
        # Monthly view: what is published for the target quarter, per series.
        monthly = X.groupby(X.index.to_period("M")).mean()
        monthly.index = monthly.index.to_timestamp()

        cols, histories, needed = [], [], []
        observed_means = {}
        for col in self.bridge_:
            if col not in monthly.columns:
                continue
            s = monthly[col].dropna()
            in_q = s.loc[(s.index >= q_start) & (s.index <= q_end)]
            n_missing = 3 - len(in_q)
            if n_missing <= 0:
                observed_means[col] = float(in_q.mean())
                continue
            hist = s.loc[s.index < q_start]
            if len(hist) < MIN_CONTEXT:
                continue
            cols.append(col)
            histories.append(hist.tail(self.max_context))
            needed.append((n_missing, in_q.to_numpy()))

        # One batched call for every series that needs completing.
        if histories:
            t0 = time.time()
            max_h = max(n for n, _ in needed)
            forecasts = self.wrapper.predict_batch(histories, horizon=max_h)
            self.predict_seconds_ = time.time() - t0
            for col, (n_missing, observed), fc in zip(cols, needed, forecasts):
                if not np.isfinite(fc.mean):
                    continue
                # Chronos returns the horizon end; use it for every missing
                # month (a flat completion), which is the conservative choice:
                # it does not invent within-quarter dynamics the model was not
                # asked for.
                filled = np.concatenate([observed, np.repeat(fc.mean, n_missing)])
                observed_means[col] = float(np.mean(filled))

        preds = []
        for col, mean_x in observed_means.items():
            eq = self.bridge_.get(col)
            if eq is None or not np.isfinite(mean_x):
                continue
            preds.append(float(eq["beta"] @ np.array([1.0, mean_x])))
        if not preds:
            logger.warning(
                "%s: abstaining at %s -- %d series had a usable quarterly "
                "mean, %d bridge equations, %d TSFM completions attempted",
                self.name,
                pd.Timestamp(target_quarter).date(),
                len(observed_means),
                len(self.bridge_),
                len(histories),
            )
            return Forecast(mean=float("nan"))
        return Forecast(mean=float(np.mean(preds)))

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        if y is not None:
            self.y_ = y.sort_index()
        q = pd.Timestamp(target_quarter)
        if self.strategy == "aggregate_q":
            return self._predict_aggregate_q(q)
        return self._predict_ragged_edge(X, q)


class ChronosFineTuner:
    """Fine-tuning of Chronos on the panel available at t (M4, RQ3).

    Supports both modes the master plan asks for:

    * ``mode="lora"`` -- adapters only (1.77M of 207M trainable, 0.85%).
    * ``mode="full"`` -- every parameter trainable.

    Counter-intuitively, FULL fine-tuning is FASTER per step here (75 ms vs
    238 ms measured on MPS): PEFT inserts extra matrix products and forward
    hooks, and on a 205M model that overhead exceeds what is saved by
    computing fewer gradients. So "full" is not the expensive option at this
    scale -- the whole 21-refit sweep costs ~0.4 h.

    The learning rate DIFFERS BY MODE and this is deliberate: 1e-4 for
    adapters, 1e-5 for full fine-tuning. Applying the adapter rate to all
    205M parameters on ~12k windows would risk catastrophic forgetting of the
    pretraining that is the entire reason to use a foundation model.

    R1 IS THE WHOLE DIFFICULTY HERE. Fine-tuning is the one place a
    foundation model can be handed its own future, so the training set is
    built EXCLUSIVELY from the information set at the nowcast date: windows
    drawn from ``X`` (already vintage-filtered by ``information_set``), never
    from the target and never from a later vintage. A model fine-tuned at
    ``t`` therefore cannot have seen any observation published after ``t``.

    The budget sweep is NESTED, so training runs ONCE to the largest budget
    and snapshots the adapter at each milestone. Three separate runs would
    cost 3x for identical results.
    """

    def __init__(
        self,
        repo: str = "amazon/chronos-bolt-base",
        budgets: tuple[int, ...] = (50, 200, 800),
        context_length: int = 128,
        prediction_length: int = 3,
        batch_size: int = 8,
        lr: float | None = None,
        lora_r: int = 8,
        seed: int = 42,
        device: str | None = None,
        mode: str = "lora",
    ):
        if mode not in ("lora", "full"):
            raise ValueError(f"mode must be 'lora' or 'full', got {mode!r}")
        self.mode = mode
        # Full fine-tuning needs a much smaller step than adapters, or the
        # pretraining is simply overwritten (see class docstring).
        self.lr = lr if lr is not None else (1e-4 if mode == "lora" else 1e-5)
        self.repo = repo
        self.budgets = tuple(sorted(b for b in budgets if b > 0))
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.lora_r = lora_r
        self.seed = seed
        self.device = device
        self.adapters_: dict[int, dict] = {}
        self.train_seconds_: dict[int, float] = {}
        self.n_windows_: int | None = None
        self.pipeline_ = None  # private, never the shared cache

    def _windows(self, X: pd.DataFrame) -> np.ndarray:
        """Training windows from the information set only (R1)."""
        need = self.context_length + self.prediction_length
        out = []
        for col in X.columns:
            v = X[col].dropna().to_numpy(dtype="float32")
            if len(v) < need:
                continue
            idx = np.arange(need)[None, :] + np.arange(len(v) - need + 1)[:, None]
            w = v[idx]
            keep = np.isfinite(w).all(axis=1) & (w.std(axis=1) > 1e-8)
            out.append(w[keep])
        if not out:
            return np.empty((0, need), dtype="float32")
        return np.concatenate(out, axis=0)

    def fit(self, X: pd.DataFrame, seed: int | None = None) -> None:
        import torch
        from chronos import BaseChronosPipeline
        from peft import LoraConfig, get_peft_model

        windows = self._windows(X)
        self.n_windows_ = len(windows)
        if len(windows) < self.batch_size:
            logger.warning(
                "chronos_ft: only %d training windows; skipping", len(windows)
            )
            return

        device = _resolve_device(self.device)
        # A PRIVATE pipeline, deliberately bypassing _PIPELINE_CACHE: PEFT
        # injects LoRA layers IN PLACE and ChronosBoltPipeline has
        # `inner_model is model`, so wrapping the shared object would turn
        # every zero-shot Chronos model in the process into a fine-tuned one.
        self.pipeline_ = BaseChronosPipeline.from_pretrained(
            self.repo, device_map=device, dtype=torch.float32
        )
        base = self.pipeline_.inner_model
        if self.mode == "lora":
            cfg = LoraConfig(
                r=self.lora_r,
                lora_alpha=2 * self.lora_r,
                target_modules=["q", "k", "v", "o"],
                lora_dropout=0.05,
                bias="none",
            )
            model = get_peft_model(base, cfg)
        else:
            model = base
            for prm in model.parameters():
                prm.requires_grad_(True)
        opt = torch.optim.AdamW(
            [q for q in model.parameters() if q.requires_grad], lr=self.lr
        )
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        torch.manual_seed(seed if seed is not None else self.seed)

        ctx_np = windows[:, : self.context_length]
        tgt_np = windows[:, self.context_length :]
        model.train()
        t0 = time.time()
        step = 0
        for budget in self.budgets:
            while step < budget:
                sel = rng.integers(0, len(windows), size=self.batch_size)
                ctx = torch.tensor(ctx_np[sel], device=device)
                tgt = torch.tensor(tgt_np[sel], device=device)
                opt.zero_grad()
                loss = model(context=ctx, target=tgt).loss
                loss.backward()
                opt.step()
                step += 1
            # Snapshot at this budget (the sweep is nested). For LoRA only
            # the adapter tensors are kept; for full fine-tuning the whole
            # state dict must be, which is ~800 MB per budget in float32 --
            # acceptable for 3 budgets, and the reason full FT snapshots are
            # not kept across refits.
            keep = (lambda k: "lora" in k) if self.mode == "lora" else (lambda k: True)
            self.adapters_[budget] = {
                k: v.detach().clone().cpu()
                for k, v in model.state_dict().items()
                if keep(k)
            }
            self.train_seconds_[budget] = time.time() - t0
            logger.info(
                "chronos_ft: budget %d reached in %.1fs (%d windows)",
                budget,
                self.train_seconds_[budget],
                len(windows),
            )
        self._model = model
        if self.mode == "lora":
            # A zeroed adapter IS the untouched base model, so budget 0 is a
            # true zero-shot control even on this private pipeline.
            self.adapters_[0] = {
                k: torch.zeros_like(v)
                for k, v in self.adapters_[self.budgets[-1]].items()
            }
        # For full fine-tuning there is no such trick: budget 0 is the
        # separately-loaded pristine pipeline, which is what the zero-shot
        # models already use.

    def apply_adapter(self, budget: int):
        """Load the adapter snapshot for ``budget`` into the PRIVATE pipeline."""
        if budget not in self.adapters_ or not hasattr(self, "_model"):
            return None
        self._model.load_state_dict(self.adapters_[budget], strict=False)
        self._model.eval()
        return self.pipeline_


# Backwards-compatible alias: the class was LoRA-only when M4 was written.
ChronosLoRAFineTuner = ChronosFineTuner


class ChronosFineTunedNowcaster(TSFMNowcaster):
    """Chronos + LoRA at a given fine-tuning budget, ragged-edge strategy.

    ``budget=0`` is exactly the zero-shot model, which makes the budget sweep
    a clean nested comparison (RQ3): the only thing that changes across the
    sweep is how many gradient steps the adapter received on data available
    at the nowcast date.
    """

    def __init__(self, budget: int = 200, shared_tuner=None, mode: str = "lora", **kw):
        super().__init__(backend="chronos", strategy="ragged_edge", **kw)
        self.budget = budget
        self.shared_tuner = shared_tuner
        self.mode = mode
        tag = "lora_ft" if mode == "lora" else "full_ft"
        self.name = f"chronos_{tag}{budget}"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        super().fit(X, y)  # the bridge regression only; the TSFM is frozen

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        """Load THIS budget's adapter immediately before forecasting.

        All budgets share one private pipeline (training once and snapshotting
        is what makes the nested sweep affordable), so the adapter must be
        installed at PREDICT time, not at fit time. Doing it at fit time meant
        whichever budget was fitted last owned the pipeline for everyone, and
        all three budgets returned bit-identical forecasts -- a silent
        collapse of the entire RQ3 sweep that looked like "fine-tuning has no
        effect". See.
        """
        if self.budget > 0 and self.shared_tuner is not None:
            adapted = self.shared_tuner.apply_adapter(self.budget)
            if adapted is not None:
                self.wrapper._pipeline = adapted
        return super().predict(X, target_quarter, y)


class TSFMCompletedDFMNowcaster:
    """The one-variable experiment: same DFM, different ragged-edge treatment.

    WHY THIS EXISTS
    ------------------------------------------
    ``TSFMNowcaster(strategy="ragged_edge")`` completes each indicator with a
    foundation model and then maps the completed panel to GDP with a SIMPLE
    BRIDGE regression. Comparing that against a full state-space DFM confounds
    two different things:

      1. how well the ragged edge is extrapolated, and
      2. how well the completed panel is mapped to GDP
         (an OLS bridge vs a factor structure).

    The ablation exposed the problem: with a near-complete panel the
    TSFM+bridge improves only 3% yet stays worse than the DFM on the real
    ragged panel, which says most of the remaining gap is the BRIDGE, not the
    foundation model. A referee would read the headline comparison as unfair,
    and would be right.

    This class removes the confound. The TSFM completes the unpublished months
    and the completed panel is handed to THE SAME ``DFMNowcaster`` that the
    baseline uses. The only difference from the baseline DFM is therefore how
    the ragged edge was handled -- the Kalman filter's implicit treatment
    versus an explicit foundation-model extrapolation. That is a single-factor
    experiment and it is the clean test of H3.
    """

    def __init__(
        self,
        backend: str = "chronos",
        robust: bool = True,
        device: str | None = None,
        min_train_quarters: int = 20,
        max_context: int = 512,
        start: str | None = None,
        **kw,
    ):
        from src.models.econ_baselines import DFMNowcaster, RobustDFMNowcaster

        self.backend_name = backend
        self.completer = TSFMNowcaster(
            backend=backend,
            strategy="ragged_edge",
            device=device,
            min_train_quarters=min_train_quarters,
            max_context=max_context,
        )
        dfm_cls = RobustDFMNowcaster if robust else DFMNowcaster
        self.dfm = dfm_cls(min_train_quarters=min_train_quarters, start=start)
        self.name = f"{backend}_completed_dfm"
        self.n_completed_: int = 0

    def _complete_panel(self, X: pd.DataFrame, target_quarter) -> pd.DataFrame:
        """Fill each series' unpublished months up to the target quarter end.

        Only forward gaps are filled, and only from the TSFM's forecast
        conditioned on that series' OWN observed history -- so nothing enters
        that was not derivable from the information set at ``t`` (R1).
        """
        q_end = pd.Timestamp(target_quarter).to_period("Q").end_time
        monthly = X.groupby(X.index.to_period("M")).mean()
        monthly.index = monthly.index.to_timestamp()
        full_idx = pd.date_range(monthly.index.min(), q_end, freq="MS")
        monthly = monthly.reindex(full_idx)

        cols, hists, needed = [], [], []
        for col in monthly.columns:
            s = monthly[col]
            observed = s.dropna()
            if len(observed) < MIN_CONTEXT:
                continue
            gap = s.loc[observed.index[-1] :].isna().sum()
            if gap <= 0:
                continue
            cols.append(col)
            hists.append(observed.tail(self.max_context_))
            needed.append(int(gap))

        self.n_completed_ = 0
        if hists:
            forecasts = self.completer.wrapper.predict_batch(hists, horizon=max(needed))
            for col, gap, fc in zip(cols, needed, forecasts):
                if not np.isfinite(fc.mean):
                    continue
                s = monthly[col]
                missing = s.loc[s.dropna().index[-1] :].index[1:]
                monthly.loc[missing[:gap], col] = fc.mean
                self.n_completed_ += gap
        return monthly

    @property
    def max_context_(self) -> int:
        return self.completer.max_context

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # The DFM is estimated on the ragged panel exactly as the baseline
        # does: completion is a PREDICT-time operation, so estimation is
        # identical between the two arms and only the nowcast input differs.
        self.dfm.fit(X, y)
        self.completer.fit(X, y)

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        try:
            completed = self._complete_panel(X, target_quarter)
        except Exception as exc:  # noqa: BLE001 - abstain rather than abort
            logger.warning("%s: completion failed (%s)", self.name, exc)
            return Forecast(mean=float("nan"))
        # A completion arm that completed nothing is not this arm. When the
        # backend is unavailable it abstains inside `_complete_panel`, which
        # returns the panel uncompleted, and the factor model then produces a
        # perfectly plausible nowcast from it. That silently turns the
        # single-factor experiment into a comparison of the factor model with
        # itself, and it produced a full 1,403-row artifact at 100% coverage
        # before anyone noticed. Abstain instead.
        if self.n_completed_ == 0:
            logger.warning(
                "%s: the completer filled no months, so this is not a "
                "foundation-model arm; abstaining",
                self.name,
            )
            return Forecast(mean=float("nan"))
        return self.dfm.predict(completed, target_quarter, y)


def rectangular_context(
    monthly: pd.DataFrame,
    usable: list,
    ctx_end,
    max_context: int,
) -> pd.DataFrame:
    """The gap-free block of history a multivariate model can condition on.

    The window is trimmed to `max_context` BEFORE incomplete columns are
    dropped. Reversing the two throws the panel away: it reaches back to
    1913, so a column filter applied to the whole history drops every series
    that starts late or carries an old gap, which is all of them, and leaves
    nothing to condition on.
    """
    return monthly.loc[:ctx_end, usable].tail(max_context).dropna(axis=1, how="any")


def _quantile_index(pipeline, n_axis: int) -> tuple[int, float]:
    """Index of the requested completion quantile on the model's own grid.

    The levels are read off the pipeline rather than inferred from the axis
    length. They are not evenly spaced over [0, 1]: the grid runs 0.01, 0.05,
    0.10, ..., 0.95, 0.99, so arithmetic on the index silently returns a
    neighbouring level.
    """
    levels = getattr(pipeline, "quantiles", None)
    if levels is None:
        raise ValueError(
            "the pipeline does not expose its quantile levels, so the "
            "completion quantile cannot be identified"
        )
    levels = np.asarray([float(q) for q in levels], dtype=float)
    if levels.shape[0] != n_axis:
        raise ValueError(
            f"{levels.shape[0]} quantile levels but the quantile axis has "
            f"{n_axis} entries"
        )
    idx = int(np.argmin(np.abs(levels - CHRONOS2_COMPLETION_QUANTILE)))
    return idx, float(levels[idx])


# Which quantile of the predictive distribution the panel is completed at.
# 0.5 is the median and the default; the imputation sensitivity sweeps it.
CHRONOS2_COMPLETION_QUANTILE = 0.5


class Chronos2CompletedDFMNowcaster:
    """Complete the ragged edge MULTIVARIATELY, then pass the panel to a DFM.

    The study's discussion states a falsifiable prediction: that a pretrained
    model which conditions on covariates, so the completion of one indicator
    can use the observed values of the others, should close the gap on the
    Kalman filter and might exceed it. Chronos-2 is such a model, so the
    prediction is testable rather than a hedge, and this arm tests it.

    The difference from :class:`TSFMCompletedDFMNowcaster` is one thing only.
    That arm completes each indicator from its OWN history, so the pretrained
    model never sees the cross-section and the comparison against a filter
    that does is partly a comparison of information sets. Here the whole panel
    goes into one multivariate call, so both arms see the same cross-section
    and the contrast is the mechanism that uses it.

    Context is the window in which every series is observed, which ends at the
    EARLIEST last observation across the panel. That discards the fresher
    observations of the less stale series, which is a real cost and the price
    of a rectangular multivariate context; the filter arm pays no such price,
    so this comparison is, if anything, unfavourable to the pretrained model.
    """

    def __init__(
        self,
        repo: str = "amazon/chronos-2",
        robust: bool = True,
        device: str | None = None,
        min_train_quarters: int = 20,
        max_context: int = 512,
        start: str | None = None,
        covariate_mode: bool = False,
        **kw,
    ):
        from src.models.econ_baselines import DFMNowcaster, RobustDFMNowcaster

        self.repo = repo
        self.device = device
        self.max_context = max_context
        # Whether the already-published months of the more current series are
        # handed to the model as known future covariates, or discarded by
        # truncating the panel to a rectangle.
        self.covariate_mode = covariate_mode
        dfm_cls = RobustDFMNowcaster if robust else DFMNowcaster
        self.dfm = dfm_cls(min_train_quarters=min_train_quarters, start=start)
        self.name = (
            "chronos2_cov_completed_dfm" if covariate_mode else "chronos2_completed_dfm"
        )
        self.n_completed_: int = 0
        # The quantile the panel was actually completed at, which is the
        # nearest level the model offers to the one requested.
        self.completion_quantile_: float | None = None
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from chronos import Chronos2Pipeline

            device = _resolve_device(self.device)
            t0 = time.time()
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.repo, device_map=device
            )
            logger.info(
                "%s: loaded %s in %.1fs", self.name, self.repo, time.time() - t0
            )
        return self._pipeline

    def _take_quantile(self, pipeline, paths: np.ndarray) -> np.ndarray:
        """Collapse the quantile axis of a (variates, quantiles, horizon) block."""
        idx_q, level = _quantile_index(pipeline, paths.shape[1])
        self.completion_quantile_ = level
        if abs(level - CHRONOS2_COMPLETION_QUANTILE) > 1e-9:
            logger.info(
                "%s: completing at quantile %.3f, the nearest the model offers "
                "to the requested %.3f",
                self.name,
                level,
                CHRONOS2_COMPLETION_QUANTILE,
            )
        return paths[:, idx_q, :]

    def _complete_with_covariates(
        self, monthly: pd.DataFrame, q_end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fill the ragged edge using the months other series have published.

        A rectangular context has to end where the laggiest series ends, which
        throws away every month the more current series have already
        published -- on this panel, up to five of them. The model accepts
        those months as known future covariates, which is what a
        covariate-aware model is for, so the edge is filled in two steps.

        First, every series that ends before the panel's last observed month
        is forecast up to that month with the fully observed series supplied
        as future covariates, which are data rather than forecasts. The panel
        is then rectangular, and a single multivariate call carries it to the
        end of the target quarter, where nothing is known for anyone.
        """
        last_obs = {c: monthly[c].dropna().index.max() for c in monthly.columns}
        usable = [c for c, t in last_obs.items() if pd.notna(t)]
        if not usable:
            return monthly
        edge = max(last_obs[c] for c in usable)
        pipeline = self._load()

        # Phase one. Series grouped by where they end; each group shares a
        # horizon, which the model requires of a single call.
        lagging = [c for c in usable if last_obs[c] < edge]
        current = [c for c in usable if last_obs[c] == edge]
        for end in sorted({last_obs[c] for c in lagging}):
            group = [c for c in lagging if last_obs[c] == end]
            horizon = len(pd.date_range(end, edge, freq="MS")) - 1
            if horizon <= 0:
                continue
            hist = monthly.loc[:end, group + current].tail(self.max_context)
            hist = hist.dropna(axis=1, how="any")
            keep = [c for c in group if c in hist.columns]
            cov = [c for c in current if c in hist.columns]
            if not keep or hist.shape[0] < MIN_CONTEXT:
                continue
            fut_idx = pd.date_range(end, edge, freq="MS")[1:]
            future = monthly.loc[fut_idx, cov] if cov else None
            if future is not None and not future.notna().all().all():
                cov, future = [], None
            item = {"target": np.ascontiguousarray(hist[keep].to_numpy(float).T)}
            if cov:
                item["past_covariates"] = {
                    c: np.ascontiguousarray(hist[c].to_numpy(float)) for c in cov
                }
                item["future_covariates"] = {
                    c: np.ascontiguousarray(future[c].to_numpy(float)) for c in cov
                }
            out = np.asarray(
                pipeline.predict([item], prediction_length=horizon)[0], dtype=float
            )
            if out.ndim == 3:
                if out.shape[2] != horizon:
                    raise ValueError(
                        f"expected the horizon on axis 2, got {out.shape} for "
                        f"horizon {horizon}"
                    )
                out = self._take_quantile(pipeline, out)
            for k, col in enumerate(keep):
                vals = out[k, :horizon]
                if not np.all(np.isfinite(vals)):
                    continue
                monthly.loc[fut_idx, col] = vals
                self.n_completed_ += len(fut_idx)

        # Phase two. The panel is rectangular to `edge`; nothing beyond it is
        # published for any series, so this is an ordinary joint forecast.
        horizon = len(pd.date_range(edge, q_end, freq="MS")) - 1
        if horizon <= 0:
            return monthly
        ctx = rectangular_context(monthly, usable, edge, self.max_context)
        if ctx.shape[0] < MIN_CONTEXT or ctx.shape[1] == 0:
            return monthly
        arr = np.ascontiguousarray(ctx.to_numpy(float).T)[None, :, :]
        paths = np.asarray(
            pipeline.predict(arr, prediction_length=horizon)[0], dtype=float
        )
        if paths.ndim == 3:
            if paths.shape[2] != horizon:
                raise ValueError(
                    f"expected the horizon on axis 2, got {paths.shape} for "
                    f"horizon {horizon}"
                )
            paths = self._take_quantile(pipeline, paths)
        fut_idx = pd.date_range(edge, q_end, freq="MS")[1:]
        for k, col in enumerate(ctx.columns):
            s = monthly[col]
            missing = s.index[s.isna() & s.index.isin(fut_idx)]
            if len(missing) == 0:
                continue
            take = min(len(missing), paths.shape[1])
            vals = paths[k, :take]
            if not np.all(np.isfinite(vals)):
                continue
            monthly.loc[missing[:take], col] = vals
            self.n_completed_ += take
        return monthly

    def _complete_panel(self, X: pd.DataFrame, target_quarter) -> pd.DataFrame:
        q_end = pd.Timestamp(target_quarter).to_period("Q").end_time
        monthly = X.groupby(X.index.to_period("M")).mean()
        monthly.index = monthly.index.to_timestamp()
        full_idx = pd.date_range(monthly.index.min(), q_end, freq="MS")
        monthly = monthly.reindex(full_idx)

        self.n_completed_ = 0
        if self.covariate_mode:
            return self._complete_with_covariates(monthly, q_end)

        last_obs = {c: monthly[c].dropna().index.max() for c in monthly.columns}
        usable = [c for c, t in last_obs.items() if pd.notna(t)]
        if not usable:
            return monthly

        # The rectangular context ends where every usable series is observed.
        ctx_end = min(last_obs[c] for c in usable)
        ctx = rectangular_context(monthly, usable, ctx_end, self.max_context)
        if ctx.shape[0] < MIN_CONTEXT or ctx.shape[1] == 0:
            return monthly

        horizon = int(
            max(
                len(pd.date_range(ctx_end, q_end, freq="MS")) - 1,
                1,
            )
        )
        pipeline = self._load()
        # One item, many variates: the model shares information across them.
        arr = np.ascontiguousarray(ctx.to_numpy(float).T)[None, :, :]
        out = pipeline.predict(arr, prediction_length=horizon)
        # The item comes back as (n_variates, n_quantiles, horizon). Taking a
        # median along the LAST axis collapses the horizon and leaves the
        # quantile fan, which is monotone and so looks exactly like a
        # plausible path; the mistake is invisible in a plot. The median is
        # along the quantile axis, which is axis 1.
        paths = np.asarray(out[0], dtype=float)
        if paths.ndim == 3:
            if paths.shape[2] != horizon:
                raise ValueError(
                    f"expected the horizon on axis 2, got shape {paths.shape} "
                    f"for horizon {horizon}"
                )
            # The median by default. `CHRONOS2_COMPLETION_QUANTILE` moves the
            # completion to another quantile of the predictive distribution,
            # which is how the multiple-imputation sensitivity propagates the
            # uncertainty a point completion otherwise discards.
            #
            paths = self._take_quantile(pipeline, paths)
        if paths.shape[1] != horizon:
            raise ValueError(
                f"completion path has length {paths.shape[1]}, " f"expected {horizon}"
            )

        future = pd.date_range(ctx_end, q_end, freq="MS")[1:]
        for k, col in enumerate(ctx.columns):
            series = monthly[col]
            missing = series.index[series.isna() & series.index.isin(future)]
            if len(missing) == 0:
                continue
            take = min(len(missing), paths.shape[1])
            values = paths[k, :take]
            if not np.all(np.isfinite(values)):
                continue
            monthly.loc[missing[:take], col] = values
            self.n_completed_ += take
        return monthly

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # Completion is a PREDICT-time operation, so estimation is identical
        # between the arms and only the nowcast input differs.
        self.dfm.fit(X, y)

    def predict(
        self, X: pd.DataFrame, target_quarter, y: pd.Series | None = None
    ) -> Forecast:
        try:
            completed = self._complete_panel(X, target_quarter)
        except Exception as exc:  # noqa: BLE001 - abstain rather than abort
            logger.warning("%s: completion failed (%s)", self.name, exc)
            return Forecast(mean=float("nan"))
        if self.n_completed_ == 0:
            logger.warning("%s: the completer filled no months; abstaining", self.name)
            return Forecast(mean=float("nan"))
        return self.dfm.predict(completed, target_quarter, y)


TSFM_MODELS = {
    "chronos_aggregate_q": lambda **kw: TSFMNowcaster(
        backend="chronos", strategy="aggregate_q", **kw
    ),
    "chronos_ragged_edge": lambda **kw: TSFMNowcaster(
        backend="chronos", strategy="ragged_edge", **kw
    ),
    "timesfm_aggregate_q": lambda **kw: TSFMNowcaster(
        backend="timesfm", strategy="aggregate_q", **kw
    ),
    "timesfm_ragged_edge": lambda **kw: TSFMNowcaster(
        backend="timesfm", strategy="ragged_edge", **kw
    ),
    "lagllama_aggregate_q": lambda **kw: TSFMNowcaster(
        backend="lagllama", strategy="aggregate_q", **kw
    ),
    "lagllama_ragged_edge": lambda **kw: TSFMNowcaster(
        backend="lagllama", strategy="ragged_edge", **kw
    ),
    # The single-factor H3 experiment: same DFM, different ragged-edge
    # treatment.
    "chronos_completed_dfm": lambda **kw: TSFMCompletedDFMNowcaster(
        backend="chronos", **kw
    ),
    "timesfm_completed_dfm": lambda **kw: TSFMCompletedDFMNowcaster(
        backend="timesfm", **kw
    ),
    "lagllama_completed_dfm": lambda **kw: TSFMCompletedDFMNowcaster(
        backend="lagllama", **kw
    ),
    # The covariate-aware arm: the panel is completed multivariately, so the
    # pretrained model sees the cross-section the filter sees.
    "chronos2_completed_dfm": lambda **kw: Chronos2CompletedDFMNowcaster(**kw),
    # The same arm given the months the other series have already published,
    # as known future covariates rather than discarded by truncation.
    "chronos2_cov_completed_dfm": lambda **kw: Chronos2CompletedDFMNowcaster(
        covariate_mode=True, **kw
    ),
}
