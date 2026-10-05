"""기준 모델과 비교 모델.

모든 행 단위 모델은 fit(X, y, sample_weight=None) / predict(X) 인터페이스를 따른다.
시퀀스 모델은 needs_full_frame=True로 표시하고 fit_frame / predict_frame을 구현한다(src/seq.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from . import config as C

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


def gbm_backend() -> str:
    return f"lightgbm {lgb.__version__}" if _HAS_LGB else "sklearn HistGradientBoosting"


def make_gbm():
    if _HAS_LGB:
        params = dict(C.LGB_PARAMS)
        return lgb.LGBMRegressor(**params, random_state=C.SEED, deterministic=True,
                                 force_row_wise=True, verbose=-1)
    params = dict(C.HGB_PARAMS)
    return HistGradientBoostingRegressor(**params, random_state=C.SEED)


def make_models() -> dict[str, tuple[str, object, str]]:
    """모델명 → (종류, 인스턴스, 입력 범위).
    종류: point(점예측) | risk(상위 분위 예측). 입력 범위: 1d(1일 전까지) | 7d(7일 전까지)."""
    models = {
        "random_forest": ("point", RandomForestRegressor(**C.RF_PARAMS, random_state=C.SEED, n_jobs=-1), "7d"),
        "gbm_1d": ("point", make_gbm(), "1d"),
        "gbm": ("point", make_gbm(), "7d"),
    }
    if lstm_available():
        from .seq import LSTMForecaster
        for name, cfg in C.LSTM_MODELS.items():
            models[name] = ("point", LSTMForecaster(cfg["window"], name, cfg["decoder_lags"]),
                            f"{cfg['window'] // 96}d")
    return models


def lstm_available() -> bool:
    from .seq import HAS_TORCH
    return C.RUN_LSTM and HAS_TORCH
