"""베이스라인과 비교 모델.

모든 모델은 fit(X, y, sample_weight=None) / predict(X) 인터페이스를 따른다.
새 모델(GRU 등)을 추가하려면 같은 인터페이스로 클래스를 만들고 make_models()에 등록하면
교차검증·홀드아웃·결과표에 자동으로 포함된다.
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
    return f"lightgbm {lgb.__version__}" if _HAS_LGB else "sklearn HistGradientBoosting (lightgbm 미설치)"


class SeasonalNaive:
    """지난주 같은 요일·같은 15분 슬롯의 실측값."""

    def fit(self, X, y, sample_weight=None):
        return self

    def predict(self, X):
        return X["lag_7d"].fillna(X["lag_1d"]).to_numpy(dtype=float)


class ProfileMean:
    """학습기간의 (휴무일 여부, 요일, 슬롯)별 평균 프로파일."""

    keys = ["is_off_day", "dow", "slot"]

    def fit(self, X, y, sample_weight=None):
        d = X[self.keys].copy()
        d["y"] = np.asarray(y, dtype=float)
        self.table_ = d.groupby(self.keys)["y"].mean()
        self.slot_ = d.groupby("slot")["y"].mean()
        self.global_ = float(d["y"].mean())
        return self

    def predict(self, X):
        p = self.table_.reindex(pd.MultiIndex.from_frame(X[self.keys])).to_numpy()
        p = np.where(np.isnan(p), X["slot"].map(self.slot_).to_numpy(dtype=float), p)
        return np.where(np.isnan(p), self.global_, p)


def make_gbm(quantile: float | None = None):
    if _HAS_LGB:
        params = dict(C.LGB_PARAMS)
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile)
        return lgb.LGBMRegressor(**params, random_state=C.SEED, deterministic=True,
                                 force_row_wise=True, verbose=-1)
    params = dict(C.HGB_PARAMS)
    if quantile is not None:
        params.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**params, random_state=C.SEED)


def make_models() -> dict[str, tuple[str, object]]:
    """이름 → (종류, 새 인스턴스). 종류: point(점예측) | risk(피크위험용 상위분위 예측)."""
    return {
        "naive_lastweek": ("point", SeasonalNaive()),
        "profile_mean": ("point", ProfileMean()),
        "random_forest": ("point", RandomForestRegressor(**C.RF_PARAMS, random_state=C.SEED, n_jobs=-1)),
        "gbm": ("point", make_gbm()),
        f"gbm_q{int(C.RISK_QUANTILE * 100)}": ("risk", make_gbm(quantile=C.RISK_QUANTILE)),
    }
