"""15분 단위 변환과 피처 생성.

모든 피처는 예측 시점(D일 00:00)에 알 수 있는 값만 사용한다.
  - 전력 래그: D-1일 이전 실측(lag_1d: 1일 전 동일 구간, lag_7d: 7일 전 동일 구간)
  - 달력: 사전에 확정
  - 과거 생산량(prod_lag_1d: 1일 전 동일 시간, prev_day_prod_total: 전일 총생산량): config 설정 시에만 사용
  - 기상, 생산계획: config 설정 시에만 사용
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SLOTS_PER_DAY = 96

CAL_COLS = ["hour", "quarter", "slot", "dow", "month", "is_holiday",
            "is_off_day", "prev_is_off_day", "is_startup", "is_lunch"]
SHUTDOWN_COLS = ["is_planned_shutdown"]
LAG_1D_COLS = ["lag_1d", "prev_day_mean", "prev_day_max", "prev_day_last"]
LAG_7D_COLS = ["lag_7d", "lastweek_day_mean"]
LAG_COLS = LAG_1D_COLS + LAG_7D_COLS
WEATHER_COLS = ["temp", "humidity", "wind", "rain", "cdd", "hdd"]
PROD_COLS = ["prod_h", "prod_prev_h", "prod_next_h", "prod_day_total"]
PAST_PROD_COLS = ["prod_lag_1d", "prev_day_prod_total"]

# 신경망 입력용 달력 인코딩
SLOT_CYCLIC_COLS = ["slot_sin", "slot_cos"]
DOW_ONEHOT_COLS = [f"dow_{k}" for k in range(7)]   # 월=dow_0 … 일=dow_6
MONTH_CYCLIC_COLS = ["month_sin", "month_cos"]
NN_FLAG_COLS = ["is_holiday", "is_off_day", "prev_is_off_day", "is_startup", "is_lunch"]


def to_long(hourly: pd.DataFrame) -> pd.DataFrame:
    """시간 단위 4개 컬럼을 15분 단위 시계열(kw)로 변환한다. ts는 구간 시작 시각."""
    id_cols = [c for c in hourly.columns if c not in C.POWER_COLS]
    long = hourly.melt(id_vars=id_cols, value_vars=C.POWER_COLS, var_name="q_col", value_name="kw")
    long["quarter"] = long["q_col"].map({c: i for i, c in enumerate(C.POWER_COLS)}).astype(int)
    long["ts"] = long["ts"] + pd.to_timedelta(15 * long["quarter"], unit="min")
    long = long.drop(columns="q_col").sort_values("ts").reset_index(drop=True)
    long["slot"] = long["hour"] * 4 + long["quarter"]
    long["kw"] = long["kw"].astype(float)

    expected = pd.date_range(long["ts"].min(), long["ts"].max(), freq="15min")
    if len(expected) != len(long) or not (expected == pd.DatetimeIndex(long["ts"])).all():
        raise ValueError("15분 시계열에 누락 또는 중복이 있습니다.")
    return long


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    f = long.copy()

    # 전력 래그
    f["lag_1d"] = f["kw"].shift(SLOTS_PER_DAY)
    f["lag_7d"] = f["kw"].shift(SLOTS_PER_DAY * 7)
    daily = f.groupby("date")["kw"].agg(day_mean="mean", day_max="max", day_last="last").asfreq("D")
    f = f.join(daily.shift(1).add_prefix("prev_"), on="date")
    f["lastweek_day_mean"] = f["date"].map(daily["day_mean"].shift(7))

    # 달력·교대 구조
    f["month"] = f["date"].dt.month
    hol = pd.DatetimeIndex(pd.to_datetime(C.HOLIDAYS))
    shut = pd.DatetimeIndex(pd.to_datetime(C.PLANNED_SHUTDOWNS))
    days = pd.DataFrame(index=daily.index)
    days["is_holiday"] = days.index.isin(hol).astype(int)
    days["is_planned_shutdown"] = days.index.isin(shut).astype(int)
    off = (days.index.dayofweek == 6) | (days["is_holiday"] == 1)
    if C.USE_PLANNED_SHUTDOWN:
        off = off | (days["is_planned_shutdown"] == 1)
    days["is_off_day"] = off.astype(int)
    days["prev_is_off_day"] = days["is_off_day"].shift(1).fillna(0).astype(int)
    f = f.join(days, on="date")
    f["is_startup"] = f["hour"].isin([7, 8]).astype(int)
    f["is_lunch"] = (f["hour"] == 12).astype(int)
    for name, val, period in [("slot", f["slot"], 96), ("month", f["month"] - 1, 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    for k in range(7):
        f[f"dow_{k}"] = (f["dow"] == k + 1).astype(int)

    # 기상 파생
    f["cdd"] = (f["temp"] - 24).clip(lower=0)
    f["hdd"] = (10 - f["temp"]).clip(lower=0)

    # 생산계획(시간 단위 값을 15분 구간에 공유)
    f["ts_hour"] = f["ts"].dt.floor("h")
    ph = f.groupby("ts_hour")["prod"].first()
    f["prod_h"] = f["ts_hour"].map(ph)
    f["prod_prev_h"] = f["ts_hour"].map(ph.shift(1)).fillna(0)
    f["prod_next_h"] = f["ts_hour"].map(ph.shift(-1)).fillna(0)
    prod_daily = ph.groupby(ph.index.normalize()).sum()
    f["prod_day_total"] = f["date"].map(prod_daily)

    # 과거 생산량(D-1일 실적)
    f["prod_lag_1d"] = f["prod_h"].shift(SLOTS_PER_DAY)
    f["prev_day_prod_total"] = f["date"].map(prod_daily.asfreq("D").shift(1))

    f = f.dropna(subset=LAG_COLS).reset_index(drop=True)   # 첫 7일은 lag_7d가 없어 제외
    na = f[CAL_COLS + SHUTDOWN_COLS + LAG_COLS + WEATHER_COLS + PROD_COLS + PAST_PROD_COLS
           + SLOT_CYCLIC_COLS + DOW_ONEHOT_COLS + MONTH_CYCLIC_COLS].isna().sum()
    if na.any():
        raise ValueError(f"피처에 결측이 있습니다: {na[na > 0].to_dict()}")
    return f


def _optional_cols(prod_plan: str, use_weather: bool) -> list[str]:
    cols = []
    if C.USE_PLANNED_SHUTDOWN:
        cols += SHUTDOWN_COLS
    if use_weather:
        cols += WEATHER_COLS
    if prod_plan == "hourly":
        cols += PROD_COLS
    elif prod_plan == "daily":
        cols += ["prod_day_total"]
    elif prod_plan != "none":
        raise ValueError(f"PRODUCTION_PLAN은 hourly/daily/none 중 하나: {prod_plan}")
    return cols


def feature_columns(prod_plan: str = C.PRODUCTION_PLAN, use_weather: bool = C.USE_WEATHER,
                    lookback: str = "7d", use_past_prod: bool | None = None) -> list[str]:
    """트리·기준 모델 피처. lookback="1d"는 1일 전 정보만, "7d"는 7일 전 정보까지 사용."""
    if lookback not in ("1d", "7d"):
        raise ValueError(f"lookback은 1d/7d 중 하나: {lookback}")
    if use_past_prod is None:
        use_past_prod = C.USE_PAST_PROD
    lags = LAG_1D_COLS + (LAG_7D_COLS if lookback == "7d" else [])
    lags += PAST_PROD_COLS if use_past_prod else []
    cal = [c for c in CAL_COLS if C.USE_MONTH or c != "month"]
    return cal + lags + _optional_cols(prod_plan, use_weather)


def nn_future_columns(prod_plan: str = C.PRODUCTION_PLAN, use_weather: bool = C.USE_WEATHER) -> list[str]:
    """LSTM 인코더·디코더 공통 입력(달력·상태). 디코더 래그는 seq.py에서 별도로 추가한다."""
    month = MONTH_CYCLIC_COLS if C.USE_MONTH else []
    return SLOT_CYCLIC_COLS + DOW_ONEHOT_COLS + month + NN_FLAG_COLS + _optional_cols(prod_plan, use_weather)


def nn_past_columns(use_past_prod: bool | None = None) -> list[str]:
    """LSTM 인코더(과거 구간) 입력: 전력 + (과거 생산량) + 공통 달력·상태."""
    if use_past_prod is None:
        use_past_prod = C.USE_PAST_PROD
    fut = nn_future_columns()
    prod = ["prod_h"] if use_past_prod and "prod_h" not in fut else []
    return ["kw"] + prod + fut
