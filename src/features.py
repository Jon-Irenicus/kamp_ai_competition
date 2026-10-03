"""15분 단위 변환과 피처 생성.

원칙: 모든 피처는 예측 시점(D일 00:00)에 알 수 있는 정보만 사용한다.
  - 전력 래그: D-1일 이전 실측만 (lag_1d = 어제 같은 슬롯, lag_7d = 지난주 같은 슬롯)
  - 달력: 미리 앎
  - 기상: 예보로 대체 가능하다는 가정 (config.USE_WEATHER)
  - 생산량: 생산계획으로 미리 안다는 가정 (config.PRODUCTION_PLAN)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

SLOTS_PER_DAY = 96

CAL_COLS = ["hour", "quarter", "slot", "dow", "month", "is_holiday",
            "is_off_day", "prev_is_off_day", "is_startup", "is_lunch"]
SHUTDOWN_COLS = ["is_planned_shutdown"]          # config.USE_PLANNED_SHUTDOWN일 때만 사용
LAG_1D_COLS = ["lag_1d", "prev_day_mean", "prev_day_max", "prev_day_last"]   # 전날 정보만
LAG_7D_COLS = ["lag_7d", "lastweek_day_mean"]                                 # 지난주 정보
LAG_COLS = LAG_1D_COLS + LAG_7D_COLS
# 신경망용 달력 인코딩.
#   시간: 순환 구조(23:45 다음 00:00)를 표현하도록 sin/cos
#   요일: 토(새벽만 가동)·일(대기)·월(재가동) 패턴이 서로 독립적이라 one-hot(월=dow_0 … 일=dow_6)
#   월: config.USE_MONTH일 때만 sin/cos
SLOT_CYCLIC_COLS = ["slot_sin", "slot_cos"]
DOW_ONEHOT_COLS = [f"dow_{k}" for k in range(7)]
MONTH_CYCLIC_COLS = ["month_sin", "month_cos"]
NN_FLAG_COLS = ["is_holiday", "is_off_day", "prev_is_off_day", "is_startup", "is_lunch"]
WEATHER_COLS = ["temp", "humidity", "wind", "rain", "cdd", "hdd"]
PROD_COLS = ["prod_h", "prod_prev_h", "prod_next_h", "prod_day_total"]


def to_long(hourly: pd.DataFrame) -> pd.DataFrame:
    """시간 x 4컬럼 → 15분 단위 시계열(kw). ts는 각 15분 구간의 시작 시각."""
    id_cols = [c for c in hourly.columns if c not in C.POWER_COLS]
    long = hourly.melt(id_vars=id_cols, value_vars=C.POWER_COLS, var_name="q_col", value_name="kw")
    long["quarter"] = long["q_col"].map({c: i for i, c in enumerate(C.POWER_COLS)}).astype(int)
    long["ts"] = long["ts"] + pd.to_timedelta(15 * long["quarter"], unit="min")
    long = long.drop(columns="q_col").sort_values("ts").reset_index(drop=True)
    long["slot"] = long["hour"] * 4 + long["quarter"]
    long["kw"] = long["kw"].astype(float)
    long["zero_power"] = long["kw"].eq(0)   # 정전 표시를 15분 단위로 다시 판정

    expected = pd.date_range(long["ts"].min(), long["ts"].max(), freq="15min")
    if len(expected) != len(long) or not (expected == pd.DatetimeIndex(long["ts"])).all():
        raise ValueError("15분 시계열이 연속적이지 않습니다. 정제 단계를 확인하세요.")
    return long


def lag_source(long: pd.DataFrame) -> pd.Series:
    """래그 계산용 전력 시계열. 정전 칸을 같은 요일·같은 칸의 최근 N주 중앙값으로 대체한다.
    과거 값만 쓰므로(t-7일, t-14일, …) 예측 시점 이후 정보는 들어가지 않는다."""
    kw = long["kw"]
    if not C.LAG_IMPUTE_OUTAGE:
        return kw
    masked = kw.mask(long["zero_power"])
    past = pd.concat([masked.shift(SLOTS_PER_DAY * 7 * k) for k in range(1, C.LAG_IMPUTE_WEEKS + 1)], axis=1)
    return masked.fillna(past.median(axis=1)).fillna(kw)   # 과거 이력이 없으면 원래 값


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    f = long.copy()

    # 전력 래그 (D일 00:00에 모두 알려진 값). 타깃 kw는 그대로, 래그만 대체 시계열에서 계산.
    src = lag_source(f)
    f["kw_lag_src"] = src
    f["lag_1d"] = src.shift(SLOTS_PER_DAY)
    f["lag_7d"] = src.shift(SLOTS_PER_DAY * 7)
    daily = f.groupby("date")["kw_lag_src"].agg(day_mean="mean", day_max="max", day_last="last").asfreq("D")
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
    days["prev_is_off_day"] = days["is_off_day"].shift(1).fillna(0).astype(int)   # 휴무 다음날 재가동
    f = f.join(days, on="date")
    f["is_startup"] = f["hour"].isin([7, 8]).astype(int)   # 기동 피크 구간
    f["is_lunch"] = (f["hour"] == 12).astype(int)
    for name, val, period in [("slot", f["slot"], 96), ("month", f["month"] - 1, 12)]:
        f[f"{name}_sin"] = np.sin(2 * np.pi * val / period)
        f[f"{name}_cos"] = np.cos(2 * np.pi * val / period)
    for k in range(7):
        f[f"dow_{k}"] = (f["dow"] == k + 1).astype(int)

    # 기상 파생
    f["cdd"] = (f["temp"] - 24).clip(lower=0)
    f["hdd"] = (10 - f["temp"]).clip(lower=0)

    # 생산계획 (시간 단위 값을 15분 슬롯에 공유)
    f["ts_hour"] = f["ts"].dt.floor("h")
    ph = f.groupby("ts_hour")["prod"].first()
    f["prod_h"] = f["ts_hour"].map(ph)
    f["prod_prev_h"] = f["ts_hour"].map(ph.shift(1)).fillna(0)
    f["prod_next_h"] = f["ts_hour"].map(ph.shift(-1)).fillna(0)
    f["prod_day_total"] = f["date"].map(ph.groupby(ph.index.normalize()).sum())

    f = f.dropna(subset=LAG_COLS).reset_index(drop=True)   # 첫 7일은 래그가 없어 제외
    na = f[CAL_COLS + SHUTDOWN_COLS + LAG_COLS + WEATHER_COLS + PROD_COLS
           + SLOT_CYCLIC_COLS + DOW_ONEHOT_COLS + MONTH_CYCLIC_COLS].isna().sum()
    if na.any():
        raise ValueError(f"피처에 결측이 남아 있습니다: {na[na > 0].to_dict()}")
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
                    lookback: str = "7d") -> list[str]:
    """트리·기준 모델용 피처. lookback="1d"면 전날 정보만, "7d"면 지난주 래그까지."""
    if lookback not in ("1d", "7d"):
        raise ValueError(f"lookback은 1d/7d 중 하나: {lookback}")
    lags = LAG_1D_COLS + (LAG_7D_COLS if lookback == "7d" else [])
    cal = [c for c in CAL_COLS if C.USE_MONTH or c != "month"]
    return cal + lags + _optional_cols(prod_plan, use_weather)


def nn_future_columns(prod_plan: str = C.PRODUCTION_PLAN, use_weather: bool = C.USE_WEATHER) -> list[str]:
    """LSTM 인코더·디코더에 공통으로 넣는 달력·상태 정보(예측일에도 미리 아는 것).
    디코더 전용 과거 전력(lag_1d 등)은 모델별 설정(config.LSTM_MODELS)으로 seq.py에서 따로 붙인다."""
    month = MONTH_CYCLIC_COLS if C.USE_MONTH else []
    return SLOT_CYCLIC_COLS + DOW_ONEHOT_COLS + month + NN_FLAG_COLS + _optional_cols(prod_plan, use_weather)
