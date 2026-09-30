"""15분 단위 변환과 피처 생성.

원칙: 모든 피처는 예측 시점(D일 00:00)에 알 수 있는 정보만 사용한다.
  - 전력 래그: D-1일 이전 실측만 (lag_1d = 어제 같은 슬롯, lag_7d = 지난주 같은 슬롯)
  - 달력: 미리 앎
  - 기상: 예보로 대체 가능하다는 가정 (config.USE_WEATHER)
  - 생산량: 생산계획으로 미리 안다는 가정 (config.PRODUCTION_PLAN)
"""
from __future__ import annotations

import pandas as pd

from . import config as C

SLOTS_PER_DAY = 96

CAL_COLS = ["hour", "quarter", "slot", "dow", "month", "is_holiday", "is_planned_shutdown",
            "is_off_day", "prev_is_off_day", "is_startup", "is_lunch"]
LAG_COLS = ["lag_1d", "lag_7d", "prev_day_mean", "prev_day_max", "prev_day_last", "lastweek_day_mean"]
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

    expected = pd.date_range(long["ts"].min(), long["ts"].max(), freq="15min")
    if len(expected) != len(long) or not (expected == pd.DatetimeIndex(long["ts"])).all():
        raise ValueError("15분 시계열이 연속적이지 않습니다. 정제 단계를 확인하세요.")
    return long


def build_features(long: pd.DataFrame) -> pd.DataFrame:
    f = long.copy()

    # 전력 래그 (D일 00:00에 모두 알려진 값)
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
    days["is_off_day"] = ((days.index.dayofweek == 6) | (days["is_holiday"] == 1)
                          | (days["is_planned_shutdown"] == 1)).astype(int)
    days["prev_is_off_day"] = days["is_off_day"].shift(1).fillna(0).astype(int)   # 휴무 다음날 재가동
    f = f.join(days, on="date")
    f["is_startup"] = f["hour"].isin([7, 8]).astype(int)   # 기동 피크 구간
    f["is_lunch"] = (f["hour"] == 12).astype(int)

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
    na = f[CAL_COLS + LAG_COLS + WEATHER_COLS + PROD_COLS].isna().sum()
    if na.any():
        raise ValueError(f"피처에 결측이 남아 있습니다: {na[na > 0].to_dict()}")
    return f


def feature_columns(prod_plan: str = C.PRODUCTION_PLAN, use_weather: bool = C.USE_WEATHER) -> list[str]:
    cols = CAL_COLS + LAG_COLS
    if use_weather:
        cols += WEATHER_COLS
    if prod_plan == "hourly":
        cols += PROD_COLS
    elif prod_plan == "daily":
        cols += ["prod_day_total"]
    elif prod_plan != "none":
        raise ValueError(f"PRODUCTION_PLAN은 hourly/daily/none 중 하나: {prod_plan}")
    return cols
