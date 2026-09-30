"""원본 CSV 로드와 정제.

모든 정제는 코드로만 수행한다(원본 파일 수작업 수정 금지). 각 단계의 근거가 되는
검증 수치는 report 딕셔너리에 남겨 data_quality.json / results.md로 출력한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

RENAME = {
    "기온": "temp", "습도": "humidity", "풍속": "wind", "강수량": "rain",
    "생산량": "prod", "평균": "kw_hour_avg", "인건비": "labor_mult",
    "전기요금(계절)": "tariff_season",
}


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=C.ENCODING)
    required = ["날짜", "시간", *C.POWER_COLS, "평균", "생산량", *C.WEATHER_COLS,
                "day", *C.LEAKAGE_COLS, *C.COST_ONLY_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}")
    df["_row"] = np.arange(len(df))   # 파일 내 원래 순서(시간 복원에 사용)
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """정제된 시간 단위 데이터, 품질 리포트, 복제 그룹표를 반환."""
    rep: dict = {"n_rows_raw": int(len(raw))}
    df = raw.sort_values(["날짜", "_row"], kind="stable").reset_index(drop=True)

    # 1) 시간 복원: 날짜별 정확히 24행인지 확인한 뒤 행 순서로 hour를 다시 만든다.
    #    (7/13, 7/15는 '시간' 컬럼에 전력값이 들어가 있음)
    per_day = df.groupby("날짜").size()
    if (per_day != 24).any():
        raise ValueError(f"24행이 아닌 날짜: {per_day[per_day != 24].to_dict()}")
    df["hour"] = df.groupby("날짜").cumcount()
    bad = df["시간"] != df["hour"]
    df["hour_repaired"] = bad
    rep["hour_repaired"] = {
        "rows": int(bad.sum()),
        "dates": sorted(df.loc[bad, "날짜"].astype(str).unique().tolist()),
    }

    df["date"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
    df["ts"] = df["date"] + pd.to_timedelta(df["hour"], unit="h")   # 구간 시작 시각 기준
    full = pd.date_range(df["ts"].min(), df["ts"].max(), freq="h")
    rep["period"] = [str(df["ts"].min()), str(df["ts"].max())]
    rep["n_days"] = int(df["date"].nunique())
    rep["missing_hours_after_repair"] = int(len(full.difference(df["ts"])))
    rep["duplicate_ts_after_repair"] = int(df["ts"].duplicated().sum())

    # 2) 정합성 검증
    rep["day_col_is_weekday_mon1_sun7"] = bool((df["date"].dt.dayofweek + 1 == df["day"]).all())
    mean4 = df[C.POWER_COLS].mean(axis=1)
    rep["avg_col_vs_mean_of_4_max_abs_diff"] = float((df["평균"] - mean4).abs().max())
    # 인건비 = 1.0 (09~17시) / 1.5 (그 외). 복구 전 '시간' 값으로 계산돼 있어 오염된 날은 전부 1.5
    #  → 규칙이 나머지 행에서 예외 없이 성립하는지 확인한 뒤, 오염된 행만 복구된 hour로 재계산
    rule = np.where(df["hour"].between(9, 17), 1.0, 1.5)
    rep["labor_rule_holds_on_unrepaired_rows"] = bool((df.loc[~bad, "인건비"] == rule[~bad.to_numpy()]).all())
    rep["labor_mult_recomputed_rows"] = int((df["인건비"] != rule).sum())
    df["인건비"] = rule
    tar = df.groupby(df["date"].dt.month)["전기요금(계절)"].agg(["nunique", "first"])
    rep["tariff_constant_within_month"] = bool((tar["nunique"] == 1).all())
    rep["tariff_by_month"] = {int(k): float(v) for k, v in tar["first"].items()}

    # 3) 누수 변수 검증 후 제거: 공장인원 == 생산량 / (15분+30분+45분+60분)
    s = df[C.POWER_COLS].sum(axis=1)
    ratio = df["생산량"] / s.where(s > 0)
    ok = df["공장인원"].notna() & ratio.notna()
    rep["leakage_factory_staff"] = {
        "max_abs_diff_vs_prod_over_power_sum": float((ratio[ok] - df.loc[ok, "공장인원"]).abs().max()),
        "nan_rows_all_0_div_0": int(df["공장인원"].isna().sum()),
        "action": "excluded (target leakage)",
    }
    df = df.drop(columns=C.LEAKAGE_COLS)

    # 4) 기상 결측: 시간순 선형보간
    rep["weather_missing_before"] = {c: int(df[c].isna().sum()) for c in C.WEATHER_COLS}
    df[C.WEATHER_COLS] = df[C.WEATHER_COLS].interpolate(limit_direction="both")

    # 5) 전력 0 구간(정전·셧다운 추정): 학습에서 제외, 평가는 포함/제외 둘 다 보고
    df["zero_power"] = df[C.POWER_COLS].sum(axis=1).eq(0)
    rep["zero_power_hours"] = {
        "n": int(df["zero_power"].sum()),
        "dates": sorted(df.loc[df["zero_power"], "date"].dt.strftime("%Y-%m-%d").unique().tolist()),
    }

    # 6) 증강 복제일 탐지
    df, dup_rep, groups = flag_duplicate_days(df)
    rep["duplicate_days"] = dup_rep

    df = df.rename(columns=RENAME).rename(columns={"day": "dow"})
    keep = ["ts", "date", "hour", "dow", *C.POWER_COLS, "kw_hour_avg", "prod",
            "temp", "humidity", "wind", "rain", "labor_mult", "tariff_season",
            "hour_repaired", "zero_power", "dup_group", "dup_group_size", "is_copy_day"]
    return df[keep], rep, groups


def flag_duplicate_days(df: pd.DataFrame):
    """하루 96포인트(24시간 x 4) 전력 프로파일이 완전히 같은 날을 같은 그룹으로 묶는다."""
    prof = df.pivot(index="date", columns="hour", values=C.POWER_COLS)
    h = pd.util.hash_pandas_object(prof, index=False)
    gid = pd.Series(pd.factorize(h)[0], index=prof.index)
    size = gid.map(gid.value_counts())
    is_copy = pd.Series(h.duplicated(keep="first").to_numpy(), index=prof.index)

    df["dup_group"] = df["date"].map(gid)
    df["dup_group_size"] = df["date"].map(size)
    df["is_copy_day"] = df["date"].map(is_copy)

    month = prof.index.month
    rep = {
        "n_days": int(len(prof)),
        "n_unique_profiles": int(gid.nunique()),
        "n_days_in_duplicate_groups": int((size > 1).sum()),
        "n_copy_days_excluding_first": int(is_copy.sum()),
        "days_in_duplicate_groups_by_month": {int(m): int(v) for m, v in (size > 1).groupby(month).sum().items()},
    }
    groups = (
        pd.DataFrame({"date": prof.index.strftime("%Y-%m-%d"), "dup_group": gid.values, "size": size.values})
        .query("size > 1")
        .groupby("dup_group")
        .agg(size=("size", "first"), dates=("date", lambda x: ", ".join(x)))
        .sort_values("size", ascending=False)
        .reset_index()
    )
    return df, rep, groups
