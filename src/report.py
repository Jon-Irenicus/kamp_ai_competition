"""결과 저장: CSV, results.md, 그림."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure   # pyplot 미사용: 전역 백엔드를 변경하지 않음

from . import config as C
from .metrics import HIT_COL, K, TOPK_MAE_COL, daily_matrix

METRIC_COLS = ["mae", "rmse", "daily_max_mae", HIT_COL, TOPK_MAE_COL]
FAMILY = {"naive_yesterday": "naive", "naive_lastweek": "naive", "gbm_1d": "gbm", "gbm": "gbm",
          "lstm_1d": "lstm", "lstm_7d": "lstm"}


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "-" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    from .experiment import wavg
    return pd.DataFrame([
        {"model": m, "lookback": g["lookback"].iloc[0],
         **{c: wavg(g, c) for c in METRIC_COLS}, "mae_std_across_folds": g["mae"].std()}
        for m, g in cv.groupby("model")
    ]).sort_values("mae")


def lookback_table(cv_mean: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """모델 계열 × 입력 범위(1d/7d)별 CV·테스트 성능."""
    rows = []
    for m, fam in FAMILY.items():
        if m not in set(test["model"]):
            continue
        c = cv_mean.set_index("model").loc[m]
        t = test.set_index("model").loc[m]
        rows.append({"family": fam, "lookback": t["lookback"], "model": m, "cv_mae": c["mae"],
                     f"cv_{HIT_COL}": c[HIT_COL], "test_mae": t["mae"], f"test_{HIT_COL}": t[HIT_COL]})
    return pd.DataFrame(rows).sort_values(["family", "lookback"])


def write_results_md(out: Path, dq: dict, groups: pd.DataFrame, cv: pd.DataFrame, test: pd.DataFrame,
                     selected: str, demo: pd.DataFrame | None,
                     slices: dict, backend: str):
    cv_mean = cv_summary(cv)
    dup = dq["duplicate_days"]
    L = [
        "# 실험 결과 요약", "",
        f"- GBM 백엔드: {backend}",
        f"- 과제: D일 00:00에 D일 15분 수요전력 96개 예측.  기상 {C.USE_WEATHER}, "
        f"계획휴무 {C.USE_PLANNED_SHUTDOWN}, 월 피처 {C.USE_MONTH}",
        f"- 최종 테스트: {C.TEST_START} ~ (선정에 미사용)",
        f"- 피크 지표: 일자별 상위 {K}개 15분 구간. {HIT_COL} = 예측 상위 {K}개 중 실제 상위 {K}개 포함 비율, "
        f"{TOPK_MAE_COL} = 정렬된 상위 {K}개 값의 MAE", "",
        "## 1. 데이터 품질 진단", "",
        f"- 기간 {dq['period'][0]} ~ {dq['period'][1]}, {dq['n_days']}일",
        f"- '시간' 컬럼 오류 복원: {dq['hour_repaired']['rows']}행 ({', '.join(dq['hour_repaired']['dates'])}), "
        f"복원 후 결측 시각 {dq['missing_hours_after_repair']}, 중복 시각 {dq['duplicate_ts_after_repair']}",
        f"- 인건비 규칙(09~17시 1.0, 그 외 1.5) 성립: {dq['labor_rule_holds_on_unrepaired_rows']}, "
        f"재계산 {dq['labor_mult_recomputed_rows']}행",
        f"- 평균 컬럼과 네 구간 평균의 최대 차이: {dq['avg_col_vs_mean_of_4_max_abs_diff']} (반올림)",
        f"- 공장인원 = 생산량/(15분+30분+45분+60분), 최대 오차 "
        f"{dq['leakage_factory_staff']['max_abs_diff_vs_prod_over_power_sum']:.1e} → 제외",
        f"- 전기요금(월별 상수): {dq['tariff_by_month']}",
        f"- 기상 결측(보간 전): {dq['weather_missing_before']}",
        f"- 전력 0 구간: {dq['zero_power_15min_slots']['n']}개 ({', '.join(dq['zero_power_15min_slots']['dates'])})",
        f"- 복제일: {dup['n_days']}일 중 {dup['n_days_in_duplicate_groups']}일이 복제그룹 소속, "
        f"고유 프로파일 {dup['n_unique_profiles']}개, 월별 {dup['days_in_duplicate_groups_by_month']}",
        "", md_table(groups.head(8)), "",
        "## 2. 교차검증 (주 단위 walk-forward, 채점일 가중 평균)", "",
        md_table(cv.groupby("fold")[["n_train_days", "n_purged_days", "n_scored_days"]].first().reset_index()), "",
        md_table(cv_mean[["model", "lookback", *METRIC_COLS, "mae_std_across_folds"]]), "",
        "폴드별 MAE", "", md_table(cv.pivot(index="model", columns="fold", values="mae").reset_index()), "",
        f"## 3. 최종 테스트 ({C.TEST_START} ~)", "",
        md_table(test[["model", "kind", "lookback", *METRIC_COLS]]), "",
        "## 3-1. 입력 범위 비교 (1d / 7d)", "",
        md_table(lookback_table(cv_mean, test)), "",
    ]
    if demo is not None:
        L += ["## 4. 무작위 분할 단위별 MAE (GBM)", "", md_table(demo), ""]
    L += [f"## 5. 오류 슬라이스 ({selected}, 최종 테스트)", ""]
    for k, v in slices.items():
        L += [f"### {k}", "", md_table(v), ""]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")


def make_figures(out: Path, preds: pd.DataFrame, selected: str, slices: dict):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # (1) 테스트 첫 2주: 실측·예측과 일자별 상위 K개 구간
    t0 = preds["ts"].min().normalize()
    w = preds[preds["ts"] < t0 + pd.Timedelta(days=14)].copy()
    a = daily_matrix(w, "kw").to_numpy()
    p = daily_matrix(w, selected).to_numpy()
    top_a = np.argsort(-a, axis=1, kind="stable")[:, :K]
    top_p = np.argsort(-p, axis=1, kind="stable")[:, :K]
    day_idx = np.arange(len(a))[:, None] * 96
    fig = Figure(figsize=(14, 4))
    ax = fig.subplots()
    ax.plot(w["ts"], w["kw"], lw=1, color="black", label="actual")
    ax.plot(w["ts"], w[selected], lw=1, label=selected)
    ax.scatter(w["ts"].to_numpy()[(day_idx + top_a).ravel()], a.ravel()[(day_idx + top_a).ravel()],
               s=10, color="black", label=f"actual top-{K}")
    ax.scatter(w["ts"].to_numpy()[(day_idx + top_p).ravel()], p.ravel()[(day_idx + top_p).ravel()],
               s=12, marker="x", color="tab:red", label=f"predicted top-{K}")
    ax.set_ylabel("15-min demand (kW)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_first_2weeks.png", dpi=130)

    # (2) 시간대별 MAE와 편향
    h = slices["by_hour"]
    fig = Figure(figsize=(10, 4))
    ax1 = fig.subplots()
    ax1.bar(h["hour"], h["mae"], color="#8aa", label="MAE")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.set_xlabel("hour")
    ax1.set_ylabel("kW")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_by_hour.png", dpi=130)

    # (3) 일자별 상위 K개 적중 수
    d = slices["by_day"]
    fig = Figure(figsize=(12, 3.5))
    ax = fig.subplots()
    ax.bar(pd.to_datetime(d["date"]), d[f"top{K}_hits"], color="#8aa")
    ax.set_ylim(0, K)
    ax.set_ylabel(f"top-{K} hits")
    fig.tight_layout()
    fig.savefig(fig_dir / f"daily_top{K}_hits.png", dpi=130)
