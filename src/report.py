"""결과물 저장: CSV, results.md(보고서 초안용 표), 그림."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

REG_COLS = ["mae", "rmse", "nmae", "mae_excl_outage", "hourly_mae", "daily_max_mae",
            "daily_max_bias", "peak_time_hit_1h"]
PEAK_COLS = ["precision", "recall", "f1", "fn", "fp"]


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


def write_results_md(out: Path, dq: dict, groups: pd.DataFrame, cv: pd.DataFrame, test: pd.DataFrame,
                     thr: float, selected: str, abl: pd.DataFrame | None, demo: pd.DataFrame | None,
                     slices: dict, backend: str):
    from .experiment import wavg
    cv_mean = pd.DataFrame([
        {"model": m, "kind": g["kind"].iloc[0], **{c: wavg(g, c) for c in REG_COLS + PEAK_COLS},
         "mae_std_across_folds": g["mae"].std()}
        for m, g in cv.groupby("model")
    ]).sort_values("mae")
    dup = dq["duplicate_days"]
    L = [
        "# 실험 결과 요약 (자동 생성)", "",
        f"- GBM 백엔드: {backend}",
        f"- 과제: day-ahead, D일 00:00에 D일 15분 수요전력 96개 예측 (생산계획 가정: {C.PRODUCTION_PLAN}, 기상 사용: {C.USE_WEATHER})",
        f"- 모델 선정: {C.CV_START}부터 {C.CV_FOLD_DAYS}일 단위 walk-forward, 점예측 모델 중 CV MAE 최소 → **{selected}**",
        f"- 최종 테스트: {C.TEST_START} ~ 끝 (선정에 미사용), 피크 임계값 {thr:.1f} kW "
        f"(학습기간 15분 수요 {int(100 * C.PEAK_QUANTILE)}분위)", "",
        "## 1. 데이터 품질 진단", "",
        f"- 기간 {dq['period'][0]} ~ {dq['period'][1]}, {dq['n_days']}일",
        f"- '시간' 컬럼 오염 복구: {dq['hour_repaired']['rows']}행 ({', '.join(dq['hour_repaired']['dates'])}) "
        f"→ 복구 후 결측 시각 {dq['missing_hours_after_repair']}, 중복 시각 {dq['duplicate_ts_after_repair']}",
        f"- 인건비 규칙(09~17시 1.0, 그 외 1.5)이 복구 외 행에서 성립: {dq['labor_rule_holds_on_unrepaired_rows']}, "
        f"오염일 재계산 {dq['labor_mult_recomputed_rows']}행",
        f"- 평균 컬럼 = 4개 15분 값의 평균(최대 차이 {dq['avg_col_vs_mean_of_4_max_abs_diff']}, 반올림)",
        f"- 공장인원 = 생산량/(15분+30분+45분+60분), 최대 오차 "
        f"{dq['leakage_factory_staff']['max_abs_diff_vs_prod_over_power_sum']:.1e} → 타깃 누수로 제외",
        f"- 전기요금: 월별 상수 {dq['tariff_by_month']} → 예측 피처 제외, 비용 시뮬레이션용",
        f"- 기상 결측(보간 전): {dq['weather_missing_before']}",
        f"- 전력 0(15분 단위): {dq['zero_power_15min_slots']['n']}칸 ({', '.join(dq['zero_power_15min_slots']['dates'])}), "
        f"시간 단위로는 {dq['zero_power_hours']['n']}시간 → 학습 제외, 래그 계산 시 같은 요일·칸 최근 {C.LAG_IMPUTE_WEEKS}주 중앙값으로 대체"
        f"({'사용' if C.LAG_IMPUTE_OUTAGE else '미사용'})",
        f"- 복제일: {dup['n_days']}일 중 {dup['n_days_in_duplicate_groups']}일이 복제그룹 소속, "
        f"고유 프로파일 {dup['n_unique_profiles']}개, 월별 {dup['days_in_duplicate_groups_by_month']}",
        "", "상위 복제그룹:", "", md_table(groups.head(8)), "",
        "## 2. 교차검증 (주 단위 walk-forward, 채점일 가중 평균)", "",
        md_table(cv.groupby("fold")[["n_train_days", "n_purged_days", "n_scored_days"]].first().reset_index()), "",
        md_table(cv_mean[["model", "kind", *REG_COLS, "mae_std_across_folds", *PEAK_COLS]]), "",
        "지표는 폴드별 값의 채점일 가중평균(fn·fp는 폴드당 평균 건수)이라 precision·recall·f1이 서로 정확히 맞물리지 않을 수 있다.", "",
        "폴드별 MAE:", "", md_table(cv.pivot(index="model", columns="fold", values="mae").reset_index()), "",
        f"## 3. 최종 테스트 ({C.TEST_START} ~, 전부 원본 데이터)", "",
        md_table(test[["model", "kind", *REG_COLS, *PEAK_COLS]]), "",
    ]
    if abl is not None:
        L += ["## 4. Ablation (GBM, CV 채점일 가중 평균)", "", md_table(abl), ""]
    if demo is not None:
        L += ["## 5. 분할 방식에 따른 점수 부풀림 (GBM)", "",
              "세 분할 모두 1년 전체에서 무작위로 20%를 뽑고 묶는 단위만 다르다. 행→일: 인접 15분 시점을 통한 누수, "
              "일→복제그룹: 복제본을 통한 누수 규모. 시간 순서를 지킨 결과는 2·3절 참고.", "",
              md_table(demo), ""]
    L += [f"## 6. 오류 슬라이스 ({selected}, 최종 테스트)", ""]
    for k, v in slices.items():
        L += [f"### {k}", "", md_table(v), ""]
    (out / "results.md").write_text("\n".join(L), encoding="utf-8")


def make_figures(out: Path, preds: pd.DataFrame, selected: str, thr: float, slices: dict):
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    # (1) 테스트 첫 2주 실측 vs 예측
    t0 = preds["ts"].min().normalize()
    w = preds[preds["ts"] < t0 + pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(w["ts"], w["kw"], lw=1, label="actual", color="black")
    ax.plot(w["ts"], w[selected], lw=1, label=selected)
    ax.plot(w["ts"], w["naive_lastweek"], lw=0.8, alpha=0.6, label="naive_lastweek")
    ax.axhline(thr, ls="--", lw=0.8, color="red", label=f"peak threshold {thr:.0f}")
    ax.set_ylabel("15-min demand (kW)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "test_first_2weeks.png", dpi=130)
    plt.close(fig)

    # (2) 시간대별 MAE와 FN/FP
    h = slices["by_hour"]
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(h["hour"], h["mae"], color="#8aa")
    ax1.set_xlabel("hour")
    ax1.set_ylabel("MAE (kW)")
    ax2 = ax1.twinx()
    ax2.plot(h["hour"], h["fn"], "o-", color="red", label="FN (missed peaks)")
    ax2.plot(h["hour"], h["fp"], "s--", color="orange", label="FP (false alarms)")
    ax2.set_ylabel("count")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_by_hour.png", dpi=130)
    plt.close(fig)

    # (3) 일 최대수요 실측 vs 예측
    g = preds.groupby("date")[["kw", selected]].max()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(g.index, g["kw"], "o-", ms=3, color="black", label="actual daily max")
    ax.plot(g.index, g[selected], "o-", ms=3, label=f"{selected} daily max")
    ax.axhline(thr, ls="--", lw=0.8, color="red")
    ax.set_ylabel("daily max 15-min demand (kW)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "daily_max_demand.png", dpi=130)
    plt.close(fig)
