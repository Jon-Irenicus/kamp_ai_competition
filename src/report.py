"""결과 저장: JSON, 요약표, 그림."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure   # pyplot 미사용: 전역 백엔드를 변경하지 않음

from .metrics import HIT_COL, K, TOL_HIT_COL, TOPK_MAE_COL, daily_matrix

METRIC_COLS = ["mae", "rmse", "daily_max_mae", HIT_COL, TOL_HIT_COL, TOPK_MAE_COL]
FAMILY = {"naive_yesterday": "naive", "naive_lastweek": "naive", "gbm_1d": "gbm", "gbm": "gbm",
          "lstm_1d": "lstm", "lstm_7d": "lstm"}


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def cv_summary(cv: pd.DataFrame) -> pd.DataFrame:
    from .experiment import wavg
    return pd.DataFrame([
        {"model": m, "lookback": g["lookback"].iloc[0],
         **{c: wavg(g, c) for c in METRIC_COLS if c in g}, "mae_std_across_folds": g["mae"].std()}
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
