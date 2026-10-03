"""평가지표.

회귀 지표 외에, 대회 평가표의 F1·FN·FP 요구를 채우기 위해
"15분 수요전력 >= 임계값"을 피크 이벤트로 정의한 분류 지표를 함께 계산한다.
임계값은 항상 해당 분할의 학습 구간에서만 정한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    e = np.asarray(p, dtype=float) - y
    mae = float(np.mean(np.abs(e)))
    return {"mae": mae, "rmse": float(np.sqrt(np.mean(e ** 2))),
            "nmae": mae / float(np.mean(y)), "bias": float(np.mean(e))}


def peak_events(y, p, thr: float) -> dict:
    t = np.asarray(y) >= thr
    q = np.asarray(p) >= thr
    tp, fp, fn = int((t & q).sum()), int((~t & q).sum()), int((t & ~q).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"peak_thr": float(thr), "peak_pos": int(t.sum()), "precision": prec,
            "recall": rec, "f1": f1, "fn": fn, "fp": fp}


def daily_peak(d: pd.DataFrame) -> dict:
    """일 최대 15분 수요전력의 크기 오차와 발생 시각 적중률(±1시간)."""
    g = d.groupby("date")
    act, pred = g["kw"].max(), g["pred"].max()
    sa = d.loc[g["kw"].idxmax().to_numpy(), "slot"].to_numpy()
    sp = d.loc[g["pred"].idxmax().to_numpy(), "slot"].to_numpy()
    return {"daily_max_mae": float((pred - act).abs().mean()),
            "daily_max_bias": float((pred - act).mean()),
            "peak_time_hit_1h": float(np.mean(np.abs(sa - sp) <= 4))}


def evaluate(te: pd.DataFrame, pred, thr: float) -> dict:
    d = te[["date", "slot", "ts_hour", "kw", "zero_power"]].copy()
    d["pred"] = np.asarray(pred, dtype=float)
    out = regression(d["kw"], d["pred"])
    ex = ~d["zero_power"]
    r = regression(d.loc[ex, "kw"], d.loc[ex, "pred"])
    out.update(mae_excl_outage=r["mae"], nmae_excl_outage=r["nmae"])
    hourly = d.groupby("ts_hour")[["kw", "pred"]].mean()   # 해석 A: 시간평균 kW = 시간당 kWh
    out["hourly_mae"] = float((hourly["pred"] - hourly["kw"]).abs().mean())
    out.update(daily_peak(d))
    out.update(peak_events(d["kw"], d["pred"], thr))
    return out
