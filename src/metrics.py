"""평가지표.

전체 오차: MAE, RMSE, NMAE, bias
일 최대수요: daily_max_mae, daily_max_bias
피크(일자별 상위 K개 15분 구간, K = config.PEAK_TOP_K)
  - top{K}_hit: 예측 상위 K개 구간 중 실제 상위 K개에 포함된 비율(순위 무관). 일자별로 계산해 평균한다.
                실제값 동률을 고려해 K번째로 큰 실제값 이상인 구간을 모두 실제 피크로 본다.
  - top{K}_mae: 실제 상위 K개 값과 예측 상위 K개 값을 각각 내림차순 정렬해 순위끼리 비교한 MAE.
                발생 시각과 무관하게 피크 수준의 정확도를 본다(시각은 top{K}_hit이 평가).
  - topk_hits_within: top{K}_hit의 시각 허용 변형. 예측 상위 K개 구간이 같은 날 실제 피크 구간과
                ±tol구간 이내이면 적중으로 본다(실제 피크 하나에 여러 예측 구간이 대응될 수 있다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

K = C.PEAK_TOP_K
HIT_COL, TOPK_MAE_COL = f"top{K}_hit", f"top{K}_mae"


def regression(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    e = np.asarray(p, dtype=float) - y
    mae = float(np.mean(np.abs(e)))
    return {"mae": mae, "rmse": float(np.sqrt(np.mean(e ** 2))),
            "nmae": mae / float(np.mean(y)), "bias": float(np.mean(e))}


def daily_matrix(d: pd.DataFrame, col: str) -> pd.DataFrame:
    """(일자 × 96구간) 행렬. 하루 96구간이 모두 있어야 한다."""
    m = d.pivot(index="date", columns="slot", values=col)
    if m.shape[1] != 96 or m.isna().any().any():
        raise ValueError("일자별 96구간이 모두 있어야 합니다.")
    return m


def topk_scores(actual: np.ndarray, pred: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """일자별 상위 k개 적중 수와 상위 k개 MAE. 입력 shape: (일자, 96)."""
    pred_top = np.argsort(-pred, axis=1, kind="stable")[:, :k]          # 예측 동률은 이른 시각 우선
    actual_sorted = -np.sort(-actual, axis=1)
    kth = actual_sorted[:, k - 1]
    hits = (np.take_along_axis(actual, pred_top, axis=1) >= kth[:, None]).sum(axis=1)
    pred_sorted = -np.sort(-pred, axis=1)
    mae = np.abs(actual_sorted[:, :k] - pred_sorted[:, :k]).mean(axis=1)
    return hits, mae


def topk_hits_within(actual: np.ndarray, pred: np.ndarray, k: int = K, tol: int = 0) -> np.ndarray:
    """일자별 예측 상위 k개 구간 중 실제 피크 구간과 ±tol구간 이내인 구간 수. 입력 shape: (일자, 96).
    tol=0이면 topk_scores의 적중 수와 같다. 허용 범위는 날짜를 넘지 않는다."""
    pred_top = np.argsort(-pred, axis=1, kind="stable")[:, :k]
    kth = -np.sort(-actual, axis=1)[:, k - 1]
    peak = actual >= kth[:, None]
    if tol > 0:
        pad = np.pad(peak, ((0, 0), (tol, tol)))
        peak = np.stack([pad[:, s:s + actual.shape[1]] for s in range(2 * tol + 1)]).any(axis=0)
    return np.take_along_axis(peak, pred_top, axis=1).sum(axis=1)


def daily_peak(d: pd.DataFrame) -> dict:
    a = daily_matrix(d, "kw").to_numpy()
    p = daily_matrix(d, "pred").to_numpy()
    diff = p.max(axis=1) - a.max(axis=1)
    hits, tmae = topk_scores(a, p)
    return {"daily_max_mae": float(np.abs(diff).mean()), "daily_max_bias": float(diff.mean()),
            HIT_COL: float((hits / K).mean()), TOPK_MAE_COL: float(tmae.mean())}


def evaluate(te: pd.DataFrame, pred) -> dict:
    d = te[["date", "slot", "kw"]].copy()
    d["pred"] = np.asarray(pred, dtype=float)
    out = regression(d["kw"], d["pred"])
    out.update(daily_peak(d))
    return out
