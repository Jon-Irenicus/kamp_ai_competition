"""시퀀스 모델(LSTM)의 SHAP 분석.

GradientExplainer(expected gradients)로 예측일 96개 구간 각각의 출력에 대한 입력 기여를 계산한다.
입력은 (시점 × 피처) 구조이므로 시점 축을 합산해 피처별 기여로 정리하고, 예측 구간 하나를 샘플 하나로 본다.
  - 과거 입력 피처값: 과거 window 구간의 평균
  - 미래 입력 피처값: 해당 예측 구간의 값
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import SLOTS_PER_DAY


def _label(col: str) -> str:
    """요일 one-hot(dow_0=월 … dow_6=일)은 요일명으로 표시한다."""
    if col.startswith("dow_") and col[4:].isdigit():
        return f"요일={'월화수목금토일'[int(col[4:])]}"
    return col


def day_starts(feat: pd.DataFrame, dates) -> np.ndarray:
    """지정 일자의 00:00 행 위치."""
    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    m = feat["date"].isin(set(d)) & (feat["slot"] == 0)
    return np.flatnonzero(m.to_numpy())


def sequence_shap(model, feat: pd.DataFrame, target_dates, background_dates, nsamples: int = 200):
    """반환: (SHAP 값 [n일×96, 피처], 피처값 [n일×96, 피처], 기준값). SHAP 값 단위는 kW."""
    import shap
    import torch

    if not getattr(model, "needs_full_frame", False):
        raise TypeError("시퀀스 모델(LSTMForecaster)만 지원한다.")
    past_cols, fut_cols = model.input_names()
    names = [f"{_label(c)} (과거)" for c in past_cols] + [f"{_label(c)} (예측일)" for c in fut_cols]

    tgt, bg = day_starts(feat, target_dates), day_starts(feat, background_dates)
    bg = bg[bg >= model.window]
    Xb, Fb = (torch.from_numpy(a) for a in model.day_inputs(feat, bg))
    Xt, Ft = (torch.from_numpy(a) for a in model.day_inputs(feat, tgt))

    net = model.net_.cpu().eval()
    explainer = shap.GradientExplainer(net, [Xb, Fb])
    sx, sf = explainer.shap_values([Xt, Ft], nsamples=nsamples, rseed=C.SEED)
    # [n, 시점, 피처, 출력 96] → 시점 합산 → [n, 출력 96, 피처]
    sx = sx.sum(axis=1).transpose(0, 2, 1)
    sf = sf.sum(axis=1).transpose(0, 2, 1)
    values = np.concatenate([sx, sf], axis=2).reshape(-1, len(names)) * model.y_std_

    h = np.arange(SLOTS_PER_DAY)
    past = np.stack([feat[past_cols].to_numpy(float)[s - model.window:s].mean(axis=0) for s in tgt])
    past = np.repeat(past, SLOTS_PER_DAY, axis=0)
    fut = feat[fut_cols].to_numpy(float)[(tgt[:, None] + h).ravel()]
    index = pd.MultiIndex.from_arrays([feat["ts"].to_numpy()[(tgt[:, None] + h).ravel()]], names=["ts"])

    with torch.no_grad():
        base = float(net(Xb, Fb).mean()) * model.y_std_ + model.y_mean_
    return (pd.DataFrame(values, columns=names, index=index),
            pd.DataFrame(np.hstack([past, fut]), columns=names, index=index), base)
