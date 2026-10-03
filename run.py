#!/usr/bin/env python
"""원본 CSV → 정제 → 15분 피처 → 교차검증 → 홀드아웃 → ablation → 결과물 생성을 한 번에 실행.

사용법:
    python run.py                                  # config.py 기본 경로 사용
    python run.py --data path/to/okm_augumented_2021.csv --out outputs
"""
from __future__ import annotations

import argparse
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src import config as C
from src import report
from src.data import clean, load_raw
from src.experiment import (error_slices, run_ablations, run_cv, run_leakage_demo,
                            run_test, select_model)
from src.features import build_features, feature_columns, to_long
from src.models import gbm_backend, lstm_available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=C.DATA_PATH)
    ap.add_argument("--out", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--fast", action="store_true", help="ablation·누수 시연 생략(반복 실험용)")
    ap.add_argument("--skip-lstm", action="store_true", help="LSTM 비교 모델 생략(torch가 없으면 자동 생략)")
    args = ap.parse_args()

    if args.skip_lstm:
        C.RUN_LSTM = False
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def step(msg):
        print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)

    step("1/7 데이터 로드·정제")
    hourly, dq, groups = clean(load_raw(args.data))
    hourly.to_csv(args.out / "clean_hourly.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(args.out / "duplicate_groups.csv", index=False, encoding="utf-8-sig")
    report.save_json(dq, args.out / "data_quality.json")

    step("2/7 15분 단위 변환·피처 생성")
    feat = build_features(to_long(hourly))
    cols = feature_columns()
    step(f"      {len(feat):,}행, 피처 {len(cols)}개(7d) / {len(feature_columns(lookback='1d'))}개(1d), "
         f"LSTM {'사용: ' + ', '.join(C.LSTM_MODELS) if lstm_available() else '생략(torch 미설치 또는 --skip-lstm)'}")

    step("3/7 주 단위 walk-forward 교차검증")
    cv = run_cv(feat)
    cv.to_csv(args.out / "metrics_cv.csv", index=False)
    selected = select_model(cv)
    step(f"      선정 모델: {selected}")

    step("4/7 최종 테스트 (테스트 직전까지 전체로 재학습)")
    test, preds, thr = run_test(feat)
    test.to_csv(args.out / "metrics_test.csv", index=False)
    preds.to_csv(args.out / "predictions_test.csv", index=False)

    abl = demo = None
    if not args.fast:
        step("5/7 ablation (CV 기준)")
        abl = run_ablations(feat)
        abl.to_csv(args.out / "ablation_cv.csv", index=False)

    if C.RUN_LEAKAGE_DEMO and not args.fast:
        step("6/7 분할 방식별 누수 시연")
        demo = run_leakage_demo(feat, cols)
        demo.to_csv(args.out / "leakage_demo.csv", index=False)

    step("7/7 오류 슬라이스·그림·요약")
    slices = error_slices(preds, selected, thr)
    for k, v in slices.items():
        v.to_csv(args.out / f"errors_{k}.csv", index=False)
    report.make_figures(args.out, preds, selected, thr, slices)
    report.write_results_md(args.out, dq, groups, cv, test, thr, selected, abl, demo, slices, gbm_backend())

    run_info = {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__, "gbm_backend": gbm_backend(), "seed": C.SEED,
        "torch": _torch_version(), "lstm_run": lstm_available(),
        "data": str(args.data), "selected_model": selected, "runtime_sec": round(time.time() - t0, 1),
        "config": {k: getattr(C, k) for k in dir(C) if k.isupper()},
    }
    report.save_json(run_info, args.out / "run_info.json")
    step(f"완료 → {args.out}/results.md")


def _torch_version():
    try:
        import torch
        return f"{torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
    except ImportError:
        return None


if __name__ == "__main__":
    main()
