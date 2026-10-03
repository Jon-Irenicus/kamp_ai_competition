#!/usr/bin/env python
"""GBM 하이퍼파라미터 탐색. 주간 walk-forward CV(run.py와 같은 폴드·채점 기준)만 사용하고 테스트는 보지 않는다.

사용법:
    python tune.py                       # 기본 60회 탐색
    python tune.py --trials 100 --timeout 3600
    python tune.py --trials 2 --seed-check 0 --max-trees 300   # 동작 확인용

- Optuna가 설치돼 있으면 TPE 탐색과 가지치기(pruning)를 쓰고, 없으면 같은 탐색 공간에서 랜덤 탐색을 한다.
- 트리 개수는 탐색하지 않고, 한 번 넉넉히 학습한 뒤 체크포인트별 예측으로 7개 폴드 공통 최적값을 고른다.
- 결과: outputs/tuning_trials.csv(모든 시도), outputs/tuning_summary.md(추천값과 config.py에 붙일 코드).
  추천값은 자동 반영하지 않는다. config.py에 옮긴 뒤 run.py를 다시 실행한다.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.data import clean, load_raw
from src.experiment import _train_rows, _weights, cv_folds
from src.features import build_features, feature_columns, to_long
from src.models import _HAS_LGB, gbm_backend

try:
    import optuna
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

TUNE_LR = 0.05
MAX_TREES = 2000
CHECKPOINTS = [100, 200, 300, 500, 700, 1000, 1500, 2000]

# (이름, 하한, 상한, 로그척도, 정수) — 백엔드별 탐색 공간
SPACE_LGB = [
    ("num_leaves", 15, 127, True, True),
    ("min_child_samples", 20, 300, True, True),
    ("subsample", 0.5, 1.0, False, False),
    ("colsample_bytree", 0.5, 1.0, False, False),
    ("reg_lambda", 1e-3, 30.0, True, False),
    ("min_split_gain", 0.0, 0.5, False, False),
]
SPACE_HGB = [
    ("max_leaf_nodes", 15, 127, True, True),
    ("min_samples_leaf", 20, 300, True, True),
    ("l2_regularization", 1e-3, 30.0, True, False),
    ("max_features", 0.5, 1.0, False, False),
]
SPACE = SPACE_LGB if _HAS_LGB else SPACE_HGB
COMPLEXITY_KEY = "num_leaves" if _HAS_LGB else "max_leaf_nodes"
MIN_LEAF_KEY = "min_child_samples" if _HAS_LGB else "min_samples_leaf"


LIB_DEFAULTS = {"min_split_gain": 0.0, "max_features": 1.0}   # config에 없으면 라이브러리 기본값


def baseline_params() -> dict:
    """현재 config.py 설정을 탐색 공간의 키로 옮긴 것(비교 기준, 0번 시도)."""
    src = C.LGB_PARAMS if _HAS_LGB else C.HGB_PARAMS
    return {name: src.get(name, LIB_DEFAULTS.get(name)) for name, *_ in SPACE}


def make_model(params: dict, seed: int):
    if _HAS_LGB:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=MAX_TREES, learning_rate=TUNE_LR, subsample_freq=1,
                                 random_state=seed, deterministic=True, force_row_wise=True,
                                 verbose=-1, **params)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=MAX_TREES, learning_rate=TUNE_LR, early_stopping=False,
                                         random_state=seed, **params)


def checkpoint_predictions(model, X) -> np.ndarray:
    """(체크포인트 수, 행 수) 예측 행렬. 한 번 학습한 모델에서 앞의 k개 트리만 쓴 예측을 꺼낸다."""
    if _HAS_LGB:
        return np.vstack([model.predict(X, num_iteration=k) for k in CHECKPOINTS])
    out, want = [], set(CHECKPOINTS)
    for i, p in enumerate(model.staged_predict(X), start=1):
        if i in want:
            out.append(p)
    return np.vstack(out)


def evaluate(params: dict, folds, cols, seed: int = C.SEED, report=None) -> dict:
    """폴드별·체크포인트별 MAE를 계산하고, 폴드 공통 최적 체크포인트를 고른다."""
    maes, weights = [], []
    for i, (_, tr, val, score, _) in enumerate(folds):
        tr = _train_rows(tr, C.COPY_POLICY)
        model = make_model(params, seed)
        model.fit(tr[cols], tr["kw"], sample_weight=_weights(tr, C.COPY_POLICY))
        y = val["kw"].to_numpy()[score]
        P = checkpoint_predictions(model, val[cols])[:, score]
        maes.append(np.abs(P - y).mean(axis=1))
        weights.append(int(val.loc[score, "date"].nunique()))
        if report is not None:
            running = np.average(np.vstack(maes), axis=0, weights=weights)
            report(float(running.min()), i)
    M = np.vstack(maes)                                   # (폴드, 체크포인트)
    cv = np.average(M, axis=0, weights=weights)
    j = int(np.argmin(cv))
    return {"cv_mae": float(cv[j]), "n_estimators": CHECKPOINTS[j],
            "fold_mae": [round(float(x), 3) for x in M[:, j]], "weights": weights}


def sample_random(rng: np.random.Generator) -> dict:
    p = {}
    for name, lo, hi, log, is_int in SPACE:
        v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
        p[name] = int(round(v)) if is_int else float(v)
    return p


def suggest(trial) -> dict:
    p = {}
    for name, lo, hi, log, is_int in SPACE:
        p[name] = trial.suggest_int(name, lo, hi, log=log) if is_int else trial.suggest_float(name, lo, hi, log=log)
    return p


def run_search(folds, cols, n_trials: int, timeout: float | None, seed: int) -> pd.DataFrame:
    rows, t_start = [], time.time()

    def record(num, params, res, state, sec):
        rows.append({"trial": num, "state": state, "seconds": round(sec, 1), **params,
                     "n_estimators": res.get("n_estimators"), "cv_mae": res.get("cv_mae"),
                     "fold_mae": json.dumps(res.get("fold_mae"))})
        best = min((r["cv_mae"] for r in rows if r["cv_mae"] is not None), default=float("nan"))
        print(f"[{time.time() - t_start:7.1f}s] trial {num:3d} {state:8s} cv_mae={res.get('cv_mae')} "
              f"(best {best:.3f})", flush=True)

    if _HAS_OPTUNA:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2))
        study.enqueue_trial(baseline_params())

        def objective(trial):
            params, t0 = suggest(trial), time.time()

            def report(value, step):
                trial.report(value, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            try:
                res = evaluate(params, folds, cols, seed, report)
            except optuna.TrialPruned:
                record(trial.number, params, {}, "pruned", time.time() - t0)
                raise
            record(trial.number, params, res, "complete", time.time() - t0)
            return res["cv_mae"]

        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    else:
        rng = np.random.default_rng(seed)
        for num in range(n_trials):
            if timeout and time.time() - t_start > timeout:
                break
            params, t0 = (baseline_params() if num == 0 else sample_random(rng)), time.time()
            record(num, params, evaluate(params, folds, cols, seed), "complete", time.time() - t0)
    return pd.DataFrame(rows)


def recommend(trials: pd.DataFrame, base_mae: float) -> tuple[pd.Series, pd.Series]:
    """1-표준오차 규칙: 최고 점수와 '통계적으로 구분되지 않는' 후보 중 가장 단순한(잎 적고, 잎 최소샘플 큰) 것.
    모든 후보가 같은 폴드로 평가되므로 폴드 난이도 차이를 빼기 위해 폴드별 '짝지은 차이'로 표준오차를 계산한다."""
    done = trials[trials["state"] == "complete"].copy()
    best = done.loc[done["cv_mae"].idxmin()]
    best_f = np.array(json.loads(best["fold_mae"]))

    def within_se(row) -> bool:
        diff = np.array(json.loads(row["fold_mae"])) - best_f
        se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
        return (row["cv_mae"] - best["cv_mae"]) <= se

    near = done[done.apply(within_se, axis=1) & (done["cv_mae"] < base_mae)]   # 기준보다 나은 후보만
    if near.empty:
        return best, best
    pick = near.sort_values([COMPLEXITY_KEY, MIN_LEAF_KEY], ascending=[True, False]).iloc[0]
    return best, pick


def main():
    global CHECKPOINTS, MAX_TREES
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=C.DATA_PATH)
    ap.add_argument("--out", type=Path, default=C.OUTPUT_DIR)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--timeout", type=float, default=None, help="초 단위 전체 제한 시간")
    ap.add_argument("--seed-check", type=int, default=3, help="추천값을 다른 시드 몇 개로 재확인할지(0이면 생략)")
    ap.add_argument("--max-trees", type=int, default=MAX_TREES, help="탐색 시 최대 트리 수(빠른 확인용으로 줄일 수 있음)")
    args = ap.parse_args()
    MAX_TREES = args.max_trees
    CHECKPOINTS = sorted({k for k in CHECKPOINTS if k < MAX_TREES} | {MAX_TREES})
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"백엔드: {gbm_backend()} / 탐색기: {'Optuna TPE + 가지치기' if _HAS_OPTUNA else '랜덤 탐색 (optuna 미설치)'}")
    hourly, _, _ = clean(load_raw(args.data))
    feat = build_features(to_long(hourly))
    cols = feature_columns()
    folds = list(cv_folds(feat))
    print(f"폴드 {len(folds)}개, 피처 {len(cols)}개, learning_rate={TUNE_LR}, 트리 체크포인트 {CHECKPOINTS}")

    trials = run_search(folds, cols, args.trials, args.timeout, C.SEED)
    trials.to_csv(args.out / "tuning_trials.csv", index=False)
    base = trials.iloc[0]
    best, pick = recommend(trials, float(base["cv_mae"]))
    params = {name: (int(pick[name]) if is_int else round(float(pick[name]), 4)) for name, _, _, _, is_int in SPACE}

    base_f, pick_f = np.array(json.loads(base["fold_mae"])), np.array(json.loads(pick["fold_mae"]))
    wins = int((pick_f < base_f).sum())

    seed_line, seed_std = "", 0.0
    if args.seed_check > 0:
        res = [evaluate(params, folds, cols, seed=C.SEED + s)["cv_mae"] for s in range(1, args.seed_check + 1)]
        seed_std = float(np.std(res))
        seed_line = f"- 추천값을 다른 시드 {args.seed_check}개로 재평가: CV MAE 평균 {np.mean(res):.3f}, 표준편차 {seed_std:.3f}"

    gain = float(base["cv_mae"] - pick["cv_mae"])
    adopt = gain > 0 and wins > len(pick_f) / 2 and gain > seed_std
    reasons = []
    if gain <= 0:
        reasons.append("CV MAE가 기준보다 낫지 않음")
    if wins <= len(pick_f) / 2:
        reasons.append(f"기준보다 나은 폴드가 과반이 아님({wins}/{len(pick_f)})")
    if gain > 0 and gain <= seed_std:
        reasons.append("개선 폭이 시드 간 표준편차 이하")
    verdict = ("**채택 권장**: 아래 코드를 config.py에 반영하고 run.py를 다시 실행하세요." if adopt
               else "**채택 비권장**: " + ", ".join(reasons) + ". 현재 config를 유지하거나 탐색 횟수를 늘리세요.")

    name = "LGB_PARAMS" if _HAS_LGB else "HGB_PARAMS"
    fixed = (dict(n_estimators=int(pick["n_estimators"]), learning_rate=TUNE_LR, subsample_freq=1) if _HAS_LGB
             else dict(max_iter=int(pick["n_estimators"]), learning_rate=TUNE_LR, early_stopping=False))
    snippet = f"{name} = dict(\n" + "".join(f"    {k}={v!r},\n" for k, v in {**fixed, **params}.items()) + ")"

    lines = [
        "# 하이퍼파라미터 탐색 결과 (자동 생성)", "",
        f"- 백엔드: {gbm_backend()}, 탐색기: {'Optuna TPE' if _HAS_OPTUNA else '랜덤 탐색'}, "
        f"완료 {int((trials['state'] == 'complete').sum())}회 / 가지치기 {int((trials['state'] == 'pruned').sum())}회",
        f"- 기준(현재 config, 0번 시도): CV MAE {base['cv_mae']:.3f} (트리 {int(base['n_estimators'])}개, learning_rate {TUNE_LR})",
        f"- 최고 점수: CV MAE {best['cv_mae']:.3f} (시도 {int(best['trial'])})",
        f"- 추천(1-표준오차 규칙, 폴드별 짝지은 차이 기준): CV MAE {pick['cv_mae']:.3f} (시도 {int(pick['trial'])}), "
        + (f"기준 대비 {'개선' if gain > 0 else '악화'} {abs(gain):.3f}" if abs(gain) > 1e-9 else "기준과 동일")
        + f", 기준보다 나은 폴드 {wins}/{len(pick_f)}개",
        seed_line, "",
        verdict, "",
        "채택 조건: 기준보다 CV MAE가 낮고, 과반의 폴드에서 낫고, 개선 폭이 시드 간 표준편차보다 클 것.", "",
        "추천 후보를 `src/config.py`에 붙일 코드 (채택 권장일 때만 반영):", "", "```python", snippet, "```", "",
        "폴드별 MAE (기준 → 추천): " + ", ".join(f"{a:.2f}→{b:.2f}" for a, b in zip(base_f, pick_f)),
    ]
    (args.out / "tuning_summary.md").write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
