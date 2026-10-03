"""교차검증(주 단위 walk-forward), 최종 테스트, ablation, 누수 시연, 오류 슬라이스."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as C
from .features import feature_columns
from .metrics import evaluate, regression
from .models import make_gbm, make_models


# ── 공통 ───────────────────────────────────────────────────────────────
def _train_rows(tr: pd.DataFrame, policy: str) -> pd.DataFrame:
    tr = tr[~tr["zero_power"]]                  # 정전·셧다운 구간은 학습에서 제외
    if policy == "drop":
        tr = tr[~tr["is_copy_day"]]
    return tr


def _weights(tr: pd.DataFrame, policy: str):
    return (1.0 / tr["dup_group_size"]).to_numpy() if policy == "weight" else None


def fit_predict(model, tr, te, cols, policy=C.COPY_POLICY):
    tr = _train_rows(tr, policy)
    model.fit(tr[cols], tr["kw"], sample_weight=_weights(tr, policy))
    return model.predict(te[cols]), tr


def peak_threshold(tr: pd.DataFrame) -> float:
    """피크 임계값은 항상 학습 구간에서만 정한다."""
    return float(tr.loc[~tr["zero_power"], "kw"].quantile(C.PEAK_QUANTILE))


def wavg(df: pd.DataFrame, col: str, w: str = "n_scored_days") -> float:
    """NaN을 건너뛰는 가중평균(폴드별 지표 → 전체 CV 지표)."""
    m = df[col].notna() & (df[w] > 0)
    return float(np.average(df.loc[m, col], weights=df.loc[m, w])) if m.any() else np.nan


# ── 폴드 ───────────────────────────────────────────────────────────────
def cv_folds(feat: pd.DataFrame):
    """(이름, 학습, 검증, 채점마스크, 제거된 학습일 수)를 주 단위로 생성."""
    test_start = pd.Timestamp(C.TEST_START)
    for a in pd.date_range(C.CV_START, test_start - pd.Timedelta(days=1), freq=f"{C.CV_FOLD_DAYS}D"):
        b = min(a + pd.Timedelta(days=C.CV_FOLD_DAYS), test_start)
        val = feat[(feat["ts"] >= a) & (feat["ts"] < b)]
        tr = feat[feat["ts"] < a]
        purged = 0
        if C.PURGE_DUPLICATE_GROUPS:
            mask = tr["dup_group"].isin(set(val["dup_group"]))
            purged = int(tr.loc[mask, "date"].nunique())
            tr = tr[~mask]
        score = (~val["is_copy_day"]).to_numpy() if C.CV_SCORE_ORIGINAL_ONLY else np.ones(len(val), bool)
        yield a.strftime("%m-%d"), tr, val, score, purged


def train_test(feat: pd.DataFrame):
    t = pd.Timestamp(C.TEST_START)
    tr, te = feat[feat["ts"] < t], feat[feat["ts"] >= t]
    overlap = int(tr["dup_group"].isin(set(te["dup_group"])).sum())
    if overlap:
        print(f"[경고] 테스트와 복제그룹이 겹치는 학습 행 {overlap}개")
    return tr, te


# ── 실험 ───────────────────────────────────────────────────────────────
def run_cv(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for fold, tr, val, score, purged in cv_folds(feat):
        thr = peak_threshold(tr)
        for name, (kind, model) in make_models().items():
            p, used = fit_predict(model, tr, val, cols)
            r = evaluate(val[score], p[score], thr)
            r.update(fold=fold, model=name, kind=kind, n_train_days=int(used["date"].nunique()),
                     n_purged_days=purged, n_scored_days=int(val.loc[score, "date"].nunique()))
            rows.append(r)
    return pd.DataFrame(rows)


def select_model(cv: pd.DataFrame) -> str:
    """선정 규칙(사전 고정): 점예측 모델 중 CV MAE(채점일 가중) 최소. 테스트는 선정에 쓰지 않는다."""
    d = cv[cv["kind"] == "point"]
    s = pd.Series({m: wavg(g, "mae") for m, g in d.groupby("model")})
    return str(s.idxmin())


def run_test(feat: pd.DataFrame, cols: list[str]):
    """테스트 직전까지 전체로 재학습 → 테스트 구간 day-ahead 예측."""
    tr, te = train_test(feat)
    thr = peak_threshold(tr)
    rows, preds = [], te[["ts", "date", "hour", "slot", "dow", "kw", "prod", "zero_power"]].copy()
    for name, (kind, model) in make_models().items():
        p, _ = fit_predict(model, tr, te, cols)
        preds[name] = p
        r = evaluate(te, p, thr)
        r.update(model=name, kind=kind)
        rows.append(r)
    return pd.DataFrame(rows), preds, thr


def run_ablations(feat: pd.DataFrame) -> pd.DataFrame:
    """GBM 기준으로 설정을 하나씩 바꿔 CV 성능 변화를 본다(테스트는 건드리지 않음)."""
    variants = [
        ("baseline (config)", C.COPY_POLICY, C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("copy days dropped", "drop", C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("copy days weighted 1/n", "weight", C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("hourly production plan", C.COPY_POLICY, "hourly", C.USE_WEATHER),
        ("daily production plan only", C.COPY_POLICY, "daily", C.USE_WEATHER),
        ("no production plan", C.COPY_POLICY, "none", C.USE_WEATHER),
        ("no weather", C.COPY_POLICY, C.PRODUCTION_PLAN, False),
    ]
    folds = list(cv_folds(feat))
    rows, seen = [], set()
    for name, policy, prod, weather in variants:
        if (policy, prod, weather) in seen:
            continue
        seen.add((policy, prod, weather))
        cols = feature_columns(prod, weather)
        fr = []
        for fold, tr, val, score, _ in folds:
            p, _ = fit_predict(make_gbm(), tr, val, cols, policy)
            r = evaluate(val[score], p[score], peak_threshold(tr))
            r["n_scored_days"] = int(val.loc[score, "date"].nunique())
            fr.append(r)
        fr = pd.DataFrame(fr)
        rows.append({"variant": name, **{c: wavg(fr, c) for c in
                     ["mae", "nmae", "daily_max_mae", "f1", "fn", "fp"]}})
    return pd.DataFrame(rows)


def run_leakage_demo(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """같은 GBM이라도 분할 방식에 따라 점수가 얼마나 부풀려지는지 보여준다."""
    d = feat[~feat["zero_power"]].reset_index(drop=True)
    splits = {}
    a, b = train_test_split(d.index, test_size=0.2, random_state=C.SEED)
    splits["random rows (80/20)"] = (a, b)
    for label, g in [("random days (80/20)", "date"), ("random duplicate-groups (80/20)", "dup_group")]:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED).split(d, groups=d[g]))
        splits[label] = (d.index[a], d.index[b])
    rows = []
    for label, (a, b) in splits.items():
        m = make_gbm().fit(d.loc[a, cols], d.loc[a, "kw"])
        r = regression(d.loc[b, "kw"], m.predict(d.loc[b, cols]))
        rows.append({"split": label, "mae": r["mae"], "nmae": r["nmae"], "n_test_rows": len(b)})
    return pd.DataFrame(rows)


def error_slices(preds: pd.DataFrame, model: str, thr: float) -> dict[str, pd.DataFrame]:
    """오차와 FN/FP가 어떤 조건에 몰리는지 — 오류분석(평가항목 3)의 출발점."""
    d = preds.copy()
    d["err"] = d[model] - d["kw"]
    d["fn"] = (d["kw"] >= thr) & (d[model] < thr)
    d["fp"] = (d["kw"] < thr) & (d[model] >= thr)
    d["prod_bin"] = pd.cut(d["prod"], [-np.inf, 0, 500, 1500, np.inf],
                           labels=["0", "1-500", "501-1500", ">1500"])

    def agg(key):
        g = d.groupby(key, observed=True)
        return pd.DataFrame({
            "mae": g["err"].apply(lambda e: e.abs().mean()), "bias": g["err"].mean(),
            "fn": g["fn"].sum(), "fp": g["fp"].sum(), "n": g.size(),
        }).reset_index()

    return {"by_hour": agg("hour"), "by_dow": agg("dow"), "by_prod": agg("prod_bin")}
