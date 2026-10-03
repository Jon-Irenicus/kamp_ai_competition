"""교차검증(주 단위 walk-forward), 최종 테스트, ablation, 분할 방식 비교, 오류 슬라이스."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config as C
from .features import feature_columns
from .metrics import HIT_COL, K, TOPK_MAE_COL, daily_matrix, evaluate, regression, topk_scores
from .models import make_gbm, make_models


# ── 공통 ───────────────────────────────────────────────────────────────
def _train_rows(tr: pd.DataFrame, policy: str) -> pd.DataFrame:
    return tr[~tr["is_copy_day"]] if policy == "drop" else tr


def _weights(tr: pd.DataFrame, policy: str):
    return (1.0 / tr["dup_group_size"]).to_numpy() if policy == "weight" else None


def fit_predict(model, tr, te, cols, policy=C.COPY_POLICY):
    tr = _train_rows(tr, policy)
    model.fit(tr[cols], tr["kw"], sample_weight=_weights(tr, policy))
    return model.predict(te[cols]), tr


def fit_predict_model(model, lookback: str, feat: pd.DataFrame, tr: pd.DataFrame, te: pd.DataFrame,
                      policy=C.COPY_POLICY):
    """행 단위 모델은 피처 행렬로, 시퀀스 모델(needs_full_frame)은 연속 프레임과 행 위치로 학습·예측한다."""
    if getattr(model, "needs_full_frame", False):
        used = _train_rows(tr, policy)
        model.fit_frame(feat, used.index)
        return model.predict_frame(feat, te.index), used
    return fit_predict(model, tr, te, feature_columns(lookback=lookback), policy)


def wavg(df: pd.DataFrame, col: str, w: str = "n_scored_days") -> float:
    """폴드별 지표의 채점일 수 가중평균(NaN 제외)."""
    m = df[col].notna() & (df[w] > 0)
    return float(np.average(df.loc[m, col], weights=df.loc[m, w])) if m.any() else np.nan


# ── 분할 ───────────────────────────────────────────────────────────────
def cv_folds(feat: pd.DataFrame):
    """(폴드명, 학습, 검증, 채점 마스크, 제거된 학습일 수)를 주 단위로 생성한다."""
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
def run_cv(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, tr, val, score, purged in cv_folds(feat):
        print(f"      폴드 {fold}", flush=True)
        for name, (kind, model, lookback) in make_models().items():
            p, used = fit_predict_model(model, lookback, feat, tr, val)
            r = evaluate(val[score], p[score])
            r.update(fold=fold, model=name, kind=kind, lookback=lookback, n_train_days=int(used["date"].nunique()),
                     n_purged_days=purged, n_scored_days=int(val.loc[score, "date"].nunique()))
            rows.append(r)
    return pd.DataFrame(rows)


def select_model(cv: pd.DataFrame) -> str:
    """점예측 모델 중 CV MAE(채점일 가중) 최소. 테스트 구간은 선정에 사용하지 않는다."""
    d = cv[cv["kind"] == "point"]
    s = pd.Series({m: wavg(g, "mae") for m, g in d.groupby("model")})
    return str(s.idxmin())


def run_test(feat: pd.DataFrame):
    """테스트 시작 전 전체 구간으로 재학습한 뒤 테스트 구간을 day-ahead로 예측한다."""
    tr, te = train_test(feat)
    rows, preds = [], te[["ts", "date", "hour", "slot", "dow", "kw", "prod"]].copy()
    for name, (kind, model, lookback) in make_models().items():
        p, _ = fit_predict_model(model, lookback, feat, tr, te)
        preds[name] = p
        r = evaluate(te, p)
        r.update(model=name, kind=kind, lookback=lookback)
        rows.append(r)
    return pd.DataFrame(rows), preds


def run_ablations(feat: pd.DataFrame) -> pd.DataFrame:
    """GBM 기준으로 설정을 하나씩 바꿨을 때의 CV 성능."""
    variants = [
        ("baseline (config)", C.COPY_POLICY, C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("copy days dropped", "drop", C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("copy days weighted 1/n", "weight", C.PRODUCTION_PLAN, C.USE_WEATHER),
        ("production plan: daily total", C.COPY_POLICY, "daily", C.USE_WEATHER),
        ("production plan: hourly", C.COPY_POLICY, "hourly", C.USE_WEATHER),
        ("production plan: none", C.COPY_POLICY, "none", C.USE_WEATHER),
        ("weather: on", C.COPY_POLICY, C.PRODUCTION_PLAN, True),
        ("weather: off", C.COPY_POLICY, C.PRODUCTION_PLAN, False),
        ("hourly plan + weather", C.COPY_POLICY, "hourly", True),
    ]
    folds = list(cv_folds(feat))
    rows, seen = [], set()
    for name, policy, prod, weather in variants:
        if (policy, prod, weather) in seen:
            continue
        seen.add((policy, prod, weather))
        cols = feature_columns(prod, weather)
        fr = []
        for _, tr, val, score, _ in folds:
            p, _ = fit_predict(make_gbm(), tr, val, cols, policy)
            r = evaluate(val[score], p[score])
            r["n_scored_days"] = int(val.loc[score, "date"].nunique())
            fr.append(r)
        fr = pd.DataFrame(fr)
        rows.append({"variant": name, **{c: wavg(fr, c) for c in
                     ["mae", "nmae", "daily_max_mae", HIT_COL, TOPK_MAE_COL]}})
    return pd.DataFrame(rows)


def run_leakage_demo(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """동일 GBM을 무작위 분할 단위(행·일·복제그룹)만 바꿔 평가한다."""
    d = feat.reset_index(drop=True)
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


def error_slices(preds: pd.DataFrame, model: str) -> dict[str, pd.DataFrame]:
    """시간대·요일·생산량 구간·일자별 오차와 피크 적중."""
    d = preds.copy()
    d["err"] = d[model] - d["kw"]
    d["prod_bin"] = pd.cut(d["prod"], [-np.inf, 0, 500, 1500, np.inf], labels=["0", "1-500", "501-1500", ">1500"])

    a = daily_matrix(d, "kw")
    p = daily_matrix(d, model)
    hits, tmae = topk_scores(a.to_numpy(), p.to_numpy())
    by_day = pd.DataFrame({"date": a.index, "dow": d.groupby("date")["dow"].first().reindex(a.index).to_numpy(),
                           "mae": d.groupby("date")["err"].apply(lambda e: e.abs().mean()).reindex(a.index).to_numpy(),
                           f"top{K}_hits": hits, TOPK_MAE_COL: tmae,
                           "daily_max_err": p.max(axis=1).to_numpy() - a.max(axis=1).to_numpy()})

    def agg(key):
        g = d.groupby(key, observed=True)
        return pd.DataFrame({"mae": g["err"].apply(lambda e: e.abs().mean()), "bias": g["err"].mean(),
                             "n": g.size()}).reset_index()

    by_dow = agg("dow").merge(by_day.groupby("dow")[[f"top{K}_hits", TOPK_MAE_COL]].mean().reset_index(), on="dow")
    return {"by_hour": agg("hour"), "by_dow": by_dow, "by_prod": agg("prod_bin"), "by_day": by_day}
