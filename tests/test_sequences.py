"""시퀀스 창 정렬 검증 (torch 없이 실행 가능).

학습하지 않고 과거 창의 특정 위치를 그대로 내보내는 가짜 모델로 확인한다.
  - 1일 창의 마지막 하루를 내보내면 예측 == lag_1d (어제 같은 칸)
  - 7일 창의 첫 하루를 내보내면 예측 == lag_7d (지난주 같은 칸)
  - 미래 입력의 slot_sin을 내보내면 예측 위치의 slot_sin과 일치
  - 디코더 래그 채널을 내보내면 lag_1d / lag_7d와 일치
그 밖에 학습 샘플 선택, 모니터 분리와 복제그룹 제거, 표준화 기준(학습 행만 사용),
시간 표기(구간 시작 기준 하루 경계, 요일·공휴일이 하루 96칸에 일관되게 붙는지), 피처 구성을 확인한다.

실행: 프로젝트 폴더에서  python tests/test_sequences.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src.data import clean, load_raw
from src.experiment import _train_rows, cv_folds, fit_predict_model, train_test
from src.features import DOW_ONEHOT_COLS, build_features, feature_columns, nn_future_columns, to_long
from src.seq import HORIZON, SequenceForecaster


class EchoPast(SequenceForecaster):
    """과거 창의 offset번째 칸부터 96칸의 전력을 그대로 출력(타깃 척도로 변환)."""

    def __init__(self, window, offset):
        super().__init__(window, f"echo_past_{offset}")
        self.offset = offset

    def _train(self, *args):
        self.trained_with_ = args[3], args[4]

    def _predict_batch(self, X, Fu):
        kw = X[:, self.offset:self.offset + HORIZON, 0] * self.p_std_["kw_lag_src"] + self.p_mean_["kw_lag_src"]
        return (kw - self.y_mean_) / self.y_std_


class EchoFuture(SequenceForecaster):
    """미래 입력의 slot_sin을 그대로 출력(타깃 척도로 변환)."""

    def _train(self, *args):
        pass

    def _predict_batch(self, X, Fu):
        j = nn_future_columns().index("slot_sin")
        v = Fu[:, :, j] * self.f_std_["slot_sin"] + self.f_mean_["slot_sin"]
        return (v - self.y_mean_) / self.y_std_


class EchoDecoderLag(SequenceForecaster):
    """디코더 래그 채널 하나를 그대로 출력(래그는 이미 타깃 척도로 스케일돼 있음)."""

    def __init__(self, window, decoder_lags, pick):
        super().__init__(window, f"echo_lag_{pick}", decoder_lags)
        self.pick = pick

    def _train(self, *args):
        pass

    def _predict_batch(self, X, Fu):
        j = len(nn_future_columns()) + self.decoder_lags.index(self.pick)
        return Fu[:, :, j]


def check_time_labels(feat):
    """구간 시작 기준: 날짜마다 00:00(slot 0)~23:45(slot 95)가 순서대로 있고, 요일·공휴일이 하루 내내 같다."""
    g = feat.groupby("date")
    assert (g.size() == HORIZON).all(), "하루 96칸이 아닌 날짜가 있음"
    assert (g["slot"].apply(lambda x: (x.to_numpy() == np.arange(HORIZON)).all())).all(), "slot 순서 오류"
    expected_ts = feat["date"] + pd.to_timedelta(feat["slot"] * 15, unit="min")
    assert (feat["ts"] == expected_ts).all(), "ts가 구간 시작 시각이 아님"
    assert (feat["dow"] == feat["date"].dt.dayofweek + 1).all(), "요일이 날짜와 다름"
    for col in ["dow", "is_holiday", "is_off_day", "prev_is_off_day"]:
        assert (g[col].nunique() == 1).all(), f"{col}이 하루 안에서 바뀜(날짜 경계 어긋남)"
    onehot = feat[DOW_ONEHOT_COLS].to_numpy()
    assert (onehot.sum(axis=1) == 1).all() and (onehot.argmax(axis=1) + 1 == feat["dow"].to_numpy()).all(), "요일 one-hot 오류"


def main():
    data = Path(__file__).resolve().parents[1] / C.DATA_PATH
    feat = build_features(to_long(clean(load_raw(data))[0]))
    tr, te = train_test(feat)
    fit_idx = _train_rows(tr, C.COPY_POLICY).index

    # 1) 1일 창: 마지막 하루 = lag_1d
    m1 = EchoPast(96, 0).fit_frame(feat, fit_idx)
    assert np.allclose(m1.predict_frame(feat, te.index), te["lag_1d"], atol=1e-3), "1일 창 정렬 오류"

    # 2) 7일 창: 첫 하루 = lag_7d, 마지막 하루 = lag_1d
    m7 = EchoPast(672, 0).fit_frame(feat, fit_idx)
    assert np.allclose(m7.predict_frame(feat, te.index), te["lag_7d"], atol=1e-3), "7일 창(첫 하루) 정렬 오류"
    m7b = EchoPast(672, 672 - 96).fit_frame(feat, fit_idx)
    assert np.allclose(m7b.predict_frame(feat, te.index), te["lag_1d"], atol=1e-3), "7일 창(마지막 하루) 정렬 오류"

    # 3) 미래 입력이 예측 칸과 정렬돼 있는지
    mf = EchoFuture(96, "echo_future").fit_frame(feat, fit_idx)
    assert np.allclose(mf.predict_frame(feat, te.index), te["slot_sin"], atol=1e-4), "미래 입력 정렬 오류"

    # 3-1) 디코더 래그: lstm_1d는 lag_1d, lstm_7d는 lag_1d·lag_7d가 예측 칸과 정렬돼 있는지
    for window, lags, pick in [(96, ["lag_1d"], "lag_1d"), (672, ["lag_1d", "lag_7d"], "lag_1d"),
                               (672, ["lag_1d", "lag_7d"], "lag_7d")]:
        md = EchoDecoderLag(window, lags, pick).fit_frame(feat, fit_idx)
        assert np.allclose(md.predict_frame(feat, te.index), te[pick], atol=1e-3), f"디코더 래그 정렬 오류({pick})"
    for name, cfg in C.LSTM_MODELS.items():   # 과거 범위 정의: 1일 창은 lag_7d를 받으면 안 됨
        assert cfg["window"] >= 672 or "lag_7d" not in cfg["decoder_lags"], f"{name}: 1일 창에 7일 래그"

    # 4) 학습 샘플: 타깃 96칸이 모두 학습 행, 과거 창이 프레임 안, stride 간격, 모니터와 겹치지 않음
    starts = m1.training_starts(feat, fit_idx)
    ok = np.zeros(len(feat), bool)
    ok[np.asarray(fit_idx)] = True
    assert ok[starts[:, None] + np.arange(HORIZON)].all(), "학습 타깃에 학습 외 행 포함"
    assert (starts >= m1.window).all() and (feat["slot"].to_numpy()[starts] % C.LSTM_TRAIN["stride"] == 0).all()
    tr_s, mon_s = m1.trained_with_
    assert tr_s.max() + HORIZON <= mon_s.min(), "학습 샘플과 모니터 구간이 겹침"
    assert feat.loc[starts.max() + HORIZON - 1, "ts"] < feat.loc[te.index[0], "ts"], "학습 타깃이 테스트 구간 침범"

    # 5) 표준화 기준은 학습 행으로만 계산
    assert np.isclose(m1.y_mean_, feat.loc[fit_idx, "kw"].mean()), "표준화에 학습 외 행 사용"

    # 6) 실험 코드 경로(fit_predict_model)를 통과해도 같은 결과인지, CV 첫 폴드로 확인
    fold, ftr, fval, score, _ = next(cv_folds(feat))
    echo = EchoPast(96, 0)
    p, _ = fit_predict_model(echo, "1d", feat, ftr, fval)
    assert np.allclose(p, fval["lag_1d"], atol=1e-3), "실험 경로 정렬 오류"

    # 7) 모니터 복제그룹 제거: 첫 폴드의 모니터(6/28~7/4)에는 원본이 학습 구간에 있는 복제일이 있다
    f_tr, f_mon = echo.trained_with_
    h = np.arange(HORIZON)
    groups = feat["dup_group"].to_numpy()
    shared = np.intersect1d(groups[(f_tr[:, None] + h).ravel()], groups[(f_mon[:, None] + h).ravel()])
    assert echo.monitor_purged_ > 0 and len(shared) == 0, "모니터와 같은 복제그룹이 학습 샘플에 남음"

    # 8) 시간 표기와 피처 구성
    check_time_labels(feat)
    if not C.USE_MONTH:
        assert "month" not in feature_columns() and "month_sin" not in nn_future_columns(), "USE_MONTH=False인데 월 피처 포함"

    print(f"모든 검증 통과: 학습 샘플 {len(starts):,}개(stride {C.LSTM_TRAIN['stride']}), "
          f"모니터 {len(mon_s):,}개, 테스트 {len(te) // HORIZON}일, "
          f"CV 첫 폴드 {fold}에서 모니터와 같은 복제그룹 학습 샘플 {echo.monitor_purged_}개 제외")


if __name__ == "__main__":
    main()
