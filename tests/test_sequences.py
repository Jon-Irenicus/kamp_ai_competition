"""시퀀스 구성, 시간 표기, 피처 설정 검증 (torch 불필요).

학습 없이 입력의 특정 위치를 그대로 출력하는 모델로 구간 정렬을 확인한다.
  - 1일 입력의 마지막 날 = lag_1d, 7일 입력의 첫날 = lag_7d
  - 미래 입력의 slot_sin, decoder_lags = 예측 구간의 값
실행: python tests/test_sequences.py
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
    def __init__(self, window, offset):
        super().__init__(window, f"echo_past_{offset}")
        self.offset = offset

    def _train(self, *args):
        self.trained_with_ = args[3], args[4]

    def _predict_batch(self, X, Fu):
        kw = X[:, self.offset:self.offset + HORIZON, 0] * self.p_std_["kw"] + self.p_mean_["kw"]
        return (kw - self.y_mean_) / self.y_std_


class EchoFuture(SequenceForecaster):
    def _train(self, *args):
        pass

    def _predict_batch(self, X, Fu):
        j = nn_future_columns().index("slot_sin")
        v = Fu[:, :, j] * self.f_std_["slot_sin"] + self.f_mean_["slot_sin"]
        return (v - self.y_mean_) / self.y_std_


class EchoDecoderLag(SequenceForecaster):
    def __init__(self, window, decoder_lags, pick):
        super().__init__(window, f"echo_lag_{pick}", decoder_lags)
        self.pick = pick

    def _train(self, *args):
        pass

    def _predict_batch(self, X, Fu):
        return Fu[:, :, len(nn_future_columns()) + self.decoder_lags.index(self.pick)]


def check_time_labels(feat):
    """구간 시작 기준: 일자마다 slot 0(00:00)~95(23:45), 요일·공휴일은 하루 동안 일정."""
    g = feat.groupby("date")
    assert (g.size() == HORIZON).all()
    assert (g["slot"].apply(lambda x: (x.to_numpy() == np.arange(HORIZON)).all())).all()
    assert (feat["ts"] == feat["date"] + pd.to_timedelta(feat["slot"] * 15, unit="min")).all()
    assert (feat["dow"] == feat["date"].dt.dayofweek + 1).all()
    for col in ["dow", "is_holiday", "is_off_day", "prev_is_off_day"]:
        assert (g[col].nunique() == 1).all(), col
    onehot = feat[DOW_ONEHOT_COLS].to_numpy()
    assert (onehot.sum(axis=1) == 1).all() and (onehot.argmax(axis=1) + 1 == feat["dow"].to_numpy()).all()


def main():
    data = Path(__file__).resolve().parents[1] / C.DATA_PATH
    feat = build_features(to_long(clean(load_raw(data))[0]))
    tr, te = train_test(feat)
    fit_idx = _train_rows(tr, C.COPY_POLICY).index

    # 과거 입력 정렬
    m1 = EchoPast(96, 0).fit_frame(feat, fit_idx)
    assert np.allclose(m1.predict_frame(feat, te.index), te["lag_1d"], atol=1e-3)
    m7 = EchoPast(672, 0).fit_frame(feat, fit_idx)
    assert np.allclose(m7.predict_frame(feat, te.index), te["lag_7d"], atol=1e-3)
    m7b = EchoPast(672, 672 - 96).fit_frame(feat, fit_idx)
    assert np.allclose(m7b.predict_frame(feat, te.index), te["lag_1d"], atol=1e-3)

    # 미래 입력·디코더 래그 정렬
    mf = EchoFuture(96, "echo_future").fit_frame(feat, fit_idx)
    assert np.allclose(mf.predict_frame(feat, te.index), te["slot_sin"], atol=1e-4)
    for window, lags, pick in [(96, ["lag_1d"], "lag_1d"), (672, ["lag_1d", "lag_7d"], "lag_1d"),
                               (672, ["lag_1d", "lag_7d"], "lag_7d")]:
        md = EchoDecoderLag(window, lags, pick).fit_frame(feat, fit_idx)
        assert np.allclose(md.predict_frame(feat, te.index), te[pick], atol=1e-3)
    for name, cfg in C.LSTM_MODELS.items():
        assert cfg["window"] >= 672 or "lag_7d" not in cfg["decoder_lags"], name

    # 학습 샘플: 타깃이 학습 행 안, stride 간격, 모니터와 비중첩, 테스트 구간 비침범
    starts = m1.training_starts(feat, fit_idx)
    ok = np.zeros(len(feat), bool)
    ok[np.asarray(fit_idx)] = True
    assert ok[starts[:, None] + np.arange(HORIZON)].all()
    assert (starts >= m1.window).all() and (feat["slot"].to_numpy()[starts] % C.LSTM_TRAIN["stride"] == 0).all()
    tr_s, mon_s = m1.trained_with_
    assert tr_s.max() + HORIZON <= mon_s.min()
    assert feat.loc[starts.max() + HORIZON - 1, "ts"] < feat.loc[te.index[0], "ts"]
    assert np.isclose(m1.y_mean_, feat.loc[fit_idx, "kw"].mean())

    # 실험 경로 + 모니터 복제그룹 제거(첫 폴드)
    fold, ftr, fval, _, _ = next(cv_folds(feat))
    echo = EchoPast(96, 0)
    p, _ = fit_predict_model(echo, "1d", feat, ftr, fval)
    assert np.allclose(p, fval["lag_1d"], atol=1e-3)
    f_tr, f_mon = echo.trained_with_
    h = np.arange(HORIZON)
    groups = feat["dup_group"].to_numpy()
    assert len(np.intersect1d(groups[(f_tr[:, None] + h).ravel()], groups[(f_mon[:, None] + h).ravel()])) == 0

    # 시간 표기, 피처 설정
    check_time_labels(feat)
    if not C.USE_MONTH:
        assert "month" not in feature_columns() and "month_sin" not in nn_future_columns()

    print(f"OK: train samples {len(starts):,} (stride {C.LSTM_TRAIN['stride']}), monitor {len(mon_s):,}, "
          f"test days {len(te) // HORIZON}, fold {fold} purged {echo.monitor_purged_}")


if __name__ == "__main__":
    main()
