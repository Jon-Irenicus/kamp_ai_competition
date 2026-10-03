"""시퀀스 모델(LSTM) 래퍼.

첨부 코드의 make_sequences와 같은 구조로 데이터를 자른다.
  - 과거 입력 X: 예측 시작 직전 window칸의 [전력(정전 대체값) + 그 시점의 달력 정보]
  - 미래 입력 C: 예측할 96칸의 '미리 아는 정보'(달력 등) + 모델별 디코더 래그(같은 칸의 1일 전·7일 전 전력)
  - 타깃 y: 예측할 96칸의 실제 전력
차이점은 세 가지다.
  1) 전체 배열을 미리 만들지 않고 시작 위치(index)만 저장한 뒤 배치마다 잘라낸다(7일 창도 메모리 부담이 작음).
  2) 학습 샘플은 stride칸 간격으로 시작하지만, 평가·예측은 항상 00:00 시작(day-ahead 과제와 동일)으로 한다.
  3) 표준화 기준(평균·표준편차)은 각 폴드의 학습 행에서만 계산한다(누수 방지).
     디코더 래그는 노트북처럼 타깃과 같은 기준으로 스케일해, 참조값과 맞힐 값이 같은 단위가 되게 한다.
  4) 조기 종료 모니터(학습 구간 마지막 며칠)와 복제그룹이 같은 학습 샘플은 뺀다(CV의 purge와 같은 규칙).

이 파일의 SequenceForecaster는 torch 없이 동작하고(창 자르기·표준화·예측 되돌리기),
학습 루프만 LSTMForecaster가 torch로 구현한다. tests/test_sequences.py가 창 정렬을 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .features import SLOTS_PER_DAY, nn_future_columns

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HORIZON = SLOTS_PER_DAY   # 하루 96칸


class SequenceForecaster:
    """창 자르기·표준화·예측 배치를 담당. 학습(_train)과 배치 예측(_predict_batch)은 하위 클래스가 구현."""

    needs_full_frame = True   # 실험 코드가 행 단위 X 대신 연속된 전체 프레임을 넘기도록 알리는 표시

    def __init__(self, window: int, name: str = "seq", decoder_lags: list[str] | None = None):
        self.window = window
        self.name = name
        self.decoder_lags = list(decoder_lags or [])

    # ── 데이터 준비 ────────────────────────────────────────────────────
    def _columns(self):
        """과거 입력 = 전력(정전 대체) + 그 시점의 달력·상태, 미래 입력 = 예측 칸의 달력·상태 (+ 디코더 래그 별도)."""
        fut = nn_future_columns()
        return ["kw_lag_src"] + fut, fut

    def _fit_scaler(self, feat: pd.DataFrame, fit_index):
        past_cols, fut_cols = self._columns()
        ref = feat.loc[fit_index]
        std = lambda s: s.std().replace(0, 1.0).fillna(1.0)
        self.p_mean_, self.p_std_ = ref[past_cols].mean(), std(ref[past_cols])
        self.f_mean_, self.f_std_ = ref[fut_cols].mean(), std(ref[fut_cols])
        self.y_mean_, self.y_std_ = float(ref["kw"].mean()), float(ref["kw"].std() or 1.0)

    def _arrays(self, feat: pd.DataFrame):
        past_cols, fut_cols = self._columns()
        P = ((feat[past_cols] - self.p_mean_) / self.p_std_).to_numpy(np.float32)
        F = ((feat[fut_cols] - self.f_mean_) / self.f_std_).to_numpy(np.float32)
        if self.decoder_lags:   # 디코더 래그: 타깃과 같은 기준으로 스케일
            lags = ((feat[self.decoder_lags] - self.y_mean_) / self.y_std_).to_numpy(np.float32)
            F = np.hstack([F, lags])
        Y = ((feat["kw"] - self.y_mean_) / self.y_std_).to_numpy(np.float32)
        return P, F, Y

    def _batch(self, P, F, Y, starts):
        """시작 위치 배열 → (과거 창, 미래 정보, 타깃). 시작 위치 i의 과거 창은 i-window ~ i-1."""
        w = np.arange(-self.window, 0)
        h = np.arange(HORIZON)
        X = P[starts[:, None] + w]
        Fu = F[starts[:, None] + h]
        y = None if Y is None else Y[starts[:, None] + h]
        return X, Fu, y

    @staticmethod
    def _check_frame(feat: pd.DataFrame):
        if not isinstance(feat.index, pd.RangeIndex) or feat.index.start != 0:
            raise ValueError("시퀀스 모델에는 build_features가 반환한 연속 프레임(0부터 시작하는 RangeIndex)을 넘겨야 합니다.")

    def training_starts(self, feat: pd.DataFrame, fit_index) -> np.ndarray:
        """타깃 96칸이 모두 '학습 허용 행'이고 과거 창이 프레임 안에 있는 시작 위치(stride 간격)."""
        ok = np.zeros(len(feat), dtype=bool)
        ok[np.asarray(fit_index)] = True
        cs = np.concatenate([[0], np.cumsum(ok)])
        starts = np.arange(self.window, len(feat) - HORIZON + 1)
        starts = starts[(cs[starts + HORIZON] - cs[starts]) == HORIZON]
        slot = feat["slot"].to_numpy()
        return starts[slot[starts] % C.LSTM_TRAIN["stride"] == 0]

    def split_monitor(self, feat: pd.DataFrame, starts: np.ndarray):
        """학습 구간 마지막 monitor_days일을 조기 종료 모니터로 떼어낸다.
        학습 샘플은 (1) 타깃이 모니터와 겹치지 않게 자르고, (2) 타깃에 모니터 날짜와 같은 복제그룹이
        들어 있으면 뺀다. (2)가 없으면 복제본을 외워서 모니터 손실이 낮아지는 쪽으로 조기 종료가 왜곡된다."""
        if len(starts) == 0:
            raise ValueError("학습 샘플이 없습니다.")
        ts = feat["ts"].to_numpy()
        cutoff = ts[starts[-1]] - np.timedelta64(C.LSTM_TRAIN["monitor_days"], "D")
        mon = starts[ts[starts] >= cutoff]
        tr = starts[starts + HORIZON <= mon[0]]
        h = np.arange(HORIZON)
        groups = feat["dup_group"].to_numpy()
        mon_groups = np.unique(groups[(mon[:, None] + h).ravel()])
        clash = np.isin(groups, mon_groups)[tr[:, None] + h].any(axis=1)
        self.monitor_purged_ = int(clash.sum())
        return tr[~clash], mon

    # ── 실험 코드가 부르는 인터페이스 ──────────────────────────────────
    def fit_frame(self, feat: pd.DataFrame, fit_index):
        self._check_frame(feat)
        self._fit_scaler(feat, fit_index)
        P, F, Y = self._arrays(feat)
        tr, mon = self.split_monitor(feat, self.training_starts(feat, fit_index))
        self._train(P, F, Y, tr, mon)
        return self

    def predict_frame(self, feat: pd.DataFrame, target_index) -> np.ndarray:
        """target_index 행들의 예측을 같은 순서로 반환. 각 날은 00:00에서 시작해 96칸을 한 번에 예측한다."""
        self._check_frame(feat)
        P, F, _ = self._arrays(feat)
        tgt = np.asarray(target_index)
        day_starts = tgt[feat["slot"].to_numpy()[tgt] == 0]
        if (day_starts < self.window).any():
            raise ValueError("예측일 앞에 과거 창을 채울 데이터가 부족합니다.")
        out = np.full(len(feat), np.nan)
        bs = C.LSTM_TRAIN["batch_size"]
        for k in range(0, len(day_starts), bs):
            s = day_starts[k:k + bs]
            X, Fu, _ = self._batch(P, F, None, s)
            out[(s[:, None] + np.arange(HORIZON)).ravel()] = self._predict_batch(X, Fu).ravel()
        pred = out[tgt] * self.y_std_ + self.y_mean_
        if np.isnan(pred).any():
            raise ValueError("하루 96칸이 모두 포함되지 않은 예측 대상이 있습니다(예측은 하루 단위로만 가능).")
        return pred

    # ── 하위 클래스가 구현 ─────────────────────────────────────────────
    def _train(self, P, F, Y, tr_starts, mon_starts):
        raise NotImplementedError

    def _predict_batch(self, X, Fu) -> np.ndarray:
        raise NotImplementedError


class LSTMForecaster(SequenceForecaster):
    """첨부 Seq2SeqLSTM을 학습한다. 손실(L1)·옵티마이저(AdamW)·스케줄러·조기 종료는 노트북 설정을 따른다."""

    def _train(self, P, F, Y, tr_starts, mon_starts):
        from .lstm_model import Seq2SeqLSTM

        cfg = C.LSTM_TRAIN
        torch.manual_seed(C.SEED)
        rng = np.random.default_rng(C.SEED)
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device_.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.net_ = Seq2SeqLSTM(input_size=P.shape[1], future_size=F.shape[1], **C.LSTM_PARAMS).to(self.device_)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["scheduler_factor"],
                                                           patience=cfg["scheduler_patience"])
        loss_fn = torch.nn.L1Loss()
        to_t = lambda a: torch.from_numpy(a).to(self.device_)

        best, best_state, bad, history = np.inf, None, 0, []
        for epoch in range(1, cfg["epochs"] + 1):
            self.net_.train()
            order = rng.permutation(tr_starts)
            for k in range(0, len(order), cfg["batch_size"]):
                X, Fu, y = self._batch(P, F, Y, order[k:k + cfg["batch_size"]])
                opt.zero_grad()
                loss = loss_fn(self.net_(to_t(X), to_t(Fu)), to_t(y))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            mon = self._monitor_mae(P, F, Y, mon_starts)
            sched.step(mon)
            history.append(mon)
            if mon < best:
                best, bad = mon, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
        if best_state is None:
            raise RuntimeError(f"[{self.name}] 모니터 손실이 한 번도 계산되지 않았습니다(NaN). 입력 데이터를 확인하세요.")
        self.net_.load_state_dict(best_state)
        self.history_ = history
        print(f"      [{self.name}] 학습 샘플 {len(tr_starts):,}개(모니터와 같은 복제그룹 {self.monitor_purged_}개 제외), "
              f"{len(history)} epoch, 최종 lr {opt.param_groups[0]['lr']:.1e}, "
              f"모니터 MAE {best * self.y_std_:.2f} kW (최적 epoch {int(np.argmin(history)) + 1})", flush=True)

    def _monitor_mae(self, P, F, Y, starts) -> float:
        errs = []
        for k in range(0, len(starts), C.LSTM_TRAIN["batch_size"]):
            X, Fu, y = self._batch(P, F, Y, starts[k:k + C.LSTM_TRAIN["batch_size"]])
            errs.append(np.abs(self._predict_batch(X, Fu) - y).ravel())
        return float(np.concatenate(errs).mean())

    def _predict_batch(self, X, Fu) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            return self.net_(torch.from_numpy(X).to(self.device_),
                             torch.from_numpy(Fu).to(self.device_)).cpu().numpy()
