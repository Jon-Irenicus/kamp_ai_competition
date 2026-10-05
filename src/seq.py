"""시퀀스 모델(LSTM) 학습·예측.

샘플 구성
  - 과거 입력: 예측 시작 직전 window개 구간의 [전력 + 해당 시점의 달력·상태]
  - 미래 입력: 예측할 96개 구간의 달력·상태 + decoder_lags(같은 구간의 과거 전력)
  - 타깃: 예측할 96개 구간의 전력

구현 사항
  - 시작 위치만 저장하고 배치마다 구간을 잘라낸다.
  - 학습 샘플은 stride 간격으로 시작하며, 예측은 항상 00:00에서 시작한다.
  - 표준화 기준은 학습 행으로만 계산한다. decoder_lags는 타깃과 같은 기준으로 스케일한다.
  - 조기 종료 모니터는 학습 구간 마지막 monitor_days일이며, 모니터와 같은 복제그룹을 타깃으로 하는 학습 샘플은 제외한다.

SequenceForecaster는 torch 없이 동작하는 부분(구간 구성, 스케일링, 예측 배치)이고,
LSTMForecaster가 torch 학습을 구현한다.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import config as C
from .features import SLOTS_PER_DAY, nn_future_columns, nn_past_columns

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

HORIZON = SLOTS_PER_DAY


class SequenceForecaster:
    needs_full_frame = True

    def __init__(self, window: int, name: str = "seq", decoder_lags: list[str] | None = None,
                 use_past_prod: bool | None = None):
        self.window = window
        self.name = name
        self.decoder_lags = list(decoder_lags or [])
        self.use_past_prod = C.USE_PAST_PROD if use_past_prod is None else use_past_prod

    # ── 입력 구성 ──────────────────────────────────────────────────────
    def _columns(self):
        # use_past_prod 도입 이전에 저장된 모델은 과거 생산량 없이 학습되었다.
        return nn_past_columns(getattr(self, "use_past_prod", False)), nn_future_columns()

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
        if self.decoder_lags:
            lags = ((feat[self.decoder_lags] - self.y_mean_) / self.y_std_).to_numpy(np.float32)
            F = np.hstack([F, lags])
        Y = ((feat["kw"] - self.y_mean_) / self.y_std_).to_numpy(np.float32)
        return P, F, Y

    def _batch(self, P, F, Y, starts):
        """시작 위치 i → 과거 입력 [i-window, i), 미래 입력·타깃 [i, i+96)."""
        w = np.arange(-self.window, 0)
        h = np.arange(HORIZON)
        X = P[starts[:, None] + w]
        Fu = F[starts[:, None] + h]
        y = None if Y is None else Y[starts[:, None] + h]
        return X, Fu, y

    def input_names(self) -> tuple[list[str], list[str]]:
        """(과거 입력 피처명, 미래 입력 피처명). 미래 입력은 decoder_lags를 포함한다."""
        past_cols, fut_cols = self._columns()
        return past_cols, fut_cols + self.decoder_lags

    def day_inputs(self, feat: pd.DataFrame, day_starts) -> tuple[np.ndarray, np.ndarray]:
        """00:00 시작 위치별 (과거 입력 [n, window, p], 미래 입력 [n, 96, f]). 스케일 적용 상태."""
        self._check_frame(feat)
        P, F, _ = self._arrays(feat)
        X, Fu, _ = self._batch(P, F, None, np.asarray(day_starts))
        return X, Fu

    @staticmethod
    def _check_frame(feat: pd.DataFrame):
        if not isinstance(feat.index, pd.RangeIndex) or feat.index.start != 0:
            raise ValueError("build_features가 반환한 연속 프레임(RangeIndex)이 필요합니다.")

    def training_starts(self, feat: pd.DataFrame, fit_index) -> np.ndarray:
        """타깃 96구간이 모두 학습 허용 행이고 과거 입력이 프레임 안에 있는 시작 위치(stride 간격)."""
        ok = np.zeros(len(feat), dtype=bool)
        ok[np.asarray(fit_index)] = True
        cs = np.concatenate([[0], np.cumsum(ok)])
        starts = np.arange(self.window, len(feat) - HORIZON + 1)
        starts = starts[(cs[starts + HORIZON] - cs[starts]) == HORIZON]
        slot = feat["slot"].to_numpy()
        return starts[slot[starts] % C.LSTM_TRAIN["stride"] == 0]

    def split_monitor(self, feat: pd.DataFrame, starts: np.ndarray):
        """조기 종료 모니터 분리. 학습 샘플은 모니터와 기간이 겹치거나 같은 복제그룹을 포함하면 제외한다."""
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

    # ── 학습·예측 ──────────────────────────────────────────────────────
    def fit_frame(self, feat: pd.DataFrame, fit_index):
        self._check_frame(feat)
        self._fit_scaler(feat, fit_index)
        P, F, Y = self._arrays(feat)
        tr, mon = self.split_monitor(feat, self.training_starts(feat, fit_index))
        self._train(P, F, Y, tr, mon)
        return self

    def predict_frame(self, feat: pd.DataFrame, target_index) -> np.ndarray:
        """target_index 행의 예측값을 같은 순서로 반환한다. 일자별로 00:00부터 96구간을 예측한다."""
        self._check_frame(feat)
        P, F, _ = self._arrays(feat)
        tgt = np.asarray(target_index)
        day_starts = tgt[feat["slot"].to_numpy()[tgt] == 0]
        if (day_starts < self.window).any():
            raise ValueError("과거 입력 구간이 부족한 예측일이 있습니다.")
        out = np.full(len(feat), np.nan)
        bs = C.LSTM_TRAIN["batch_size"]
        for k in range(0, len(day_starts), bs):
            s = day_starts[k:k + bs]
            X, Fu, _ = self._batch(P, F, None, s)
            out[(s[:, None] + np.arange(HORIZON)).ravel()] = self._predict_batch(X, Fu).ravel()
        pred = out[tgt] * self.y_std_ + self.y_mean_
        if np.isnan(pred).any():
            raise ValueError("예측 대상은 하루 96구간 단위여야 합니다.")
        return pred

    def _train(self, P, F, Y, tr_starts, mon_starts):
        raise NotImplementedError

    def _predict_batch(self, X, Fu) -> np.ndarray:
        raise NotImplementedError


class LSTMForecaster(SequenceForecaster):
    """Seq2SeqLSTM 학습. L1 손실, AdamW, ReduceLROnPlateau, 모니터 손실 기준 조기 종료."""

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
            raise RuntimeError(f"[{self.name}] 모니터 손실이 계산되지 않았습니다.")
        self.net_.load_state_dict(best_state)
        self.history_ = history
        print(f"      [{self.name}] samples={len(tr_starts):,} (purged {self.monitor_purged_}), "
              f"epochs={len(history)}, best_epoch={int(np.argmin(history)) + 1}, "
              f"lr={opt.param_groups[0]['lr']:.1e}, monitor_mae={best * self.y_std_:.2f} kW", flush=True)

    def __getstate__(self):
        """GPU에서 학습한 모델도 CPU 환경에서 불러올 수 있도록 CPU로 옮겨 저장한다."""
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_"] = copy.deepcopy(self.net_).cpu()
            state["device_"] = torch.device("cpu")
        return state

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
