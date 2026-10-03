"""실험 설정."""
from pathlib import Path

SEED = 42

# ── 데이터 ─────────────────────────────────────────────────────────────
DATA_PATH = Path("dataset/okm_augumented_2021.csv")
OUTPUT_DIR = Path("outputs")
ENCODING = "utf-8-sig"

# 15분·30분·45분·60분: 각 15분 구간의 평균 수요전력(kW). 시각은 구간 시작 기준(00:00 = 00:00~00:15).
POWER_COLS = ["15분", "30분", "45분", "60분"]
WEATHER_COLS = ["기온", "습도", "풍속", "강수량"]
LEAKAGE_COLS = ["공장인원"]                     # 생산량 / sum(POWER_COLS)로 계산된 값 → 타깃 누수
COST_ONLY_COLS = ["인건비", "전기요금(계절)"]    # 시간대·월로 결정되는 값 → 비용 계산에만 사용

HOLIDAYS = [
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-08-16",
]
PLANNED_SHUTDOWNS: list[str] = []   # 사전에 확정된 휴무일만 입력

# ── 예측 과제 ──────────────────────────────────────────────────────────
# D일 00:00 시점에 D일 96개 15분 구간을 예측(day-ahead).
# 예측 시점에 확정되지 않는 정보는 기본적으로 사용하지 않는다.
PRODUCTION_PLAN = "none"     # 생산계획: "hourly" | "daily" | "none"
USE_WEATHER = False          # 기상(실측값을 예보로 가정)
USE_PLANNED_SHUTDOWN = False
USE_MONTH = False            # 학습 기간의 8월 휴가 주간이 월 효과와 혼동되어 제외

# ── LSTM ───────────────────────────────────────────────────────────────
# window: 인코더 입력 길이(15분 구간 수)
# decoder_lags: 디코더에 추가하는 예측 구간의 과거 전력. 입력 범위(1d/7d)에 맞춰 지정.
LSTM_MODELS = {
    "lstm_1d": dict(window=96, decoder_lags=["lag_1d"]),
    "lstm_7d": dict(window=96 * 7, decoder_lags=["lag_1d", "lag_7d"]),
}
RUN_LSTM = True
LSTM_PARAMS = dict(hidden_size=128, num_layers=2, dropout=0.2)
LSTM_TRAIN = dict(
    epochs=150, batch_size=64,
    lr=2e-4, weight_decay=1e-5,                    # AdamW
    scheduler_factor=0.5, scheduler_patience=10,   # ReduceLROnPlateau
    patience=20,       # 조기 종료
    monitor_days=7,    # 학습 구간 마지막 N일을 조기 종료 모니터로 사용
    stride=4,          # 학습 샘플 시작 간격(15분 구간 수)
)

# ── 검증 ───────────────────────────────────────────────────────────────
# 1~6월은 증강 복제일이 대부분이므로, 원본이 연속되는 7월 이후 구간에서 주 단위 walk-forward로 모델을 선정한다.
CV_START = "2021-07-05"
CV_FOLD_DAYS = 7
TEST_START = "2021-08-23"        # 최종 평가 전용
PURGE_DUPLICATE_GROUPS = True    # 검증일과 같은 복제그룹의 학습일 제거
CV_SCORE_ORIGINAL_ONLY = True    # 검증 구간에서 복제본이 아닌 날만 채점
COPY_POLICY = "keep"             # 학습 시 복제일 처리: "keep" | "drop" | "weight"

# ── 피크 평가 ──────────────────────────────────────────────────────────
PEAK_TOP_K = 10        # 일자별 상위 K개 15분 구간을 피크로 정의
RISK_QUANTILE = 0.80   # 분위수 회귀 모델의 분위

# ── 모델 하이퍼파라미터 ────────────────────────────────────────────────
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5)
LGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
)
HGB_PARAMS = dict(   # LightGBM 미설치 시 대체 구현
    max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=False,
)

RUN_LEAKAGE_DEMO = True
