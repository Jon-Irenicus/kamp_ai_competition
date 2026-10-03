"""실험 설정. 여기 값만 바꾸면 run.py 전체가 같은 순서로 다시 실행된다."""
from pathlib import Path

SEED = 42

# ── 데이터 ─────────────────────────────────────────────────────────────
DATA_PATH = Path("dataset/okm_augumented_2021.csv")
OUTPUT_DIR = Path("outputs")
ENCODING = "utf-8-sig"

# 해석 A: 각 컬럼 = 해당 15분 구간의 평균수요전력(kW).
#   15분 = hh:00~15, 30분 = hh:15~30, 45분 = hh:30~45, 60분 = hh:45~(h+1):00
#   → 시간 평균(평균 컬럼)은 숫자상 그 시간의 사용량(kWh)과 같다.
POWER_COLS = ["15분", "30분", "45분", "60분"]
WEATHER_COLS = ["기온", "습도", "풍속", "강수량"]
LEAKAGE_COLS = ["공장인원"]                    # = 생산량 / sum(POWER_COLS) → 타깃 누수, 제외
COST_ONLY_COLS = ["인건비", "전기요금(계절)"]   # 시간·월로 결정되는 상수 → 피처 제외, 비용 시뮬레이션용

# 2021 한국 공휴일(데이터 기간 내). 달력으로 미리 아는 정보.
HOLIDAYS = [
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-08-16",
]
# 공장 휴무 계획(여름휴가 등). "사전에 아는 계획"일 때만 채운다. USE_PLANNED_SHUTDOWN이 True일 때만 사용.
# 데이터를 보고 전력이 낮은 날을 골라 채우면 누수가 된다.
PLANNED_SHUTDOWNS: list[str] = []

# ── 예측 과제 정의 ─────────────────────────────────────────────────────
# Day-ahead: D일 00:00 시점에, D-1일까지의 실측 전력으로 D일 96개 15분 슬롯을 예측.
# 기본값은 "예측 시점에 확실히 알 수 있는 정보만" 쓴다(달력, 과거 전력).
# 아래 정보는 미래에 확정되지 않으므로 기본적으로 끄고, ablation에서 켰을 때의 효과만 보고한다.
#   PRODUCTION_PLAN: "hourly" 시간별 계획 | "daily" 일 총량만 | "none" 사용 안 함
#   USE_WEATHER: 기상(실측을 예보 대용으로 쓰는 가정)
#   USE_PLANNED_SHUTDOWN: 계획 휴무 달력(PLANNED_SHUTDOWNS)
PRODUCTION_PLAN = "none"
USE_WEATHER = False
USE_PLANNED_SHUTDOWN = False

# 월(month) 피처. 학습 기간 8월 평일의 절반이 휴가 주간(8/2~8/6)이라, 생산계획 없이는 월 피처가
# 휴가를 "8월의 특성"으로 학습했다. 빼면 CV MAE가 gbm 33.4→27.1, gbm_1d 25.8→21.5로 개선되어
# 사전 규칙(CV 기준 결정)에 따라 끈다. 트리 모델(month)과 LSTM(month sin/cos)에 함께 적용된다.
USE_MONTH = False

# ── 입력 과거 범위(lookback) 비교 ─────────────────────────────────────
# 같은 모델 계열을 "전날만 입력(1d)"과 "지난주까지 입력(7d)" 두 경우로 학습해 비교한다.
#   트리 모델: 1d = lag_1d + 전날 통계, 7d = 여기에 lag_7d와 지난주 평균 추가
#   LSTM: 인코더 입력 길이 = 96칸(1d) 또는 672칸(7d)
# 학습 데이터 범위는 두 경우 모두 "예측일 이전 전체"(확장 윈도우)로 같다.
# LSTM 모델별 설정. window = 인코더(과거 입력) 길이, decoder_lags = 디코더(미래 입력)에 넣는 예측 칸별 과거 전력.
# 노트북(LSTM_prediction.ipynb)처럼 디코더에 같은 칸의 과거 전력을 넣되, 과거 범위 정의를 지키도록
# lstm_1d는 1일 전 값만, lstm_7d는 1일 전·7일 전 값을 받는다(트리 gbm_1d / gbm과 같은 구분).
# 인코더에는 래그를 넣지 않는다(과거 칸의 래그는 창보다 더 먼 과거를 보게 되므로).
LSTM_MODELS = {
    "lstm_1d": dict(window=96, decoder_lags=["lag_1d"]),
    "lstm_7d": dict(window=96 * 7, decoder_lags=["lag_1d", "lag_7d"]),
}
RUN_LSTM = True                  # torch가 없거나 --skip-lstm이면 자동으로 건너뜀
LSTM_PARAMS = dict(hidden_size=128, num_layers=2, dropout=0.2)   # 첨부 Seq2SeqLSTM 기본값
# 학습 설정은 LSTM_prediction.ipynb에서 쓴 값을 따른다.
LSTM_TRAIN = dict(
    epochs=150, batch_size=64,
    lr=2e-4, weight_decay=1e-5,                    # AdamW
    scheduler_factor=0.5, scheduler_patience=10,   # ReduceLROnPlateau (모니터 손실 기준)
    patience=20,       # 모니터 손실이 이만큼 연속 개선되지 않으면 중단
    monitor_days=7,    # 학습 구간 마지막 7일을 조기 종료 모니터로 사용(검증 주와 별개, 복제그룹 제거)
    stride=4,          # 학습 샘플 시작 간격(15분 칸 수). 1이면 노트북과 같이 모든 칸에서 시작(약 4배 느림)
)

# 정전(15분 수요가 정확히 0인 칸)은 타깃으로는 그대로 두되, 래그·전날 통계를 계산할 때는
# "같은 요일·같은 15분 칸의 최근 N주 중앙값"으로 대체한다. 직전 값 유지는 정전 직전 값이 이미
# 비정상일 수 있고, 기간 평균은 요일·시간대 구조를 지워서 채택하지 않았다.
# 사후 보정(정전 후 증산 등)은 학습 기간에 정전이 없어 검증할 수 없으므로 넣지 않는다.
LAG_IMPUTE_OUTAGE = True
LAG_IMPUTE_WEEKS = 4

# ── 검증 설계 ──────────────────────────────────────────────────────────
# 연속된 원본(복제 없는) 데이터는 7월 이후에만 있다. 1~6월 대부분은 증강 복제본이라
# 그 기간을 검증에 쓰면 복제본이 래그 피처로 새어 들어가고, 학습 데이터도 거의 가짜라서
# 최종 운영 상황(원본을 포함해 학습)을 대표하지 못한다. 그래서:
#   - 모델 선정: 7/5부터 테스트 직전까지 주 단위 walk-forward (매 폴드마다 그 이전 전체로 재학습)
#     → 휴가 주간·광복절 대체공휴일 같은 불규칙 주간이 검증에 포함된다.
#   - 최종 테스트: TEST_START 이후. 선정·튜닝에 절대 쓰지 않는다.
CV_START = "2021-07-05"          # 첫 검증 주 시작(월요일)
CV_FOLD_DAYS = 7
TEST_START = "2021-08-23"        # 이후 전부 원본 → 최종 평가 전용
PURGE_DUPLICATE_GROUPS = True    # 검증일과 프로파일이 같은(복제) 학습일을 학습에서 제거
CV_SCORE_ORIGINAL_ONLY = True    # 검증 구간에서 복제본이 아닌 날(각 그룹의 첫 등장일)만 채점
COPY_POLICY = "keep"             # 학습 시 복제일 처리: "keep" | "drop" | "weight"(1/복제수)

# ── 피크 이벤트 정의 ───────────────────────────────────────────────────
PEAK_QUANTILE = 0.90   # 15분 수요전력 >= 학습기간 90분위 → 피크 이벤트
RISK_QUANTILE = 0.80   # 피크 위험용 분위수 회귀 모델의 분위

# ── 모델 하이퍼파라미터 (튜닝 전 기본값) ──────────────────────────────
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, max_features=0.5)
LGB_PARAMS = dict(
    n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
)
HGB_PARAMS = dict(   # LightGBM이 없을 때 쓰는 sklearn 대체 구현
    max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=20,
    l2_regularization=1.0, early_stopping=False,   # 내부 랜덤 검증분할 끔(시계열·재현성)
)

RUN_LEAKAGE_DEMO = True
