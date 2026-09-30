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
# 공장 휴무 계획(여름휴가 등). "사전에 아는 계획"일 때만 채운다.
# 데이터를 보고 전력이 낮은 날을 골라 채우면 누수가 된다.
PLANNED_SHUTDOWNS: list[str] = []

# ── 예측 과제 정의 ─────────────────────────────────────────────────────
# Day-ahead: D일 00:00 시점에, D-1일까지의 실측 전력으로 D일 96개 15분 슬롯을 예측.
# D일 생산량을 생산계획으로 미리 안다는 가정의 수준. 성능에 가장 큰 영향을 주는 가정이므로 보고서에 명시.
#   "hourly": 시간별 생산계획을 안다(낙관적) | "daily": 일 생산계획 총량만 안다 | "none": 모른다
PRODUCTION_PLAN = "hourly"
USE_WEATHER = True           # 기상예보 ≈ 실측 가정 (실제 운영 시 예보 오차만큼 성능 하락)

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
