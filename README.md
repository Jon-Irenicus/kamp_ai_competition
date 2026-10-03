# 15분 수요전력 day-ahead 예측

원본 CSV에서 정제, 15분 단위 피처, 교차검증, 최종 테스트, 결과물 생성까지 명령 하나로 실행한다.

## 실행

```bash
pip install -r requirements.txt
# data/okm_augumented_2021.csv 위치에 원본을 둔다(경로는 --data로 변경 가능)
python run.py              # 전체 실행
python run.py --fast       # ablation·누수 시연 생략(반복 실험용)
```

분석 노트북은 `run.py` 실행 후에 연다.

```bash
pip install jupyter                      # 선택: shap도 설치하면 02 노트북에 SHAP 그림이 추가됨
jupyter notebook notebooks/
# 명령줄에서 실행 결과까지 저장하려면
jupyter nbconvert --to notebook --execute notebooks/01_data_diagnosis.ipynb --output-dir outputs/notebooks
jupyter nbconvert --to notebook --execute notebooks/02_results_analysis.ipynb --output-dir outputs/notebooks
```

CPU 1코어 기준 전체 약 6~7분, `--fast` 약 4분. LightGBM을 설치하면 자동으로 GBM 백엔드로 쓰이고 더 빨라진다.
정제는 전부 코드로 수행하며 원본 파일은 수정하지 않는다.

## 구조

```
run.py                 단일 진입점 (7단계, 진행 로그 출력)
tune.py                GBM 하이퍼파라미터 탐색 (선택 실행, 결과는 config.py에 수동 반영)

src/config.py          모든 설정(경로, 시드, 과제 가정, 검증 구간, 하이퍼파라미터)
src/data.py            로드·정제·품질 검증·복제일 탐지
src/features.py        15분 long 변환, 예측 시점 기준 피처
src/lstm_model.py      LSTM 모델 정의
src/models.py          나이브, 프로파일 평균, 랜덤포레스트, GBM, 분위수 GBM
src/metrics.py         회귀 지표, 일 최대수요 지표, 피크 이벤트 F1·FN·FP
src/experiment.py      walk-forward CV, 최종 테스트, ablation, 누수 시연, 오류 슬라이스
src/report.py          results.md, CSV, 그림
src/seq.py             LSTM용 데이터 처리와 모델 학습
src/viz.py             노트북 공용 그림 설정(한글 폰트 자동 탐색, 히트맵)

notebooks/
  01_data_diagnosis.ipynb    평가항목 1: 정제 근거, 변수 의미 검증, 복제일, 전력 패턴
  02_results_analysis.ipynb  평가항목 2·3·4: 모델 비교, 오류분석, 영향변수·상호작용, 피크 위험 캘린더
```

## 정제 규칙

| 문제 | 검증 근거 | 처리 |
|---|---|---|
| 7/13, 7/15의 `시간` 컬럼에 전력값(70~188) | 날짜별 정확히 24행, 행 순서 복원 후 결측·중복 시각 0 | 행 순서로 시간 복원 |
| 같은 두 날의 `인건비`가 전 시간 1.5 | 나머지 모든 행에서 "09~17시 1.0, 그 외 1.5" 규칙이 예외 없이 성립 | 복원된 시간으로 재계산 |
| `공장인원` | 생산량 ÷ (15분+30분+45분+60분)과 오차 1e-8 이내로 일치, 결측 17개는 0/0 | 타깃 누수로 제외 |
| `인건비`, `전기요금(계절)` | 각각 시간대·월로 완전히 결정되는 상수 | 예측 피처 제외, 비용 시뮬레이션용 보존 |
| 풍속 3, 강수량 1 결측 | | 시간순 선형보간 |
| 증강 복제일 | 96포인트 프로파일 해시가 동일한 날 160일(1~6월 집중, 8~9월 없음) | 그룹 ID 부여 → 검증 시 purge, 채점은 원본일만 |

## 과제 정의와 가정

- **Day-ahead**: D일 00:00 시점에 D-1일까지의 실측 전력으로 D일 96개 15분 슬롯을 예측한다.
- **기상**: 예보가 실측과 같다고 가정한다(실제 운영에서는 예보 오차만큼 성능이 떨어진다).

## 검증 설계

연속된 원본 데이터는 7월 이후에만 있다. 4~6월을 검증에 쓰면 채점할 원본일이 거의 없고,
학습 데이터도 대부분 복제본이라 최종 운영 상황을 대표하지 못한다.
실제로 4~7월 월별 검증을 쓰면 "지난주 같은 시각" 나이브가 선정되지만, 여름휴가 주간이 있는 8월에는 가장 나쁜 모델이 된다.

- **모델 선정**: 2021-07-05부터 주 단위 walk-forward 7개 폴드. 매 폴드마다 그 이전 전체로 재학습하며, 휴가 주간과 대체공휴일이 포함된다. 선정 규칙은 점예측 모델 중 CV MAE(채점일 가중) 최소로 사전에 고정했다.
- **최종 테스트**: 2021-08-23 ~ 09-14(전부 원본). 선정과 튜닝에 쓰지 않는다.
- **누수 방지**: 검증일과 같은 복제그룹의 학습일을 제거하고, 피크 임계값은 학습 구간에서만 정한다.
- 누수 시연(`leakage_demo.csv`): 같은 GBM으로 랜덤 행·랜덤 일·랜덤 복제그룹 분할의 점수를 비교한다.


## 하이퍼파라미터 탐색 (`tune.py`)

```bash
python tune.py --trials 60                 # Optuna가 있으면 TPE + 가지치기, 없으면 랜덤 탐색
python tune.py --trials 2 --seed-check 0 --max-trees 300   # 동작 확인용
```

- `run.py`와 같은 주간 CV 폴드·채점 기준(채점일 가중 MAE)만 쓰고 테스트는 보지 않는다. 0번 시도는 현재 config 값(비교 기준)이다.
- 트리 개수는 탐색하지 않고, `learning_rate=0.05`로 최대 2,000개를 한 번 학습한 뒤 체크포인트별 예측으로 7개 폴드 공통 최적값을 고른다.
- 추천값은 1-표준오차 규칙으로 고른다(최고 점수에서 표준오차 이내 후보 중 가장 단순한 것). 기준보다 나은 폴드 수와 시드별 재평가도 함께 보고한다.
- 결과: `outputs/tuning_trials.csv`(모든 시도), `outputs/tuning_summary.md`(추천값과 config.py에 붙일 코드).
- 추천값은 자동 반영하지 않는다. 대부분의 폴드에서 기준보다 나을 때만 `src/config.py`에 옮기고 `python run.py`를 다시 실행한다.
  하이퍼파라미터는 백엔드(LightGBM/sklearn)마다 다르므로, 최종 제출 환경과 같은 백엔드에서 탐색한다.

## 출력물(`outputs/`)

| 파일 | 내용 |
|---|---|
| `results.md` | 데이터 진단, CV, 테스트, 입력 범위 비교, ablation, 분할 비교, 오류 슬라이스 |
| `data_quality.json`, `duplicate_groups.csv`, `clean_hourly.csv` | 정제 결과 |
| `metrics_cv.csv`, `metrics_test.csv`, `predictions_test.csv` | 지표, 예측값 |
| `ablation_cv.csv`, `leakage_demo.csv` | 설정 비교, 무작위 분할 단위별 MAE |
| `errors_by_hour/dow/prod/day.csv` | 오류 슬라이스 |
| `figures/` | 테스트 예측, 시간대별 오차, 일자별 상위 10개 적중 |
| `run_info.json` | 실행 환경, 설정 |

## 평가지표

- 회귀: MAE, RMSE, NMAE, 정전 제외 MAE, 시간 단위 MAE(= kWh 오차), 일 최대수요 오차, 피크 발생 시각 적중률(±1시간)
- 피크 이벤트: "15분 수요 ≥ 학습기간 90분위"를 이벤트로 정의하고 Precision, Recall, F1, FN(놓친 피크), FP(헛경보)를 계산한다.
  대회 평가표의 F1·FN·FP 요구를 회귀 과제에 맞게 옮긴 것이다.

