# 15분 수요전력 day-ahead 예측

D일 00:00 시점에 D일의 15분 단위 수요전력 96개를 예측하고, 일자별 피크 구간 예측 성능을 평가한다.

## 실행

```bash
pip install -r requirements.txt
# data/okm_augumented_2021.csv 위치에 원본 데이터 배치
python run.py                    # 전체 실행
python run.py --fast             # ablation, 분할 비교 생략
python run.py --skip-lstm        # LSTM 생략(torch 미설치 시 자동 생략)
python tests/test_metrics.py     # 피크 지표 검증
python tests/test_sequences.py   # 시퀀스 구성·시간 표기 검증
python tune.py --trials 60       # GBM 하이퍼파라미터 탐색(선택)
```

분석 노트북은 `run.py` 실행 후 `notebooks/`에서 실행한다.

## 구조

```
run.py                       전체 파이프라인
tune.py                      GBM 하이퍼파라미터 탐색
src/config.py                설정
src/data.py                  로드, 정합성 검증, 정제, 복제일 탐지
src/features.py              15분 단위 변환, 피처
src/models.py                기준 모델, 랜덤포레스트, GBM, 모델 목록
src/seq.py                   LSTM 입력 구성, 학습, 예측
src/lstm_model.py            Seq2Seq LSTM
src/metrics.py               평가지표
src/experiment.py            교차검증, 테스트, ablation, 오류 슬라이스
src/report.py                results.md, 그림
src/viz.py                   노트북 그림 설정
notebooks/01_data_diagnosis.ipynb   데이터 이해·진단
notebooks/02_results_analysis.ipynb 결과·오류·영향 변수 분석
tests/                       검증 스크립트
```

## 데이터 처리

| 항목 | 근거 | 처리 |
|---|---|---|
| 7/13, 7/15 `시간` 오류(전력값 기록) | 날짜별 24행 유지 | 날짜 내 행 순서로 복원 |
| 같은 날의 `인건비` | 나머지 행에서 09~17시 1.0, 그 외 1.5 규칙 성립 | 복원된 시간으로 재계산 |
| `공장인원` | 생산량 ÷ (15분+30분+45분+60분)과 일치 | 제외(타깃 누수) |
| `인건비`, `전기요금(계절)` | 시간대·월로 결정 | 피처 제외, 비용 계산용 |
| 기상 결측 4개 | | 시간순 선형보간 |
| 전력 0 구간 74개(8/28~29, 9/8) | 정전 추정 | 별도 처리 없음 |
| 증강 복제일 160일 | 하루 96개 값 해시 일치, 1~6월 집중 | 검증 시 복제그룹 제거, 원본일만 채점 |

시각은 구간 시작 기준이다(`15분` 컬럼 = 00:00~00:15 → 00:00). 하루는 00:00~23:45의 96구간이다.

## 사용 정보

예측 시점에 확정된 정보(달력, 과거 전력)만 기본으로 사용한다.

| 설정 | 기본값 | 내용 |
|---|---|---|
| `PRODUCTION_PLAN` | `"none"` | 생산계획(`hourly` / `daily` / `none`) |
| `USE_WEATHER` | `False` | 기상 |
| `USE_PLANNED_SHUTDOWN` | `False` | 계획 휴무일 |
| `USE_MONTH` | `False` | 월 피처(학습 기간 8월 휴가 주간과 혼동) |

## 입력 범위(1d / 7d)

학습 데이터는 모든 모델이 예측일 이전 전체이며, 예측 시 입력으로 사용하는 과거 범위를 두 가지로 비교한다.

| 계열 | 1d | 7d |
|---|---|---|
| 기준 | `naive_yesterday`(1일 전 동일 구간) | `naive_lastweek`(7일 전 동일 구간) |
| GBM | `gbm_1d`(1일 전 래그·전일 통계) | `gbm`(+ 7일 전 래그·지난주 평균) |
| LSTM | `lstm_1d`(인코더 96구간, 디코더 1일 전 값) | `lstm_7d`(인코더 672구간, 디코더 1일·7일 전 값) |

## LSTM

| 입력 | 내용 |
|---|---|
| 인코더 | 직전 window개 구간의 [전력 + 달력·상태] |
| 디코더 | 예측 96구간의 달력·상태(시간 sin/cos, 요일 one-hot, 공휴일·휴무·기동·점심) + 같은 구간의 과거 전력 |
| 타깃 | 예측 96구간의 전력 |

- 학습: L1 손실, AdamW(lr 2e-4, weight decay 1e-5), ReduceLROnPlateau(factor 0.5, patience 10), batch 64, 최대 150 epoch, 조기 종료 patience 20, gradient clipping 1.0.
- 학습 샘플은 4구간(1시간) 간격으로 시작하며 예측은 00:00 시작이다. 표준화 기준은 폴드별 학습 행으로 계산한다.
- 조기 종료 모니터는 학습 구간 마지막 7일이며, 모니터와 같은 복제그룹을 타깃으로 하는 학습 샘플은 제외한다.
- GPU 사용을 권장한다(CUDA 사용 시 실행 간 미세한 차이가 있을 수 있다).

## 검증

- 모델 선정: 2021-07-05부터 7일 단위 walk-forward 7개 폴드. 각 폴드는 검증 주 이전 전체 구간으로 학습한다.
  점예측 모델 중 CV MAE(채점일 가중 평균)가 최소인 모델을 선정한다.
- 최종 테스트: 2021-08-23 ~ 2021-09-14. 선정·튜닝에 사용하지 않는다.
- 1~6월은 증강 복제일이 대부분이므로 원본이 연속되는 7월 이후를 검증 구간으로 사용한다.

## 평가지표

| 지표 | 정의 |
|---|---|
| `mae`, `rmse` | 15분 구간 예측 오차 |
| `nmae` | MAE ÷ 실제 평균 수요 |
| `bias` | 평균(예측 − 실제) |
| `daily_max_mae`, `daily_max_bias` | 일 최대 15분 수요의 오차(절댓값 평균, 부호 평균) |
| `top10_hit` | 일자별 예측 상위 10개 구간 중 실제 상위 10개에 포함된 비율의 평균. 순위 무관, 10번째 실제값과 동률인 구간은 모두 실제 상위로 인정 |
| `top10_mae` | 일자별 실제·예측 상위 10개 값을 각각 내림차순 정렬해 순위끼리 비교한 MAE의 평균. 발생 시각과 무관한 피크 수준 오차 |

피크 개수는 `config.PEAK_TOP_K`로 변경할 수 있다.

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

## 하이퍼파라미터 탐색

`tune.py`는 교차검증 폴드로만 탐색하며, 결과(`outputs/tuning_summary.md`)의 채택 조건(CV MAE 개선, 과반 폴드 우세, 개선 폭 > 시드 간 표준편차)을 만족할 때 `src/config.py`에 수동으로 반영한다.
