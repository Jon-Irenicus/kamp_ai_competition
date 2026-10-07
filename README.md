# KAMP 자원 최적화: 15분 수요전력 day-ahead 예측 파이프라인

제조 공정 상에서 발생하는 15분 간격의 최대수요전력과 생산량 등의 데이터를 이용하여 최대수요전력을 예측하는 모델을 제작한다. 
본 코드는 baseline 코드와 실제 모델 제작 및 분석 파트로 구분된다. 

## 실행

### Baseline LSTM 실행

Baseline은 `baseline_LSTM.py`, `baseline_preprocessing.ipynb`, `baseline_LSTM_prediction.ipynb`로 구성된다.
Baseline을 실행하기 전에 아래 명령으로 필요한 패키지를 설치한다.

```bash
pip install -r requirements.txt
```

그 다음 `baseline_preprocessing.ipynb`를 처음부터 끝까지 실행하면 원본 데이터를 전처리하고
`dataset/data_final.csv`로 저장한다. 

전처리가 끝나면 `baseline_LSTM_prediction.ipynb`를 처음부터 끝까지 실행한다.

이 노트북은 `baseline_LSTM.py`의 LSTM 모델과 `dataset/data_final.csv`를 불러와 예측을 수행하고 결과를 분석한 후 모델의 가중치를 저장한다.

따라서 Baseline은 별도의 실행 명령 없이 두 노트북을 순서대로 실행하면 된다.

### 본 모델 실행
```bash
pip install -r requirements.txt
# data/okm_augumented_2021.csv 위치에 원본을 둔다(경로는 --data로 변경 가능)
python run.py              # 전체 실행
python run.py --fast       # ablation·누수 시연 생략(반복 실험용)
```

분석 노트북은 `run.py` 실행 후에 연다.

```bash
pip install jupyter                  
jupyter notebook notebooks/
# 명령줄에서 실행 결과까지 저장하려면
jupyter nbconvert --to notebook --execute notebooks/01_data_diagnosis.ipynb --output-dir outputs/notebooks
jupyter nbconvert --to notebook --execute notebooks/02_results_analysis.ipynb --output-dir outputs/notebooks
jupyter nbconvert --to notebook --execute notebooks/03_model_application.ipynb --output-dir outputs/notebooks
```

## 구조

```
run.py                 단일 진입점 (7단계, 진행 로그 출력)
tune.py                GBM 하이퍼파라미터 탐색 (선택 실행, 결과는 config.py에 수동 반영)

src/config.py          모든 설정(경로, 시드, 과제 가정, 검증 구간, 하이퍼파라미터)
src/data.py            로드·정제·품질 검증·복제일 탐지
src/features.py        15분 long 변환, 예측 시점 기준 피처
src/lstm_model.py      LSTM 모델 정의
src/models.py          나이브, 랜덤포레스트, GBM
src/metrics.py         회귀 지표, 일 최대수요 지표, 피크 이벤트
src/experiment.py      walk-forward CV, 최종 테스트, ablation, 누수 시연, 오류 슬라이스
src/report.py          JSON 저장, 요약표, 그림
src/seq.py             LSTM용 데이터 처리와 모델 학습
src/viz.py             노트북 공용 그림 설정(한글 폰트 자동 탐색, 히트맵)

notebooks/
  01_data_diagnosis.ipynb    정제 근거, 변수 의미 검증, 복제일, 전력 패턴
  02_results_analysis.ipynb  모델 비교, 오류분석, 피크 위험 캘린더
  03_model_applicatoin.ipynb 전력량 예측 모델 활용 방안

baseline_LSTM.py       Baseline LSTM 모델 구현
baseline_preprocessing.ipynb
                       Baseline 실행 전 원본 데이터 전처리 및 dataset/data_final.csv 생성
baseline_LSTM_prediction.ipynb
                       Baseline LSTM 모델과 전처리 결과를 불러와 예측 및 결과 분석
baseline_lstm.pth      Baseline 코드 실행 후 저장한 모델의 가중치

```

위 구조에서 `baseline_`이 붙은 파일들은 LSTM을 이용한 Baseline 분석을 위한 파일이다.
반면 `src/`, `run.py`, `tune.py`, `notebooks/`, `outputs/`는 실제 분석 파이프라인을 구성하며,
데이터 전처리부터 여러 예측 모델의 비교·검증·오류 분석과 결과물 저장까지 수행한다.

즉, Baseline은 LSTM 단일 모델의 예측 결과를 확인하는 별도 구성이고,
나머지 파일들은 나이브 모델, 랜덤포레스트, GBM, 분위수 GBM 등 여러 모델을 비교 분석하는 구조이다.

노트북 규칙: (1) 로직은 `src/`에서 불러오기만 하고 노트북에서 새로 만들지 않는다. 좋은 아이디어는 `src/`로 옮겨 `run.py`에 연결한다.
(2) 수정 후 항상 "커널 재시작 → 전체 실행"이 통과해야 한다. (3) 첫 셀의 `%autoreload 2`로 `src/` 수정이 바로 반영된다.
02 노트북은 최종 모델을 다시 학습해 `run.py`의 예측과 일치하는지 확인한다(재현성 점검).

## 하이퍼파라미터 탐색 (`tune.py`)

```bash
python tune.py --trials 60                 # Optuna가 있으면 TPE + 가지치기, 없으면 랜덤 탐색
python tune.py --trials 2 --seed-check 0 --max-trees 300   # 동작 확인용
```

- 'GBM' 의 최적 탐색 조건을 찾기 위해 하이퍼 파라미터를 탐색한다.
- `run.py`와 같은 주간 CV 폴드·채점 기준(채점일 가중 MAE)만 쓰고 테스트는 보지 않는다. 0번 시도는 현재 config 값(비교 기준)이다.
- 트리 개수는 탐색하지 않고, `learning_rate=0.05`로 최대 2,000개를 한 번 학습한 뒤 체크포인트별 예측으로 7개 폴드 공통 최적값을 고른다.
- 추천값은 1-표준오차 규칙으로 고른다(최고 점수에서 표준오차 이내 후보 중 가장 단순한 것). 기준보다 나은 폴드 수와 시드별 재평가도 함께 보고한다.
- 결과: `outputs/tuning_trials.csv`(모든 시도), `outputs/tuning_summary.md`(추천값과 config.py에 붙일 코드).
- 추천값은 자동 반영하지 않는다. 대부분의 폴드에서 기준보다 나을 때만 `src/config.py`에 옮기고 `python run.py`를 다시 실행한다.
  하이퍼파라미터는 백엔드(LightGBM/sklearn)마다 다르므로, 최종 제출 환경과 같은 백엔드에서 탐색한다.

## 출력물(`outputs/`)

| 파일 | 내용 |
|---|---|
| `data_quality.json`, `duplicate_groups.csv`, `clean_hourly.csv` | 정제 결과 |
| `metrics_cv.csv`, `metrics_test.csv`, `predictions_test.csv` | 지표, 예측값 |
| `ablation_cv.csv`, `leakage_demo.csv` | 설정 비교, 무작위 분할 단위별 MAE |
| `errors_by_hour/dow/prod/day.csv` | 오류 슬라이스 |
| `figures/` | 테스트 예측, 시간대별 오차, 일자별 상위 10개 적중 |
| `run_info.json` | 실행 환경, 설정 |

## 평가지표

- 회귀: MAE, RMSE, 정전 제외 MAE, 시간 단위 MAE(= kWh 오차), 일 최대수요 오차, 피크 발생 시각 적중률
- 피크 이벤트: 각 일자당 모델이 예측한 상위 10개 피크와 실제 상위 10개 피크를 비교

