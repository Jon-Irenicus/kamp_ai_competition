# 하이퍼파라미터 탐색 결과

- 백엔드: lightgbm 4.7.0, 탐색기: Optuna TPE, 완료 41회 / 가지치기 19회
- 기준(현재 config, 0번 시도): CV MAE 26.869 (트리 100개, learning_rate 0.05)
- 최고 점수: CV MAE 24.549 (시도 35)
- 추천(1-표준오차 규칙, 폴드별 짝지은 차이 기준): CV MAE 24.951 (시도 55), 기준 대비 개선 1.918, 기준보다 나은 폴드 4/7개
- 추천값을 다른 시드 3개로 재평가: CV MAE 평균 25.765, 표준편차 0.398

**채택**: 아래 설정을 config.py에 반영 후 run.py 재실행

채택 조건: CV MAE 개선, 과반 폴드에서 우세, 개선 폭 > 시드 간 표준편차

추천 후보 설정:

```python
LGB_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.05,
    subsample_freq=1,
    num_leaves=15,
    min_child_samples=218,
    subsample=0.6931,
    colsample_bytree=0.7319,
    reg_lambda=0.003,
    min_split_gain=0.2881,
)
```

폴드별 MAE (기준 → 추천): 16.31→14.24, 15.18→15.85, 15.60→12.40, 13.71→15.82, 54.67→47.93, 61.76→56.71, 8.98→10.40