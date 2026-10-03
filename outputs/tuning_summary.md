# 하이퍼파라미터 탐색 결과 (자동 생성)

- 백엔드: lightgbm 4.7.0, 탐색기: Optuna TPE, 완료 55회 / 가지치기 5회
- 기준(현재 config, 0번 시도): CV MAE 17.027 (트리 700개, learning_rate 0.05)
- 최고 점수: CV MAE 16.344 (시도 33)
- 추천(1-표준오차 규칙, 폴드별 짝지은 차이 기준): CV MAE 16.751 (시도 44), 기준 대비 개선 0.276, 기준보다 나은 폴드 4/7개
- 추천값을 다른 시드 3개로 재평가: CV MAE 평균 17.126, 표준편차 0.362

**채택 비권장**: 개선 폭이 시드 간 표준편차 이하. 현재 config를 유지하거나 탐색 횟수를 늘리세요.

채택 조건: 기준보다 CV MAE가 낮고, 과반의 폴드에서 낫고, 개선 폭이 시드 간 표준편차보다 클 것.

추천 후보를 `src/config.py`에 붙일 코드 (채택 권장일 때만 반영):

```python
LGB_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.05,
    subsample_freq=1,
    num_leaves=25,
    min_child_samples=33,
    subsample=0.8383,
    colsample_bytree=0.7956,
    reg_lambda=0.167,
    min_split_gain=0.1037,
)
```

폴드별 MAE (기준 → 추천): 14.57→15.17, 22.39→23.75, 14.22→13.82, 11.19→10.30, 20.34→20.16, 25.69→23.01, 9.96→10.12