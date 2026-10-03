"""피크 지표(top-K) 검증.

실행: python tests/test_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.metrics import topk_scores


def day(values):
    a = np.zeros(96)
    a[:len(values)] = values
    return a[None, :]


def main():
    # 완전 일치: 적중 3, MAE 0
    a = day([5, 9, 8, 7, 1])
    h, m = topk_scores(a, a.copy(), k=3)
    assert h[0] == 3 and m[0] == 0

    # 시각이 하나 어긋남: 적중 2, 값은 정렬 후 비교
    p = day([7, 9, 8, 0, 1])
    h, m = topk_scores(a, p, k=3)
    assert h[0] == 2 and np.isclose(m[0], 0.0)

    # 실제값 동률: 3위 값(9)과 같은 구간은 모두 실제 피크로 인정
    a = day([5, 9, 9, 9, 9, 1])
    p = day([0, 1, 1, 9, 8, 7])
    h, m = topk_scores(a, p, k=3)
    assert h[0] == 2 and np.isclose(m[0], (0 + 1 + 2) / 3)

    # 일자 여러 개
    A = np.vstack([day([3, 2, 1]), day([1, 2, 3])])
    P = np.vstack([day([3, 2, 1]), day([3, 2, 1])])
    h, m = topk_scores(A, P, k=1)
    assert list(h) == [1, 0] and np.allclose(m, [0, 0])

    print("OK: top-K metric checks")


if __name__ == "__main__":
    main()
