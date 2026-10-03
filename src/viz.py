"""노트북 그림 설정."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Noto Sans CJK JP에도 한글 글리프가 포함되어 있다.
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]
DOW_KR = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}


def setup_plot_style() -> str | None:
    """설치된 한글 폰트를 적용한다."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in available), None)
    if font:
        plt.rcParams["font.family"] = font
    else:
        print("한글 폰트 없음: 그림의 한글이 표시되지 않을 수 있음")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (10, 4)
    plt.rcParams["figure.dpi"] = 110
    return font


def heatmap(df: pd.DataFrame, ax=None, cmap: str = "viridis", annotate: bool = False,
            fmt: str = "{:.0f}", cbar_label: str | None = None, fontsize: int = 7):
    """DataFrame 히트맵(행: 세로축, 열: 가로축)."""
    ax = ax or plt.gca()
    values = df.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(df.shape[1]), [str(c) for c in df.columns])
    ax.set_yticks(range(df.shape[0]), [str(i) for i in df.index])
    if annotate:
        mid = np.nanmean(values)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                            color="white" if v < mid else "black")
    cb = plt.colorbar(im, ax=ax)
    if cbar_label:
        cb.set_label(cbar_label)
    return ax
