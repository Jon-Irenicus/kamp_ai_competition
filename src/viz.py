"""노트북 공용 그림 설정과 도우미. 분석 로직은 두지 않는다."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

# Windows, Mac, Linux 순서. Noto Sans CJK는 폰트 파일 이름이 JP로 등록돼도 한글 글리프를 포함한다.
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]
DOW_KR = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}


def setup_plot_style() -> str | None:
    """설치된 한글 폰트를 찾아 적용한다(Windows·Mac·Linux 순서로 탐색)."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in available), None)
    if font:
        plt.rcParams["font.family"] = font
    else:
        print("한글 폰트를 찾지 못했습니다. 그림의 한글이 깨지면 나눔고딕 등을 설치하세요.")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.figsize"] = (10, 4)
    plt.rcParams["figure.dpi"] = 110
    return font


def heatmap(df: pd.DataFrame, ax=None, cmap: str = "viridis", annotate: bool = False,
            fmt: str = "{:.0f}", cbar_label: str | None = None, fontsize: int = 7):
    """DataFrame을 그대로 히트맵으로 그린다(행 = 세로축, 열 = 가로축)."""
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
