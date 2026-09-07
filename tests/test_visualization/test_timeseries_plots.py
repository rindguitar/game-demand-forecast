"""
時系列の可視化モジュールのテスト

図の見た目はテストできないので、落ちずにファイルが作られることと、
配色の決定（オレンジ ↔ アクア）が守られていることだけを確認する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import pytest
from src.visualization.timeseries_plots import (
    HIGH_COLOR,
    LOW_COLOR,
    plot_overview,
    plot_positive_rate_grid,
    plot_series_grid,
)


@pytest.fixture
def series():
    """2単位 × 10週の最小の系列"""
    weeks = pd.date_range('2024-01-01', periods=10, freq='W-MON')
    rows = []
    for unit, kw in [(1, 'cards, decks'), (2, 'hunting, animals')]:
        for i, w in enumerate(weeks):
            rate = 0.5 + i * 0.04
            rows.append({'week': w, 'unit': unit, 'keywords': kw,
                         'count': 10 + i, 'share': 0.1, 'positive_rate': rate,
                         'expected_positive_rate': 0.6,
                         'positive_rate_gap': rate - 0.6})
    return pd.DataFrame(rows)


def test_plot_series_grid_creates_file(series, tmp_path):
    """系列ごとの折れ線が書き出される"""
    path = str(tmp_path / 'grid.png')
    assert os.path.getsize(plot_series_grid(series, 'count', 'title', path)) > 0


def test_plot_positive_rate_grid_creates_file(series, tmp_path):
    """充足度の図が書き出される"""
    path = str(tmp_path / 'rate.png')
    assert os.path.getsize(plot_positive_rate_grid(series, 'title', path)) > 0


def test_plot_overview_creates_file(series, tmp_path):
    """重ね描きの図が書き出される"""
    path = str(tmp_path / 'overview.png')
    assert os.path.getsize(plot_overview(series, 'count', 'title', path)) > 0


def test_satisfaction_colors_are_orange_and_aqua():
    """充足度の配色は赤↔緑ではない（色覚多様性で識別できないため却下されている）

    docs/decisions.md 2026-08-18。
    """
    for color in (LOW_COLOR, HIGH_COLOR):
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 100 and b < 100), '赤系は使わない'
        assert not (g > 150 and r < 100 and b < 100), '緑系は使わない'
