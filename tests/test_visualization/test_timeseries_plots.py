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
    FORECAST_STYLES,
    HIGH_COLOR,
    LOW_COLOR,
    plot_forecast_grid,
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


@pytest.fixture
def forecasts():
    """2単位 × 10週（学習6週・テスト4週）の最小の予測表。予測の列はテスト週だけ値を持つ"""
    weeks = pd.date_range('2024-01-01', periods=10, freq='W-MON')
    rows = []
    for unit, kw in [(1, 'cards, decks'), (2, 'hunting, animals')]:
        for i, w in enumerate(weeks):
            is_test = i >= 6
            row = {'unit': unit, 'week': w, 'split': 'test' if is_test else 'train',
                   'keywords': kw, 'actual': 0.1 + 0.01 * i}
            row.update({column: 0.12 if is_test else float('nan') for column in FORECAST_STYLES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_plot_forecast_grid_creates_file(forecasts, tmp_path):
    """実績と4つの予測を重ねた図が書き出される"""
    path = str(tmp_path / 'forecast.png')
    assert os.path.getsize(plot_forecast_grid(forecasts, 'title', path)) > 0


def test_plot_forecast_grid_draws_only_given_units_and_weeks(forecasts, tmp_path):
    """描く単位と、図に出す学習期間の週数を絞っても書き出される"""
    path = str(tmp_path / 'forecast_one.png')
    out = plot_forecast_grid(forecasts, 'title', path, units=[2], history_weeks=3)
    assert os.path.getsize(out) > 0


def test_forecast_styles_are_all_distinguishable():
    """4本の予測は、色と線種の組が互いに違う（同じ見た目の線が2本ないこと）"""
    looks = {(color, linestyle) for _, color, linestyle in FORECAST_STYLES.values()}
    assert len(looks) == len(FORECAST_STYLES)


def test_forecast_styles_separate_prophet_and_baseline_by_color():
    """Prophet の2本は同じ色、比べる相手の2本は別の同じ色（同じ色の2本は線種で見分ける）"""
    colors = {name: color for name, (_, color, _) in FORECAST_STYLES.items()}
    assert (colors['prophet_yearly'] == colors['prophet_no_yearly']
            != colors['baseline_mean'] == colors['baseline_recent'])


def test_satisfaction_colors_are_orange_and_aqua():
    """充足度の配色は赤↔緑ではない（色覚多様性で識別できないため却下されている）

    docs/decisions.md 2026-08-18。
    """
    for color in (LOW_COLOR, HIGH_COLOR):
        r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        assert not (r > 150 and g < 100 and b < 100), '赤系は使わない'
        assert not (g > 150 and r < 100 and b < 100), '緑系は使わない'
