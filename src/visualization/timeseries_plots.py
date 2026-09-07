"""
週次時系列の可視化

需要の「大きさ」（言及数・シェア）と「充足度」（ポジ率）を別の図にする
（docs/decisions.md 2026-08-18）。充足度の配色はオレンジ ↔ アクア
（赤 ↔ 緑は色覚多様性で識別できないため却下されている）。

コンテナに日本語フォントが無いので、図のラベルは英語で書く。
"""

from typing import Optional, Sequence
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# 充足度の配色（2026-08-18 の決定）
LOW_COLOR = '#E8833A'    # オレンジ = 満たされていない
HIGH_COLOR = '#2AA9A0'   # アクア   = 満たされている
LINE_COLOR = '#3A4A5A'
ROLLING_COLOR = '#C64B3C'


def _grid(n: int, columns: int = 3, height: float = 2.2):
    """小さい図を並べる格子を作る（1枚に線を詰め込まず、系列ごとに分ける）"""
    rows = math.ceil(n / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * (height + 0.5)),
                             squeeze=False)
    return fig, axes.ravel(), rows * columns


def _format_date_axis(ax, months: int = 6) -> None:
    """X軸の日付を間引く（週次155点をそのまま出すとラベルが重なって読めない）"""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=months))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')


def _label(keywords: str, limit: int = 34) -> str:
    """凡例に出すトピック名（キーワードの先頭だけ）"""
    text = str(keywords)
    return text if len(text) <= limit else text[:limit - 1] + '…'


def plot_series_grid(series: pd.DataFrame, value_column: str, title: str,
                     save_path: str, rolling: int = 4,
                     units: Optional[Sequence] = None) -> str:
    """
    単位ごとに1枚ずつ並べた折れ線を描く

    1枚に何本も重ねると読めないので、系列ごとに小さい図を並べる。
    週次はぎざぎざするので、移動平均を重ねて傾きが見えるようにする。
    """
    order = units if units is not None else (
        series.groupby('unit')[value_column].median().sort_values(ascending=False).index)
    fig, axes, slots = _grid(len(order))

    for ax, unit in zip(axes, order):
        one = series[series['unit'] == unit].sort_values('week')
        ax.plot(one['week'], one[value_column], color=LINE_COLOR, linewidth=0.8, alpha=0.55)
        if rolling:
            ax.plot(one['week'], one[value_column].rolling(rolling, min_periods=1).mean(),
                    color=ROLLING_COLOR, linewidth=1.6)
        ax.set_title(f"t{unit}  {_label(one['keywords'].iloc[0])}", fontsize=8, loc='left')
        ax.tick_params(labelsize=7)
        ax.margins(x=0.01)
        _format_date_axis(ax)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in axes[len(order):slots]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_positive_rate_grid(series: pd.DataFrame, title: str, save_path: str,
                            rolling: int = 4) -> str:
    """
    充足度を「期待ポジ率との差」で描く

    実際のポジ率をそのまま描くと、そのトピックがどのゲームの話かを映すだけになる
    （評判の悪いゲームのトピックは常に低く、良いゲームのトピックは常に高く出る）。
    ゲーム構成から期待される率を引いた差を描くことで、要素そのものの効き方が残る。

    0より上（アクア）= その要素に触れた人はゲームを高く評価する傾向
    0より下（オレンジ）= 低く評価する傾向
    """
    order = series.groupby('unit')['count'].median().sort_values(ascending=False).index
    fig, axes, slots = _grid(len(order))
    limit = series['positive_rate_gap'].abs().quantile(0.99)

    for ax, unit in zip(axes, order):
        one = series[series['unit'] == unit].sort_values('week')
        gap = one['positive_rate_gap'].rolling(rolling, min_periods=1).mean()
        ax.plot(one['week'], gap, color=LINE_COLOR, linewidth=1.2)
        ax.axhline(0, color='#999999', linewidth=0.8, linestyle='--')
        ax.fill_between(one['week'], 0, gap, where=gap >= 0,
                        color=HIGH_COLOR, alpha=0.45, interpolate=True)
        ax.fill_between(one['week'], 0, gap, where=gap < 0,
                        color=LOW_COLOR, alpha=0.45, interpolate=True)
        ax.set_ylim(-limit, limit)

        # 絶対値も分かるようにタイトルへ入れる（差だけだと水準が見えないため）
        actual = (one['positive_rate'] * one['count']).sum() / one['count'].sum()
        expected = (one['expected_positive_rate'] * one['count']).sum() / one['count'].sum()
        ax.set_title(f"t{unit}  {_label(one['keywords'].iloc[0], 26)}\n"
                     f"actual {actual:.0%} vs expected {expected:.0%}"
                     f"  ({actual - expected:+.0%})", fontsize=8, loc='left')
        ax.tick_params(labelsize=7)
        ax.margins(x=0.01)
        _format_date_axis(ax)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)

    for ax in axes[len(order):slots]:
        ax.set_visible(False)

    fig.suptitle(f"{title}  (gap vs the rate expected from the game mix)", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def plot_overview(series: pd.DataFrame, value_column: str, title: str, save_path: str,
                  top: int = 8, rolling: int = 4) -> str:
    """
    上位の系列だけを1枚に重ねる（全体の関係を見るため）

    重ねる本数は8本まで。それ以上は線が絡んで読めなくなる。
    """
    order = (series.groupby('unit')[value_column].median()
             .sort_values(ascending=False).head(top).index)
    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap('tab10')

    for i, unit in enumerate(order):
        one = series[series['unit'] == unit].sort_values('week')
        ax.plot(one['week'], one[value_column].rolling(rolling, min_periods=1).mean(),
                color=cmap(i % 10), linewidth=1.8,
                label=f"t{unit}  {_label(one['keywords'].iloc[0], 28)}")

    ax.set_title(title, fontsize=12, loc='left')
    ax.set_ylabel(value_column)
    ax.legend(fontsize=8, ncol=2, frameon=False, loc='upper left')
    ax.margins(x=0.01)
    _format_date_axis(ax, months=3)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path
