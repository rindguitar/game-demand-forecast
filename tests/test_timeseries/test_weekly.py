"""
週次時系列モジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import pytest
from src.timeseries.weekly import (
    add_week_column,
    build_weekly_series,
    trim_partial_weeks,
)

WEEK = 7 * 24 * 3600
# 2023-09-04 は月曜日。ここを起点に週を作る
MONDAY = int(pd.Timestamp('2023-09-04').timestamp())


def _reviews(rows):
    """(topic_id, 週番号, ゲーム名, voted_up) からレビューのDataFrameを作る"""
    return pd.DataFrame([
        {'topic_id': t, 'timestamp_created': MONDAY + w * WEEK + 3600,
         'game_name': g, 'voted_up': v}
        for t, w, g, v in rows
    ])


def test_add_week_column_snaps_to_monday():
    """週の列はその週の月曜日を指す"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 0, 'A', True)]))
    assert df['week'].nunique() == 1
    assert df['week'].iloc[0] == pd.Timestamp('2023-09-04')


def test_trim_partial_weeks_drops_incomplete_tail():
    """最後の週が7日そろっていなければ落とす"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 1, 'A', True)]))
    # 週0は月曜1時から日曜まで無いので、両端とも部分週になる
    assert trim_partial_weeks(df).empty


def test_trim_partial_weeks_keeps_complete_weeks():
    """真ん中の完全な週は残る"""
    rows = [(1, w, 'A', True) for w in range(4)]
    df = add_week_column(_reviews(rows))
    # 端を人工的に完全にする（週0の頭と週3の尻にレビューを足す）
    df.loc[len(df)] = {'topic_id': 1, 'timestamp_created': MONDAY,
                       'game_name': 'A', 'voted_up': True,
                       'week': pd.Timestamp('2023-09-04'),
                       '_date': pd.Timestamp('2023-09-04')}
    end = pd.Timestamp('2023-09-04') + pd.Timedelta(weeks=3, days=6, hours=23, minutes=59)
    df.loc[len(df)] = {'topic_id': 1, 'timestamp_created': int(end.timestamp()),
                       'game_name': 'A', 'voted_up': True,
                       'week': pd.Timestamp('2023-09-25'), '_date': end}
    assert trim_partial_weeks(df)['week'].nunique() == 4


def test_build_weekly_series_fills_gap_weeks_with_zero():
    """観測が始まった後に空いた週は0件として埋める"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 2, 'A', True)]))
    series = build_weekly_series(df)
    assert series.set_index('week').loc[pd.Timestamp('2023-09-11'), 'count'] == 0


def test_build_weekly_series_masks_weeks_before_first_observation():
    """初出より前の週は0ではなく欠測にする（需要ゼロではなく観測対象外のため）"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (2, 2, 'B', True)]))
    series = build_weekly_series(df)
    t2 = series[series['unit'] == 2].set_index('week')
    assert pd.isna(t2.loc[pd.Timestamp('2023-09-04'), 'count'])
    assert t2.loc[pd.Timestamp('2023-09-18'), 'count'] == 1


def test_build_weekly_series_share_uses_week_total():
    """シェアはその週の総言及数に対する割合になる"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (2, 0, 'B', True),
                                   (2, 0, 'B', True), (2, 0, 'B', True)]))
    series = build_weekly_series(df)
    assert series[series['unit'] == 1]['share'].iloc[0] == pytest.approx(0.25)


def test_build_weekly_series_positive_rate_needs_minimum():
    """件数が少ない週のポジ率は欠測にする（比率が暴れるため）"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 0, 'A', False)]))
    series = build_weekly_series(df, min_reviews_for_rate=5)
    assert pd.isna(series['positive_rate'].iloc[0])


def test_build_weekly_series_positive_rate_value():
    """件数が足りていればポジ率が出る"""
    rows = [(1, 0, 'A', True)] * 3 + [(1, 0, 'A', False)] * 1
    series = build_weekly_series(add_week_column(_reviews(rows)), min_reviews_for_rate=4)
    assert series['positive_rate'].iloc[0] == pytest.approx(0.75)


def test_expected_positive_rate_reflects_game_mix():
    """期待ポジ率は、そのトピック・その週のゲーム構成から決まる

    ゲームAは全レビューがポジ（100%）、ゲームBは全部ネガ（0%）。
    半々のトピックなら期待は50%になる。
    """
    rows = [(1, 0, 'A', True), (1, 0, 'B', False),
            (2, 0, 'A', True), (2, 0, 'A', True)]
    series = build_weekly_series(add_week_column(_reviews(rows)), min_reviews_for_rate=1)
    by_unit = series.set_index('unit')['expected_positive_rate']
    assert by_unit[1] == pytest.approx(0.5)   # AとBが半々
    assert by_unit[2] == pytest.approx(1.0)   # Aだけ


def test_positive_rate_gap_removes_game_reputation():
    """評判の悪いゲームに偏ったトピックでも、期待どおりなら差はゼロになる

    実測: cards は実際50%だが、中身の93%を占める MTG Arena 自体が53%で、
    差はほぼ無かった。これを絶対値のまま読むと「満たされていない需要」と誤読する。
    """
    # ゲームBは評判が悪い（4件中1件だけポジ = 25%）。トピック1はBだけで構成される
    rows = [(1, 0, 'B', True), (1, 0, 'B', False), (1, 0, 'B', False), (1, 0, 'B', False)]
    series = build_weekly_series(add_week_column(_reviews(rows)), min_reviews_for_rate=1)
    row = series.iloc[0]
    assert row['positive_rate'] == pytest.approx(0.25)   # 絶対値は低い
    assert row['positive_rate_gap'] == pytest.approx(0.0)  # ゲームの評判どおりなので差はゼロ


def test_positive_rate_gap_is_missing_when_rate_is():
    """件数が足りない週は、差も欠測にする"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 0, 'A', False)]))
    series = build_weekly_series(df, min_reviews_for_rate=5)
    assert pd.isna(series['positive_rate_gap'].iloc[0])


def test_build_weekly_series_counts_games():
    """参加ゲーム数を数える"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 0, 'B', True), (1, 0, 'A', True)]))
    assert build_weekly_series(df)['games'].iloc[0] == 2


def test_measure_topic_panels_uses_median_density():
    """週あたり件数は中央値で測る（発売スパイク型を密度十分にしないため）"""
    from src.timeseries.weekly import measure_topic_panels
    # トピック1は最初の週だけ100件、以降は毎週1件（平均は高いが中央値は1）
    rows = [(1, 0, 'A', True)] * 100 + [(1, w, 'A', True) for w in range(1, 10)]
    rows += [(2, w, 'A', True) for w in range(10) for _ in range(5)]
    df = _reviews(rows)
    # 端の週を完全にして落とされないようにする
    df.loc[len(df)] = {'topic_id': 1, 'timestamp_created': MONDAY,
                       'game_name': 'A', 'voted_up': True}
    df.loc[len(df)] = {'topic_id': 1, 'timestamp_created': MONDAY + 10 * WEEK - 60,
                       'game_name': 'A', 'voted_up': True}
    panels, _ = measure_topic_panels(df, backbone_games=['A'])
    assert panels.loc[1, 'per_week_all'] < panels.loc[2, 'per_week_all']


def test_measure_topic_panels_excludes_outlier():
    """Outlier（-1）は単位として数えない"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(-1, w, 'A', True) for w in range(5)] + [(1, w, 'A', True) for w in range(5)]
    panels, totals = measure_topic_panels(_reviews(rows), backbone_games=['A'])
    assert -1 not in panels.index
    assert totals['assigned'] < totals['all_reviews']


def test_measure_topic_panels_top1_share_and_backbone():
    """集中度は最も多いゲームの割合。土台パネルは指定したゲームだけで測る"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(1, w, 'A', True) for w in range(6) for _ in range(3)]
    rows += [(1, w, 'B', True) for w in range(6)]
    panels, _ = measure_topic_panels(_reviews(rows), backbone_games=['B'])
    assert panels.loc[1, 'top1_game'] == 'A'
    assert panels.loc[1, 'top1_share'] == pytest.approx(0.75)
    assert panels.loc[1, 'count_backbone'] < panels.loc[1, 'count_all']


def test_measure_topic_panels_respects_unit_column():
    """束ねた単位（unit 列）でも同じ物差しで測れる"""
    from src.timeseries.weekly import measure_topic_panels
    df = _reviews([(1, w, 'A', True) for w in range(6)])
    df['unit'] = 7
    panels, _ = measure_topic_panels(df, backbone_games=['A'], unit_column='unit')
    assert panels.index.tolist() == [7]
