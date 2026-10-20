"""
発売後の山の続き方モジュールのテスト

窓の切り方・残った割合（件数・割合）・発売前の比・測れない発売の除外・区分ファイルの検査・
組の分け方・3本未満の組で中央値を出さないこと・相対値・食い違いの印を、手で数えられる小さい表で確かめる。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import pytest
from src.timeseries.persistence import (
    add_group_comparison,
    assign_group,
    measure_persistence,
    persistence_curves,
    validate_labels,
    weekly_counts,
)

START = pd.Timestamp('2024-01-01')  # 月曜日
WEEKS = 60


def _reviews(weekly: dict) -> pd.DataFrame:
    """{ゲーム名: [START から始まる週ごとの件数]} から、レビュー1件1行の表を作る"""
    rows = []
    for name, per_week in weekly.items():
        for i, n in enumerate(per_week):
            ts = int((START + pd.Timedelta(weeks=i, hours=12)).timestamp())
            rows += [{'game_name': name, 'timestamp_created': ts}] * n
    return pd.DataFrame(rows)


def _game(first: int, rest: int, pre: int = 0, release_week: int = 10):
    """発売週 release_week の前を pre、直後8週を first、中盤18週・後半を rest にした週次件数"""
    series = [pre] * release_week
    series += [first] * 8 + [rest] * (WEEKS - release_week - 8)
    return series


def _release(weeks_after_start: int) -> str:
    return str((START + pd.Timedelta(weeks=weeks_after_start, days=2)).date())


@pytest.fixture
def basic():
    """A: 直後100・以降20・発売前10 / B: 直後100・以降50（発売週は両方10週目）"""
    reviews = _reviews({'A': _game(100, 20, pre=10), 'B': _game(100, 50), 'Z': [5] * WEEKS})
    counts, oldest = weekly_counts(reviews)
    releases = pd.Series({'A': _release(10), 'B': _release(10)})
    return counts, oldest, releases


def test_weekly_counts_fills_missing_weeks():
    counts, _ = weekly_counts(_reviews({'A': [3, 0, 2]}))
    assert list(counts['A']) == [3, 0, 2]


def test_windows_and_persistence_counts(basic):
    counts, oldest, releases = basic
    measured, excluded = measure_persistence(counts, oldest, releases)
    a = measured.set_index('name').loc['A']
    assert excluded.empty
    assert a['release_week'] == START + pd.Timedelta(weeks=10)
    # 発売直後 = 0〜7週の平均、中盤 = 8〜25週、後半 = 26週〜最後
    assert a['immediate_count'] == 100
    assert a['mid_count'] == 20
    assert a['late_count'] == 20
    assert a['persist_mid_count'] == pytest.approx(0.2)
    assert a['persist_late_count'] == pytest.approx(0.2)


def test_window_boundary_weeks(basic):
    counts, oldest, releases = basic
    # 7週目までが直後・8週目から中盤: 7週目だけ違う値にして境目を確かめる
    weekly = _game(100, 20)
    weekly[10 + 7] = 200
    weekly[10 + 8] = 40
    weekly[10 + 25] = 60
    weekly[10 + 26] = 80
    c, o = weekly_counts(_reviews({'A': weekly}))
    a = measure_persistence(c, o, pd.Series({'A': _release(10)}))[0].iloc[0]
    assert a['immediate_count'] == pytest.approx((7 * 100 + 200) / 8)
    assert a['mid_count'] == pytest.approx((16 * 20 + 40 + 60) / 18)
    late_weeks = WEEKS - 36
    assert a['late_count'] == pytest.approx(((late_weeks - 1) * 20 + 80) / late_weeks)


def test_share_uses_all_games_total(basic):
    counts, oldest, releases = basic
    measured, _ = measure_persistence(counts, oldest, releases)
    a = measured.set_index('name').loc['A']
    # 直後の週の合計は A100 + B100 + Z5 = 205、中盤は 20 + 50 + 5 = 75
    assert a['immediate_share'] == pytest.approx(100 / 205)
    assert a['mid_share'] == pytest.approx(20 / 75)
    assert a['persist_mid_share'] == pytest.approx((20 / 75) / (100 / 205))


def test_pre_ratio(basic):
    counts, oldest, releases = basic
    measured, _ = measure_persistence(counts, oldest, releases)
    m = measured.set_index('name')
    assert m.loc['A', 'pre_count'] == 10
    assert m.loc['A', 'pre_ratio_count'] == pytest.approx(0.1)
    # B は発売前のレビューが無い（最古が発売週）ので空
    assert np.isnan(m.loc['B', 'pre_count'])


def test_excludes_missing_immediate_and_short_window(basic):
    counts, oldest, _ = basic
    reviews = _reviews({'A': _game(100, 20), 'L': [0] * 20 + [50] * 40,
                        'S': _game(100, 20), 'Z': [5] * WEEKS})
    c, o = weekly_counts(reviews)
    releases = pd.Series({
        'A': _release(10),
        'L': _release(10),    # 最古のレビューが発売日の70日あと → 発売直後が欠けている
        'S': _release(40),    # 発売40週目 + 25週が60週の外 → 中盤が最後を越える
    })
    measured, excluded = measure_persistence(c, o, releases)
    assert list(measured['name']) == ['A']
    reasons = dict(zip(excluded['name'], excluded['reason']))
    assert '発売直後' in reasons['L']
    assert '中盤' in reasons['S']


def test_missing_days_threshold():
    reviews = _reviews({'A': [10] * WEEKS, 'Z': [5] * WEEKS})
    c, o = weekly_counts(reviews)
    # 最古は START の12時。発売日がその7日前なら測れる・8日前なら外す（8日以上あと）
    oldest_date = o['A'].normalize()
    ok = pd.Series({'A': str((oldest_date - pd.Timedelta(days=7)).date())})
    ng = pd.Series({'A': str((oldest_date - pd.Timedelta(days=8)).date())})
    assert len(measure_persistence(c, o, ok)[0]) == 1
    assert len(measure_persistence(c, o, ng)[1]) == 1


def _labels(rows):
    return pd.DataFrame(rows, columns=['name', 'fan_base', 'launch_type', 'note'])


def test_validate_labels_stops_on_missing_game():
    labels = _labels([['A', 'none', 'first', '']])
    with pytest.raises(ValueError, match='区分ファイルに無い'):
        validate_labels(labels, ['A', 'B'])


def test_validate_labels_stops_on_unknown_word():
    with pytest.raises(ValueError, match='決められた語'):
        validate_labels(_labels([['A', 'fans', 'first', '']]), ['A'])
    with pytest.raises(ValueError, match='決められた語'):
        validate_labels(_labels([['A', 'none', 'remake', '']]), ['A'])


def test_assign_group():
    assert assign_group('series', 'first') == 'first_series'
    assert assign_group('none', 'rerelease') == 'rerelease'
    assert assign_group('series', 'ea_to_full') == 'ea_to_full'


def _measured(values: dict) -> pd.DataFrame:
    """{名前: (中盤・件数, 中盤・割合)} から、比較に必要な列だけの表を作る"""
    rows = []
    for name, (count, share) in values.items():
        rows.append({'name': name, 'persist_mid_count': count, 'persist_late_count': count,
                     'persist_mid_share': share, 'persist_late_share': share})
    return pd.DataFrame(rows)


def test_group_median_relative_and_small_group():
    measured = _measured({'A': (0.1, 0.1), 'B': (0.2, 0.2), 'C': (0.4, 0.4), 'D': (0.3, 0.3)})
    labels = _labels([['A', 'none', 'first', ''], ['B', 'none', 'first', ''],
                      ['C', 'none', 'first', ''], ['D', 'series', 'first', '']]).set_index('name')
    result, summary = add_group_comparison(measured, labels, min_group=3)
    s = summary.set_index('group')
    assert s.loc['first_none', 'n'] == 3
    assert s.loc['first_none', 'median_mid_count'] == pytest.approx(0.2)
    r = result.set_index('name')
    assert r.loc['C', 'relative_mid_count'] == pytest.approx(2.0)
    assert r.loc['A', 'relative_mid_count'] == pytest.approx(0.5)
    # 1本だけの組は中央値も相対値も出さない
    assert np.isnan(s.loc['first_series', 'median_mid_count'])
    assert np.isnan(r.loc['D', 'relative_mid_count'])
    assert pd.isna(r.loc['D', 'mismatch_mid'])


def test_mismatch_flag():
    # 件数版の中央値は 0.2、割合版の中央値は 0.2。A は件数で上・割合で下 → 食い違い
    measured = _measured({'A': (0.3, 0.1), 'B': (0.2, 0.2), 'C': (0.1, 0.3)})
    labels = _labels([[n, 'none', 'first', ''] for n in 'ABC']).set_index('name')
    result, _ = add_group_comparison(measured, labels, min_group=3)
    r = result.set_index('name')
    assert bool(r.loc['A', 'mismatch_mid']) and bool(r.loc['C', 'mismatch_mid'])
    assert not bool(r.loc['B', 'mismatch_mid'])


def test_persistence_curves(basic):
    counts, oldest, releases = basic
    measured, _ = measure_persistence(counts, oldest, releases)
    measured['group'] = 'g'
    curves = persistence_curves(counts, measured)
    a = curves[curves['name'] == 'A'].set_index('weeks_since_launch')
    assert a.loc[0, 'ratio'] == pytest.approx(1.0)
    assert a.loc[8, 'ratio'] == pytest.approx(0.2)
