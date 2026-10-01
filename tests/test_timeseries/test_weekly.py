"""
週次時系列モジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import pytest
from src.timeseries.weekly import (
    MIN_WEEKS_SINCE_RELEASE,
    add_week_column,
    build_weekly_series,
    common_window,
    decide_window_and_backbone,
    describe_window_and_backbone,
    select_backbone,
    trim_to_window,
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


def _window(first_week, last_week):
    """週番号（MONDAY の週を0とする）から、共通の期間 (最初の週の月曜, 最後の週の月曜) を作る"""
    monday = pd.Timestamp('2023-09-04')
    return monday + pd.Timedelta(weeks=first_week), monday + pd.Timedelta(weeks=last_week)


def test_add_week_column_snaps_to_monday():
    """週の列はその週の月曜日を指す"""
    df = add_week_column(_reviews([(1, 0, 'A', True), (1, 0, 'A', True)]))
    assert df['week'].nunique() == 1
    assert df['week'].iloc[0] == pd.Timestamp('2023-09-04')


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
    panels, _ = measure_topic_panels(_reviews(rows), backbone_games=['A'], window=_window(0, 9))
    assert panels.loc[1, 'per_week_all'] < panels.loc[2, 'per_week_all']


def test_measure_topic_panels_excludes_outlier():
    """Outlier（-1）は単位として数えない"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(-1, w, 'A', True) for w in range(5)] + [(1, w, 'A', True) for w in range(5)]
    panels, totals = measure_topic_panels(_reviews(rows), backbone_games=['A'],
                                          window=_window(0, 4))
    assert -1 not in panels.index
    assert totals['assigned'] < totals['all_reviews']


def test_measure_topic_panels_top1_share_and_backbone():
    """集中度は最も多いゲームの割合。土台パネルは指定したゲームだけで測る"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(1, w, 'A', True) for w in range(6) for _ in range(3)]
    rows += [(1, w, 'B', True) for w in range(6)]
    panels, _ = measure_topic_panels(_reviews(rows), backbone_games=['B'], window=_window(0, 5))
    assert panels.loc[1, 'top1_game'] == 'A'
    assert panels.loc[1, 'top1_share'] == pytest.approx(0.75)
    assert panels.loc[1, 'count_backbone'] < panels.loc[1, 'count_all']


def test_measure_topic_panels_respects_unit_column():
    """束ねた単位（unit 列）でも同じ物差しで測れる"""
    from src.timeseries.weekly import measure_topic_panels
    df = _reviews([(1, w, 'A', True) for w in range(6)])
    df['unit'] = 7
    panels, _ = measure_topic_panels(df, backbone_games=['A'], window=_window(0, 5),
                                     unit_column='unit')
    assert panels.index.tolist() == [7]


def test_measure_topic_panels_counts_only_reviews_inside_window():
    """期間の外のレビューは数えない（端の週が密度の中央値に混ざらないように）"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(1, w, 'A', True) for w in range(6)]
    _, totals = measure_topic_panels(_reviews(rows), backbone_games=['A'], window=_window(1, 4))
    assert totals['all_reviews'] == 4


def test_measure_topic_panels_week_axis_is_the_window():
    """週の軸は期間そのもの。期間の端にレビューが無い週も0件として数える"""
    from src.timeseries.weekly import measure_topic_panels
    rows = [(1, w, 'A', True) for w in range(2, 4)]
    _, totals = measure_topic_panels(_reviews(rows), backbone_games=['A'], window=_window(0, 5))
    assert totals['weeks'] == 6


def test_measure_topic_panels_without_window_raises_type_error():
    """期間を省略すると TypeError（端の週が混ざる古いやり方へ黙って戻さないため）"""
    from src.timeseries.weekly import measure_topic_panels
    with pytest.raises(TypeError):
        measure_topic_panels(_reviews([(1, 0, 'A', True)]), backbone_games=['A'])


def test_trim_to_window_keeps_edge_weeks_and_drops_outside():
    """期間の最初の週と最後の週は残し、その外の週は落とす"""
    df = add_week_column(_reviews([(1, w, 'A', True) for w in range(6)]))
    window = (pd.Timestamp('2023-09-11'), pd.Timestamp('2023-10-02'))  # 週1〜週4
    kept = sorted(trim_to_window(df, window)['week'].unique())
    assert kept == list(pd.date_range('2023-09-11', '2023-10-02', freq='W-MON'))


# --- 共通の期間（common_window）と土台の選び方（select_backbone） ---
# 日付は実データの収集ログを模している。ロスターを2回に分けて集めたため、
# 最初の24本は 2023-09-05〜2026-09-03、追加の40本は 2023-09-20ごろ〜2026-09-20ごろだった

WINDOW_START = pd.Timestamp('2023-09-25')  # 月曜


def _log(*rows):
    """(ゲーム名, oldest, newest) から収集ログのDataFrameを作る"""
    return pd.DataFrame(list(rows), columns=['name', 'oldest', 'newest'])


def _games(*rows):
    """(ゲーム名, 発売日, tier) からゲーム台帳のDataFrameを作る"""
    return pd.DataFrame(list(rows), columns=['name', 'release_date', 'tier'])


def _release(weeks_before, extra_days=0):
    """期間開始（WINDOW_START）の weeks_before 週前から、extra_days 日あとの日付（文字列）"""
    day = WINDOW_START - pd.Timedelta(weeks=weeks_before) + pd.Timedelta(days=extra_days)
    return day.strftime('%Y-%m-%d')


def _two_batches():
    """2回に分けて集めたロスター。極端な日付を持つゲームが行の真ん中に来るようにしてある"""
    return _log(('first', '2023-09-05', '2026-09-03'),   # 最初の24本: 最も早い newest
                ('second', '2023-09-22', '2026-09-19'),  # 追加の40本: 最も遅い oldest
                ('third', '2023-09-21', '2026-09-13'),
                ('fourth', '2023-09-06', '2026-09-04'))


def test_common_window_start_skips_week_of_oldest():
    """開始は、oldest（金曜）を含む週の次の月曜。その週は途中からしか集めていない"""
    log = _log(('A', '2023-09-22', '2026-09-19'))
    assert common_window(log, ['A'])[0] == pd.Timestamp('2023-09-25')


def test_common_window_start_skips_week_even_when_oldest_is_monday():
    """oldest が月曜でも、その週は使わない"""
    log = _log(('A', '2023-09-18', '2026-09-19'))
    assert common_window(log, ['A'])[0] == pd.Timestamp('2023-09-25')


def test_common_window_end_skips_week_of_newest():
    """終了は、newest（木曜）を含む週の前の週の月曜。その週は途中までしか集めていない"""
    log = _log(('A', '2023-09-05', '2026-09-03'))
    assert common_window(log, ['A'])[1] == pd.Timestamp('2026-08-24')


def test_common_window_end_skips_week_even_when_newest_is_sunday():
    """newest が日曜でも、その週は使わない"""
    log = _log(('A', '2023-09-05', '2026-09-13'))
    assert common_window(log, ['A'])[1] == pd.Timestamp('2026-08-31')


def test_common_window_start_follows_latest_oldest():
    """開始は、全ゲームの oldest のうち最も遅いもので決まる"""
    names = ['first', 'second', 'third', 'fourth']
    assert common_window(_two_batches(), names)[0] == pd.Timestamp('2023-09-25')


def test_common_window_end_follows_earliest_newest():
    """終了は、全ゲームの newest のうち最も早いもので決まる"""
    names = ['first', 'second', 'third', 'fourth']
    assert common_window(_two_batches(), names)[1] == pd.Timestamp('2026-08-24')


def test_common_window_ignores_games_not_in_names():
    """対象に渡していないゲームは期間を狭めない（発売が期間より後なら oldest が発売日になる）"""
    log = _log(('A', '2023-09-05', '2026-09-19'), ('late', '2025-04-10', '2026-09-19'))
    assert common_window(log, ['A'])[0] == pd.Timestamp('2023-09-11')


def test_common_window_game_missing_from_log_raises():
    """収集ログに無いゲームがあれば、既定値で埋めずに止まる"""
    log = _log(('A', '2023-09-05', '2026-09-03'))
    with pytest.raises(ValueError, match='収集ログに無い'):
        common_window(log, ['A', 'B'])


def test_common_window_empty_date_raises():
    """oldest が空のゲームがあれば、黙って飛ばさず止まる"""
    log = _log(('A', '2023-09-05', '2026-09-03'), ('B', None, None))
    with pytest.raises(ValueError, match='空のゲーム'):
        common_window(log, ['A', 'B'])


def test_common_window_duplicate_rows_use_last():
    """収集ログは追記式。失敗して空の行のあと取り直した場合は、最後の行を使う"""
    log = _log(('A', None, None), ('A', '2023-09-22', '2026-09-03'))
    assert common_window(log, ['A'])[0] == pd.Timestamp('2023-09-25')


def test_common_window_no_names_raises():
    """期間を決めるゲームが1本も無ければ止まる"""
    with pytest.raises(ValueError, match='1本も'):
        common_window(_two_batches(), [])


def test_common_window_without_overlap_raises():
    """全ゲームの収集がそろう週が無ければ止まる"""
    log = _log(('A', '2026-06-01', '2026-09-03'), ('B', '2023-09-05', '2026-06-02'))
    with pytest.raises(ValueError, match='そろう週が無い'):
        common_window(log, ['A', 'B'])


def test_select_backbone_release_exactly_min_weeks_before_is_included():
    """期間開始のちょうど min_weeks 週前に発売したゲームは含める（「以前」）"""
    games = _games(('A', _release(MIN_WEEKS_SINCE_RELEASE), '土台'))
    assert select_backbone(games, WINDOW_START) == ['A']


def test_select_backbone_release_one_day_after_boundary_is_excluded():
    """境目より1日でも後に発売したゲームは外れる"""
    games = _games(('A', _release(MIN_WEEKS_SINCE_RELEASE, extra_days=1), '土台'))
    assert select_backbone(games, WINDOW_START) == []


def test_select_backbone_min_weeks_argument_moves_the_boundary():
    """週数を指定すれば境目が動く"""
    games = _games(('A', _release(10), '土台'))
    assert select_backbone(games, WINDOW_START, min_weeks=10) == ['A']


def test_select_backbone_other_tier_is_excluded():
    """tier が違うゲームは、発売が古くても入らない"""
    games = _games(('A', _release(100), '土台'), ('B', _release(100), '直近'))
    assert select_backbone(games, WINDOW_START) == ['A']


def test_select_backbone_tier_argument_selects_that_tier():
    """tier を指定すれば、その tier のゲームから選ぶ"""
    games = _games(('A', _release(100), '土台'), ('B', _release(100), '直近'))
    assert select_backbone(games, WINDOW_START, tier='直近') == ['B']


def test_select_backbone_exclude_removes_named_game():
    """exclude に挙げたゲームは、条件を満たしていても外れる"""
    games = _games(('A', _release(100), '土台'), ('B', _release(100), '土台'))
    assert select_backbone(games, WINDOW_START, exclude=['A']) == ['B']


def test_select_backbone_default_does_not_exclude_by_name():
    """既定では名前で外さない（Starfield を手書きで外していたのをやめた）"""
    games = _games(('Starfield', _release(100), '土台'))
    assert select_backbone(games, WINDOW_START) == ['Starfield']


def test_select_backbone_missing_release_date_raises():
    """発売日が空のゲームがあれば、黙って含めたり外したりせず止まる"""
    games = _games(('A', _release(100), '土台'), ('B', None, '土台'))
    with pytest.raises(ValueError, match='発売日が読めない'):
        select_backbone(games, WINDOW_START)


def test_select_backbone_non_date_release_date_raises():
    """発売日が日付として読めないゲームがあれば止まる"""
    games = _games(('A', 'Coming soon', '土台'))
    with pytest.raises(ValueError, match='発売日が読めない'):
        select_backbone(games, WINDOW_START)


def test_select_backbone_unreadable_date_in_other_tier_is_ignored():
    """選ぶかどうかに関わらない tier のゲームの発売日は問わない"""
    games = _games(('A', _release(100), '土台'), ('B', None, '直近'))
    assert select_backbone(games, WINDOW_START) == ['A']


# --- 期間と土台をまとめて決める入口（decide_window_and_backbone） ---
# build_weekly_series.py / categorize_topics.py / compare_topic_granularity.py の3本が呼ぶ

def _decide_inputs():
    """土台3本（古い2本と、期間開始の5週前に発売した1本）と、土台でない1本の台帳と収集ログ"""
    games = _games(('old1', _release(100), '土台'),
                   ('old2', _release(100), '土台'),
                   ('recent', _release(5), '土台'),
                   ('other', _release(100), '直近'))
    log = _log(('old1', '2023-09-05', '2026-09-19'),
               ('old2', '2023-09-22', '2026-09-03'),   # 最も遅い oldest・最も早い newest
               ('recent', '2023-08-22', '2026-09-19'),
               ('other', '2025-04-10', '2026-09-19'))  # 土台でないゲームは期間を狭めない
    return games, log


def test_decide_window_and_backbone_window_ignores_games_of_other_tiers():
    """期間は tier が土台のゲームの収集ログだけで決まる"""
    games, log = _decide_inputs()
    window, _, _ = decide_window_and_backbone(games, log)
    assert window == (pd.Timestamp('2023-09-25'), pd.Timestamp('2026-08-24'))


def test_decide_window_and_backbone_backbone_is_chosen_from_window_start():
    """土台は、決めた期間の開始から数えて選ぶ"""
    games, log = _decide_inputs()
    _, backbone, _ = decide_window_and_backbone(games, log)
    assert backbone == ['old1', 'old2']


def test_decide_window_and_backbone_left_out_is_tier_games_not_selected():
    """外れたゲームは、tier が土台なのに選ばれなかったもの（tier が違うゲームは数えない）"""
    games, log = _decide_inputs()
    _, _, left_out = decide_window_and_backbone(games, log)
    assert left_out == ['recent']


def test_decide_window_and_backbone_manual_exclude_is_in_left_out():
    """手で外したゲームも、外れたゲームに入る"""
    games, log = _decide_inputs()
    _, _, left_out = decide_window_and_backbone(games, log, exclude=['old1'])
    assert left_out == ['old1', 'recent']


def test_decide_window_and_backbone_min_weeks_is_passed_to_selection():
    """min_weeks を渡せば、土台の条件が変わる"""
    games, log = _decide_inputs()
    _, backbone, _ = decide_window_and_backbone(games, log, min_weeks=4)
    assert backbone == ['old1', 'old2', 'recent']


def test_decide_window_and_backbone_tier_argument_changes_target_games():
    """tier を渡せば、その tier のゲームから期間も土台も決める"""
    games, log = _decide_inputs()
    _, backbone, _ = decide_window_and_backbone(games, log, tier='直近')
    assert backbone == ['other']


def test_decide_window_and_backbone_game_missing_from_log_raises():
    """収集ログに無いゲームがあれば止まる（既定値で埋めない）"""
    games, log = _decide_inputs()
    with pytest.raises(ValueError, match='収集ログに無い'):
        decide_window_and_backbone(games, log[log['name'] != 'old1'])


# --- 画面に出す文（describe_window_and_backbone） ---

REAL_WINDOW = (pd.Timestamp('2023-09-25'), pd.Timestamp('2026-08-24'))


def test_describe_window_and_backbone_shows_period_and_week_count():
    """期間の開始・終了と、週の数を出す"""
    text = describe_window_and_backbone(REAL_WINDOW, ['A'], [], '土台', 26)
    assert '期間: 2023-09-25 〜 2026-08-24（153週' in text


def test_describe_window_and_backbone_shows_backbone_count_and_rule():
    """土台の本数と、選んだ条件（tier と週数）を出す"""
    text = describe_window_and_backbone(REAL_WINDOW, ['A', 'B'], [], '土台', 26)
    assert '土台パネル: 2本（tier=土台 で、発売が期間開始の26週以上前）' in text


def test_describe_window_and_backbone_lists_left_out_games():
    """外れたゲームを名前で並べる"""
    text = describe_window_and_backbone(REAL_WINDOW, ['A'], ['Starfield', 'Overwatch®'], '土台', 26)
    assert '外れたゲーム（2本）: Starfield, Overwatch®' in text


def test_describe_window_and_backbone_without_left_out_says_none():
    """外れたゲームが無ければ「なし」と出す"""
    text = describe_window_and_backbone(REAL_WINDOW, ['A'], [], '土台', 26)
    assert '外れたゲーム（0本）: なし' in text


# --- 実データでの確認（data/ がある環境だけ。無ければ飛ばす） ---

GAMES_CSV = os.path.join(os.path.dirname(__file__), '../../data/timeseries/games.csv')
LOG_CSV = os.path.join(os.path.dirname(__file__), '../../data/timeseries/collection_log.csv')


@pytest.fixture(scope='module')
def real_basis():
    """実データ（64本のロスターと収集ログ）で決めた (期間, 土台, 外れたゲーム)"""
    if not (os.path.exists(GAMES_CSV) and os.path.exists(LOG_CSV)):
        pytest.skip('data/timeseries/games.csv と collection_log.csv が無い環境では飛ばす')
    return decide_window_and_backbone(pd.read_csv(GAMES_CSV), pd.read_csv(LOG_CSV))


def test_decide_window_and_backbone_real_data_window(real_basis):
    """実データ（64本）の期間は 2023-09-25〜2026-08-24"""
    assert real_basis[0] == REAL_WINDOW


def test_decide_window_and_backbone_real_data_window_has_153_weeks(real_basis):
    """実データ（64本）の期間は153週"""
    assert len(pd.date_range(*real_basis[0], freq='W-MON')) == 153


def test_decide_window_and_backbone_real_data_backbone_has_36_games(real_basis):
    """実データ（64本）の土台は、tier が土台の41本から5本が外れた36本"""
    assert len(real_basis[1]) == 36


def test_decide_window_and_backbone_real_data_left_out_games(real_basis):
    """実データ（64本）で外れるのは、期間開始の26週前より後に発売した5本"""
    assert set(real_basis[2]) == {'Starfield', 'Overwatch®', 'DAVE THE DIVER',
                                  'Magic: The Gathering Arena', 'Darkest Dungeon® II'}
