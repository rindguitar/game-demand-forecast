"""
Prophet 予測モジュールのテスト

Prophet そのものの出来は確かめない（実データで回して見る）。ここでは、学習とテストの切り方・
比べる相手・MAE と比・勝ちの数え方が仕様どおりであることと、Prophet が年次季節性あり・なしの
両方で回ることを確かめる。発売を出来事として渡す部分（発売の選び方・holidays の形・
比べる相手が発売後の週を除くこと）も確かめる。数値を厳密に確かめたいところは、
Prophet を偽物に差し替える。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import pytest
from prophet import Prophet
from src.timeseries import forecast
from src.timeseries.forecast import (
    COMPARISONS,
    METHODS,
    FitFallbackWarning,
    baseline_mean,
    baseline_recent,
    drop_holiday_weeks,
    evaluate_unit,
    find_target_launches,
    forecast_prophet,
    launch_holidays,
    mae,
    mae_ratio,
    select_launch_events,
    split_train_test,
    summarize_comparisons,
)

# 2024-01-01 は月曜日。ここを起点に週を作る
MONDAY = pd.Timestamp('2024-01-01')


def _weekly(values, first_week=0):
    """値の並びから、月曜始まりの週次DataFrame（week / share）を作る。first_week は起点からの週数"""
    weeks = pd.date_range(MONDAY + pd.Timedelta(weeks=first_week), periods=len(values),
                          freq='W-MON')
    return pd.DataFrame({'week': weeks, 'share': values})


def _seasonal(n_weeks, first_week=0):
    """1年周期の波を持つ週次の値（ノイズなし）。Prophet の動作確認に使う"""
    t = np.arange(first_week, first_week + n_weeks)
    return _weekly(0.02 + 0.01 * np.sin(2 * np.pi * t / 52), first_week)


# ---------------------------------------------------------------- 学習とテストの切り方

def test_split_train_test_takes_last_weeks_as_test():
    """最後の test_weeks 週がテスト、その前が学習になる"""
    train, test, _ = split_train_test(_weekly(range(10)), test_weeks=3)
    assert (len(train), len(test)) == (7, 3)


def test_split_train_test_returns_first_test_week_as_cutoff():
    """切る週は、テストの最初の週"""
    _, _, cutoff = split_train_test(_weekly(range(10)), test_weeks=3)
    assert cutoff == MONDAY + pd.Timedelta(weeks=7)


def test_split_train_test_does_not_overlap():
    """学習は切る週より前、テストは切る週から。同じ週が両方に入らない"""
    train, test, cutoff = split_train_test(_weekly(range(10)), test_weeks=3)
    assert train['week'].max() < cutoff <= test['week'].min()


def test_split_train_test_drops_missing_values_from_both():
    """値が欠測の行は、学習・テストとも除く"""
    values = [0, np.nan, 2, 3, 4, 5, 6, 7, np.nan, 9]
    train, test, _ = split_train_test(_weekly(values), test_weeks=3)
    assert (len(train), len(test)) == (6, 2)


def test_split_train_test_counts_missing_weeks_when_cutting():
    """切る週は欠測の週も数えて決める（最後の週が欠測でも、テストは最後の3週ぶん）"""
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, np.nan]
    _, _, cutoff = split_train_test(_weekly(values), test_weeks=3)
    assert cutoff == MONDAY + pd.Timedelta(weeks=7)


def test_split_train_test_cuts_all_units_at_same_week():
    """単位ごとに実績の末尾が違っても、全単位で同じ週で切る

    単位 b は最後の2週の行が無い。単位ごとに数えると b だけ切る週が前にずれ、テストが3行になる。
    """
    unit_a = _weekly(range(10)).assign(unit='a')
    unit_b = _weekly(range(8)).assign(unit='b')
    _, test, _ = split_train_test(pd.concat([unit_a, unit_b]), test_weeks=3)
    assert len(test[test['unit'] == 'b']) == 1


def test_split_train_test_with_too_few_weeks_raises():
    """学習の週が残らないほどテストを長くしたら止まる"""
    with pytest.raises(ValueError):
        split_train_test(_weekly(range(5)), test_weeks=5)


def test_split_train_test_with_zero_test_weeks_raises():
    """テストが0週なら止まる"""
    with pytest.raises(ValueError):
        split_train_test(_weekly(range(5)), test_weeks=0)


# ---------------------------------------------------------------- 発売の選び方

# 発売の選び方のテストで使う期間。起点（MONDAY）からの週数で、データ期間は4週目から、切る週は40週目
FIRST_WEEK = MONDAY + pd.Timedelta(weeks=4)
CUTOFF = MONDAY + pd.Timedelta(weeks=40)

# select_launch_events が返す表の列
EVENT_COLUMNS = ['unit', 'game', 'release_week', 'game_mentions', 'unit_mentions', 'share']


def _games(release_week_by_game):
    """ゲーム名 → 発売週（起点からの週数）から、ゲーム台帳（name / release_date）を作る

    発売日は、その週の水曜にする（発売週は月曜に直されることの確認にもなる）。
    """
    dates = [MONDAY + pd.Timedelta(weeks=week, days=2) for week in release_week_by_game.values()]
    return pd.DataFrame({'name': list(release_week_by_game),
                         'release_date': [date.strftime('%Y-%m-%d') for date in dates]})


def _mentions(game, topic, first_week, counts):
    """ゲームのレビュー（game_name / week / topic_id）を作る。first_week 週目から、1週ごとの件数 counts で並べる"""
    weeks = [MONDAY + pd.Timedelta(weeks=first_week + i)
             for i, count in enumerate(counts) for _ in range(count)]
    return pd.DataFrame({'game_name': game, 'week': pd.DatetimeIndex(weeks), 'topic_id': topic})


def _launch_reviews(alpha_counts, old_counts=(), first_week=10):
    """Alpha（first_week 週目に発売）と、ずっと前に発売した Old の、同じトピック（1）のレビュー"""
    return pd.concat([_mentions('Alpha', 1, first_week, alpha_counts),
                      _mentions('Old', 1, first_week, old_counts)], ignore_index=True)


def _select_events(reviews, games, units=(1, 2), **options):
    """テスト用の期間（FIRST_WEEK 〜 CUTOFF）で、発売を選ぶ"""
    return select_launch_events(reviews, games, units, FIRST_WEEK, CUTOFF, **options)


@pytest.fixture
def games():
    """10週目に発売した Alpha と、ずっと前（-100週目）に発売した Old の台帳"""
    return _games({'Alpha': 10, 'Old': -100})


def test_find_target_launches_includes_launch_whose_last_week_is_first_week():
    """発売週から8週のうち最後の週が、データ期間の最初の週にちょうど当たる発売は入る

    期間の直前に出たゲーム（Starfield など）も、発売後の山が期間の中に入るため。
    """
    targets = find_target_launches(_games({'Alpha': -3}), FIRST_WEEK, CUTOFF)
    assert targets['game'].tolist() == ['Alpha']


def test_find_target_launches_excludes_launch_ended_before_first_week():
    """発売週から8週が、データ期間の最初の週より前に終わる発売は入らない"""
    targets = find_target_launches(_games({'Alpha': -4}), FIRST_WEEK, CUTOFF)
    assert targets.empty


def test_find_target_launches_includes_launch_week_before_cutoff():
    """発売週が切る週の1週前の発売は入る（打ち切られて、期間は1週になる）"""
    targets = find_target_launches(_games({'Alpha': 39}), FIRST_WEEK, CUTOFF)
    assert targets['game'].tolist() == ['Alpha']


@pytest.mark.parametrize('release_week', [40, 41])
def test_find_target_launches_excludes_launch_at_or_after_cutoff(release_week):
    """発売週が切る週と同じか、それ以降の発売は入らない（テスト期間のレビューを使わないため）"""
    targets = find_target_launches(_games({'Alpha': release_week}), FIRST_WEEK, CUTOFF)
    assert targets.empty


def test_find_target_launches_launch_weeks_changes_range():
    """週数を4週にすると、8週なら入る発売（-3週目の発売。8週目の最後が期間の最初の週）は入らない"""
    targets = find_target_launches(_games({'Alpha': -3}), FIRST_WEEK, CUTOFF, launch_weeks=4)
    assert targets.empty


def test_find_target_launches_gives_monday_of_release_date():
    """発売週は、発売日（水曜）を含む週の月曜"""
    targets = find_target_launches(_games({'Alpha': 10}), FIRST_WEEK, CUTOFF)
    assert targets['release_week'].tolist() == [MONDAY + pd.Timedelta(weeks=10)]


@pytest.mark.parametrize('bad_date', ['TBA', None])
def test_find_target_launches_with_unreadable_release_date_raises(bad_date):
    """発売日が読めないゲームがあれば止まる（黙って含めたり外したりしない）"""
    ledger = _games({'Alpha': 10}).assign(release_date=bad_date)
    with pytest.raises(ValueError):
        find_target_launches(ledger, FIRST_WEEK, CUTOFF)


def test_find_target_launches_with_zero_launch_weeks_raises():
    """週数が0なら止まる"""
    with pytest.raises(ValueError):
        find_target_launches(_games({'Alpha': 10}), FIRST_WEEK, CUTOFF, launch_weeks=0)


def test_select_launch_events_returns_documented_columns(games):
    """返す表の列は unit / game / release_week / game_mentions / unit_mentions / share"""
    events = _select_events(_launch_reviews([10] * 8), games)
    assert events.columns.tolist() == EVENT_COLUMNS


def test_select_launch_events_counts_mentions_in_launch_period(games):
    """確かめる期間の中で、そのゲームのレビュー数と、その単位の全レビュー数を数える

    Alpha 100件（20・20・20・10・10・10・5・5）と、Old 20件で、単位の全体は120件。
    """
    reviews = _launch_reviews([20, 20, 20, 10, 10, 10, 5, 5], old_counts=[20])
    events = _select_events(reviews, games)
    assert events[['game_mentions', 'unit_mentions']].values.tolist() == [[100, 120]]


def test_select_launch_events_share_is_game_mentions_over_unit_mentions(games):
    """share は、その単位のレビューのうちそのゲームが占める割合（100 ÷ 120）"""
    reviews = _launch_reviews([20, 20, 20, 10, 10, 10, 5, 5], old_counts=[20])
    assert _select_events(reviews, games)['share'].tolist() == pytest.approx([100 / 120])


def test_select_launch_events_gives_unit_game_and_release_week(games):
    """単位・ゲーム・発売週（発売日を含む週の月曜）が付く"""
    events = _select_events(_launch_reviews([10] * 8), games)
    assert events[['unit', 'game', 'release_week']].values.tolist() == [
        [1, 'Alpha', MONDAY + pd.Timedelta(weeks=10)]]


def test_select_launch_events_keeps_pair_when_share_is_exactly_half(games):
    """ゲームが占める割合がちょうど0.5の組は選ぶ（過半数 = 0.5以上）"""
    events = _select_events(_launch_reviews([10] * 8, old_counts=[10] * 8), games)
    assert events['game'].tolist() == ['Alpha']


def test_select_launch_events_drops_pair_when_share_is_below_half(games):
    """割合が0.5を少しでも下回る組（80 ÷ 161）は選ばない"""
    events = _select_events(_launch_reviews([10] * 8, old_counts=[10] * 7 + [11]), games)
    assert events.empty


def test_select_launch_events_min_share_argument_changes_threshold(games):
    """割合の下限は引数で変えられる（割合0.5の組が、下限0.6なら外れる）"""
    reviews = _launch_reviews([10] * 8, old_counts=[10] * 8)
    assert _select_events(reviews, games, min_share=0.6).empty


def test_select_launch_events_keeps_pair_with_exactly_minimum_weekly_mentions(games):
    """そのゲームのレビューがちょうど週10件 × 8週 = 80件の組は選ぶ"""
    events = _select_events(_launch_reviews([10] * 8), games)
    assert events['game_mentions'].tolist() == [80]


def test_select_launch_events_drops_pair_below_minimum_weekly_mentions(games):
    """割合が100%でも、レビューが79件（週10件 × 8週に1件足りない）の組は選ばない"""
    events = _select_events(_launch_reviews([10] * 7 + [9]), games)
    assert events.empty


def test_select_launch_events_min_weekly_argument_changes_threshold(games):
    """週あたりの下限は引数で変えられる（79件の組が、週5件なら選ばれる）"""
    events = _select_events(_launch_reviews([10] * 7 + [9]), games, min_weekly_mentions=5)
    assert events['game_mentions'].tolist() == [79]


def test_select_launch_events_launch_weeks_argument_changes_period(games):
    """確かめる期間の週数は引数で変えられる（発売の5〜8週目だけに集まった言及は、4週では数えない）"""
    reviews = _launch_reviews([0, 0, 0, 0, 20, 20, 20, 20])
    assert _select_events(reviews, games, launch_weeks=4).empty


def test_select_launch_events_does_not_count_reviews_from_cutoff_on():
    """切る週以降のレビューは数えない

    37週目に発売した Alpha の期間は、切る週（40週目）の前の3週で打ち切られる。
    テスト期間（40週目）に Old のレビューが500件あっても、数えると割合が下がって外れるが、
    数えないので 30 ÷ 30 のまま選ばれる。
    """
    reviews = pd.concat([_mentions('Alpha', 1, 37, [10] * 8), _mentions('Old', 1, 40, [500])],
                        ignore_index=True)
    events = _select_events(reviews, _games({'Alpha': 37, 'Old': -100}))
    assert events[['game_mentions', 'unit_mentions']].values.tolist() == [[30, 30]]


def test_select_launch_events_scales_weekly_minimum_down_when_period_is_truncated():
    """打ち切りで3週になった発売は、週10件 × 3週 = 30件で足りる（8週ぶんの80件は要らない）"""
    events = _select_events(_mentions('Alpha', 1, 37, [10] * 3), _games({'Alpha': 37}))
    assert events['game'].tolist() == ['Alpha']


def test_select_launch_events_drops_pair_below_scaled_weekly_minimum_when_truncated():
    """打ち切りで3週になった発売は、29件（週10件 × 3週に1件足りない）なら選ばない"""
    events = _select_events(_mentions('Alpha', 1, 37, [10, 10, 9]), _games({'Alpha': 37}))
    assert events.empty


def test_select_launch_events_counts_reviews_before_first_week_for_prior_launch():
    """期間の直前に出た発売は入り、期間の最初の週より前のレビューも数える

    -3週目に発売した Alpha のレビュー80件のうち、7週ぶんはデータ期間（4週目から）より前。
    """
    reviews = _mentions('Alpha', 1, -3, [10] * 8)
    events = _select_events(reviews, _games({'Alpha': -3}))
    assert events['game_mentions'].tolist() == [80]


def test_select_launch_events_excludes_launch_at_cutoff():
    """切る週に発売したゲームは、レビューが多くても入らない"""
    reviews = _mentions('Alpha', 1, 40, [100] * 8)
    assert _select_events(reviews, _games({'Alpha': 40})).empty


def test_select_launch_events_ignores_topics_outside_units(games):
    """単位の集合に無いトピック（外れ値の -1 も含む）のレビューは無視する"""
    reviews = pd.concat([_mentions('Alpha', 1, 10, [10] * 8), _mentions('Alpha', 2, 10, [10] * 8),
                         _mentions('Alpha', -1, 10, [10] * 8)], ignore_index=True)
    assert _select_events(reviews, games, units=[1])['unit'].tolist() == [1]


def test_select_launch_events_gives_each_launch_of_a_unit_its_own_row():
    """同じ単位に別々の時期の発売が2つあれば、2行になる（発売週の順）"""
    reviews = pd.concat([_mentions('Beta', 1, 25, [10] * 8), _mentions('Alpha', 1, 10, [10] * 8)],
                        ignore_index=True)
    events = _select_events(reviews, _games({'Alpha': 10, 'Beta': 25}))
    assert events['game'].tolist() == ['Alpha', 'Beta']


def test_select_launch_events_without_pairs_returns_empty_table_with_columns(games):
    """組が1つも無ければ、列だけを持つ空の表を返す"""
    events = _select_events(_launch_reviews([1]), games)
    assert (events.columns.tolist(), len(events)) == (EVENT_COLUMNS, 0)


def test_select_launch_events_with_unreadable_release_date_raises():
    """台帳に発売日が読めないゲームがあれば止まる"""
    ledger = _games({'Alpha': 10, 'Old': -100}).assign(release_date=['2024-03-06', 'unknown'])
    with pytest.raises(ValueError):
        _select_events(_launch_reviews([10] * 8), ledger)


# ---------------------------------------------------------------- Prophet に渡す発売（holidays）

@pytest.fixture
def events():
    """単位1に2つ（Alpha・Beta）、単位2に1つ（Gamma）の発売を持つ、選んだ組の表"""
    return pd.DataFrame({
        'unit': [1, 1, 2], 'game': ['Alpha', 'Beta', 'Gamma'],
        'release_week': [MONDAY + pd.Timedelta(weeks=week) for week in (10, 25, 30)],
        'game_mentions': [100, 90, 80], 'unit_mentions': [120, 100, 100],
        'share': [100 / 120, 0.9, 0.8]})


def test_launch_holidays_names_each_launch_by_game(events):
    """発売ごとに別の出来事にする（holiday = ゲーム名）"""
    assert launch_holidays(events, 1)['holiday'].tolist() == ['Alpha', 'Beta']


def test_launch_holidays_ds_is_release_week(events):
    """ds は発売週"""
    assert launch_holidays(events, 1)['ds'].tolist() == [
        MONDAY + pd.Timedelta(weeks=10), MONDAY + pd.Timedelta(weeks=25)]


def test_launch_holidays_lower_window_is_zero(events):
    """出来事は発売週から始まる（発売より前には効かせない）"""
    assert (launch_holidays(events, 1)['lower_window'] == 0).all()


@pytest.mark.parametrize('launch_weeks, upper_window', [(8, 49), (4, 21), (1, 0)])
def test_launch_holidays_upper_window_reaches_last_launch_week(events, launch_weeks, upper_window):
    """upper_window は 7 × (週数 − 1) 日。週次データなので、発売週から週数ぶんの週に効く"""
    holidays = launch_holidays(events, 1, launch_weeks=launch_weeks)
    assert (holidays['upper_window'] == upper_window).all()


def test_launch_holidays_takes_only_rows_of_given_unit(events):
    """渡された単位の発売だけを出来事にする"""
    assert launch_holidays(events, 2)['holiday'].tolist() == ['Gamma']


def test_launch_holidays_for_unit_without_launch_is_none(events):
    """発売が付かない単位は None（Prophet に holidays を渡さない）"""
    assert launch_holidays(events, 3) is None


def test_launch_holidays_with_empty_events_is_none():
    """選んだ組が1つも無い（空の表）ときも None"""
    assert launch_holidays(pd.DataFrame(columns=EVENT_COLUMNS), 1) is None


# ---------------------------------------------------------------- 比べる相手

def test_baseline_mean_repeats_train_mean():
    """学習期間の平均を、テスト週の数だけ並べる"""
    result = baseline_mean(_weekly([1, 2, 3, 6]), horizon=3)
    assert result.tolist() == [3.0, 3.0, 3.0]


def test_baseline_mean_skips_missing_values():
    """欠測を除いて平均する"""
    assert baseline_mean(_weekly([1, np.nan, 3]), horizon=1).tolist() == [2.0]


def test_baseline_mean_without_values_raises():
    """実績が1つも無ければ止まる"""
    with pytest.raises(ValueError):
        baseline_mean(_weekly([np.nan, np.nan]), horizon=1)


def test_baseline_recent_averages_last_four_weeks_by_default():
    """既定では、学習期間の最後の4週の平均になる"""
    result = baseline_recent(_weekly(range(1, 11)), horizon=2)
    assert result.tolist() == [8.5, 8.5]


def test_baseline_recent_weeks_argument_changes_window():
    """週数は引数で変えられる"""
    result = baseline_recent(_weekly(range(1, 11)), horizon=1, recent_weeks=2)
    assert result.tolist() == [9.5]


def test_baseline_recent_skips_missing_before_taking_last_weeks():
    """欠測を除いた最後の4つで平均する（欠測の週を数に入れない）

    欠測を除くと [1, 2, 3, 4, 5, 7]。最後の4つは [3, 4, 5, 7] で平均4.75。
    先に末尾4行を取ってから欠測を除くと、[5, 7] の平均6.0になってしまう。
    """
    values = [1, 2, 3, 4, 5, np.nan, 7, np.nan]
    assert baseline_recent(_weekly(values), horizon=1).tolist() == [4.75]


def test_baseline_recent_uses_latest_weeks_even_if_rows_are_shuffled():
    """行が週の順に並んでいなくても、時間で最後の週を使う"""
    shuffled = _weekly(range(1, 11)).sample(frac=1, random_state=0)
    assert baseline_recent(shuffled, horizon=1).tolist() == [8.5]


def test_baseline_recent_with_zero_weeks_raises():
    """週数が0なら止まる"""
    with pytest.raises(ValueError):
        baseline_recent(_weekly([1, 2, 3]), horizon=1, recent_weeks=0)


def _holidays(*launches, upper_window=14):
    """(ゲーム名, 起点からの週数) の並びから、出来事の表を作る。窓は既定で3週（発売週 + 14日）"""
    return pd.DataFrame({
        'holiday': [name for name, _ in launches],
        'ds': [MONDAY + pd.Timedelta(weeks=week) for _, week in launches],
        'lower_window': 0, 'upper_window': upper_window})


def test_drop_holiday_weeks_removes_weeks_from_release_week_through_window():
    """発売週から窓（14日 = 3週）ぶんの週を除く。窓の次の週は残す

    2週目に発売なら、2・3・4週目を除いて、0・1・5週目以降が残る。
    """
    remaining = drop_holiday_weeks(_weekly(range(12)), _holidays(('Alpha', 2)))
    assert remaining['share'].tolist() == [0, 1, 5, 6, 7, 8, 9, 10, 11]


def test_drop_holiday_weeks_without_holidays_returns_train_as_is():
    """出来事が無い（None）ときは、何も除かない"""
    train = _weekly(range(12))
    assert drop_holiday_weeks(train, None) is train


def test_drop_holiday_weeks_removes_only_overlapping_weeks_of_launch_before_train():
    """学習期間より前に始まった発売は、学習期間と重なる週だけ除く

    -1週目に発売なら窓は -1・0・1週目。学習期間にあるのは 0・1週目の2週だけ。
    """
    remaining = drop_holiday_weeks(_weekly(range(12)), _holidays(('Alpha', -1)))
    assert remaining['share'].tolist() == list(range(2, 12))


def test_drop_holiday_weeks_removes_weeks_of_every_launch():
    """発売が複数あれば、それぞれの窓の週をすべて除く（2〜4週目と、8〜10週目）"""
    remaining = drop_holiday_weeks(_weekly(range(12)), _holidays(('Alpha', 2), ('Beta', 8)))
    assert remaining['share'].tolist() == [0, 1, 5, 6, 7, 11]


def test_drop_holiday_weeks_overlapping_launches_remove_each_week_once():
    """窓が重なる発売があっても、重なった週は1回除くだけで、ほかの週は残る"""
    remaining = drop_holiday_weeks(_weekly(range(12)), _holidays(('Alpha', 2), ('Beta', 3)))
    assert remaining['share'].tolist() == [0, 1, 6, 7, 8, 9, 10, 11]


def test_drop_holiday_weeks_with_everything_removed_raises():
    """除いた結果、実績が1つも残らなければ止まる"""
    with pytest.raises(ValueError):
        drop_holiday_weeks(_weekly(range(3)), _holidays(('Alpha', 0)))


# ---------------------------------------------------------------- MAE と比

def test_mae_is_mean_absolute_difference():
    """MAE は、予測と実績の差の絶対値の平均（差は 1, 0, 2 → 平均1）"""
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_mae_with_different_lengths_raises():
    """長さが違えば止まる"""
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])


def test_mae_ratio_divides_prophet_by_baseline():
    """比 = Prophet の MAE ÷ 比べる相手の MAE。1未満なら Prophet の勝ち"""
    assert mae_ratio(0.5, 2.0) == pytest.approx(0.25)


def test_mae_ratio_with_zero_baseline_is_infinite():
    """比べる相手の MAE が0で Prophet が外していれば、割れないので inf（Prophet の負け）"""
    assert mae_ratio(0.5, 0.0) == float('inf')


def test_mae_ratio_with_both_zero_is_nan():
    """どちらも0なら比は決まらないので nan"""
    assert np.isnan(mae_ratio(0.0, 0.0))


def test_comparisons_pair_each_prophet_with_each_baseline():
    """比べるのは Prophet 2つ × 比べる相手 2つの4通り"""
    assert set(COMPARISONS) == {
        ('prophet_yearly', 'baseline_mean'), ('prophet_yearly', 'baseline_recent'),
        ('prophet_no_yearly', 'baseline_mean'), ('prophet_no_yearly', 'baseline_recent'),
    }


# ---------------------------------------------------------------- Prophet

def test_forecast_prophet_with_seasonal_data_returns_finite_values():
    """学習から予測まで回り、有限の値が返る"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    result = forecast_prophet(train, test['week'], yearly=True)
    assert np.isfinite(result).all()


@pytest.mark.parametrize('yearly', [True, False])
def test_forecast_prophet_returns_one_value_per_test_week(yearly):
    """年次季節性あり・なしのどちらでも、テスト週と同じ長さの予測が返る"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    assert len(forecast_prophet(train, test['week'], yearly=yearly)) == len(test)


def test_forecast_prophet_yearly_switch_changes_forecast():
    """年次季節性の有無で、予測が変わる（引数が学習に届いている）"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    with_yearly = forecast_prophet(train, test['week'], yearly=True)
    without_yearly = forecast_prophet(train, test['week'], yearly=False)
    assert not np.allclose(with_yearly, without_yearly)


def test_forecast_prophet_yearly_follows_seasonal_pattern():
    """年次の波があるデータでは、年次季節性ありのほうが波に沿って当たる

    テスト期間は波の山から下りに入る。年次季節性なしは直前の上り坂を延ばして外れる。
    あり・なしの指定が逆に渡っていないことの確認。
    """
    train, test = _seasonal(117), _seasonal(8, first_week=117)
    actual = test['share'].to_numpy()
    error_with = mae(actual, forecast_prophet(train, test['week'], yearly=True))
    error_without = mae(actual, forecast_prophet(train, test['week'], yearly=False))
    assert error_with < error_without


def test_forecast_prophet_with_unsorted_weeks_raises():
    """予測する週が昇順でなければ止まる（順がずれて、黙って違う週に当てはまるのを防ぐ）"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    with pytest.raises(ValueError):
        forecast_prophet(train, test['week'].iloc[::-1], yearly=False)


@pytest.fixture
def stalling_prophet(monkeypatch):
    """1回目の学習は時間切れになり、Newton 法を指定した2回目で成功する偽の Prophet に差し替える

    本物の Stan が終わらなくなる学習は再現できないので、時間切れ（TimeoutError）だけ真似る。
    学習のたびに渡された引数を calls に残す。
    """
    calls = []

    class Stub:
        def __init__(self, **kwargs):
            pass

        def fit(self, history, **options):
            calls.append(options)
            if options.get('algorithm') != 'Newton':
                raise TimeoutError
            return self

        def predict(self, future):
            return pd.DataFrame({'ds': future['ds'], 'yhat': np.full(len(future), 0.5)})

    monkeypatch.setattr(forecast, 'Prophet', Stub)
    return calls


def test_forecast_prophet_timeout_refits_with_newton(stalling_prophet):
    """学習が時間切れになったら、Newton 法で学び直す"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    with pytest.warns(FitFallbackWarning):
        forecast_prophet(train, test['week'], yearly=False)
    assert [options.get('algorithm') for options in stalling_prophet] == [None, 'Newton']


def test_forecast_prophet_timeout_returns_refit_forecast(stalling_prophet):
    """時間切れのあとは、学び直したモデルの予測が返る"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    with pytest.warns(FitFallbackWarning):
        result = forecast_prophet(train, test['week'], yearly=False)
    assert result.tolist() == [0.5] * 8


def test_forecast_prophet_real_timeout_falls_back_to_newton():
    """本物の Stan でも、時間切れから学び直しに切り替わって、予測が返る

    上限を 0.001 秒にして、必ず時間切れにする（Stan の起動だけでもそれ以上かかる）。
    """
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    with pytest.warns(FitFallbackWarning):
        result = forecast_prophet(train, test['week'], yearly=False, fit_timeout=0.001)
    assert len(result) == len(test)


@pytest.fixture
def recorded_prophet(monkeypatch):
    """Prophet を、作るときに渡された引数を記録する偽物に差し替える"""
    created = []

    class Stub:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def fit(self, history, **options):
            return self

        def predict(self, future):
            return pd.DataFrame({'ds': future['ds'], 'yhat': np.zeros(len(future))})

    monkeypatch.setattr(forecast, 'Prophet', Stub)
    return created


def test_forecast_prophet_passes_holidays_to_prophet(recorded_prophet):
    """渡した出来事の表が、Prophet にそのまま届く"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    holidays = _holidays(('Alpha', 10))
    forecast_prophet(train, test['week'], yearly=True, holidays=holidays)
    assert recorded_prophet[0]['holidays'] is holidays


def test_forecast_prophet_without_holidays_passes_none_to_prophet(recorded_prophet):
    """出来事を渡さなければ、Prophet にも holidays を渡さない（None）"""
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    forecast_prophet(train, test['week'], yearly=True)
    assert recorded_prophet[0]['holidays'] is None


def test_forecast_prophet_timeout_refit_keeps_holidays(monkeypatch):
    """学習が時間切れになって Newton 法で学び直すときも、同じ出来事の表を渡す"""
    created = []

    class Stub:
        def __init__(self, **kwargs):
            created.append(kwargs['holidays'])

        def fit(self, history, **options):
            if options.get('algorithm') != 'Newton':
                raise TimeoutError
            return self

        def predict(self, future):
            return pd.DataFrame({'ds': future['ds'], 'yhat': np.zeros(len(future))})

    monkeypatch.setattr(forecast, 'Prophet', Stub)
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    holidays = _holidays(('Alpha', 10))
    with pytest.warns(FitFallbackWarning):
        forecast_prophet(train, test['week'], yearly=False, holidays=holidays)
    assert [item is holidays for item in created] == [True, True]


def test_forecast_prophet_without_holidays_matches_plain_prophet():
    """出来事を渡さない予測は、年次季節性以外すべて既定値の素の Prophet とまったく同じ

    発売が付かない単位の結果が、発売を渡せるようにする前と変わらないことの確認。
    """
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    plain = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    plain.fit(pd.DataFrame({'ds': train['week'], 'y': train['share']}),
              timeout=forecast.FIT_TIMEOUT_SECONDS)
    expected = plain.predict(pd.DataFrame({'ds': test['week']}))['yhat'].to_numpy()
    assert forecast_prophet(train, test['week'], yearly=True).tolist() == expected.tolist()


def test_forecast_prophet_with_holidays_returns_finite_values():
    """出来事を渡して本物の Prophet で学習しても、有限の予測が返る

    出来事は、学習期間より前に始まるもの・学習期間の真ん中にあるもの・学習期間の終わりをまたぐもの。
    """
    train, test = _seasonal(110), _seasonal(8, first_week=110)
    holidays = _holidays(('Before', -3), ('Middle', 60), ('Straddle', 108), upper_window=49)
    result = forecast_prophet(train, test['week'], yearly=True, holidays=holidays)
    assert np.isfinite(result).all()


# 1回きりの発売の山。発売週から8週に、水準へ足す量（だんだん減る）
LAUNCH_BUMP = [0.06, 0.045, 0.03, 0.02, 0.012, 0.008, 0.004, 0.002]


def test_forecast_prophet_launch_holiday_weakens_yearly_replay_of_one_off_bump():
    """発売を渡すと、1回きりの発売の山が、1年後の山として再生されにくくなる（年次季節性あり）

    水準0.02で一定の系列の70週目に発売の山があり、110週ぶんを学習する。発売を渡さないと、
    年次季節性が山を毎年の山と覚えて、1年後（122週目から8週）に水準より約0.014高く予測する。
    渡すと山は発売のせいと学ばれ、1年後の予測はほぼ水準のままになる。
    """
    values = np.full(136, 0.02)
    values[70:78] += LAUNCH_BUMP
    series = _weekly(values)
    train, test = series.iloc[:110], series.iloc[110:]
    events = pd.DataFrame({'unit': [1], 'game': ['Alpha'],
                           'release_week': [MONDAY + pd.Timedelta(weeks=70)]})

    anniversary = slice(12, 20)    # テスト期間のうち、発売の1年後（70 + 52 = 122 週目）から8週
    without = forecast_prophet(train, test['week'], yearly=True)[anniversary]
    with_launch = forecast_prophet(train, test['week'], yearly=True,
                                   holidays=launch_holidays(events, 1))[anniversary]
    assert (with_launch - 0.02).mean() < 0.5 * (without - 0.02).mean()


# ---------------------------------------------------------------- 1単位の評価

@pytest.fixture
def fake_prophet(monkeypatch):
    """Prophet を、年次季節性ありなら常に 0.3、なしなら常に -0.1 を予測する偽物に差し替える

    MAE と比の数値を厳密に確かめるため。-0.1 は負の予測がそのまま残ることの確認にも使う。
    """
    def fake(train, weeks, yearly, **kwargs):
        return np.full(len(weeks), 0.3 if yearly else -0.1)

    monkeypatch.setattr(forecast, 'forecast_prophet', fake)


@pytest.fixture
def unit_data():
    """学習6週（平均0.35・最後の2週の平均0.55）とテスト2週（0.5, 0.7）"""
    return _weekly([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), _weekly([0.5, 0.7], first_week=6)


def test_evaluate_unit_returns_mae_of_four_methods(fake_prophet, unit_data):
    """4つの方法それぞれの MAE を返す"""
    metrics, _ = evaluate_unit(*unit_data, recent_weeks=2)
    assert {key: value for key, value in metrics.items() if key.startswith('mae_')} == (
        pytest.approx({'mae_prophet_yearly': 0.3, 'mae_prophet_no_yearly': 0.7,
                       'mae_baseline_mean': 0.25, 'mae_baseline_recent': 0.1}))


def test_evaluate_unit_returns_ratio_for_four_comparisons(fake_prophet, unit_data):
    """Prophet 2つ × 比べる相手 2つの4通りで、MAE(Prophet) ÷ MAE(比べる相手) を返す"""
    metrics, _ = evaluate_unit(*unit_data, recent_weeks=2)
    assert {key: value for key, value in metrics.items() if key.startswith('ratio_')} == (
        pytest.approx({'ratio_prophet_yearly_vs_baseline_mean': 1.2,
                       'ratio_prophet_yearly_vs_baseline_recent': 3.0,
                       'ratio_prophet_no_yearly_vs_baseline_mean': 2.8,
                       'ratio_prophet_no_yearly_vs_baseline_recent': 7.0}))


def test_evaluate_unit_passes_recent_weeks_to_baseline(fake_prophet, unit_data):
    """直近の平均の週数は引数で変えられる（最後の2週なら0.55、4週なら0.45）"""
    _, predictions = evaluate_unit(*unit_data, recent_weeks=4)
    assert predictions['baseline_recent'].tolist() == pytest.approx([0.45, 0.45])


def test_evaluate_unit_returns_predictions_by_test_week(fake_prophet, unit_data):
    """予測は、テスト週ごとに実績と4つの予測が並ぶ"""
    _, predictions = evaluate_unit(*unit_data)
    assert predictions.columns.tolist() == ['week', 'actual', *METHODS]


def test_evaluate_unit_keeps_negative_forecast_as_is(fake_prophet, unit_data):
    """Prophet の予測が負でも、0に切り上げない"""
    _, predictions = evaluate_unit(*unit_data)
    assert (predictions['prophet_no_yearly'] == -0.1).all()


def test_evaluate_unit_without_test_values_raises(fake_prophet, unit_data):
    """テスト期間に実績が無ければ止まる"""
    train, test = unit_data
    with pytest.raises(ValueError):
        evaluate_unit(train, test.iloc[:0])


@pytest.fixture(scope='module')
def prophet_evaluation():
    """本物の Prophet で、合成データ1単位（学習110週・テスト8週）を評価した結果"""
    return evaluate_unit(_seasonal(110), _seasonal(8, first_week=110))


def test_evaluate_unit_with_prophet_returns_finite_metrics(prophet_evaluation):
    """本物の Prophet でも、8つの指標がすべて有限の値で返る"""
    metrics, _ = prophet_evaluation
    assert np.isfinite(list(metrics.values())).all()


def test_evaluate_unit_with_prophet_returns_one_row_per_test_week(prophet_evaluation):
    """本物の Prophet でも、予測はテスト週の数だけ並ぶ"""
    _, predictions = prophet_evaluation
    assert len(predictions) == 8


@pytest.fixture
def recorded_fake_prophet(monkeypatch):
    """Prophet を常に 0.3 を予測する偽物に差し替え、呼ばれたときの引数（学習期間・holidays）を記録する"""
    calls = []

    def fake(train, weeks, yearly, **kwargs):
        calls.append({'train': train, **kwargs})
        return np.full(len(weeks), 0.3)

    monkeypatch.setattr(forecast, 'forecast_prophet', fake)
    return calls


def test_evaluate_unit_passes_holidays_to_both_prophet_fits(recorded_fake_prophet, unit_data):
    """年次季節性あり・なしの両方の学習に、同じ出来事の表を渡す"""
    holidays = _holidays(('Alpha', 4))
    evaluate_unit(*unit_data, holidays=holidays)
    assert [call['holidays'] is holidays for call in recorded_fake_prophet] == [True, True]


def test_evaluate_unit_gives_prophet_whole_train_even_with_holidays(recorded_fake_prophet,
                                                                    unit_data):
    """Prophet には、発売後の週を除かない学習期間をそのまま渡す（除くのは比べる相手だけ）"""
    evaluate_unit(*unit_data, holidays=_holidays(('Alpha', 4)))
    assert [len(call['train']) for call in recorded_fake_prophet] == [6, 6]


def test_evaluate_unit_without_holidays_passes_no_holidays_to_prophet(recorded_fake_prophet,
                                                                      unit_data):
    """出来事を渡さなければ、Prophet にも渡さない（None）。発売が付かない単位がこの動きになる"""
    evaluate_unit(*unit_data)
    assert [call['holidays'] for call in recorded_fake_prophet] == [None, None]


def test_evaluate_unit_baseline_mean_excludes_launch_weeks(fake_prophet, unit_data):
    """比べる相手①の平均は、発売後の週を除いた学習期間で出す

    学習は 0.1〜0.6 の6週。4週目に発売で窓が2週なら、4・5週目（0.5・0.6）を除いて平均は 0.25
    （除かなければ 0.35）。
    """
    _, predictions = evaluate_unit(*unit_data, holidays=_holidays(('Alpha', 4), upper_window=7))
    assert predictions['baseline_mean'].tolist() == pytest.approx([0.25, 0.25])


def test_evaluate_unit_baseline_recent_excludes_launch_weeks(fake_prophet, unit_data):
    """比べる相手②の直近の平均は、発売後の週を除いたあとの、最後の recent_weeks 個で出す

    4・5週目を除くと残りは 0.1〜0.4。最後の2個（0.3・0.4）の平均は 0.35
    （除かなければ最後の2個は 0.5・0.6 で 0.55）。
    """
    _, predictions = evaluate_unit(*unit_data, recent_weeks=2,
                                   holidays=_holidays(('Alpha', 4), upper_window=7))
    assert predictions['baseline_recent'].tolist() == pytest.approx([0.35, 0.35])


def test_evaluate_unit_with_holidays_removing_all_train_raises(fake_prophet, unit_data):
    """発売後の週を除くと学習期間に実績が1つも残らないなら止まる"""
    with pytest.raises(ValueError):
        evaluate_unit(*unit_data, holidays=_holidays(('Alpha', 0), upper_window=35))


def test_evaluate_unit_with_prophet_and_holidays_returns_finite_metrics():
    """本物の Prophet に発売を渡しても、8つの指標がすべて有限の値で返る"""
    metrics, _ = evaluate_unit(_seasonal(110), _seasonal(8, first_week=110),
                               holidays=_holidays(('Alpha', 60), upper_window=49))
    assert np.isfinite(list(metrics.values())).all()


# ---------------------------------------------------------------- まとめ

def _ratios(**ratio_by_comparison):
    """4通りの比の列を持つ評価表を作る（1行1単位）。引数の名前は 'prophet_yearly__baseline_mean' の形"""
    columns = {f"ratio_{key.replace('__', '_vs_')}": values
               for key, values in ratio_by_comparison.items()}
    return pd.DataFrame(columns)


@pytest.fixture
def four_unit_metrics():
    """4単位ぶんの評価表。比が 1 未満・ちょうど1・1超・欠測 が混ざる"""
    return _ratios(
        prophet_yearly__baseline_mean=[0.5, 0.9, 1.0, 1.5],
        prophet_yearly__baseline_recent=[1.0, 1.0, 1.0, 1.0],
        prophet_no_yearly__baseline_mean=[0.1, 0.2, 0.3, 0.4],
        prophet_no_yearly__baseline_recent=[2.0, 3.0, 4.0, np.nan],
    )


def test_summarize_comparisons_has_four_rows_in_comparison_order(four_unit_metrics):
    """4通りの比較が、COMPARISONS の順に1行ずつ並ぶ"""
    summary = summarize_comparisons(four_unit_metrics)
    assert list(zip(summary['prophet'], summary['baseline'])) == list(COMPARISONS)


def test_summarize_comparisons_counts_wins_when_ratio_is_below_one(four_unit_metrics):
    """比が1未満の単位を、勝ちと数える"""
    assert summarize_comparisons(four_unit_metrics)['wins'].tolist() == [2, 0, 4, 0]


def test_summarize_comparisons_does_not_count_ratio_of_exactly_one_as_win():
    """比がちょうど1は、勝ちにしない"""
    metrics = _ratios(prophet_yearly__baseline_mean=[1.0, 1.0],
                      prophet_yearly__baseline_recent=[1.0, 1.0],
                      prophet_no_yearly__baseline_mean=[1.0, 1.0],
                      prophet_no_yearly__baseline_recent=[1.0, 1.0])
    assert summarize_comparisons(metrics)['wins'].tolist() == [0, 0, 0, 0]


def test_summarize_comparisons_does_not_count_missing_ratio_as_win():
    """比が欠測の単位は、勝ちにしない"""
    metrics = _ratios(prophet_yearly__baseline_mean=[0.5, np.nan],
                      prophet_yearly__baseline_recent=[0.5, np.nan],
                      prophet_no_yearly__baseline_mean=[0.5, np.nan],
                      prophet_no_yearly__baseline_recent=[0.5, np.nan])
    assert summarize_comparisons(metrics)['wins'].tolist() == [1, 1, 1, 1]


def test_summarize_comparisons_units_counts_all_units(four_unit_metrics):
    """全単位数は、比が欠測の単位も含めた単位の数"""
    assert summarize_comparisons(four_unit_metrics)['units'].tolist() == [4, 4, 4, 4]


def test_summarize_comparisons_gives_median_ratio(four_unit_metrics):
    """比の中央値を出す（欠測は除く）"""
    medians = summarize_comparisons(four_unit_metrics)['median_ratio']
    assert medians.tolist() == pytest.approx([0.95, 1.0, 0.25, 3.0])
