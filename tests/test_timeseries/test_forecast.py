"""
Prophet 予測モジュールのテスト

Prophet そのものの出来は確かめない（実データで回して見る）。ここでは、学習とテストの切り方・
比べる相手・MAE と比・勝ちの数え方が仕様どおりであることと、Prophet が年次季節性あり・なしの
両方で回ることを確かめる。数値を厳密に確かめたいところは、Prophet を偽物に差し替える。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
import pytest
from src.timeseries import forecast
from src.timeseries.forecast import (
    COMPARISONS,
    METHODS,
    FitFallbackWarning,
    baseline_mean,
    baseline_recent,
    evaluate_unit,
    forecast_prophet,
    mae,
    mae_ratio,
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
