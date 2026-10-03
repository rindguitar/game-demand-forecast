"""
週次のシェアを Prophet で予測し、単純な予測（比べる相手）と当たり具合を比べるモジュール

方法・比べ方・学習の打ち切りの説明は scripts/README.md の「予測と評価」。ファイルの読み書きはしない。
"""

import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet

# テスト期間の長さ（週）。学習期間は、全単位に共通の週のうち、これより前の週すべて
TEST_WEEKS = 26

# 「直近の平均」に使う、学習期間の最後の週数
RECENT_WEEKS = 4

# Prophet の学習1回に待つ上限（秒）。Stan の L-BFGS が稀に終わらなくなるため（scripts/README.md「Prophet の注意点」）
FIT_TIMEOUT_SECONDS = 10

# 予測する4つの方法。forecasts.csv の列名にもなる
PROPHET_METHODS = ('prophet_yearly', 'prophet_no_yearly')
BASELINE_METHODS = ('baseline_mean', 'baseline_recent')
METHODS = PROPHET_METHODS + BASELINE_METHODS

# 比べる4通り = Prophet 2つ × 比べる相手 2つ。(Prophet, 比べる相手) の組
COMPARISONS = tuple((prophet, baseline)
                    for prophet in PROPHET_METHODS for baseline in BASELINE_METHODS)


def mae_column(method: str) -> str:
    """方法ごとの MAE の列名（metrics.csv の列名）"""
    return f'mae_{method}'


def ratio_column(prophet: str, baseline: str) -> str:
    """比 MAE(prophet) ÷ MAE(baseline) の列名（metrics.csv の列名）"""
    return f'ratio_{prophet}_vs_{baseline}'


def split_train_test(df: pd.DataFrame, test_weeks: int = TEST_WEEKS,
                     week_column: str = 'week', value_column: str = 'share'
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    全単位を同じ週で、学習とテストに分ける

    処理の流れ:
      1. 全単位の週をまとめて昇順に並べ、最後の test_weeks 週の先頭を「切る週」にする
      2. 値が欠測の行を除く（学習・テストとも）
      3. 切る週より前を学習、切る週以降をテストにする

    切る週は、欠測の行も含めた週の軸から決める。欠測を除いてから単位ごとに数えると、
    単位によって切る週がずれてしまうため。

    Args:
        df: 週の列と値の列を持つ縦長のDataFrame（全単位ぶん）
        test_weeks: テスト期間の週数

    Returns:
        (学習, テスト, 切る週)。切る週はテストの最初の週の月曜
    """
    if test_weeks < 1:
        raise ValueError(f'test_weeks は1以上にする: {test_weeks}')
    weeks = pd.Index(df[week_column].unique()).sort_values()
    if len(weeks) <= test_weeks:
        raise ValueError(f'学習の週が残らない（全{len(weeks)}週に対してテスト{test_weeks}週）')
    cutoff = weeks[-test_weeks]

    observed = df.dropna(subset=[value_column])
    train = observed[observed[week_column] < cutoff]
    test = observed[observed[week_column] >= cutoff]
    return train, test, cutoff


class FitFallbackWarning(UserWarning):
    """Prophet の学習が時間内に終わらず、Newton 法で学び直したときの警告"""


def _fit_prophet(history: pd.DataFrame, yearly: bool, **fit_options) -> Prophet:
    """
    Prophet を学習して返す

    年次季節性以外は既定値（チューニング・holidays・changepoint の指定はしない）。
    学習したモデルは1回しか使えないので、学び直すときは作り直す。
    """
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=False,
                    daily_seasonality=False)
    return model.fit(history, **fit_options)


def forecast_prophet(train: pd.DataFrame, weeks: Sequence, yearly: bool,
                     week_column: str = 'week', value_column: str = 'share',
                     fit_timeout: Optional[float] = FIT_TIMEOUT_SECONDS) -> np.ndarray:
    """
    Prophet で学習して、指定した週の値を予測する

    処理の流れ:
      1. 学習データを Prophet の列名（ds / y）に直す
      2. 年次季節性を yearly で切り替えて学習する（週・日の季節性は使わない）。
         fit_timeout 秒で終わらなければ、警告を出して Newton 法で学び直す
      3. 予測したい週だけを渡して、予測値（yhat）を取り出す

    予測が負になってもクリップしない。

    Args:
        train: 学習期間の週と値（1単位ぶん）
        weeks: 予測する週。昇順で渡す（Prophet は昇順で返すので、順がずれないように）
        yearly: 年次季節性を入れるか
        fit_timeout: 学習1回に待つ上限（秒）。None なら待ち続ける

    Returns:
        weeks と同じ長さの予測値
    """
    weeks = pd.DatetimeIndex(weeks)
    if not weeks.is_monotonic_increasing:
        raise ValueError('weeks は昇順で渡す')

    history = pd.DataFrame({'ds': train[week_column].to_numpy(),
                            'y': train[value_column].to_numpy()})
    try:
        model = _fit_prophet(history, yearly, timeout=fit_timeout)
    except TimeoutError:
        warnings.warn(f'Prophet の学習が{fit_timeout}秒で終わらなかったので、Newton 法で学び直した'
                      f'（年次季節性{"あり" if yearly else "なし"}）', FitFallbackWarning)
        model = _fit_prophet(history, yearly, algorithm='Newton')
    return model.predict(pd.DataFrame({'ds': weeks}))['yhat'].to_numpy()


def _observed_values(train: pd.DataFrame, week_column: str, value_column: str) -> pd.Series:
    """学習期間の値を、週の昇順に並べて欠測を除いたものにする"""
    values = train.sort_values(week_column)[value_column].dropna()
    if values.empty:
        raise ValueError('学習期間に実績が1つも無い')
    return values


def baseline_mean(train: pd.DataFrame, horizon: int,
                  week_column: str = 'week', value_column: str = 'share') -> np.ndarray:
    """比べる相手①: 学習期間の平均を、horizon 週ぶん並べる（欠測は除いて平均する）"""
    values = _observed_values(train, week_column, value_column)
    return np.full(horizon, values.mean())


def baseline_recent(train: pd.DataFrame, horizon: int, recent_weeks: int = RECENT_WEEKS,
                    week_column: str = 'week', value_column: str = 'share') -> np.ndarray:
    """
    比べる相手②: 学習期間の最後の recent_weeks 週の平均を、horizon 週ぶん並べる

    欠測を除いた最後の recent_weeks 個で平均する（欠測の週は数えない）。
    実績が recent_weeks 個に満たなければ、あるぶんで平均する。
    """
    if recent_weeks < 1:
        raise ValueError(f'recent_weeks は1以上にする: {recent_weeks}')
    values = _observed_values(train, week_column, value_column)
    return np.full(horizon, values.tail(recent_weeks).mean())


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """予測と実績の差の絶対値の平均（MAE）"""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f'実績と予測の長さが違う: {actual.shape} / {predicted.shape}')
    if actual.size == 0:
        raise ValueError('MAE を出す週が1つも無い')
    return float(np.mean(np.abs(actual - predicted)))


def mae_ratio(mae_prophet: float, mae_baseline: float) -> float:
    """
    比 = MAE(Prophet) ÷ MAE(比べる相手)。1未満なら Prophet の勝ち

    比べる相手の MAE が0のときは割れない。Prophet の MAE が正なら inf（Prophet の負け）、
    Prophet も0なら nan（勝ちにも負けにも数えない）にする。
    """
    if mae_baseline == 0:
        return float('nan') if mae_prophet == 0 else float('inf')
    return mae_prophet / mae_baseline


def evaluate_unit(train: pd.DataFrame, test: pd.DataFrame, recent_weeks: int = RECENT_WEEKS,
                  week_column: str = 'week', value_column: str = 'share'
                  ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    1単位を4つの方法で予測し、テスト期間の当たり具合を測る

    処理の流れ:
      1. テスト週を昇順に並べ、実績を取り出す
      2. Prophet を年次季節性あり・なしで学習し、テスト週を予測する
      3. 比べる相手2つ（学習期間の平均・直近の平均）を、テスト週と同じ長さで作る
      4. 4つの方法それぞれの MAE を出す
      5. Prophet 2つ × 比べる相手 2つ の4通りで、比を出す

    Args:
        train: 学習期間の週と値（split_train_test の学習のうち、この単位）
        test: テスト期間の週と値（同テストのうち、この単位）。欠測は除いてあること
        recent_weeks: 「直近の平均」に使う、学習期間の最後の週数

    Returns:
        (指標, 予測)。指標は mae_<方法> 4つと ratio_<Prophet>_vs_<比べる相手> 4つの dict。
        予測は week / actual / 4つの方法の予測 を、テスト週ごとに並べたDataFrame
    """
    if test.empty:
        raise ValueError('テスト期間に実績が1つも無い')
    test = test.sort_values(week_column)
    weeks = pd.DatetimeIndex(test[week_column])
    actual = test[value_column].to_numpy(dtype=float)

    predictions = pd.DataFrame({week_column: weeks, 'actual': actual})
    predictions['prophet_yearly'] = forecast_prophet(
        train, weeks, yearly=True, week_column=week_column, value_column=value_column)
    predictions['prophet_no_yearly'] = forecast_prophet(
        train, weeks, yearly=False, week_column=week_column, value_column=value_column)
    predictions['baseline_mean'] = baseline_mean(
        train, len(weeks), week_column=week_column, value_column=value_column)
    predictions['baseline_recent'] = baseline_recent(
        train, len(weeks), recent_weeks=recent_weeks,
        week_column=week_column, value_column=value_column)

    metrics = {mae_column(method): mae(actual, predictions[method]) for method in METHODS}
    for prophet, baseline in COMPARISONS:
        metrics[ratio_column(prophet, baseline)] = mae_ratio(
            metrics[mae_column(prophet)], metrics[mae_column(baseline)])
    return metrics, predictions


def summarize_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    全単位の評価表から、4通りの比較それぞれの勝ち負けをまとめる

    処理の流れ:
      1. 比較（Prophet × 比べる相手）ごとに、比の列を取る
      2. 比が1未満の単位を勝ちと数える（ちょうど1や欠測は勝ちにしない）
      3. 全単位数と、比の中央値（欠測は除く）を添える

    Args:
        metrics: 1行1単位の評価表。ratio_<Prophet>_vs_<比べる相手> の4列を持つ

    Returns:
        比較ごとに1行。prophet / baseline / wins（勝った単位数）/
        units（全単位数）/ median_ratio（比の中央値）
    """
    rows = []
    for prophet, baseline in COMPARISONS:
        ratio = metrics[ratio_column(prophet, baseline)]
        rows.append({'prophet': prophet, 'baseline': baseline,
                     'wins': int((ratio < 1).sum()), 'units': len(ratio),
                     'median_ratio': ratio.median()})
    return pd.DataFrame(rows)
