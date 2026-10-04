"""
週次のシェアを Prophet で予測し、単純な予測（比べる相手）と当たり具合を比べるモジュール

方法・比べ方・学習の打ち切り・発売を出来事として渡す選び方・発売を段差の印としても渡す方法の説明は
scripts/README.md の「予測と評価」。ファイルの読み書きはしない。
"""

import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.utilities import regressor_coefficients

# テスト期間の長さ（週）。学習期間は、全単位に共通の週のうち、これより前の週すべて
TEST_WEEKS = 26

# 「直近の平均」に使う、学習期間の最後の週数
RECENT_WEEKS = 4

# 発売を出来事として渡すときの既定値（選び方は scripts/README.md「予測と評価」）
LAUNCH_WEEKS = 8                  # 発売週から数えて、発売の影響を見る週数
LAUNCH_MIN_SHARE = 0.5            # その単位のレビューのうち、発売したゲームが占める割合の下限（過半数）
LAUNCH_MIN_WEEKLY_MENTIONS = 10   # 発売したゲームの、その単位でのレビュー数の下限（1週あたり）

# Prophet の学習1回に待つ上限（秒）。Stan の L-BFGS が稀に終わらなくなるため（scripts/README.md「Prophet の注意点」）
FIT_TIMEOUT_SECONDS = 10

# 予測する4つの方法。forecasts.csv の列名にもなる
PROPHET_METHODS = ('prophet_yearly', 'prophet_no_yearly')
BASELINE_METHODS = ('baseline_mean', 'baseline_recent')
METHODS = PROPHET_METHODS + BASELINE_METHODS

# 比べる4通り = Prophet 2つ × 比べる相手 2つ。(Prophet, 比べる相手) の組
COMPARISONS = tuple((prophet, baseline)
                    for prophet in PROPHET_METHODS for baseline in BASELINE_METHODS)

# 勝ちの基準（Issue #60）。Prophet の型ごとに、比べる相手2つの両方に、この単位数以上で勝てば届いた扱い
WIN_CRITERION_UNITS = 40

# launch_effects.csv の列（1行 = Prophet の型 × 単位 × 発売）
LAUNCH_EFFECT_COLUMNS = ['prophet', 'unit', 'keywords', 'game', 'release_week',
                         'has_step', 'spike_peak', 'step_size']


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


def find_target_launches(games: pd.DataFrame, first_week: pd.Timestamp, cutoff: pd.Timestamp,
                         launch_weeks: int = LAUNCH_WEEKS) -> pd.DataFrame:
    """
    出来事として渡す対象の発売を、ゲーム台帳から選ぶ

    処理の流れ:
      1. 発売日が読めないゲームがあれば止まる（select_backbone と同じ扱い）
      2. 発売日をその週の月曜（発売週）にする
      3. 次の2つを満たす発売を残す
         - 発売週から launch_weeks 週が、データ期間の最初の週以降と重なる
           （期間の直前に出たゲームも、発売後の山が期間に入るので対象）
         - 発売週が切る週より前（テスト期間に入る発売は使わない）

    Args:
        games: ゲーム台帳。name / release_date 列が必要
        first_week: データ期間の最初の週（月曜）
        cutoff: 切る週（split_train_test が返すテストの最初の週）
        launch_weeks: 発売週から数える週数

    Returns:
        game / release_week を持つDataFrame（台帳の並び）
    """
    if launch_weeks < 1:
        raise ValueError(f'launch_weeks は1以上にする: {launch_weeks}')

    released = pd.to_datetime(games['release_date'], errors='coerce')
    unreadable = games.loc[released.isna(), 'name'].tolist()
    if unreadable:
        raise ValueError(f'発売日が読めないゲームがある: {unreadable}')

    release_week = released.dt.to_period('W').dt.start_time
    last_week = release_week + pd.Timedelta(weeks=launch_weeks - 1)
    target = (last_week >= pd.Timestamp(first_week)) & (release_week < pd.Timestamp(cutoff))
    return pd.DataFrame({'game': games.loc[target, 'name'],
                         'release_week': release_week[target]}).reset_index(drop=True)


def select_launch_events(reviews: pd.DataFrame, games: pd.DataFrame, units: Sequence,
                         first_week: pd.Timestamp, cutoff: pd.Timestamp,
                         launch_weeks: int = LAUNCH_WEEKS, min_share: float = LAUNCH_MIN_SHARE,
                         min_weekly_mentions: float = LAUNCH_MIN_WEEKLY_MENTIONS,
                         game_column: str = 'game_name', unit_column: str = 'topic_id',
                         week_column: str = 'week') -> pd.DataFrame:
    """
    各単位に出来事として渡す発売を、発売直後のレビューから選ぶ

    ある単位の言及が発売直後にそのゲームで占められていれば、その山は「発売のせい」と見なせる。

    処理の流れ:
      1. 対象の発売を決める（find_target_launches）
      2. 発売ごとに「確かめる期間」を決める = 発売週から launch_weeks 週。
         ただし切る週より前で打ち切る（テスト期間のレビューを使わないため）。
         期間の最初の週より前のレビューは、あれば使う
      3. 確かめる期間の中で、単位ごとに unit_mentions（その単位の全レビュー数）と
         game_mentions（そのうち、そのゲームのレビュー数）を数える
      4. 次の2つを両方満たす（単位, 発売）の組を残す
         - game_mentions / unit_mentions >= min_share（ちょうどでも残す）
         - game_mentions >= min_weekly_mentions × 確かめる期間の週数
           （打ち切りで週数が減ったら、その週数で掛ける）

    Args:
        reviews: game_column / week_column / unit_column を持つレビュー（1行1レビュー）
        games: ゲーム台帳。name / release_date 列が必要
        units: 週次系列にある単位。これに無いトピックのレビューは数えない
        first_week: データ期間の最初の週（月曜）
        cutoff: 切る週（split_train_test が返すテストの最初の週）

    Returns:
        unit / game / release_week / game_mentions / unit_mentions / share を持つDataFrame
        （unit・release_week・game の順に並べる）。組が1つも無ければ空
    """
    launches = find_target_launches(games, first_week, cutoff, launch_weeks)
    cutoff = pd.Timestamp(cutoff)
    in_units = reviews[reviews[unit_column].isin(units)]

    chosen_by_launch = []
    for launch in launches.itertuples():
        end = min(launch.release_week + pd.Timedelta(weeks=launch_weeks), cutoff)
        weeks_checked = (end - launch.release_week) // pd.Timedelta(weeks=1)

        in_period = in_units[(in_units[week_column] >= launch.release_week)
                             & (in_units[week_column] < end)]
        unit_mentions = in_period.groupby(unit_column).size()
        game_mentions = (in_period[in_period[game_column] == launch.game]
                         .groupby(unit_column).size())

        pairs = pd.DataFrame({'game_mentions': game_mentions,
                              'unit_mentions': unit_mentions.reindex(game_mentions.index)})
        pairs['share'] = pairs['game_mentions'] / pairs['unit_mentions']
        passed = pairs[(pairs['share'] >= min_share)
                       & (pairs['game_mentions'] >= min_weekly_mentions * weeks_checked)]
        if not passed.empty:
            chosen_by_launch.append(passed.rename_axis('unit').reset_index()
                                    .assign(game=launch.game, release_week=launch.release_week))

    columns = ['unit', 'game', 'release_week', 'game_mentions', 'unit_mentions', 'share']
    if not chosen_by_launch:
        return pd.DataFrame(columns=columns)
    events = pd.concat(chosen_by_launch, ignore_index=True)[columns]
    return events.sort_values(['unit', 'release_week', 'game']).reset_index(drop=True)


def launch_holidays(events: pd.DataFrame, unit: int, launch_weeks: int = LAUNCH_WEEKS
                    ) -> Optional[pd.DataFrame]:
    """
    1単位に渡す発売を、Prophet の holidays（出来事の表）にする

    処理の流れ:
      1. events（select_launch_events の結果）から、この単位の行を取る。無ければ None
      2. 発売ごとに別の出来事にする（holiday = ゲーム名・ds = 発売週）
      3. 窓は発売週から launch_weeks 週。週次データなので、発売週の月曜（lower_window = 0）から
         7×(launch_weeks−1) 日後（upper_window）までにする

    Returns:
        holiday / ds / lower_window / upper_window を持つDataFrame。
        発売が付かない単位は None（Prophet に holidays を渡さない）
    """
    one = events[events['unit'] == unit]
    if one.empty:
        return None
    return pd.DataFrame({'holiday': one['game'].to_numpy(), 'ds': one['release_week'].to_numpy(),
                         'lower_window': 0,
                         'upper_window': pd.Timedelta(weeks=launch_weeks - 1).days})


def launch_steps(holidays: Optional[pd.DataFrame], first_week: pd.Timestamp) -> pd.DataFrame:
    """
    1単位に渡した発売のうち、段差の印を付けるものを選び、印の列名を振る

    発売前の週が学習期間に無いと印がずっと1になって段差を学べないので、
    発売週が学習期間の最初の週より後のものだけに付ける。

    処理の流れ:
      1. holidays（launch_holidays が返す形）が None なら、空の表を返す
      2. 発売週が first_week より後の発売だけを残す
      3. 残った発売に、holidays の並び順に step_0, step_1 ... と名前を振る
         （ゲーム名は列名にしない。対応は game 列に持つ）

    Args:
        holidays: この単位に渡す発売。None なら発売が付かない単位
        first_week: 学習期間の最初の週（この単位の学習データの最初の週）

    Returns:
        step / game / release_week を持つDataFrame。付ける発売が無ければ空
    """
    if holidays is None:
        return pd.DataFrame(columns=['step', 'game', 'release_week'])
    later = holidays[holidays['ds'] > pd.Timestamp(first_week)]
    return pd.DataFrame({'step': [f'step_{i}' for i in range(len(later))],
                         'game': later['holiday'].to_numpy(),
                         'release_week': later['ds'].to_numpy()})


def with_step_columns(frame: pd.DataFrame, steps: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    ds 列を持つ表に、段差の印（0/1 の列）を足して返す

    印は、ds が発売週より前なら0、発売週から後はずっと1（予測する未来の週も1）。
    steps が None か空なら、何も足さない。
    """
    if steps is None:
        return frame
    return frame.assign(**{row.step: (frame['ds'] >= row.release_week).astype(int)
                           for row in steps.itertuples()})


class FitFallbackWarning(UserWarning):
    """Prophet の学習が時間内に終わらず、Newton 法で学び直したときの警告"""


def _fit_prophet(history: pd.DataFrame, yearly: bool, holidays: Optional[pd.DataFrame] = None,
                 steps: Optional[pd.DataFrame] = None, **fit_options) -> Prophet:
    """
    Prophet を学習して返す

    年次季節性・holidays・段差の印（渡されたときだけ add_regressor）以外は既定値
    （チューニング・changepoint の指定はしない）。
    学習したモデルは1回しか使えないので、学び直すときは作り直す。
    """
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=False,
                    daily_seasonality=False, holidays=holidays)
    if steps is not None:
        for name in steps['step']:
            model.add_regressor(name)
    return model.fit(history, **fit_options)


def fit_prophet(train: pd.DataFrame, yearly: bool,
                week_column: str = 'week', value_column: str = 'share',
                fit_timeout: Optional[float] = FIT_TIMEOUT_SECONDS,
                holidays: Optional[pd.DataFrame] = None,
                steps: Optional[pd.DataFrame] = None) -> Prophet:
    """
    Prophet を学習して返す。時間内に終わらなければ、警告を出して Newton 法で学び直す

    処理の流れ:
      1. 学習データを Prophet の列名（ds / y）に直し、steps があれば段差の印の列を足す
      2. 年次季節性を yearly で切り替えて学習する（週・日の季節性は使わない。
         holidays があれば出来事、steps があれば説明変数として渡す）
      3. fit_timeout 秒で終わらなければ、同じ引数のまま Newton 法で学び直す

    Args:
        train: 学習期間の週と値（1単位ぶん）
        yearly: 年次季節性を入れるか
        fit_timeout: 学習1回に待つ上限（秒）。None なら待ち続ける
        holidays: 出来事の表（launch_holidays が返す形）。None なら渡さない
        steps: 段差の印の表（launch_steps が返す形）。None なら渡さない
    """
    history = with_step_columns(pd.DataFrame({'ds': train[week_column].to_numpy(),
                                              'y': train[value_column].to_numpy()}), steps)
    try:
        return _fit_prophet(history, yearly, holidays, steps, timeout=fit_timeout)
    except TimeoutError:
        warnings.warn(f'Prophet の学習が{fit_timeout}秒で終わらなかったので、Newton 法で学び直した'
                      f'（年次季節性{"あり" if yearly else "なし"}）', FitFallbackWarning)
        return _fit_prophet(history, yearly, holidays, steps, algorithm='Newton')


def predict_prophet(model: Prophet, weeks: Sequence,
                    steps: Optional[pd.DataFrame] = None) -> np.ndarray:
    """学習した Prophet で、指定した週（昇順）の値（yhat）を予測する。steps があれば段差の印の列を足して渡す"""
    future = with_step_columns(pd.DataFrame({'ds': pd.DatetimeIndex(weeks)}), steps)
    return model.predict(future)['yhat'].to_numpy()


def forecast_prophet(train: pd.DataFrame, weeks: Sequence, yearly: bool,
                     week_column: str = 'week', value_column: str = 'share',
                     fit_timeout: Optional[float] = FIT_TIMEOUT_SECONDS,
                     holidays: Optional[pd.DataFrame] = None,
                     steps: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Prophet で学習して、指定した週の値を予測する

    処理の流れ:
      1. weeks が昇順であることを確かめる
      2. Prophet を学習する（fit_prophet。holidays・steps があれば渡す。
         fit_timeout 秒で終わらなければ、警告を出して Newton 法で学び直す）
      3. 予測したい週だけを渡して、予測値（yhat）を取り出す（predict_prophet）

    予測が負になってもクリップしない。

    Args:
        train: 学習期間の週と値（1単位ぶん）
        weeks: 予測する週。昇順で渡す（Prophet は昇順で返すので、順がずれないように）
        yearly: 年次季節性を入れるか
        fit_timeout: 学習1回に待つ上限（秒）。None なら待ち続ける
        holidays: 出来事の表（launch_holidays が返す形）。None なら渡さない
        steps: 段差の印の表（launch_steps が返す形）。None なら渡さない

    Returns:
        weeks と同じ長さの予測値
    """
    weeks = pd.DatetimeIndex(weeks)
    if not weeks.is_monotonic_increasing:
        raise ValueError('weeks は昇順で渡す')

    model = fit_prophet(train, yearly, week_column, value_column, fit_timeout, holidays, steps)
    return predict_prophet(model, weeks, steps)


def read_launch_effects(model: Prophet, train: pd.DataFrame, holidays: Optional[pd.DataFrame],
                        steps: Optional[pd.DataFrame], week_column: str = 'week') -> pd.DataFrame:
    """
    学習した Prophet から、発売ごとの効き目（山の印・段差の印）を読む

    処理の流れ:
      1. holidays が None なら、列だけの空の表を返す
      2. 学習期間の週で当てはめ直し、発売ごと（列名 = ゲーム名）の holidays の成分を取り出す
      3. 発売ごとに、窓（ds + lower_window 日 〜 ds + upper_window 日）の中の成分の最大値を
         spike_peak にする（窓が学習期間と重ならなければ NaN）
      4. 段差の印のある発売には、その係数を step_size に入れる。印の無い発売は NaN

    値の単位はシェア（Prophet が返す、元の値の単位）。

    Args:
        model: fit_prophet が返した、学習済みのモデル
        train: そのモデルを学習した、学習期間の週と値
        holidays: そのモデルに渡した発売
        steps: そのモデルに渡した段差の印の表

    Returns:
        game / release_week / has_step / spike_peak / step_size を持つDataFrame（発売1つにつき1行）
    """
    columns = ['game', 'release_week', 'has_step', 'spike_peak', 'step_size']
    if holidays is None:
        return pd.DataFrame(columns=columns)

    fitted = model.predict(with_step_columns(
        pd.DataFrame({'ds': train[week_column].to_numpy()}), steps))
    step_by_game = {} if steps is None else dict(zip(steps['game'], steps['step']))
    coefficient = ({} if not step_by_game else
                   regressor_coefficients(model).set_index('regressor')['coef'].to_dict())

    rows = []
    for launch in holidays.itertuples():
        start = launch.ds + pd.Timedelta(days=launch.lower_window)
        end = launch.ds + pd.Timedelta(days=launch.upper_window)
        in_window = (fitted['ds'] >= start) & (fitted['ds'] <= end)
        step = step_by_game.get(launch.holiday)
        rows.append({'game': launch.holiday, 'release_week': launch.ds,
                     'has_step': step is not None,
                     'spike_peak': fitted.loc[in_window, launch.holiday].max(),
                     'step_size': coefficient.get(step, np.nan)})
    return pd.DataFrame(rows, columns=columns)


def drop_holiday_weeks(train: pd.DataFrame, holidays: Optional[pd.DataFrame],
                       week_column: str = 'week', value_column: str = 'share') -> pd.DataFrame:
    """
    学習期間から、出来事の期間に入る週を除く（比べる相手に Prophet と同じ情報を渡すため）

    処理の流れ:
      1. holidays が None なら何も除かない
      2. 出来事ごとに、ds + lower_window 日 〜 ds + upper_window 日 に入る週を印付けする
         （発売なら、発売週から launch_weeks 週）
      3. 印の付いた週を除く。実績が1つも残らなければ止まる
    """
    if holidays is None:
        return train

    weeks = pd.DatetimeIndex(train[week_column])
    in_holiday = np.zeros(len(train), dtype=bool)
    for row in holidays.itertuples():
        start = row.ds + pd.Timedelta(days=row.lower_window)
        end = row.ds + pd.Timedelta(days=row.upper_window)
        in_holiday |= (weeks >= start) & (weeks <= end)

    remaining = train[~in_holiday]
    if remaining[value_column].dropna().empty:
        raise ValueError('発売後の週を除くと、学習期間に実績が1つも残らない')
    return remaining


def keep_weeks_after_latest_launch(train: pd.DataFrame, holidays: Optional[pd.DataFrame],
                                   week_column: str = 'week', value_column: str = 'share'
                                   ) -> pd.DataFrame:
    """
    学習期間から、最新の発売の窓が終わった次の週以降だけを残す（段差の印を渡すときの、比べる相手のため）

    段差の印を渡すと、予測したいテスト期間は印が1の状態になる。発売前の週は「いまのふだんの高さ」を
    表さないので、比べる相手は最新の発売の後の週だけで作る。

    処理の流れ:
      1. holidays が None なら何も除かない
      2. 発売ごとの窓の終わり（ds + upper_window 日）の最も遅いものを出す
         （段差の印を付けなかった発売も数える）
      3. 窓の終わりより後の週だけを残す。実績が1つも残らなければ止まる
    """
    if holidays is None:
        return train

    window_end = max(row.ds + pd.Timedelta(days=row.upper_window)
                     for row in holidays.itertuples())
    remaining = train[pd.DatetimeIndex(train[week_column]) > window_end]
    if remaining[value_column].dropna().empty:
        raise ValueError('最新の発売の後の週が、学習期間に1つも残らない')
    return remaining


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


def _score_predictions(actual: np.ndarray, predictions: pd.DataFrame) -> Dict[str, float]:
    """4つの方法それぞれの MAE と、Prophet 2つ × 比べる相手 2つの4通りの比を出す"""
    metrics = {mae_column(method): mae(actual, predictions[method]) for method in METHODS}
    for prophet, baseline in COMPARISONS:
        metrics[ratio_column(prophet, baseline)] = mae_ratio(
            metrics[mae_column(prophet)], metrics[mae_column(baseline)])
    return metrics


def evaluate_unit(train: pd.DataFrame, test: pd.DataFrame, recent_weeks: int = RECENT_WEEKS,
                  week_column: str = 'week', value_column: str = 'share',
                  holidays: Optional[pd.DataFrame] = None
                  ) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    1単位を4つの方法で予測し、テスト期間の当たり具合を測る

    処理の流れ:
      1. テスト週を昇順に並べ、実績を取り出す
      2. Prophet を年次季節性あり・なしで学習し、テスト週を予測する（holidays があれば渡す）
      3. 比べる相手2つ（学習期間の平均・直近の平均）を、テスト週と同じ長さで作る。
         holidays があれば、その期間の週を学習期間から除いてから作る
      4. 4つの方法それぞれの MAE を出す
      5. Prophet 2つ × 比べる相手 2つ の4通りで、比を出す

    Args:
        train: 学習期間の週と値（split_train_test の学習のうち、この単位）
        test: テスト期間の週と値（同テストのうち、この単位）。欠測は除いてあること
        recent_weeks: 「直近の平均」に使う、学習期間の最後の週数
        holidays: この単位に渡す発売（launch_holidays が返す形）。None なら発売を渡さない

    Returns:
        (指標, 予測)。指標は mae_<方法> 4つと ratio_<Prophet>_vs_<比べる相手> 4つの dict。
        予測は week / actual / 4つの方法の予測 を、テスト週ごとに並べたDataFrame
    """
    if test.empty:
        raise ValueError('テスト期間に実績が1つも無い')
    test = test.sort_values(week_column)
    weeks = pd.DatetimeIndex(test[week_column])
    actual = test[value_column].to_numpy(dtype=float)
    baseline_train = drop_holiday_weeks(train, holidays, week_column, value_column)

    predictions = pd.DataFrame({week_column: weeks, 'actual': actual})
    predictions['prophet_yearly'] = forecast_prophet(
        train, weeks, yearly=True, week_column=week_column, value_column=value_column,
        holidays=holidays)
    predictions['prophet_no_yearly'] = forecast_prophet(
        train, weeks, yearly=False, week_column=week_column, value_column=value_column,
        holidays=holidays)
    predictions['baseline_mean'] = baseline_mean(
        baseline_train, len(weeks), week_column=week_column, value_column=value_column)
    predictions['baseline_recent'] = baseline_recent(
        baseline_train, len(weeks), recent_weeks=recent_weeks,
        week_column=week_column, value_column=value_column)

    return _score_predictions(actual, predictions), predictions


def evaluate_unit_with_steps(train: pd.DataFrame, test: pd.DataFrame,
                             recent_weeks: int = RECENT_WEEKS,
                             week_column: str = 'week', value_column: str = 'share',
                             holidays: Optional[pd.DataFrame] = None
                             ) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    1単位を4つの方法で予測し、テスト期間の当たり具合と、発売ごとの効き目を測る（段差の印を渡す版）

    evaluate_unit との違いは3つ。Prophet に、山の印（holidays）に加えて段差の印も渡す／
    比べる相手は、最新の発売の窓が終わった次の週以降の学習期間で作る／
    学習した Prophet から、発売ごとの効き目を読む。

    処理の流れ:
      1. テスト週を昇順に並べ、実績を取り出す
      2. 段差の印を付ける発売を選ぶ（発売週が、この単位の学習期間の最初の週より後のもの）
      3. Prophet を年次季節性あり・なしで学習し、テスト週を予測する。
         学習したモデルから、発売ごとの効き目を読む
      4. 比べる相手2つ（学習期間の平均・直近の平均）を、最新の発売の窓が終わった次の週以降の
         学習期間で、テスト週と同じ長さで作る
      5. 4つの方法それぞれの MAE と、4通りの比を出す

    発売が付かない単位（holidays が None）は、evaluate_unit と同じ予測・比べる相手になる。

    Args:
        train: 学習期間の週と値（split_train_test の学習のうち、この単位）。欠測は除いてあること
        test: テスト期間の週と値（同テストのうち、この単位）。欠測は除いてあること
        recent_weeks: 「直近の平均」に使う、最後の週数
        holidays: この単位に渡す発売（launch_holidays が返す形）。None なら発売を渡さない

    Returns:
        (指標, 予測, 発売の効き目)。指標と予測は evaluate_unit と同じ形。
        発売の効き目は prophet / game / release_week / has_step / spike_peak / step_size を持つ
        DataFrame（Prophet の型 × 発売ごとに1行。発売が付かない単位は空）
    """
    if test.empty:
        raise ValueError('テスト期間に実績が1つも無い')
    test = test.sort_values(week_column)
    weeks = pd.DatetimeIndex(test[week_column])
    actual = test[value_column].to_numpy(dtype=float)
    steps = launch_steps(holidays, train[week_column].min())

    predictions = pd.DataFrame({week_column: weeks, 'actual': actual})
    effects = []
    for method, yearly in (('prophet_yearly', True), ('prophet_no_yearly', False)):
        model = fit_prophet(train, yearly, week_column, value_column,
                            holidays=holidays, steps=steps)
        predictions[method] = predict_prophet(model, weeks, steps)
        effects.append(read_launch_effects(model, train, holidays, steps, week_column)
                       .assign(prophet=method))

    baseline_train = keep_weeks_after_latest_launch(train, holidays, week_column, value_column)
    predictions['baseline_mean'] = baseline_mean(
        baseline_train, len(weeks), week_column=week_column, value_column=value_column)
    predictions['baseline_recent'] = baseline_recent(
        baseline_train, len(weeks), recent_weeks=recent_weeks,
        week_column=week_column, value_column=value_column)

    return (_score_predictions(actual, predictions), predictions,
            pd.concat(effects, ignore_index=True))


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


def launch_effects_table(effects_by_unit: Mapping[int, pd.DataFrame],
                         keywords: Mapping[int, str]) -> pd.DataFrame:
    """
    全単位の発売の効き目を、launch_effects.csv の形にまとめる

    処理の流れ:
      1. 単位ごとの表（evaluate_unit_with_steps が返す形）に、単位とキーワードの列を足して縦に積む
      2. 列を LAUNCH_EFFECT_COLUMNS の順にし、Prophet の型 → 単位 → 発売週の順に並べる

    Args:
        effects_by_unit: 単位 → その単位の発売の効き目
        keywords: 単位 → キーワード（その単位が何の話かを読むための列）

    Returns:
        LAUNCH_EFFECT_COLUMNS を持つDataFrame。発売が1つも付かなければ空
    """
    frames = [effects.assign(unit=unit, keywords=keywords[unit])
              for unit, effects in effects_by_unit.items() if not effects.empty]
    if not frames:
        return pd.DataFrame(columns=LAUNCH_EFFECT_COLUMNS)

    table = pd.concat(frames, ignore_index=True)[LAUNCH_EFFECT_COLUMNS]
    table = table.sort_values(['unit', 'release_week', 'game'])
    table = table.sort_values('prophet', key=lambda column: column.map(PROPHET_METHODS.index),
                              kind='stable')
    return table.reset_index(drop=True)


def check_win_criterion(summary: pd.DataFrame, required_wins: int = WIN_CRITERION_UNITS
                        ) -> pd.DataFrame:
    """
    勝ちの基準に届いたかを、Prophet の型ごとに判定する

    基準 = 比べる相手2つの両方に、required_wins 単位以上で勝つ（ちょうどでも届いた扱い）。
    片方の相手にしか届かなければ、届かない。

    処理の流れ:
      1. summary（summarize_comparisons の結果）から、Prophet の型ごとに、
         比べる相手2つに勝った単位数を取る
      2. 2つとも required_wins 以上なら、届いたと判定する

    Args:
        summary: summarize_comparisons が返す表
        required_wins: 基準の単位数

    Returns:
        Prophet の型ごとに1行。prophet / wins_<比べる相手> 2つ / units（全単位数）/
        required_wins / reached（届いたか）
    """
    rows = []
    for prophet in PROPHET_METHODS:
        one = summary[summary['prophet'] == prophet].set_index('baseline')
        row = {'prophet': prophet,
               **{f'wins_{baseline}': int(one.at[baseline, 'wins'])
                  for baseline in BASELINE_METHODS},
               'units': int(one['units'].iloc[0]), 'required_wins': required_wins}
        row['reached'] = all(row[f'wins_{baseline}'] >= required_wins
                             for baseline in BASELINE_METHODS)
        rows.append(row)
    return pd.DataFrame(rows)
