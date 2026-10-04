"""
週次のシェアを Prophet で予測し、単純な予測（比べる相手）と当たり具合を比べるモジュール

方法・比べ方・学習の打ち切り・発売を出来事として渡す選び方・発売を段差の印としても渡す方法・
確かめ用の期間で Prophet の設定を選ぶ方法の説明は scripts/README.md の「予測と評価」。
ファイルの読み書きはしない。
"""

import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

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

# Prophet の型 → 年次季節性を入れるか
YEARLY_BY_METHOD = dict(zip(PROPHET_METHODS, (True, False)))

# 比べる4通り = Prophet 2つ × 比べる相手 2つ。(Prophet, 比べる相手) の組
COMPARISONS = tuple((prophet, baseline)
                    for prophet in PROPHET_METHODS for baseline in BASELINE_METHODS)

# 勝ちの基準（Issue #60）。Prophet の型ごとに、比べる相手2つの両方に、この単位数以上で勝てば届いた扱い
WIN_CRITERION_UNITS = 40

# 確かめ用の期間で選ぶ Prophet の設定2つ（Prophet の引数名。tuning.csv の列名にもなる）
CHANGEPOINT_SCALE = 'changepoint_prior_scale'    # トレンドの曲がりやすさ
SEASONALITY_SCALE = 'seasonality_prior_scale'    # 季節性の効き具合（年次季節性ありの型だけ）

# 設定を選ぶ段階の既定値（Issue #60 ②）。確かめ用の期間の週数と、試す設定
VALIDATION_WEEKS = 26             # テスト期間と同じ長さ
CPS_GRID = (0.001, 0.01, 0.05, 0.5)
SPS_GRID = (0.01, 0.1, 1.0, 10.0)

# Prophet 自身の既定値。設定の成績が同点のとき、これに近い方を選ぶ
PROPHET_DEFAULT_SETTINGS = {CHANGEPOINT_SCALE: 0.05, SEASONALITY_SCALE: 10.0}

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


def split_validation(df: pd.DataFrame, test_cutoff: pd.Timestamp,
                     validation_weeks: int = VALIDATION_WEEKS,
                     week_column: str = 'week', value_column: str = 'share'
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """
    学習期間を、設定を選ぶための「学ぶ期間」と「確かめ用の期間」に分ける（テスト期間は使わない）

    処理の流れ:
      1. テストの切る週より前の行だけを残す（テスト期間の行は、ここで捨てる）
      2. split_train_test と同じ規則で、最後の validation_weeks 週を確かめ用の期間、
         その前を学ぶ期間にする

    Args:
        df: 週の列と値の列を持つ縦長のDataFrame（全単位ぶん。テスト期間の行を含んでよい）
        test_cutoff: テストの切る週（split_train_test が返したもの）
        validation_weeks: 確かめ用の期間の週数

    Returns:
        (学ぶ期間, 確かめ用の期間, 確かめ用の切る週)。確かめ用の切る週は、確かめ用の期間の最初の週の月曜
    """
    before_test = df[df[week_column] < pd.Timestamp(test_cutoff)]
    try:
        return split_train_test(before_test, validation_weeks, week_column, value_column)
    except ValueError as error:
        raise ValueError(f'確かめ用の期間を切れない（validation_weeks={validation_weeks}）: {error}'
                         ) from error


def parse_grid(text: str) -> Tuple[float, ...]:
    """
    カンマ区切りの数（例: '0.001,0.01,0.05'）を、試す設定の並び（タプル）にする

    数でない・空の要素・0以下や有限でない値・重複があれば ValueError。
    重複を許さないのは、同じ設定を二重に数えて、勝った単位数が膨らむのを防ぐため。
    """
    try:
        values = tuple(float(item) for item in text.split(','))
    except ValueError:
        raise ValueError(f'カンマ区切りの数で渡す（例: 0.01,0.1）: {text!r}') from None
    if not all(np.isfinite(value) and value > 0 for value in values):
        raise ValueError(f'0より大きい有限の数にする: {text!r}')
    if len(set(values)) != len(values):
        raise ValueError(f'同じ値が重なっている: {text!r}')
    return values


def prophet_settings_grid(prophet: str, cps_grid: Sequence[float] = CPS_GRID,
                          sps_grid: Sequence[float] = SPS_GRID) -> List[Dict[str, float]]:
    """
    Prophet の型ごとに、試す設定（Prophet に渡す引数の dict）を並べる

    年次季節性ありは、トレンドの曲がりやすさ × 季節性の効き具合の全組み合わせ（曲がりやすさが外側）。
    年次季節性なしは、季節性が無く効き具合が意味を持たないので、曲がりやすさだけ（効き具合は渡さない）。
    """
    if YEARLY_BY_METHOD[prophet]:
        return [{CHANGEPOINT_SCALE: cps, SEASONALITY_SCALE: sps}
                for cps in cps_grid for sps in sps_grid]
    return [{CHANGEPOINT_SCALE: cps} for cps in cps_grid]


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
                 steps: Optional[pd.DataFrame] = None,
                 settings: Optional[Mapping[str, float]] = None, **fit_options) -> Prophet:
    """
    Prophet を学習して返す

    年次季節性・holidays・段差の印（渡されたときだけ add_regressor）・settings（渡されたときだけ）
    以外は既定値（changepoint の位置の指定などはしない）。
    学習したモデルは1回しか使えないので、学び直すときは作り直す。
    """
    model = Prophet(yearly_seasonality=yearly, weekly_seasonality=False,
                    daily_seasonality=False, holidays=holidays, **(settings or {}))
    if steps is not None:
        for name in steps['step']:
            model.add_regressor(name)
    return model.fit(history, **fit_options)


def fit_prophet(train: pd.DataFrame, yearly: bool,
                week_column: str = 'week', value_column: str = 'share',
                fit_timeout: Optional[float] = FIT_TIMEOUT_SECONDS,
                holidays: Optional[pd.DataFrame] = None,
                steps: Optional[pd.DataFrame] = None,
                settings: Optional[Mapping[str, float]] = None) -> Prophet:
    """
    Prophet を学習して返す。時間内に終わらなければ、警告を出して Newton 法で学び直す

    処理の流れ:
      1. 学習データを Prophet の列名（ds / y）に直し、steps があれば段差の印の列を足す
      2. 年次季節性を yearly で切り替えて学習する（週・日の季節性は使わない。
         holidays があれば出来事、steps があれば説明変数、settings があれば Prophet の設定として渡す）
      3. fit_timeout 秒で終わらなければ、同じ引数のまま Newton 法で学び直す

    Args:
        train: 学習期間の週と値（1単位ぶん）
        yearly: 年次季節性を入れるか
        fit_timeout: 学習1回に待つ上限（秒）。None なら待ち続ける
        holidays: 出来事の表（launch_holidays が返す形）。None なら渡さない
        steps: 段差の印の表（launch_steps が返す形）。None なら渡さない
        settings: Prophet の設定（CHANGEPOINT_SCALE / SEASONALITY_SCALE をキーにした dict）。
            None か、入れていないキーは Prophet の既定値のまま
    """
    history = with_step_columns(pd.DataFrame({'ds': train[week_column].to_numpy(),
                                              'y': train[value_column].to_numpy()}), steps)
    try:
        return _fit_prophet(history, yearly, holidays, steps, settings, timeout=fit_timeout)
    except TimeoutError:
        warnings.warn(f'Prophet の学習が{fit_timeout}秒で終わらなかったので、Newton 法で学び直した'
                      f'（年次季節性{"あり" if yearly else "なし"}）', FitFallbackWarning)
        return _fit_prophet(history, yearly, holidays, steps, settings, algorithm='Newton')


def predict_prophet(model: Prophet, weeks: Sequence,
                    steps: Optional[pd.DataFrame] = None) -> np.ndarray:
    """学習した Prophet で、指定した週（昇順）の値（yhat）を予測する。steps があれば段差の印の列を足して渡す"""
    future = with_step_columns(pd.DataFrame({'ds': pd.DatetimeIndex(weeks)}), steps)
    return model.predict(future)['yhat'].to_numpy()


def forecast_prophet(train: pd.DataFrame, weeks: Sequence, yearly: bool,
                     week_column: str = 'week', value_column: str = 'share',
                     fit_timeout: Optional[float] = FIT_TIMEOUT_SECONDS,
                     holidays: Optional[pd.DataFrame] = None,
                     steps: Optional[pd.DataFrame] = None,
                     settings: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """
    Prophet で学習して、指定した週の値を予測する

    処理の流れ:
      1. weeks が昇順であることを確かめる
      2. Prophet を学習する（fit_prophet。holidays・steps・settings があれば渡す。
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
        settings: Prophet の設定（fit_prophet と同じ）。None なら既定値のまま

    Returns:
        weeks と同じ長さの予測値
    """
    weeks = pd.DatetimeIndex(weeks)
    if not weeks.is_monotonic_increasing:
        raise ValueError('weeks は昇順で渡す')

    model = fit_prophet(train, yearly, week_column, value_column, fit_timeout, holidays, steps,
                        settings)
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


class NoBaselineWeeksError(ValueError):
    """最新の発売の窓が学習期間の終わりまで続き、その後の週が1つも残らない（比べる相手を作れない）エラー"""


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
      3. 窓の終わりより後の週だけを残す。実績が1つも残らなければ NoBaselineWeeksError で止まる
    """
    if holidays is None:
        return train

    window_end = max(row.ds + pd.Timedelta(days=row.upper_window)
                     for row in holidays.itertuples())
    remaining = train[pd.DatetimeIndex(train[week_column]) > window_end]
    if remaining[value_column].dropna().empty:
        raise NoBaselineWeeksError('最新の発売の後の週が、学習期間に1つも残らない')
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
                             holidays: Optional[pd.DataFrame] = None,
                             prophet_settings: Optional[Mapping[str, Mapping[str, float]]] = None
                             ) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    1単位を4つの方法で予測し、テスト期間の当たり具合と、発売ごとの効き目を測る（段差の印を渡す版）

    evaluate_unit との違いは3つ。Prophet に、山の印（holidays）に加えて段差の印も渡す／
    比べる相手は、最新の発売の窓が終わった次の週以降の学習期間で作る／
    学習した Prophet から、発売ごとの効き目を読む。

    処理の流れ:
      1. テスト週を昇順に並べ、実績を取り出す
      2. 段差の印を付ける発売を選ぶ（発売週が、この単位の学習期間の最初の週より後のもの）
      3. Prophet を年次季節性あり・なしで学習し、テスト週を予測する（prophet_settings があれば、
         型ごとの設定を渡す）。学習したモデルから、発売ごとの効き目を読む
      4. 比べる相手2つ（学習期間の平均・直近の平均）を、最新の発売の窓が終わった次の週以降の
         学習期間で、テスト週と同じ長さで作る
      5. 4つの方法それぞれの MAE と、4通りの比を出す

    発売が付かない単位（holidays が None）は、evaluate_unit と同じ予測・比べる相手になる。

    Args:
        train: 学習期間の週と値（split_train_test の学習のうち、この単位）。欠測は除いてあること
        test: テスト期間の週と値（同テストのうち、この単位）。欠測は除いてあること
        recent_weeks: 「直近の平均」に使う、最後の週数
        holidays: この単位に渡す発売（launch_holidays が返す形）。None なら発売を渡さない
        prophet_settings: Prophet の型（PROPHET_METHODS の名前）→ その型に渡す設定（selected_settings
            が返す形）。None か、入れていない型は Prophet の既定値のまま

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
    for method, yearly in YEARLY_BY_METHOD.items():
        model = fit_prophet(train, yearly, week_column, value_column,
                            holidays=holidays, steps=steps,
                            settings=(prophet_settings or {}).get(method))
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


def evaluate_unit_settings(train: pd.DataFrame, validation: pd.DataFrame,
                           cps_grid: Sequence[float] = CPS_GRID,
                           sps_grid: Sequence[float] = SPS_GRID,
                           recent_weeks: int = RECENT_WEEKS,
                           week_column: str = 'week', value_column: str = 'share',
                           holidays: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    1単位で、Prophet の型 × 設定ごとに確かめ用の期間を予測し、比べる相手2つとの比を出す

    evaluate_unit_with_steps と同じ形（山の印と段差の印を渡し、比べる相手は最新の発売の窓が終わった
    次の週以降の学習期間で作る）で、Prophet の設定だけを変えて回す。

    処理の流れ:
      1. 確かめ週を昇順に並べ、実績を取り出す
      2. 段差の印を付ける発売を選ぶ（evaluate_unit_with_steps と同じ）
      3. 比べる相手2つの MAE を出す（設定によらないので1回だけ）。最新の発売の窓が学ぶ期間の終わりまで
         続いて、その後の週が1つも無ければ、比べる相手を作れないので NoBaselineWeeksError で止まる
         （呼び出し側が、その単位を設定を選ぶ段階から外す）
      4. 型ごとに、設定の一覧（prophet_settings_grid）を順に、Prophet を学習して確かめ週を予測し、
         MAE と、比べる相手2つとの比を出す

    Args:
        train: 学ぶ期間の週と値（split_validation の学ぶ期間のうち、この単位）。欠測は除いてあること
        validation: 確かめ用の期間の週と値（同じく、この単位）。欠測は除いてあること
        cps_grid: 試すトレンドの曲がりやすさ
        sps_grid: 試す季節性の効き具合（年次季節性ありの型だけ）
        recent_weeks: 「直近の平均」に使う、最後の週数
        holidays: この単位に渡す発売（設定を選ぶ段階の切る週で選んだもの）。None なら発売を渡さない

    Returns:
        型 × 設定ごとに1行のDataFrame。prophet / changepoint_prior_scale / seasonality_prior_scale
        （年次季節性なしの型は欠測）/ ratio_baseline_mean / ratio_baseline_recent
    """
    if validation.empty:
        raise ValueError('確かめ用の期間に実績が1つも無い')
    validation = validation.sort_values(week_column)
    weeks = pd.DatetimeIndex(validation[week_column])
    actual = validation[value_column].to_numpy(dtype=float)
    steps = launch_steps(holidays, train[week_column].min())

    baseline_train = keep_weeks_after_latest_launch(train, holidays, week_column, value_column)
    baseline_mae = {
        'baseline_mean': mae(actual, baseline_mean(
            baseline_train, len(weeks), week_column=week_column, value_column=value_column)),
        'baseline_recent': mae(actual, baseline_recent(
            baseline_train, len(weeks), recent_weeks=recent_weeks,
            week_column=week_column, value_column=value_column))}

    rows = []
    for method, yearly in YEARLY_BY_METHOD.items():
        for settings in prophet_settings_grid(method, cps_grid, sps_grid):
            model = fit_prophet(train, yearly, week_column, value_column,
                                holidays=holidays, steps=steps, settings=settings)
            mae_prophet = mae(actual, predict_prophet(model, weeks, steps))
            rows.append({'prophet': method,
                         CHANGEPOINT_SCALE: settings[CHANGEPOINT_SCALE],
                         SEASONALITY_SCALE: settings.get(SEASONALITY_SCALE, np.nan),
                         **{f'ratio_{baseline}': mae_ratio(mae_prophet, baseline_mae[baseline])
                            for baseline in BASELINE_METHODS}})
    return pd.DataFrame(rows)


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


def summarize_settings(results: pd.DataFrame) -> pd.DataFrame:
    """
    全単位の（型 × 設定）ごとの比から、型 × 設定ごとの勝った単位数と比の中央値をまとめる

    処理の流れ:
      1. 型 × 設定ごとに、比べる相手2つそれぞれで、比が1未満の単位を勝ちと数える
         （ちょうど1や欠測は勝ちにしない）
      2. 数えた単位数（units）と、2つの勝ち数の小さい方（min_wins）、2つの比の中央値（欠測は除く）を添える

    Args:
        results: 単位ごとの evaluate_unit_settings の結果を縦に積んだ表。
            prophet / changepoint_prior_scale / seasonality_prior_scale / ratio_<比べる相手> 2つの列を持つ

    Returns:
        型 × 設定ごとに1行（results に現れた順）。prophet / changepoint_prior_scale /
        seasonality_prior_scale / units（数えた単位数）/ wins_<比べる相手> 2つ / min_wins /
        median_ratio_<比べる相手> 2つ
    """
    keys = ['prophet', CHANGEPOINT_SCALE, SEASONALITY_SCALE]
    rows = []
    # 年次季節性なしの型は seasonality_prior_scale が欠測なので、欠測もキーとして残す
    for key, group in results.groupby(keys, sort=False, dropna=False):
        wins = {f'wins_{baseline}': int((group[f'ratio_{baseline}'] < 1).sum())
                for baseline in BASELINE_METHODS}
        medians = {f'median_ratio_{baseline}': group[f'ratio_{baseline}'].median()
                   for baseline in BASELINE_METHODS}
        rows.append({**dict(zip(keys, key)), 'units': len(group), **wins,
                     'min_wins': min(wins.values()), **medians})
    return pd.DataFrame(rows)


def choose_settings(summary: pd.DataFrame) -> pd.DataFrame:
    """
    型ごとに、Prophet の設定を1つ選ぶ（設定は全単位で1つ。単位ごとには選ばない）

    確かめ用の期間は1回きりなので、単位ごとに設定を選ぶと「たまたま当たった設定」を選んでしまう。

    処理の流れ:
      1. 並べ替えのキー（型の順・2つの中央値の平均・既定値との近さ・表の順）を作る
      2. 型ごとに、下の「選ぶ順」で並べ替える
      3. 型ごとに、いちばん上の行に selected を付ける

    選ぶ順（型ごと。上のものほど優先）:
      1. min_wins（比べる相手2つに勝った単位数の小さい方）が大きい
      2. 2つの比の中央値の平均が小さい（欠測は最後）
      3. Prophet の既定値（PROPHET_DEFAULT_SETTINGS）に近い。近さは、設定ごとに
         「既定値との比の常用対数の絶対値」を足したもの（年次季節性なしの型は曲がりやすさだけ）
      4. それでも同点なら、summary で先に並んでいる方

    Args:
        summary: summarize_settings の結果

    Returns:
        summary を、型（PROPHET_METHODS の順）ごとに選ぶ順に並べ替え、selected 列
        （型ごとにいちばん上の行だけ True）を足した表
    """
    median_columns = [f'median_ratio_{baseline}' for baseline in BASELINE_METHODS]

    def log_distance(column):
        """既定値との比の常用対数の絶対値（欠測は欠測のまま）"""
        return np.log10(summary[column] / PROPHET_DEFAULT_SETTINGS[column]).abs()

    ranking = summary.assign(
        _type=summary['prophet'].map(PROPHET_METHODS.index),
        _median=summary[median_columns].mean(axis=1, skipna=False),
        # 浮動小数点の誤差で、本来同じ近さが違って見えないよう丸める
        _distance=(log_distance(CHANGEPOINT_SCALE)
                   + log_distance(SEASONALITY_SCALE).fillna(0)).round(9),
        _order=np.arange(len(summary)))
    ranking = ranking.sort_values(['_type', 'min_wins', '_median', '_distance', '_order'],
                                  ascending=[True, False, True, True, True])
    ranking['selected'] = (~ranking['prophet'].duplicated()).to_numpy()
    return ranking[[*summary.columns, 'selected']].reset_index(drop=True)


def selected_settings(table: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    choose_settings の表から、選んだ設定を、型 → Prophet に渡す設定（dict）にする

    年次季節性なしの型には、季節性の効き具合を入れない（欠測の設定は入れない）。
    """
    chosen = {}
    for row in table[table['selected']].to_dict('records'):
        settings = {name: row[name] for name in (CHANGEPOINT_SCALE, SEASONALITY_SCALE)}
        chosen[row['prophet']] = {name: value for name, value in settings.items()
                                  if not pd.isna(value)}
    return chosen
