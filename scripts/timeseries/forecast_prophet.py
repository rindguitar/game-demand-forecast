"""
週次シェアを Prophet で予測し、単純な予測と当たり具合を比べる（Phase 7・Issue #41）

build_weekly_series.py が出した weekly_series_all.csv（全ゲームの中でのシェア）を読み、
単位（トピック）ごとに4つの方法でテスト期間を予測して、MAE を比べる
（予測するのがシェアである理由は docs/decisions.md 2026-10-03）。

  Prophet（年次季節性あり） / Prophet（年次季節性なし）
  比べる相手①: 学習期間の平均 / 比べる相手②: 直近の平均（学習期間の最後の --recent-weeks 週）

全単位で同じ週で切り、最後の --test-weeks 週をテスト、その前を学習にする。
比 = MAE(Prophet) ÷ MAE(比べる相手)。1未満なら Prophet の勝ち。
Prophet の予測が負になってもクリップしない（負になった数を画面に出す）。
既定値は64本のファイルを指す（data/timeseries/weekly_64/）。

--launch-events を付けると、発売を Prophet に出来事（holidays）として渡す。
発売直後に言及の過半数をそのゲームが占める単位には、その発売を渡して、山を「発売のせい」と
学ばせる（年次季節性が、1回きりの山を毎年の山として覚えるのを防ぐ）。
比べる相手2つも、学習期間から発売後の週を除いて作る。出力先は既定で別のディレクトリになる。

--launch-steps を付けると（--launch-events も有効になる）、発売を水準の段差としても渡す
（発売が水準を押し上げたまま残る場合のため。Issue #60）。発売週が学習期間の最初の週より後の発売には、
発売週から後を1とする説明変数（段差の印）を足す。比べる相手2つは、最新の発売の窓が終わった次の週
から学習期間の最後までで作る。発売ごとの効き目を launch_effects.csv に書き、勝ちの基準の判定を出す。

--tune を付けると（--launch-steps も有効になる）、テスト期間を見ずに Prophet の設定を選ぶ（Issue #60 ②）。
学習期間の後ろ --validation-weeks 週を「確かめ用の期間」にして、その前で学び、設定ごと（--cps-grid ×
--sps-grid）に確かめ用の期間を予測する。比べる相手2つの両方に勝った単位数の小さい方が最も大きい設定を、
Prophet の型ごとに全単位で1つ選び、その設定で学習期間の全部から学び直してテスト期間を測る。
確かめ用の期間の成績は tuning.csv に書く。

処理の流れ:
  1. 週次シェアを読み、全単位を同じ週で学習とテストに分ける
     （--tune のときは、学習期間をさらに、学ぶ期間と確かめ用の期間に分ける）
  2. （--launch-events のとき）レビューと台帳を読み、単位ごとに渡す発売を選ぶ
     （--tune のときは、確かめ用の期間の最初の週で切って選び直した発売も）
  3. （--tune のとき）確かめ用の期間で、設定ごとに全単位を予測し、型ごとに設定を1つ選ぶ
  4. 単位ごとに4つの方法で予測し、MAE と比を出す
  5. forecasts.csv / metrics.csv / summary.csv（--launch-events のときは launch_events.csv も、
     --launch-steps のときは launch_effects.csv も、--tune のときは tuning.csv も）を書く
  6. 4通りの比較（勝った単位数・比の中央値）と、負の予測の数を表示する
     （--launch-steps のときは、段差の印の数と、勝ちの基準の判定も）
  7. 単位ごとの小さい図を並べる（--no-plot で省く）

使い方:
    make forecast-prophet
    make forecast-prophet FORECAST_ARGS="--launch-events"
    make forecast-prophet FORECAST_ARGS="--launch-steps"
    make forecast-prophet FORECAST_ARGS="--tune"
    docker compose exec dev python scripts/timeseries/forecast_prophet.py
"""

import argparse
import logging
import os
import sys
import time
import warnings

import pandas as pd
from cmdstanpy.utils import get_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.timeseries.forecast import (  # noqa: E402
    BASELINE_METHODS,
    CHANGEPOINT_SCALE,
    CPS_GRID,
    LAUNCH_MIN_SHARE,
    LAUNCH_MIN_WEEKLY_MENTIONS,
    LAUNCH_WEEKS,
    METHODS,
    PROPHET_METHODS,
    RECENT_WEEKS,
    SEASONALITY_SCALE,
    SPS_GRID,
    TEST_WEEKS,
    VALIDATION_WEEKS,
    WIN_CRITERION_UNITS,
    FitFallbackWarning,
    NoBaselineWeeksError,
    check_win_criterion,
    choose_settings,
    evaluate_unit,
    evaluate_unit_settings,
    evaluate_unit_with_steps,
    find_target_launches,
    launch_effects_table,
    launch_holidays,
    parse_grid,
    select_launch_events,
    selected_settings,
    split_train_test,
    split_validation,
    summarize_comparisons,
    summarize_settings,
)
from src.timeseries.weekly import add_week_column  # noqa: E402
from src.visualization.timeseries_plots import plot_forecast_grid  # noqa: E402

# 図1枚に並べる単位の数。plot_weekly_series.py の全ゲームパネルは59単位を1枚に載せていて
# 縦に長すぎるので、その半分ほどで分ける（64本なら2枚）
UNITS_PER_PLOT = 30

# 設定を選ぶ段階で、同じ内容を1回にまとめて表示するログの出し元（Prophet 本体・Prophet の最適化・cmdstanpy）
PROPHET_LOGGERS = ('prophet', 'prophet.models', 'cmdstanpy')

# 出力先の既定値。--launch-events / --launch-steps / --tune のときは、それより前の結果を上書きしないよう別にする
DEFAULT_OUTPUT_DIR = 'data/timeseries/forecast_64'
LAUNCH_OUTPUT_DIR = 'data/timeseries/forecast_64_launch'
STEP_OUTPUT_DIR = 'data/timeseries/forecast_64_launch_step'
TUNE_OUTPUT_DIR = 'data/timeseries/forecast_64_tuned'


def grid_argument(text):
    """--cps-grid / --sps-grid の値（カンマ区切りの数）を読む。読めなければ、理由つきで argparse に伝える"""
    try:
        return parse_grid(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default='data/timeseries/weekly_64/weekly_series_all.csv',
                        help='build_weekly_series.py が出した週次系列CSV（既定は64本のシェア）')
    parser.add_argument('--value-col', default='share',
                        help='予測する値の列（既定 %(default)s）')
    parser.add_argument('--test-weeks', type=int, default=TEST_WEEKS,
                        help='テスト期間の週数。最後からこの週数をテストにする（既定 %(default)s）')
    parser.add_argument('--recent-weeks', type=int, default=RECENT_WEEKS,
                        help='「直近の平均」に使う、学習期間の最後の週数（既定 %(default)s）')
    parser.add_argument('--launch-events', action='store_true',
                        help='発売を Prophet に出来事（holidays）として渡す。比べる相手の学習期間からも'
                             '発売後の週を除く')
    parser.add_argument('--launch-steps', action='store_true',
                        help='発売を水準の段差としても渡す（--launch-events も有効になる）。'
                             '発売週が学習期間の最初の週より後の発売に、発売週から後を1とする説明変数を足す。'
                             '比べる相手は、最新の発売の窓が終わった次の週から学習期間の最後までで作る。'
                             'launch_effects.csv を書き、勝ちの基準の判定を出す')
    parser.add_argument('--tune', action='store_true',
                        help='テスト期間を見ずに Prophet の設定を選ぶ（--launch-steps も有効になる）。'
                             '学習期間の後ろを確かめ用の期間にして、設定ごとに予測し、'
                             '比べる相手2つに勝った単位数の小さい方が最も大きい設定を、型ごとに全単位で'
                             '1つ選ぶ。その設定で学習期間の全部から学び直してテスト期間を測る。'
                             'tuning.csv を書く')
    parser.add_argument('--validation-weeks', type=int, default=VALIDATION_WEEKS,
                        help='確かめ用の期間の週数。学習期間の最後からこの週数（--tune のときだけ使う。'
                             '既定 %(default)s）')
    parser.add_argument('--cps-grid', type=grid_argument, default=CPS_GRID,
                        help='試すトレンドの曲がりやすさ（changepoint_prior_scale）。カンマ区切り'
                             '（--tune のときだけ使う。既定 %(default)s）')
    parser.add_argument('--sps-grid', type=grid_argument, default=SPS_GRID,
                        help='試す季節性の効き具合（seasonality_prior_scale）。カンマ区切り。'
                             '年次季節性ありの型だけで試す（--tune のときだけ使う。既定 %(default)s）')
    parser.add_argument('--win-criterion', type=int, default=WIN_CRITERION_UNITS,
                        help='勝ちの基準。Prophet の型ごとに、比べる相手2つの両方に、この単位数以上で'
                             '勝てば届いた扱い（--launch-steps のときだけ判定する。既定 %(default)s）')
    parser.add_argument('--reviews',
                        default='data/timeseries/reviews_timeseries_with_topics_64.csv',
                        help='トピック付与済みレビューCSV。--launch-events のときだけ読む'
                             '（game_name / timestamp_created / topic_id 列だけ使う）')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。--launch-events のときだけ読む（発売日を使う）')
    parser.add_argument('--launch-weeks', type=int, default=LAUNCH_WEEKS,
                        help='発売週から数えて、発売の影響を見る週数（既定 %(default)s）')
    parser.add_argument('--launch-min-share', type=float, default=LAUNCH_MIN_SHARE,
                        help='発売を渡す条件。その単位のレビューのうち、発売したゲームが占める割合の'
                             '下限（既定 %(default)s = 過半数。ちょうどでも渡す）')
    parser.add_argument('--launch-min-weekly', type=int, default=LAUNCH_MIN_WEEKLY_MENTIONS,
                        help='発売を渡す条件。発売したゲームの、その単位でのレビュー数の下限'
                             '（1週あたり。既定 %(default)s）')
    parser.add_argument('--output-dir', default=None,
                        help=f'出力先ディレクトリ（既定 {DEFAULT_OUTPUT_DIR}。'
                             f'--launch-events のときは {LAUNCH_OUTPUT_DIR}、'
                             f'--launch-steps のときは {STEP_OUTPUT_DIR}、'
                             f'--tune のときは {TUNE_OUTPUT_DIR}）')
    parser.add_argument('--plot-weeks', type=int, default=52,
                        help='図に出す学習期間の週数。学習期間の最後からこの週数（既定 %(default)s）')
    parser.add_argument('--no-plot', action='store_true',
                        help='図を描かない')
    args = parser.parse_args()

    # --tune は --launch-steps の上に、--launch-steps は --launch-events の上に重ねるので、
    # 付けたら下のオプションも有効にする
    if args.tune:
        args.launch_steps = True
    if args.launch_steps:
        args.launch_events = True
    if args.output_dir is None:
        args.output_dir = (TUNE_OUTPUT_DIR if args.tune
                           else STEP_OUTPUT_DIR if args.launch_steps
                           else LAUNCH_OUTPUT_DIR if args.launch_events else DEFAULT_OUTPUT_DIR)
    return args


def quiet_prophet_logs():
    """Prophet と cmdstanpy の INFO ログを止める（WARNING 以上は残す）"""
    # cmdstanpy のロガーは最初に呼ばれたときにレベルを初期化するので、先に作ってから設定する
    get_logger().setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)


def choose_launch_events(args, series, units, cutoff, validation_cutoff=None):
    """
    レビューと台帳を読み、単位ごとに渡す発売を選んで、件数を表示する

    validation_cutoff（確かめ用の期間の最初の週）を渡したとき（--tune）は、設定を選ぶ段階の発売も、
    その週で切って選び直す（その週以降のレビュー・発売は使わない）。
    返すのは (テスト期間を測る発売, 設定を選ぶ段階の発売)。validation_cutoff が None なら後ろは None
    """
    # 1. レビューは必要な3列だけ読み（726MBあるため）、週の列を足す。台帳も読む
    reviews = add_week_column(pd.read_csv(
        args.reviews, usecols=['game_name', 'timestamp_created', 'topic_id']))
    games = pd.read_csv(args.games)
    first_week = series['week'].min()

    # 2. テスト期間を測る発売を選ぶ（切る週 = テストの最初の週）
    targets, events = choose_launch_events_at(
        args, reviews, games, units, first_week, cutoff, '発売を出来事として渡す（--launch-events）')
    if validation_cutoff is None:
        return events, None

    # 3. 設定を選ぶ段階の発売を選び直し、この段階で対象外になった発売を表示する
    tune_targets, tune_events = choose_launch_events_at(
        args, reviews, games, units, first_week, validation_cutoff,
        '設定を選ぶ段階の発売（--tune。確かめ用の期間の最初の週で切って選び直す）')
    excluded = targets[~targets['game'].isin(tune_targets['game'])]
    names = '・'.join(f"{row.game}（{row.release_week:%Y-%m-%d}）" for row in excluded.itertuples())
    print(f"  この段階で対象外の発売（テスト期間を測る発売のうち、発売週が確かめ用の期間以降）: "
          f"{names or 'なし'}")
    return events, tune_events


def choose_launch_events_at(args, reviews, games, units, first_week, cutoff, title):
    """切る週 cutoff で、単位ごとに渡す発売を選んで、件数を表示する。返すのは (対象の発売, 選んだ組)"""
    # 1. 対象の発売を数え、単位ごとに渡す発売を選ぶ
    targets = find_target_launches(games, first_week, cutoff, args.launch_weeks)
    events = select_launch_events(
        reviews, games, units, first_week, cutoff, launch_weeks=args.launch_weeks,
        min_share=args.launch_min_share, min_weekly_mentions=args.launch_min_weekly)

    # 2. 発売の数・組の数・発売が付いた単位／付かなかった単位の数を表示する
    with_launch = events['unit'].nunique()
    print(f"\n{title}")
    print(f"  対象の発売: {len(targets)}本"
          f"（発売週から{args.launch_weeks}週がデータ期間と重なり、発売週が切る週"
          f" {cutoff:%Y-%m-%d} より前）")
    print(f"  選んだ組（単位 × 発売）: {len(events)}組・発売{events['game'].nunique()}本"
          f"（発売したゲームの割合が{args.launch_min_share:g}以上、かつレビュー数が"
          f"週{args.launch_min_weekly}件 × 確かめる週数 以上）")
    print(f"  発売が付いた単位: {with_launch} / 付かなかった単位: {len(units) - with_launch}"
          f"（付かなかった単位は発売を渡さない）")
    return targets, events


def print_summary(summary, forecasts_test):
    """4通りの比較の表と、Prophet の負の予測の数を表示する"""
    print(f"\n{'=' * 74}\n4通りの比較（比 = Prophet の MAE ÷ 比べる相手の MAE。1未満なら Prophet の勝ち）"
          f"\n{'=' * 74}")
    print(summary.to_string(index=False, formatters={'median_ratio': '{:.3f}'.format}))

    print('\nProphet の負の予測（クリップはしていない）:')
    for method in PROPHET_METHODS:
        negative = forecasts_test[method] < 0
        print(f"  {method:<18}: {negative.sum():,} / {len(negative):,}件"
              f"（{forecasts_test.loc[negative, 'unit'].nunique()}単位）")


def print_step_counts(launch_effects):
    """段差の印を付けた組・付けなかった組（単位 × 発売）の数を表示する"""
    # Prophet の型で組の数は変わらないので、1つ目の型だけで数える
    pairs = launch_effects[launch_effects['prophet'] == PROPHET_METHODS[0]]
    with_step = int(pairs['has_step'].sum())
    print(f"\n段差の印（--launch-steps）: 付けた組 {with_step} / 付けなかった組 {len(pairs) - with_step}"
          f"（発売週が学習期間の最初の週以前の発売には付けない）")


def print_win_criterion(criterion):
    """勝ちの基準に届いたかを、Prophet の型ごとに表示する"""
    required, units = criterion['required_wins'].iloc[0], criterion['units'].iloc[0]
    print(f"\n勝ちの基準（比べる相手2つの両方に {required} / {units} 単位以上で勝つ）:")
    for row in criterion.itertuples():
        wins = '・'.join(f"{baseline} {getattr(row, f'wins_{baseline}')}"
                        for baseline in BASELINE_METHODS)
        print(f"  {row.prophet:<18}: {wins} → {'届いた' if row.reached else '届かない'}")


def call_reporting_warnings(unit, function, *args, **kwargs):
    """
    function を呼び、出た警告を、どの単位かを添えてその場で表示する

    返すのは (function の結果, Newton 法に切り替えた学習の数)。
    警告（学習が時間内に終わらなかった等）は、どの単位かが分からないと読めないため。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(*args, **kwargs)
    for item in caught:
        print(f"  ⚠️ t{unit}: {item.message}")
    return result, sum(issubclass(item.category, FitFallbackWarning) for item in caught)


def print_tuning(tuning, units):
    """確かめ用の期間の、型 × 設定ごとの成績を、選ぶ順（型ごとに上ほど先に選ばれる）に表示する"""
    print(f"\n{'=' * 74}\n確かめ用の期間での設定ごとの成績（型ごとに選ぶ順。* = 選んだ設定）\n{'=' * 74}")
    names = {CHANGEPOINT_SCALE: 'cps', SEASONALITY_SCALE: 'sps',
             'wins_baseline_mean': 'wins_mean', 'wins_baseline_recent': 'wins_recent',
             'min_wins': 'min', 'median_ratio_baseline_mean': 'median_mean',
             'median_ratio_baseline_recent': 'median_recent'}
    formatters = {'cps': '{:g}'.format, 'sps': '{:g}'.format,
                  'median_mean': '{:.3f}'.format, 'median_recent': '{:.3f}'.format,
                  'selected': lambda value: '*' if value else ''}
    # 年次季節性なしの型には季節性の効き具合が無いので、欠測は - で表す
    shown = tuning.drop(columns='units').rename(columns=names)    # units はどの行も同じなので、凡例に書く
    print(shown.to_string(index=False, formatters=formatters, na_rep='-'))
    print(f"cps = トレンドの曲がりやすさ・sps = 季節性の効き具合（年次季節性ありの型だけ）\n"
          f"wins_mean / wins_recent = 学習期間の平均 / 直近の平均 に勝った単位数（全{units}単位）・"
          f"min = その小さい方・median = 比の中央値")


def print_chosen_settings(tuning, units):
    """選んだ設定（型ごとに1つ）と、その確かめ用の期間での成績を表示する"""
    chosen = selected_settings(tuning)
    print("\n選んだ設定（全単位で1つ。確かめ用の期間で、比べる相手2つに勝った単位数の小さい方が"
          "最も大きい設定）:")
    for row in tuning[tuning['selected']].itertuples():
        settings = '・'.join(f"{name}={value:g}" for name, value in chosen[row.prophet].items())
        print(f"  {row.prophet:<18}: {settings}"
              f"（確かめ用の期間で、学習期間の平均に {row.wins_baseline_mean} / "
              f"直近の平均に {row.wins_baseline_recent} / {units} 単位で勝った）")


class OncePerMessage(logging.Filter):
    """同じ内容のログを、最初の1回だけ通し、回数を数える（学習のたびに出る同じ警告で、画面が埋まらないように）"""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def filter(self, record):
        message = record.getMessage()
        self.counts[message] = self.counts.get(message, 0) + 1
        return self.counts[message] == 1


def tune_settings(args, learn, validation, units, tune_events):
    """
    確かめ用の期間で、型 × 設定ごとに全単位を予測し、設定を型ごとに1つ選ぶ。返すのは tuning 表

    処理の流れ:
      1. 単位ごとに、型 × 設定ごとの、比べる相手2つとの比を出す（evaluate_unit_settings）。
         比べる相手を作れない単位（最新の発売の窓が学ぶ期間の終わりまで続く）は、ここから外す
      2. 型 × 設定ごとに、勝った単位数と比の中央値をまとめる（summarize_settings）
      3. 型ごとに、設定を1つ選ぶ（choose_settings）。全単位で1つで、選ぶ順に並べた表になる
      4. 設定ごとの成績の表と、選んだ設定を表示する
    """
    started = time.perf_counter()
    learn_by_unit = dict(tuple(learn.groupby('unit')))
    validation_by_unit = dict(tuple(validation.groupby('unit')))

    # 1. 単位ごとに、型 × 設定ごとの比を出す
    print(f"\n確かめ用の期間で設定を選ぶ（{len(units)}単位 × Prophet の型 × 設定を学習する。"
          f"Prophet の同じログは1回だけ表示し、回数を最後に出す）")
    once = OncePerMessage()
    for name in PROPHET_LOGGERS:
        logging.getLogger(name).addFilter(once)
    results, skipped, fallbacks = [], [], 0
    for done, unit in enumerate(units, start=1):
        holidays = launch_holidays(tune_events, unit, args.launch_weeks)
        try:
            unit_results, unit_fallbacks = call_reporting_warnings(
                unit, evaluate_unit_settings, learn_by_unit[unit], validation_by_unit[unit],
                args.cps_grid, args.sps_grid, recent_weeks=args.recent_weeks,
                value_column=args.value_col, holidays=holidays)
        except NoBaselineWeeksError as error:
            print(f"  ⚠️ t{unit}: 設定を選ぶ段階から外した（{error}）")
            skipped.append(unit)
        else:
            fallbacks += unit_fallbacks
            results.append(unit_results.assign(unit=unit))
        if done % 10 == 0 or done == len(units):
            print(f"  {done}/{len(units)}単位を確かめた")
    for name in PROPHET_LOGGERS:
        logging.getLogger(name).removeFilter(once)

    # 2. 型 × 設定ごとにまとめ、3. 型ごとに設定を選ぶ
    tuning = choose_settings(summarize_settings(pd.concat(results, ignore_index=True)))

    # 4. 設定ごとの成績の表と、選んだ設定を表示する
    scored = len(units) - len(skipped)
    print_tuning(tuning, scored)
    print_chosen_settings(tuning, scored)
    if skipped:
        print(f"\n設定を選ぶ段階から外した単位: {len(skipped)}"
              f"（{'・'.join(f't{unit}' for unit in skipped)}。比べる相手を作れなかった）")
    if once.counts:
        print("\nProphet・cmdstanpy のログ（同じ内容は最初の1回だけ表示した。回数）:")
        for message, count in once.counts.items():
            print(f"  {count}回: {message[:70]}")
    print(f"\n確かめ用の期間で Newton 法に切り替えた学習: {fallbacks}回"
          f"（確かめ用の期間にかかった時間: {time.perf_counter() - started:.1f}秒）")
    return tuning


def plot_forecasts(forecasts, keywords, args, cutoff, events=None):
    """単位ごとの小さい図を、UNITS_PER_PLOT 単位ずつ何枚かに分けて描く。events があれば発売の週も引く"""
    plot_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 中央値の大きい順（plot_weekly_series.py の図と同じ並び）に、1枚ぶんずつ区切る
    order = forecasts.groupby('unit')['actual'].median().sort_values(ascending=False).index
    pages = [order[i:i + UNITS_PER_PLOT] for i in range(0, len(order), UNITS_PER_PLOT)]
    plot_data = forecasts.assign(keywords=forecasts['unit'].map(keywords))

    for number, units in enumerate(pages, start=1):
        first = (number - 1) * UNITS_PER_PLOT + 1
        title = (f"Weekly {args.value_col}: actual vs forecast  "
                 f"(topics ranked {first}-{first + len(units) - 1} by median, "
                 f"page {number}/{len(pages)})\n"
                 f"test = last {args.test_weeks} weeks from {cutoff:%Y-%m-%d}   "
                 f"recent mean = last {args.recent_weeks} weeks   "
                 f"grey = last {args.plot_weeks} train weeks")
        path = os.path.join(plot_dir, f'forecast_{number:02d}.png')
        plot_forecast_grid(plot_data, title, path, units=units, history_weeks=args.plot_weeks,
                           launches=events)
        print(f"  ✅ 図: {path}（{len(units)}単位）")


def main():
    args = parse_args()
    started = time.perf_counter()
    quiet_prophet_logs()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 週次シェアを読み、全単位を同じ週で学習とテストに分ける
    # 実績を元のCSVと1桁も違えず読むため、float_precision を指定する
    series = pd.read_csv(args.input, parse_dates=['week'], float_precision='round_trip')
    train, test, cutoff = split_train_test(series, args.test_weeks, value_column=args.value_col)
    units = sorted(series['unit'].unique())
    print(f"入力: {args.input}（{len(units)}単位 × {series['week'].nunique()}週）")
    print(f"学習: {train['week'].nunique()}週（{train['week'].min():%Y-%m-%d} 〜 "
          f"{train['week'].max():%Y-%m-%d}）")
    print(f"テスト: {test['week'].nunique()}週（{test['week'].min():%Y-%m-%d} 〜 "
          f"{test['week'].max():%Y-%m-%d}）")
    print(f"  {args.value_col} が欠測のため除いた行: {series[args.value_col].isna().sum():,}")

    # --tune のときは、学習期間をさらに、学ぶ期間と確かめ用の期間に分ける（テスト期間の行は使わない）
    learn = validation = validation_cutoff = None
    if args.tune:
        learn, validation, validation_cutoff = split_validation(
            series, cutoff, args.validation_weeks, value_column=args.value_col)
        print("\n確かめ用の期間（--tune。設定を選ぶ段階では、テスト期間のデータを使わない）")
        print(f"  学ぶ期間: {learn['week'].nunique()}週（{learn['week'].min():%Y-%m-%d} 〜 "
              f"{learn['week'].max():%Y-%m-%d}）")
        print(f"  確かめ用の期間: {validation['week'].nunique()}週"
              f"（{validation['week'].min():%Y-%m-%d} 〜 {validation['week'].max():%Y-%m-%d}）")

    # 2. 発売を渡すときは、レビューと台帳から、単位ごとに渡す発売を選ぶ
    #    （--tune なら、確かめ用の期間の最初の週で切って選び直した発売も）
    events, tune_events = (choose_launch_events(args, series, units, cutoff, validation_cutoff)
                           if args.launch_events else (None, None))

    # 3. --tune のときは、確かめ用の期間で、設定ごとに全単位を予測し、型ごとに設定を1つ選ぶ
    tuning = chosen = None
    if args.tune:
        tuning = tune_settings(args, learn, validation, units, tune_events)
        chosen = selected_settings(tuning)

    # 4. 単位ごとに4つの方法で予測し、MAE と比を出す（--tune のときは、選んだ設定で学習期間の全部から学ぶ）
    info = series.drop_duplicates('unit').set_index('unit')[['category', 'keywords']]
    train_by_unit = dict(tuple(train.groupby('unit')))
    test_by_unit = dict(tuple(test.groupby('unit')))
    metrics_rows, prediction_frames, effects_by_unit, fallbacks = [], [], {}, 0
    for done, unit in enumerate(units, start=1):
        # 発売が付かない単位は None（Prophet に holidays を渡さない）になる
        holidays = (launch_holidays(events, unit, args.launch_weeks)
                    if events is not None else None)
        options = {'recent_weeks': args.recent_weeks, 'value_column': args.value_col,
                   'holidays': holidays}
        # --launch-steps のときは、段差の印を渡し、発売の効き目も一緒に受け取る
        if args.launch_steps:
            result, unit_fallbacks = call_reporting_warnings(
                unit, evaluate_unit_with_steps, train_by_unit[unit], test_by_unit[unit],
                prophet_settings=chosen, **options)
            unit_metrics, predictions, effects_by_unit[unit] = result
        else:
            result, unit_fallbacks = call_reporting_warnings(
                unit, evaluate_unit, train_by_unit[unit], test_by_unit[unit], **options)
            unit_metrics, predictions = result
        fallbacks += unit_fallbacks
        metrics_rows.append({'unit': unit, 'category': info.at[unit, 'category'],
                             'keywords': info.at[unit, 'keywords'], **unit_metrics})
        prediction_frames.append(predictions.assign(unit=unit))
        if done % 10 == 0 or done == len(units):
            print(f"  {done}/{len(units)}単位を評価した")

    # 5. 結果を書く。forecasts.csv は学習週も入れ、予測の列はテスト週だけ値を持つ
    metrics = pd.DataFrame(metrics_rows)
    summary = summarize_comparisons(metrics)
    future = pd.concat(prediction_frames, ignore_index=True).assign(split='test')
    history = (train[['unit', 'week', args.value_col]]
               .rename(columns={args.value_col: 'actual'}).assign(split='train'))
    columns = ['unit', 'week', 'split', 'actual', *METHODS]
    forecasts = (pd.concat([history, future], ignore_index=True)
                 .sort_values(['unit', 'week'])[columns])

    tables = [('forecasts', forecasts), ('metrics', metrics), ('summary', summary)]
    if events is not None:
        tables.append(('launch_events', events))
    if args.launch_steps:
        launch_effects = launch_effects_table(effects_by_unit, info['keywords'])
        tables.append(('launch_effects', launch_effects))
    if tuning is not None:
        tables.append(('tuning', tuning))
    print()
    for name, table in tables:
        path = os.path.join(args.output_dir, f'{name}.csv')
        table.to_csv(path, index=False)
        print(f"  ✅ 出力: {path}（{len(table):,}行）")

    # 6. 4通りの比較と、負の予測の数を表示する（--launch-steps なら、段差の印の数と勝ちの基準の判定も）
    if args.launch_steps:
        print_step_counts(launch_effects)
    print_summary(summary, future)
    if args.launch_steps:
        print_win_criterion(check_win_criterion(summary, args.win_criterion))
    print(f"\nNewton 法に切り替えた学習: {fallbacks}回"
          f"（時間内に終わらなかったもの。ほかの学習は Prophet の既定のまま）")

    # 7. 単位ごとの小さい図を並べる
    if not args.no_plot:
        print()
        plot_forecasts(forecasts, info['keywords'], args, cutoff, events)

    print(f"\n実行時間: {time.perf_counter() - started:.1f}秒")


if __name__ == '__main__':
    main()
