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

処理の流れ:
  1. 週次シェアを読み、全単位を同じ週で学習とテストに分ける
  2. 単位ごとに4つの方法で予測し、MAE と比を出す
  3. forecasts.csv / metrics.csv / summary.csv を書く
  4. 4通りの比較（勝った単位数・比の中央値）と、負の予測の数を表示する
  5. 単位ごとの小さい図を並べる（--no-plot で省く）

使い方:
    make forecast-prophet
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
    METHODS,
    PROPHET_METHODS,
    RECENT_WEEKS,
    TEST_WEEKS,
    FitFallbackWarning,
    evaluate_unit,
    split_train_test,
    summarize_comparisons,
)
from src.visualization.timeseries_plots import plot_forecast_grid  # noqa: E402

# 図1枚に並べる単位の数。plot_weekly_series.py の全ゲームパネルは59単位を1枚に載せていて
# 縦に長すぎるので、その半分ほどで分ける（64本なら2枚）
UNITS_PER_PLOT = 30


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
    parser.add_argument('--output-dir', default='data/timeseries/forecast_64',
                        help='出力先ディレクトリ')
    parser.add_argument('--plot-weeks', type=int, default=52,
                        help='図に出す学習期間の週数。学習期間の最後からこの週数（既定 %(default)s）')
    parser.add_argument('--no-plot', action='store_true',
                        help='図を描かない')
    return parser.parse_args()


def quiet_prophet_logs():
    """Prophet と cmdstanpy の INFO ログを止める（WARNING 以上は残す）"""
    # cmdstanpy のロガーは最初に呼ばれたときにレベルを初期化するので、先に作ってから設定する
    get_logger().setLevel(logging.WARNING)
    logging.getLogger('prophet').setLevel(logging.WARNING)


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


def plot_forecasts(forecasts, keywords, args, cutoff):
    """単位ごとの小さい図を、UNITS_PER_PLOT 単位ずつ何枚かに分けて描く"""
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
        plot_forecast_grid(plot_data, title, path, units=units, history_weeks=args.plot_weeks)
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

    # 2. 単位ごとに4つの方法で予測し、MAE と比を出す
    info = series.drop_duplicates('unit').set_index('unit')[['category', 'keywords']]
    train_by_unit = dict(tuple(train.groupby('unit')))
    test_by_unit = dict(tuple(test.groupby('unit')))
    metrics_rows, prediction_frames, fallbacks = [], [], 0
    for done, unit in enumerate(units, start=1):
        # 警告（学習が時間内に終わらなかった等）は、どの単位かを添えてその場で表示する
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            unit_metrics, predictions = evaluate_unit(
                train_by_unit[unit], test_by_unit[unit],
                recent_weeks=args.recent_weeks, value_column=args.value_col)
        for item in caught:
            print(f"  ⚠️ t{unit}: {item.message}")
        fallbacks += sum(issubclass(item.category, FitFallbackWarning) for item in caught)
        metrics_rows.append({'unit': unit, 'category': info.at[unit, 'category'],
                             'keywords': info.at[unit, 'keywords'], **unit_metrics})
        prediction_frames.append(predictions.assign(unit=unit))
        if done % 10 == 0 or done == len(units):
            print(f"  {done}/{len(units)}単位を評価した")

    # 3. 結果を書く。forecasts.csv は学習週も入れ、予測の列はテスト週だけ値を持つ
    metrics = pd.DataFrame(metrics_rows)
    summary = summarize_comparisons(metrics)
    future = pd.concat(prediction_frames, ignore_index=True).assign(split='test')
    history = (train[['unit', 'week', args.value_col]]
               .rename(columns={args.value_col: 'actual'}).assign(split='train'))
    columns = ['unit', 'week', 'split', 'actual', *METHODS]
    forecasts = (pd.concat([history, future], ignore_index=True)
                 .sort_values(['unit', 'week'])[columns])

    print()
    for name, table in [('forecasts', forecasts), ('metrics', metrics), ('summary', summary)]:
        path = os.path.join(args.output_dir, f'{name}.csv')
        table.to_csv(path, index=False)
        print(f"  ✅ 出力: {path}（{len(table):,}行）")

    # 4. 4通りの比較と、負の予測の数を表示する
    print_summary(summary, future)
    print(f"\nNewton 法に切り替えた学習: {fallbacks}回"
          f"（時間内に終わらなかったもの。ほかの学習は Prophet の既定のまま）")

    # 5. 単位ごとの小さい図を並べる
    if not args.no_plot:
        print()
        plot_forecasts(forecasts, info['keywords'], args, cutoff)

    print(f"\n実行時間: {time.perf_counter() - started:.1f}秒")


if __name__ == '__main__':
    main()
