"""
週次時系列を折れ線グラフにする

build_weekly_series.py が出したCSVを読んで、パネルごとに図を描く。
土台13本は絶対数（言及数）、全24本はシェアが主軸になる
（docs/decisions.md 2026-09-05 / 2026-08-31）。

使い方:
    docker compose exec dev python scripts/timeseries/plot_weekly_series.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.visualization.timeseries_plots import (  # noqa: E402
    plot_overview,
    plot_positive_rate_grid,
    plot_series_grid,
)

# パネルごとの主軸（土台13本は絶対数で引ける／全24本は参加ゲームが入れ替わるのでシェア）
PANELS = {
    'backbone13': ('count', 'Backbone 13 games - weekly mentions'),
    'all24': ('share', 'All 24 games - share of weekly mentions'),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default='data/timeseries',
                        help='weekly_series_*.csv があるディレクトリ')
    parser.add_argument('--output-dir', default='data/timeseries/plots',
                        help='図の出力先')
    parser.add_argument('--rolling', type=int, default=4,
                        help='移動平均の窓（週）。0で移動平均なし')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for panel, (value_column, title) in PANELS.items():
        path = os.path.join(args.input_dir, f'weekly_series_{panel}.csv')
        series = pd.read_csv(path, parse_dates=['week'])
        units = series['unit'].nunique()
        print(f"\n{panel}: {units}単位 × {series['week'].nunique()}週（{path}）")

        # 1. 単位ごとの小さい図を並べる（1枚に重ねると読めないため）
        out = plot_series_grid(series, value_column,
                               f"{title}  (thin = weekly, thick = {args.rolling}w average)",
                               os.path.join(args.output_dir, f'{panel}_{value_column}.png'),
                               rolling=args.rolling)
        print(f"  ✅ {out}")

        # 2. 上位だけ重ねて関係を見る
        out = plot_overview(series, value_column, f"{title}  (top 8, {args.rolling}w average)",
                            os.path.join(args.output_dir, f'{panel}_{value_column}_overview.png'),
                            rolling=args.rolling)
        print(f"  ✅ {out}")

        # 3. 充足度（ポジ率）
        out = plot_positive_rate_grid(
            series, f"{title.split(' - ')[0]} - satisfaction (positive rate)",
            os.path.join(args.output_dir, f'{panel}_positive_rate.png'), rolling=args.rolling)
        print(f"  ✅ {out}")


if __name__ == '__main__':
    main()
