"""
仕分け済みのトピックから、週次の時系列データを作る（Phase 6）

パネルは2つ作る（docs/decisions.md 2026-09-05）:
  土台13本 … 参加ゲームが入れ替わらないので、絶対数で引ける
  全24本   … 参加ゲームが入れ替わるので、シェアで見る

処理の流れ:
  1. 仕分け結果から、パネルごとに乗せる単位（トピック）を選ぶ
  2. レビューに週の列を足し、端の部分週を落とす
  3. 単位 × 週で 件数・シェア・ポジ率・参加ゲーム数 を集計する
  4. 系列の健全性（0件週・欠測・振れ幅）を点検して表示する

使い方:
    docker compose exec dev python scripts/timeseries/build_weekly_series.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.topic_category import BUSINESS, ELEMENT, QUALITY  # noqa: E402
from src.timeseries.weekly import (  # noqa: E402
    add_week_column,
    build_weekly_series,
    trim_partial_weeks,
)

# 時系列に乗せる分類（中身なし・固有名詞・要手動判定は乗せない）
SERIES_CATEGORIES = [ELEMENT, QUALITY, BUSINESS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--categories-csv', default='data/timeseries/topic_categories.csv',
                        help='categorize_topics.py が出したトピックの仕分け結果')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv',
                        help='トピック付与済みレビューCSV')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。土台パネルの顔ぶれを tier 列から取る')
    parser.add_argument('--output-dir', default='data/timeseries',
                        help='出力先ディレクトリ')
    parser.add_argument('--backbone-tier', default='土台',
                        help='土台パネルとして扱う tier の値')
    parser.add_argument('--exclude-game', action='append', default=None,
                        help='土台パネルから外すゲーム名（複数指定可）')
    parser.add_argument('--min-reviews-for-rate', type=int, default=5,
                        help='ポジ率を出す最小件数。これ未満の週は欠測にする')
    return parser.parse_args()


def report(name, series, categories):
    """系列の健全性を点検して表示する"""
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    counts = series.pivot(index='week', columns='unit', values='count')
    print(f"  系列 {counts.shape[1]}個 × {counts.shape[0]}週")

    observed = counts.notna()
    zeros = (counts == 0).sum()
    missing = (~observed).sum()
    print(f"  観測前の欠測を持つ系列: {(missing > 0).sum()}個"
          f"（最大 {missing.max()}週 = 全体の {missing.max() / len(counts):.0%}）")
    print(f"  0件の週を持つ系列      : {(zeros > 0).sum()}個（最大 {zeros.max()}週）")

    medians = counts.median()
    print(f"  週あたり件数の中央値   : 全系列で {medians.median():.1f}"
          f"（最小の系列 {medians.min():.1f} / 最大の系列 {medians.max():.1f}）")

    rate = series.pivot(index='week', columns='unit', values='positive_rate')
    print(f"  ポジ率が出せない週     : {rate.isna().sum().sum():,} / {rate.size:,} セル"
          f"（{rate.isna().sum().sum() / rate.size:.0%}）")

    thin = medians[medians < 10]
    if len(thin):
        print(f"  ⚠️ 週10件を割る系列 {len(thin)}個:")
        for unit, m in thin.sort_values().items():
            print(f"       t{unit}: 中央値 {m:.1f}件  {categories.loc[unit, 'keywords'][:44]}")


def main():
    args = parse_args()
    exclude = args.exclude_game if args.exclude_game is not None else ['Starfield']

    # 1. 乗せる単位を選ぶ
    categories = pd.read_csv(args.categories_csv).set_index('topic_id')
    on_series = categories[categories['category'].isin(SERIES_CATEGORIES)]
    games = pd.read_csv(args.games)
    backbone = [g for g in games.loc[games['tier'] == args.backbone_tier, 'name']
                if g not in exclude]

    # 2. 週の列を足して端の部分週を落とす
    reviews = pd.read_csv(args.reviews, usecols=['game_name', 'timestamp_created',
                                                 'topic_id', 'voted_up'])
    reviews = add_week_column(reviews)
    before = reviews['week'].nunique()
    reviews = trim_partial_weeks(reviews)
    dropped = before - reviews['week'].nunique()
    print(f"レビュー {len(reviews):,}件 / {reviews['week'].nunique()}週"
          f"（端の部分週を{dropped}週ぶん落とした）")
    print(f"土台パネル: {len(backbone)}本（tier={args.backbone_tier} から"
          f" {', '.join(exclude)} を除く）")

    assigned = reviews[reviews['topic_id'] != -1]

    # 3. パネルごとに集計する
    panels = {
        'all24': (assigned, on_series[on_series['reaches_all']]),
        'backbone13': (assigned[assigned['game_name'].isin(backbone)],
                       on_series[on_series['reaches_backbone']]),
    }
    for name, (data, units) in panels.items():
        subset = data[data['topic_id'].isin(units.index)]
        series = build_weekly_series(
            subset,
            denominator=data.groupby('week').size(),
            min_reviews_for_rate=args.min_reviews_for_rate,
        )
        series['category'] = series['unit'].map(categories['category'])
        series['keywords'] = series['unit'].map(categories['keywords'])

        label = '全24本パネル（シェア主軸）' if name == 'all24' else '土台13本パネル（絶対数）'
        report(f"{label} — {len(units)}単位", series, categories)

        path = os.path.join(args.output_dir, f'weekly_series_{name}.csv')
        series.to_csv(path, index=False)
        print(f"  ✅ 出力: {path}")


if __name__ == '__main__':
    main()
