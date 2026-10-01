"""
仕分け済みのトピックから、週次の時系列データを作る（Phase 6）

パネルは2つ作る（docs/decisions.md 2026-09-05）:
  土台パネル     … 参加ゲームが入れ替わらないので、絶対数で引ける
  全ゲームパネル … 参加ゲームが入れ替わるので、シェアで見る

期間は、全ゲームの収集がそろう範囲を収集ログから出す。土台は、tier が土台で、かつ
発売が期間開始の --min-weeks-since-release 週以上前のゲームに限る（docs/decisions.md
2026-10-01）。どちらも src/timeseries/weekly.py の decide_window_and_backbone を通す
（分類・粒度比較と同じ定義）。
24本のときは「全24本・土台13本」だったが、本数は台帳と収集ログから決まる。

処理の流れ:
  1. 仕分け結果から、パネルごとに乗せる単位（トピック）を選ぶ
  2. 収集ログから共通の期間を出し、土台のゲームを選ぶ
  3. レビューに週の列を足し、共通の期間に絞る
  4. 単位 × 週で 件数・シェア・ポジ率・参加ゲーム数 を集計する
  5. 系列の健全性（0件週・欠測・振れ幅）を点検して表示する

使い方:
    docker compose exec dev python scripts/timeseries/build_weekly_series.py
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.topic_category import MEANINGFUL_CATEGORIES  # noqa: E402
from src.timeseries.weekly import (  # noqa: E402
    MIN_WEEKS_SINCE_RELEASE,
    add_week_column,
    build_weekly_series,
    decide_window_and_backbone,
    describe_window_and_backbone,
    trim_to_window,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--categories-csv', default='data/timeseries/topic_categories.csv',
                        help='categorize_topics.py が出したトピックの仕分け結果')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv',
                        help='トピック付与済みレビューCSV')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。土台パネルの顔ぶれを tier 列と発売日から取る')
    parser.add_argument('--collection-log', default='data/timeseries/collection_log.csv',
                        help='収集ログCSV。全ゲームの収集がそろう期間を oldest / newest から出す')
    parser.add_argument('--output-dir', default='data/timeseries',
                        help='出力先ディレクトリ')
    parser.add_argument('--backbone-tier', default='土台',
                        help='土台パネルとして扱う tier の値')
    parser.add_argument('--min-weeks-since-release', type=int, default=MIN_WEEKS_SINCE_RELEASE,
                        help='土台に入れる条件。発売が期間開始のこの週数以上前であること（既定 %(default)s）')
    parser.add_argument('--exclude-game', action='append', default=None,
                        help='土台パネルから手で外すゲーム名（複数指定可。既定は外さない）')
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
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 乗せる単位を選ぶ（①②③以外は乗せない）
    categories = pd.read_csv(args.categories_csv).set_index('topic_id')
    on_series = categories[categories['category'].isin(MEANINGFUL_CATEGORIES)]

    # 2. 共通の期間と土台のゲームを、収集ログと台帳から機械的に決める
    window, backbone, left_out = decide_window_and_backbone(
        pd.read_csv(args.games), pd.read_csv(args.collection_log),
        tier=args.backbone_tier, min_weeks=args.min_weeks_since_release,
        exclude=args.exclude_game)
    print(describe_window_and_backbone(window, backbone, left_out,
                                       args.backbone_tier, args.min_weeks_since_release))

    # 3. 週の列を足して、共通の期間に絞る
    reviews = pd.read_csv(args.reviews, usecols=['game_name', 'timestamp_created',
                                                 'topic_id', 'voted_up'])
    reviews = add_week_column(reviews)
    in_window = trim_to_window(reviews, window)
    print(f"レビュー {len(in_window):,}件 / {in_window['week'].nunique()}週"
          f"（期間の外の{len(reviews) - len(in_window):,}件を除いた）")

    assigned = in_window[in_window['topic_id'] != -1]

    # 4. パネルごとに集計する
    panels = {
        'all': (assigned, on_series[on_series['reaches_all']]),
        'backbone': (assigned[assigned['game_name'].isin(backbone)],
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

        n_games = data['game_name'].nunique()
        label = (f'全{n_games}本パネル（シェア主軸）' if name == 'all'
                 else f'土台{n_games}本パネル（絶対数）')
        report(f"{label} — {len(units)}単位", series, categories)

        path = os.path.join(args.output_dir, f'weekly_series_{name}.csv')
        series.to_csv(path, index=False)
        print(f"  ✅ 出力: {path}")


if __name__ == '__main__':
    main()
