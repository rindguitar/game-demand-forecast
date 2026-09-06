"""
トピックを ①ゲーム要素 / ②品質・運営 / ③ビジネス条件 / 中身なし に仕分ける

docs/decisions.md 2026-08-18「トピックを3分類し、仕分けは ルール→手動→分類器 で実装する」
の第一段（ルール）。曖昧なものは AMBIGUOUS として一覧に出すので、そこだけ人が見る。

処理の流れ:
  1. トピック統計（topic_statistics.csv）を読む
  2. 分類語彙（configs/topic_categories.txt）で各トピックを仕分ける
  3. レビュー本体から、パネルごとの週あたり件数とゲーム集中度を測る
  4. 分類 × パネル到達 の内訳を表示し、CSVに書き出す

使い方:
    docker compose exec dev python scripts/nlp/categorize_topics.py
    docker compose exec dev python scripts/nlp/categorize_topics.py --show ambiguous
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.topic_category import (  # noqa: E402
    CATEGORY_LABELS,
    ELEMENT,
    QUALITY,
    BUSINESS,
    CONTENTLESS,
    PROPERNOUN,
    AMBIGUOUS,
    classify_topics,
    format_hits,
    load_category_words,
)

# 表示の並び順（要素を先頭に置く）
CATEGORY_ORDER = [ELEMENT, QUALITY, BUSINESS, CONTENTLESS, PROPERNOUN, AMBIGUOUS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stats', default='data/timeseries/topic_statistics.csv',
                        help='トピック統計CSV（topic_id / keywords / count 列が必要）')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv',
                        help='トピック付与済みレビューCSV。週あたり件数と集中度の測定に使う')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。土台パネルの顔ぶれを tier 列から取る')
    parser.add_argument('--categories', default='configs/topic_categories.txt',
                        help='分類語彙のファイル')
    parser.add_argument('--output', default=None,
                        help='出力CSV（未指定なら統計と同ディレクトリの topic_categories.csv）')
    parser.add_argument('--min-per-week', type=float, default=10.0,
                        help='時系列に乗せる下限（週あたり件数）')
    parser.add_argument('--backbone-tier', default='土台',
                        help='土台パネルとして扱う tier の値')
    parser.add_argument('--exclude-game', action='append', default=None,
                        help='土台パネルから外すゲーム名（複数指定可）')
    parser.add_argument('--show', choices=CATEGORY_ORDER, default=None,
                        help='この分類のトピックを全件表示する（既定は要約のみ）')
    return parser.parse_args()


def measure_panels(reviews_path, games_path, backbone_tier, exclude_games):
    """レビュー本体から、パネルごとの週あたり件数とゲーム集中度を測る"""
    df = pd.read_csv(reviews_path, usecols=['game_name', 'timestamp_created', 'topic_id'])
    df['date'] = pd.to_datetime(df['timestamp_created'], unit='s')
    weeks = (df['date'].max() - df['date'].min()).days / 7
    assigned = df[df['topic_id'] != -1]

    grouped = assigned.groupby('topic_id')['game_name']
    panels = pd.DataFrame({
        'count_all': assigned['topic_id'].value_counts(),
        'games': grouped.nunique(),
        'top1_share': grouped.apply(lambda s: s.value_counts(normalize=True).iloc[0]),
        'top1_game': grouped.apply(lambda s: s.value_counts().index[0]),
    })
    panels['per_week_all'] = panels['count_all'] / weeks

    # 土台パネル（tier が backbone_tier のゲームから、除外指定を引いたもの）
    games = pd.read_csv(games_path)
    backbone = games.loc[games['tier'] == backbone_tier, 'name'].tolist()
    backbone = [g for g in backbone if g not in (exclude_games or [])]
    bb = assigned[assigned['game_name'].isin(backbone)]
    panels['count_backbone'] = bb['topic_id'].value_counts()
    panels['count_backbone'] = panels['count_backbone'].fillna(0).astype(int)
    panels['per_week_backbone'] = panels['count_backbone'] / weeks

    totals = {'all_reviews': len(df), 'assigned': len(assigned),
              'backbone_assigned': len(bb), 'backbone_games': backbone, 'weeks': weeks}
    return panels, totals


def main():
    args = parse_args()
    exclude = args.exclude_game if args.exclude_game is not None else ['Starfield']

    # 1. トピック統計を読む（Outlier行は分類しない）
    stats = pd.read_csv(args.stats)
    stats = stats[stats['topic_id'] != -1].copy()
    print(f"トピック統計: {len(stats)}件（{args.stats}）")

    # 2. 語彙で仕分ける
    category_words = load_category_words(args.categories)
    loaded = {c: len(w) for c, w in category_words.items()}
    print(f"分類語彙: {loaded}（{args.categories}）")
    classified = classify_topics(list(zip(stats['topic_id'], stats['keywords'])), category_words)
    stats['category'] = [c for _, _, c, _ in classified]
    stats['matched'] = [format_hits(h) for _, _, _, h in classified]

    # 3. パネルごとの密度・集中度を測る
    panels, totals = measure_panels(args.reviews, args.games, args.backbone_tier, exclude)
    stats = stats.merge(panels, left_on='topic_id', right_index=True, how='left')
    stats['reaches_all'] = stats['per_week_all'] >= args.min_per_week
    stats['reaches_backbone'] = stats['per_week_backbone'] >= args.min_per_week
    print(f"土台パネル: {len(totals['backbone_games'])}本"
          f"（tier={args.backbone_tier} から {', '.join(exclude)} を除く）")

    # 4. 内訳を出す
    print(f"\n{'=' * 78}\n分類の内訳（全{len(stats)}トピック）\n{'=' * 78}")
    header = f"{'分類':<14}{'個数':>6}{'件数':>12}{'割合':>8}   週{args.min_per_week:g}件以上"
    print(header)
    print('-' * 78)
    for category in CATEGORY_ORDER:
        sub = stats[stats['category'] == category]
        if sub.empty:
            continue
        reach = sub[sub['reaches_all']]
        print(f"{CATEGORY_LABELS[category]:<14}{len(sub):>6}{sub['count'].sum():>12,}"
              f"{sub['count'].sum() / totals['assigned']:>8.1%}   "
              f"{len(reach):>3}個 / {reach['count'].sum():>9,}件")

    reach_all = stats[stats['reaches_all']]
    print('-' * 78)
    print(f"合計{'':<10}{len(stats):>6}{stats['count'].sum():>12,}{1.0:>8.1%}   "
          f"{len(reach_all):>3}個 / {reach_all['count'].sum():>9,}件")
    print(f"\n全レビュー {totals['all_reviews']:,}件に対する到達率: "
          f"{reach_all['count'].sum() / totals['all_reviews']:.1%}"
          f"（Outlier {1 - totals['assigned'] / totals['all_reviews']:.1%} を含む母数）")

    element_reach = reach_all[reach_all['category'] == ELEMENT]
    bb_reach = stats[stats['reaches_backbone']]
    bb_element = bb_reach[bb_reach['category'] == ELEMENT]
    print(f"\n需要スコアの対象（①ゲーム要素 × 週{args.min_per_week:g}件以上）")
    print(f"  全24本パネル : {len(element_reach)}個 / {element_reach['count'].sum():,}件")
    print(f"  土台パネル   : {len(bb_element)}個 / {bb_element['count_backbone'].sum():,}件")

    # 5. 指定があれば中身を全部出す
    if args.show:
        sub = stats[stats['category'] == args.show].sort_values('count', ascending=False)
        print(f"\n{'=' * 78}\n{CATEGORY_LABELS[args.show]}（{len(sub)}件）\n{'=' * 78}")
        for _, r in sub.iterrows():
            mark = '★' if r['reaches_all'] else ' '
            print(f"{mark}t{int(r['topic_id']):<5} {int(r['count']):>7,}件 "
                  f"top1 {r['top1_share']:.0%} {str(r['top1_game'])[:20]:22s} {r['keywords']}")
            if r['matched']:
                print(f"       └ {r['matched']}")

    # 6. 書き出す
    output = args.output or os.path.join(os.path.dirname(args.stats), 'topic_categories.csv')
    columns = ['topic_id', 'category', 'keywords', 'count', 'games', 'top1_share', 'top1_game',
               'per_week_all', 'per_week_backbone', 'reaches_all', 'reaches_backbone', 'matched']
    stats.sort_values('count', ascending=False)[columns].to_csv(output, index=False)
    print(f"\n✅ 出力: {output}")

    ambiguous = stats[stats['category'] == AMBIGUOUS]
    if not ambiguous.empty:
        print(f"⚠️ 手動判定が必要: {len(ambiguous)}件"
              f"（`--show ambiguous` で一覧、うち週{args.min_per_week:g}件以上は"
              f"{int(ambiguous['reaches_all'].sum())}件）")


if __name__ == '__main__':
    main()
