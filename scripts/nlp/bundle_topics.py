"""
小さいトピックをSteamタグの語彙に束ねる

docs/decisions.md 2026-08-31「一律圧縮は誤り・手を入れるのは小さいトピックのみ」
「候補4（Steamタグ）を語彙として実行。寄らないものは『その他』に集約」の実行部分。

処理の流れ:
  1. 仕分け済みのトピック（topic_categories.csv）を読む
  2. 束ねる対象を絞る（週10件以上は単独で残す。固有名詞・中身なし・要手動判定は外す）
  3. 残りをゲーム台帳のタグ語彙に寄せる。寄らないものは分類ごとの「その他」へ
  4. 束ねた単位で週あたり件数と参加ゲーム数を測り、時系列に乗る単位を数える

使い方:
    docker compose exec dev python scripts/nlp/bundle_topics.py
    docker compose exec dev python scripts/nlp/bundle_topics.py --show
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.topic_bundle import OTHER, bundle_topics, load_tag_vocabulary  # noqa: E402
from src.nlp.topic_category import CATEGORY_LABELS, ELEMENT, QUALITY, BUSINESS  # noqa: E402

# 束ねる対象にする分類（中身なし・固有名詞・要手動判定は束ねない）
BUNDLED_CATEGORIES = [ELEMENT, QUALITY, BUSINESS]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--categories-csv', default='data/timeseries/topic_categories.csv',
                        help='categorize_topics.py が出したトピックの仕分け結果')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv',
                        help='トピック付与済みレビューCSV。参加ゲーム数の測定に使う')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。ジャンル列とタグ列から束ね先の語彙を作る')
    parser.add_argument('--output', default=None,
                        help='出力CSV（未指定なら仕分け結果と同ディレクトリの topic_bundles.csv）')
    parser.add_argument('--min-per-week', type=float, default=10.0,
                        help='時系列に乗せる下限（週あたり件数）')
    parser.add_argument('--show', action='store_true',
                        help='束ねた単位を全件表示する')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 仕分け済みのトピックと台帳を読む
    topics = pd.read_csv(args.categories_csv)
    games = pd.read_csv(args.games)
    vocabulary = load_tag_vocabulary(games['genres'], games['tags'])
    print(f"トピック: {len(topics)}件 / 束ね先の語彙: {len(vocabulary)}種（{args.games}）")

    # 2. 束ねる対象を絞る
    single = topics[topics['reaches_all']].copy()          # 大きいので単独で残す
    small = topics[~topics['reaches_all']].copy()
    target = small[small['category'].isin(BUNDLED_CATEGORIES)].copy()
    dropped = small[~small['category'].isin(BUNDLED_CATEGORIES)]
    print(f"  単独で残す（週{args.min_per_week:g}件以上）: {len(single)}個 / "
          f"{single['count'].sum():,}件")
    print(f"  束ねる対象                        : {len(target)}個 / {target['count'].sum():,}件")
    print(f"  束ねない（中身なし・固有名詞ほか）: {len(dropped)}個 / {dropped['count'].sum():,}件")

    # 3. タグ語彙に寄せる
    assignment = bundle_topics(list(zip(target['topic_id'], target['keywords'])), vocabulary)
    target['bundle'] = target['topic_id'].map(assignment)
    single['bundle'] = single['topic_id'].map(lambda i: f'（単独）t{i}')
    hit = target[target['bundle'] != OTHER]
    print(f"  → タグに寄った: {len(hit)}個 / {hit['count'].sum():,}件"
          f"（{len(hit) / len(target):.0%}）、"
          f"その他: {len(target) - len(hit)}個 / {target['count'].sum() - hit['count'].sum():,}件")

    # 4. 束ねた単位で密度と参加ゲーム数を測る
    reviews = pd.read_csv(args.reviews, usecols=['game_name', 'timestamp_created', 'topic_id'])
    reviews['date'] = pd.to_datetime(reviews['timestamp_created'], unit='s')
    weeks = (reviews['date'].max() - reviews['date'].min()).days / 7

    key = dict(zip(target['topic_id'], target['category'] + ' / ' + target['bundle']))
    reviews['bundle'] = reviews['topic_id'].map(key)
    grouped = reviews.dropna(subset=['bundle']).groupby('bundle')
    bundles = pd.DataFrame({
        'count': grouped.size(),
        'games': grouped['game_name'].nunique(),
        'topics': target.groupby(target['category'] + ' / ' + target['bundle']).size(),
        'top1_share': grouped['game_name'].apply(lambda s: s.value_counts(normalize=True).iloc[0]),
    })
    bundles['per_week'] = bundles['count'] / weeks
    bundles['reaches'] = bundles['per_week'] >= args.min_per_week
    bundles = bundles.sort_values('count', ascending=False)

    reached = bundles[bundles['reaches']]
    print(f"\n{'=' * 78}\n束ねた結果\n{'=' * 78}")
    print(f"束ねた単位: {len(bundles)}個（うち「その他」は分類ごとに1個ずつ）")
    print(f"週{args.min_per_week:g}件以上になった単位: {len(reached)}個 / "
          f"{reached['count'].sum():,}件")
    gain = reached[~reached.index.str.contains(OTHER)]
    print(f"  うち「その他」を除く: {len(gain)}個 / {gain['count'].sum():,}件")

    before = single['count'].sum()
    after = before + reached['count'].sum()
    print(f"\n時系列に乗る言及量: {before:,}件 → {after:,}件"
          f"（全{len(reviews):,}件の {before / len(reviews):.1%} → {after / len(reviews):.1%}）")
    print(f"時系列に乗る単位数 : {len(single)}個 → {len(single) + len(reached)}個")

    if args.show:
        print(f"\n{'=' * 78}\n束ねた単位（週{args.min_per_week:g}件以上に★）\n{'=' * 78}")
        for name, r in bundles.iterrows():
            mark = '★' if r['reaches'] else ' '
            print(f"{mark} {name[:44]:46s} {int(r['topics']):>3}トピック "
                  f"{int(r['count']):>7,}件 週{r['per_week']:5.1f} "
                  f"{int(r['games']):>2}本 top1 {r['top1_share']:.0%}")

    # 5. 書き出す（トピック単位の割り当てと、束ねた単位の要約の2枚）
    output = args.output or os.path.join(os.path.dirname(args.categories_csv),
                                         'topic_bundles.csv')
    columns = ['topic_id', 'category', 'bundle', 'keywords', 'count', 'per_week_all', 'top1_game']
    pd.concat([target[columns], single[columns]]).sort_values(
        ['category', 'bundle', 'count'], ascending=[True, True, False]).to_csv(output, index=False)
    summary = output.replace('.csv', '_summary.csv')
    bundles.to_csv(summary)
    print(f"\n✅ 出力: {output}\n✅ 出力: {summary}")


if __name__ == '__main__':
    main()
