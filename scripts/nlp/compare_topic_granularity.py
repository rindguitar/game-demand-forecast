"""
トピックの粒度を粗い側へ動かし、レベルごとに同じ物差しで測って比べる

Issue #37「トピックの目標粒度を1回だけ決める」の判断材料を作る。
455トピックのマージ木を作り、300 / 200 / 100 ... と切りながら、
どのレベルでも同じ指標を出して並べる。再学習はしない。

物差しの中心は **有効単位数**:
    週あたり件数の中央値が下限以上（時系列に乗る）
  × top1_share が上限未満（1本のゲームの話ではない）
  × 分類が ①要素 / ②品質 / ③ビジネス（中身なし・固有名詞ではない）

粗くすると到達率は上がるが、中身なしのトピックが要素の束に紛れ込む。
その代償を **混入率** として併記する。

使い方:
    docker compose exec dev python scripts/nlp/compare_topic_granularity.py
    docker compose exec dev python scripts/nlp/compare_topic_granularity.py \
        --distance embedding --levels 455,200,100,50,25
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.pool_tags import load_pool  # noqa: E402
from src.nlp.tag_semantics import element_tags, load_encoder  # noqa: E402
from src.nlp.topic_category import (  # noqa: E402
    CONTENTLESS, ELEMENT, MEANINGFUL_CATEGORIES, PROPERNOUN,
    classify_with_evidence, load_category_words,
)
from src.nlp.topic_granularity import (  # noqa: E402
    CTFIDF, EMBEDDING, build_linkage, cut_levels, group_dispersion,
    merged_keywords, topic_matrix,
)
from src.timeseries.weekly import (  # noqa: E402
    MIN_WEEKS_SINCE_RELEASE,
    add_week_column,
    decide_window_and_backbone,
    describe_window_and_backbone,
    measure_topic_panels,
    trim_to_window,
)

# 束に紛れ込むと需要スコアを水増しする分類
NOISE = (CONTENTLESS, PROPERNOUN)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='models/topic_full')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv')
    parser.add_argument('--stats', default='data/timeseries/topic_statistics.csv')
    parser.add_argument('--games', default='data/timeseries/games.csv',
                        help='ゲーム台帳CSV。土台パネルの顔ぶれを tier 列と発売日から取る')
    parser.add_argument('--collection-log', default='data/timeseries/collection_log.csv',
                        help='収集ログCSV。全ゲームの収集がそろう期間を oldest / newest から出す')
    parser.add_argument('--categories', default='configs/topic_categories.txt')
    parser.add_argument('--pool', default='data/timeseries/pool_cache.json',
                        help='①の語彙にするSteamタグの取得元（必須。証拠なしで分類しないため）')
    parser.add_argument('--outdir', default='data/timeseries/granularity')
    parser.add_argument('--levels', default='455,300,200,150,100,70,50,35,25,15',
                        help='切り出すトピック数をカンマ区切りで（粗い側のみ）')
    parser.add_argument('--distance', choices=[CTFIDF, EMBEDDING], default=CTFIDF,
                        help='ctfidf=語の重なりで束ねる / embedding=意味の近さで束ねる')
    parser.add_argument('--min-per-week', type=float, default=10.0,
                        help='時系列に乗せる下限（週あたり件数の中央値）')
    parser.add_argument('--cross-max', type=float, default=0.5,
                        help='横断とみなす上限（top1_share がこれ未満なら横断）')
    parser.add_argument('--backbone-tier', default='土台',
                        help='土台パネルとして扱う tier の値')
    parser.add_argument('--min-weeks-since-release', type=int, default=MIN_WEEKS_SINCE_RELEASE,
                        help='土台に入れる条件。発売が期間開始のこの週数以上前であること（既定 %(default)s）')
    parser.add_argument('--exclude-game', action='append', default=None,
                        help='土台パネルから手で外すゲーム名（複数指定可。既定は外さない）')
    return parser.parse_args()


def original_categories(stats_path, category_words, tags, encoder):
    """元の455トピックの分類。混入率を測るときの基準になる"""
    stats = pd.read_csv(stats_path)
    stats = stats[stats['topic_id'] != -1]
    classified = classify_with_evidence(list(zip(stats['topic_id'], stats['keywords'])),
                                        category_words, tags, encoder=encoder)
    return {int(c.topic_id): c.category for c in classified}


def measure_level(df, backbone, window, mapping, model, weights, category_words,
                  origin_category, tags, encoder, args):
    """1つの粒度レベルを測って、単位ごとの表と要約の dict を返す

    処理の流れ:
      1. 各レビューの topic_id を束IDに置き換える（Outlier は Outlier のまま）
      2. 束ごとの密度・集中度をパネル共通の物差しで測る（df は期間に絞ってある。window は物差しの必須の引数）
      3. 束のキーワードを作って分類する（tags・encoder は呼び出し側で1回だけ読んだものを使い回す）
      4. 到達 / 横断 / 意味ありの3条件で有効単位を数え、混入率を出す
    """
    work = df.copy()
    work['unit'] = work['topic_id'].map(mapping).fillna(-1).astype(int)

    panels, totals = measure_topic_panels(work, backbone, window, unit_column='unit')
    keywords = merged_keywords(model, mapping, weights)

    units = panels.reset_index().rename(columns={'index': 'unit', 'unit': 'unit'})
    units['keywords'] = units['unit'].map(keywords)
    classified = classify_with_evidence(list(zip(units['unit'], units['keywords'])),
                                        category_words, tags, encoder=encoder)
    units['category'] = [c.category for c in classified]
    units['members'] = units['unit'].map(
        pd.Series(list(mapping.values())).value_counts().to_dict())
    units['dispersion'] = units['unit'].map(group_dispersion(model, mapping))

    # 束の中で、元が中身なし・固有名詞だったレビューが何件混ざったか
    work['_origin'] = work['topic_id'].map(origin_category)
    noise = (work[work['_origin'].isin(NOISE)].groupby('unit').size())
    units['noise_count'] = units['unit'].map(noise).fillna(0).astype(int)

    units['reaches_all'] = units['per_week_all'] >= args.min_per_week
    units['reaches_backbone'] = units['per_week_backbone'] >= args.min_per_week
    units['is_cross'] = units['top1_share'] < args.cross_max
    units['is_meaningful'] = units['category'].isin(MEANINGFUL_CATEGORIES)
    units['is_effective'] = units['reaches_all'] & units['is_cross'] & units['is_meaningful']

    reach = units[units['reaches_all']]
    effective = units[units['is_effective']]
    element = effective[effective['category'] == ELEMENT]
    # 混入率は「有効と数えた束の中に、中身なし由来がどれだけ混ざっているか」
    contamination = (effective['noise_count'].sum() / effective['count_all'].sum()
                     if len(effective) else 0.0)

    summary = {
        'units': len(units),
        'reach_all': len(reach),
        'reach_backbone': int(units['reaches_backbone'].sum()),
        'effective': len(effective),
        'effective_element': len(element),
        'coverage': reach['count_all'].sum() / totals['all_reviews'],
        'effective_coverage': effective['count_all'].sum() / totals['all_reviews'],
        'contamination': contamination,
        'median_members': float(effective['members'].median()) if len(effective) else 0.0,
        'dispersion': (float((effective['dispersion'] * effective['count_all']).sum()
                             / effective['count_all'].sum()) if len(effective) else 0.0),
    }
    return units, summary


def main():
    args = parse_args()
    levels = [int(x) for x in args.levels.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 材料を読む。共通の期間と土台のゲームは、収集ログと台帳から機械的に決める。
    #    レビューは先に期間へ絞る（密度と混入率を同じレビューで測るため）。
    #    束の重み（トピックごとの件数）は期間と関係ないので、全レビューから数える
    from bertopic import BERTopic
    print(f"モデルを読む: {args.model}")
    model = BERTopic.load(args.model)
    window, backbone, left_out = decide_window_and_backbone(
        pd.read_csv(args.games), pd.read_csv(args.collection_log),
        tier=args.backbone_tier, min_weeks=args.min_weeks_since_release,
        exclude=args.exclude_game)
    print(describe_window_and_backbone(window, backbone, left_out,
                                       args.backbone_tier, args.min_weeks_since_release))

    df = pd.read_csv(args.reviews, usecols=['game_name', 'timestamp_created', 'topic_id'])
    weights = df['topic_id'].value_counts().to_dict()
    total = len(df)
    df = trim_to_window(add_week_column(df), window)
    category_words = load_category_words(args.categories)
    tags = element_tags(load_pool(args.pool))
    encoder = load_encoder()
    origin_category = original_categories(args.stats, category_words, tags, encoder)
    print(f"レビュー {len(df):,}件（期間の外の{total - len(df):,}件を除いた）"
          f" / 距離 {args.distance}")

    # 2. マージ木を作り、レベルごとに切る
    matrix, topic_ids = topic_matrix(model, args.distance)
    tree = build_linkage(matrix)
    mappings = cut_levels(tree, topic_ids, levels)
    print(f"マージ木: {len(topic_ids)}トピック → {len(levels)}レベル\n")

    # 3. レベルごとに同じ物差しで測る
    rows = []
    for level in levels:
        units, summary = measure_level(df, backbone, window, mappings[level], model, weights,
                                       category_words, origin_category, tags, encoder, args)
        summary['level'] = level
        rows.append(summary)
        units.sort_values('count_all', ascending=False).to_csv(
            os.path.join(args.outdir, f'units_{args.distance}_{level:03d}.csv'), index=False)
        print(f"  level {level:>4} → 単位{summary['units']:>4} "
              f"到達{summary['reach_all']:>3} 有効{summary['effective']:>3} "
              f"（うち要素{summary['effective_element']:>3}） "
              f"到達率{summary['coverage']:>6.1%} 混入{summary['contamination']:>5.1%}")

    # 4. 比較表を書き出す
    table = pd.DataFrame(rows)[[
        'level', 'units', 'reach_all', 'reach_backbone', 'effective', 'effective_element',
        'coverage', 'effective_coverage', 'contamination', 'dispersion', 'median_members']]
    out = os.path.join(args.outdir, f'comparison_{args.distance}.csv')
    table.to_csv(out, index=False)

    print(f"\n{'=' * 88}\n粒度レベルの比較（距離: {args.distance}"
          f" / 週{args.min_per_week:g}件以上 / top1 {args.cross_max:.0%}未満）\n{'=' * 88}")
    print(f"{'レベル':>6}{'単位数':>7}{'到達':>6}{'有効':>6}{'内要素':>7}"
          f"{'到達率':>9}{'有効到達率':>11}{'混入率':>8}{'束内距離':>10}{'束の中央値':>11}")
    print('-' * 88)
    for r in rows:
        print(f"{r['level']:>6}{r['units']:>7}{r['reach_all']:>6}{r['effective']:>6}"
              f"{r['effective_element']:>7}{r['coverage']:>9.1%}"
              f"{r['effective_coverage']:>11.1%}{r['contamination']:>8.1%}"
              f"{r['dispersion']:>10.3f}{r['median_members']:>11.0f}")

    best = max(rows, key=lambda r: r['effective'])
    print('-' * 88)
    print(f"有効単位数が最大: レベル {best['level']}（{best['effective']}単位・"
          f"有効到達率 {best['effective_coverage']:.1%}・混入率 {best['contamination']:.1%}"
          f"・束内距離 {best['dispersion']:.3f}）")
    print(f"\n✅ 比較表: {out}")
    print(f"✅ 各レベルの単位一覧: {args.outdir}/units_{args.distance}_*.csv")


if __name__ == '__main__':
    main()
