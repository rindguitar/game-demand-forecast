"""
ゲーム単位の共起を出す（どの部品が同じゲームに同居しているか）

Issue #42「需要スコアを部品の合算から組み合わせへ広げるか」の材料。
部品ごとに合算すると「どの組み合わせが未充足か」が消えるので、その手前を見る。

**出現ではなくリフトで測る。** 素朴に「同じゲームに出るか」で数えると、
時系列に乗る35単位のうち21個が全24本に出るためほぼ全結合になる。

粒度は Issue #37 の結論に従い、既定でマージ木のレベル300を使う（→ docs/decisions.md 2026-09-09）。

⚠️ 24本では「このペアが無い = 未開拓」は言えない。300部品ならペアは44,850通りあり、
24本では原理的に埋まらない。読めるのは①レシピと②観測された共起まで。

使い方:
    docker compose exec dev python scripts/nlp/build_topic_cooccurrence.py
    docker compose exec dev python scripts/nlp/build_topic_cooccurrence.py --min-lift 3.0
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.topic_category import (  # noqa: E402
    CONTENTLESS, PROPERNOUN, classify_topic, load_category_words,
)
from src.nlp.topic_cooccurrence import (  # noqa: E402
    build_cooccurrence, build_game_unit_matrix, compute_lift, extract_recipes,
)
from src.nlp.topic_granularity import (  # noqa: E402
    CTFIDF, EMBEDDING, build_linkage, cut_levels, merged_keywords, topic_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='models/topic_full')
    parser.add_argument('--reviews', default='data/timeseries/reviews_timeseries_with_topics.csv')
    parser.add_argument('--categories', default='configs/topic_categories.txt')
    parser.add_argument('--outdir', default='data/timeseries/cooccurrence')
    parser.add_argument('--level', type=int, default=300,
                        help='マージ木を切る個数（既定300は Issue #37 の結論）')
    parser.add_argument('--distance', choices=[CTFIDF, EMBEDDING], default=EMBEDDING)
    parser.add_argument('--min-lift', type=float, default=2.0,
                        help='そのゲームらしいと数える下限（平均の何倍か）')
    parser.add_argument('--min-count', type=int, default=50,
                        help='レシピに入れる最小件数。リフトは分母が小さいと跳ねるため')
    parser.add_argument('--keep-noise', action='store_true',
                        help='中身なし・固有名詞の単位もレシピに残す（既定は外す）')
    parser.add_argument('--top', type=int, default=8, help='画面に出すレシピの行数')
    parser.add_argument('--top-pairs', type=int, default=15, help='画面に出す共起ペアの行数')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1. モデルを読み、マージ木を指定のレベルで切る
    from bertopic import BERTopic
    print(f"モデルを読む: {args.model}")
    model = BERTopic.load(args.model)
    df = pd.read_csv(args.reviews, usecols=['game_name', 'topic_id'])
    weights = df['topic_id'].value_counts().to_dict()

    matrix_source, topic_ids = topic_matrix(model, args.distance)
    mapping = cut_levels(build_linkage(matrix_source), topic_ids, [args.level])[args.level]
    keywords = merged_keywords(model, mapping, weights)
    df['unit'] = df['topic_id'].map(mapping).fillna(-1).astype(int)
    print(f"レビュー {len(df):,}件 / ゲーム {df['game_name'].nunique()}本 "
          f"/ 単位 {len(set(mapping.values()))}個（レベル{args.level}・{args.distance}）")

    # 2. 単位を仕分ける（中身なし・固有名詞はレシピから外す）
    category_words = load_category_words(args.categories)
    categories = {u: classify_topic(k, category_words)[0] for u, k in keywords.items()}
    if not args.keep_noise:
        noise = {u for u, c in categories.items() if c in (CONTENTLESS, PROPERNOUN)}
        df = df[~df['unit'].isin(noise)]
        print(f"レシピから外した単位: {len(noise)}個（中身なし・固有名詞）")

    # 3. ゲーム × 単位 の表からリフトを出し、レシピを作る
    counts = build_game_unit_matrix(df)
    lift = compute_lift(counts)
    recipes = extract_recipes(counts, lift, args.min_lift, args.min_count)
    recipes['keywords'] = recipes['unit'].map(keywords)
    recipes['category'] = recipes['unit'].map(categories)

    # 4. レシピから共起表を作る
    pairs = build_cooccurrence(recipes)
    if not pairs.empty:
        pairs['keywords_a'] = pairs['unit_a'].map(keywords)
        pairs['keywords_b'] = pairs['unit_b'].map(keywords)

    # 5. 書き出す
    recipes.to_csv(os.path.join(args.outdir, 'recipes.csv'), index=False)
    pairs.to_csv(os.path.join(args.outdir, 'pairs.csv'), index=False)

    # 6. 画面に出す
    sizes = recipes.groupby('game_name').size()
    print(f"\n{'=' * 84}\nゲームごとのレシピ（リフト{args.min_lift:g}倍以上・"
          f"{args.min_count}件以上）\n{'=' * 84}")
    for game, sub in recipes.groupby('game_name'):
        print(f"\n{game}（{len(sub)}部品）")
        for _, r in sub.head(args.top).iterrows():
            print(f"   ×{r['lift']:>5.1f} {int(r['count']):>6,}件 "
                  f"[{r['category']:<8}] {r['keywords']}")

    print(f"\n{'=' * 84}\n部品の数（1ゲームあたり）: 中央値 {sizes.median():.0f} / "
          f"最小 {sizes.min()} / 最大 {sizes.max()}\n{'=' * 84}")
    print(f"\n共起したペア: {len(pairs):,}組（44,850通り中）"
          if not pairs.empty else "\n共起なし")
    if not pairs.empty:
        print(f"\n2本以上のゲームで同居したペア（上位{args.top_pairs}）")
        for _, r in pairs[pairs['games'] >= 2].head(args.top_pairs).iterrows():
            print(f"  {int(r['games'])}本  {str(r['keywords_a'])[:34]:36s} × "
                  f"{str(r['keywords_b'])[:34]}")

    print(f"\n✅ レシピ: {os.path.join(args.outdir, 'recipes.csv')}")
    print(f"✅ 共起表: {os.path.join(args.outdir, 'pairs.csv')}")
    print("\n⚠️ 24本では「このペアが無い = 未開拓」は言えない（ペアは44,850通り）")


if __name__ == '__main__':
    main()
