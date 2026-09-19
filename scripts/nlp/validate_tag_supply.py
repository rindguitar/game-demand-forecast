"""
供給側のタグが、ロスター拡大の代わりになるか検証する

Issue #42。24本のレビューでは部品の組み合わせが観測できなかった（最大同居3本）。
ロスターを増やすのが本筋だがマシンの制約で難しいので、
母集団キャッシュの354本のタグでどこまで迫れるかを測る。

## 2つの検証

**検証1: 同じ24本で、タグとトピックが同じ構造を測っているか**
   語彙が違うのでペアは直接比べられない。「ゲーム同士の似方」に落として突き合わせる。
   ここが合わなければタグは代用にならない（違うものを測っていることになる）。
   偶然どれくらい相関するかを見るため、シャッフルした場合とも比べる。

**検証2: 本数を増やすと組み合わせが見えるようになるか**
   3通りを並べて、効いているのが語彙なのかサンプル数なのかを分離する。

       24本 × トピック   いまの結果（比較の基準）
       24本 × タグ       同じ本数・違う語彙  → 語彙の効果
      354本 × タグ       違う本数・同じ語彙  → サンプル数の効果

## 合否の基準

24本のトピックが失敗したのと同じ条件で測る。
**企画書の粗さ（1ゲーム6部品程度）で、2本以上のゲームに同居するペアが出るか。**

使い方:
    docker compose exec dev python scripts/nlp/validate_tag_supply.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.pool_tags import load_pool_tags  # noqa: E402
from src.nlp.topic_cooccurrence import build_cooccurrence, game_similarity  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pool', default='data/timeseries/pool_cache.json')
    parser.add_argument('--games', default='data/timeseries/games.csv')
    parser.add_argument('--recipes', default='data/timeseries/cooccurrence/recipes.csv')
    parser.add_argument('--outdir', default='data/timeseries/cooccurrence')
    parser.add_argument('--recipe-size', type=int, default=6,
                        help='検証2で使う部品数。タグ側（上位6個）に揃えるのが既定')
    parser.add_argument('--agreement-size', type=int, default=0,
                        help='検証1で使う部品数。0ならレシピ全部（相関は標本が要るため）')
    parser.add_argument('--shuffles', type=int, default=200,
                        help='偶然の相関を測る回数')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-informative', type=int, default=30,
                        help='相関を判定するのに必要な「両方が非ゼロ」のペア数')
    return parser.parse_args()


def top_units_per_game(recipes: pd.DataFrame, size: int) -> pd.DataFrame:
    """レシピをゲームごとにリフトの高い順で上位 size 個に切る（タグ側と粒度を揃える）"""
    return (recipes.sort_values('lift', ascending=False)
            .groupby('game_name').head(size)[['game_name', 'unit']])


def null_correlation(a: pd.Series, b: pd.Series, shuffles: int, rng) -> float:
    """片方を並べ替えたときの相関の絶対値の95パーセンタイル（偶然の目安）"""
    values = b.to_numpy()
    return float(np.percentile(
        [abs(spearmanr(a, rng.permutation(values)).statistic) for _ in range(shuffles)], 95))


def summarize(pairs: pd.DataFrame, n_games: int, n_items: int, label: str) -> dict:
    """共起表を1行にまとめる"""
    possible = n_items * (n_items - 1) // 2
    return {
        'label': label, 'ゲーム数': n_games, '語彙数': n_items,
        '観測ペア': len(pairs), '可能ペア': possible,
        '2本以上': int((pairs['games'] >= 2).sum()) if len(pairs) else 0,
        '5本以上': int((pairs['games'] >= 5).sum()) if len(pairs) else 0,
        '最大同居': int(pairs['games'].max()) if len(pairs) else 0,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1. 材料を読む
    roster = pd.read_csv(args.games)['name'].tolist()
    recipes = pd.read_csv(args.recipes)
    topics24 = top_units_per_game(recipes, args.recipe_size)
    tags_all = load_pool_tags(args.pool)
    tags24 = tags_all[tags_all['game_name'].isin(roster)]
    print(f"ロスター {len(roster)}本 / タグを持つ母集団 {tags_all['game_name'].nunique()}本")
    print(f"部品数を上位{args.recipe_size}個に揃えて比べる\n")

    # 2. 検証1: 同じ24本で、タグとトピックが同じ構造を測っているか
    # 検証1は標本が要るので、切らずにレシピ全体を使う（既定）
    agreement_units = (recipes[['game_name', 'unit']] if args.agreement_size <= 0
                       else top_units_per_game(recipes, args.agreement_size))
    sim_topic = game_similarity(agreement_units, item_column='unit')
    sim_tag = game_similarity(tags24, item_column='tag')
    merged = sim_topic.merge(sim_tag, on=['game_a', 'game_b'], suffixes=('_topic', '_tag'))
    rho = spearmanr(merged['jaccard_topic'], merged['jaccard_tag'])
    chance = null_correlation(merged['jaccard_topic'], merged['jaccard_tag'], args.shuffles, rng)

    print(f"{'=' * 76}\n検証1: タグとトピックは同じ構造を測っているか（同じ24本）\n{'=' * 76}")
    # 両方が0のペアは「似ていない」ではなく「重なりが無い」なので、相関の材料にならない。
    # これが少なすぎるときは、不一致ではなく判定不能として報告する
    informative = int(((merged['jaccard_topic'] > 0) & (merged['jaccard_tag'] > 0)).sum())
    zero_topic = int((merged['jaccard_topic'] == 0).sum())
    zero_tag = int((merged['jaccard_tag'] == 0).sum())

    print(f"  ゲームのペア        {len(merged)}組")
    print(f"  重なりゼロ          トピック {zero_topic}組（{zero_topic/len(merged):.0%}） / "
          f"タグ {zero_tag}組（{zero_tag/len(merged):.0%}）")
    print(f"  両方が非ゼロ        {informative}組  ← 相関を測れる実質的な標本")
    print(f"  Spearman 相関       ρ = {rho.statistic:+.3f}  (p = {rho.pvalue:.4f})")
    print(f"  偶然の目安（95%）   |ρ| = {chance:.3f}  ← {args.shuffles}回シャッフルした結果")

    if informative < args.min_informative:
        print(f"  → ⚠️ 判定不能。両方が非ゼロのペアが {informative}組しかない"
              f"（{args.min_informative}組は要る）。")
        print("     ρ が小さいのは『タグとトピックが違うものを測っている』証拠にはならない。"
              "レシピがゲーム間で重ならないため、そもそも比べられていない")
        print("     直し方: build_topic_cooccurrence.py を --min-lift 1.0 で回して"
              "レシピを厚くし、その出力を --recipes に渡す")
    elif abs(rho.statistic) > chance and rho.pvalue < 0.05:
        print("  → ✅ 偶然を超えている。タグとトピックは同じ構造を捉えている")
    else:
        print("  → ❌ 偶然の範囲。タグはトピックと違うものを測っている")

    # 3. 検証2: 本数を増やすと組み合わせが見えるか（語彙とサンプル数を分離）
    rows = [
        summarize(build_cooccurrence(topics24.rename(columns={'unit': 'unit'})),
                  topics24['game_name'].nunique(), topics24['unit'].nunique(),
                  '24本 × トピック'),
        summarize(build_cooccurrence(tags24.rename(columns={'tag': 'unit'})),
                  tags24['game_name'].nunique(), tags24['tag'].nunique(),
                  '24本 × タグ'),
        summarize(build_cooccurrence(tags_all.rename(columns={'tag': 'unit'})),
                  tags_all['game_name'].nunique(), tags_all['tag'].nunique(),
                  '354本 × タグ'),
    ]
    print(f"\n{'=' * 76}\n検証2: 本数を増やすと組み合わせが見えるか\n{'=' * 76}")
    print(f"{'':16s}{'ゲーム':>6}{'語彙':>6}{'観測ペア':>9}{'可能ペア':>9}"
          f"{'2本以上':>8}{'5本以上':>8}{'最大同居':>9}")
    print('-' * 76)
    for r in rows:
        print(f"{r['label']:16s}{r['ゲーム数']:>6}{r['語彙数']:>6}{r['観測ペア']:>9,}"
              f"{r['可能ペア']:>9,}{r['2本以上']:>8,}{r['5本以上']:>8,}{r['最大同居']:>9}")

    # 4. 合否
    base, tag24, tag354 = rows
    print(f"\n{'=' * 76}\n判定\n{'=' * 76}")
    print(f"  基準: 企画書の粗さ（1ゲーム{args.recipe_size}部品）で2本以上に同居するペアが出るか")
    print(f"    24本 × トピック : {base['2本以上']:,}組（最大{base['最大同居']}本）")
    print(f"    354本 × タグ    : {tag354['2本以上']:,}組（最大{tag354['最大同居']}本）")
    verdict2 = tag354['2本以上'] > base['2本以上']
    print(f"  → {'✅ 組み合わせが観測できるようになった' if verdict2 else '❌ 増えなかった'}")
    print(f"\n  語彙の効果（24本で比較）  : {base['2本以上']:,}組 → {tag24['2本以上']:,}組")
    print(f"  サンプル数の効果（タグで）: {tag24['2本以上']:,}組 → {tag354['2本以上']:,}組")

    out = os.path.join(args.outdir, 'tag_supply_validation.csv')
    pd.DataFrame(rows).to_csv(out, index=False)
    merged.to_csv(os.path.join(args.outdir, 'similarity_agreement.csv'), index=False)
    print(f"\n✅ 比較表: {out}")
    print(f"✅ 類似度の突き合わせ: {os.path.join(args.outdir, 'similarity_agreement.csv')}")
    print("\n⚠️ タグは供給（何が出荷されたか）であって需要ではない。"
          "未充足を言うには需要側と突き合わせが要る")


if __name__ == '__main__':
    main()
