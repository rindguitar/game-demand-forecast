"""
よく続いた発売と続かなかった発売で、持っている要素が違うかをのぞく（Issue #63 段階3）

要素ごと・2つの組み合わせごとに、持つ発売と持たない発売の相対値（相対値 = 区分の組の中での
続き方。1より大きければふつうより続いた）の中央値を比べる。発売と相対値の対応を入れ替えて（シャッフル）、
偶然でも出る差かを確かめる。要素は、レビューの話題（recipes.csv の①ゲーム要素）を主に、
Steam のタグ（上位6個のうち①にするもの）で補う。発売が少ないうちの「当たり付け」で、結論には使わない。

処理の流れ:
  1. launch_persistence.csv から相対値のある発売を取り出す
  2. 発売ごとの要素（話題・タグ）の真偽表を作り、ペアの表も作る
  3. 要素・ペア・タグそれぞれで中央値の差を出し、シャッフルで p_shuffle と max_diff_95 を出す
  4. 3つの CSV を書き、画面に上位・下位を出す

使い方:
    make persistence-elements
    make persistence-elements ELEMENTS_ARGS="--n-shuffle 5000 --seed 1"
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.nlp.tag_semantics import DEFAULT_TAG_JUDGMENTS, load_tag_judgments  # noqa: E402
from src.timeseries.persistence_elements import (  # noqa: E402
    MIN_COUNT,
    N_SHUFFLE,
    SEED,
    TOP_TAGS,
    compare_with_shuffle,
    game_tag_sets,
    membership_matrix,
    pair_membership,
    recipe_element_sets,
    unit_keywords,
    usable_releases,
)

DEFAULT_PERSISTENCE = 'data/timeseries/persistence_64/launch_persistence.csv'
DEFAULT_RECIPES = 'data/timeseries/cooccurrence_64/recipes.csv'
DEFAULT_GAMES = 'data/timeseries/games.csv'
DEFAULT_OUTPUT_DIR = 'data/timeseries/persistence_64/elements'
SHOW_COLS = ['keywords', 'n_have', 'n_lack', 'median_have', 'median_lack', 'diff',
             'p_shuffle', 'beyond_line']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--persistence', default=DEFAULT_PERSISTENCE)
    parser.add_argument('--recipes', default=DEFAULT_RECIPES)
    parser.add_argument('--games', default=DEFAULT_GAMES)
    parser.add_argument('--tag-judgments', default=DEFAULT_TAG_JUDGMENTS)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--min-count', type=int, default=MIN_COUNT, help='要素を見る最小の発売数')
    parser.add_argument('--n-shuffle', type=int, default=N_SHUFFLE, help='シャッフル回数')
    parser.add_argument('--seed', type=int, default=SEED, help='乱数の種')
    parser.add_argument('--top-tags', type=int, default=TOP_TAGS, help='発売ごとに見る上位タグの数')
    return parser.parse_args()


def show(title, table, head=0, tail=0):
    """表の上位 head 行と下位 tail 行を画面に出す"""
    print(f'\n--- {title} ---')
    if table.empty:
        print('(該当なし)')
        return
    parts = [table.head(head)] if head else []
    if tail:
        parts.append(table.tail(tail))
    shown = pd.concat(parts).drop_duplicates(subset=['item'])
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        print(shown[SHOW_COLS].round(3).to_string())


def main():
    args = parse_args()

    # 1. 相対値のある発売
    values = usable_releases(pd.read_csv(args.persistence))
    print(f'対象の発売: {len(values)} 本（相対値が空の発売は除外）')

    # 2. 要素の真偽表（話題・ペア・タグ）
    recipes = pd.read_csv(args.recipes)
    games = pd.read_csv(args.games)
    topic_m = membership_matrix(recipe_element_sets(recipes), values.index)
    pair_m = pair_membership(topic_m, args.min_count)
    judgments = load_tag_judgments(args.tag_judgments)
    tag_sets = game_tag_sets(games, judgments, args.top_tags, only=values.index)
    tag_m = membership_matrix(tag_sets, values.index)
    kw = unit_keywords(recipes)

    # 3. 差とシャッフル
    kwargs = dict(min_count=args.min_count, n_shuffle=args.n_shuffle, seed=args.seed)
    topics, topic_line = compare_with_shuffle(topic_m, values, labels=kw, **kwargs)
    pair_labels = {p: f'{kw.get(p[0], "")} / {kw.get(p[1], "")}' for p in pair_m.columns}
    pairs, pair_line = compare_with_shuffle(pair_m, values, labels=pair_labels, **kwargs)
    tags, tag_line = compare_with_shuffle(tag_m, values, labels={t: t for t in tag_m.columns},
                                          **kwargs)
    for t in (pairs,):
        t['item'] = t['item'].map(lambda p: f'{p[0]}+{p[1]}')

    # 4. 書き出しと画面
    os.makedirs(args.output_dir, exist_ok=True)
    topics.to_csv(os.path.join(args.output_dir, 'element_persistence.csv'), index=False)
    pairs.to_csv(os.path.join(args.output_dir, 'pair_persistence.csv'), index=False)
    tags.to_csv(os.path.join(args.output_dir, 'tag_persistence.csv'), index=False)

    print(f'見た数: 話題の要素 {len(topics)} / ペア {len(pairs)} / タグ {len(tags)}')
    print(f'max_diff_95（たくさん試すと偶然出る差の線）: 話題 {topic_line:.3f} / '
          f'ペア {pair_line:.3f} / タグ {tag_line:.3f}')
    show('話題の要素 上位5・下位5', topics, 5, 5)
    show('話題の要素のペア 上位3', pairs, 3)
    show('タグ 上位3・下位3', tags, 3, 3)
    print(f'\n出力: {args.output_dir}')


if __name__ == '__main__':
    main()
