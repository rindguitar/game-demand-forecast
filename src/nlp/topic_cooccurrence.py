"""
ゲーム単位の共起を測るモジュール

「どの部品が同じゲームに同居しているか」を出す。需要スコアを部品ごとに合算すると
どの組み合わせが未充足かが消えるため（→ Issue #42）、その手前の材料を作る。

**出現では測らない。** 「同じゲームに出るか」で測るとほぼ全結合になり情報にならない
（24本・300単位での実測: 時系列に乗る35単位のうち21個が全24本に出た）。
そのゲームらしさ（リフト）で重み付けし、閾値を超えたものだけを同居と数える。
"""

from typing import Dict, List

import pandas as pd


def has_bundled_units(mapping: Dict[int, int]) -> bool:
    """束ねた単位（2トピック以上）が1つでもあるか

    1. 単位の種類を数える → 2. トピックの数と比べる。
    トピックは必ず1つの単位に入るので、単位のほうが少なければどこかで束ねている。
    """
    return len(set(mapping.values())) < len(mapping)


def unit_categories_from_topics(mapping: Dict[int, int],
                                topic_categories: Dict[int, str]) -> Dict[int, str]:
    """束ねないときの「単位 → 分類」を、トピックごとの公式の分類から引く

    トピック単位の分類は、複数トピックの束には当てられないので束ねたときは使えない。
    分類に無いトピックは既定値で埋めない（別のモデル・別の版のCSVを黙って通さないため）。

    Args:
        mapping: 元トピック → 単位（cut_levels の戻り値の1レベル分）
        topic_categories: 元トピック → 分類（categorize_topics.py の出力から作る）

    Returns:
        単位 → 分類

    Raises:
        ValueError: 束ねた単位がある / 分類に無いトピックがある
    """
    # 1. 束ねた単位があれば止める → 2. 分類に無いトピックがあれば止める
    # → 3. 各単位に、その単位に入ったトピックの分類をそのまま当てる
    if has_bundled_units(mapping):
        raise ValueError('束ねた単位（2トピック以上）があるので、トピック単位の分類は当てられません')
    missing = sorted(set(mapping) - set(topic_categories))
    if missing:
        raise ValueError(f'分類に無いトピックが{len(missing)}個あります'
                         f'（既定値で埋めずに止める）: {missing}')
    return {unit: topic_categories[topic] for topic, unit in mapping.items()}


def build_game_unit_matrix(df: pd.DataFrame, unit_column: str = 'unit',
                           game_column: str = 'game_name',
                           outlier_id: int = -1) -> pd.DataFrame:
    """ゲーム × 単位 の件数表を作る（Outlier は除く）"""
    assigned = df[df[unit_column] != outlier_id]
    return pd.crosstab(assigned[game_column], assigned[unit_column])


def compute_lift(matrix: pd.DataFrame) -> pd.DataFrame:
    """そのゲームらしさ（リフト）を出す

    リフト = そのゲームでの出現率 ÷ 全体での出現率。
    1.0 なら平均通り、2.0 なら平均の2倍そのゲームで語られている。
    大きい汎用トピック（friends / price）が全ゲームを埋めるのを防ぐために割る。
    """
    in_game = matrix.div(matrix.sum(axis=1), axis=0)
    overall = matrix.sum(axis=0) / matrix.values.sum()
    return in_game.div(overall, axis=1)


def extract_recipes(matrix: pd.DataFrame, lift: pd.DataFrame,
                    min_lift: float, min_count: int) -> pd.DataFrame:
    """ゲームごとの「レシピ」= そのゲームを特徴づける単位の並びを作る

    件数の下限も置く。リフトは分母が小さいと跳ねるので、
    数件しかない単位が「そのゲームらしい」と出てしまうため。

    Returns:
        game_name / unit / count / share_in_game / lift を持つ縦長のDataFrame
    """
    long = (lift.stack().rename('lift').reset_index()
            .merge(matrix.stack().rename('count').reset_index(),
                   on=[matrix.index.name, matrix.columns.name]))
    long = long.rename(columns={matrix.index.name: 'game_name',
                                matrix.columns.name: 'unit'})
    long['share_in_game'] = long['count'] / long['game_name'].map(matrix.sum(axis=1))
    keep = (long['lift'] >= min_lift) & (long['count'] >= min_count)
    return long[keep].sort_values(['game_name', 'lift'], ascending=[True, False])


def build_cooccurrence(recipes: pd.DataFrame) -> pd.DataFrame:
    """レシピから、単位ペアの共起表を作る

    同じゲームのレシピに一緒に入っていれば1回と数える。
    ペアは順序を持たない（小さいID側を a に置く）。

    Returns:
        unit_a / unit_b / games（同居したゲーム数）/ game_list
    """
    rows: List[dict] = []
    for game, sub in recipes.groupby('game_name'):
        units = sorted(sub['unit'].tolist())
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                rows.append({'unit_a': a, 'unit_b': b, 'game_name': game})
    if not rows:
        return pd.DataFrame(columns=['unit_a', 'unit_b', 'games', 'game_list'])

    pairs = pd.DataFrame(rows)
    grouped = pairs.groupby(['unit_a', 'unit_b'])['game_name']
    return (pd.DataFrame({'games': grouped.size(),
                          'game_list': grouped.apply(lambda s: ' / '.join(sorted(s)))})
            .reset_index().sort_values('games', ascending=False))


def game_similarity(memberships: pd.DataFrame, game_column: str = 'game_name',
                    item_column: str = 'unit') -> pd.DataFrame:
    """ゲーム同士の似方を Jaccard で出す（共有している部品の割合）

    タグとトピックは語彙が違うので、ペアを直接は比べられない。
    「ゲーム同士がどれだけ似ているか」に落とせば、語彙によらず突き合わせられる。

    Returns:
        game_a / game_b / jaccard を持つDataFrame（同じ組は1行）
    """
    sets = memberships.groupby(game_column)[item_column].apply(set)
    games = sorted(sets.index)
    rows = []
    for i, a in enumerate(games):
        for b in games[i + 1:]:
            union = sets[a] | sets[b]
            rows.append({'game_a': a, 'game_b': b,
                         'jaccard': len(sets[a] & sets[b]) / len(union) if union else 0.0})
    return pd.DataFrame(rows, columns=['game_a', 'game_b', 'jaccard'])
