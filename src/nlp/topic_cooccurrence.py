"""
ゲーム単位の共起を測るモジュール

「どの部品が同じゲームに同居しているか」を出す。需要スコアを部品ごとに合算すると
どの組み合わせが未充足かが消えるため（→ Issue #42）、その手前の材料を作る。

**出現では測らない。** 時系列に乗る35単位のうち21個が全24本に出るので、
「同じゲームに出るか」で測るとほぼ全結合になり情報にならない。
そのゲームらしさ（リフト）で重み付けし、閾値を超えたものだけを同居と数える。
"""

from typing import List

import pandas as pd


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
