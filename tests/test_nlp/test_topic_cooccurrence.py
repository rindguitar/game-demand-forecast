"""
ゲーム単位の共起を測るモジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import pytest
from src.nlp.topic_cooccurrence import (
    build_cooccurrence,
    build_game_unit_matrix,
    compute_lift,
    extract_recipes,
)


def _reviews(rows):
    """(ゲーム名, 単位, 件数) からレビューのDataFrameを作る"""
    return pd.DataFrame([{'game_name': g, 'unit': u}
                         for g, u, n in rows for _ in range(n)])


def test_matrix_excludes_outlier():
    """Outlier（-1）は共起の対象にしない"""
    matrix = build_game_unit_matrix(_reviews([('A', 1, 3), ('A', -1, 5), ('B', 1, 2)]))
    assert -1 not in matrix.columns
    assert matrix.loc['A', 1] == 3


def test_lift_is_one_when_evenly_spread():
    """どのゲームにも同じ割合で出る単位のリフトは1.0"""
    matrix = build_game_unit_matrix(
        _reviews([('A', 1, 10), ('A', 2, 10), ('B', 1, 10), ('B', 2, 10)]))
    lift = compute_lift(matrix)
    assert lift.loc['A', 1] == pytest.approx(1.0)
    assert lift.loc['B', 2] == pytest.approx(1.0)


def test_lift_rises_for_concentrated_unit():
    """1本のゲームに偏っている単位のリフトは1より大きい"""
    matrix = build_game_unit_matrix(
        _reviews([('A', 1, 90), ('A', 2, 10), ('B', 1, 10), ('B', 2, 90)]))
    lift = compute_lift(matrix)
    assert lift.loc['A', 1] > 1.0
    assert lift.loc['B', 1] < 1.0


def test_lift_does_not_reward_size_alone():
    """全ゲームで大きいだけの単位は、そのゲームらしいとは数えない

    friends や price のような汎用トピックが全ゲームのレシピを埋めるのを防ぐ。
    """
    matrix = build_game_unit_matrix(
        _reviews([('A', 1, 900), ('A', 2, 100), ('B', 1, 900), ('B', 2, 100)]))
    lift = compute_lift(matrix)
    assert lift.loc['A', 1] == pytest.approx(1.0)


def test_extract_recipes_applies_both_thresholds():
    """リフトと件数の両方を満たしたものだけがレシピに入る"""
    matrix = build_game_unit_matrix(
        _reviews([('A', 1, 90), ('A', 2, 3), ('B', 1, 10), ('B', 2, 100)]))
    lift = compute_lift(matrix)
    # 単位2はAでのリフトは低い。単位1はAで高いが、件数の下限で落とせる
    assert set(extract_recipes(matrix, lift, min_lift=1.5, min_count=50)['unit']) == {1, 2}
    assert extract_recipes(matrix, lift, min_lift=1.5, min_count=200).empty


def test_cooccurrence_counts_games_not_reviews():
    """共起はゲーム数で数える（同じゲーム内で何件あっても1回）"""
    recipes = pd.DataFrame([
        {'game_name': 'A', 'unit': 1}, {'game_name': 'A', 'unit': 2},
        {'game_name': 'B', 'unit': 1}, {'game_name': 'B', 'unit': 2},
        {'game_name': 'C', 'unit': 1}, {'game_name': 'C', 'unit': 3},
    ])
    pairs = build_cooccurrence(recipes).set_index(['unit_a', 'unit_b'])
    assert pairs.loc[(1, 2), 'games'] == 2
    assert pairs.loc[(1, 3), 'games'] == 1
    assert (2, 3) not in pairs.index


def test_cooccurrence_pairs_are_unordered():
    """ペアは順序を持たない（小さいID側が a）"""
    recipes = pd.DataFrame([{'game_name': 'A', 'unit': 5},
                            {'game_name': 'A', 'unit': 2}])
    pairs = build_cooccurrence(recipes)
    assert pairs.iloc[0]['unit_a'] == 2
    assert pairs.iloc[0]['unit_b'] == 5


def test_cooccurrence_empty_when_no_game_has_two_units():
    """1ゲームに単位が1つしか無ければ共起は生まれない"""
    recipes = pd.DataFrame([{'game_name': 'A', 'unit': 1},
                            {'game_name': 'B', 'unit': 2}])
    assert build_cooccurrence(recipes).empty


def test_game_similarity_is_jaccard():
    """ゲーム同士の似方は、共有している部品の割合（Jaccard）"""
    from src.nlp.topic_cooccurrence import game_similarity
    memberships = pd.DataFrame([
        {'game_name': 'A', 'unit': 1}, {'game_name': 'A', 'unit': 2},
        {'game_name': 'B', 'unit': 2}, {'game_name': 'B', 'unit': 3},
    ])
    sim = game_similarity(memberships).set_index(['game_a', 'game_b'])
    # 共有1個 / 合計3個
    assert sim.loc[('A', 'B'), 'jaccard'] == pytest.approx(1 / 3)


def test_game_similarity_is_zero_without_overlap():
    """重なりが無ければ0（『似ていない』ではなく『比べる材料が無い』印にもなる）"""
    from src.nlp.topic_cooccurrence import game_similarity
    memberships = pd.DataFrame([{'game_name': 'A', 'unit': 1},
                                {'game_name': 'B', 'unit': 2}])
    assert game_similarity(memberships).iloc[0]['jaccard'] == 0.0


def test_game_similarity_covers_every_pair_once():
    """全ペアが1行ずつ出る（順序違いの重複を作らない）"""
    from src.nlp.topic_cooccurrence import game_similarity
    memberships = pd.DataFrame([{'game_name': g, 'unit': 1} for g in 'ABCD'])
    sim = game_similarity(memberships)
    assert len(sim) == 6
    assert (sim['game_a'] < sim['game_b']).all()
