"""
山の続き方と要素の関係モジュールのテスト

要素の表の作り方・最小本数の絞り込み・中央値と差・ペア・シャッフル・max_diff_95・
相対値が空の発売の除外を、手で数えられる小さい表で確かめる。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest
import pandas as pd
from src.timeseries.persistence_elements import (
    compare_items,
    compare_with_shuffle,
    game_tag_sets,
    membership_matrix,
    pair_membership,
    recipe_element_sets,
    shuffle_test,
    unit_keywords,
    usable_releases,
)


def _values(n=10):
    return pd.Series(np.linspace(0.5, 1.5, n), index=[f'g{i}' for i in range(n)])


def test_usable_releases_drops_missing():
    df = pd.DataFrame({'name': ['a', 'b', 'c'], 'relative_mid_count': [1.0, np.nan, 0.8]})
    assert list(usable_releases(df).index) == ['a', 'c']


def test_recipe_element_sets_only_element_category():
    recipes = pd.DataFrame({
        'game_name': ['a', 'a', 'b'], 'unit': [1, 2, 3],
        'keywords': ['x', 'y', 'z'], 'category': ['element', 'quality', 'element']})
    assert recipe_element_sets(recipes) == {'a': {1}, 'b': {3}}
    assert unit_keywords(recipes) == {1: 'x', 3: 'z'}


def test_game_tag_sets_top_n_and_element_only():
    games = pd.DataFrame({'name': ['a'], 'tags': ['T1|T2|T3|T4']})
    judg = {'T1': 'element', 'T2': 'impression', 'T3': 'element', 'T4': 'impression'}
    # 上位2個のうち element は T1 だけ。T4 は impression で外れる
    assert game_tag_sets(games, judg, top_n=2) == {'a': {'T1'}}
    assert game_tag_sets(games, judg, top_n=4) == {'a': {'T1', 'T3'}}


def test_game_tag_sets_unjudged_tag_stops():
    games = pd.DataFrame({'name': ['a', 'b'], 'tags': ['T1|NEW', 'T1']})
    with pytest.raises(ValueError, match=r'NEW（a）'):
        game_tag_sets(games, {'T1': 'element'})
    # 対象外の発売にだけある未判定タグでは止まらない
    assert game_tag_sets(games, {'T1': 'element'}, only=['b']) == {'b': {'T1'}}


def test_membership_matrix_shape():
    m = membership_matrix({'a': {1, 2}, 'b': {2}}, ['a', 'b', 'c'])
    assert m.loc['a', 1] and not m.loc['b', 1] and not m.loc['c', 2]


def test_compare_items_median_diff_and_min_count():
    values = pd.Series([2.0, 1.8, 1.6, 0.4, 0.6, 0.8], index=list('abcdef'))
    sets = {'a': {1}, 'b': {1}, 'c': {1}, 'd': {2}, 'e': {2}}
    m = membership_matrix(sets, values.index)
    table = compare_items(m, values, min_count=3)
    # 要素2は2本しか持たないので外れる
    assert list(table['item']) == [1]
    row = table.iloc[0]
    assert row['n_have'] == 3 and row['n_lack'] == 3
    assert row['median_have'] == 1.8 and row['median_lack'] == 0.6
    assert np.isclose(row['diff'], 1.2)
    assert row['games'] == 'a, b, c'


def test_pair_membership_needs_both_and_min_count():
    sets = {f'g{i}': {1, 2} for i in range(3)}
    sets.update({'g3': {1}, 'g4': {2}})
    m = membership_matrix(sets, [f'g{i}' for i in range(5)])
    pairs = pair_membership(m, min_count=3)
    assert list(pairs.columns) == [(1, 2)]
    assert pairs[(1, 2)].sum() == 3
    assert pair_membership(m, min_count=4).shape[1] == 0


def test_shuffle_same_seed_same_result():
    values = _values()
    m = membership_matrix({f'g{i}': {1} for i in range(5)}, values.index)
    p1, l1 = shuffle_test(m, values, [1], n_shuffle=200, seed=3)
    p2, l2 = shuffle_test(m, values, [1], n_shuffle=200, seed=3)
    assert np.array_equal(p1, p2) and l1 == l2


def test_shuffle_separating_item_has_small_p_random_item_large_p():
    values = _values(12)
    # 要素1は相対値の上位6本をきれいに分ける。要素2は交互でほぼ差が出ない
    sets = {f'g{i}': ({1} if i >= 6 else set()) | ({2} if i % 2 == 0 else set())
            for i in range(12)}
    m = membership_matrix(sets, values.index)
    table, _ = compare_with_shuffle(m, values, min_count=3, n_shuffle=500, seed=0)
    p = table.set_index('item')['p_shuffle']
    assert p[1] < 0.05
    assert p[2] > 0.3


def test_max_diff_95_and_beyond_line():
    values = _values(12)
    sets = {f'g{i}': {1} for i in range(6, 12)}
    m = membership_matrix(sets, values.index)
    table, line = compare_with_shuffle(m, values, min_count=3, n_shuffle=500, seed=0)
    # 要素が1つなら、線は「その要素の |差| のシャッフル分布の95パーセンタイル」
    assert line > 0
    assert bool(table.iloc[0]['beyond_line']) == (abs(table.iloc[0]['diff']) > line)
    assert table.iloc[0]['beyond_line']  # 最も極端な分け方なので線を越える


def test_empty_when_no_item_reaches_min_count():
    values = _values(6)
    m = membership_matrix({'g0': {1}}, values.index)
    table, line = compare_with_shuffle(m, values, min_count=3, n_shuffle=10)
    assert table.empty and np.isnan(line)
