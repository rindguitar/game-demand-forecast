"""
ゲーム選定（select_from_pool）のテスト

選定は収集量の問題ではなく需要スコアの定義の一部なので（docs/decisions.md）、
3つの条件が同時に効いていることを固定する。
  1. 各ジャンル最低3本  2. 土台は上限まで  3. タグが2個以上重なるものは入れない
"""

import sys
import os
from argparse import Namespace
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/collect'))

from collect_timeseries_dataset import select_from_pool  # noqa: E402

WINDOW_START = '2023-09-01'   # これ以前の発売＝土台
MIN_HISTORY = '2025-09-01'    # これ以降の発売は履歴が短すぎるので対象外


def game(name, genres, tags, release, total=50000):
    return {'app_id': abs(hash(name)) % 10**6, 'name': name, 'genres': frozenset(genres),
            'tags': list(tags), 'release_date': release, 'total_reviews': total,
            'total_positive': 0, 'total_negative': 0}


def args(**kw):
    base = dict(seed=42, n_games=6, genre_floor=3, max_backbone=14,
                max_per_genre=12, tag_overlap_threshold=2, overlap_tags=6)
    base.update(kw)
    return Namespace(**base)


def make_pool(n_backbone=10, n_recent=10):
    """タグが重ならないゲームを並べた素直な母集団"""
    pool = [game(f'old{i}', ['Action'], [f'tag{i}'], '2020-01-01') for i in range(n_backbone)]
    pool += [game(f'new{i}', ['Action'], [f'ntag{i}'], '2024-06-01') for i in range(n_recent)]
    return pool


def test_backbone_cap_leaves_room_for_new_games():
    """土台の上限が、新しい側の下限として働く"""
    chosen, _ = select_from_pool(make_pool(), WINDOW_START, MIN_HISTORY,
                                 args(n_games=6, max_backbone=2, genre_floor=0))
    backbone = [g for g in chosen if g['release_date'] <= WINDOW_START]
    assert len(chosen) == 6
    assert len(backbone) == 2
    assert len(chosen) - len(backbone) == 4


def test_rare_genre_reaches_the_floor():
    """希少なジャンルでも下限まで選ばれる（頻出ジャンルに枠を食われない）"""
    pool = make_pool(n_backbone=20, n_recent=20)
    pool += [game(f'race{i}', ['Racing'], [f'rtag{i}'], '2020-01-01') for i in range(3)]
    chosen, counts = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                      args(n_games=10, genre_floor=3))
    assert counts['Racing'] == 3
    assert len(chosen) == 10


def test_similar_games_are_excluded():
    """タグが2個以上重なるゲームは同時に選ばれない"""
    pool = [
        game('base', ['Action'], ['Looter Shooter', 'Open World', 'x'], '2020-01-01'),
        game('twin', ['Action'], ['Looter Shooter', 'Open World', 'y'], '2020-01-01'),
        game('other', ['Action'], ['City Builder', 'Management', 'z'], '2020-01-01'),
    ]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=3, genre_floor=0))
    names = {g['name'] for g in chosen}
    assert 'other' in names
    assert not {'base', 'twin'} <= names, 'タグが2個重なる2本が両方選ばれている'


def test_one_shared_tag_is_allowed():
    """1個だけの重なりでは弾かない（しきい値は2個以上）"""
    pool = [
        game('a', ['Action'], ['Open World', 'p'], '2020-01-01'),
        game('b', ['Action'], ['Open World', 'q'], '2020-01-01'),
    ]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=2, genre_floor=0))
    assert len(chosen) == 2


def test_games_without_enough_history_are_excluded():
    """発売から日が浅いゲームは母集団から外れる"""
    pool = [game('fresh', ['Action'], ['t1'], '2026-01-01'),
            game('ok', ['Action'], ['t2'], '2024-01-01')]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=5, genre_floor=0))
    assert [g['name'] for g in chosen] == ['ok']


def test_same_seed_gives_same_result():
    """同じ母集団・同じシードなら選定は再現する"""
    pool = make_pool(15, 15)
    a, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY, args())
    b, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY, args())
    assert [g['name'] for g in a] == [g['name'] for g in b]


def test_genre_cap_prevents_one_genre_from_taking_everything():
    """1ジャンルが全部を占める退化を防ぐ"""
    pool = make_pool(20, 20)
    chosen, counts = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                      args(n_games=10, genre_floor=0, max_per_genre=4))
    assert counts['Action'] == 4
    assert len(chosen) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def existing_entry(g, tier='土台'):
    """台帳から読んだ形（tier を持ち、genres は set）にする"""
    return dict(g, genres=set(g['genres']), tier=tier)


def test_extend_keeps_every_existing_game():
    """既存の台帳は1本も落とさない（収集済みを無駄にしないため）"""
    pool = make_pool()
    keep = [existing_entry(pool[0]), existing_entry(pool[1])]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=6), existing=keep)
    names = [g['name'] for g in chosen]
    assert names[:2] == ['old0', 'old1']
    assert len(chosen) == 6


def test_extend_does_not_pick_existing_games_twice():
    """既存は候補から外す（同じゲームが二重に入らない）"""
    pool = make_pool()
    keep = [existing_entry(g) for g in pool[:3]]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=6), existing=keep)
    names = [g['name'] for g in chosen]
    assert len(names) == len(set(names))


def test_extend_counts_existing_in_genre_cap():
    """ジャンル上限は既存分も数える"""
    pool = make_pool()
    keep = [existing_entry(g) for g in pool[:4]]   # 全部 Action
    chosen, counts = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                      args(n_games=10, max_per_genre=5, genre_floor=0),
                                      existing=keep)
    assert counts['Action'] <= 5
    assert len(chosen) == 5


def test_extend_counts_existing_in_backbone_cap():
    """土台の上限も既存分から数える"""
    pool = make_pool()
    keep = [existing_entry(g) for g in pool[:3]]   # 3本とも土台
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=10, max_backbone=3, genre_floor=0),
                                 existing=keep)
    backbone = [g for g in chosen if g['release_date'] <= WINDOW_START]
    assert len(backbone) == 3


def test_extend_excludes_games_overlapping_with_existing():
    """既存と似たゲームは入らない（重なり判定に既存が効いている）"""
    pool = [game('keepme', ['Action'], ['a', 'b'], '2020-01-01'),
            game('twin', ['Action'], ['a', 'b'], '2020-02-01'),
            game('other', ['Action'], ['x', 'y'], '2020-03-01')]
    chosen, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=3, genre_floor=0),
                                 existing=[existing_entry(pool[0])])
    names = {g['name'] for g in chosen}
    assert 'twin' not in names
    assert names == {'keepme', 'other'}


def test_relaxing_the_threshold_only_adds():
    """条件を緩めて足し直すと、前に選んだものはそのまま残って追加だけ起きる

    段階的に広げても収集をやり直さずに済むための性質。
    """
    # どの2本も 'a' と 'b' の2個で重なる。閾値2なら1本しか入らず、3なら全部入る
    pool = [game(f'g{i}', ['Action'], ['a', 'b', f't{i}'], f'2020-0{i + 1}-01')
            for i in range(5)]
    strict, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                 args(n_games=5, genre_floor=0, tag_overlap_threshold=2),
                                 existing=[])
    loose, _ = select_from_pool(pool, WINDOW_START, MIN_HISTORY,
                                args(n_games=5, genre_floor=0, tag_overlap_threshold=3),
                                existing=[existing_entry(g) for g in strict])
    assert {g['name'] for g in strict} <= {g['name'] for g in loose}
    assert len(loose) > len(strict)


def test_game_master_roundtrip_keeps_numbers_numeric(tmp_path):
    """台帳を保存して読み戻すと件数は数値のまま

    文字列のままだと母集団から選んだゲームと型が食い違い、
    既存を固定して追加するとき（--extend）に表示や集計で落ちる。
    """
    from collect_timeseries_dataset import load_game_master, save_game_master
    path = str(tmp_path / 'games.csv')
    original = dict(game('A', ['Action'], ['x'], '2020-01-01', total=12345), tier='土台')
    save_game_master(path, [original])
    loaded = load_game_master(path)[0]
    assert loaded['total_reviews'] == 12345
    assert isinstance(loaded['app_id'], int)
    assert loaded['genres'] == {'Action'}
    assert loaded['tags'] == ['x']   # 順位を保つのでリスト
    assert loaded['tier'] == '土台'


def test_game_master_roundtrip_tolerates_blank_counts(tmp_path):
    """件数が空でも読める（古い台帳や取得失敗の行で落ちない）"""
    from collect_timeseries_dataset import load_game_master
    path = str(tmp_path / 'games.csv')
    open(path, 'w', encoding='utf-8').write(
        'app_id,name,genres,tags,total_reviews,total_positive,total_negative,'
        'release_date,tier\n1,A,Action,x,,,,2020-01-01,土台\n')
    assert load_game_master(path)[0]['total_reviews'] == 0


def test_top_tag_set_takes_the_highest_ranked():
    """似ている判定は上位n個だけを見る（下位の汎用タグに埋もれないため）"""
    from collect_timeseries_dataset import top_tag_set
    g = {'tags': ['Racing', 'Automobile Sim', 'Open World', 'Action', 'Adventure']}
    assert top_tag_set(g, 2) == {'Racing', 'Automobile Sim'}
    assert len(top_tag_set(g, 99)) == 5
    assert top_tag_set({}, 3) == set()


def test_overlap_judgment_uses_only_the_top_tags():
    """保存数を増やしても、判定に使う数を絞れば似ていないと判定できる

    実測: 上位20個まで見ると Action / Adventure などの汎用タグで
    無関係なゲーム同士が「似ている」と出る。
    """
    racing = game('racing', ['Racing'], [], '2020-01-01')
    social = game('social', ['Casual'], [], '2020-02-01')
    racing['tags'] = ['Racing', 'Automobile Sim', 'Action', 'Adventure', 'Open World']
    social['tags'] = ['Social', 'VR', 'Action', 'Adventure', 'Open World']

    chosen, _ = select_from_pool([racing, social], WINDOW_START, MIN_HISTORY,
                                 args(n_games=2, genre_floor=0, overlap_tags=2))
    assert len(chosen) == 2, '上位2個なら別物と判定されるはず'

    chosen, _ = select_from_pool([racing, social], WINDOW_START, MIN_HISTORY,
                                 args(n_games=2, genre_floor=0, overlap_tags=5))
    assert len(chosen) == 1, '5個まで見ると汎用タグで似ていると誤判定される'


def test_game_master_roundtrip_keeps_tag_order(tmp_path):
    """台帳はタグの順位を保つ（並べ替えると「上位n個」が意味を失う）"""
    from collect_timeseries_dataset import load_game_master, save_game_master
    path = str(tmp_path / 'games.csv')
    original = dict(game('A', ['Action'], [], '2020-01-01'), tier='土台')
    original['tags'] = ['Souls-like', 'Difficult', 'Action', 'Adventure']
    save_game_master(path, [original])
    assert load_game_master(path)[0]['tags'] == ['Souls-like', 'Difficult', 'Action', 'Adventure']


def test_refresh_ledger_tags_replaces_tags_but_not_the_roster(tmp_path):
    """タグだけ入れ替える。ゲームの顔ぶれは変えない"""
    from collect_timeseries_dataset import (load_game_master, refresh_ledger_tags,
                                            save_game_master)
    path = str(tmp_path / 'games.csv')
    a = dict(game('A', ['Action'], [], '2020-01-01'), tier='土台')
    a['tags'] = ['Old']
    b = dict(game('B', ['Action'], [], '2020-02-01'), tier='土台')
    b['tags'] = ['Keep']
    save_game_master(path, [a, b])

    pool = [dict(a, tags=['New', 'Extra', 'More'])]   # Aだけ新しいタグを持つ
    assert refresh_ledger_tags(path, pool) == 1
    loaded = {g['name']: g['tags'] for g in load_game_master(path)}
    assert loaded['A'] == ['New', 'Extra', 'More']
    assert loaded['B'] == ['Keep'], '母集団に無いゲームは触らない'
    assert set(loaded) == {'A', 'B'}, '顔ぶれは変わらない'
