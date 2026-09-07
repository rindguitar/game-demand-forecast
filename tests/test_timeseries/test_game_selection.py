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
            'tags': set(tags), 'release_date': release, 'total_reviews': total,
            'total_positive': 0, 'total_negative': 0}


def args(**kw):
    base = dict(seed=42, n_games=6, genre_floor=3, max_backbone=14,
                max_per_genre=12, tag_overlap_threshold=2)
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
