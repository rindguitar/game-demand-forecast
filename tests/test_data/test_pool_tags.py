"""
母集団キャッシュからタグを読むモジュールのテスト
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.pool_tags import load_pool, load_pool_tags


def _cache(tmp_path, data):
    path = tmp_path / 'pool.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return str(path)


def test_load_pool_drops_order_key(tmp_path):
    """__order__ はゲームではないので除く"""
    path = _cache(tmp_path, {'__order__': [1, 2], '10': {'name': 'A', 'tags': ['X']}})
    pool = load_pool(path)
    assert list(pool) == ['10']


def test_load_pool_tags_makes_long_format(tmp_path):
    """ゲーム × タグ の縦長になる"""
    path = _cache(tmp_path, {'10': {'name': 'A', 'tags': ['X', 'Y']},
                             '11': {'name': 'B', 'tags': ['Y']}})
    df = load_pool_tags(path)
    assert len(df) == 3
    assert set(df.columns) == {'game_name', 'tag'}
    assert set(df[df.game_name == 'A']['tag']) == {'X', 'Y'}


def test_load_pool_tags_drops_games_without_tags(tmp_path):
    """タグを持たないゲームは落とす（母集団の143本がこれに当たる）"""
    path = _cache(tmp_path, {'10': {'name': 'A', 'tags': ['X']},
                             '11': {'name': 'B', 'tags': []},
                             '12': {'name': 'C'}})
    assert set(load_pool_tags(path)['game_name']) == {'A'}


def test_load_pool_tags_filters_to_given_games(tmp_path):
    """ロスターとの比較用に、指定したゲームだけへ絞れる"""
    path = _cache(tmp_path, {'10': {'name': 'A', 'tags': ['X']},
                             '11': {'name': 'B', 'tags': ['Y']}})
    assert set(load_pool_tags(path, only_games=['B'])['game_name']) == {'B'}


def test_load_pool_tags_returns_empty_frame_with_columns(tmp_path):
    """1本も残らなくても列は保つ（呼び出し側が落ちないように）"""
    df = load_pool_tags(_cache(tmp_path, {'10': {'name': 'A', 'tags': []}}))
    assert df.empty and list(df.columns) == ['game_name', 'tag']
