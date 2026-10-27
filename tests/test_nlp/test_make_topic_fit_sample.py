"""
学習用サンプル作成スクリプト（make_topic_fit_sample.py）のテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/nlp'))

import pandas as pd
from make_topic_fit_sample import make_fit_sample


def _write_csv(path, games, n_each, empty_rows=0):
    """ゲームごとに n_each 件のレビューを持つ小さな CSV を作る"""
    rows = []
    for gid, name in enumerate(games):
        for i in range(n_each):
            rows.append({'game_id': gid, 'game_name': name, 'review_text': f'{name} review {i}'})
    for _ in range(empty_rows):
        rows.append({'game_id': 99, 'game_name': games[0], 'review_text': ''})
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_make_fit_sample_per_game_limit_caps_each_game(tmp_path):
    """1本あたりの上限を超えない"""
    p = _write_csv(tmp_path / 'a.csv', ['A', 'B'], 20)
    result = make_fit_sample([p], per_game=5, total=0, seed=1)
    assert result['game_name'].value_counts().max() == 5


def test_make_fit_sample_total_limit_caps_rows(tmp_path):
    """全体の上限を超えない"""
    p = _write_csv(tmp_path / 'a.csv', ['A', 'B'], 20)
    result = make_fit_sample([p], per_game=10, total=7, seed=1)
    assert len(result) == 7


def test_make_fit_sample_empty_review_text_is_dropped(tmp_path):
    """review_text が空の行は落とす"""
    p = _write_csv(tmp_path / 'a.csv', ['A'], 3, empty_rows=4)
    result = make_fit_sample([p], per_game=100, total=0, seed=1)
    assert len(result) == 3


def test_make_fit_sample_two_files_both_games_included(tmp_path):
    """2ファイルのゲームが両方入る"""
    p1 = _write_csv(tmp_path / 'a.csv', ['A'], 5)
    p2 = _write_csv(tmp_path / 'b.csv', ['B'], 5)
    result = make_fit_sample([p1, p2], per_game=3, total=0, seed=1)
    assert set(result['game_name']) == {'A', 'B'}


def test_make_fit_sample_same_seed_same_result(tmp_path):
    """同じ seed なら同じ結果"""
    p = _write_csv(tmp_path / 'a.csv', ['A', 'B'], 30)
    r1 = make_fit_sample([p], per_game=10, total=12, seed=3)
    r2 = make_fit_sample([p], per_game=10, total=12, seed=3)
    assert r1['review_text'].tolist() == r2['review_text'].tolist()
