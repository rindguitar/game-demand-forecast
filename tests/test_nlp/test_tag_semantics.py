"""
トピックとSteamタグを意味の近さで照合するモジュールのテスト

文字列照合では `Souls-like` と `soulslike` が別物になる。実測で一致率が43.4%止まり、
語彙を241語→321語に広げても5ポイントしか動かなかったため、照合方法を変えた。
"""

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pytest
from src.nlp.tag_semantics import element_tags, load_tag_judgments, match_terms


class _FakeEncoder:
    """語ごとに決め打ちのベクトルを返す差し替え（本物のモデルは重いので使わない）"""

    VECTORS = {
        'souls, soulslike': [1.0, 0.0, 0.0],
        'hard, challenge': [0.0, 1.0, 0.0],
        'Souls-like': [0.99, 0.1, 0.0],
        'Difficult': [0.0, 1.0, 0.0],
        'Fishing': [0.0, 0.0, 1.0],
    }

    def encode(self, texts, **kwargs):
        vecs = [np.array(self.VECTORS.get(t, [0.3, 0.3, 0.3]), dtype=float) for t in texts]
        return np.array([v / np.linalg.norm(v) for v in vecs])


def test_match_terms_picks_the_nearest_tag():
    """いちばん意味の近いタグとその近さを返す"""
    got = match_terms(['souls, soulslike', 'hard, challenge'],
                      ['Souls-like', 'Difficult', 'Fishing'], encoder=_FakeEncoder())
    assert [tag for tag, _ in got] == ['Souls-like', 'Difficult']
    assert got[0][1] > 0.9


def test_match_terms_always_returns_a_best_even_when_nothing_fits():
    """何も合わなくても最も近いものが1つ返る。低いスコアが『該当なし』の印になる"""
    (tag, score), = match_terms(['totally unrelated'], ['Fishing'], encoder=_FakeEncoder())
    assert tag == 'Fishing'
    assert score < 0.9


def test_match_terms_handles_empty_inputs():
    """語彙やテキストが空でも落ちない"""
    assert match_terms([], ['Fishing'], encoder=_FakeEncoder()) == []
    assert match_terms(['x'], [], encoder=_FakeEncoder()) == [('', 0.0)]


def test_element_tags_drops_non_elements():
    """中身を表さないタグは①の語彙にしない（生成側と照合側で同じ判定を使う）"""
    judgments = {
        'Horror': 'element', 'Roguelike': 'element', 'Indie': 'not_content',
        'Early Access': 'not_content', 'Free to Play': 'not_content',
    }
    pool = {'1': {'tags': ['Horror', 'Indie', 'Free to Play']},
            '2': {'tags': ['Early Access', 'Roguelike']}}
    assert element_tags(pool, judgments) == ['Horror', 'Roguelike']


def test_element_tags_keeps_how_you_play():
    """「誰と遊ぶか」は①の語彙に残す（企画で決められる要素なので）"""
    judgments = {'Co-op': 'element', 'PvP': 'element'}
    assert element_tags({'1': {'tags': ['Co-op', 'PvP']}}, judgments) == ['Co-op', 'PvP']


def test_element_tags_tolerates_missing_tags():
    """タグを持たないゲームが混ざっても落ちない"""
    judgments = {'Horror': 'element'}
    pool = {'1': {'tags': ['Horror']}, '2': {}, '3': {'tags': []}}
    assert element_tags(pool, judgments) == ['Horror']


def test_free_to_play_is_excluded_as_business_condition():
    """Free to Play は③ビジネス条件。①に入れると 2026-08-18 の決定と矛盾する"""
    judgments = {'Free to Play': 'not_content'}
    assert element_tags({'1': {'tags': ['Free to Play']}}, judgments) == []


def test_element_tags_impression_tag_excluded():
    """遊んだ結果の感想（Addictive 等）は遊ぶ前に分からないので①の語彙にしない"""
    judgments = {'Addictive': 'impression', 'Horror': 'element'}
    pool = {'1': {'tags': ['Addictive', 'Horror']}}
    assert element_tags(pool, judgments) == ['Horror']


def test_element_tags_not_content_tag_excluded():
    """ゲームの中身を表さないタグ（シリーズ物等）は①の語彙にしない"""
    judgments = {'Sequel': 'not_content', 'Horror': 'element'}
    pool = {'1': {'tags': ['Sequel', 'Horror']}}
    assert element_tags(pool, judgments) == ['Horror']


def test_element_tags_unjudged_tag_raises():
    """判定の無いタグがあると、新しいタグを黙って①に入れず ValueError で止める"""
    with pytest.raises(ValueError, match='Mystery Tag'):
        element_tags({'1': {'tags': ['Mystery Tag']}}, judgments={})


def test_load_tag_judgments_missing_file_raises(tmp_path):
    """判定ファイルが無ければエラー（無いものを「除外なし」と取り違えないため）"""
    with pytest.raises(FileNotFoundError):
        load_tag_judgments(str(tmp_path / 'missing.txt'))


def test_load_tag_judgments_parses_each_section(tmp_path):
    """見出し（[not_content] 等）ごとにタグを判定として読む"""
    path = tmp_path / 'judgments.txt'
    path.write_text('[not_content]\nIndie\n\n[impression]\nAddictive\n\n[element]\nHorror\n',
                    encoding='utf-8')
    expected = {'Indie': 'not_content', 'Addictive': 'impression', 'Horror': 'element'}
    assert load_tag_judgments(str(path)) == expected


def test_load_tag_judgments_duplicate_tag_raises(tmp_path):
    """同じタグが2つの見出しにあれば止まる（移し替えで元の行を消し忘れても①に残らない）"""
    path = tmp_path / 'judgments.txt'
    path.write_text('[impression]\nCute\n\n[element]\nCute\nHorror\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Cute'):
        load_tag_judgments(str(path))


def test_steam_tags_judge_every_pool_tag():
    """本物の configs/steam_tags.txt が、本物の母集団の全タグを判定していること

    新しいタグが出たら element_tags が ValueError になるので、そのとき気づけるようにする。
    """
    pool_path = 'data/timeseries/pool_cache.json'
    if not os.path.exists(pool_path):
        pytest.skip(f'{pool_path} が無い環境ではskip')
    with open(pool_path, encoding='utf-8') as f:
        pool = {k: v for k, v in json.load(f).items() if k != '__order__'}
    element_tags(pool)  # 判定の無いタグがあれば ValueError になる
