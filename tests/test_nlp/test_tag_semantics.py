"""
トピックとSteamタグを意味の近さで照合するモジュールのテスト

文字列照合では `Souls-like` と `soulslike` が別物になる。実測で一致率が43.4%止まり、
語彙を241語→321語に広げても5ポイントしか動かなかったため、照合方法を変えた。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from src.nlp.tag_semantics import NOT_ELEMENT_TAGS, element_tags, match_terms


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
    """中身を表さないタグは①の語彙にしない（生成側と照合側で同じ定義を使う）"""
    pool = {'1': {'tags': ['Horror', 'Indie', 'Free to Play']},
            '2': {'tags': ['Early Access', 'Roguelike']}}
    assert element_tags(pool) == ['Horror', 'Roguelike']


def test_element_tags_keeps_how_you_play():
    """「誰と遊ぶか」は①の語彙に残す（企画で決められる要素なので）"""
    assert element_tags({'1': {'tags': ['Co-op', 'PvP']}}) == ['Co-op', 'PvP']


def test_element_tags_tolerates_missing_tags():
    """タグを持たないゲームが混ざっても落ちない"""
    assert element_tags({'1': {'tags': ['Horror']}, '2': {}, '3': {'tags': []}}) == ['Horror']


def test_free_to_play_is_excluded_as_business_condition():
    """Free to Play は③ビジネス条件。①に入れると 2026-08-18 の決定と矛盾する"""
    assert 'Free to Play' in NOT_ELEMENT_TAGS
    assert element_tags({'1': {'tags': ['Free to Play']}}) == []
