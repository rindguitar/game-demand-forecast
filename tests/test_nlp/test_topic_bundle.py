"""
トピックを束ねるモジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
from src.nlp.topic_bundle import (
    OTHER,
    assign_bundle,
    bundle_topics,
    load_tag_vocabulary,
)


@pytest.fixture
def vocabulary():
    """台帳の2列から作った束ね先の語彙（長い順）"""
    return load_tag_vocabulary(
        genre_cells=['Action|Strategy', 'Simulation'],
        tag_cells=['Open World|Survival|Open World Survival Craft', 'Card Game|Deckbuilding'],
    )


def test_load_tag_vocabulary_splits_and_dedupes(vocabulary):
    """`|` 区切りを割り、小文字にして重複を潰す"""
    assert 'open world' in vocabulary
    assert 'card game' in vocabulary
    assert vocabulary.count('open world') == 1


def test_load_tag_vocabulary_is_longest_first(vocabulary):
    """長いタグを先に置く（"Open World Survival Craft" を "Survival" より先に当てるため）"""
    assert vocabulary[0] == 'open world survival craft'


def test_load_tag_vocabulary_ignores_empty(vocabulary):
    """空セル・nan は読み飛ばす"""
    assert load_tag_vocabulary(['nan', ''], [None]) == []


def test_assign_bundle_matches_tag(vocabulary):
    """キーワードにタグが現れれば、そのタグに寄る"""
    assert assign_bundle('survival, survival game, best survival', vocabulary) == 'survival'


def test_assign_bundle_matches_multi_word_tag(vocabulary):
    """複数語のタグも当たる（語の間の区切りは問わない）"""
    assert assign_bundle('open, world, best open world game', vocabulary) == 'open world'


def test_assign_bundle_prefers_more_hits(vocabulary):
    """当たった回数が多いタグを採る"""
    keywords = 'survival, survival game, best survival, open world'
    assert assign_bundle(keywords, vocabulary) == 'survival'


def test_assign_bundle_returns_none_when_no_tag(vocabulary):
    """どのタグにも当たらなければ None"""
    assert assign_bundle('tutorial, finished tutorial, tutorials', vocabulary) is None


def test_bundle_topics_falls_back_to_other(vocabulary):
    """タグに寄らないトピックは「その他」に集約する"""
    result = bundle_topics([(1, 'survival game'), (2, 'tutorial, tutorials')], vocabulary)
    assert result == {1: 'survival', 2: OTHER}
