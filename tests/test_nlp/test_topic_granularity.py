"""
トピックの粒度を粗くするモジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.nlp.topic_granularity import (
    CTFIDF,
    EMBEDDING,
    build_linkage,
    cut_levels,
    merged_keywords,
    topic_matrix,
)

VOCAB = np.array(['alpha', 'beta', 'gamma', 'delta'])
# Outlier + 4トピック。0と1、2と3がそれぞれ近い
ROWS = np.array([
    [0.5, 0.5, 0.5, 0.5],   # -1（Outlier）
    [1.0, 0.0, 0.0, 0.0],   # 0
    [0.9, 0.1, 0.0, 0.0],   # 1
    [0.0, 0.0, 1.0, 0.0],   # 2
    [0.0, 0.0, 0.9, 0.1],   # 3
])


class _FakeVectorizer:
    def get_feature_names_out(self):
        return VOCAB


class _FakeModel:
    """BERTopic のうち、粒度モジュールが触る部分だけを持つ差し替え"""

    def __init__(self):
        self.c_tf_idf_ = csr_matrix(ROWS)
        self.topic_embeddings_ = ROWS
        self.vectorizer_model = _FakeVectorizer()

    def get_topic_info(self):
        return pd.DataFrame({'Topic': [-1, 0, 1, 2, 3]})


def test_topic_matrix_drops_outlier():
    """Outlier（-1）は束ねる対象ではないので行列から外す"""
    matrix, topic_ids = topic_matrix(_FakeModel(), CTFIDF)
    assert topic_ids == [0, 1, 2, 3]
    assert matrix.shape[0] == 4


def test_topic_matrix_embedding_source():
    """embedding を指定すると topic_embeddings_ 側を使う"""
    matrix, topic_ids = topic_matrix(_FakeModel(), EMBEDDING)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)


def test_cut_levels_at_full_size_gives_singletons():
    """トピック数と同じ個数で切ると、1トピック1束になる"""
    matrix, topic_ids = topic_matrix(_FakeModel(), CTFIDF)
    mapping = cut_levels(build_linkage(matrix), topic_ids, [4])[4]
    assert len(set(mapping.values())) == 4


def test_cut_levels_merges_similar_topics():
    """近いトピック同士が同じ束に入る（0と1 / 2と3）"""
    matrix, topic_ids = topic_matrix(_FakeModel(), CTFIDF)
    mapping = cut_levels(build_linkage(matrix), topic_ids, [2])[2]
    assert mapping[0] == mapping[1]
    assert mapping[2] == mapping[3]
    assert mapping[0] != mapping[2]


def test_cut_levels_are_nested():
    """粗いレベルは細かいレベルの入れ子になる（同じ木を切っているため）

    ここが崩れると「粒度だけを動かした」比較にならない。
    """
    matrix, topic_ids = topic_matrix(_FakeModel(), CTFIDF)
    mappings = cut_levels(build_linkage(matrix), topic_ids, [4, 3, 2])
    fine, coarse = mappings[3], mappings[2]
    for a in topic_ids:
        for b in topic_ids:
            if fine[a] == fine[b]:
                assert coarse[a] == coarse[b]


def test_merged_keywords_reflects_members():
    """束のキーワードは、中身のトピックの語を件数で重み付けして作る"""
    model = _FakeModel()
    mapping = {0: 1, 1: 1, 2: 2, 3: 2}
    weights = {0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0}
    keywords = merged_keywords(model, mapping, weights, top_n=2)
    assert keywords[1].split(', ')[0] == 'alpha'
    assert keywords[2].split(', ')[0] == 'gamma'


def test_merged_keywords_weighting_prefers_larger_topic():
    """件数の多いトピックの語が上に来る"""
    model = _FakeModel()
    # トピック3（gamma/delta）が支配的になるように重みを振る
    keywords = merged_keywords(model, {2: 9, 3: 9},
                               {2: 1.0, 3: 10000.0}, top_n=2)
    assert 'delta' in keywords[9]


def test_group_dispersion_is_zero_for_singletons():
    """1トピックだけの束はばらつき0"""
    from src.nlp.topic_granularity import group_dispersion
    values = group_dispersion(_FakeModel(), {0: 1, 1: 2, 2: 3, 3: 4})
    assert all(v == 0.0 for v in values.values())


def test_group_dispersion_grows_when_unrelated_topics_are_merged():
    """無関係なトピックを混ぜた束は、近いもの同士の束より値が大きい

    語の重なりで束ねると意味の遠いものが同居しうるので、その検出に使う。
    """
    from src.nlp.topic_granularity import group_dispersion
    tight = group_dispersion(_FakeModel(), {0: 1, 1: 1})[1]      # 0と1は近い
    loose = group_dispersion(_FakeModel(), {0: 1, 2: 1})[1]      # 0と2は遠い
    assert loose > tight
