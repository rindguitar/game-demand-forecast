"""
トピックの粒度を粗い側へ動かすモジュール

抽出済みのトピックをマージ木にまとめ、任意の個数で切って「もっと粗い版」を作る。
再学習せずに済むのが要点で、どのレベルも同じ455トピックの入れ子になるため
「クラスタリングが変わったから結果も変わった」の混同を避けられる。

距離とマージ方式は BERTopic の hierarchical_topics に合わせている（コサイン距離 + ward）。
本家と違い fit 時の文書を必要としないので、保存済みモデルだけで動く。

細かい側（トピックを増やす方向）はこの方法では作れない。min_topic_size を下げた
再学習が要る。
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances

# 距離の作り方（--distance の選択肢）
CTFIDF = 'ctfidf'
EMBEDDING = 'embedding'


def topic_matrix(model, source: str = CTFIDF) -> Tuple[np.ndarray, List[int]]:
    """モデルから、トピックを行とする行列と、その行に対応するトピックIDを取り出す

    c_tf_idf_ / topic_embeddings_ の行は get_topic_info() の Topic 列と同じ並びで、
    先頭行が Outlier（-1）。Outlier は束ねる対象ではないので落とす。

    Args:
        source: ctfidf なら語の重なり、embedding なら意味の近さで測る
    """
    topic_ids = model.get_topic_info()['Topic'].tolist()
    matrix = model.c_tf_idf_ if source == CTFIDF else np.asarray(model.topic_embeddings_)
    keep = [i for i, t in enumerate(topic_ids) if t != -1]
    return matrix[keep], [topic_ids[i] for i in keep]


def build_linkage(matrix, method: str = 'ward') -> np.ndarray:
    """トピック間のコサイン距離からマージ木を作る"""
    distances = cosine_distances(matrix)
    # 数値誤差で対角が0でなかったり非対称になったりすると squareform が弾く
    np.fill_diagonal(distances, 0.0)
    distances = (distances + distances.T) / 2
    return linkage(squareform(distances, checks=False), method, optimal_ordering=True)


def cut_levels(tree: np.ndarray, topic_ids: Sequence[int],
               levels: Sequence[int]) -> Dict[int, Dict[int, int]]:
    """マージ木を指定の個数で切り、レベルごとに「元トピック → 束ID」の対応を作る

    同じ高さの木を切るだけなので、粗いレベルは細かいレベルの入れ子になる。
    距離が同じトピックが並ぶと要求した個数ぴったりにならないことがあるので、
    実際にできた個数は呼び出し側で数え直すこと。
    """
    mappings = {}
    for n in levels:
        labels = fcluster(tree, t=n, criterion='maxclust')
        mappings[n] = {int(t): int(g) for t, g in zip(topic_ids, labels)}
    return mappings


def merged_keywords(model, mapping: Dict[int, int], weights: Dict[int, float],
                    top_n: int = 6) -> Dict[int, str]:
    """束ごとのキーワードを、元トピックの c-TF-IDF を件数で重み付けして足して作る

    束の中身を人が読むための表示用。分類語彙との突き合わせにも使う。
    """
    topic_ids = model.get_topic_info()['Topic'].tolist()
    row_of = {t: i for i, t in enumerate(topic_ids)}
    vocab = np.asarray(model.vectorizer_model.get_feature_names_out())

    groups: Dict[int, List[int]] = {}
    for topic_id, group in mapping.items():
        groups.setdefault(group, []).append(topic_id)

    result = {}
    for group, members in groups.items():
        rows = [row_of[t] for t in members]
        w = np.array([max(weights.get(t, 0.0), 1.0) for t in members])
        summed = np.asarray((model.c_tf_idf_[rows].T @ w)).ravel()
        top = vocab[np.argsort(-summed)[:top_n]]
        result[group] = ', '.join(top)
    return result


def group_dispersion(model, mapping: Dict[int, int]) -> Dict[int, float]:
    """束ごとに、中に入ったトピック同士の意味的な離れ具合を出す

    束ね方（c-TF-IDF / embedding）に関わらず、審判はいつも埋め込み空間で行う。
    語の重なりで束ねると `hunting + price` のように語が違えば無関係でも同じ束に入るため、
    「有効単位が増えた」が中身の伴わない増え方かどうかをこの値で見分ける。

    Returns:
        束ID → メンバー同士のコサイン距離の平均（単独の束は 0.0）
    """
    topic_ids = model.get_topic_info()['Topic'].tolist()
    row_of = {t: i for i, t in enumerate(topic_ids)}
    embeddings = np.asarray(model.topic_embeddings_)

    groups: Dict[int, List[int]] = {}
    for topic_id, group in mapping.items():
        groups.setdefault(group, []).append(topic_id)

    result = {}
    for group, members in groups.items():
        if len(members) < 2:
            result[group] = 0.0
            continue
        distances = cosine_distances(embeddings[[row_of[t] for t in members]])
        # 対角（自分自身との距離0）を除いた平均
        n = len(members)
        result[group] = float(distances.sum() / (n * (n - 1)))
    return result
