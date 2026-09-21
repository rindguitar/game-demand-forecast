"""
トピックとSteamタグを「意味の近さ」で照合するモジュール

①ゲーム要素の語彙にタグを使うと決めたが（→ docs/decisions.md 2026-09-21）、
文字列照合では拾えない。概念はタグ語彙にあるのに、書かれ方が違うため。

    souls, soulslike   ↔  Souls-like     ハイフンの有無
    hard, challenge    ↔  Difficult      同義語
    play friends       ↔  Co-op          まったく別の語

実測では文字列照合が43.4%止まりで、語彙を241語→321語に広げても5ポイントしか
動かなかった。不足しているのは語彙ではなく照合方法だった。

トピック抽出と同じ埋め込みモデルを使うので、追加の学習は要らない。
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# トピック抽出（BERTopic）と同じモデル。別のモデルを使うと空間が揃わない
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# ゲームの中身を表さないタグ。①ゲーム要素の語彙にも、意味の照合にも使わない
#   Indie / Early Access  開発規模・販売形態であって遊びの中身ではない
#   Free to Play          ③ビジネス条件。需要量に合算せず阻害要因の別枠に置く
#                         （docs/decisions.md 2026-08-18）ので①に入れてはいけない
NOT_ELEMENT_TAGS = {'Indie', 'Early Access', 'Free to Play', 'Free To Play'}


def element_tags(pool: dict) -> list:
    """母集団から①ゲーム要素の語彙になるタグを集める

    生成側（configs への書き出し）と照合側（意味の近さ）で同じ語彙を使うため、
    ここに1つだけ置く。片方にしか除外が効いていないと、`free` が①に入る。
    """
    tags = {t for v in pool.values() if isinstance(v, dict) for t in (v.get('tags') or [])}
    return sorted(t for t in tags if t not in NOT_ELEMENT_TAGS)


def load_encoder(model_name: str = DEFAULT_MODEL):
    """埋め込みモデルを読む（重いので呼び出し側で使い回すこと）"""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def match_terms(texts: Sequence[str], vocabulary: Sequence[str],
                encoder=None) -> List[Tuple[str, float]]:
    """それぞれのテキストに、いちばん意味の近い語とその近さを返す

    近さは0〜1のコサイン類似度。1に近いほど同じ意味。
    実測の目安: Souls-like ↔ soulslike が 0.69、Difficult ↔ hard が 0.53、
    RPG ↔ game game（無関係）が 0.52。**0.5前後は当てにならない帯**なので、
    呼び出し側で閾値を分けて扱うこと。

    Returns:
        (いちばん近い語, その近さ) の並び。語彙が空なら ('', 0.0)
    """
    if not len(vocabulary) or not len(texts):
        return [('', 0.0)] * len(texts)

    encoder = encoder or load_encoder()
    vocab_vec = encoder.encode(list(vocabulary), show_progress_bar=False,
                               normalize_embeddings=True)
    text_vec = encoder.encode([str(t) for t in texts], show_progress_bar=False,
                              normalize_embeddings=True)
    # 正規化済みなので内積がそのままコサイン類似度になる
    sim = np.asarray(text_vec) @ np.asarray(vocab_vec).T
    best = sim.argmax(axis=1)
    return [(vocabulary[i], float(sim[row, i])) for row, i in enumerate(best)]


def score_topics(keywords: Sequence[str], vocabulary: Sequence[str],
                 encoder=None) -> Dict[int, Tuple[str, float]]:
    """トピックの並びに対して、行番号をキーに (近い語, 近さ) を返す"""
    return dict(enumerate(match_terms(keywords, vocabulary, encoder)))
