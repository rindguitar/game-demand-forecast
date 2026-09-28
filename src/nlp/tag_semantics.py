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

from typing import Dict, List, Optional, Sequence, Tuple
import os
import re

import numpy as np

# トピック抽出（BERTopic）と同じモデル。別のモデルを使うと空間が揃わない
DEFAULT_MODEL = 'all-MiniLM-L6-v2'

# タグ判定ファイルの既定パス。生成側（configs への書き出し）と照合側（意味の近さ）で
# 同じ定義を使うため、判定は1か所（このファイル）だけに置く
DEFAULT_TAG_JUDGMENTS = 'configs/steam_tags.txt'

# 判定ファイルに書ける見出し（[not_content] など）
NOT_CONTENT = 'not_content'
IMPRESSION = 'impression'
ELEMENT = 'element'
_JUDGMENT_SECTIONS = (NOT_CONTENT, IMPRESSION, ELEMENT)


def load_tag_judgments(path: str = DEFAULT_TAG_JUDGMENTS) -> Dict[str, str]:
    """タグの判定ファイルを読む（`[見出し]` で区切り・1行1タグ・# はコメント）

    見出しに無いセクションは無視する（topic_category.load_category_words にならう）。
    ファイルが無いときは黙って空を返さずエラーにする
    （判定なしを「除外なし」と取り違えて、未判定のタグまで①に入るのを防ぐため）。
    同じタグが2つの見出しにあってもエラーにする（見出しを移すとき元の行を消し忘れると、
    後に書いた [element] が黙って勝つため）。
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'タグの判定ファイルが見つかりません: {path}')

    judgments: Dict[str, str] = {}
    current = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            header = re.fullmatch(r'\[(\w+)\]', line)
            if header:
                current = header.group(1).lower()
                continue
            if current in _JUDGMENT_SECTIONS:
                if line in judgments:
                    raise ValueError(f'タグ {line} が [{judgments[line]}] と [{current}] の'
                                     f'両方にあります: {path}')
                judgments[line] = current
    return judgments


def element_tags(pool: dict, judgments: Optional[Dict[str, str]] = None) -> list:
    """母集団から①ゲーム要素の語彙になるタグを集める

    1. 母集団のタグを集める
    2. 判定の無いタグがあれば ValueError で止める（新しいタグを黙って①に入れないため）
    3. `element` と判定されたタグだけを sorted で返す

    Args:
        judgments: タグ → 判定（not_content / impression / element）。
            省略時は DEFAULT_TAG_JUDGMENTS を読む
    """
    if judgments is None:
        judgments = load_tag_judgments()

    tags = {t for v in pool.values() if isinstance(v, dict) for t in (v.get('tags') or [])}
    unjudged = sorted(t for t in tags if t not in judgments)
    if unjudged:
        raise ValueError(f'判定の無いタグがあります: {", ".join(unjudged)}'
                         '（新しいタグは判定ファイルに追記すること）')
    return sorted(t for t in tags if judgments[t] == ELEMENT)


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
