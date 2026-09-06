"""
小さいトピックを束ねるモジュール

密度が足りないのは小さいトピックだけなので、大きいトピックはそのまま残し、
小さいものだけをSteamタグの語彙に寄せる（docs/decisions.md 2026-08-31）。
タグに寄らないものは「その他」に集約し、供給側と繋がらない旨を明示する。

束ねてはいけないもの（固有名詞・中身なし）は呼び出し側で除いてから渡す。
"""

from typing import Dict, List, Optional, Sequence, Tuple
import re

# タグに寄らなかったトピックの行き先
OTHER = 'その他'


def load_tag_vocabulary(genre_cells: Sequence[str], tag_cells: Sequence[str]) -> List[str]:
    """
    ゲーム台帳のジャンル列・タグ列から、束ね先の候補を作る

    `Action|Adventure|Racing` のように `|` 区切りで入っているので割って集める。
    長い順に返すのは、`Open World Survival Craft` を `Survival` より先に当てるため。
    """
    vocabulary = set()
    for cells in (genre_cells, tag_cells):
        for cell in cells:
            if not cell or str(cell) == 'nan':
                continue
            for item in str(cell).split('|'):
                item = item.strip().lower()
                if item:
                    vocabulary.add(item)
    return sorted(vocabulary, key=len, reverse=True)


def _tag_matches(keywords: str, tag: str) -> int:
    """トピックのキーワードにタグが何回現れるか（語の間の区切りは問わない）"""
    pattern = r'\b' + r'\W+'.join(re.escape(p) for p in tag.split()) + r'\b'
    return len(re.findall(pattern, str(keywords).lower()))


def assign_bundle(keywords: str, vocabulary: Sequence[str]) -> Optional[str]:
    """
    トピックの束ね先タグを1つ決める（当たらなければ None）

    1. 各タグがキーワードに何回当たるかを数える
    2. 最多のタグを採る。同数なら長いタグ（＝より具体的な方）を採る
       語彙が長い順に並んでいるので、同数のときは先に見つかったものが残る
    """
    best_tag, best_hits = None, 0
    for tag in vocabulary:
        hits = _tag_matches(keywords, tag)
        if hits > best_hits:
            best_tag, best_hits = tag, hits
    return best_tag


def bundle_topics(topics: Sequence[Tuple[int, str]],
                  vocabulary: Sequence[str]) -> Dict[int, str]:
    """
    トピックの一覧に束ね先を割り当てる

    Args:
        topics: (topic_id, keywords) の並び。束ねる対象だけを渡すこと
        vocabulary: load_tag_vocabulary() の戻り値

    Returns:
        topic_id → 束ね先（タグ名、または OTHER）
    """
    return {topic_id: (assign_bundle(keywords, vocabulary) or OTHER)
            for topic_id, keywords in topics}
