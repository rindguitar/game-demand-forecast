"""
トピックの分類モジュール

抽出したトピックを ①ゲーム要素 ②品質・運営 ③ビジネス条件 ＋ 中身なし に仕分ける。

①ゲーム要素は**証拠のある分類**にする。かつては「どの語彙にも当たらなければ①」という
残余だったため、語彙の穴がすべて需要スコアの対象に落ちていた（実測: 64本で boobs /
braindead / money / ruined life が①に混ざった）。①にも語彙を持たせ、
どこにも当たらないものは「未分類」に落とす。
需要スコアは①だけを合算し、③は阻害要因として別枠に持つ（docs/decisions.md 2026-08-18）。

仕分けは「ルール → 曖昧なものだけ手動 → その結果を教師データに分類器」の3段構えで、
このモジュールが担うのは第一段のルール。
"""

from typing import Dict, List, Tuple, Optional
import os
import re

# 分類の識別子（CSVにもこの値が入る）
ELEMENT = 'element'
QUALITY = 'quality'
BUSINESS = 'business'
CONTENTLESS = 'contentless'
PROPERNOUN = 'propernoun'
AMBIGUOUS = 'ambiguous'
UNCLASSIFIED = 'unclassified'

# ①の証拠としてタグとの近さを使うときの閾値（実測で決める。→ docs/decisions.md）
# 0.5前後は当てにならない帯なので、強い証拠と弱い証拠を分けて扱う
STRONG_ELEMENT = 0.65
WEAK_ELEMENT = 0.50

# 表示用の日本語名
CATEGORY_LABELS = {
    ELEMENT: '①ゲーム要素',
    QUALITY: '②品質・運営',
    BUSINESS: '③ビジネス条件',
    CONTENTLESS: '中身なし',
    PROPERNOUN: '固有名詞',
    AMBIGUOUS: '要手動判定',
    UNCLASSIFIED: '未分類',
}

# 設定ファイルに書ける見出し（[quality] など）
_SECTIONS = (ELEMENT, QUALITY, BUSINESS, CONTENTLESS, PROPERNOUN)


def load_category_words(path: str) -> Dict[str, List[str]]:
    """
    分類語彙のファイルを読む（`[見出し]` で区切り・1行1語・# はコメント）

    見出しに無いセクションは無視する。ファイルが無いときは黙って空を返さず警告する
    （「分類したつもり」で全部が①ゲーム要素になるのを防ぐため）。
    """
    words: Dict[str, List[str]] = {s: [] for s in _SECTIONS}
    if not path or not os.path.exists(path):
        print(f"⚠️ 分類語彙のファイルが見つかりません: {path}（全トピックが①ゲーム要素になります）")
        return words

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
            if current in words:
                words[current].append(line.lower())
    return words


def _matched_words(keywords: str, vocabulary: List[str]) -> List[str]:
    """キーワード文字列の中に、語彙の語（語句）が単語として現れるものを返す"""
    text = str(keywords).lower()
    hits = []
    for w in vocabulary:
        # 語句もそのまま扱う。語の間の空白は空白以外の区切りも許す
        pattern = r'\b' + r'\W+'.join(re.escape(p) for p in w.split()) + r'\b'
        if re.search(pattern, text):
            hits.append(w)
    return hits


def classify_topic(keywords: str,
                   category_words: Dict[str, List[str]],
                   element_score: float = 0.0,
                   strong_element: float = STRONG_ELEMENT,
                   weak_element: float = WEAK_ELEMENT
                   ) -> Tuple[str, Dict[str, List[str]]]:
    """
    トピックのキーワードから分類を1つ決める

    1. 分類ごとに、当たった語を数える
    2. ①の証拠は語の一致だけでなく「タグとの意味の近さ」でも受け取る（element_score）
    3. 1つも当たらなければ **未分類**（①ではない。証拠が無いものを需要スコアに入れない）
    4. 固有名詞に当たっていればそれで確定（束ねる対象から確実に外すため）
    5. ①の証拠が強ければそれで確定。語の数では拾えない形の証拠なので、多数決に混ぜない
    6. 最多の分類が1つに決まればそれ。同数で並んだら AMBIGUOUS（手動送り）

    語の一致（○×）だけだと証拠の強さを比べられない。実測では
    `sandbox, sandbox game, best sandbox` が `game best` の1票だけで中身なしに落ちていた。
    タグ `Sandbox` との近さ 0.78 を強い証拠として扱えば、正しく①になる。

    Args:
        element_score: ①の語彙（Steamタグ）といちばん近い語との近さ（0〜1）
        strong_element: これ以上なら①で確定する
        weak_element: これ以上なら①に1票入れる（多数決に参加する）

    Returns:
        (分類の識別子, 分類ごとに当たった語)
    """
    hits = {c: _matched_words(keywords, ws) for c, ws in category_words.items()}
    counts = {c: len(v) for c, v in hits.items() if v}
    if element_score >= weak_element:
        counts[ELEMENT] = counts.get(ELEMENT, 0) + 1
    if not counts:
        return UNCLASSIFIED, hits

    # 固有名詞は当たった時点で確定させる。束ねてはいけないものなので、
    # 他の分類と同数で並んで ambiguous に落ちると取りこぼす
    if counts.get(PROPERNOUN):
        return PROPERNOUN, hits

    # 意味の近さは語の数と単位が違うので、強い証拠は多数決の外で確定させる
    if element_score >= strong_element:
        return ELEMENT, hits

    top = max(counts.values())
    winners = [c for c, n in counts.items() if n == top]
    return (winners[0] if len(winners) == 1 else AMBIGUOUS), hits


def classify_topics(topics: List[Tuple[int, str]],
                    category_words: Dict[str, List[str]],
                    element_scores: Optional[Dict[int, float]] = None,
                    **thresholds
                    ) -> List[Tuple[int, str, str, Dict[str, List[str]]]]:
    """
    トピックの一覧をまとめて分類する

    Args:
        topics: (topic_id, keywords) の並び。Outlier（topic_id = -1）は呼び出し側で除く
        category_words: load_category_words() の戻り値
        element_scores: topic_id → ①の語彙との近さ。省略すると語の一致だけで決める

    Returns:
        (topic_id, keywords, 分類, 当たった語) の並び
    """
    scores = element_scores or {}
    result = []
    for topic_id, keywords in topics:
        category, hits = classify_topic(keywords, category_words,
                                        element_score=scores.get(topic_id, 0.0),
                                        **thresholds)
        result.append((topic_id, keywords, category, hits))
    return result


def format_hits(hits: Dict[str, List[str]], limit: Optional[int] = 3) -> str:
    """当たった語を「quality: bug, crash / business: price」の形にする（表示用）"""
    parts = []
    for category, words in hits.items():
        if not words:
            continue
        shown = words[:limit] if limit else words
        tail = f"+{len(words) - len(shown)}" if limit and len(words) > len(shown) else ''
        parts.append(f"{category}: {', '.join(shown)}{tail}")
    return ' / '.join(parts)
