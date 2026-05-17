"""
誤分類サンプルのヒューリスティックタグ付け

misclassified.csvの各レビューに、該当する全パターンをタグとして付与する。
カテゴリ排他方式と異なり、1レビューに複数タグが付与可能。

入力:
    data/experiments/sarcasm_baseline/misclassified.csv

出力:
    data/experiments/sarcasm_baseline/misclassified_tagged.csv
    （tags列・tag_count列を追加・confidence降順でソート）
"""

import os
import re
from collections import Counter
from itertools import combinations

import pandas as pd


NEGATION_WORDS = [
    'not', 'never', 'no', "don't", "doesn't", "didn't",
    "can't", "couldn't", "isn't", "wasn't", "aren't", "weren't",
    "won't", "wouldn't", 'nothing', 'nobody', 'nowhere',
]

# 二重否定検出用のネガティブ語
NEGATIVE_WORDS_FOR_DOUBLE_NEG = [
    'bad', 'awful', 'terrible', 'horrible', 'boring', 'broken',
    'disappointed', 'disappointing', 'wrong', 'useless', 'impossible',
    'worst', 'poor', 'weak', 'dull',
]

MIXED_WORDS = ['but', 'however', 'although', 'though', 'despite']

# スラング・口語的なネガティブ表現（標準的なネガティブ語と区別）
STRONG_NEGATIVE_WORDS = [
    'bs', 'garbage', 'trash', 'shit', 'crap', 'suck', 'sucks', 'sucked',
    'cheaters', 'stupid', 'cancer', 'toxic',
]

RATING_PATTERNS = [
    r'\b0/10\b', r'\b10/10\b', r'★', r'\b100%\b',
    r'\b1/10\b', r'\b9/10\b',
]

SLANG_WORDS = ['fr', 'no cap', 'slaps', 'mid', 'ngl', 'based', 'cringe']

# 表示順（集計表示時の順番）
TAG_ORDER = [
    '否定語あり', '二重否定構造', '接続詞混合', '強ネガ表現',
    '評価記号', 'スラング', '全大文字', '短文', '長文',
]


def _contains_word(text_lower: str, words: list[str]) -> bool:
    """単語境界を考慮した部分一致判定"""
    for w in words:
        pattern = r'\b' + re.escape(w) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def _has_double_negation(text_lower: str) -> bool:
    """否定語の直後5単語以内にネガティブ語があるか"""
    tokens = re.findall(r"[a-z']+", text_lower)
    negation_indices = [i for i, t in enumerate(tokens) if t in NEGATION_WORDS]
    negative_set = set(NEGATIVE_WORDS_FOR_DOUBLE_NEG)

    for idx in negation_indices:
        for j in range(idx + 1, min(idx + 6, len(tokens))):
            if tokens[j] in negative_set:
                return True
    return False


def _is_all_caps(text: str, threshold: float = 0.5) -> bool:
    """テキストの大文字率が閾値以上か"""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:
        return False
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters) >= threshold


def assign_tags(text: str) -> list[str]:
    """1レビューに該当する全タグを付与（マルチラベル）"""
    if not isinstance(text, str) or not text:
        return []

    text_lower = text.lower()
    tags = []

    # 否定関連
    if _contains_word(text_lower, NEGATION_WORDS):
        tags.append('否定語あり')
    if _has_double_negation(text_lower):
        tags.append('二重否定構造')

    # 接続詞混合
    if _contains_word(text_lower, MIXED_WORDS):
        tags.append('接続詞混合')

    # 強ネガ表現
    if _contains_word(text_lower, STRONG_NEGATIVE_WORDS):
        tags.append('強ネガ表現')

    # 評価記号
    for pattern in RATING_PATTERNS:
        if re.search(pattern, text):
            tags.append('評価記号')
            break

    # スラング
    if _contains_word(text_lower, SLANG_WORDS):
        tags.append('スラング')

    # 全大文字強調
    if _is_all_caps(text):
        tags.append('全大文字')

    # 長さ
    if len(text) < 50:
        tags.append('短文')
    elif len(text) >= 300:
        tags.append('長文')

    return tags


def main():
    input_path = 'data/experiments/sarcasm_baseline/misclassified.csv'
    output_path = 'data/experiments/sarcasm_baseline/misclassified_tagged.csv'

    print('=' * 70)
    print('誤分類サンプルのヒューリスティックタグ付け')
    print('=' * 70)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'{input_path} が見つかりません。\n'
            f'先に analyze_misclassified.py を実行してください。'
        )

    df = pd.read_csv(input_path)
    print(f'\n[1/3] 入力読み込み: {len(df)}件')

    # タグ付与
    print('[2/3] タグ付け中...')
    df['tags_list'] = df['review_text'].apply(assign_tags)
    df['tags'] = df['tags_list'].apply(lambda lst: ','.join(lst))
    df['tag_count'] = df['tags_list'].apply(len)

    # confidence降順でソート
    df = df.sort_values(by='confidence', ascending=False).reset_index(drop=True)

    # 保存（tags_listは内部用なので除外）
    save_df = df.drop(columns=['tags_list'])
    save_df.to_csv(output_path, index=False)
    print(f'[3/3] 出力保存: {output_path}')

    # --- 集計1: 各タグの出現件数 ---
    print('\n【各タグの出現件数（共起含む・重複カウント）】')
    tag_counter = Counter()
    for tags in df['tags_list']:
        tag_counter.update(tags)

    total = len(df)
    print(f'  {"タグ":12} | {"件数":>5} | {"全体に占める割合":>15}')
    print('  ' + '-' * 50)
    for tag in TAG_ORDER:
        cnt = tag_counter.get(tag, 0)
        pct = cnt / total * 100
        print(f'  {tag:12} | {cnt:>5} | {pct:>14.1f}%')

    # --- 集計2: タグ数の分布 ---
    print('\n【1レビューあたりのタグ数分布】')
    count_dist = df['tag_count'].value_counts().sort_index()
    for n, cnt in count_dist.items():
        pct = cnt / total * 100
        bar = '#' * int(pct / 2)
        print(f'  {n}個: {cnt:>4}件 ({pct:>5.1f}%) {bar}')

    # --- 集計3: タグなしレビュー数 ---
    no_tag = (df['tag_count'] == 0).sum()
    print(f'\n【真の「その他」(タグなし)】: {no_tag}件 ({no_tag / total * 100:.1f}%)')

    # --- 集計4: タグ共起の頻出パターン ---
    print('\n【タグ共起の頻出パターン Top10（2タグ組み合わせ）】')
    pair_counter = Counter()
    for tags in df['tags_list']:
        if len(tags) >= 2:
            for pair in combinations(sorted(tags), 2):
                pair_counter[pair] += 1

    if pair_counter:
        for (a, b), cnt in pair_counter.most_common(10):
            pct = cnt / total * 100
            print(f'  {a} + {b}: {cnt}件 ({pct:.1f}%)')
    else:
        print('  （共起なし）')

    # --- 集計5: タグ × FP/FN ---
    print('\n【タグ × FP/FN】')
    print(f'  {"タグ":12} | {"FP":>5} | {"FN":>5}')
    print('  ' + '-' * 32)
    for tag in TAG_ORDER:
        sub = df[df['tags_list'].apply(lambda lst: tag in lst)]
        fp = (sub['error_type'] == 'FP').sum()
        fn = (sub['error_type'] == 'FN').sum()
        if fp + fn > 0:
            print(f'  {tag:12} | {fp:>5} | {fn:>5}')

    print('\n' + '=' * 70)
    print('タグ付け完了')
    print('=' * 70)


if __name__ == '__main__':
    main()
