"""
誤分類サンプルのヒューリスティック自動分類

misclassified.csvの各レビューにルールベースでカテゴリを付与する。
モデルの弱点パターン（皮肉・否定・混合感情など）を傾向把握するための一次分類。

入力:
    data/experiments/sarcasm_baseline/misclassified.csv

出力:
    data/experiments/sarcasm_baseline/misclassified_categorized.csv
    （category列追加・category順 + 各カテゴリ内confidence降順でソート）
"""

import os
import re

import pandas as pd


NEGATION_WORDS = [
    'not', 'never', 'no', "don't", "doesn't", "didn't",
    "can't", "couldn't", "isn't", "wasn't", "aren't", "weren't",
    "won't", "wouldn't", 'nothing', 'nobody', 'nowhere',
]

NEGATIVE_WORDS = [
    'bad', 'awful', 'terrible', 'horrible', 'boring', 'broken',
    'disappointed', 'disappointing', 'wrong', 'useless', 'impossible',
    'worst', 'poor', 'weak', 'dull',
]

MIXED_WORDS = ['but', 'however', 'although', 'though', 'despite']

RATING_PATTERNS = [
    r'\b0/10\b', r'\b10/10\b', r'★', r'\b100%\b',
    r'\b1/10\b', r'\b9/10\b',
]

SLANG_WORDS = ['fr', 'no cap', 'slaps', 'mid', 'ngl', 'based', 'cringe']

# カテゴリ判定の優先順位
CATEGORY_ORDER = [
    '二重否定', '単純否定', '混合感情', '評価記号矛盾',
    'スラング', '全大文字強調', '短文', 'その他',
]


def _contains_word(text_lower: str, words: list[str]) -> bool:
    """単語の境界を考慮した部分一致判定"""
    for w in words:
        # 単語境界で囲んで検索（アポストロフィを含む場合は別処理）
        pattern = r'\b' + re.escape(w) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def _has_double_negation(text_lower: str) -> bool:
    """否定語の近くにネガティブ語があるか（5単語以内）"""
    tokens = re.findall(r"[a-z']+", text_lower)
    negation_indices = [i for i, t in enumerate(tokens) if t in NEGATION_WORDS]
    negative_set = set(NEGATIVE_WORDS)

    for idx in negation_indices:
        # 否定語の直後5単語以内にネガティブ語があるか
        for j in range(idx + 1, min(idx + 6, len(tokens))):
            if tokens[j] in negative_set:
                return True
    return False


def _is_all_caps(text: str, threshold: float = 0.5) -> bool:
    """テキストの大文字率が閾値以上か（アルファベットに対する割合）"""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 10:  # 短すぎる場合は判定しない
        return False
    upper_count = sum(1 for c in letters if c.isupper())
    return upper_count / len(letters) >= threshold


def classify(text: str) -> str:
    """1レビューにカテゴリを付与"""
    if not isinstance(text, str) or not text:
        return 'その他'

    text_lower = text.lower()

    # 1. 二重否定（最優先 - 二重否定は意味が反転して肯定になる）
    if _has_double_negation(text_lower):
        return '二重否定'

    # 2. 単純否定
    if _contains_word(text_lower, NEGATION_WORDS):
        return '単純否定'

    # 3. 混合感情
    if _contains_word(text_lower, MIXED_WORDS):
        return '混合感情'

    # 4. 評価記号矛盾
    for pattern in RATING_PATTERNS:
        if re.search(pattern, text):
            return '評価記号矛盾'

    # 5. スラング
    if _contains_word(text_lower, SLANG_WORDS):
        return 'スラング'

    # 6. 全大文字強調
    if _is_all_caps(text):
        return '全大文字強調'

    # 7. 短文
    if len(text) < 50:
        return '短文'

    return 'その他'


def main():
    input_path = 'data/experiments/sarcasm_baseline/misclassified.csv'
    output_path = 'data/experiments/sarcasm_baseline/misclassified_categorized.csv'

    print('=' * 70)
    print('誤分類サンプルのヒューリスティック自動分類')
    print('=' * 70)

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f'{input_path} が見つかりません。\n'
            f'先に make exec CMD="python scripts/experiments/analyze_misclassified.py" を実行してください。'
        )

    df = pd.read_csv(input_path)
    print(f'\n[1/3] 入力読み込み: {len(df)}件')

    # カテゴリ付与
    print('[2/3] カテゴリ判定中...')
    df['category'] = df['review_text'].apply(classify)

    # ソート: category順 → 各カテゴリ内confidence降順
    category_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    df['_category_rank'] = df['category'].map(category_rank)
    df = df.sort_values(
        by=['_category_rank', 'confidence'],
        ascending=[True, False]
    ).drop(columns=['_category_rank']).reset_index(drop=True)

    # 保存
    df.to_csv(output_path, index=False)
    print(f'[3/3] 出力保存: {output_path}')

    # 集計表示
    print('\n【カテゴリ別件数（全体）】')
    counts = df['category'].value_counts()
    for cat in CATEGORY_ORDER:
        n = counts.get(cat, 0)
        pct = n / len(df) * 100 if len(df) else 0
        print(f'  {cat:10} : {n:>4}件 ({pct:5.1f}%)')

    # confidence帯別の集計
    print('\n【カテゴリ × confidence帯】')
    print(f'  {"カテゴリ":10} | {"≥0.9":>5} | {"0.7-0.9":>8} | {"<0.7":>5}')
    print('  ' + '-' * 45)
    for cat in CATEGORY_ORDER:
        sub = df[df['category'] == cat]
        high = (sub['confidence'] >= 0.9).sum()
        mid = ((sub['confidence'] >= 0.7) & (sub['confidence'] < 0.9)).sum()
        low = (sub['confidence'] < 0.7).sum()
        if high + mid + low > 0:
            print(f'  {cat:10} | {high:>5} | {mid:>8} | {low:>5}')

    # FP/FN別の集計
    print('\n【カテゴリ × FP/FN】')
    print(f'  {"カテゴリ":10} | {"FP":>5} | {"FN":>5}')
    print('  ' + '-' * 30)
    for cat in CATEGORY_ORDER:
        sub = df[df['category'] == cat]
        fp = (sub['error_type'] == 'FP').sum()
        fn = (sub['error_type'] == 'FN').sum()
        if fp + fn > 0:
            print(f'  {cat:10} | {fp:>5} | {fn:>5}')

    print('\n' + '=' * 70)
    print('分類完了')
    print('=' * 70)


if __name__ == '__main__':
    main()
