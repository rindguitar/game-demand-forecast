"""
2モデルの誤分類CSVを差分し、変化したレビューを抽出する

同じ未知データに対する旧モデル(--before)と新モデル(--after)の誤分類CSVを比較し、
review_text単位で集合差を取る:
    fixed : beforeは誤り → afterは正解（新モデルが直したレビュー）
    broke : beforeは正解 → afterは誤り（新モデルが壊したレビュー）

各CSVは analyze_misclassified.py の出力（誤分類のみ・review_text/error_type列を持つ）を想定。
出力した fixed.csv / broke.csv は categorize_misclassified.py でタグ付け分析できる。
"""

import os
import argparse

import pandas as pd


def _fp_fn(df: pd.DataFrame) -> str:
    """FP/FN内訳を文字列で返す"""
    fp = (df['error_type'] == 'FP').sum()
    fn = (df['error_type'] == 'FN').sum()
    return f'{len(df)}件（FP {fp} / FN {fn}）'


def main():
    parser = argparse.ArgumentParser(description='2モデルの誤分類を差分（fixed/broke抽出）')
    parser.add_argument('--before', required=True, help='旧モデルの誤分類CSV')
    parser.add_argument('--after', required=True, help='新モデルの誤分類CSV')
    parser.add_argument('--output-dir', required=True, help='fixed.csv / broke.csv の出力先')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    before = pd.read_csv(args.before)
    after = pd.read_csv(args.after)

    before_texts = set(before['review_text'])
    after_texts = set(after['review_text'])

    # fixed: beforeで誤り かつ afterの誤りに含まれない → 新モデルが直した
    fixed = before[~before['review_text'].isin(after_texts)].copy()
    # broke: afterで誤り かつ beforeの誤りに含まれない → 新モデルが壊した
    broke = after[~after['review_text'].isin(before_texts)].copy()

    fixed_path = os.path.join(args.output_dir, 'fixed.csv')
    broke_path = os.path.join(args.output_dir, 'broke.csv')
    fixed.to_csv(fixed_path, index=False)
    broke.to_csv(broke_path, index=False)

    print('=' * 70)
    print('誤分類の差分（before → after）')
    print('=' * 70)
    print(f'  before誤分類: {len(before)}件 / after誤分類: {len(after)}件')
    print(f'  fixed（afterが直した）: {_fp_fn(fixed)} → {fixed_path}')
    print(f'  broke（afterが壊した）: {_fp_fn(broke)} → {broke_path}')
    print(f'  純改善: {len(fixed) - len(broke):+d}件')


if __name__ == '__main__':
    main()
