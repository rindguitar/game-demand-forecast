"""
DAPT前後の誤分類差分を可視化

diff_misclassified.py / categorize_misclassified.py の出力を読み、2枚のグラフを生成する。
- 図1（error type別）: FP/FN の「直した(fixed) vs 壊した(broke)」発散バー
- 図2（パターン別）: タグごとの fixed率 vs broke率 グループバー

フォントはDejaVu Sans（日本語非対応）のため、ラベルは英語。

入力（--input-dir）:
    fixed.csv / broke.csv          … error_type列（FP/FN）
    fixed_tagged.csv / broke_tagged.csv … tags列（カンマ区切り）
出力（--output-dir）:
    dapt_diff_errortype.png / dapt_diff_tags.png
"""

import os
import argparse

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

FIXED_COLOR = '#2ca02c'  # 緑＝直した
BROKE_COLOR = '#d62728'  # 赤＝壊した

# 日本語タグ → グラフ用の英語ラベル（表示順）
TAG_LABELS = [
    ('否定語あり', 'Negation'),
    ('接続詞混合', 'Contrastive (but)'),
    ('長文', 'Long text'),
    ('短文', 'Short text'),
    ('条件法', 'Conditional'),
    ('二重否定構造', 'Double negation'),
    ('強ネガ表現', 'Strong negative'),
    ('スラング', 'Slang'),
]


def plot_error_type(fixed: pd.DataFrame, broke: pd.DataFrame, out_path: str):
    """図1: FP/FN の fixed vs broke 発散バー"""
    cats = ['FP (Neg→Pos)', 'FN (Pos→Neg)']
    fixed_cnt = [int((fixed['error_type'] == t).sum()) for t in ('FP', 'FN')]
    broke_cnt = [int((broke['error_type'] == t).sum()) for t in ('FP', 'FN')]

    y = range(len(cats))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(y, fixed_cnt, color=FIXED_COLOR, label='Fixed (preDAPT wrong → DAPT right)')
    ax.barh(y, [-b for b in broke_cnt], color=BROKE_COLOR, label='Broke (preDAPT right → DAPT wrong)')
    ax.axvline(0, color='black', lw=0.8)

    # 件数と純改善を注記
    for i, (f, b) in enumerate(zip(fixed_cnt, broke_cnt)):
        ax.text(f + 1, i, str(f), va='center', ha='left', fontsize=11, color=FIXED_COLOR, fontweight='bold')
        ax.text(-b - 1, i, str(b), va='center', ha='right', fontsize=11, color=BROKE_COLOR, fontweight='bold')
        ax.text(0, i + 0.28, f'net {f - b:+d}', va='center', ha='center', fontsize=9, color='black')

    ax.set_yticks(list(y))
    ax.set_yticklabels(cats, fontsize=11)
    ax.set_xlabel('← broke    |    fixed →   (count)', fontsize=11)
    ax.set_title('What DAPT fixed vs broke on OOD (by error type)', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  ✅ {out_path}  (fixed FP{fixed_cnt[0]}/FN{fixed_cnt[1]} · broke FP{broke_cnt[0]}/FN{broke_cnt[1]})')


def _tag_rate(tagged: pd.DataFrame, jp_tag: str) -> float:
    """そのタグを含む行の割合(%)"""
    tags = tagged['tags'].fillna('')
    has = tags.apply(lambda s: jp_tag in s.split(','))
    return has.sum() / len(tagged) * 100


def plot_tags(fixed_tagged: pd.DataFrame, broke_tagged: pd.DataFrame, out_path: str):
    """図2: タグ別 fixed率 vs broke率 グループバー"""
    labels = [en for _, en in TAG_LABELS]
    fixed_pct = [_tag_rate(fixed_tagged, jp) for jp, _ in TAG_LABELS]
    broke_pct = [_tag_rate(broke_tagged, jp) for jp, _ in TAG_LABELS]

    x = range(len(labels))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([i - w / 2 for i in x], fixed_pct, width=w, color=FIXED_COLOR, label='Fixed (n=%d)' % len(fixed_tagged))
    ax.bar([i + w / 2 for i in x], broke_pct, width=w, color=BROKE_COLOR, label='Broke (n=%d)' % len(broke_tagged))

    for i, (f, b) in enumerate(zip(fixed_pct, broke_pct)):
        ax.text(i - w / 2, f + 0.8, f'{f:.0f}', ha='center', fontsize=8, color=FIXED_COLOR)
        ax.text(i + w / 2, b + 0.8, f'{b:.0f}', ha='center', fontsize=8, color=BROKE_COLOR)

    ax.set_ylabel('% of bucket', fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, rotation=20, ha='right')
    ax.set_title('What DAPT fixed vs broke on OOD (by review pattern)', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  ✅ {out_path}')


def main():
    parser = argparse.ArgumentParser(description='DAPT前後の誤分類差分を可視化')
    parser.add_argument('--input-dir', default='data/experiments/ood_benchmark', help='fixed/broke CSVの場所')
    parser.add_argument('--output-dir', default=None, help='PNG出力先（未指定なら--input-dirと同じ）')
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or in_dir
    os.makedirs(out_dir, exist_ok=True)

    fixed = pd.read_csv(os.path.join(in_dir, 'fixed.csv'))
    broke = pd.read_csv(os.path.join(in_dir, 'broke.csv'))
    fixed_tagged = pd.read_csv(os.path.join(in_dir, 'fixed_tagged.csv'))
    broke_tagged = pd.read_csv(os.path.join(in_dir, 'broke_tagged.csv'))

    print('DAPT誤分類差分の可視化')
    plot_error_type(fixed, broke, os.path.join(out_dir, 'dapt_diff_errortype.png'))
    plot_tags(fixed_tagged, broke_tagged, os.path.join(out_dir, 'dapt_diff_tags.png'))
    print('完了')


if __name__ == '__main__':
    main()
