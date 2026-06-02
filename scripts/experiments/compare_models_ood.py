"""
自作モデル vs 既製汎用モデル(sst-2) のOOD性能比較 (Issue #21)

OODテストセット（学習に使っていない20ゲーム・2000件balanced）で両モデルを評価し、
accuracy / precision / recall / F1 / 混同行列 をグラフで比較する。
「Steamデータでのファインチューニングに意味があったか」を未知データで検証するのが目的。

ラベル整合性（重要）:
  ラベルの対応(0/1)はベースのアーキテクチャではなく学習データのラベル付けで決まるため、
  2モデルで一致する保証はない。整数IDで揃えず「意味ラベル」で揃える:
  - 自作モデル: 我々が 1=positive / 0=negative で学習 → argmax をそのまま使用
  - sst-2: pipelineが "POSITIVE"/"NEGATIVE" を文字列で返す → POSITIVE=1 / NEGATIVE=0 に変換
  さらに、明確にポジティブな文1つで sst-2 のラベル意味を動作確認してから本番に回す。

出力（data/experiments/ood_benchmark/）:
  - metrics_comparison.png : accuracy/precision/recall/F1 のグループ棒グラフ
  - confusion_matrices.png : 両モデルの混同行列ヒートマップ
  - metrics.json           : 数値結果
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # GUI不要のバックエンド
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.nlp.dataset import SteamReviewDataset
from src.nlp.model import load_model
from src.nlp.evaluation import evaluate_sentiment_model, print_evaluation_metrics

# --- 設定 ---
SELF_MODEL_DIR = 'models/best_model'                              # 自作モデル
SST2_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'    # 既製汎用モデル
TEST_CSV = 'data/test/reviews_ood_2000.csv'                       # OODテストセット
OUT_DIR = 'data/experiments/ood_benchmark'
MAX_LENGTH = 128  # 自作モデルの学習時と同じ


def predict_self_trained(df: pd.DataFrame, device: str) -> list:
    """自作モデルで予測（argmaxの結果が既に 1=positive/0=negative）"""
    model, tokenizer = load_model(SELF_MODEL_DIR, device=device)

    dataset = SteamReviewDataset(
        texts=df['review_text'].tolist(),
        labels=df['sentiment'].tolist(),  # 予測には使わないが引数上必要
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="自作モデル予測中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds


def predict_sst2(df: pd.DataFrame, device: str) -> list:
    """sst-2で予測。意味ラベル(POSITIVE/NEGATIVE)で揃え、事前に動作確認する"""
    from transformers import pipeline

    device_id = 0 if device == 'cuda' else -1
    pipe = pipeline('sentiment-analysis', model=SST2_MODEL, device=device_id)

    # 動作確認（裏取り）: 明らかにポジティブな文が POSITIVE になるか
    sanity = pipe("This game is absolutely amazing, I love every minute of it!")[0]
    print(f"\n[sst-2 動作確認] 明確にポジティブな文 → {sanity}")
    assert sanity['label'] == 'POSITIVE', \
        f"sst-2のラベル意味が想定と異なる: {sanity}（ラベル変換を見直すこと）"
    print("  → POSITIVE と判定。ラベル意味は想定通り。\n")

    texts = df['review_text'].tolist()
    results = pipe(texts, batch_size=64, truncation=True, max_length=512)
    # 意味ラベルで変換: POSITIVE=1 / NEGATIVE=0
    return [1 if r['label'] == 'POSITIVE' else 0 for r in results]


def mcnemar_test(y_true: list, pred_a: list, pred_b: list) -> dict:
    """
    McNemar検定: 同一テストセットで2モデルの正誤が食い違うサンプル(b,c)だけに注目し、
    accuracyの差が統計的に有意か（＝誤差で説明できないか）を検定する。

    同じデータを両モデルが見ている（対応のある比較）ため、独立2標本の検定ではなく
    McNemar検定が正しい道具。判定が一致したサンプルは情報を持たないので無視する。

    Args:
        y_true: 正解ラベル
        pred_a: モデルAの予測（自作）
        pred_b: モデルBの予測（sst-2）

    Returns:
        food違いの内訳と p値（連続補正カイ二乗・正確二項）を含むdict
    """
    from scipy.stats import chi2 as chi2_dist, binomtest

    correct_a = [p == t for p, t in zip(pred_a, y_true)]
    correct_b = [p == t for p, t in zip(pred_b, y_true)]

    both_correct = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and cb)
    both_wrong = sum(1 for ca, cb in zip(correct_a, correct_b) if not ca and not cb)
    only_a = sum(1 for ca, cb in zip(correct_a, correct_b) if ca and not cb)  # Aだけ正解
    only_b = sum(1 for ca, cb in zip(correct_a, correct_b) if cb and not ca)  # Bだけ正解

    n = only_a + only_b  # 食い違い総数（discordant pairs）
    if n == 0:
        chi2_stat, p_chi2, p_exact = 0.0, 1.0, 1.0
    else:
        # 連続補正つきカイ二乗（df=1）
        chi2_stat = (abs(only_a - only_b) - 1) ** 2 / n
        p_chi2 = float(chi2_dist.sf(chi2_stat, df=1))
        # 正確二項検定（小サンプルでも妥当）: H0のもと only_a ~ Binomial(n, 0.5)
        p_exact = float(binomtest(min(only_a, only_b), n, 0.5).pvalue)

    return {
        'both_correct': both_correct,
        'both_wrong': both_wrong,
        'only_self_correct': only_a,
        'only_sst2_correct': only_b,
        'discordant': n,
        'chi2': float(chi2_stat),
        'p_chi2': p_chi2,
        'p_exact': p_exact,
        'significant_at_0.05': p_exact < 0.05,
    }


def plot_metric_comparison(m_self: dict, m_sst2: dict, mcnemar: dict, out_path: str) -> None:
    """accuracy/precision/recall/F1 のグループ棒グラフ"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    self_vals = [m_self[k] * 100 for k in metrics]
    sst2_vals = [m_sst2[k] * 100 for k in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, self_vals, width, label='Self-trained (Steam)', color='#2E86DE')
    bars2 = ax.bar(x + width / 2, sst2_vals, width, label='SST-2 (generic)', color='#E67E22')

    # 棒の上に数値ラベル
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Self-trained vs SST-2 on OOD Test Set (2000 reviews / 20 unseen games)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # McNemar検定の結論をグラフ上に明記（「差は意味があったか」を一目で）
    sig = mcnemar['significant_at_0.05']
    verdict = 'significant' if sig else 'NOT significant (within noise)'
    acc_diff = (m_self['accuracy'] - m_sst2['accuracy']) * 100
    note = (f"Accuracy diff: {acc_diff:+.1f}pt   |   "
            f"McNemar test: p = {mcnemar['p_exact']:.3f}  → {verdict}")
    ax.text(0.5, 0.97, note, transform=ax.transAxes, ha='center', va='top',
            fontsize=11, fontweight='bold',
            color=('#27AE60' if sig else '#C0392B'),
            bbox=dict(boxstyle='round', facecolor='#F4F6F7', edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")


def plot_confusion_matrices(m_self: dict, m_sst2: dict, out_path: str) -> None:
    """両モデルの混同行列ヒートマップ（[[TN,FP],[FN,TP]]）"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tick_labels = ['Neg', 'Pos']

    for ax, m, title, cmap in (
        (axes[0], m_self, 'Self-trained (Steam)', 'Blues'),
        (axes[1], m_sst2, 'SST-2 (generic)', 'Oranges'),
    ):
        cm = np.array(m['confusion_matrix'])
        im = ax.imshow(cm, cmap=cmap)
        # セルに件数を表示
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=11)
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f"{title}  (Acc {m['accuracy']*100:.1f}%)", fontsize=12, fontweight='bold')

    plt.suptitle('Confusion Matrix on OOD Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print("自作モデル vs sst-2 OOD性能比較 (Issue #21)")
    print("=" * 60)
    print(f"  device: {device}")
    print(f"  テストセット: {TEST_CSV}")

    # 1. データ読み込み
    df = pd.read_csv(TEST_CSV).dropna(subset=['review_text']).reset_index(drop=True)
    y_true = df['sentiment'].tolist()
    print(f"  件数: {len(df)} (Positive {sum(y_true)} / Negative {len(y_true) - sum(y_true)})")

    # 2. 両モデルで予測
    print("\n[1/3] 自作モデルで予測")
    y_self = predict_self_trained(df, device)
    print("\n[2/3] sst-2で予測")
    y_sst2 = predict_sst2(df, device)

    # 3. 評価
    print("\n[3/3] 評価と可視化")
    m_self = evaluate_sentiment_model(y_true, y_self)
    m_sst2 = evaluate_sentiment_model(y_true, y_sst2)

    print_evaluation_metrics(m_self, "自作モデル (Steam fine-tuned)")
    print_evaluation_metrics(m_sst2, "sst-2 (generic)")

    # McNemar検定: accuracyの差が誤差で説明できるか
    mc = mcnemar_test(y_true, y_self, y_sst2)
    print("\n" + "=" * 60)
    print("McNemar検定（自作 vs sst-2 のaccuracy差は有意か）")
    print("=" * 60)
    print(f"  両方正解: {mc['both_correct']} / 両方不正解: {mc['both_wrong']}")
    print(f"  自作だけ正解: {mc['only_self_correct']} / sst-2だけ正解: {mc['only_sst2_correct']}"
          f"  (食い違い計 {mc['discordant']})")
    print(f"  p値（正確二項検定）: {mc['p_exact']:.4f}")
    if mc['significant_at_0.05']:
        print("  → p < 0.05: 差は統計的に有意（ファインチューニングの効果と言える）")
    else:
        print("  → p ≥ 0.05: 差は有意でない＝誤差圏（このテストでは効果を主張できない）")

    # 4. 可視化・保存
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_metric_comparison(m_self, m_sst2, mc, os.path.join(OUT_DIR, 'metrics_comparison.png'))
    plot_confusion_matrices(m_self, m_sst2, os.path.join(OUT_DIR, 'confusion_matrices.png'))

    # 数値をJSONで保存（confusion_matrixはlistに変換）
    def to_serializable(m):
        out = {k: v for k, v in m.items() if k != 'confusion_matrix'}
        out['confusion_matrix'] = np.array(m['confusion_matrix']).tolist()
        return out

    metrics_path = os.path.join(OUT_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_samples': len(df),
            'self_trained': to_serializable(m_self),
            'sst2': to_serializable(m_sst2),
            'mcnemar': mc,
        }, f, ensure_ascii=False, indent=2)
    print(f"  保存: {metrics_path}")

    # 5. 一言サマリ
    diff = (m_self['accuracy'] - m_sst2['accuracy']) * 100
    print("\n" + "=" * 60)
    print(f"Accuracy: 自作 {m_self['accuracy']*100:.1f}% vs sst-2 {m_sst2['accuracy']*100:.1f}% "
          f"(差 {diff:+.1f}pt)")
    print("=" * 60)


if __name__ == '__main__':
    main()
