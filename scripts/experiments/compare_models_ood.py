"""
モデルのOOD性能比較（複数対応・--modelsで増減可）

OODテストセットで各モデルを評価し、accuracy/precision/recall/F1/混同行列を比較、
ペアごとにMcNemar検定する。使い方は --help、設計は scripts/README.md を参照。
"""

import os
import sys
import json
import argparse
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from src.nlp.dataset import SteamReviewDataset
from src.nlp.model import load_model
from src.nlp.evaluation import evaluate_sentiment_model, print_evaluation_metrics

TEST_CSV = 'data/test/reviews_ood_2000.csv'
OUT_DIR = 'data/experiments/ood_benchmark'
MAX_LENGTH = 128

# デフォルト比較対象：本プロジェクトの本番モデル(DAPT後) vs 汎用sst-2
DEFAULT_MODELS = [
    'self:models/best_model:Self(DAPT)',
    'hf:distilbert-base-uncased-finetuned-sst-2-english:SST-2',
]


def parse_model_spec(spec: str) -> tuple:
    """"kind:path[:label]" をパース（kind=self|hf）"""
    parts = spec.split(':', 2)
    if len(parts) < 2 or parts[0] not in ('self', 'hf'):
        raise ValueError(f"不正なモデル指定: {spec}（'self:path[:label]' / 'hf:name[:label]'）")
    kind, path = parts[0], parts[1]
    label = parts[2] if len(parts) == 3 else os.path.basename(path.rstrip('/'))
    return kind, path, label


def predict_self(model_path: str, df: pd.DataFrame, device: str) -> list:
    """自作SentimentClassifierで予測（argmaxが 1=positive/0=negative）"""
    model, tokenizer = load_model(model_path, device=device)
    dataset = SteamReviewDataset(df['review_text'].tolist(), df['sentiment'].tolist(),
                                 tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  予測中(self)'):
            logits = model(input_ids=batch['input_ids'].to(device),
                           attention_mask=batch['attention_mask'].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    return preds


def predict_hf(model_name: str, df: pd.DataFrame, device: str) -> list:
    """HuggingFace pipelineで予測"""
    from transformers import pipeline
    pipe = pipeline('sentiment-analysis', model=model_name, device=0 if device == 'cuda' else -1)

    # ラベル名は機種ごとに違うので、ポジ/ネガ文で正例ラベルを動的判定してから1/0に変換
    pos_label = pipe("I absolutely love this game, it is amazing and fantastic!")[0]['label']
    neg_label = pipe("This game is terrible, awful, broken garbage. I hate it.")[0]['label']
    print(f"    [ラベル判定] positive文→'{pos_label}' / negative文→'{neg_label}'")
    if pos_label == neg_label:
        raise ValueError(f"ポジ/ネガ文が同ラベル('{pos_label}')→2値sentimentでない可能性。手動確認を")

    results = pipe(df['review_text'].tolist(), batch_size=64, truncation=True, max_length=512)
    return [1 if r['label'] == pos_label else 0 for r in results]


def mcnemar_test(y_true: list, pred_a: list, pred_b: list) -> dict:
    """McNemar検定：食い違いサンプルからaccuracy差が有意かを検定"""
    from scipy.stats import binomtest
    ca = [p == t for p, t in zip(pred_a, y_true)]
    cb = [p == t for p, t in zip(pred_b, y_true)]
    only_a = sum(1 for x, y in zip(ca, cb) if x and not y)
    only_b = sum(1 for x, y in zip(ca, cb) if y and not x)
    n = only_a + only_b
    p_exact = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n > 0 else 1.0
    return {'only_a_correct': only_a, 'only_b_correct': only_b, 'discordant': n,
            'p_exact': p_exact, 'significant_at_0.05': p_exact < 0.05}


def plot_metric_comparison(results: list, out_path: str) -> None:
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    n = len(results)
    x = np.arange(len(metrics))
    width = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(2.5 * len(metrics) + n, 6))
    for i, r in enumerate(results):
        vals = [r['metrics'][k] * 100 for k in metrics]
        bars = ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=r['label'], color=colors[i])
        for b in bars:
            ax.annotate(f'{b.get_height():.1f}', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model comparison on OOD test set (2000 reviews / 20 unseen games)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, ncol=min(n, 3))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")


def plot_confusion_matrices(results: list, out_path: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]
    ticks = ['Neg', 'Pos']
    for ax, r in zip(axes, results):
        cm = np.array(r['metrics']['confusion_matrix'])
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=15,
                        fontweight='bold', color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(ticks, fontsize=10)
        ax.set_yticks([0, 1]); ax.set_yticklabels(ticks, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f"{r['label']}  (Acc {r['metrics']['accuracy']*100:.1f}%)",
                     fontsize=11, fontweight='bold')
    plt.suptitle('Confusion Matrix on OOD Test Set', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='モデルのOOD性能比較（複数対応）')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='比較対象 "kind:path[:label]"（kind=self|hf）を空白区切りで')
    parser.add_argument('--test-csv', type=str, default=TEST_CSV)
    parser.add_argument('--output', type=str, default=OUT_DIR)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print(f"モデルのOOD性能比較  (device={device})")
    print("=" * 60)

    df = pd.read_csv(args.test_csv).dropna(subset=['review_text']).reset_index(drop=True)
    y_true = df['sentiment'].tolist()
    print(f"  {args.test_csv}: {len(df)}件 (Pos {sum(y_true)} / Neg {len(y_true) - sum(y_true)})")
    print(f"  比較対象: {len(args.models)}モデル")

    results = []
    for spec in args.models:
        kind, path, label = parse_model_spec(spec)
        print(f"\n[{label}] kind={kind} path={path}")
        preds = predict_self(path, df, device) if kind == 'self' else predict_hf(path, df, device)
        metrics = evaluate_sentiment_model(y_true, preds)
        print_evaluation_metrics(metrics, label)
        results.append({'label': label, 'kind': kind, 'path': path, 'preds': preds, 'metrics': metrics})

    print("\n" + "=" * 60)
    print("McNemar検定（ペアごと・accuracy差が有意か）")
    print("=" * 60)
    mcnemar_results = []
    for a, b in combinations(results, 2):
        mc = mcnemar_test(y_true, a['preds'], b['preds'])
        verdict = '有意' if mc['significant_at_0.05'] else '誤差圏'
        print(f"  {a['label']} vs {b['label']}: 食い違い{mc['discordant']} "
              f"({a['label']}のみ{mc['only_a_correct']}/{b['label']}のみ{mc['only_b_correct']}) "
              f"p={mc['p_exact']:.4f} → {verdict}")
        mcnemar_results.append({'pair': [a['label'], b['label']], **mc})

    print("\n[可視化・保存]")
    os.makedirs(args.output, exist_ok=True)
    plot_metric_comparison(results, os.path.join(args.output, 'metrics_comparison.png'))
    plot_confusion_matrices(results, os.path.join(args.output, 'confusion_matrices.png'))

    def serialize(m):
        out = {k: v for k, v in m.items() if k != 'confusion_matrix'}
        out['confusion_matrix'] = np.array(m['confusion_matrix']).tolist()
        return out

    with open(os.path.join(args.output, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'n_samples': len(df),
                   'models': [{'label': r['label'], 'kind': r['kind'], 'path': r['path'],
                               'metrics': serialize(r['metrics'])} for r in results],
                   'mcnemar': mcnemar_results}, f, ensure_ascii=False, indent=2)
    print(f"  保存: {os.path.join(args.output, 'metrics.json')}")

    print("\n" + "=" * 60)
    print("Accuracy サマリ")
    print("=" * 60)
    for r in sorted(results, key=lambda r: -r['metrics']['accuracy']):
        print(f"  {r['label']:20} {r['metrics']['accuracy']*100:.1f}%")


if __name__ == '__main__':
    main()
