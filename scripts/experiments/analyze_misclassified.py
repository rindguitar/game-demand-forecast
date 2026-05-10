"""
誤分類分析スクリプト

現行の感情分析モデル（models/best_model）でreviews_10000.csv全件を予測し、
正解ラベル（voted_up由来のlabel列）と一致しない事例を抽出する。

将来的なモデル改善のベースライン取得が目的。

出力:
    - data/experiments/sarcasm_baseline/misclassified.csv（最新のみ・上書き）
    - data/experiments/sarcasm_baseline/summary.csv（全モデルの集計・追記）
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.nlp.dataset import SteamReviewDataset
from src.nlp.model import load_model


def predict_all(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str
) -> tuple[list[int], list[float]]:
    """全レビューに対して予測を実行"""
    model.eval()
    predictions = []
    confidences = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="予測中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            batch_confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

            predictions.extend(preds.cpu().tolist())
            confidences.extend(batch_confidences.cpu().tolist())

    return predictions, confidences


def label_to_pn(label: int) -> str:
    """1→P, 0→N に変換"""
    return 'P' if label == 1 else 'N'


def append_summary(summary_path: str, row: dict):
    """summary.csv に1行追記（同タイムスタンプは重複させない）"""
    columns = [
        'model_timestamp', 'dataset', 'seed', 'lr', 'patience', 'dropout',
        'batch_size', 'accuracy', 'fp', 'fn', 'misclassified',
        'high_confidence_errors'
    ]

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        # 同タイムスタンプは上書きせず重複追加防止
        if (df['model_timestamp'] == row['model_timestamp']).any():
            print(f"⚠️  既存のmodel_timestamp（{row['model_timestamp']}）が存在するため追記しません")
            return
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=columns)

    df.to_csv(summary_path, index=False)


def main():
    print("=" * 70)
    print("誤分類分析: reviews_10000.csv vs models/best_model")
    print("=" * 70)

    # パス設定
    input_path = 'data/train/reviews_10000.csv'
    model_dir = 'models/best_model'
    output_dir = 'data/experiments/sarcasm_baseline'
    misclassified_path = os.path.join(output_dir, 'misclassified.csv')
    summary_path = os.path.join(output_dir, 'summary.csv')

    os.makedirs(output_dir, exist_ok=True)

    # 1. モデルメタデータ読み込み（training_results.jsonから）
    results_path = os.path.join(model_dir, 'training_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"training_results.json が見つかりません: {results_path}\n"
            f"先に make train-sentiment で学習結果を生成してください。"
        )

    with open(results_path, 'r', encoding='utf-8') as f:
        training_info = json.load(f)

    print(f"\n[1/4] モデルメタデータ読み込み")
    print(f"  学習タイムスタンプ: {training_info['timestamp']}")
    print(f"  ハイパーパラメータ: {training_info['hyperparameters']}")

    # 2. データ読み込み
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['review_text']).reset_index(drop=True)
    print(f"\n[2/4] データ読み込み: {len(df)}件")

    # 3. モデル読み込み・予測
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[3/4] モデル読み込み・予測 (device={device})")
    model, tokenizer = load_model(model_dir, device=device)

    dataset = SteamReviewDataset(
        texts=df['review_text'].tolist(),
        labels=df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=128
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions, confidences = predict_all(model, dataloader, device)

    # 4. 誤分類抽出と保存
    print(f"\n[4/4] 誤分類抽出と保存")
    df['predicted_label'] = predictions
    df['confidence'] = confidences
    df['is_correct'] = df['label'] == df['predicted_label']

    total = len(df)
    correct = int(df['is_correct'].sum())
    misclassified_count = total - correct
    accuracy = correct / total

    fp = int(((df['label'] == 0) & (df['predicted_label'] == 1)).sum())
    fn = int(((df['label'] == 1) & (df['predicted_label'] == 0)).sum())

    print(f"\n【全体統計】")
    print(f"  総件数:     {total}")
    print(f"  正解:       {correct} ({accuracy:.2%})")
    print(f"  誤分類:     {misclassified_count} ({misclassified_count / total:.2%})")
    print(f"  FP (NをPと誤認): {fp} ({fp / total:.2%})")
    print(f"  FN (PをNと誤認): {fn} ({fn / total:.2%})")

    # 誤分類サンプルのCSV出力（P/N・FP/FN形式で軽量化）
    misclassified_df = df[~df['is_correct']].copy()
    misclassified_df['actual'] = misclassified_df['label'].apply(label_to_pn)
    misclassified_df['predicted'] = misclassified_df['predicted_label'].apply(label_to_pn)
    misclassified_df['error_type'] = misclassified_df.apply(
        lambda r: 'FP' if r['label'] == 0 else 'FN',
        axis=1
    )

    save_cols = ['review_text', 'actual', 'predicted', 'confidence', 'error_type']
    if 'game_name' in misclassified_df.columns:
        save_cols.insert(0, 'game_name')

    misclassified_df[save_cols].to_csv(misclassified_path, index=False)
    print(f"\n✅ 誤分類サンプル保存: {misclassified_path}")

    # 信頼度の分布
    high_conf_errors = int((misclassified_df['confidence'] >= 0.9).sum())
    print(f"\n【誤分類の信頼度分布】")
    print(f"  平均confidence: {misclassified_df['confidence'].mean():.3f}")
    print(f"  高信頼誤分類 (>= 0.9): {high_conf_errors}件")

    # summary.csv に追記
    summary_row = {
        'model_timestamp': training_info['timestamp'],
        'dataset': training_info['dataset_path'],
        'seed': training_info['hyperparameters']['random_seed'],
        'lr': training_info['hyperparameters']['learning_rate'],
        'patience': training_info['hyperparameters']['patience'],
        'dropout': 0.3,  # model.pyのデフォルト
        'batch_size': training_info['hyperparameters']['batch_size'],
        'accuracy': round(accuracy, 4),
        'fp': fp,
        'fn': fn,
        'misclassified': misclassified_count,
        'high_confidence_errors': high_conf_errors,
    }
    append_summary(summary_path, summary_row)
    print(f"✅ サマリー追記: {summary_path}")

    print("\n" + "=" * 70)
    print("分析完了")
    print("=" * 70)


if __name__ == '__main__':
    main()
