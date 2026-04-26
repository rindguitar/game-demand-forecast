"""
感情分析modelの学習実行スクリプト

DistilBERTをSteamレビューでファインチューニングする。
"""

import sys
import os
import argparse

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer

from src.nlp.dataset import load_datasets_from_csv, create_dataloaders
from src.nlp.model import SentimentClassifier, save_model
from src.nlp.train import train_model, evaluate
from src.nlp.evaluation import evaluate_sentiment_model, print_evaluation_metrics


def main():
    """学習メイン関数"""

    # コマンドライン引数
    parser = argparse.ArgumentParser(description='DistilBERT感情分析モデル学習')
    parser.add_argument('--dataset-size', type=int, default=1000, choices=[1000, 5000],
                        help='Dataset size (1000 or 5000)')
    args = parser.parse_args()

    dataset_size = args.dataset_size
    train_size = int(dataset_size * 0.7)
    val_size = int(dataset_size * 0.15)
    test_size = int(dataset_size * 0.15)

    print("=" * 60)
    print(f"DistilBERT 感情分析モデル学習 (Dataset: {dataset_size}件)")
    print("=" * 60)

    # デバイス確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✅ Device: {device}")

    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. データセット読み込み
    print("\n" + "=" * 60)
    print("1. データセット読み込み")
    print("=" * 60)

    train_df, val_df, test_df = load_datasets_from_csv(
        f'data/train/train_{train_size}.csv',
        f'data/train/val_{val_size}.csv',
        f'data/train/test_{test_size}.csv'
    )

    print(f"✅ Train: {len(train_df)} reviews")
    print(f"✅ Val: {len(val_df)} reviews")
    print(f"✅ Test: {len(test_df)} reviews")

    # 2. Tokenizer & DataLoader
    print("\n" + "=" * 60)
    print("2. Tokenizer & DataLoader作成")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    print("✅ Tokenizer loaded")

    batch_size = 64  # ベンチマーク結果からbatch_size=64が最適
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size
    )

    print(f"✅ DataLoader created (batch_size={batch_size})")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # 3. モデル初期化
    print("\n" + "=" * 60)
    print("3. モデル初期化")
    print("=" * 60)

    model = SentimentClassifier()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✅ Model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 4. 学習実行
    print("\n" + "=" * 60)
    print("4. 学習実行")
    print("=" * 60)

    trained_model, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,
        lr=2e-5,
        device=device,
        patience=2
    )

    # 5. Train/Val/Test評価
    print("\n" + "=" * 60)
    print("5. 全データセットで評価")
    print("=" * 60)

    # Train評価
    print("\n【Train評価】")
    train_predictions, train_labels = evaluate(trained_model, train_loader, device)
    train_results = evaluate_sentiment_model(train_labels, train_predictions)
    print_evaluation_metrics(train_results, "Train")

    # Val評価
    print("\n【Validation評価】")
    val_predictions, val_labels = evaluate(trained_model, val_loader, device)
    val_results = evaluate_sentiment_model(val_labels, val_predictions)
    print_evaluation_metrics(val_results, "Validation")

    # Test評価（最終評価）
    print("\n【Test評価】")
    test_predictions, test_labels = evaluate(trained_model, test_loader, device)
    test_results = evaluate_sentiment_model(test_labels, test_predictions)
    print_evaluation_metrics(test_results, "Test")

    # 6. パフォーマンスサマリー
    print("\n" + "=" * 60)
    print("6. パフォーマンスサマリー")
    print("=" * 60)

    print(f"{'Dataset':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 60)
    print(f"{'Train':<12} {train_results['accuracy']:>10.2%}  {train_results['precision']:>10.2%}  {train_results['recall']:>10.2%}  {train_results['f1_score']:>10.2%}")
    print(f"{'Validation':<12} {val_results['accuracy']:>10.2%}  {val_results['precision']:>10.2%}  {val_results['recall']:>10.2%}  {val_results['f1_score']:>10.2%}")
    print(f"{'Test':<12} {test_results['accuracy']:>10.2%}  {test_results['precision']:>10.2%}  {test_results['recall']:>10.2%}  {test_results['f1_score']:>10.2%}")

    # 7. モデル保存
    print("\n" + "=" * 60)
    print("7. モデル保存")
    print("=" * 60)

    save_model(trained_model, tokenizer, save_path='models/sentiment_model')

    # 8. 成功判定
    print("\n" + "=" * 60)
    print("8. 成功判定")
    print("=" * 60)

    target_accuracy = 0.85

    if test_results['accuracy'] >= target_accuracy:
        print(f"✅ 目標達成！Test Accuracy {test_results['accuracy']:.2%} ≥ {target_accuracy:.0%}")
        print(f"\n🎉 Issue #6 Phase 1 完了！")
        return 0
    else:
        print(f"⚠️  目標未達: Test Accuracy {test_results['accuracy']:.2%} < {target_accuracy:.0%}")
        print(f"\n💡 Phase 2（5000件dataset）の検討が必要です")
        return 1


if __name__ == '__main__':
    exit(main())
