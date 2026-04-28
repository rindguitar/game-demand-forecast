"""
単一トライアルの学習実行スクリプト

Learning Curve実験用に、1つのデータセットで1回の学習を実行し、結果を返す。
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from src.nlp.dataset import create_dataloaders
from src.nlp.model import SentimentClassifier, save_model
from src.nlp.train import train_model, evaluate
from src.nlp.evaluation import evaluate_sentiment_model


def train_single_trial(
    dataset_path: str,
    output_dir: str,
    random_seed: int = 42,
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    patience: int = 2,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    verbose: bool = True
) -> dict:
    """
    単一トライアルの学習を実行

    Args:
        dataset_path: データセットCSVファイルのパス
        output_dir: モデル保存先ディレクトリ
        random_seed: ランダムシード
        batch_size: バッチサイズ
        epochs: エポック数
        learning_rate: 学習率
        patience: Early Stoppingの忍耐値
        train_ratio: Train setの割合
        val_ratio: Validation setの割合
        verbose: 詳細ログ出力

    Returns:
        学習結果のdict（train_acc, val_acc, test_acc, best_epoch等）
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"学習実行: {dataset_path}")
        print(f"Random Seed: {random_seed}")
        print(f"{'=' * 70}")

    # デバイス確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"\n✅ Device: {device}")
        if device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # 1. データセット読み込み
    if verbose:
        print(f"\n1. データセット読み込み")

    df = pd.read_csv(dataset_path)

    # NaN除去
    df = df.dropna(subset=['review_text'])

    if verbose:
        print(f"   Total: {len(df)} reviews")

    # Train/Val/Test分割
    test_ratio = 1.0 - train_ratio - val_ratio

    # Train/Temp分割
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=df['label']
    )

    # Val/Test分割
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_seed,
        stratify=temp_df['label']
    )

    if verbose:
        print(f"   Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
        print(f"   Val:   {len(val_df)} ({len(val_df)/len(df):.1%})")
        print(f"   Test:  {len(test_df)} ({len(test_df)/len(df):.1%})")

    # 2. Tokenizer & DataLoader
    if verbose:
        print(f"\n2. Tokenizer & DataLoader作成")

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, tokenizer, batch_size=batch_size
    )

    if verbose:
        print(f"   Batch size: {batch_size}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

    # 3. モデル初期化
    if verbose:
        print(f"\n3. モデル初期化")

    model = SentimentClassifier()
    model.to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

    # 4. 学習実行
    if verbose:
        print(f"\n4. 学習実行")
        print(f"   Epochs: {epochs}, LR: {learning_rate}, Patience: {patience}")

    trained_model, best_epoch = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=learning_rate,
        device=device,
        patience=patience,
        test_loader=test_loader  # Test精度も記録
    )

    # 5. Train/Val/Test評価
    if verbose:
        print(f"\n5. 評価")

    # Train評価
    train_predictions, train_labels = evaluate(trained_model, train_loader, device)
    train_results = evaluate_sentiment_model(train_labels, train_predictions)

    # Val評価
    val_predictions, val_labels = evaluate(trained_model, val_loader, device)
    val_results = evaluate_sentiment_model(val_labels, val_predictions)

    # Test評価
    test_predictions, test_labels = evaluate(trained_model, test_loader, device)
    test_results = evaluate_sentiment_model(test_labels, test_predictions)

    if verbose:
        print(f"   Train Acc: {train_results['accuracy']:.2%}")
        print(f"   Val Acc:   {val_results['accuracy']:.2%}")
        print(f"   Test Acc:  {test_results['accuracy']:.2%}")

    # 6. モデル保存
    os.makedirs(output_dir, exist_ok=True)
    save_model(trained_model, tokenizer, save_path=output_dir)

    if verbose:
        print(f"\n✅ モデル保存: {output_dir}")

    # 結果を返す
    return {
        'train_acc': train_results['accuracy'] * 100,  # パーセント表記
        'val_acc': val_results['accuracy'] * 100,
        'test_acc': test_results['accuracy'] * 100,
        'train_f1': train_results['f1_score'] * 100,
        'val_f1': val_results['f1_score'] * 100,
        'test_f1': test_results['f1_score'] * 100,
        'best_epoch': best_epoch,
        'random_seed': random_seed,
        'dataset_size': len(df)
    }


def main():
    """テスト実行"""
    import argparse

    parser = argparse.ArgumentParser(description='単一トライアル学習')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset CSV path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--patience', type=int, default=2, help='Early stopping patience')

    args = parser.parse_args()

    results = train_single_trial(
        dataset_path=args.dataset,
        output_dir=args.output,
        random_seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience
    )

    print("\n" + "=" * 70)
    print("学習完了")
    print("=" * 70)
    print(f"Train Acc: {results['train_acc']:.2f}%")
    print(f"Val Acc:   {results['val_acc']:.2f}%")
    print(f"Test Acc:  {results['test_acc']:.2f}%")
    print(f"Best Epoch: {results['best_epoch']}")


if __name__ == '__main__':
    main()
