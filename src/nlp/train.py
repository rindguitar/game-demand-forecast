"""
感情分析modelの学習モジュール

PyTorchでDistilBERTをファインチューニングする機能を提供。
Early Stopping、学習ログ、ベストmodel保存に対応。
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Tuple, List
import time


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = 'cuda'
) -> float:
    """
    1エポックの学習

    Args:
        model: 学習するmodel
        dataloader: Train DataLoader
        optimizer: optimizer
        device: 'cuda'または'cpu'

    Returns:
        平均loss

    Example:
        >>> loss = train_epoch(model, train_loader, optimizer, device='cuda')
        >>> print(f"Train Loss: {loss:.4f}")
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[List[int], List[int]]:
    """
    評価（推論のみ）

    Args:
        model: 評価するmodel
        dataloader: Val/Test DataLoader
        device: 'cuda'または'cpu'

    Returns:
        (predictions, true_labels)のタプル

    Example:
        >>> predictions, true_labels = evaluate(model, val_loader, device='cuda')
        >>> accuracy = (np.array(predictions) == np.array(true_labels)).mean()
        >>> print(f"Accuracy: {accuracy:.2%}")
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 2e-5,
    device: str = 'cuda',
    patience: int = 2,
    model_save_path: str = 'models/sentiment_model/best_model.pth',
    test_loader: DataLoader = None
) -> Tuple[nn.Module, int]:
    """
    学習メイン関数（Early Stopping対応）

    Args:
        model: 学習するmodel
        train_loader: Train DataLoader
        val_loader: Validation DataLoader
        epochs: 最大エポック数（デフォルト5）
        lr: learning rate（デフォルト2e-5）
        device: 'cuda'または'cpu'
        patience: Early Stopping patience（デフォルト2）
        model_save_path: ベストmodelの保存パス
        test_loader: Test DataLoader（省略可。指定時は各エポックでTest精度も記録）

    Returns:
        (学習済みmodel（ベストmodelの重みを読み込み済み）, best_epoch)

    Example:
        >>> model = SentimentClassifier()
        >>> model.to('cuda')
        >>> trained_model, best_epoch = train_model(
        ...     model, train_loader, val_loader,
        ...     epochs=5, lr=2e-5, device='cuda', patience=2,
        ...     test_loader=test_loader  # Test精度も記録
        ... )
    """
    import os
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=lr)

    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []

    print("=" * 60)
    print("学習開始")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Patience: {patience}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_predictions, val_labels = evaluate(model, val_loader, device)
        val_accuracy = (np.array(val_predictions) == np.array(val_labels)).mean()
        print(f"Val Accuracy: {val_accuracy:.2%}")

        # Test評価（test_loaderが指定されている場合）
        test_accuracy = None
        if test_loader is not None:
            test_predictions, test_labels = evaluate(model, test_loader, device)
            test_accuracy = (np.array(test_predictions) == np.array(test_labels)).mean()
            print(f"Test Accuracy: {test_accuracy:.2%}")

        epoch_time = time.time() - start_time
        print(f"Time: {epoch_time:.1f}s")

        # 履歴保存
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'time': epoch_time
        }
        if test_accuracy is not None:
            history_entry['test_accuracy'] = test_accuracy
        training_history.append(history_entry)

        # Early Stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            # ベストmodelを保存
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved! (Val Acc: {val_accuracy:.2%})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered!")
                break

    # ベストmodelを読み込み
    model.load_state_dict(torch.load(model_save_path))
    print(f"\n✅ Best model loaded (Val Acc: {best_val_accuracy:.2%})")

    # 学習履歴サマリー
    print("\n" + "=" * 60)
    print("学習完了サマリー")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_accuracy:.2%} (Epoch {best_epoch})")
    print(f"Total Epochs: {len(training_history)}")
    total_time = sum(h['time'] for h in training_history)
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.2f}min)")

    # Test平均精度を表示（test_loaderが指定されている場合）
    if test_loader is not None:
        test_accuracies = [h['test_accuracy'] for h in training_history if 'test_accuracy' in h]
        if test_accuracies:
            avg_test_accuracy = np.mean(test_accuracies)
            print(f"Average Test Accuracy (全{len(test_accuracies)}エポック): {avg_test_accuracy:.2%}")

    print("=" * 60)

    return model, best_epoch
