"""
Sentiment analysis evaluation module

This module provides functions to evaluate sentiment analysis models
and compare results across different languages.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_sentiment_model(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Evaluate sentiment analysis model performance

    Args:
        y_true: True labels (0 or 1)
        y_pred: Predicted labels (0 or 1)

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1_score: F1 score
            - confusion_matrix: 2x2 numpy array [[TN, FP], [FN, TP]]

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0]
        >>> results = evaluate_sentiment_model(y_true, y_pred)
        >>> results['accuracy']
        0.8
    """
    if not y_true or not y_pred:
        raise ValueError("y_true and y_pred must not be empty")

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm
    }


def print_evaluation_metrics(results: Dict[str, float], language: str = "モデル") -> None:
    """
    Print evaluation metrics in a formatted way

    Args:
        results: Dictionary from evaluate_sentiment_model()
        language: Language name for display (e.g., "英語", "日本語")

    Example:
        >>> results = {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.95, 'f1_score': 0.92, ...}
        >>> print_evaluation_metrics(results, "英語")
    """
    print(f"\n{'=' * 60}")
    print(f"{language}モデル評価結果")
    print(f"{'=' * 60}")
    print(f"正解率（Accuracy）:  {results['accuracy']:.2%}")
    print(f"適合率（Precision）: {results['precision']:.2%}")
    print(f"再現率（Recall）:    {results['recall']:.2%}")
    print(f"F1スコア:            {results['f1_score']:.2%}")

    # Print confusion matrix
    cm = results['confusion_matrix']
    print(f"\n混同行列（Confusion Matrix）:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"Actual  Neg    [{cm[0][0]:4d}] [{cm[0][1]:4d}]")
    print(f"        Pos    [{cm[1][0]:4d}] [{cm[1][1]:4d}]")


def print_comparison_report(
    results_en: Dict[str, float],
    results_ja: Dict[str, float]
) -> None:
    """
    Print comparison report for English and Japanese models

    Args:
        results_en: Evaluation results for English model
        results_ja: Evaluation results for Japanese model

    Example:
        >>> print_comparison_report(results_en, results_ja)
    """
    print("\n" + "=" * 60)
    print("言語比較レポート")
    print("=" * 60)

    # Table header
    print(f"\n{'評価指標':<15} {'英語':<12} {'日本語':<12} {'差分':<12}")
    print("-" * 60)

    # Metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = {
        'accuracy': '正解率',
        'precision': '適合率',
        'recall': '再現率',
        'f1_score': 'F1スコア'
    }
    for metric in metrics:
        en_val = results_en[metric]
        ja_val = results_ja[metric]
        diff = en_val - ja_val
        diff_str = f"{diff:+.2%}" if diff != 0 else " 0.00%"

        print(f"{metric_names[metric]:<15} {en_val:>10.2%}  {ja_val:>10.2%}  {diff_str:>10}")

    # Interpretation
    print("\n" + "=" * 60)
    print("評価")
    print("=" * 60)

    en_acc = results_en['accuracy']
    ja_acc = results_ja['accuracy']

    # English model assessment
    if en_acc >= 0.90:
        en_status = "✅ 優秀 (≥90%)"
    elif en_acc >= 0.85:
        en_status = "✅ 良好 (≥85%)"
    elif en_acc >= 0.80:
        en_status = "⚠️  許容範囲 (≥80%)"
    else:
        en_status = "❌ 不良 (<80%)"

    # Japanese model assessment
    if ja_acc >= 0.85:
        ja_status = "✅ 優秀 (≥85%)"
    elif ja_acc >= 0.80:
        ja_status = "⚠️  許容範囲 (≥80%)"
    else:
        ja_status = "❌ 不良 (<80%)"

    print(f"英語モデル:   {en_status}")
    print(f"日本語モデル: {ja_status}")

    # Recommendation
    print("\n" + "=" * 60)
    print("推奨事項")
    print("=" * 60)

    if en_acc >= 0.90 and ja_acc >= 0.85:
        print("✅ 両モデルとも良好な性能です。ターゲット市場に基づいて選択してください:")
        print("   - 英語: より広範なデータセット、一般的な洞察")
        print("   - 日本語: 日本市場特有の洞察")
    elif en_acc >= 0.90 and ja_acc < 0.85:
        print("✅ 英語モデルの使用を推奨:")
        print(f"   - 英語の正解率 ({en_acc:.2%}) が大幅に高い")
        print(f"   - 日本語の正解率 ({ja_acc:.2%}) は信頼性が低い可能性")
    elif en_acc < 0.85:
        print("❌ 両モデルの正解率が低いです。以下を調査してください:")
        print("   - データ品質の問題")
        print("   - モデル選択（別のモデルを試す）")
        print("   - 実装のバグ")
    else:
        print("⚠️  判断する前に両モデルを慎重に確認してください")


def calculate_error_rate_by_class(
    y_true: List[int],
    y_pred: List[int]
) -> Dict[str, Tuple[int, int, float]]:
    """
    Calculate error rate for each class (Positive/Negative)

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with keys 'positive' and 'negative', each containing:
            - (correct_count, total_count, accuracy)

    Example:
        >>> y_true = [1, 1, 0, 0]
        >>> y_pred = [1, 0, 0, 0]
        >>> result = calculate_error_rate_by_class(y_true, y_pred)
        >>> result['positive']
        (1, 2, 0.5)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Positive class (label=1)
    pos_mask = y_true == 1
    pos_correct = np.sum((y_true[pos_mask] == y_pred[pos_mask]))
    pos_total = np.sum(pos_mask)
    pos_acc = pos_correct / pos_total if pos_total > 0 else 0.0

    # Negative class (label=0)
    neg_mask = y_true == 0
    neg_correct = np.sum((y_true[neg_mask] == y_pred[neg_mask]))
    neg_total = np.sum(neg_mask)
    neg_acc = neg_correct / neg_total if neg_total > 0 else 0.0

    return {
        'positive': (int(pos_correct), int(pos_total), float(pos_acc)),
        'negative': (int(neg_correct), int(neg_total), float(neg_acc))
    }


def print_detailed_classification_report(
    y_true: List[int],
    y_pred: List[int],
    language: str = "モデル"
) -> None:
    """
    Print detailed classification report with sklearn

    Args:
        y_true: True labels
        y_pred: Predicted labels
        language: Language name for display

    Example:
        >>> print_detailed_classification_report(y_true, y_pred, "英語")
    """
    print(f"\n{'=' * 60}")
    print(f"{language} - 詳細分類レポート")
    print(f"{'=' * 60}")

    # sklearn classification report
    target_names = ['Negative (0)', 'Positive (1)']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # Per-class accuracy
    class_stats = calculate_error_rate_by_class(y_true, y_pred)

    print(f"\nクラス別正解率:")
    for class_name, (correct, total, acc) in class_stats.items():
        print(f"  {class_name.capitalize()}: {correct}/{total} = {acc:.2%}")
