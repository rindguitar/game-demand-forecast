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


def print_evaluation_metrics(results: Dict[str, float], language: str = "Model") -> None:
    """
    Print evaluation metrics in a formatted way

    Args:
        results: Dictionary from evaluate_sentiment_model()
        language: Language name for display (e.g., "English", "Japanese")

    Example:
        >>> results = {'accuracy': 0.92, 'precision': 0.90, 'recall': 0.95, 'f1_score': 0.92, ...}
        >>> print_evaluation_metrics(results, "English")
    """
    print(f"\n{'=' * 60}")
    print(f"{language} Model Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {results['accuracy']:.2%}")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall:    {results['recall']:.2%}")
    print(f"F1 Score:  {results['f1_score']:.2%}")

    # Print confusion matrix
    cm = results['confusion_matrix']
    print(f"\nConfusion Matrix:")
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
    print("Language Comparison Report")
    print("=" * 60)

    # Table header
    print(f"\n{'Metric':<15} {'English':<12} {'Japanese':<12} {'Difference':<12}")
    print("-" * 60)

    # Metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        en_val = results_en[metric]
        ja_val = results_ja[metric]
        diff = en_val - ja_val
        diff_str = f"{diff:+.2%}" if diff != 0 else " 0.00%"

        print(f"{metric.capitalize():<15} {en_val:>10.2%}  {ja_val:>10.2%}  {diff_str:>10}")

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    en_acc = results_en['accuracy']
    ja_acc = results_ja['accuracy']

    # English model assessment
    if en_acc >= 0.90:
        en_status = "✅ Excellent (≥90%)"
    elif en_acc >= 0.85:
        en_status = "✅ Good (≥85%)"
    elif en_acc >= 0.80:
        en_status = "⚠️  Acceptable (≥80%)"
    else:
        en_status = "❌ Poor (<80%)"

    # Japanese model assessment
    if ja_acc >= 0.85:
        ja_status = "✅ Excellent (≥85%)"
    elif ja_acc >= 0.80:
        ja_status = "⚠️  Acceptable (≥80%)"
    else:
        ja_status = "❌ Poor (<80%)"

    print(f"English Model:  {en_status}")
    print(f"Japanese Model: {ja_status}")

    # Recommendation
    print("\n" + "=" * 60)
    print("Recommendation")
    print("=" * 60)

    if en_acc >= 0.90 and ja_acc >= 0.85:
        print("✅ Both models perform well. Choose based on target market:")
        print("   - English: Broader dataset, more general insights")
        print("   - Japanese: Japan-specific insights")
    elif en_acc >= 0.90 and ja_acc < 0.85:
        print("✅ Recommend using English model:")
        print(f"   - English accuracy ({en_acc:.2%}) is significantly better")
        print(f"   - Japanese accuracy ({ja_acc:.2%}) may not be reliable")
    elif en_acc < 0.85:
        print("❌ Both models have low accuracy. Investigate:")
        print("   - Data quality issues")
        print("   - Model selection (try different models)")
        print("   - Implementation bugs")
    else:
        print("⚠️  Review both models carefully before deciding")


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
    language: str = "Model"
) -> None:
    """
    Print detailed classification report with sklearn

    Args:
        y_true: True labels
        y_pred: Predicted labels
        language: Language name for display

    Example:
        >>> print_detailed_classification_report(y_true, y_pred, "English")
    """
    print(f"\n{'=' * 60}")
    print(f"{language} - Detailed Classification Report")
    print(f"{'=' * 60}")

    # sklearn classification report
    target_names = ['Negative (0)', 'Positive (1)']
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    # Per-class accuracy
    class_stats = calculate_error_rate_by_class(y_true, y_pred)

    print(f"\nPer-Class Accuracy:")
    for class_name, (correct, total, acc) in class_stats.items():
        print(f"  {class_name.capitalize()}: {correct}/{total} = {acc:.2%}")
