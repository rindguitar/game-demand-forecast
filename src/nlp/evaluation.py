"""
感情分析評価モジュール

感情分析モデルの評価と異なる言語間での結果比較機能を提供します。
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
    感情分析modelの性能を評価

    Args:
        y_true: 正解label（0または1）
        y_pred: 予測label（0または1）

    Returns:
        以下を含むdict:
            - accuracy: 全体のaccuracy
            - precision: precision
            - recall: recall
            - f1_score: F1 score
            - confusion_matrix: 2x2 numpy配列 [[TN, FP], [FN, TP]]

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

    # metricsを計算
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
    評価metricsをフォーマットして表示

    Args:
        results: evaluate_sentiment_model()からのdict
        language: 表示用の言語名（例: "英語", "日本語"）

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

    # confusion matrixを表示
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
    英語modelと日本語modelの比較レポートを表示

    Args:
        results_en: 英語modelの評価結果
        results_ja: 日本語modelの評価結果

    Example:
        >>> print_comparison_report(results_en, results_ja)
    """
    print("\n" + "=" * 60)
    print("言語比較レポート")
    print("=" * 60)

    # テーブルヘッダー
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

    # 解釈
    print("\n" + "=" * 60)
    print("評価")
    print("=" * 60)

    en_acc = results_en['accuracy']
    ja_acc = results_ja['accuracy']

    # 英語modelの評価
    if en_acc >= 0.90:
        en_status = "✅ 優秀 (≥90%)"
    elif en_acc >= 0.85:
        en_status = "✅ 良好 (≥85%)"
    elif en_acc >= 0.80:
        en_status = "⚠️  許容範囲 (≥80%)"
    else:
        en_status = "❌ 不良 (<80%)"

    # 日本語modelの評価
    if ja_acc >= 0.85:
        ja_status = "✅ 優秀 (≥85%)"
    elif ja_acc >= 0.80:
        ja_status = "⚠️  許容範囲 (≥80%)"
    else:
        ja_status = "❌ 不良 (<80%)"

    print(f"英語モデル:   {en_status}")
    print(f"日本語モデル: {ja_status}")

    # 推奨
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
    各class（Positive/Negative）のerror rateを計算

    Args:
        y_true: 正解label
        y_pred: 予測label

    Returns:
        'positive'と'negative'をkeyとするdict、各valueは以下を含む:
            - (正解数, 総数, accuracy)

    Example:
        >>> y_true = [1, 1, 0, 0]
        >>> y_pred = [1, 0, 0, 0]
        >>> result = calculate_error_rate_by_class(y_true, y_pred)
        >>> result['positive']
        (1, 2, 0.5)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Positiveクラス（label=1）
    pos_mask = y_true == 1
    pos_correct = np.sum((y_true[pos_mask] == y_pred[pos_mask]))
    pos_total = np.sum(pos_mask)
    pos_acc = pos_correct / pos_total if pos_total > 0 else 0.0

    # Negativeクラス（label=0）
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
    sklearnを使用して詳細なclassification reportを表示

    Args:
        y_true: 正解label
        y_pred: 予測label
        language: 表示用の言語名

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

    # classごとのaccuracy
    class_stats = calculate_error_rate_by_class(y_true, y_pred)

    print(f"\nクラス別正解率:")
    for class_name, (correct, total, acc) in class_stats.items():
        print(f"  {class_name.capitalize()}: {correct}/{total} = {acc:.2%}")
