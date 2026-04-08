"""
Test script for evaluation module

This script tests the evaluation module with dummy data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nlp.evaluation import (
    evaluate_sentiment_model,
    print_evaluation_metrics,
    print_comparison_report,
    calculate_error_rate_by_class,
    print_detailed_classification_report
)


def test_perfect_predictions():
    """Test with perfect predictions (100% accuracy)"""
    print("=" * 60)
    print("Test 1: Perfect predictions (100% accuracy)")
    print("=" * 60)

    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # Same as true

    results = evaluate_sentiment_model(y_true, y_pred)

    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")

    if results['accuracy'] == 1.0:
        print(f"✅ Perfect accuracy: {results['accuracy']:.2%}")
    else:
        print(f"❌ Expected 100% accuracy, got {results['accuracy']:.2%}")
        return False

    if results['precision'] == 1.0 and results['recall'] == 1.0 and results['f1_score'] == 1.0:
        print(f"✅ Perfect precision, recall, F1: {results['precision']:.2%}")
    else:
        print(f"❌ Expected perfect metrics")
        return False

    return True


def test_imperfect_predictions():
    """Test with realistic predictions (80% accuracy)"""
    print("\n" + "=" * 60)
    print("Test 2: Imperfect predictions (~80% accuracy)")
    print("=" * 60)

    # 8 correct, 2 incorrect out of 10
    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]  # 2 errors

    results = evaluate_sentiment_model(y_true, y_pred)

    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    print(f"Errors at index:  4, 7")

    if 0.75 <= results['accuracy'] <= 0.85:
        print(f"✅ Accuracy in expected range: {results['accuracy']:.2%}")
    else:
        print(f"❌ Unexpected accuracy: {results['accuracy']:.2%}")
        return False

    # Print metrics
    print_evaluation_metrics(results, "Imperfect Model")

    return True


def test_class_imbalance():
    """Test with class imbalance"""
    print("\n" + "=" * 60)
    print("Test 3: Class imbalance (7 positive, 3 negative)")
    print("=" * 60)

    y_true = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]  # 7 positive, 3 negative
    y_pred = [1, 1, 1, 0, 1, 1, 1, 0, 0, 1]  # 1 FN, 1 FP

    results = evaluate_sentiment_model(y_true, y_pred)

    print(f"True distribution: 7 positive, 3 negative")
    print(f"Accuracy: {results['accuracy']:.2%}")

    # Calculate per-class accuracy
    class_stats = calculate_error_rate_by_class(y_true, y_pred)

    print(f"\nPer-class accuracy:")
    print(f"  Positive: {class_stats['positive'][0]}/{class_stats['positive'][1]} = {class_stats['positive'][2]:.2%}")
    print(f"  Negative: {class_stats['negative'][0]}/{class_stats['negative'][1]} = {class_stats['negative'][2]:.2%}")

    if results['accuracy'] == 0.8:
        print(f"✅ Accuracy matches expected: {results['accuracy']:.2%}")
        return True
    else:
        print(f"❌ Unexpected accuracy: {results['accuracy']:.2%}")
        return False


def test_comparison_report():
    """Test comparison report for English vs Japanese"""
    print("\n" + "=" * 60)
    print("Test 4: Language comparison report")
    print("=" * 60)

    # Simulate English model (90% accuracy)
    y_true_en = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10  # 100 samples
    y_pred_en = y_true_en[:]
    # Introduce 10 errors (90% accuracy)
    for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
        y_pred_en[i] = 1 - y_pred_en[i]

    results_en = evaluate_sentiment_model(y_true_en, y_pred_en)

    # Simulate Japanese model (85% accuracy)
    y_true_ja = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10  # 100 samples
    y_pred_ja = y_true_ja[:]
    # Introduce 15 errors (85% accuracy)
    for i in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 10, 20, 30, 40, 50]:
        y_pred_ja[i] = 1 - y_pred_ja[i]

    results_ja = evaluate_sentiment_model(y_true_ja, y_pred_ja)

    print(f"English model accuracy:  {results_en['accuracy']:.2%}")
    print(f"Japanese model accuracy: {results_ja['accuracy']:.2%}")

    # Print comparison
    print_comparison_report(results_en, results_ja)

    if 0.88 <= results_en['accuracy'] <= 0.92 and 0.83 <= results_ja['accuracy'] <= 0.87:
        print(f"\n✅ Comparison report generated successfully")
        return True
    else:
        print(f"\n❌ Unexpected accuracy values")
        return False


def test_detailed_classification_report():
    """Test detailed classification report"""
    print("\n" + "=" * 60)
    print("Test 5: Detailed classification report")
    print("=" * 60)

    y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5  # 50 samples
    y_pred = y_true[:]
    # Introduce 5 errors
    for i in [5, 15, 25, 35, 45]:
        y_pred[i] = 1 - y_pred[i]

    print_detailed_classification_report(y_true, y_pred, "Test Model")

    print(f"\n✅ Detailed report generated successfully")
    return True


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "=" * 60)
    print("Test 6: Edge cases")
    print("=" * 60)

    # Test 1: All predictions correct
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    results = evaluate_sentiment_model(y_true, y_pred)
    if results['accuracy'] == 1.0:
        print(f"✅ All correct predictions: {results['accuracy']:.2%}")
    else:
        print(f"❌ Failed all correct test")
        return False

    # Test 2: All predictions wrong
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]
    results = evaluate_sentiment_model(y_true, y_pred)
    if results['accuracy'] == 0.0:
        print(f"✅ All wrong predictions: {results['accuracy']:.2%}")
    else:
        print(f"❌ Failed all wrong test")
        return False

    # Test 3: Error handling - empty lists
    try:
        evaluate_sentiment_model([], [])
        print(f"❌ Should have raised ValueError for empty lists")
        return False
    except ValueError:
        print(f"✅ Correctly raised ValueError for empty lists")

    # Test 4: Error handling - length mismatch
    try:
        evaluate_sentiment_model([1, 0], [1])
        print(f"❌ Should have raised ValueError for length mismatch")
        return False
    except ValueError:
        print(f"✅ Correctly raised ValueError for length mismatch")

    return True


def main():
    """Run all tests"""
    print("\n" + "📊 " * 20)
    print("Evaluation Module Tests")
    print("📊 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("Perfect predictions", test_perfect_predictions()))
    results.append(("Imperfect predictions", test_imperfect_predictions()))
    results.append(("Class imbalance", test_class_imbalance()))
    results.append(("Comparison report", test_comparison_report()))
    results.append(("Detailed report", test_detailed_classification_report()))
    results.append(("Edge cases", test_edge_cases()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
