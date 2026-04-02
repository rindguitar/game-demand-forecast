"""
Test script for data preprocessing module

This script tests the preprocessing module with sample data.
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import (
    clean_review_text,
    steam_reviews_to_dataframe,
    balance_dataset,
    prepare_validation_dataset
)


def test_clean_review_text():
    """Test text cleaning function"""
    print("=" * 60)
    print("Test 1: Text cleaning")
    print("=" * 60)

    test_cases = [
        ("<b>Great game!</b>", "Great game!"),
        ("Too many   spaces", "Too many spaces"),
        ("  Trim whitespace  ", "Trim whitespace"),
        ("Line\nbreaks\nhere", "Line breaks here"),
        ("<div>HTML <span>tags</span></div>", "HTML tags"),
    ]

    passed = 0
    for input_text, expected in test_cases:
        result = clean_review_text(input_text)
        if result == expected:
            print(f"✅ '{input_text}' → '{result}'")
            passed += 1
        else:
            print(f"❌ '{input_text}' → '{result}' (expected '{expected}')")

    print(f"\n{passed}/{len(test_cases)} test cases passed")
    return passed == len(test_cases)


def test_steam_reviews_to_dataframe():
    """Test DataFrame conversion"""
    print("\n" + "=" * 60)
    print("Test 2: Steam reviews to DataFrame conversion")
    print("=" * 60)

    # Load sample data
    sample_file = 'data/raw/sample_reviews.json'
    if not os.path.exists(sample_file):
        print(f"❌ Sample data not found: {sample_file}")
        print("  Run test_steam_api.py first to generate sample data")
        return False

    with open(sample_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    # Convert to DataFrame
    df = steam_reviews_to_dataframe(reviews)

    print(f"✅ Converted {len(reviews)} reviews to DataFrame")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())

    # Verify columns
    required_columns = ['review_text', 'label', 'votes_up', 'language', 'timestamp_created', 'author']
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False

    # Verify labels are 0 or 1
    if not df['label'].isin([0, 1]).all():
        print(f"❌ Invalid labels found (should be 0 or 1)")
        return False

    print(f"✅ All required columns present")
    print(f"✅ Labels are valid (0 or 1)")

    # Show sample
    print(f"\nSample row:")
    sample = df.iloc[0]
    print(f"  - Text: {sample['review_text'][:100]}...")
    print(f"  - Label: {sample['label']} ({'Positive' if sample['label'] == 1 else 'Negative'})")
    print(f"  - Language: {sample['language']}")

    return True


def test_balance_dataset():
    """Test dataset balancing"""
    print("\n" + "=" * 60)
    print("Test 3: Dataset balancing")
    print("=" * 60)

    import pandas as pd

    # Create imbalanced dataset
    df = pd.DataFrame({
        'review_text': ['text' + str(i) for i in range(15)],
        'label': [1] * 10 + [0] * 5,  # 10 positive, 5 negative
        'votes_up': list(range(15)),
        'language': ['english'] * 15,
        'timestamp_created': list(range(15)),
        'author': ['user' + str(i) for i in range(15)],
    })

    print(f"Original distribution:")
    print(df['label'].value_counts())

    # Balance
    balanced_df = balance_dataset(df)

    print(f"\nBalanced distribution:")
    print(balanced_df['label'].value_counts())

    # Verify balance
    label_counts = balanced_df['label'].value_counts()
    if len(label_counts) == 2 and label_counts[0] == label_counts[1]:
        print(f"✅ Dataset is balanced ({label_counts[0]} samples per class)")
        return True
    else:
        print(f"❌ Dataset is not balanced")
        return False


def test_prepare_validation_dataset():
    """Test validation dataset preparation"""
    print("\n" + "=" * 60)
    print("Test 4: Validation dataset preparation")
    print("=" * 60)

    # Create sample reviews
    positive_reviews = [
        {'review_text': f'Great game {i}!', 'voted_up': True, 'votes_up': i, 'language': 'english', 'timestamp_created': i, 'author': f'user{i}'}
        for i in range(10)
    ]

    negative_reviews = [
        {'review_text': f'Bad game {i}!', 'voted_up': False, 'votes_up': i, 'language': 'english', 'timestamp_created': i, 'author': f'user{i}'}
        for i in range(10)
    ]

    # Prepare validation dataset
    df = prepare_validation_dataset(positive_reviews, negative_reviews, n_per_class=5)

    print(f"Validation dataset prepared:")
    print(f"  - Shape: {df.shape}")
    print(f"  - Label distribution:")
    print(df['label'].value_counts())

    # Verify
    if len(df) == 10:  # 5 positive + 5 negative
        label_counts = df['label'].value_counts()
        if label_counts[0] == 5 and label_counts[1] == 5:
            print(f"✅ Validation dataset is correctly balanced")
            return True

    print(f"❌ Validation dataset is not correctly balanced")
    return False


def test_save_cleaned_data():
    """Test saving cleaned data to CSV"""
    print("\n" + "=" * 60)
    print("Test 5: Save cleaned data to CSV")
    print("=" * 60)

    # Load sample data
    sample_file = 'data/raw/sample_reviews.json'
    if not os.path.exists(sample_file):
        print(f"❌ Sample data not found: {sample_file}")
        return False

    with open(sample_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    # Convert and clean
    df = steam_reviews_to_dataframe(reviews)

    # Create validation directory
    os.makedirs('data/validation', exist_ok=True)

    # Save to CSV
    output_path = 'data/validation/reviews_cleaned.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"✅ Saved {len(df)} cleaned reviews to {output_path}")
    print(f"  - File size: {os.path.getsize(output_path)} bytes")

    # Verify by loading
    df_loaded = pd.read_csv(output_path)
    if len(df_loaded) == len(df):
        print(f"✅ Verified: CSV file loaded correctly")
        return True
    else:
        print(f"❌ CSV file verification failed")
        return False


def main():
    """Run all tests"""
    print("\n" + "🧹 " * 20)
    print("Data Preprocessing Tests")
    print("🧹 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("Text cleaning", test_clean_review_text()))
    results.append(("DataFrame conversion", test_steam_reviews_to_dataframe()))
    results.append(("Dataset balancing", test_balance_dataset()))
    results.append(("Validation dataset prep", test_prepare_validation_dataset()))
    results.append(("Save cleaned data", test_save_cleaned_data()))

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
    # Import pandas here to avoid import error if not installed
    import pandas as pd
    exit(main())
