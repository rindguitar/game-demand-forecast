"""
Test script for Steam API data collection

This script tests the steam_collector module with a small sample.
"""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.steam_collector import get_steam_reviews, collect_balanced_reviews


def test_basic_collection():
    """Test basic review collection"""
    print("=" * 60)
    print("Test 1: Basic review collection (10 reviews)")
    print("=" * 60)

    # Use CS:GO (app_id=730) as test game - very popular with many reviews
    app_id = 730
    num_reviews = 10

    try:
        reviews = get_steam_reviews(
            app_id=app_id,
            language='english',
            review_type='all',
            num=num_reviews
        )

        print(f"✅ Successfully collected {len(reviews)} reviews")
        print(f"\nSample review:")
        if reviews:
            sample = reviews[0]
            print(f"  - Recommended: {sample['voted_up']}")
            print(f"  - Language: {sample['language']}")
            print(f"  - Upvotes: {sample['votes_up']}")
            print(f"  - Text (first 100 chars): {sample['review_text'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_japanese_collection():
    """Test Japanese review collection"""
    print("\n" + "=" * 60)
    print("Test 2: Japanese review collection (5 reviews)")
    print("=" * 60)

    app_id = 730  # CS:GO
    num_reviews = 5

    try:
        reviews = get_steam_reviews(
            app_id=app_id,
            language='japanese',
            review_type='all',
            num=num_reviews
        )

        print(f"✅ Successfully collected {len(reviews)} Japanese reviews")
        if reviews:
            sample = reviews[0]
            print(f"\nSample Japanese review:")
            print(f"  - Recommended: {sample['voted_up']}")
            print(f"  - Text (first 100 chars): {sample['review_text'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_balanced_collection():
    """Test balanced positive/negative collection"""
    print("\n" + "=" * 60)
    print("Test 3: Balanced collection (5 positive + 5 negative)")
    print("=" * 60)

    app_id = 730  # CS:GO

    try:
        reviews = collect_balanced_reviews(
            app_id=app_id,
            language='english',
            n_positive=5,
            n_negative=5
        )

        print(f"✅ Successfully collected balanced reviews:")
        print(f"  - Positive: {len(reviews['positive'])} reviews")
        print(f"  - Negative: {len(reviews['negative'])} reviews")

        # Verify labels
        positive_labels = [r['voted_up'] for r in reviews['positive']]
        negative_labels = [r['voted_up'] for r in reviews['negative']]

        if all(positive_labels) and not any(negative_labels):
            print(f"✅ Labels are correct (Positive=True, Negative=False)")
        else:
            print(f"⚠️  Label mismatch detected")
            print(f"  - Positive labels: {positive_labels}")
            print(f"  - Negative labels: {negative_labels}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_save_sample_data():
    """Save sample data to data/raw/"""
    print("\n" + "=" * 60)
    print("Test 4: Save sample data to data/raw/")
    print("=" * 60)

    app_id = 730
    num_reviews = 10

    try:
        # Create data/raw directory if not exists
        os.makedirs('data/raw', exist_ok=True)

        reviews = get_steam_reviews(
            app_id=app_id,
            language='english',
            review_type='all',
            num=num_reviews
        )

        # Save to JSON
        output_path = 'data/raw/sample_reviews.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reviews, f, indent=2, ensure_ascii=False)

        print(f"✅ Successfully saved {len(reviews)} reviews to {output_path}")
        print(f"  - File size: {os.path.getsize(output_path)} bytes")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("Steam API Data Collection Tests")
    print("🚀 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("Basic collection", test_basic_collection()))
    results.append(("Japanese collection", test_japanese_collection()))
    results.append(("Balanced collection", test_balanced_collection()))
    results.append(("Save sample data", test_save_sample_data()))

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
