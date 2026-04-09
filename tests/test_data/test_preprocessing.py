"""
データ前処理モジュールのテストスクリプト

sampleデータを使用してpreprocessingモジュールをテストします。
"""

import sys
import os
import json
import pandas as pd

# srcをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.preprocessing import (
    clean_review_text,
    steam_reviews_to_dataframe,
    balance_dataset,
    prepare_validation_dataset
)


def test_clean_review_text():
    """テキストクリーニング機能のテスト"""
    print("=" * 60)
    print("テスト1: テキストクリーニング")
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
            print(f"❌ '{input_text}' → '{result}' (期待値: '{expected}')")

    print(f"\n{passed}/{len(test_cases)} テストケース成功")
    return passed == len(test_cases)


def test_steam_reviews_to_dataframe():
    """DataFrame変換のテスト"""
    print("\n" + "=" * 60)
    print("テスト2: SteamレビューからDataFrameへの変換")
    print("=" * 60)

    # Load sample data
    sample_file = 'data/raw/sample_reviews.json'
    if not os.path.exists(sample_file):
        print(f"❌ サンプルデータが見つかりません: {sample_file}")
        print("  先にtest_steam_api.pyを実行してサンプルデータを生成してください")
        return False

    with open(sample_file, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    # Convert to DataFrame
    df = steam_reviews_to_dataframe(reviews)

    print(f"✅ Converted {len(reviews)} reviews to DataFrame")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    print(f"\nゲームへの評価（game_rating）の分布:")
    print(df['game_rating'].value_counts())

    # Verify columns
    required_columns = ['review_text', 'game_rating', 'review_helpfulness', 'language', 'posted_date', 'user_id']
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        print(f"❌ 不足しているカラム: {missing_columns}")
        return False

    # Verify game_rating are 0 or 1
    if not df['game_rating'].isin([0, 1]).all():
        print(f"❌ game_ratingに無効な値が含まれています（0または1である必要があります）")
        return False

    print(f"✅ すべての必須カラムが存在します")
    print(f"✅ game_ratingは有効です（0または1）")

    # Show sample
    print(f"\nサンプル行:")
    sample = df.iloc[0]
    print(f"  - レビュー本文: {sample['review_text'][:100]}...")
    print(f"  - ゲームへの評価: {sample['game_rating']} ({'おすすめ' if sample['game_rating'] == 1 else 'おすすめしない'})")
    print(f"  - 言語: {sample['language']}")

    return True


def test_balance_dataset():
    """datasetバランシングのテスト"""
    print("\n" + "=" * 60)
    print("テスト3: データセットバランシング")
    print("=" * 60)

    import pandas as pd

    # Create imbalanced dataset
    df = pd.DataFrame({
        'review_text': ['text' + str(i) for i in range(15)],
        'game_rating': [1] * 10 + [0] * 5,  # 10 positive, 5 negative
        'review_helpfulness': list(range(15)),
        'language': ['english'] * 15,
        'posted_date': list(range(15)),
        'user_id': ['user' + str(i) for i in range(15)],
    })

    print(f"元のデータ分布:")
    print(df['game_rating'].value_counts())

    # Balance
    balanced_df = balance_dataset(df)

    print(f"\nバランス後のデータ分布:")
    print(balanced_df['game_rating'].value_counts())

    # Verify balance
    label_counts = balanced_df['game_rating'].value_counts()
    if len(label_counts) == 2 and label_counts[0] == label_counts[1]:
        print(f"✅ データセットがバランスされました（クラスあたり{label_counts[0]}サンプル）")
        return True
    else:
        print(f"❌ データセットがバランスされていません")
        return False


def test_prepare_validation_dataset():
    """検証用dataset準備のテスト"""
    print("\n" + "=" * 60)
    print("テスト4: 検証用データセット準備")
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

    print(f"検証用データセットの準備完了:")
    print(f"  - Shape: {df.shape}")
    print(f"  - ゲームへの評価の分布:")
    print(df['game_rating'].value_counts())

    # Verify
    if len(df) == 10:  # 5 positive + 5 negative
        label_counts = df['game_rating'].value_counts()
        if label_counts[0] == 5 and label_counts[1] == 5:
            print(f"✅ 検証用データセットが正しくバランスされました")
            return True

    print(f"❌ 検証用データセットが正しくバランスされていません")
    return False


def test_save_cleaned_data():
    """クリーニング済みデータのCSV保存テスト"""
    print("\n" + "=" * 60)
    print("テスト5: クリーニング済みデータをCSVに保存")
    print("=" * 60)

    # Load sample data
    sample_file = 'data/raw/sample_reviews.json'
    if not os.path.exists(sample_file):
        print(f"❌ サンプルデータが見つかりません: {sample_file}")
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

    print(f"✅ {len(df)}件のクリーニング済みレビューを保存しました: {output_path}")
    print(f"  - ファイルサイズ: {os.path.getsize(output_path)} bytes")

    # Verify by loading
    df_loaded = pd.read_csv(output_path)
    if len(df_loaded) == len(df):
        print(f"✅ 検証完了: CSVファイルが正しく読み込まれました")
        return True
    else:
        print(f"❌ CSVファイルの検証に失敗しました")
        return False


def main():
    """全テスト実行"""
    print("\n" + "🧹 " * 20)
    print("データ前処理テスト")
    print("🧹 " * 20 + "\n")

    results = []

    # Run tests
    results.append(("テキストクリーニング", test_clean_review_text()))
    results.append(("DataFrame変換", test_steam_reviews_to_dataframe()))
    results.append(("データセットバランシング", test_balance_dataset()))
    results.append(("検証用データセット準備", test_prepare_validation_dataset()))
    results.append(("クリーニング済みデータ保存", test_save_cleaned_data()))

    # Summary
    print("\n" + "=" * 60)
    print("テスト結果サマリー")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ 成功" if passed else "❌ 失敗"
        print(f"{status} - {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\n合計: {passed}/{total} テスト成功")

    if passed == total:
        print("\n🎉 すべてのテストが成功しました！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} テスト失敗")
        return 1


if __name__ == '__main__':
    exit(main())
