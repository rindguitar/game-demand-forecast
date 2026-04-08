"""
感情分析モジュールのテストスクリプト

事前学習済みDistilBERTを使用した感情分析機能をテストします。
"""

import sys
import os

# src をパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nlp.sentiment import (
    analyze_sentiment,
    predict_sentiment_labels,
    get_recommended_model,
    check_gpu_available,
    get_device
)


def test_gpu_check():
    """GPU利用可能性のテスト"""
    print("=" * 60)
    print("テスト1: GPU利用可能性チェック")
    print("=" * 60)

    gpu_available = check_gpu_available()
    device = get_device()

    if gpu_available:
        print(f"✅ GPUが利用可能です")
        print(f"  - デバイス: {device} (0=GPU)")
    else:
        print(f"⚠️  GPUが利用できません（CPUで実行）")
        print(f"  - デバイス: {device} (-1=CPU)")

    return True


def test_recommended_models():
    """推奨モデル取得のテスト"""
    print("\n" + "=" * 60)
    print("テスト2: 推奨モデル取得")
    print("=" * 60)

    try:
        en_model = get_recommended_model('english')
        ja_model = get_recommended_model('japanese')

        print(f"✅ 英語モデル: {en_model}")
        print(f"✅ 日本語モデル: {ja_model}")

        # 不正な言語でエラーチェック
        try:
            get_recommended_model('french')
            print(f"❌ 不正な言語で例外が発生しませんでした")
            return False
        except ValueError:
            print(f"✅ 不正な言語で正しくValueErrorが発生")

        return True

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_single_text_analysis():
    """単一テキストの感情分析テスト"""
    print("\n" + "=" * 60)
    print("テスト3: 単一テキストの感情分析")
    print("=" * 60)

    test_cases = [
        ("This game is absolutely amazing! Best game ever!", "POSITIVE"),
        ("Terrible game. Waste of money. Do not buy.", "NEGATIVE"),
        ("I love this game so much!", "POSITIVE"),
        ("Worst purchase I've ever made.", "NEGATIVE"),
    ]

    passed = 0
    for text, expected in test_cases:
        try:
            result = analyze_sentiment(text, device=get_device())
            predicted = result['label']
            score = result['score']

            if predicted == expected:
                print(f"✅ '{text[:40]}...'")
                print(f"   予測: {predicted} (信頼度: {score:.2%})")
                passed += 1
            else:
                print(f"❌ '{text[:40]}...'")
                print(f"   期待: {expected}, 実際: {predicted} (信頼度: {score:.2%})")

        except Exception as e:
            print(f"❌ エラー: {e}")
            return False

    print(f"\n{passed}/{len(test_cases)} テストケース成功")
    return passed == len(test_cases)


def test_batch_analysis():
    """バッチ処理のテスト"""
    print("\n" + "=" * 60)
    print("テスト4: バッチ処理（複数テキスト）")
    print("=" * 60)

    texts = [
        "Great game!",
        "Terrible experience",
        "Amazing graphics and gameplay",
        "Worst game ever",
        "I love it!",
        "Complete waste of time"
    ]

    expected = ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE"]

    try:
        results = analyze_sentiment(texts, batch_size=32, device=get_device())

        passed = 0
        for i, (text, exp, result) in enumerate(zip(texts, expected, results)):
            predicted = result['label']
            score = result['score']

            if predicted == exp:
                print(f"✅ テキスト{i+1}: {text}")
                print(f"   予測: {predicted} (信頼度: {score:.2%})")
                passed += 1
            else:
                print(f"❌ テキスト{i+1}: {text}")
                print(f"   期待: {exp}, 実際: {predicted} (信頼度: {score:.2%})")

        print(f"\n{passed}/{len(texts)} テストケース成功")
        return passed == len(texts)

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_label_prediction():
    """ラベル予測のテスト（0/1形式）"""
    print("\n" + "=" * 60)
    print("テスト5: ラベル予測（0/1形式）")
    print("=" * 60)

    texts = [
        "Excellent game!",
        "Horrible experience",
        "Best game I've played",
    ]

    expected = [1, 0, 1]  # 1=POSITIVE, 0=NEGATIVE

    try:
        predictions = predict_sentiment_labels(texts, device=get_device())

        print(f"予測ラベル: {predictions}")
        print(f"期待ラベル: {expected}")

        if predictions == expected:
            print(f"✅ すべての予測が正しい")
            return True
        else:
            print(f"❌ 予測が期待と一致しません")
            return False

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def test_steam_review_format():
    """Steamレビュー形式のテスト"""
    print("\n" + "=" * 60)
    print("テスト6: Steamレビュー形式での感情分析")
    print("=" * 60)

    import pandas as pd

    # Steamレビュー形式のDataFrame作成
    reviews_df = pd.DataFrame({
        'review_text': [
            "Great game with awesome graphics!",
            "Terrible bugs, unplayable.",
            "I love this game!",
        ]
    })

    expected = [1, 0, 1]

    try:
        from src.nlp.sentiment import analyze_steam_reviews
        predictions = analyze_steam_reviews(reviews_df, device=get_device())

        print(f"予測ラベル: {predictions}")
        print(f"期待ラベル: {expected}")

        if predictions == expected:
            print(f"✅ Steamレビュー形式で正しく動作")
            return True
        else:
            print(f"❌ 予測が期待と一致しません")
            return False

    except Exception as e:
        print(f"❌ エラー: {e}")
        return False


def main():
    """全テスト実行"""
    print("\n" + "🎭 " * 20)
    print("感情分析モジュールテスト")
    print("🎭 " * 20 + "\n")

    results = []

    # テスト実行
    results.append(("GPU利用可能性チェック", test_gpu_check()))
    results.append(("推奨モデル取得", test_recommended_models()))
    results.append(("単一テキスト分析", test_single_text_analysis()))
    results.append(("バッチ処理", test_batch_analysis()))
    results.append(("ラベル予測", test_label_prediction()))
    results.append(("Steamレビュー形式", test_steam_review_format()))

    # サマリー
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
