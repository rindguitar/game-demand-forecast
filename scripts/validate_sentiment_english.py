"""
英語レビュー100件での感情分析精度検証スクリプト

Steam APIから英語レビュー100件（Positive 50件 + Negative 50件）を収集し、
事前学習済みDistilBERTで感情分析を実行して精度を評価します。

目標精度: 90%以上
"""

import sys
import os
import pandas as pd

# srcをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.steam_collector import collect_balanced_reviews
from src.data.preprocessing import prepare_validation_dataset
from src.nlp.sentiment import predict_sentiment_labels, get_device
from src.nlp.evaluation import (
    evaluate_sentiment_model,
    print_evaluation_metrics,
    print_detailed_classification_report
)


def main():
    """メイン処理"""
    print("\n" + "=" * 60)
    print("英語レビュー100件での感情分析精度検証")
    print("=" * 60)

    # 設定
    APP_ID = 570  # Dota 2（人気ゲーム、レビュー多数）
    LANGUAGE = 'english'
    N_POSITIVE = 50
    N_NEGATIVE = 50

    print(f"\n設定:")
    print(f"  - ゲームID: {APP_ID} (Dota 2)")
    print(f"  - 言語: {LANGUAGE}")
    print(f"  - Positive: {N_POSITIVE}件")
    print(f"  - Negative: {N_NEGATIVE}件")
    print(f"  - 合計: {N_POSITIVE + N_NEGATIVE}件")

    # Step 1: データ収集
    print("\n" + "=" * 60)
    print("Step 1: Steam APIからレビュー収集")
    print("=" * 60)

    try:
        reviews = collect_balanced_reviews(
            app_id=APP_ID,
            language=LANGUAGE,
            n_positive=N_POSITIVE,
            n_negative=N_NEGATIVE
        )

        print(f"✅ 収集完了:")
        print(f"  - Positive: {len(reviews['positive'])}件")
        print(f"  - Negative: {len(reviews['negative'])}件")

    except Exception as e:
        print(f"❌ データ収集エラー: {e}")
        return 1

    # Step 2: データ前処理
    print("\n" + "=" * 60)
    print("Step 2: データ前処理")
    print("=" * 60)

    try:
        df = prepare_validation_dataset(
            reviews['positive'],
            reviews['negative'],
            n_per_class=50
        )

        print(f"✅ 前処理完了:")
        print(f"  - DataFrame shape: {df.shape}")
        print(f"  - Columns: {list(df.columns)}")
        print(f"\nゲームへの評価の分布:")
        print(df['game_rating'].value_counts())

        # CSVに保存
        os.makedirs('data/validation', exist_ok=True)
        output_path = 'data/validation/english_reviews_100.csv'
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ 保存完了: {output_path}")

    except Exception as e:
        print(f"❌ 前処理エラー: {e}")
        return 1

    # Step 3: 感情分析実行
    print("\n" + "=" * 60)
    print("Step 3: 感情分析実行")
    print("=" * 60)

    try:
        device = get_device()
        print(f"使用デバイス: {'GPU (0)' if device == 0 else 'CPU (-1)'}")

        # 正解ラベル（game_rating: 1=Recommended, 0=Not Recommended）
        y_true = df['game_rating'].tolist()

        # 感情分析で予測
        print(f"\n感情分析を実行中...")
        y_pred = predict_sentiment_labels(
            df['review_text'].tolist(),
            device=device
        )

        print(f"✅ 感情分析完了: {len(y_pred)}件の予測")

    except Exception as e:
        print(f"❌ 感情分析エラー: {e}")
        return 1

    # Step 4: 精度評価
    print("\n" + "=" * 60)
    print("Step 4: 精度評価")
    print("=" * 60)

    try:
        # 評価指標を計算
        results = evaluate_sentiment_model(y_true, y_pred)

        # 結果表示
        print_evaluation_metrics(results, "英語")

        # 詳細レポート
        print_detailed_classification_report(y_true, y_pred, "英語")

    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        return 1

    # Step 5: 判定
    print("\n" + "=" * 60)
    print("判定結果")
    print("=" * 60)

    accuracy = results['accuracy']

    if accuracy >= 0.90:
        print(f"\n✅ 成功！ 精度 {accuracy:.2%} ≥ 90%")
        print(f"\n英語モデルは十分な精度を持っています。")
        print(f"英語で感情分析を進めることを推奨します。")
        print(f"\n次のステップ:")
        print(f"  - Issue #5を完了")
        print(f"  - Issue #6（1000件でPyTorch学習）に進む")
        return 0
    elif accuracy >= 0.85:
        print(f"\n⚠️  精度 {accuracy:.2%} は85-90%の範囲")
        print(f"許容範囲ですが、90%未満です。")
        print(f"モデルまたはデータを見直すことを検討してください。")
        return 0
    else:
        print(f"\n❌ 精度 {accuracy:.2%} < 85%")
        print(f"実装に問題がある可能性があります。")
        print(f"以下を確認してください:")
        print(f"  - モデルの選択")
        print(f"  - データの品質")
        print(f"  - 実装のバグ")
        return 1


if __name__ == '__main__':
    exit(main())
