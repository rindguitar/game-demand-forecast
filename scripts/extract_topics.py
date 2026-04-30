"""
トピック抽出実行スクリプト（本番用）

10000件のレビューデータからBERTopicでトピックを抽出し、結果をCSVに保存する。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from src.nlp.topic import (
    filter_english_reviews,
    remove_game_names,
    create_topic_model,
    extract_topics,
    print_topic_summary,
    get_topic_info,
    get_topic_words
)


def main():
    """トピック抽出のメイン実行関数"""

    print("=" * 80)
    print("🔍 トピック抽出実行（10000件レビュー）")
    print("=" * 80)

    # 1. データ読み込み
    print("\n[1/5] データ読み込み")
    input_path = 'data/train/reviews_10000.csv'
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['review_text'])
    print(f"   ✓ 読み込み完了: {len(df)}件（{input_path}）")

    # 2. 英語レビューのみフィルタリング
    print("\n[2/5] 英語レビューフィルタリング")
    original_count = len(df)
    df_english = filter_english_reviews(df, text_column='review_text')
    english_count = len(df_english)
    filtered_count = original_count - english_count

    print(f"   ✓ 英語レビュー: {english_count}件")
    print(f"   ✓ 除外（非英語）: {filtered_count}件 ({filtered_count/original_count*100:.1f}%)")

    # 3. ゲーム名除去
    print("\n[3/5] ゲーム名除去")
    df_english = remove_game_names(df_english, text_column='review_text', game_name_column='game_name')

    texts = df_english['review_text'].tolist()

    # 4. BERTopicモデル作成
    print("\n[4/6] BERTopicモデル初期化")
    # 10000件データなので min_topic_size を調整（10 → 20に増加）
    min_topic_size = 20
    topic_model = create_topic_model(
        min_topic_size=min_topic_size,
        embedding_model_name='all-MiniLM-L6-v2',
        ngram_range=(1, 2),
        min_df=2,
        verbose=True
    )
    print(f"   ✓ min_topic_size: {min_topic_size}")
    print(f"   ✓ 埋め込みモデル: all-MiniLM-L6-v2")
    print(f"   ✓ n-gram範囲: (1, 2)")
    print(f"   ✓ ストップワード除去: ON")

    # 5. トピック抽出実行
    print("\n[5/6] トピック抽出実行中...")
    print("   ⏳ 処理には数分かかる場合があります...")

    topic_model, topics, probabilities = extract_topics(
        texts=texts,
        topic_model=topic_model,
        verbose=True
    )

    num_topics = len(set(topics)) - 1  # -1はOutlier除外
    outlier_count = len([t for t in topics if t == -1])

    print(f"   ✓ トピック抽出完了")
    print(f"   ✓ 抽出トピック数: {num_topics}")
    print(f"   ✓ Outlier: {outlier_count}件 ({outlier_count/len(topics)*100:.1f}%)")

    # 6. 結果をサマリー表示
    print("\n[6/6] 結果サマリー")
    print_topic_summary(
        topic_model=topic_model,
        topics=topics,
        texts=texts,
        max_topics=20,  # 上位20トピックまで表示
        top_n_words=10,
        sample_reviews=3
    )

    # 6. 結果をCSVに保存
    print("\n" + "=" * 80)
    print("💾 結果保存")
    print("=" * 80)

    # レビュー単位の結果
    df_english['topic_id'] = topics
    df_english['topic_probability'] = probabilities

    # トピック情報を追加
    topic_info_df = get_topic_info(topic_model)
    topic_name_map = dict(zip(topic_info_df['Topic'], topic_info_df['Name']))
    df_english['topic_name'] = df_english['topic_id'].map(topic_name_map)

    # 代表単語を追加（上位5単語）
    topic_words_map = {}
    for topic_id in range(num_topics):
        words = get_topic_words(topic_model, topic_id, top_n=5)
        if words:
            topic_words_map[topic_id] = ", ".join([word for word, _ in words])
        else:
            topic_words_map[topic_id] = ""
    topic_words_map[-1] = "Outlier"  # Outlier用

    df_english['topic_keywords'] = df_english['topic_id'].map(topic_words_map)

    # 保存
    output_path = 'data/train/reviews_10000_with_topics.csv'
    df_english.to_csv(output_path, index=False)
    print(f"\n   ✓ レビュー+トピック結果: {output_path}")
    print(f"     - カラム: review_text, topic_id, topic_probability, topic_name, topic_keywords")

    # トピック統計を別途保存
    topic_stats = []
    for topic_id in range(num_topics):
        count = len([t for t in topics if t == topic_id])
        percentage = count / len(topics) * 100
        keywords = topic_words_map.get(topic_id, "")
        topic_name = topic_name_map.get(topic_id, "Unknown")

        topic_stats.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'keywords': keywords,
            'count': count,
            'percentage': percentage
        })

    # Outlier統計も追加
    topic_stats.append({
        'topic_id': -1,
        'topic_name': 'Outlier',
        'keywords': 'N/A',
        'count': outlier_count,
        'percentage': outlier_count / len(topics) * 100
    })

    df_stats = pd.DataFrame(topic_stats)
    df_stats = df_stats.sort_values('count', ascending=False)

    stats_output_path = 'data/train/topic_statistics.csv'
    df_stats.to_csv(stats_output_path, index=False)
    print(f"   ✓ トピック統計: {stats_output_path}")
    print(f"     - カラム: topic_id, topic_name, keywords, count, percentage")

    print("\n" + "=" * 80)
    print("✅ トピック抽出完了")
    print("=" * 80)
    print(f"\n📊 概要:")
    print(f"   - 処理レビュー数: {len(texts)}件")
    print(f"   - 抽出トピック数: {num_topics}")
    print(f"   - Outlier: {outlier_count}件 ({outlier_count/len(topics)*100:.1f}%)")
    print(f"\n📁 出力ファイル:")
    print(f"   1. {output_path}")
    print(f"   2. {stats_output_path}")
    print()


if __name__ == '__main__':
    main()
