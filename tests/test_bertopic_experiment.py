"""
BERTopic実験スクリプト

英語レビューのみを使用し、最適なパラメータでトピック抽出を実行。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re


def is_english(text: str) -> bool:
    """
    テキストが英語かどうかを判定

    Args:
        text: 判定するテキスト

    Returns:
        英語の場合True
    """
    # ASCIIと基本的な記号のみで構成されているか
    return bool(re.match(r'^[\x00-\x7F]+$', text))


def main():
    """BERTopic実験実行（英語のみ、1000件）"""

    print("=" * 70)
    print("BERTopic実験（英語レビューのみ、1000件）")
    print("=" * 70)

    # 1. データ読み込み
    print("\n1. データ読み込み")
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])

    print(f"   元データ: {len(df)}件")

    # 2. 英語レビューのみフィルタリング
    print("\n2. 英語レビューのみフィルタリング")
    df['is_english'] = df['review_text'].apply(is_english)
    df_english = df[df['is_english']].copy()

    print(f"   英語レビュー: {len(df_english)}件")
    print(f"   非英語レビュー: {len(df) - len(df_english)}件（除外）")

    texts = df_english['review_text'].tolist()

    # サンプル表示
    print(f"\n   サンプルレビュー:")
    for i, text in enumerate(texts[:3]):
        print(f"   {i+1}. {text[:80]}...")

    # 3. CountVectorizer設定（ストップワード除去・n-gram）
    print("\n3. CountVectorizer設定")
    vectorizer = CountVectorizer(
        stop_words='english',  # ストップワード除去
        ngram_range=(1, 2),    # 1-gram + 2-gram
        min_df=2               # 最低2回出現する単語のみ
    )
    print("   ストップワード除去: ON")
    print("   n-gram範囲: (1, 2)")
    print("   min_df: 2（最低出現回数）")

    # 4. BERTopicモデル初期化
    print("\n4. BERTopicモデル初期化")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=10,  # min_topic_sizeを10に設定
        verbose=False
    )
    print("   min_topic_size: 10")
    print("   モデル初期化完了")

    # 5. トピック抽出実行
    print("\n5. トピック抽出実行")
    topics, probs = topic_model.fit_transform(texts)

    num_topics = len(set(topics)) - 1  # -1はOutlier除外
    print(f"   抽出されたトピック数: {num_topics}")

    # 6. トピック情報表示
    print("\n6. トピック情報")
    topic_info = topic_model.get_topic_info()
    print(topic_info[['Topic', 'Count', 'Name']])

    # 7. 各トピックの代表的な単語を表示
    print("\n7. 各トピックの代表単語（上位10単語）")
    for topic_id in range(min(10, num_topics)):
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            words = ", ".join([word for word, _ in topic_words[:10]])
            count = len([t for t in topics if t == topic_id])
            print(f"\n   Topic {topic_id} ({count}件):")
            print(f"   {words}")

    # 8. サンプルレビュー（各トピックから3件ずつ）
    print("\n8. サンプルレビュー（各トピックから3件ずつ）")
    for topic_id in range(min(10, num_topics)):
        print(f"\n   === Topic {topic_id} ===")
        topic_reviews = [texts[i] for i, t in enumerate(topics) if t == topic_id]
        for i, review in enumerate(topic_reviews[:3]):
            print(f"   {i+1}. {review[:100]}...")

    # 9. Outlierの状況
    outlier_count = len([t for t in topics if t == -1])
    print(f"\n9. Outlier（未分類）")
    print(f"   Outlier数: {outlier_count} ({outlier_count/len(topics)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("実験完了")
    print("=" * 70)

    return topic_model, topics


if __name__ == '__main__':
    main()
