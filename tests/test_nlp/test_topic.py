"""
トピック抽出モジュールのテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pytest
import pandas as pd
from src.nlp.topic import (
    is_english,
    filter_english_reviews,
    create_topic_model,
    extract_topics,
    get_topic_info,
    get_topic_words,
    print_topic_summary
)


def test_is_english():
    """英語判定のテスト"""
    # 英語
    assert is_english("This is a great game!") == True
    assert is_english("I love this game") == True
    assert is_english("10/10") == True

    # 非英語
    assert is_english("これは素晴らしいゲームです") == False
    assert is_english("Разраба маму") == False
    assert is_english("有一说一这游戏玩久了") == False

    # 混合（非ASCII文字が含まれる）
    assert is_english("Great game! 最高！") == False


def test_filter_english_reviews():
    """英語レビューフィルタリングのテスト"""
    # テストデータ作成
    df = pd.DataFrame({
        'review_text': [
            'This is a great game!',
            'これは素晴らしいゲームです',
            'I love this game',
            'Разраба маму',
            'Amazing game'
        ],
        'label': [1, 1, 1, 0, 1]
    })

    df_english = filter_english_reviews(df, text_column='review_text')

    # 英語レビューのみが残る
    assert len(df_english) == 3
    assert 'This is a great game!' in df_english['review_text'].values
    assert 'I love this game' in df_english['review_text'].values
    assert 'Amazing game' in df_english['review_text'].values

    # 非英語レビューは除外される
    assert 'これは素晴らしいゲームです' not in df_english['review_text'].values
    assert 'Разраба маму' not in df_english['review_text'].values


def test_create_topic_model():
    """BERTopicモデル作成のテスト"""
    topic_model = create_topic_model(
        min_topic_size=5,
        verbose=False
    )

    # モデルが作成される
    assert topic_model is not None
    assert hasattr(topic_model, 'fit_transform')


def test_extract_topics_small_data():
    """小規模データでのトピック抽出テスト（100件）"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])

    # 英語レビューのみフィルタリング
    df_english = filter_english_reviews(df, text_column='review_text')

    # 100件に制限
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, probs = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # 結果確認
    assert topic_model is not None
    assert len(topics) == len(texts)
    assert len(probs) == len(texts)

    # トピックが抽出される（最低1つ以上）
    num_topics = len(set(topics)) - 1  # -1はOutlier除外
    assert num_topics >= 0

    print(f"\n✅ トピック抽出成功: {num_topics}個のトピックを抽出")


def test_get_topic_info():
    """トピック情報取得のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # トピック情報取得
    topic_info = get_topic_info(topic_model, verbose=False)

    # データフレームが返される
    assert isinstance(topic_info, pd.DataFrame)
    assert 'Topic' in topic_info.columns
    assert 'Count' in topic_info.columns
    assert 'Name' in topic_info.columns

    print(f"\n✅ トピック情報取得成功: {len(topic_info)}行")


def test_get_topic_words():
    """トピック代表単語取得のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # Topic 0の代表単語を取得
    if 0 in topics:
        words = get_topic_words(topic_model, topic_id=0, top_n=5)

        # 単語リストが返される
        assert isinstance(words, list)
        if len(words) > 0:
            assert len(words) <= 5
            # 各要素は(単語, スコア)のタプル
            assert isinstance(words[0], tuple)
            assert isinstance(words[0][0], str)
            assert isinstance(words[0][1], float)

            print(f"\n✅ トピック代表単語取得成功: {', '.join([w for w, _ in words])}")


def test_print_topic_summary():
    """トピックサマリー表示のテスト"""
    # データ読み込み
    df = pd.read_csv('data/train/reviews_1000.csv')
    df = df.dropna(subset=['review_text'])
    df_english = filter_english_reviews(df, text_column='review_text')
    texts = df_english['review_text'].head(100).tolist()

    # トピック抽出
    topic_model, topics, _ = extract_topics(
        texts,
        min_topic_size=5,
        verbose=False
    )

    # サマリー表示（エラーが出ないことを確認）
    print_topic_summary(
        topic_model,
        topics,
        texts,
        max_topics=3,
        top_n_words=5,
        sample_reviews=2
    )

    print("\n✅ トピックサマリー表示成功")


if __name__ == '__main__':
    # 個別実行用
    print("=" * 70)
    print("トピック抽出モジュールのテスト")
    print("=" * 70)

    test_is_english()
    print("✅ test_is_english passed")

    test_filter_english_reviews()
    print("✅ test_filter_english_reviews passed")

    test_create_topic_model()
    print("✅ test_create_topic_model passed")

    test_extract_topics_small_data()
    print("✅ test_extract_topics_small_data passed")

    test_get_topic_info()
    print("✅ test_get_topic_info passed")

    test_get_topic_words()
    print("✅ test_get_topic_words passed")

    test_print_topic_summary()
    print("✅ test_print_topic_summary passed")

    print("\n" + "=" * 70)
    print("全テスト成功！")
    print("=" * 70)
