"""
Steamレビューのデータ前処理モジュール

Steam APIレスポンスをクリーニングし、感情分析に適した
構造化データに変換する機能を提供します。
"""

import re
import pandas as pd
from typing import List, Dict


def clean_review_text(text: str) -> str:
    """
    感情分析用にレビューテキストをクリーニング

    Args:
        text: Steam APIから取得した生のレビューテキスト

    Returns:
        クリーニングされたテキスト:
            - HTMLタグ除去済み
            - 連続する空白を正規化
            - 先頭・末尾の空白を削除

    Example:
        >>> clean_review_text("<b>Great game!</b>  ")
        'Great game!'
    """
    if not isinstance(text, str):
        return ""

    # HTMLタグを除去
    text = re.sub(r'<[^>]+>', '', text)

    # 空白を正規化（複数のスペース/改行を単一スペースに）
    text = re.sub(r'\s+', ' ', text)

    # 先頭・末尾の空白を削除
    text = text.strip()

    return text


def steam_reviews_to_dataframe(reviews: List[Dict]) -> pd.DataFrame:
    """
    Steam APIレビューレスポンスをpandas DataFrameに変換

    Args:
        reviews: Steam APIから取得したレビューのdictリスト
                 各dictは以下を含む: review_text, voted_up, votes_up, language等

    Returns:
        以下のカラムを持つDataFrame:
            - review_text: クリーニング済みレビュー本文
            - game_rating: バイナリラベル（1=おすすめ, 0=おすすめしない）
            - review_helpfulness: 高評価数
            - language: レビューの言語
            - posted_date: レビュー作成timestamp
            - user_id: 投稿者のSteam ID

    Example:
        >>> reviews = [{'review_text': 'Great!', 'voted_up': True, ...}]
        >>> df = steam_reviews_to_dataframe(reviews)
        >>> df['game_rating'].iloc[0]
        1
    """
    if not reviews:
        return pd.DataFrame()

    # 関連フィールドを抽出
    data = []
    for review in reviews:
        data.append({
            'review_text': clean_review_text(review.get('review_text', '')),
            'game_rating': 1 if review.get('voted_up', False) else 0,  # 1=おすすめ, 0=おすすめしない
            'review_helpfulness': review.get('votes_up', 0),  # 高評価数
            'language': review.get('language', ''),
            'posted_date': review.get('timestamp_created', 0),
            'user_id': review.get('author', ''),
        })

    df = pd.DataFrame(data)

    # 空のレビューを除外
    df = df[df['review_text'].str.len() > 0]

    return df


def balance_dataset(df: pd.DataFrame, n_samples_per_class: int = None) -> pd.DataFrame:
    """
    positive/negativeレビューを同数samplingしてdatasetをバランス化

    Args:
        df: 'game_rating'カラム（0または1）を持つDataFrame
        n_samples_per_class: クラスごとのsample数（Noneの場合、最小クラスサイズを使用）

    Returns:
        positive/negativeが同数のbalanced DataFrame

    Example:
        >>> df = pd.DataFrame({'game_rating': [1, 1, 1, 0], 'review_text': ['a', 'b', 'c', 'd']})
        >>> balanced = balance_dataset(df)
        >>> balanced['game_rating'].value_counts()
        0    1
        1    1
    """
    if df.empty:
        return df

    # クラスごとのsample数をカウント
    positive_df = df[df['game_rating'] == 1]
    negative_df = df[df['game_rating'] == 0]

    n_positive = len(positive_df)
    n_negative = len(negative_df)

    # sample数を決定
    if n_samples_per_class is None:
        n_samples_per_class = min(n_positive, n_negative)

    if n_samples_per_class <= 0:
        return pd.DataFrame()

    # 各クラスから同数sampling
    positive_sampled = positive_df.sample(n=min(n_samples_per_class, n_positive), random_state=42)
    negative_sampled = negative_df.sample(n=min(n_samples_per_class, n_negative), random_state=42)

    # 結合してシャッフル
    balanced_df = pd.concat([positive_sampled, negative_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def prepare_validation_dataset(
    positive_reviews: List[Dict],
    negative_reviews: List[Dict],
    n_per_class: int = 50
) -> pd.DataFrame:
    """
    positive/negativeレビューからbalancedな検証用datasetを作成

    Args:
        positive_reviews: Steam APIから取得したpositiveレビューのdictリスト
        negative_reviews: Steam APIから取得したnegativeレビューのdictリスト
        n_per_class: クラスごとのsample数（デフォルト: 50）

    Returns:
        感情分析検証用のbalanced DataFrame

    Example:
        >>> pos = [{'review_text': 'Good!', 'voted_up': True, ...}]
        >>> neg = [{'review_text': 'Bad!', 'voted_up': False, ...}]
        >>> df = prepare_validation_dataset(pos, neg, n_per_class=10)
        >>> len(df)
        20
        >>> df['game_rating'].value_counts()
        0    10
        1    10
    """
    # DataFrameに変換
    df_positive = steam_reviews_to_dataframe(positive_reviews)
    df_negative = steam_reviews_to_dataframe(negative_reviews)

    # 結合
    df = pd.concat([df_positive, df_negative])

    # バランス化
    df_balanced = balance_dataset(df, n_samples_per_class=n_per_class)

    return df_balanced
