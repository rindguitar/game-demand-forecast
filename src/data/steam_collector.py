"""
Steam APIゲームレビュー収集モジュール

Steam APIからゲームレビューを収集する機能を提供します。
"""

import requests
import time
from typing import List, Dict, Optional


def get_steam_reviews(
    app_id: int,
    language: str = 'english',
    review_type: str = 'all',
    num: int = 100,
    max_retries: int = 3
) -> List[Dict]:
    """
    Steam APIからレビューを収集

    Args:
        app_id: SteamゲームID（例: 730=CS:GO, 570=Dota 2）
        language: 'english', 'japanese', 'all'のいずれか
        review_type: 'positive', 'negative', 'all'のいずれか
        num: 収集するレビュー数
        max_retries: API retry試行回数の上限

    Returns:
        レビューのdictリスト、各dictは以下を含む:
            - review_text: レビュー本文
            - voted_up: True=おすすめ, False=おすすめしない
            - votes_up: 高評価数
            - language: レビューの言語
            - timestamp_created: レビュー作成時刻
            - author: 投稿者のSteam ID

    Raises:
        ValueError: app_idまたはパラメータが不正な場合
        requests.exceptions.RequestException: APIリクエスト失敗時

    Example:
        >>> reviews = get_steam_reviews(app_id=730, language='english', num=100)
        >>> print(f"Collected {len(reviews)} reviews")
    """
    if app_id <= 0:
        raise ValueError(f"Invalid app_id: {app_id}")

    if language not in ['english', 'japanese', 'all']:
        raise ValueError(f"Invalid language: {language}. Must be 'english', 'japanese', or 'all'")

    if review_type not in ['positive', 'negative', 'all']:
        raise ValueError(f"Invalid review_type: {review_type}. Must be 'positive', 'negative', or 'all'")

    if num <= 0:
        raise ValueError(f"Invalid num: {num}. Must be positive")

    # Steam APIエンドポイント
    base_url = "https://store.steampowered.com/appreviews/"

    # APIパラメータ
    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',  # 最新レビューを取得
        'review_type': review_type,
        'purchase_type': 'all',
        'num_per_page': min(100, num),  # APIの上限は1リクエスト100件
    }

    reviews = []
    cursor = '*'  # 初期cursor

    while len(reviews) < num:
        params['cursor'] = cursor

        # retry付きAPIリクエスト
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{base_url}{app_id}", params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise requests.exceptions.RequestException(
                        f"Failed to fetch reviews after {max_retries} attempts: {e}"
                    )
                time.sleep(1)  # retry前の待機

        # APIレスポンス確認
        if data.get('success') != 1:
            raise requests.exceptions.RequestException(
                f"Steam API returned error: {data.get('error', 'Unknown error')}"
            )

        # レビュー抽出
        api_reviews = data.get('reviews', [])
        if not api_reviews:
            break  # これ以上レビューなし

        for review in api_reviews:
            if len(reviews) >= num:
                break

            reviews.append({
                'review_text': review.get('review', ''),
                'voted_up': review.get('voted_up', False),
                'votes_up': review.get('votes_up', 0),
                'language': review.get('language', ''),
                'timestamp_created': review.get('timestamp_created', 0),
                'author': review.get('author', {}).get('steamid', ''),
            })

        # 次のcursorを取得
        cursor = data.get('cursor')
        if not cursor:
            break  # ページなし

        # Rate limiting: Steam APIを尊重
        time.sleep(0.5)

    return reviews


def collect_balanced_reviews(
    app_id: int,
    language: str = 'english',
    n_positive: int = 50,
    n_negative: int = 50
) -> Dict[str, List[Dict]]:
    """
    検証用にbalancedなpositive/negativeレビューを収集

    Args:
        app_id: SteamゲームID
        language: 'english'または'japanese'
        n_positive: positiveレビュー数（おすすめ）
        n_negative: negativeレビュー数（おすすめしない）

    Returns:
        'positive'と'negative'をkeyとするdict、各valueはレビューのリスト

    Example:
        >>> reviews = collect_balanced_reviews(app_id=730, language='english')
        >>> print(f"Positive: {len(reviews['positive'])}, Negative: {len(reviews['negative'])}")
    """
    print(f"Collecting {n_positive} positive reviews...")
    positive_reviews = get_steam_reviews(
        app_id=app_id,
        language=language,
        review_type='positive',
        num=n_positive
    )

    print(f"Collecting {n_negative} negative reviews...")
    negative_reviews = get_steam_reviews(
        app_id=app_id,
        language=language,
        review_type='negative',
        num=n_negative
    )

    return {
        'positive': positive_reviews,
        'negative': negative_reviews
    }
