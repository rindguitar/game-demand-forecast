"""
Steam API data collector for game reviews

This module provides functions to collect game reviews from Steam API.
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
    Collect reviews from Steam API

    Args:
        app_id: Steam game ID (e.g., 730 for CS:GO, 570 for Dota 2)
        language: 'english' or 'japanese' or 'all'
        review_type: 'positive', 'negative', or 'all'
        num: Number of reviews to collect
        max_retries: Maximum number of API retry attempts

    Returns:
        List of review dictionaries containing:
            - review_text: Review content
            - voted_up: True=Recommended, False=Not Recommended
            - votes_up: Number of upvotes
            - language: Review language
            - timestamp_created: Review creation timestamp
            - author: Author's Steam ID

    Raises:
        ValueError: If app_id is invalid or parameters are incorrect
        requests.exceptions.RequestException: If API request fails

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

    # Steam API endpoint
    base_url = "https://store.steampowered.com/appreviews/"

    # API parameters
    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',  # Get recent reviews
        'review_type': review_type,
        'purchase_type': 'all',
        'num_per_page': min(100, num),  # API max is 100 per request
    }

    reviews = []
    cursor = '*'  # Initial cursor

    while len(reviews) < num:
        params['cursor'] = cursor

        # API request with retries
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
                time.sleep(1)  # Wait before retry

        # Check API response
        if data.get('success') != 1:
            raise requests.exceptions.RequestException(
                f"Steam API returned error: {data.get('error', 'Unknown error')}"
            )

        # Extract reviews
        api_reviews = data.get('reviews', [])
        if not api_reviews:
            break  # No more reviews available

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

        # Get next cursor
        cursor = data.get('cursor')
        if not cursor:
            break  # No more pages

        # Rate limiting: respect Steam API
        time.sleep(0.5)

    return reviews


def collect_balanced_reviews(
    app_id: int,
    language: str = 'english',
    n_positive: int = 50,
    n_negative: int = 50
) -> Dict[str, List[Dict]]:
    """
    Collect balanced positive and negative reviews for validation

    Args:
        app_id: Steam game ID
        language: 'english' or 'japanese'
        n_positive: Number of positive reviews (Recommended)
        n_negative: Number of negative reviews (Not Recommended)

    Returns:
        Dictionary with keys 'positive' and 'negative', each containing list of reviews

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
