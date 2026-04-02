"""
Data preprocessing module for Steam reviews

This module provides functions to clean and transform Steam API responses
into structured data suitable for sentiment analysis.
"""

import re
import pandas as pd
from typing import List, Dict


def clean_review_text(text: str) -> str:
    """
    Clean review text for sentiment analysis

    Args:
        text: Raw review text from Steam API

    Returns:
        Cleaned text with:
            - HTML tags removed
            - Multiple whitespaces normalized
            - Leading/trailing whitespace removed

    Example:
        >>> clean_review_text("<b>Great game!</b>  ")
        'Great game!'
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace (replace multiple spaces/newlines with single space)
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def steam_reviews_to_dataframe(reviews: List[Dict]) -> pd.DataFrame:
    """
    Convert Steam API review response to pandas DataFrame

    Args:
        reviews: List of review dictionaries from Steam API
                 Each dict should have: review_text, voted_up, votes_up, language, etc.

    Returns:
        DataFrame with columns:
            - review_text: Cleaned review text
            - label: Binary label (1=Positive/Recommended, 0=Negative/Not Recommended)
            - votes_up: Number of upvotes
            - language: Review language
            - timestamp_created: Review creation timestamp
            - author: Author's Steam ID

    Example:
        >>> reviews = [{'review_text': 'Great!', 'voted_up': True, ...}]
        >>> df = steam_reviews_to_dataframe(reviews)
        >>> df['label'].iloc[0]
        1
    """
    if not reviews:
        return pd.DataFrame()

    # Extract relevant fields
    data = []
    for review in reviews:
        data.append({
            'review_text': clean_review_text(review.get('review_text', '')),
            'label': 1 if review.get('voted_up', False) else 0,  # 1=Positive, 0=Negative
            'votes_up': review.get('votes_up', 0),
            'language': review.get('language', ''),
            'timestamp_created': review.get('timestamp_created', 0),
            'author': review.get('author', ''),
        })

    df = pd.DataFrame(data)

    # Filter out empty reviews
    df = df[df['review_text'].str.len() > 0]

    return df


def balance_dataset(df: pd.DataFrame, n_samples_per_class: int = None) -> pd.DataFrame:
    """
    Balance dataset by sampling equal number of positive and negative reviews

    Args:
        df: DataFrame with 'label' column (0 or 1)
        n_samples_per_class: Number of samples per class (if None, use minimum class size)

    Returns:
        Balanced DataFrame with equal number of positive and negative reviews

    Example:
        >>> df = pd.DataFrame({'label': [1, 1, 1, 0], 'text': ['a', 'b', 'c', 'd']})
        >>> balanced = balance_dataset(df)
        >>> balanced['label'].value_counts()
        0    1
        1    1
    """
    if df.empty:
        return df

    # Count samples per class
    positive_df = df[df['label'] == 1]
    negative_df = df[df['label'] == 0]

    n_positive = len(positive_df)
    n_negative = len(negative_df)

    # Determine sample size
    if n_samples_per_class is None:
        n_samples_per_class = min(n_positive, n_negative)

    if n_samples_per_class <= 0:
        return pd.DataFrame()

    # Sample equal number from each class
    positive_sampled = positive_df.sample(n=min(n_samples_per_class, n_positive), random_state=42)
    negative_sampled = negative_df.sample(n=min(n_samples_per_class, n_negative), random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([positive_sampled, negative_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return balanced_df


def prepare_validation_dataset(
    positive_reviews: List[Dict],
    negative_reviews: List[Dict],
    n_per_class: int = 50
) -> pd.DataFrame:
    """
    Prepare balanced validation dataset from positive and negative reviews

    Args:
        positive_reviews: List of positive review dicts from Steam API
        negative_reviews: List of negative review dicts from Steam API
        n_per_class: Number of samples per class (default: 50)

    Returns:
        Balanced DataFrame ready for sentiment analysis validation

    Example:
        >>> pos = [{'review_text': 'Good!', 'voted_up': True, ...}]
        >>> neg = [{'review_text': 'Bad!', 'voted_up': False, ...}]
        >>> df = prepare_validation_dataset(pos, neg, n_per_class=10)
        >>> len(df)
        20
        >>> df['label'].value_counts()
        0    10
        1    10
    """
    # Convert to DataFrames
    df_positive = steam_reviews_to_dataframe(positive_reviews)
    df_negative = steam_reviews_to_dataframe(negative_reviews)

    # Combine
    df = pd.concat([df_positive, df_negative])

    # Balance
    df_balanced = balance_dataset(df, n_samples_per_class=n_per_class)

    return df_balanced
