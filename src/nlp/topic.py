"""
トピック抽出モジュール

BERTopicを使用してゲームレビューからトピック（ゲーム要素）を抽出する。
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def is_english(text: str) -> bool:
    """
    テキストが英語かどうかを判定

    Args:
        text: 判定するテキスト

    Returns:
        英語の場合True

    Example:
        >>> is_english("This is a great game!")
        True
        >>> is_english("これは素晴らしいゲームです")
        False
    """
    # ASCIIと基本的な記号のみで構成されているか
    return bool(re.match(r'^[\x00-\x7F]+$', text))


def filter_english_reviews(df: pd.DataFrame, text_column: str = 'review_text') -> pd.DataFrame:
    """
    英語レビューのみをフィルタリング

    Args:
        df: レビューデータフレーム
        text_column: テキストカラム名

    Returns:
        英語レビューのみのデータフレーム

    Example:
        >>> df_filtered = filter_english_reviews(df)
        >>> print(f"英語レビュー: {len(df_filtered)}件")
    """
    df_copy = df.copy()
    df_copy['is_english'] = df_copy[text_column].apply(is_english)
    df_english = df_copy[df_copy['is_english']].copy()

    print(f"元データ: {len(df)}件")
    print(f"英語レビュー: {len(df_english)}件")
    print(f"非英語レビュー: {len(df) - len(df_english)}件（除外）")

    return df_english.drop(columns=['is_english'])


def remove_game_names(df: pd.DataFrame, text_column: str = 'review_text', game_name_column: str = 'game_name') -> pd.DataFrame:
    """
    各レビューから自ゲームのタイトル単語を除去

    自己言及問題（GTA Onlineのレビューに"gta online"が含まれる等）を防ぐため、
    各レビューのgame_name列の単語をレビューテキストから除去する。

    除去フィルター:
        - 2文字以下の単語（"v", "of", "a"等）
        - 数字のみの単語（"2077", "3", "5"等）

    Args:
        df: レビューデータフレーム
        text_column: テキストカラム名
        game_name_column: ゲーム名カラム名

    Returns:
        ゲーム名除去済みのデータフレーム
    """
    df_copy = df.copy()

    def _get_game_words(game_name: str) -> List[str]:
        """ゲーム名から除去対象の単語リストを生成"""
        words = re.findall(r'[a-zA-Z0-9]+', game_name.lower())
        return [w for w in words if len(w) > 3 and not w.isnumeric()]

    def _remove_words(text: str, words: List[str]) -> str:
        """テキストから単語リストを除去"""
        for word in words:
            text = re.sub(rf'\b{re.escape(word)}\b', '', text, flags=re.IGNORECASE)
        return text.strip()

    for game_name, group_idx in df_copy.groupby(game_name_column).groups.items():
        game_words = _get_game_words(game_name)
        if not game_words:
            continue
        df_copy.loc[group_idx, text_column] = df_copy.loc[group_idx, text_column].apply(
            lambda text: _remove_words(str(text), game_words)
        )

    print(f"ゲーム名除去完了")
    for game_name in df_copy[game_name_column].unique():
        words = _get_game_words(game_name)
        print(f"  {game_name}: {words}")

    return df_copy


def create_topic_model(
    min_topic_size: int = 10,
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    verbose: bool = True
) -> BERTopic:
    """
    BERTopicモデルを作成

    Args:
        min_topic_size: 最小トピックサイズ
        embedding_model_name: 埋め込みモデル名
        ngram_range: n-gram範囲（デフォルト: (1, 2) = 単語 + 2単語の組み合わせ）
        min_df: 最小出現頻度
        verbose: 詳細ログ表示

    Returns:
        BERTopicモデル

    Example:
        >>> topic_model = create_topic_model(min_topic_size=10)
    """
    if verbose:
        print("=" * 70)
        print("BERTopicモデル作成")
        print("=" * 70)
        print(f"min_topic_size: {min_topic_size}")
        print(f"embedding_model: {embedding_model_name}")
        print(f"ngram_range: {ngram_range}")
        print(f"min_df: {min_df}")

    # CountVectorizer設定（ストップワード除去・n-gram・3文字以上の単語のみ）
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=ngram_range,
        min_df=min_df,
        token_pattern=r'(?u)\b[a-zA-Z]{4,}\b'  # 4文字以上のアルファベットのみ対象
    )

    # 埋め込みモデル
    embedding_model = SentenceTransformer(embedding_model_name)

    # BERTopicモデル
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        verbose=False
    )

    if verbose:
        print("モデル作成完了")
        print("=" * 70)

    return topic_model


def extract_topics(
    texts: List[str],
    topic_model: Optional[BERTopic] = None,
    min_topic_size: int = 10,
    verbose: bool = True
) -> Tuple[BERTopic, List[int], List[float]]:
    """
    テキストからトピックを抽出

    Args:
        texts: レビューテキストのリスト
        topic_model: 既存のBERTopicモデル（Noneの場合は新規作成）
        min_topic_size: 最小トピックサイズ（topic_modelがNoneの場合のみ使用）
        verbose: 詳細ログ表示

    Returns:
        (topic_model, topics, probabilities)のタプル
        - topic_model: 学習済みBERTopicモデル
        - topics: 各テキストのトピックID（-1はOutlier）
        - probabilities: 各テキストのトピック確率

    Example:
        >>> topic_model, topics, probs = extract_topics(texts)
        >>> print(f"抽出されたトピック数: {len(set(topics)) - 1}")
    """
    if topic_model is None:
        topic_model = create_topic_model(
            min_topic_size=min_topic_size,
            verbose=verbose
        )

    if verbose:
        print("\n" + "=" * 70)
        print("トピック抽出実行")
        print("=" * 70)
        print(f"レビュー数: {len(texts)}")

    # トピック抽出
    topics, probabilities = topic_model.fit_transform(texts)

    num_topics = len(set(topics)) - 1  # -1はOutlier除外
    outlier_count = len([t for t in topics if t == -1])

    if verbose:
        print(f"抽出されたトピック数: {num_topics}")
        print(f"Outlier数: {outlier_count} ({outlier_count/len(topics)*100:.1f}%)")
        print("=" * 70)

    return topic_model, topics, probabilities


def get_topic_info(topic_model: BERTopic, verbose: bool = True) -> pd.DataFrame:
    """
    トピック情報を取得

    Args:
        topic_model: 学習済みBERTopicモデル
        verbose: 詳細ログ表示

    Returns:
        トピック情報のデータフレーム（Topic, Count, Name列を含む）

    Example:
        >>> topic_info = get_topic_info(topic_model)
        >>> print(topic_info.head())
    """
    topic_info = topic_model.get_topic_info()

    if verbose:
        print("\n" + "=" * 70)
        print("トピック情報")
        print("=" * 70)
        print(topic_info[['Topic', 'Count', 'Name']])
        print("=" * 70)

    return topic_info


def get_topic_words(
    topic_model: BERTopic,
    topic_id: int,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    特定トピックの代表単語を取得

    Args:
        topic_model: 学習済みBERTopicモデル
        topic_id: トピックID
        top_n: 取得する単語数

    Returns:
        (単語, スコア)のリスト

    Example:
        >>> words = get_topic_words(topic_model, topic_id=0, top_n=10)
        >>> print(", ".join([w for w, _ in words]))
    """
    return topic_model.get_topic(topic_id)[:top_n]


def print_topic_summary(
    topic_model: BERTopic,
    topics: List[int],
    texts: List[str],
    max_topics: int = 10,
    top_n_words: int = 10,
    sample_reviews: int = 3
):
    """
    トピック抽出結果のサマリーを表示

    Args:
        topic_model: 学習済みBERTopicモデル
        topics: 各テキストのトピックID
        texts: レビューテキストのリスト
        max_topics: 表示する最大トピック数
        top_n_words: トピックごとの代表単語数
        sample_reviews: トピックごとのサンプルレビュー数

    Example:
        >>> print_topic_summary(topic_model, topics, texts)
    """
    num_topics = len(set(topics)) - 1
    outlier_count = len([t for t in topics if t == -1])

    print("\n" + "=" * 80)
    print("📊 トピック抽出結果サマリー")
    print("=" * 80)
    print(f"抽出トピック数: {num_topics}")
    print(f"処理レビュー数: {len(texts)}")
    print(f"Outlier（未分類）: {outlier_count}件 ({outlier_count/len(topics)*100:.1f}%)")
    print("=" * 80)

    # トピック情報を取得（Name列を含む）
    topic_info = topic_model.get_topic_info()

    # トピックを件数順にソート
    topic_counts = {}
    for topic_id in range(num_topics):
        topic_counts[topic_id] = len([t for t in topics if t == topic_id])

    sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)

    # 上位トピックを表示
    for rank, (topic_id, count) in enumerate(sorted_topics[:max_topics], 1):
        topic_words = get_topic_words(topic_model, topic_id, top_n_words)
        if not topic_words:
            continue

        # 代表単語（上位5個のみ表示）
        top_words = ", ".join([word for word, _ in topic_words[:5]])

        # トピック名を取得
        topic_name = topic_info[topic_info['Topic'] == topic_id]['Name'].values
        topic_label = topic_name[0] if len(topic_name) > 0 else "Unknown"

        print(f"\n┌─ Topic {topic_id} (#{rank}) ─ {count}件 ({count/len(topics)*100:.1f}%) ─")
        print(f"│ トピック名: {topic_label}")
        print(f"│ 代表単語: {top_words}")

        # サンプルレビュー
        topic_reviews = [texts[i] for i, t in enumerate(topics) if t == topic_id]
        print(f"│ サンプル:")
        for i, review in enumerate(topic_reviews[:sample_reviews]):
            # レビューの最初の80文字を表示
            review_text = review.replace('\n', ' ')[:80]
            print(f"│   {i+1}. {review_text}...")
        print("└" + "─" * 78)

    print("\n" + "=" * 80)
    print("トピック詳細表示完了")
    print("=" * 80)
