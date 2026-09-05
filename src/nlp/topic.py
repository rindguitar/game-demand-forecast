"""
トピック抽出モジュール

BERTopicを使用してゲームレビューからトピック（ゲーム要素）を抽出する。
"""

from typing import List, Tuple, Dict, Optional
import pandas as pd
import os
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


def load_proper_nouns(path: str) -> List[str]:
    """
    除去する固有名詞のリストを読む（1行1語・# はコメント）

    ロスター外の開発元名・タイトル名を置く。収集対象ゲームの名前は台帳から
    自動で作れるので、ここには含めない。
    """
    if not path:
        return []
    if not os.path.exists(path):
        # 黙って空を返すと「除去したつもり」で進んでしまうので必ず知らせる
        print(f"⚠️ 固有名詞のファイルが見つかりません: {path}（追加の除去は行われません）")
        return []
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f
                if line.strip() and not line.lstrip().startswith('#')]


def remove_game_names(df: pd.DataFrame, text_column: str = 'review_text',
                      game_name_column: str = 'game_name',
                      all_games: bool = False,
                      extra_words: Optional[List[str]] = None) -> pd.DataFrame:
    """
    各レビューから自ゲームのタイトル単語を除去

    自己言及問題（GTA Onlineのレビューに"gta online"が含まれる等）を防ぐため、
    各レビューのgame_name列の単語をレビューテキストから除去する。

    除去フィルター:
        - 2文字以下の単語（"v", "of", "a"等）
        - 数字のみの単語（"2077", "3", "5"等）

    all_games=True にすると、**全ゲームの名前を全レビューから**除去する。自分の名前
    だけを消すと他ゲームへの言及が残り、トピックがそのゲーム専用になる（実測: Starfield
    のレビューに bethesda / skyrim / fallout が残り、1,409件のトピックが Starfield 95%
    になった）。extra_words にはロスター外の固有名詞（開発元名など）を渡す。

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

    # 全レビューから除去する場合は、語をまとめて1つの正規表現にして1回で走らせる
    # （ゲームごとに何十回も全文を走査すると、数十万件では時間が桁で変わる）
    if all_games or extra_words:
        words = {w.lower() for w in (extra_words or []) if len(w) > 3}
        for game_name in df_copy[game_name_column].dropna().unique():
            words.update(_get_game_words(game_name))
        if words:
            # 長い語から順に消す（"dark souls" を "souls" より先に処理する）
            ordered = sorted(words, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(w) for w in ordered) + r')\b'
            df_copy[text_column] = (df_copy[text_column].astype(str)
                                    .str.replace(pattern, ' ', regex=True, case=False)
                                    .str.replace(r'\s+', ' ', regex=True)
                                    .str.strip())
        print(f"固有名詞の除去完了: {len(words)}語を全レビューから除去")
        return df_copy

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


def assign_topics(
    topic_model: BERTopic,
    texts: List[str],
    verbose: bool = True
) -> Tuple[List[int], List[float]]:
    """
    学習済みモデルで、テキストにトピックを割り当てる（transform）

    学習（fit）と割り当て（transform）を分けるのは、次元削減とクラスタリングが件数に
    弱いため。重い処理は一部のデータで済ませ、割り当ては「既にできている塊のどれに
    近いか」を見るだけなので全件に掛けられる（詳細は Wiki「トピック抽出」）。

    Args:
        topic_model: 学習済みBERTopicモデル
        texts: 割り当て対象のテキスト
        verbose: 詳細ログ表示

    Returns:
        (トピックIDのリスト, 確率のリスト)
    """
    if verbose:
        print(f"トピック割り当て（transform）: {len(texts):,}件")

    topics, probabilities = topic_model.transform(texts)
    topics = [int(t) for t in topics]

    if verbose:
        outlier = topics.count(-1)
        print(f"割り当て完了: {len(set(topics) - {-1})}トピック / "
              f"Outlier {outlier:,}件 ({outlier / len(topics) * 100:.1f}%)")

    return topics, probabilities


def save_topic_model(
    topic_model: BERTopic,
    path: str,
    embedding_model_name: str = 'all-MiniLM-L6-v2'
) -> None:
    """
    学習済みモデルを保存する

    埋め込みモデルは本体を書き出さず名前だけ記録する（サイズが大きく、
    名前があれば再取得できるため）。
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    topic_model.save(
        path,
        serialization='safetensors',
        save_ctfidf=True,
        save_embedding_model=embedding_model_name,
    )


def load_topic_model(
    path: str,
    embedding_model_name: str = 'all-MiniLM-L6-v2'
) -> BERTopic:
    """保存済みモデルを読み込む（再学習を避けるため）"""
    return BERTopic.load(path, embedding_model=embedding_model_name)


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
