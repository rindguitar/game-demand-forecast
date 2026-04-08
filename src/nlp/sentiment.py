"""
事前学習済みモデルを使用した感情分析モジュール

Hugging Face Transformersの事前学習済みモデルを使用して
感情分析機能を提供します。
"""

from typing import List, Dict, Union
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


def analyze_sentiment(
    texts: Union[str, List[str]],
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
    device: int = -1
) -> Union[Dict, List[Dict]]:
    """
    事前学習済みモデルを使用してテキストの感情分析を実行

    Args:
        texts: 分析するテキスト（単一文字列またはリスト）
        model_name: Hugging Faceモデル名
            - 英語: "distilbert-base-uncased-finetuned-sst-2-english" (デフォルト)
            - 日本語: "daigo/bert-base-japanese-sentiment"
        batch_size: 複数テキスト処理時のバッチサイズ
        device: 使用デバイス（-1=CPU, 0=GPU）

    Returns:
        辞書または辞書のリスト（キー: label, score）
            - label: "POSITIVE" または "NEGATIVE"
            - score: 信頼度スコア (0.0-1.0)

    Example:
        >>> # 単一テキスト
        >>> result = analyze_sentiment("This game is amazing!")
        >>> result['label']
        'POSITIVE'

        >>> # 複数テキスト
        >>> texts = ["Great game!", "Terrible experience"]
        >>> results = analyze_sentiment(texts)
        >>> [r['label'] for r in results]
        ['POSITIVE', 'NEGATIVE']
    """
    # パイプライン初期化
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        device=device
    )

    # 感情分析実行
    if isinstance(texts, str):
        # 単一テキスト
        return sentiment_pipeline(texts)[0]
    else:
        # 複数テキスト（バッチ処理）
        return sentiment_pipeline(texts, batch_size=batch_size)


def predict_sentiment_labels(
    texts: List[str],
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
    device: int = -1
) -> List[int]:
    """
    テキストの感情ラベル（0または1）を予測

    Args:
        texts: 分析するテキストのリスト
        model_name: Hugging Faceモデル名
        batch_size: バッチサイズ
        device: 使用デバイス（-1=CPU, 0=GPU）

    Returns:
        2値ラベルのリスト (1=POSITIVE, 0=NEGATIVE)

    Example:
        >>> texts = ["Great game!", "Terrible experience", "Amazing!"]
        >>> labels = predict_sentiment_labels(texts)
        >>> labels
        [1, 0, 1]
    """
    # 感情分析実行
    results = analyze_sentiment(texts, model_name, batch_size, device)

    # 2値ラベルに変換
    labels = []
    for result in results:
        # POSITIVE -> 1, NEGATIVE -> 0
        label = 1 if result['label'] == 'POSITIVE' else 0
        labels.append(label)

    return labels


def analyze_steam_reviews(
    reviews_df,
    text_column: str = 'review_text',
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    batch_size: int = 32,
    device: int = -1
) -> List[int]:
    """
    SteamレビューDataFrameの感情分析を実行

    Args:
        reviews_df: レビューテキストを含むDataFrame
        text_column: レビューテキストのカラム名
        model_name: Hugging Faceモデル名
        batch_size: バッチサイズ
        device: 使用デバイス（-1=CPU, 0=GPU）

    Returns:
        予測ラベルのリスト (1=POSITIVE, 0=NEGATIVE)

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'review_text': ['Great!', 'Bad!']})
        >>> predictions = analyze_steam_reviews(df)
        >>> predictions
        [1, 0]
    """
    texts = reviews_df[text_column].tolist()
    return predict_sentiment_labels(texts, model_name, batch_size, device)


def get_recommended_model(language: str = 'english') -> str:
    """
    言語に応じた推奨事前学習済みモデルを取得

    Args:
        language: 'english' または 'japanese'

    Returns:
        Hugging Faceモデル名

    Example:
        >>> get_recommended_model('english')
        'distilbert-base-uncased-finetuned-sst-2-english'
        >>> get_recommended_model('japanese')
        'daigo/bert-base-japanese-sentiment'
    """
    models = {
        'english': 'distilbert-base-uncased-finetuned-sst-2-english',
        'japanese': 'daigo/bert-base-japanese-sentiment'
    }

    language = language.lower()
    if language not in models:
        raise ValueError(f"Unsupported language: {language}. Choose 'english' or 'japanese'.")

    return models[language]


def check_gpu_available() -> bool:
    """
    PyTorchでGPUが利用可能かチェック

    Returns:
        CUDA GPUが利用可能な場合True、それ以外False

    Example:
        >>> check_gpu_available()
        True  # GPUが利用可能な場合
    """
    return torch.cuda.is_available()


def get_device() -> int:
    """
    感情分析に推奨されるデバイスを取得

    Returns:
        GPUが利用可能な場合0、CPUの場合-1

    Example:
        >>> device = get_device()
        >>> device
        0  # GPUが利用可能な場合
    """
    return 0 if check_gpu_available() else -1
