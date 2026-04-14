"""
PyTorch Dataset for Steam レビュー感情分析

Steam レビューデータをPyTorchで学習するためのDatasetとDataLoaderを提供。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Tuple
import pandas as pd


class SteamReviewDataset(Dataset):
    """
    Steam レビューデータセット（PyTorch用）

    Args:
        texts: レビューテキストのリスト
        labels: ラベルのリスト（0=Negative, 1=Positive）
        tokenizer: Hugging Face Tokenizer
        max_length: 最大トークン長（デフォルト128）

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        >>> dataset = SteamReviewDataset(
        ...     texts=["Great game!", "Terrible"],
        ...     labels=[1, 0],
        ...     tokenizer=tokenizer
        ... )
        >>> print(len(dataset))
        2
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: AutoTokenizer,
        max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # バリデーション
        if len(texts) != len(labels):
            raise ValueError(
                f"textsとlabelsの長さが一致しません: {len(texts)} != {len(labels)}"
            )

    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        """
        指定されたインデックスのデータを返す

        Returns:
            dict with keys:
                - input_ids: トークンID（torch.Tensor）
                - attention_mask: attentionマスク（torch.Tensor）
                - label: ラベル（torch.Tensor）
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # トークン化
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Train/Val/TestのDataLoaderを作成

    Args:
        train_df: Train DataFrame（'review_text', 'label'列必須）
        val_df: Validation DataFrame
        test_df: Test DataFrame
        tokenizer: Hugging Face Tokenizer
        batch_size: batch size（デフォルト32）
        max_length: 最大トークン長（デフォルト128）
        num_workers: DataLoaderのworker数（デフォルト0）

    Returns:
        (train_loader, val_loader, test_loader)のタプル

    Raises:
        ValueError: DataFrameに必要な列がない場合

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_df, val_df, test_df, tokenizer, batch_size=32
        ... )
        >>> print(f"Train batches: {len(train_loader)}")
    """
    # バリデーション
    for df_name, df in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
        if 'review_text' not in df.columns:
            raise ValueError(f"{df_name}に'review_text'列が必要です")
        if 'label' not in df.columns:
            raise ValueError(f"{df_name}に'label'列が必要です")

    # Dataset作成
    train_dataset = SteamReviewDataset(
        texts=train_df['review_text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = SteamReviewDataset(
        texts=val_df['review_text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = SteamReviewDataset(
        texts=test_df['review_text'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Trainはシャッフル
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Val/Testはシャッフル不要
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def load_datasets_from_csv(
    train_csv: str,
    val_csv: str,
    test_csv: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    CSVファイルからDataFrameを読み込み

    Args:
        train_csv: Train CSVファイルパス
        val_csv: Validation CSVファイルパス
        test_csv: Test CSVファイルパス

    Returns:
        (train_df, val_df, test_df)のタプル

    Example:
        >>> train_df, val_df, test_df = load_datasets_from_csv(
        ...     'data/train/train_700.csv',
        ...     'data/train/val_150.csv',
        ...     'data/train/test_150.csv'
        ... )
        >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    return train_df, val_df, test_df
