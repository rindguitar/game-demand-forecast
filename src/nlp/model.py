"""
DistilBERTベースの感情分析モデル

PyTorchでゼロから学習する感情分析モデルを提供。
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Tuple
import os


class SentimentClassifier(nn.Module):
    """
    DistilBERTベースの感情分析モデル

    Args:
        model_name: 事前学習済みmodelの名前（デフォルト'distilbert-base-uncased'）
        n_classes: 分類クラス数（デフォルト2: Positive/Negative）
        dropout: dropout率（デフォルト0.3）

    Example:
        >>> model = SentimentClassifier()
        >>> input_ids = torch.randint(0, 30522, (16, 128))  # batch_size=16, seq_len=128
        >>> attention_mask = torch.ones(16, 128)
        >>> logits = model(input_ids, attention_mask)
        >>> print(logits.shape)
        torch.Size([16, 2])
    """

    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        n_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        # モデル情報保存（保存・読み込み用）
        self.model_name = model_name
        self.n_classes = n_classes
        self.dropout_rate = dropout

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        forward pass

        Args:
            input_ids: トークンID（shape: [batch_size, seq_len]）
            attention_mask: attentionマスク（shape: [batch_size, seq_len]）

        Returns:
            logits（shape: [batch_size, n_classes]）
        """
        # DistilBERT forward
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS] tokenの出力を取得（最初のtoken）
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Dropout + 分類層
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def save_model(
    model: SentimentClassifier,
    tokenizer: AutoTokenizer,
    save_path: str = 'models/sentiment_model'
) -> None:
    """
    modelとtokenizerを保存

    Args:
        model: 保存するSentimentClassifier
        tokenizer: 保存するtokenizer
        save_path: 保存先ディレクトリ

    Example:
        >>> model = SentimentClassifier()
        >>> tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        >>> save_model(model, tokenizer, 'models/sentiment_model')
        ✅ Model saved to models/sentiment_model
    """
    os.makedirs(save_path, exist_ok=True)

    # モデルの重みを保存
    model_path = os.path.join(save_path, 'model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_name': model.model_name,
        'n_classes': model.n_classes,
        'dropout': model.dropout_rate,
    }, model_path)

    # トークナイザーを保存
    tokenizer.save_pretrained(save_path)

    print(f"✅ Model saved to {save_path}")
    print(f"  - model.pth: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")


def load_model(
    model_path: str = 'models/sentiment_model',
    device: str = 'cuda'
) -> Tuple[SentimentClassifier, AutoTokenizer]:
    """
    modelとtokenizerを読み込み

    Args:
        model_path: modelの保存ディレクトリ
        device: 'cuda'または'cpu'

    Returns:
        (model, tokenizer)のタプル

    Example:
        >>> model, tokenizer = load_model('models/sentiment_model', device='cuda')
        ✅ Model loaded from models/sentiment_model
        >>> model.eval()
    """
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # モデル読み込み
    checkpoint_path = os.path.join(model_path, 'model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # モデル初期化
    model = SentimentClassifier(
        model_name=checkpoint['model_name'],
        n_classes=checkpoint['n_classes'],
        dropout=checkpoint['dropout']
    )

    # 重みを読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✅ Model loaded from {model_path}")
    print(f"  - Device: {device}")
    print(f"  - Classes: {checkpoint['n_classes']}")

    return model, tokenizer
