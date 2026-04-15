"""
感情分析結果をDBに保存

Issue #6で学習したDistilBERTモデルを使用して、
既存のレビューデータを分析し、結果をDBに保存する。
"""

import sys
import os
import pandas as pd
import torch
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.nlp.model import SentimentClassifier
from src.data.database import Database


def load_model(model_path: str, device: str = 'cuda') -> SentimentClassifier:
    """
    学習済みモデルをロード

    Args:
        model_path: モデルファイルパス
        device: 'cuda' or 'cpu'

    Returns:
        DistilBERTSentimentClassifier: ロード済みモデル
    """
    print(f"Loading model from {model_path}...")
    model = SentimentClassifier(n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    return model


def analyze_reviews(model, texts: list, device: str = 'cuda', batch_size: int = 64) -> list:
    """
    レビューテキストを分析

    Args:
        model: DistilBERTモデル
        texts: レビューテキストのリスト
        device: 'cuda' or 'cpu'
        batch_size: バッチサイズ

    Returns:
        list: 分析結果（各要素は{'label': str, 'score': float}）
    """
    from transformers import DistilBertTokenizer

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    results = []

    # バッチ処理
    for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing"):
        batch_texts = texts[i:i + batch_size]

        # トークン化
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # 推論
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        # 結果を格納
        for pred, prob in zip(predictions, probs):
            label = 'POSITIVE' if pred.item() == 1 else 'NEGATIVE'
            score = prob[pred.item()].item()
            results.append({'label': label, 'score': score})

    return results


def save_to_db(db: Database, df: pd.DataFrame, results: list):
    """
    分析結果をDBに保存

    Args:
        db: Databaseインスタンス
        df: レビューデータのDataFrame
        results: 分析結果のリスト
    """
    print("\nSaving to database...")

    # データ準備
    data_to_insert = []
    for idx, (_, row) in enumerate(df.iterrows()):
        data_to_insert.append((
            int(row['game_id']),
            str(row['game_name']),
            str(row['review_text']),
            int(row['timestamp_created']),
            results[idx]['label'],
            float(results[idx]['score'])
        ))

    # バッチインサート
    query = """
        INSERT INTO sentiment_results
        (app_id, game_name, review_text, timestamp_created, sentiment, sentiment_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """

    db.executemany(query, data_to_insert)
    print(f"✅ Saved {len(data_to_insert)} records to database")


def main():
    """メイン処理"""

    print("=" * 60)
    print("感情分析結果のDB保存")
    print("=" * 60)

    # デバイス確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # データロード
    print("\n" + "=" * 60)
    print("Loading review data...")
    print("=" * 60)

    csv_path = 'data/train/reviews_5000.csv'
    df = pd.read_csv(csv_path)

    # NaN除去
    original_count = len(df)
    df = df.dropna(subset=['review_text'])
    print(f"✅ Loaded {len(df)} reviews (removed {original_count - len(df)} NaN)")

    # データベース初期化
    print("\n" + "=" * 60)
    print("Initializing database...")
    print("=" * 60)

    db = Database('data/game_demand.db')
    db.create_tables()

    # 既存データ確認
    existing = db.fetchone("SELECT COUNT(*) as count FROM sentiment_results")
    if existing and existing['count'] > 0:
        print(f"⚠️  Database already contains {existing['count']} records")
        response = input("Do you want to clear existing data? (y/n): ")
        if response.lower() == 'y':
            db.execute("DELETE FROM sentiment_results")
            print("✅ Cleared existing data")
        else:
            print("❌ Aborted")
            return

    # モデルロード
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)

    model_path = 'models/sentiment_model/best_model.pth'
    model = load_model(model_path, device)

    # 感情分析実行
    print("\n" + "=" * 60)
    print("Analyzing reviews...")
    print("=" * 60)

    texts = df['review_text'].tolist()
    results = analyze_reviews(model, texts, device=device, batch_size=64)

    # 結果をDBに保存
    print("\n" + "=" * 60)
    print("Saving to database...")
    print("=" * 60)

    save_to_db(db, df, results)

    # 統計表示
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)

    stats = db.fetchall("""
        SELECT
            game_name,
            sentiment,
            COUNT(*) as count
        FROM sentiment_results
        GROUP BY game_name, sentiment
        ORDER BY game_name, sentiment
    """)

    print("\nGame-wise sentiment distribution:")
    current_game = None
    for row in stats:
        if current_game != row['game_name']:
            if current_game is not None:
                print()
            current_game = row['game_name']
            print(f"\n{current_game}:")
        print(f"  {row['sentiment']}: {row['count']}")

    # 全体統計
    total_stats = db.fetchall("""
        SELECT
            sentiment,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sentiment_results), 2) as percentage
        FROM sentiment_results
        GROUP BY sentiment
    """)

    print("\n\nOverall sentiment distribution:")
    for row in total_stats:
        print(f"  {row['sentiment']}: {row['count']} ({row['percentage']}%)")

    print("\n" + "=" * 60)
    print("✅ Complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
