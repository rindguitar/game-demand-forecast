# Game Demand Forecast

ゲーム業界のプレイヤー需要を予測するシステム

## 概要

SNS（Reddit）とSteam APIからデータを収集し、NLPと時系列分析を用いて、プレイヤーが求めるゲーム要素・トレンドを予測する。

## 技術スタック

- **Python**: 3.11
- **NLP**: Hugging Face Transformers, BERTopic
- **時系列**: Prophet, LSTM (PyTorch)
- **データ**: Reddit API, Steam API
- **DB**: SQLite
- **可視化**: Matplotlib
- **環境**: Docker, WSL2

## セットアップ

### 前提条件
- WSL2（Ubuntu）
- Docker Desktop for Windows
- Git

### 1. リポジトリクローン
```bash
git clone https://github.com/rindguitar/game-demand-forecast.git
cd game-demand-forecast
```

### 2. 環境変数設定
```bash
cp .env.example .env
# .envを編集してAPIキーを設定
```

### 3. Docker起動
```bash
docker-compose up -d
```

### 4. コンテナに入る
```bash
docker-compose exec dev bash
```

### 5. セットアップ実行
```bash
make setup
```

## 使い方
```bash
# Jupyter起動
make notebook

# データ収集
make collect-data

# テスト実行
make test

# コードフォーマット
make format
```

## プロジェクト構成
```
game-demand-forecast/
├── notebooks/          # Jupyter Notebook
├── src/               # ソースコード
│   ├── data/         # データ収集
│   ├── models/       # モデル実装
│   ├── utils/        # ユーティリティ
│   └── visualization/ # 可視化
├── data/              # データ保存
├── docs/              # ドキュメント
├── scripts/           # 自動化スクリプト
└── tests/             # テストコード
```
## プロジェクト構成（詳細）
```
src/
├── data/           # データ収集
├── nlp/            # 自然言語処理
│   ├── sentiment.py     - 感情分析
│   └── topic.py         - トピック抽出
├── timeseries/     # 時系列予測
│   ├── prophet_model.py - Prophet実装
│   └── lstm_model.py    - LSTM実装
├── integration/    # NLP + 時系列統合
├── utils/          # 共通ユーティリティ
└── visualization/  # 可視化
```

## 開発フェーズ

- **Phase 2-3**: データ収集 + NLP
- **Phase 6-9**: 時系列予測
- **Phase 10**: 統合
