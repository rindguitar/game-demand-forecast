# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ゲーム業界のプレイヤー需要予測システム。Reddit APIとSteam APIからデータを収集し、NLP（自然言語処理）と時系列分析を組み合わせて、プレイヤーが求めるゲーム要素やトレンドを予測する機械学習プロジェクト。

### 技術スタック
- Python 3.11（Docker環境）
- GPU: NVIDIA CUDA 11.8（nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04）
- NLP: Hugging Face Transformers, BERTopic, DistilBERT
- 時系列予測: Prophet（Phase 1）、LSTM + PyTorch（Phase 2）
- データソース: Reddit API（PRAW）、Steam API
- データベース: SQLite

## 開発環境セットアップ

### Docker環境の起動
```bash
# コンテナのビルドと起動
docker-compose up -d

# コンテナに入る
docker-compose exec dev bash

# GPU認識確認
python -c "import torch; print(torch.cuda.is_available())"
```

### 依存関係管理
**重要**: `requirements.txt`は段階的に追加する運用。現在はコメント行のみで、プロジェクト進行に合わせて必要なライブラリを追加していく。

```bash
# コンテナ内でライブラリをインストール
pip install <package-name>

# requirements.txtに追記
pip freeze | grep <package-name> >> requirements.txt

# コンテナを再ビルド
docker-compose build
```

### Makeコマンド
```bash
make help          # 利用可能なコマンド一覧
make setup         # 初期セットアップ（requirements.txtからインストール）
make notebook      # Jupyter Lab起動（ポート8888）
make test          # 全テスト実行
make test-nlp      # NLP関連テストのみ実行
make test-ts       # 時系列予測関連テストのみ実行
make lint          # Linter実行（flake8 + pylint）
make format        # コードフォーマット（black + isort）
make clean         # キャッシュ削除（__pycache__, .pyc, .pytest_cache等）
make collect-data  # データ収集実行
make train-prophet # Prophet学習実行
make train-lstm    # LSTM学習実行
```

## アーキテクチャ

### データフロー
1. **データ収集** (`src/data/`): Reddit/Steam APIからデータ取得 → SQLiteに保存
2. **NLP処理** (`src/nlp/`): 感情分析（sentiment.py）+ トピック抽出（topic.py）
3. **時系列データ準備** (`src/timeseries/data_preparation.py`): NLP結果を時系列データに整形
4. **予測** (`src/timeseries/`): Prophet or LSTM で需要予測
5. **統合** (`src/integration/`): 需要スコア計算 + ランキング生成
6. **可視化** (`src/visualization/`): グラフ・ダッシュボード生成

### ディレクトリ構成の思想

#### `notebooks/`（試行錯誤用）
Jupyter Notebookで実験・分析・可視化確認を行う。確定したコードは`src/`に移植する。
- `01_data_collection.ipynb`: データ収集の動作確認
- `02_nlp_sentiment.ipynb`: 感情分析の実験
- `03_nlp_topic.ipynb`: トピック抽出の実験
- `04_timeseries_prophet.ipynb`: Prophet実装の実験
- `05_timeseries_lstm.ipynb`: LSTM実装の実験
- `06_integration.ipynb`: 統合処理の検証

#### `src/`（本番コード）
再利用可能な関数・クラスを実装。自動実行スクリプトもここに配置。

- `data/`: データ収集モジュール
  - `reddit_collector.py`: Reddit APIからの投稿・コメント収集
  - `steam_collector.py`: Steam APIからのゲーム情報・プレイヤー数取得
  - `database.py`: SQLiteデータベース操作

- `nlp/`: 自然言語処理モジュール
  - `sentiment.py`: Hugging Faceを使った感情分析
  - `topic.py`: BERTopicを使ったトピック抽出
  - `preprocessing.py`: テキスト前処理

- `timeseries/`: 時系列予測モジュール
  - `prophet_model.py`: Prophet実装（Phase 1）
  - `lstm_model.py`: LSTM実装（Phase 2）
  - `data_preparation.py`: NLP結果から時系列データ生成

- `integration/`: NLPと時系列の統合
  - `demand_score.py`: 需要スコア計算ロジック
  - `ranking.py`: ゲーム要素ランキング生成

- `utils/`: 共通ユーティリティ
  - `config.py`: 設定管理
  - `logger.py`: ロギング

- `visualization/`: 可視化
  - `plots.py`: グラフ生成
  - `dashboard.py`: ダッシュボード（将来実装予定）

### 環境変数
`.env`ファイルに以下を設定（`.env.example`をコピーして作成）:
```
REDDIT_CLIENT_ID=<Reddit APIのClient ID>
REDDIT_CLIENT_SECRET=<Reddit APIのClient Secret>
REDDIT_USER_AGENT=game_demand_forecast_v1.0
STEAM_API_KEY=<Steam APIキー>
```

## 開発フェーズ

現在: **Phase 0（環境構築中）**

- Phase 0: プロジェクト準備（Week 0）
- Phase 1: 環境構築（Week 0-1）
- Phase 2: データ収集基盤（Week 1-2）
- Phase 3: NLP - 感情分析（Week 3-4）
- Phase 4: NLP - トピック抽出（Week 5-6）
- Phase 5: データ蓄積・分析（Week 7-8）
- Phase 6: 時系列データ整形（Week 9-10）
- Phase 7: Prophet実装（Week 11-12）- V1完成
- Phase 8-9: LSTM実装（Week 13-18）- V2完成
- Phase 10: 統合・完成（Week 19-22）
- Phase 11: 仕上げ・納品（Week 23-24）

## 開発方針

### 段階的開発
小さく始めて動作確認してから拡張する。例：
1. 10投稿で動作確認
2. 100投稿でテスト
3. 1000投稿で本格実装
4. バッチ処理・最適化

### NotebookとPythonファイルの使い分け
- **Notebook**: 試行錯誤、データ分析、可視化確認
- **Pythonファイル**: 確定したコード、再利用可能な関数、自動実行スクリプト

### ブランチ戦略
- `main`: 常にデプロイ可能な状態
- `feature/xxx`: 新機能開発（例: `feature/data-collection`, `feature/sentiment-analysis`）

### コミットメッセージ規約
```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント更新
refactor: リファクタリング
```

## テスト実行

```bash
# 全テスト実行
make test

# 特定のモジュールのみ
make test-nlp    # NLPテスト
make test-ts     # 時系列テスト

# 個別のテストファイル実行
pytest tests/test_nlp/test_sentiment.py -v

# 特定のテスト関数のみ実行
pytest tests/test_nlp/test_sentiment.py::test_sentiment_analysis -v
```

## データベース

SQLite（`data/game_demand.db`）を使用。主要テーブル（設計予定）:
- `reddit_posts`: Reddit投稿データ
- `reddit_comments`: Redditコメントデータ
- `steam_games`: Steamゲーム情報
- `steam_player_counts`: Steamプレイヤー数（時系列）
- `sentiment_results`: 感情分析結果
- `topic_results`: トピック抽出結果
- `predictions`: 需要予測結果

## 注意事項

### Dockerfileの特殊対応
`requirements.txt`が空（コメント行のみ）でもビルドエラーにならないよう、Dockerfileで条件分岐処理を実装済み。

### GPU利用
docker-compose.ymlでNVIDIA GPUを利用する設定済み。PyTorchでGPU利用時は`torch.cuda.is_available()`で確認すること。

### 基本ルール
- 常にハードコード禁止
- 本物の秘密鍵が含まれていないか慎重にチェック
  API キー、トークン、パスワードなどの実際の値が含まれていないか
  疑わしい場合はコミットせず、チームで相談