# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ゲーム業界のプレイヤー需要予測システム。Steam APIからゲームデータ（レビュー、プレイヤー数、タグ等）を収集し、NLP（自然言語処理）と時系列分析を組み合わせて、プレイヤーが求めるゲーム要素やトレンドを予測する機械学習プロジェクト。

**データソース変更**: 当初はReddit APIも使用予定でしたが、規約改訂により使用困難と判断。Steam APIのみで進行し、必要に応じてKaggleデータセットで補完する方針。

### 技術スタック
- Python 3.10（Docker環境・Ubuntu 22.04デフォルト）
- GPU: NVIDIA CUDA 11.8（nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04）
- PyTorch: 2.7.1+cu118（CUDA 11.8対応）
- NLP: Hugging Face Transformers, BERTopic, DistilBERT
- 時系列予測: Prophet（Phase 1）、LSTM + PyTorch（Phase 2）
- データソース: Steam API（+ Kaggleデータセット※必要に応じて）
- データベース: SQLite

## 開発環境セットアップ

### ⚠️ 重要：Makeコマンドを優先使用すること
このプロジェクトでは、Docker操作や開発タスクは**Makeコマンドを優先して使用**してください。
直接`docker-compose`コマンドを使う代わりに、`make`コマンドを使うことで一貫性のある操作が可能です。

### Docker環境の起動（Makeコマンド使用）
```bash
# コンテナのビルド
make build

# コンテナを起動
make up

# コンテナに入る
make shell

# GPU認識確認
make gpu-check

# Pythonバージョン確認
make python-version

# コンテナ停止・削除
make down

# コンテナ再起動
make restart

# ログ確認
make logs
```

### Makeコマンド一覧
```bash
# 全コマンドを表示
make help

# Docker操作
make build         # Dockerイメージをビルド
make up            # Dockerコンテナを起動
make down          # Dockerコンテナを停止・削除
make restart       # Dockerコンテナを再起動
make logs          # Dockerログを表示
make shell         # コンテナ内にbashで入る
make exec CMD=xxx  # コンテナ内でコマンド実行

# 環境確認
make gpu-check     # GPU認識確認
make python-version # Pythonバージョン確認

# 開発
make setup         # 初期セットアップ（requirements.txtからインストール）
make notebook      # Jupyter Lab起動（ポート8888）
make test          # 全テスト実行
make test-nlp      # NLP関連テストのみ実行
make test-ts       # 時系列予測関連テストのみ実行
make lint          # Linter実行（flake8 + pylint）
make format        # コードフォーマット（black + isort）
make clean         # キャッシュ削除（__pycache__, .pyc, .pytest_cache等）

# 機械学習
make collect-data  # データ収集実行
make train-prophet # Prophet学習実行
make train-lstm    # LSTM学習実行
```

### 依存関係管理
**重要**: `requirements.txt`は段階的に追加する運用。現在はコメント行のみで、プロジェクト進行に合わせて必要なライブラリを追加していく。

```bash
# コンテナ内でライブラリをインストール
make shell
pip install <package-name>

# requirements.txtに追記
pip freeze | grep <package-name> >> requirements.txt

# コンテナを再ビルド
make build
```

## アーキテクチャ

### データフロー
1. **データ収集** (`src/data/`): Steam APIからデータ取得 → SQLiteに保存
   - ゲーム情報（タイトル、ジャンル、タグ）
   - プレイヤー数（同時接続数の時系列データ）
   - レビュー（NLP用テキストデータ）
   - 価格推移
2. **NLP処理** (`src/nlp/`): Steamレビューの感情分析 + トピック抽出
3. **時系列データ準備** (`src/timeseries/data_preparation.py`): NLP結果とプレイヤー数を時系列データに整形
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
  - `steam_collector.py`: Steam APIからのゲーム情報・プレイヤー数・レビュー取得
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
STEAM_API_KEY=<Steam APIキー>
```

Steam APIキーの取得: https://steamcommunity.com/dev/apikey

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

**⚠️ 重要：新しいブランチ作成時のルール**
1. ブランチを作成したら、**作業開始前に必ずリモートにプッシュ**すること
2. 手順：
   ```bash
   git checkout -b feature/xxx
   git push -u origin feature/xxx  # ← 必ず実行！
   # その後、作業開始
   ```
3. これにより、作業の追跡とバックアップが確実になる

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
- `steam_games`: Steamゲーム情報（タイトル、ジャンル、タグ、価格等）
- `steam_player_counts`: Steamプレイヤー数（同時接続数の時系列データ）
- `steam_reviews`: Steamレビューデータ（NLP用テキスト）
- `sentiment_results`: 感情分析結果
- `topic_results`: トピック抽出結果
- `predictions`: 需要予測結果

## ドキュメント管理

### 📚 GitHub Wiki
**重要**: 技術ドキュメントはGitHub Wikiで管理します。リポジトリには含めません。

#### ドキュメント配置ルール
- **Wiki**: 技術分析、設計ドキュメント、ベンチマーク結果など
  - 例: GPU性能分析、言語選択分析、クラウドGPU比較など
  - アクセス: https://github.com/rindguitar/game-demand-forecast/wiki

- **リポジトリ**: コードで呼び出すファイルのみ
  - 例: プロンプトテンプレート、設定ファイル、READMEなど

#### Wikiへのドキュメント追加手順
```bash
# Wikiリポジトリをクローン
git clone https://github.com/rindguitar/game-demand-forecast.wiki.git /tmp/wiki

# ドキュメントを追加
cp your_document.md /tmp/wiki/Your-Document.md

# Homeページにリンクを追加
# /tmp/wiki/Home.md を編集

# コミット・プッシュ
cd /tmp/wiki
git add .
git commit -m "docs: ドキュメント追加"
git push origin master
```

## 注意事項

### 🔒 セキュリティ - APIキー・機密情報の取り扱い

**重要**: `.env`ファイルは`.claudeignore`に追加済みです。Claude Codeは`.env`ファイルを読み込めません。

#### Steam APIキーの設定手順
1. https://steamcommunity.com/dev/apikey でAPIキーを取得
2. `.env.example`をコピーして`.env`を作成
3. **手動で**Steam APIキーを設定（Claude Codeには依頼しない）
4. `.env`ファイルは絶対にコミットしない（`.gitignore`に含まれています）

#### .claudeignoreについて
以下のファイル・ディレクトリはClaude Codeから保護されています：
- `.env`, `.env.local`, `.env.*.local`（APIキー・機密情報）
- `*.db`, `data/game_demand.db`（データベースファイル）
- `data/raw/**`, `data/processed/**`（大容量データ）
- `models/**`, `*.pth`, `*.pt`（モデルファイル）
- その他機密情報を含む可能性のあるファイル

詳細は`.claudeignore`を参照してください。

### Dockerfileの特殊対応
`requirements.txt`が空（コメント行のみ）でもビルドエラーにならないよう、Dockerfileで条件分岐処理を実装済み。

### GPU利用
docker-compose.ymlでNVIDIA GPUを利用する設定済み。PyTorchでGPU利用時は`torch.cuda.is_available()`で確認すること。

### 基本ルール
- 常にハードコード禁止
- 本物の秘密鍵が含まれていないか慎重にチェック
  API キー、トークン、パスワードなどの実際の値が含まれていないか
  疑わしい場合はコミットせず、チームで相談