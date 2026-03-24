# Game Demand Forecast

ゲーム業界のプレイヤー需要を予測するシステム

## 概要

Steam APIからゲームデータ（レビュー、プレイヤー数、タグ等）を収集し、NLPと時系列分析を用いて、プレイヤーが求めるゲーム要素・トレンドを予測する。

**データソース**: Steam API（将来的にデータが不足する場合はKaggleデータセットで補完）

## 技術スタック

- **Python**: 3.10（Ubuntu 22.04デフォルト）
- **PyTorch**: 2.7.1+cu118（CUDA 11.8対応）
- **NLP**: Hugging Face Transformers, BERTopic
- **時系列**: Prophet, LSTM (PyTorch)
- **データ**: Steam API（+ Kaggleデータセット※必要に応じて）
- **DB**: SQLite
- **可視化**: Matplotlib
- **環境**: Docker, WSL2, NVIDIA CUDA 11.8

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

### 2. API設定
```bash
cp .env.example .env
# .envを**手動で**編集してAPIキーを設定
```

**⚠️ セキュリティ注意**:
- `.env`ファイルは`.claudeignore`で保護されており、Claude Codeから読み込めません
- Steam APIキーは**必ず手動で**設定してください（https://steamcommunity.com/dev/apikey）
- `.env`ファイルは絶対にコミットしないでください（`.gitignore`に含まれています）

### 3. Docker起動
```bash
make build  # 初回のみ
make up
```

### 4. GPU認識確認
```bash
make gpu-check
```

### 5. コンテナに入る
```bash
make shell
```

## 使い方

### よく使うコマンド
```bash
# コンテナ起動・停止
make up      # コンテナ起動
make down    # コンテナ停止・削除
make restart # コンテナ再起動
make shell   # コンテナに入る

# 開発
make notebook      # Jupyter起動
make gpu-check     # GPU認識確認
make python-version # Pythonバージョン確認
make test          # テスト実行
make format        # コードフォーマット

# 全コマンド表示
make help
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
