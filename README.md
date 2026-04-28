# Game Demand Forecast

ゲーム業界のプレイヤー需要を予測するシステム

## 概要

Steam APIからゲームデータ（レビュー、プレイヤー数、タグ等）を収集し、NLPと時系列分析を用いて、プレイヤーが求めるゲーム要素・トレンドを予測する。

**データソース**: Steam API

## 技術スタック

- **Python**: 3.10（Ubuntu 22.04デフォルト）
- **PyTorch**: 2.7.1+cu118（CUDA 11.8対応）
- **NLP**: Hugging Face Transformers, BERTopic, DistilBERT
- **時系列**: Prophet, LSTM (PyTorch)
- **データ**: Steam API
- **DB**: SQLite
- **可視化**: Matplotlib
- **環境**: Docker, WSL2, NVIDIA CUDA 11.8

## セットアップ

### 前提条件
- WSL2（Ubuntu）
- Docker Desktop for Windows
- Git
- NVIDIA GPU（推奨）

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

## 機械学習コマンド

### 感情分析モデルの学習

```bash
# テスト用学習（1000件・短時間・best_modelを上書きしない）
make train-test

# 推奨設定で学習（10000件・⚠️ best_modelを上書き）
make train-sentiment

# カスタム設定で学習
make train-custom DATASET=data/train/reviews_5000.csv OUTPUT=models/my_model SEED=123
```

### テスト実行

```bash
# 全テスト実行
make test

# NLPテストのみ
make test-nlp
```

## プロジェクト構成
```
game-demand-forecast/
├── notebooks/          # Jupyter Notebook（実験・分析用）
├── src/               # ソースコード
│   ├── data/         # データ収集
│   ├── nlp/          # 自然言語処理
│   │   ├── sentiment.py  - 感情分析
│   │   ├── model.py      - DistilBERTモデル
│   │   └── train.py      - 学習ループ
│   ├── timeseries/   # 時系列予測
│   ├── integration/  # NLP + 時系列統合
│   ├── utils/        # ユーティリティ
│   └── visualization/ # 可視化
├── data/              # データ保存（.gitignoreで除外）
├── models/            # 学習済みモデル（.gitignoreで除外）
├── scripts/           # 自動化スクリプト
│   └── train_single_trial.py  - 学習スクリプト
└── tests/             # テストコード
```

## ドキュメント

詳細は[GitHub Wiki](https://github.com/rindguitar/game-demand-forecast/wiki)を参照してください。

**主要ドキュメント**:
- [Train/Val/Test Split](https://github.com/rindguitar/game-demand-forecast/wiki/Train-Val-Test-Split) - データセット分割の基礎
- [ハイパーパラメータとモデルパラメータ](https://github.com/rindguitar/game-demand-forecast/wiki/Hyperparameters-vs-Model-Parameters) - 学習パラメータの解説
- [Random Seed Selection](https://github.com/rindguitar/game-demand-forecast/wiki/Random-Seed-Selection) - ランダムシードの影響と選択基準
- [過学習（Overfitting）](https://github.com/rindguitar/game-demand-forecast/wiki/Overfitting) - 過学習とは何か、防ぐ方法

## ライセンス

このプロジェクトは個人学習目的で作成されています。

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
