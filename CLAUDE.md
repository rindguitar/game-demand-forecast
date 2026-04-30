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
- データベース: SQLite（※必要に応じて）

## 開発環境セットアップ

### ⚠️ 重要：Makeコマンドを優先使用すること
このプロジェクトでは、Docker操作や開発タスクは**Makeコマンドを優先して使用**してください。
直接`docker-compose`コマンドを使う代わりに、`make`コマンドを使うことで一貫性のある操作が可能です。

### ⚠️ 重要：時間のかかる操作はユーザーが実行
以下の操作は時間がかかるため、**Claude Codeではなくユーザーが手動で実行**してください：
- `make build`（Dockerイメージの再ビルド）- 数分～10分以上かかる
- `make down && make build && make up`（完全な再ビルド）
- 大量データのダウンロード・処理

**Claude Codeの役割**：
- コードの実装・編集
- `requirements.txt`の更新
- 短時間で完了するテスト実行（`make exec CMD="..."`等）
- ユーザーに「再ビルドが必要です」と通知

### 依存関係管理
**重要**: `requirements.txt`は段階的に追加する運用。現在はコメント行のみで、プロジェクト進行に合わせて必要なライブラリを追加していく。

## アーキテクチャ

### データフロー
1. **データ収集** (`src/data/`): Steam APIからデータ取得
   - ゲーム情報（タイトル、ジャンル、タグ）
   - プレイヤー数（同時接続数の時系列データ）
   - レビュー（NLP用テキストデータ）
   - 価格推移
2. **NLP処理** (`src/nlp/`): Steamレビューの感情分析 + トピック抽出
3. **時系列データ準備** (`src/timeseries/data_preparation.py`): NLP結果とプレイヤー数を時系列データに整形
4. **予測** (`src/timeseries/`): Prophet or LSTM で需要予測
5. **統合** (`src/integration/`): 需要スコア計算 + ランキング生成
6. **可視化** (`src/visualization/`): グラフ・ダッシュボード生成

### 環境変数
`.env`ファイルに以下を設定（`.env.example`をコピーして作成）:
```
STEAM_API_KEY=<Steam APIキー>
```

## 開発方針

### 段階的開発
小さく始めて動作確認してから拡張する。例：
1. 10投稿で動作確認
2. 100投稿でテスト
3. 1000投稿で本格実装
4. バッチ処理・最適化

### コメント・ドキュメントの日本語化
**重要**: コード内のコメントとdocstringは日本語で記述すること

- **docstring**: 関数・クラスの説明は日本語で記述
- **インラインコメント**: `#`コメントも日本語で記述
- **変数名・関数名**: 英語のまま（Pythonの慣例に従う）
- **技術用語**: 無理に日本語訳せず、英語のままかカタカナ読みを使用
  - 良い例: `def test_perfect_predictions()` → `"""perfect predictionsのテスト"""`
  - 悪い例: `"""完璧な予測のテスト"""` （ぎこちない日本語）
  - 技術用語の例: accuracy, precision, recall, F1 score, batch, pipeline, model, dataset, など

**例**:
```python
def analyze_sentiment(texts: List[str]) -> List[int]:
    """
    テキストの感情分析を実行

    Args:
        texts: 分析するテキストのリスト

    Returns:
        予測ラベルのリスト (1=POSITIVE, 0=NEGATIVE)
    """
    # パイプライン初期化
    pipeline = load_model()

    # 感情分析実行
    results = pipeline(texts)

    return results
```

### ブランチ戦略
- `main`: 常にデプロイ可能な状態
- `feature/xxx`: 新機能開発（例: `feature/data-collection`, `feature/sentiment-analysis`）

**⚠️ 重要：新しいブランチ作成時のルール**
 ブランチを作成したら、**作業開始前に必ずリモートにプッシュ**すること

### コミットメッセージ規約
```
feat: 新機能追加
fix: バグ修正
docs: ドキュメント更新
refactor: リファクタリング
```

### ⚠️ 重要：コミット・プッシュのワークフロー
**必ず以下の順序を守ること**：

1. **実装**: コードを書く
2. **テスト実行**: 実装したコードをテストし、成功を確認する
3. **コミット**: 変更をコミットする
4. **プッシュ**: コミットとセットで必ずリモートにプッシュする

**絶対にやってはいけないこと**：
- ❌ テストを実行せずにコミット・プッシュ
- ❌ コミットだけしてプッシュを忘れる

## ドキュメント管理

### 📚 GitHub Wiki
**重要**: 技術ドキュメントはGitHub Wikiで管理します。リポジトリには含めません。

#### ドキュメント配置ルール
- **Wiki**: 技術分析、設計ドキュメント、ベンチマーク結果など
  - 例: GPU性能分析、言語選択分析、クラウドGPU比較など
  - アクセス: https://github.com/rindguitar/game-demand-forecast/wiki

- **リポジトリ**: コードで呼び出すファイルのみ
  - 例: プロンプトテンプレート、設定ファイル、READMEなど

## 注意事項

### 🔒 セキュリティ - APIキー・機密情報の取り扱い

**重要**: `.env`ファイルは`.claudeignore`に追加済みです。Claude Codeは`.env`ファイルを読み込めません。

#### Steam APIキーの設定手順
1. `.env.example`をコピーして`.env`を作成
2. **手動で**Steam APIキーを設定（Claude Codeには依頼しない）
3. `.env`ファイルは絶対にコミットしない（`.gitignore`に含まれています）

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