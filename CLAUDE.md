# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

ゲーム業界のプレイヤー需要予測システム。Steam APIからレビュー・プレイヤー数・タグ等を収集し、NLPと時系列分析でプレイヤー需要を予測するMLプロジェクト。

技術スタック: Python 3.10 / Docker（Ubuntu 22.04・CUDA 11.8）/ PyTorch 2.7.1+cu118 / Transformers・BERTopic・DistilBERT / 時系列: Prophet→LSTM / データソース: Steam APIのみ

### データフロー
1. データ収集 (`src/data/`): Steam APIからレビュー・プレイヤー数等を取得
2. NLP (`src/nlp/`): 感情分析 + トピック抽出
3. 時系列準備 (`src/timeseries/data_preparation.py`): NLP結果とプレイヤー数を整形
4. 予測 (`src/timeseries/`): Prophet or LSTM
5. 統合 (`src/integration/`): 需要スコア + ランキング
6. 可視化 (`src/visualization/`)

## 開発ルール

### ⚠️ Makeコマンドを優先使用
Docker操作・開発タスクは直接`docker-compose`ではなく`make`を使う（`make help`参照）。

### ⚠️ 時間のかかる操作はユーザーが実行
`make build`等の再ビルド・大量データのダウンロードや長時間学習はClaude Codeが実行せず、必要になったらユーザーに依頼する。短時間のテスト実行（`make exec CMD="..."`）はClaude CodeがOK。

### ⚠️ コミット・プッシュのワークフロー
実装 → **テストで動作確認** → コミット → **必ずセットでプッシュ**。
- ❌ テストせずにコミット・プッシュ
- ❌ コミットだけしてプッシュを忘れる

### ブランチ戦略
- `main`: 常にデプロイ可能 / `feature/xxx`: 新機能開発
- **ブランチ作成したら作業開始前に必ずリモートにプッシュ**

### コミットメッセージ規約
`feat:` 新機能 / `fix:` バグ修正 / `docs:` ドキュメント / `refactor:` リファクタリング / `chore:` 雑務

### 段階的開発
小さく始めて動作確認してから拡張する（例: 10件→100件→1000件→本格実装）。

### コメント・docstringは日本語
- docstring・`#`コメントは日本語。変数名・関数名は英語のまま
- 技術用語は無理に訳さない（accuracy, batch, pipeline等は英語のままかカタカナ）
  - 良い例: `"""perfect predictionsのテスト"""` / 悪い例: `"""完璧な予測のテスト"""`

## ドキュメント管理
技術ドキュメント（分析・設計・ベンチマーク・用語解説）は**GitHub Wiki**で管理し、リポジトリに含めない。リポジトリはコードが参照するファイルのみ。
Wiki: https://github.com/rindguitar/game-demand-forecast/wiki

## セキュリティ・注意事項
- `.env`にSTEAM_API_KEYを設定（`.env.example`をコピー・**キー設定はユーザーが手動で**行う。コミット禁止）
- `.env`・`*.db`・`data/raw|processed`・`models/`等は`.claudeignore`で保護済み（詳細は`.claudeignore`参照）
- ハードコード禁止
- コミット前に本物のAPIキー・トークン等が含まれていないかチェック。疑わしければコミットしない
- GPU利用時は`torch.cuda.is_available()`で確認
- `requirements.txt`は必要になったライブラリを段階的に追加する運用
