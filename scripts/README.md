# scripts/

実行スクリプト群。用途別に3つのサブディレクトリに分類されています。

## ディレクトリ構成

```
scripts/
├── collect/        # データ収集
├── nlp/            # NLP本番実行
├── experiments/    # 実験・検証・分析
└── benchmarks/     # 性能・実行可能性の計測
```

---

## collect/ — データ収集

Steam APIからレビューデータを収集するスクリプト。  
収集したデータは `data/train/` に保存されます。

| ファイル | 説明 |
|---|---|
| `collect_dataset_10k.py` | 10000件のbalancedレビューを収集（学習用・推奨） |
| `collect_dataset_20k.py` | 20000件のbalancedレビューを収集 |
| `collect_ood_testset.py` | OOD評価用テストセット収集（未知20ゲーム・ジャンル/タグ偏り対策） |
| `collect_dapt_corpus.py` | DAPT用の未ラベルコーパス収集（多様な10万件・OOD/学習ゲーム除外） |

**使用方法:**
```bash
make collect-10k           # 10000件（学習用）
make collect-ood           # OODテストセット
make collect-dapt-corpus   # DAPT用コーパス（10万件・未ラベル）
```

---

## nlp/ — NLP本番実行

感情分析モデルの学習とトピック抽出の本番実行スクリプト。  
通常は `make` コマンド経由で実行します。

| ファイル | 説明 |
|---|---|
| `train_sentiment.py` | DistilBERTの感情分析モデル学習（本番・実験兼用） |
| `train_dapt.py` | DAPT（未ラベルレビューでMLM継続学習・ドメイン適応モデル作成） |
| `extract_topics.py` | BERTopicによるトピック抽出（本番実行） |

**使用方法:**
```bash
make train-sentiment       # vanillaベースライン（best_model_pre_dapt上書き）
make train-dapt            # DAPT（MLM継続学習・要コーパス）
make train-sentiment-dapt  # DAPT baseで微調整（best_model上書き・本番）
make train-test            # パイプライン確認用（短時間）
make extract-topics        # トピック抽出
```

`train_sentiment.py` は `scripts/experiments/learning_curve_experiment.py` からもimportされます。

---

## experiments/ — 実験・検証・分析

モデル開発時に使用した実験・検証・分析用スクリプト。  
本番運用では通常使用しません。

| ファイル | 説明 |
|---|---|
| `learning_curve_experiment.py` | データ量と精度の関係を複数seedで検証 |
| `analyze_learning_curve.py` | Learning Curve実験結果の分析・可視化 |
| `validate_sentiment_english.py` | 英語100件での感情分析精度検証 |
| `bertopic_experiment.py` | BERTopicパラメータ実験 |
| `analyze_misclassified.py` | 誤分類の抽出（モデル誤りのbaseline取得） |
| `categorize_misclassified.py` | 誤分類のヒューリスティックタグ付け（誤りの傾向分析） |
| `explain_misclassified.py` | 誤分類の解釈（Layer Integrated Gradientsで寄与語抽出） |
| `compare_models_ood.py` | 複数モデルのOOD性能比較（accuracy/P/R/F1・McNemar） |

**使用方法:**
```bash
make learning-curve                        # 10k vs 20k で比較（デフォルト）
make learning-curve SIZES="5000 10000"     # サイズを指定して比較
make analyze-curve                         # 実験結果の分析・可視化
```

---

## benchmarks/ — 性能・実行可能性の計測

GPU性能・ファインチューニング負荷・DAPTの実行可能性などを「測る」スクリプト。

| ファイル | 説明 |
|---|---|
| `gpu_benchmark.py` | GPU性能計測 |
| `benchmark_finetuning.py` | ファインチューニングのGPU負荷検証 |
| `dapt_feasibility.py` | DAPT着手前の実行可能性（メモリ・所要時間）計測 |
