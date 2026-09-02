# scripts/

実行スクリプト群。用途別のサブディレクトリに分類されています。

## ディレクトリ構成

```
scripts/
├── collect/            # データ収集
├── nlp/                # NLP本番実行
├── misclassification/  # 誤分類の分析パイプライン
├── evaluation/         # モデル評価・比較・多シード検証
├── learning_curve/     # データ量と精度の関係
├── topic/              # トピック抽出実験
└── benchmarks/         # 性能・実行可能性の計測
```

---

## 全体の流れ

大きくは4段です。詳しい図は、それぞれのディレクトリの節にあります。

```mermaid
flowchart LR
    A["collect/<br/>Steam APIから集める"] --> B["nlp/<br/>学習・トピック抽出"]
    B --> C["evaluation/<br/>精度を測る"]
    C --> D["misclassification/<br/>間違いを分析する"]
```

図の中の図形は共通で、**四角＝スクリプト / 円筒＝データ（CSV） / 六角形＝モデル** です。
矢印は「この出力が次の入力になる」という流れを表します。

---

## collect/ — データ収集

Steam APIからレビューデータを収集するスクリプト。  
収集したデータは `data/train/` に保存されます。

```mermaid
flowchart LR
    API(["Steam API"]) --> C1["collect_dataset_10k.py"] --> D1[("data/train/reviews_10000.csv")]
    API --> C2["collect_ood_testset.py"] --> D2[("data/test/reviews_ood_2000.csv")]
    API --> C3["collect_dapt_corpus.py"] --> D3[("data/dapt/corpus.csv")]
```

| ファイル | 説明 |
|---|---|
| `collect_dataset_10k.py` | 10000件のbalancedレビューを収集（学習用・推奨） |
| `collect_dataset_20k.py` | 20000件のbalancedレビューを収集 |
| `collect_ood_testset.py` | OOD評価用テストセット収集（未知20ゲーム・ジャンル/タグ偏り対策） |
| `collect_dapt_corpus.py` | DAPT用の未ラベルコーパス収集（多様な10万件・OOD/学習ゲーム除外） |
| `collect_timeseries_dataset.py` | 時系列予測用のレビュー収集（期間固定・自然比率・レビュー本文を保存）。Issue #32 |
| `inspect_timeseries_dataset.py` | 収集した時系列データの偏り点検（自然比率・ゲーム別シェア・ジャンルの本数/量の乖離・参加ゲーム数の推移）。APIを叩かずCSVだけ読む |

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

**感情分析モデルができるまで**（左から順に実行する）

```mermaid
flowchart LR
    D3[("data/dapt/corpus.csv")] --> T1["train_dapt.py"] --> M1{{"models/dapt_distilbert"}}
    M1 --> T2["train_sentiment.py"] --> M2{{"models/best_model"}}
    D1[("data/train/reviews_10000.csv")] --> T2
```

`train_dapt.py` が作るのは「Steamの言い回しに慣れただけ」のモデルで、まだ感情は判定できません。
それを土台に `train_sentiment.py` で微調整して、本番モデル `models/best_model` になります。

**トピック抽出**（上とは独立に動く）

```mermaid
flowchart LR
    D1b[("data/train/reviews_10000.csv")] --> T3["extract_topics.py"] --> D4[("reviews_10000_with_topics.csv")]
```

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

`train_sentiment.py` は `scripts/learning_curve/learning_curve_experiment.py` と `scripts/evaluation/seed_study.py` からもimportされます。

---

## misclassification/ — 誤分類の分析パイプライン

誤分類を「抽出 → タグ付け → 2モデル差分 → 解釈 → 可視化」する分析ツール群。

| ファイル | 説明 |
|---|---|
| `analyze_misclassified.py` | 任意モデル×未知データで誤分類を抽出（`--input`/`--model`） |
| `categorize_misclassified.py` | 誤分類のヒューリスティックタグ付け（`--input`） |
| `diff_misclassified.py` | 2モデルの誤分類差分（fixed/broke抽出・`--before`/`--after`） |
| `explain_misclassified.py` | 誤分類の解釈（Layer Integrated Gradientsで寄与語抽出・`--input`/`--model`） |
| `plot_dapt_diff.py` | DAPT前後の誤分類差分を可視化（fixed/broke・タグ別） |

### 手順1: 2つのモデルの誤分類を取り、差分を出す

同じ `analyze_misclassified.py` を、DAPT前とDAPT後で**2回**走らせます。

```mermaid
flowchart LR
    M1{{"best_model_pre_dapt<br/>DAPT前"}} --> A1["analyze_misclassified.py<br/>1回目"] --> C1[("misclassified_best_model_pre_dapt.csv")]
    M2{{"best_model<br/>DAPT後"}} --> A2["analyze_misclassified.py<br/>2回目"] --> C2[("misclassified_best_model.csv")]
    C1 --> DF["diff_misclassified.py"] --> FB[("fixed.csv / broke.csv")]
    C2 --> DF
```

2回とも入力データは同じ `data/test/reviews_ood_2000.csv` です（線が増えて読みにくくなるため図では省略）。
`fixed` は「DAPT後に直ったレビュー」、`broke` は「DAPT後に壊れたレビュー」です。

### 手順2: 差分を分析する

```mermaid
flowchart LR
    FB[("fixed.csv / broke.csv")] --> CAT["categorize_misclassified.py"] --> TG[("fixed_tagged.csv<br/>broke_tagged.csv")]
    TG --> PL["plot_dapt_diff.py"] --> PNG[("dapt_diff_errortype.png<br/>dapt_diff_tags.png")]
```

`plot_dapt_diff.py` はタグ付きCSVだけでなく、素の `fixed.csv` / `broke.csv` も読みます。
そのため `categorize_misclassified.py` を先に通しておく必要があります。

### 手順3: なぜ間違えたかを調べる（手順2とは独立）

```mermaid
flowchart LR
    C2b[("misclassified_best_model.csv")] --> EX["explain_misclassified.py"] --> TK[("token_scores.csv<br/>top_words.csv<br/>summary.json")]
```

---

## evaluation/ — モデル評価・比較・検証

| ファイル | 説明 |
|---|---|
| `compare_models_ood.py` | 複数モデルのOOD性能比較（accuracy/P/R/F1・McNemar） |
| `seed_study.py` | 多シードでDAPT効果を検証（Issue#24・平均±SD＋ペア検定・代表モデル選定） |
| `validate_sentiment_english.py` | 英語100件での感情分析精度検証 |

```mermaid
flowchart LR
    M2{{"models/best_model"}} --> E1["compare_models_ood.py"] --> O1[("data/experiments/ood_benchmark/<br/>metrics.json・比較グラフ")]
    D2[("data/test/reviews_ood_2000.csv")] --> E1
```

`seed_study.py` と `learning_curve_experiment.py` だけは、CSVを介さず
`train_sentiment.py` の関数を**直接呼んで**何度も学習を回します。

```mermaid
flowchart LR
    S["seed_study.py<br/>シードを変えて15回"] --> TS["train_sentiment.py<br/>を関数として呼ぶ"]
    L["learning_curve_experiment.py<br/>データ量を変えて複数回"] --> TS
    TS --> R[("それぞれの results.csv")]
```

**使用方法:**
```bash
make compare-ood            # OOD性能比較
make seed-study             # 多シード検証（GPU長時間。SEEDS=15で数変更）
make seed-study-analyze     # 多シード検証の集計のみ
```

---

## learning_curve/ — データ量と精度の関係

| ファイル | 説明 |
|---|---|
| `learning_curve_experiment.py` | データ量と精度の関係を複数seedで検証 |
| `analyze_learning_curve.py` | Learning Curve実験結果の分析・可視化 |

**使用方法:**
```bash
make learning-curve                        # 10k vs 20k で比較（デフォルト）
make learning-curve SIZES="5000 10000"     # サイズを指定して比較
make analyze-curve                         # 実験結果の分析・可視化
```

---

## topic/ — トピック抽出実験

| ファイル | 説明 |
|---|---|
| `bertopic_experiment.py` | BERTopicパラメータ実験 |

---

## benchmarks/ — 性能・実行可能性の計測

GPU性能・ファインチューニング負荷・DAPTの実行可能性などを「測る」スクリプト。

| ファイル | 説明 |
|---|---|
| `gpu_benchmark.py` | GPU性能計測 |
| `benchmark_finetuning.py` | ファインチューニングのGPU負荷検証 |
| `dapt_feasibility.py` | DAPT着手前の実行可能性（メモリ・所要時間）計測 |
| `timeseries_feasibility.py` | 時系列予測着手前の実行可能性計測。レビュー発生密度・英語フィルタ通過率・自然ポジ率を実測し、集計粒度（日次/週次）と必要ゲーム数を試算（Issue #31） |
| `seasonality_check.py` | レビュー投稿数の季節性測定。複数ゲームを合算し、年ごとの月別シェアの形が繰り返されるかで年次周期の有無を判定（Issue #32 の遡る期間を決めるため） |

---

## 関連

- [../src/README.md](../src/README.md) — スクリプトが使う部品（モジュール）の一覧と依存関係
- [../tests/README.md](../tests/README.md) — テストと対象モジュールの対応
- [ドキュメントマップ](https://github.com/rindguitar/game-demand-forecast/wiki/Documentation-Map) — Wiki全体の繋がり
