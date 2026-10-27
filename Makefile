.PHONY: help setup test lint format clean collect-data train-prophet train-lstm notebook \
        build up down restart logs shell exec gpu-check python-version \
        extract-topics compare-granularity cooccurrence validate-tags element-vocabulary test-topic \
        collect-10k collect-20k learning-curve analyze-curve collect-launch-extra \
        topics-fit-sample topics-fit topics-assign-64 topics-assign-85 topics-149

help:
	@echo "Available commands:"
	@echo ""
	@echo "【Docker操作】"
	@echo "  make build         - Dockerイメージをビルド"
	@echo "  make up            - Dockerコンテナを起動"
	@echo "  make down          - Dockerコンテナを停止・削除"
	@echo "  make restart       - Dockerコンテナを再起動"
	@echo "  make logs          - Dockerログを表示"
	@echo "  make shell         - コンテナ内にbashで入る"
	@echo "  make exec CMD=xxx  - コンテナ内でコマンド実行"
	@echo ""
	@echo "【環境確認】"
	@echo "  make gpu-check     - GPU認識確認"
	@echo "  make python-version - Pythonバージョン確認"
	@echo ""
	@echo "【開発】"
	@echo "  make setup         - 初期セットアップ"
	@echo "  make notebook      - Jupyter起動"
	@echo "  make test          - テスト実行"
	@echo "  make test-nlp      - NLPテスト"
	@echo "  make test-topic    - トピック抽出テスト"
	@echo "  make test-ts       - 時系列テスト"
	@echo "  make lint          - Linter実行"
	@echo "  make format        - コードフォーマット"
	@echo "  make clean         - キャッシュ削除"
	@echo ""
	@echo "【機械学習】"
	@echo "  make collect-data       - データ収集（Steam API）"
	@echo "  make collect-10k        - 10000件レビュー収集"
	@echo "  make collect-20k        - 20000件レビュー収集"
	@echo "  make extract-topics     - トピック抽出（10000件レビュー）"
	@echo "  make compare-granularity - トピックの粒度レベルを比較（再学習なし）"
	@echo "  make cooccurrence       - ゲームごとのレシピと部品ペアの共起"
	@echo "  make validate-tags      - 供給側タグが代用になるかの検証"
	@echo "  make element-vocabulary - ①ゲーム要素の語彙をタグから作り直す"
	@echo "  make learning-curve     - Learning Curve実験（10k vs 20k）"
	@echo "  make analyze-curve      - Learning Curve結果分析・可視化"
	@echo "  make train-sentiment      - vanillaベースライン学習（⚠️best_model_pre_dapt上書き）"
	@echo "  make train-sentiment-dapt - DAPT本番モデル学習（⚠️best_model上書き）"
	@echo "  make train-test         - テスト用学習（1000件・test_modelに保存）"
	@echo "  make train-custom       - カスタム設定で学習"
	@echo "  make train-prophet      - Prophet学習"
	@echo "  make train-lstm         - LSTM学習"
	@echo ""
	@echo "【時系列予測パイプライン】"
	@echo "  make timeseries-dataset   - 収集→偏り点検を通しで実行（1.5〜2時間）"
	@echo "  make collect-timeseries   - 時系列用レビュー収集のみ（中断後も再開可）"
	@echo "  make inspect-timeseries   - 収集データの偏り点検のみ（収集途中でも可）"
	@echo "  make collect-launch-extra - 期間中の発売を選ばずに全部集める（#63 段階2・実績で約18時間）"
	@echo "  make topics-149           - 149本で話題を学び直し、64本・85本に別々に割り当てる（#63 段階2・以下4つをまとめて）"
	@echo "  make topics-fit-sample    - 149本から学習用サンプルを作る（目安: 数分）"
	@echo "  make topics-fit           - サンプルでトピックモデルを学習しpickle保存（目安: 10〜20分）"
	@echo "  make topics-assign-64     - 64本に149本のモデルで話題を割り当てる（目安: 約1時間）"
	@echo "  make topics-assign-85     - 85本に149本のモデルで話題を割り当てる（目安: 約1.5時間）"
	@echo "  make launch-persistence   - 発売後の山の続き方を測る（区分ごとの中央値との比・Issue #63）"
	@echo "                              引数は COLLECT_ARGS/INSPECT_ARGS で個別に渡す"
	@echo "  make persistence-elements - 山の続き方と、発売が持つ要素の関係をのぞく（シャッフルで偶然の差を確認・Issue #63）"
	@echo "  make forecast-prophet     - Prophetで週次シェアを予測し平均と比べる（Issue #41・数十秒。--tune は約7分）"
	@echo ""
	@echo "【DAPTパイプライン】"
	@echo "  make collect-ood          - OODテストセット収集（評価用）"
	@echo "  make collect-dapt-corpus  - DAPT用コーパス収集（10万件・未ラベル）"
	@echo "  make train-dapt           - DAPT実行（MLM継続学習）"
	@echo "  make compare-ood          - OOD性能比較（DAPT vs sst-2 ほか）"
	@echo "  make seed-study           - 多シード検証（Issue#24・GPU長時間。SEEDS=15で数変更）"
	@echo "  make seed-study-analyze   - 多シード検証の集計のみ"

# ============================================================
# Docker操作
# ============================================================
build:
	docker compose build

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

shell:
	docker compose exec dev bash

exec:
	docker compose exec dev $(CMD)

# ============================================================
# 環境確認
# ============================================================
gpu-check:
	docker compose exec dev python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python-version:
	docker compose exec dev python --version
	docker compose exec dev pip --version

# ============================================================
# 開発
# ============================================================
setup:
	pip install -r requirements.txt

test:
	docker compose exec dev pytest tests/ -v

test-nlp:
	docker compose exec dev pytest tests/test_nlp/ -v

test-topic:
	docker compose exec dev pytest tests/test_nlp/test_topic.py -v

test-ts:
	docker compose exec dev pytest tests/test_timeseries/ -v

lint:
	docker compose exec dev flake8 src/

format:
	docker compose exec dev black src/ tests/
	docker compose exec dev isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# ============================================================
# 機械学習
# ============================================================
collect-data:
	docker compose exec dev python src/data/steam_collector.py

collect-10k:
	docker compose exec dev python scripts/collect/collect_dataset_10k.py

collect-20k:
	docker compose exec dev python scripts/collect/collect_dataset_20k.py

# Learning Curve実験（SIZES で比較サイズを指定可能。例: make learning-curve SIZES="10000 20000"）
SIZES ?= 10000 20000
learning-curve:
	docker compose exec dev python scripts/learning_curve/learning_curve_experiment.py --sizes $(SIZES)

# Learning Curve結果分析・可視化
analyze-curve:
	docker compose exec dev python scripts/learning_curve/analyze_learning_curve.py

# トピック抽出（data/train/reviews_10000.csv → reviews_10000_with_topics.csv）
extract-topics:
	docker compose exec dev python scripts/nlp/extract_topics.py

# トピックの粒度レベルを比較（Issue #37 の判断材料・再学習しないので数分で終わる）
compare-granularity:
	docker compose exec dev python scripts/nlp/compare_topic_granularity.py $(GRANULARITY_ARGS)

# ゲームごとのレシピと部品ペアの共起（Issue #42 の材料・再学習なし）
cooccurrence:
	docker compose exec dev python scripts/nlp/build_topic_cooccurrence.py $(COOCCURRENCE_ARGS)

# 供給側タグがロスター拡大の代用になるかの検証（Issue #42）
# 検証1は標本が要るので、リフト下限を緩めたレシピを渡す
# ①ゲーム要素の語彙を母集団のタグから作り直す（Issue #45）
element-vocabulary:
	docker compose exec dev python scripts/nlp/update_element_vocabulary.py $(ELEMENT_VOCAB_ARGS)

validate-tags:
	docker compose exec dev python scripts/nlp/build_topic_cooccurrence.py \
		--min-lift 1.0 --min-count 20 --outdir data/timeseries/cooccurrence_full \
		--top 1 --top-pairs 1
	docker compose exec dev python scripts/nlp/validate_tag_supply.py \
		--recipes data/timeseries/cooccurrence_full/recipes.csv $(VALIDATE_TAGS_ARGS)

# 感情分析モデル学習（vanilla base＝DAPT前のベースライン）
# ⚠️ 警告: models/best_model_pre_dapt/ を上書きします
train-sentiment:
	docker compose exec dev python scripts/nlp/train_sentiment.py \
		--dataset data/train/reviews_10000.csv \
		--output models/best_model_pre_dapt \
		--seed 0 \
		--batch-size 16 \
		--epochs 10 \
		--lr 1e-5 \
		--patience 3

# DAPT済みモデルをbaseに感情分析を微調整（③・本番モデル）
# ⚠️ models/best_model/ を上書き。事前に models/dapt_distilbert/ が必要（python scripts/nlp/train_dapt.py）
train-sentiment-dapt:
	docker compose exec dev python scripts/nlp/train_sentiment.py \
		--dataset data/train/reviews_10000.csv \
		--base-model models/dapt_distilbert \
		--output models/best_model \
		--seed 0 \
		--batch-size 16 \
		--epochs 10 \
		--lr 1e-5 \
		--patience 3

# ============================================================
# 時系列予測パイプライン（Issue #32）
#   collect-timeseries → inspect-timeseries → extract-topics → ...
# ============================================================
# 時系列用レビュー収集（期間固定・自然比率・3年×30本。1.5〜2時間）
# 中断しても同じコマンドで再開できる（収集済みゲームは自動スキップ）
collect-timeseries:
	docker compose exec dev python scripts/collect/collect_timeseries_dataset.py $(COLLECT_ARGS)

# 収集データの偏り点検（APIを叩かずCSVだけ読むので収集途中でも実行可）
inspect-timeseries:
	docker compose exec dev python scripts/collect/inspect_timeseries_dataset.py $(INSPECT_ARGS)

# 収集 → 点検を通しで実行。収集が失敗したら点検はしない
timeseries-dataset: collect-timeseries inspect-timeseries

# 期間中に発売したゲームを、選定の規則を通さずに全部集める（Issue #63 段階2・実績で約18時間・何回かに分けてよい）
# 64本とは別のディレクトリに置く。期間はゲームごとに発売の8週前から今まで。中断しても同じコマンドで再開できる
# 顔ぶれだけ見るなら: make collect-launch-extra COLLECT_ARGS=--dry-run
LAUNCH_EXTRA_DIR ?= data/timeseries/launch_extra
LAUNCH_FROM ?= 2023-07-31
LAUNCH_TO ?= 2026-03-01
LAUNCH_PRE_WEEKS ?= 8
collect-launch-extra:
	docker compose exec dev python scripts/collect/collect_timeseries_dataset.py \
		--launched-from $(LAUNCH_FROM) --launched-to $(LAUNCH_TO) \
		--exclude-games data/timeseries/games.csv \
		--weeks-before-release $(LAUNCH_PRE_WEEKS) \
		--output $(LAUNCH_EXTRA_DIR)/reviews_timeseries.csv \
		--games-output $(LAUNCH_EXTRA_DIR)/games.csv \
		--progress-output $(LAUNCH_EXTRA_DIR)/collection_progress.json \
		--log-output $(LAUNCH_EXTRA_DIR)/collection_log.csv \
		$(COLLECT_ARGS)

# 64本＋85本（149本）で話題を学び直し、64本・85本それぞれに割り当てる（Issue #63 段階2）
# なぜ学び直すか: 64本のモデル（models/topic_64）は safetensors 保存で UMAP/HDBSCAN を持たず、
#   読み込むと割り当てが「話題との類似度」になり（Outlier がほぼ出ない）、学習した実行の
#   中での割り当てと揃わない。UMAP の乱数も固定していなかったので、学び直しても同じにならない。
#   そこで149本から新しいモデルを作り、pickle で UMAP/HDBSCAN ごと保存する（models/topic_149.pkl）
# なぜ別々に割り当てるか: 2つで約2GBありメモリが足りなくなるため。同じモデル・同じ基準なので結果は揃う
# なぜ相手の台帳の名前も消すか: 学習データは149本のタイトルを消したので、割り当てる側も
#   入力にない方のタイトルを --extra-titles で消して条件をそろえる
# 試走: make topics-assign-64 TOPIC_ARGS="--limit 20000"（topics-fit も TOPIC_ARGS で絞れる）
# 時間は目安。試走で直す
TOPIC_CHUNK ?= 500000
TOPIC_MODEL ?= models/topic_149.pkl
TOPIC_FIT_SAMPLE ?= data/timeseries/topic_fit_sample_149.csv
topics-149: topics-fit-sample topics-fit topics-assign-64 topics-assign-85

topics-fit-sample:
	docker compose exec dev python scripts/nlp/make_topic_fit_sample.py \
		--inputs data/timeseries/reviews_timeseries.csv $(LAUNCH_EXTRA_DIR)/reviews_timeseries.csv \
		--output $(TOPIC_FIT_SAMPLE)

topics-fit:
	docker compose exec dev python scripts/nlp/extract_topics.py \
		--input $(TOPIC_FIT_SAMPLE) \
		--output data/timeseries/topic_fit_sample_149_with_topics.csv \
		--stats-output data/timeseries/topic_statistics_fit_149.csv \
		--skip-english-filter --remove-all-game-names \
		--umap-seed 42 --model-format pickle --model-output $(TOPIC_MODEL) $(TOPIC_ARGS)

topics-assign-64:
	docker compose exec dev python scripts/nlp/extract_topics.py \
		--input data/timeseries/reviews_timeseries.csv \
		--output data/timeseries/reviews_timeseries_with_topics_149.csv \
		--stats-output data/timeseries/topic_statistics_149_64.csv \
		--model-input $(TOPIC_MODEL) \
		--skip-english-filter --remove-all-game-names \
		--extra-titles $(LAUNCH_EXTRA_DIR)/games.csv \
		--chunk-size $(TOPIC_CHUNK) $(TOPIC_ARGS)

topics-assign-85:
	docker compose exec dev python scripts/nlp/extract_topics.py \
		--input $(LAUNCH_EXTRA_DIR)/reviews_timeseries.csv \
		--output $(LAUNCH_EXTRA_DIR)/reviews_timeseries_with_topics_149.csv \
		--stats-output $(LAUNCH_EXTRA_DIR)/topic_statistics_149.csv \
		--model-input $(TOPIC_MODEL) \
		--skip-english-filter --remove-all-game-names \
		--extra-titles data/timeseries/games.csv \
		--chunk-size $(TOPIC_CHUNK) $(TOPIC_ARGS)

# 週次シェア（全64本）を Prophet で予測し、学習期間の平均・直近の平均と当たり具合を比べる（Issue #41・数十秒）
# 引数は FORECAST_ARGS で渡す。例: make forecast-prophet FORECAST_ARGS="--recent-weeks 8 --no-plot"
forecast-prophet:
	docker compose exec dev python scripts/timeseries/forecast_prophet.py $(FORECAST_ARGS)

# 発売後の山の続き方（残った割合）を測り、区分ごとの中央値との比を出す（Issue #63・数十秒）
# 引数は PERSISTENCE_ARGS で渡す。例: make launch-persistence PERSISTENCE_ARGS="--min-group 4"
launch-persistence:
	docker compose exec dev python scripts/timeseries/measure_launch_persistence.py $(PERSISTENCE_ARGS)

# 山の続き方と要素の関係をのぞく（Issue #63 段階3・launch-persistence のあとに回す）
# 引数は ELEMENTS_ARGS で渡す。例: make persistence-elements ELEMENTS_ARGS="--n-shuffle 5000"
persistence-elements:
	docker compose exec dev python scripts/timeseries/compare_persistence_elements.py $(ELEMENTS_ARGS)

# ============================================================
# DAPT（ドメイン適応事前学習）パイプライン
#   collect-dapt-corpus → train-dapt → train-sentiment-dapt → compare-ood
# ============================================================
# OODテストセット収集（評価用・未知20ゲーム2000件）
collect-ood:
	docker compose exec dev python scripts/collect/collect_ood_testset.py

# DAPT用の未ラベルコーパス収集（多様なゲーム10万件・OOD/学習ゲームは除外）
collect-dapt-corpus:
	docker compose exec dev python scripts/collect/collect_dapt_corpus.py

# DAPT実行可能性の測定（GTX1060のメモリ・所要時間）
dapt-feasibility:
	docker compose exec dev python scripts/benchmarks/dapt_feasibility.py

# DAPT本体（MLM継続学習 → models/dapt_distilbert）
train-dapt:
	docker compose exec dev python scripts/nlp/train_dapt.py

# OOD性能比較（デフォルト DAPT vs sst-2。3つ巴は --models で指定）
compare-ood:
	docker compose exec dev python scripts/evaluation/compare_models_ood.py

# 多シード検証（Issue #24・GPU長時間。SEEDSでシード数変更可）
seed-study:
	docker compose exec dev python scripts/evaluation/seed_study.py --n-seeds $(or $(SEEDS),15)

# 多シード検証の集計のみ（学習せず既存結果を分析）
seed-study-analyze:
	docker compose exec dev python scripts/evaluation/seed_study.py --analyze-only

# テスト用学習（best_modelを上書きしない）
train-test:
	docker compose exec dev python scripts/nlp/train_sentiment.py \
		--dataset data/train/reviews_10000.csv \
		--output models/test_model \
		--seed 0 \
		--batch-size 16 \
		--epochs 3 \
		--lr 1e-5 \
		--patience 3

# カスタム設定で学習（変数で設定を上書き可能）
# 例: make train-custom DATASET=data/train/reviews_5000.csv LR=1e-5
train-custom:
	docker compose exec dev python scripts/nlp/train_sentiment.py \
		--dataset $(DATASET) \
		--output $(OUTPUT) \
		--seed $(SEED) \
		--batch-size $(BATCH_SIZE) \
		--epochs $(EPOCHS) \
		--lr $(LR) \
		--patience $(PATIENCE)

train-prophet:
	python src/timeseries/prophet_model.py

train-lstm:
	python src/timeseries/lstm_model.py

notebook:
	jupyter lab --ip=0.0.0.0 --allow-root --no-browser
