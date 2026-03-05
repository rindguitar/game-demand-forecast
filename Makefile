.PHONY: help setup test lint format clean collect-data train-prophet train-lstm notebook

help:
	@echo "Available commands:"
	@echo "  make setup         - 初期セットアップ"
	@echo "  make test          - テスト実行"
	@echo "  make test-nlp      - NLPテスト"
	@echo "  make test-ts       - 時系列テスト"
	@echo "  make lint          - Linter実行"
	@echo "  make format        - コードフォーマット"
	@echo "  make clean         - キャッシュ削除"
	@echo "  make collect-data  - データ収集"
	@echo "  make train-prophet - Prophet学習"
	@echo "  make train-lstm    - LSTM学習"
	@echo "  make notebook      - Jupyter起動"

setup:
	pip install -r requirements.txt

test:
	pytest tests/ -v

test-nlp:
	pytest tests/test_nlp/ -v

test-ts:
	pytest tests/test_timeseries/ -v

lint:
	flake8 src/ --max-line-length=100
	pylint src/ --max-line-length=100 --disable=C0114,C0115,C0116

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

collect-data:
	python src/data/reddit_collector.py
	python src/data/steam_collector.py

train-prophet:
	python src/timeseries/prophet_model.py

train-lstm:
	python src/timeseries/lstm_model.py

notebook:
	jupyter lab --ip=0.0.0.0 --allow-root --no-browser
