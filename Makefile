.PHONY: help setup test lint format clean collect-data train-prophet train-lstm notebook \
        build up down restart logs shell exec gpu-check python-version

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
	@echo "  make test-ts       - 時系列テスト"
	@echo "  make lint          - Linter実行"
	@echo "  make format        - コードフォーマット"
	@echo "  make clean         - キャッシュ削除"
	@echo ""
	@echo "【機械学習】"
	@echo "  make collect-data  - データ収集"
	@echo "  make train-prophet - Prophet学習"
	@echo "  make train-lstm    - LSTM学習"

# ============================================================
# Docker操作
# ============================================================
build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec dev bash

exec:
	docker-compose exec dev $(CMD)

# ============================================================
# 環境確認
# ============================================================
gpu-check:
	docker-compose exec dev python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

python-version:
	docker-compose exec dev python --version
	docker-compose exec dev pip --version

# ============================================================
# 開発
# ============================================================
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
