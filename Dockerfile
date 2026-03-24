FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 基本パッケージインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python3（3.10）をデフォルトに
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# requirements.txtをコピー
COPY requirements.txt .

# Pythonライブラリインストール
RUN pip install --no-cache-dir --upgrade pip

# requirements.txtが空でなければインストール（コメント行のみの場合も考慮）
RUN if grep -q '^[^#]' requirements.txt 2>/dev/null; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# PyTorch（CUDA 11.8対応）- 専用インデックスURLが必要なため直接インストール
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Jupyter拡張（最低限必要）
RUN pip install --no-cache-dir jupyterlab

# ポート開放
EXPOSE 8888

# デフォルトコマンド
CMD ["/bin/bash"]
