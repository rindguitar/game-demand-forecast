FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 環境変数
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 基本パッケージインストール
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python3.11をデフォルトに
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# requirements.txtをコピー
COPY requirements.txt .

# Pythonライブラリインストール
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Jupyter拡張
RUN pip install --no-cache-dir jupyterlab

# ポート開放
EXPOSE 8888

# デフォルトコマンド
CMD ["/bin/bash"]
