"""
ファインチューニングベンチマークスクリプト

GTX 1060 6GBでDistilBERTのファインチューニングが可能か検証します。
- VRAM使用量（batch_size別）
- 学習時間（1エポックあたり）
- 小規模データでの精度
"""

import sys
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import numpy as np

# srcをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class DummyDataset(Dataset):
    """ベンチマーク用のダミーデータセット"""

    def __init__(self, size=100, max_length=128):
        self.size = size
        self.max_length = max_length
        # ダミーテキスト（ランダムなトークンID）
        self.data = [
            {
                'input_ids': torch.randint(0, 30522, (max_length,)),
                'attention_mask': torch.ones(max_length, dtype=torch.long),
                'label': torch.randint(0, 2, (1,)).item()
            }
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class SentimentClassifier(nn.Module):
    """DistilBERTベースの感情分析model"""

    def __init__(self, model_name='distilbert-base-uncased', n_classes=2, dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        # DistilBERT forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] tokenの出力を取得
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Dropout + 分類層
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def get_gpu_memory():
    """GPU memory使用量を取得（MB単位）"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024  # MB
        return allocated, reserved, total
    return 0, 0, 0


def benchmark_batch_size(batch_sizes=[8, 16, 32, 64], dataset_size=100):
    """batch_size別のVRAM使用量をベンチマーク"""
    print("=" * 80)
    print("Batch Size別 VRAM使用量ベンチマーク")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("❌ GPUが利用できません。CPUで実行します（VRAMベンチマークは不可）")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    _, _, total_vram = get_gpu_memory()
    print(f"Total VRAM: {total_vram:.0f} MB\n")

    results = []

    for batch_size in batch_sizes:
        print(f"\n{'=' * 80}")
        print(f"Batch Size: {batch_size}")
        print(f"{'=' * 80}")

        try:
            # GPUメモリをクリア
            torch.cuda.empty_cache()

            # Modelを作成
            model = SentimentClassifier().to(device)
            optimizer = AdamW(model.parameters(), lr=1e-5)

            # Dataset & DataLoader
            dataset = DummyDataset(size=dataset_size)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 初期メモリ使用量
            allocated_before, reserved_before, _ = get_gpu_memory()
            print(f"初期VRAM使用量: {allocated_before:.0f} MB (Reserved: {reserved_before:.0f} MB)")

            # 1回forward + backward
            model.train()
            batch = next(iter(dataloader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Forward + Backward後のメモリ使用量
            allocated_after, reserved_after, _ = get_gpu_memory()
            print(f"学習時VRAM使用量: {allocated_after:.0f} MB (Reserved: {reserved_after:.0f} MB)")

            vram_used = allocated_after
            vram_usage_percent = (vram_used / total_vram) * 100

            print(f"✅ 成功: {vram_used:.0f} MB ({vram_usage_percent:.1f}% of {total_vram:.0f} MB)")

            results.append({
                'batch_size': batch_size,
                'vram_mb': vram_used,
                'vram_percent': vram_usage_percent,
                'status': 'success'
            })

            # クリーンアップ
            del model, optimizer, dataloader, dataset
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ メモリ不足: Batch Size {batch_size} は動作不可")
                results.append({
                    'batch_size': batch_size,
                    'vram_mb': 0,
                    'vram_percent': 0,
                    'status': 'OOM'
                })
                torch.cuda.empty_cache()
            else:
                raise e

    # 結果サマリー
    print("\n" + "=" * 80)
    print("VRAM使用量サマリー")
    print("=" * 80)
    print(f"{'Batch Size':<12} {'VRAM (MB)':<12} {'VRAM (%)':<12} {'Status':<12}")
    print("-" * 80)

    for result in results:
        status_symbol = "✅" if result['status'] == 'success' else "❌"
        print(f"{result['batch_size']:<12} {result['vram_mb']:<12.0f} {result['vram_percent']:<12.1f} {status_symbol} {result['status']}")

    # 推奨batch_size
    print("\n" + "=" * 80)
    print("推奨設定")
    print("=" * 80)

    successful = [r for r in results if r['status'] == 'success']
    if successful:
        # VRAM使用率80%以下の最大batch_size
        safe_batch_sizes = [r for r in successful if r['vram_percent'] <= 80]
        if safe_batch_sizes:
            recommended = max(safe_batch_sizes, key=lambda x: x['batch_size'])
            print(f"✅ 推奨batch_size: {recommended['batch_size']} (VRAM使用率: {recommended['vram_percent']:.1f}%)")
        else:
            print(f"⚠️  VRAM使用率80%以下のbatch_sizeはありません")
            max_batch = max(successful, key=lambda x: x['batch_size'])
            print(f"   最大batch_size: {max_batch['batch_size']} (VRAM使用率: {max_batch['vram_percent']:.1f}%)")
    else:
        print("❌ 動作可能なbatch_sizeがありません")


def benchmark_training_time(batch_size=32, dataset_size=100, num_batches=10):
    """学習時間をベンチマーク"""
    print("\n" + "=" * 80)
    print("学習時間ベンチマーク")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Dataset Size: {dataset_size}")
    print(f"Benchmarking {num_batches} batches...\n")

    # Model作成
    model = SentimentClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Dataset & DataLoader
    dataset = DummyDataset(size=dataset_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 学習時間計測
    model.train()
    batch_times = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        start_time = time.time()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward
        logits = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time = time.time() - start_time
        batch_times.append(batch_time)

        print(f"Batch {i+1}/{num_batches}: {batch_time:.3f}秒 (Loss: {loss.item():.4f})")

    # 統計
    avg_batch_time = np.mean(batch_times)
    std_batch_time = np.std(batch_times)

    print(f"\n平均batch処理時間: {avg_batch_time:.3f}秒 (±{std_batch_time:.3f}秒)")

    # 1エポックあたりの推定時間
    batches_per_epoch_700 = int(700 / batch_size)  # Train 700件
    batches_per_epoch_1000 = int(1000 / batch_size)  # 全1000件

    epoch_time_700 = batches_per_epoch_700 * avg_batch_time
    epoch_time_1000 = batches_per_epoch_1000 * avg_batch_time

    print(f"\n推定学習時間:")
    print(f"  - 1エポック (Train 700件): {epoch_time_700/60:.1f}分")
    print(f"  - 1エポック (全1000件):   {epoch_time_1000/60:.1f}分")
    print(f"  - 5エポック (Train 700件): {epoch_time_700*5/60:.1f}分")

    # クリーンアップ
    del model, optimizer, dataloader, dataset
    torch.cuda.empty_cache()


def main():
    """ベンチマーク実行"""
    print("\n" + "🔥 " * 30)
    print("DistilBERTファインチューニング ベンチマーク")
    print("🔥 " * 30 + "\n")

    # GPU情報
    if torch.cuda.is_available():
        print(f"✅ GPU利用可能")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        _, _, total_vram = get_gpu_memory()
        print(f"   Total VRAM: {total_vram:.0f} MB\n")
    else:
        print(f"⚠️  GPU利用不可（CPUで実行）\n")

    # ベンチマーク1: Batch Size別VRAM使用量
    benchmark_batch_size(batch_sizes=[8, 16, 32, 64], dataset_size=100)

    # ベンチマーク2: 学習時間
    benchmark_training_time(batch_size=32, dataset_size=100, num_batches=10)

    print("\n" + "=" * 80)
    print("✅ ベンチマーク完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
