"""
GPU性能ベンチマークスクリプト

GTX 1060 6GBで処理可能なデータ規模を測定
- GPUメモリ使用量
- NLPモデル（DistilBERT）のロード
- バッチサイズの限界値
"""

import torch
import time
from transformers import AutoTokenizer, AutoModel

def gpu_info():
    """GPU基本情報を表示"""
    print("=" * 60)
    print("GPU基本情報")
    print("=" * 60)
    print(f"GPU名: {torch.cuda.get_device_name(0)}")
    print(f"総メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"利用可能メモリ: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    print("=" * 60)

def memory_test():
    """メモリ使用量テスト"""
    print("\n" + "=" * 60)
    print("メモリ使用量テスト")
    print("=" * 60)

    # 初期メモリ
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0) / 1024**3
    print(f"初期メモリ使用量: {initial_memory:.3f} GB")

    # テンソル作成テスト（段階的にサイズを増やす）
    sizes = [
        (1000, 1000),
        (5000, 5000),
        (10000, 10000),
    ]

    for size in sizes:
        try:
            torch.cuda.empty_cache()
            tensor = torch.randn(size).cuda()
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"テンソルサイズ {size}: メモリ使用量 {memory_used:.3f} GB")
            del tensor
        except RuntimeError as e:
            print(f"テンソルサイズ {size}: メモリ不足 - {e}")
            break

    torch.cuda.empty_cache()

def nlp_model_test():
    """NLPモデル（DistilBERT）ロードテスト"""
    print("\n" + "=" * 60)
    print("NLPモデル（DistilBERT）ロードテスト")
    print("=" * 60)

    try:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(0) / 1024**3

        print("DistilBERTをロード中...")
        start_time = time.time()

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased").cuda()

        load_time = time.time() - start_time
        model_memory = torch.cuda.memory_allocated(0) / 1024**3

        print(f"✅ ロード成功")
        print(f"ロード時間: {load_time:.2f}秒")
        print(f"モデルメモリ使用量: {model_memory - initial_memory:.3f} GB")
        print(f"総メモリ使用量: {model_memory:.3f} GB")

        return tokenizer, model
    except Exception as e:
        print(f"❌ モデルロード失敗: {e}")
        return None, None

def batch_size_test(tokenizer, model):
    """バッチサイズ限界値テスト"""
    print("\n" + "=" * 60)
    print("バッチサイズ限界値テスト")
    print("=" * 60)

    if tokenizer is None or model is None:
        print("❌ モデルが未ロードのためスキップ")
        return

    # テストテキスト（Steamレビュー想定）
    sample_text = "This game is amazing! Great graphics and gameplay. Highly recommended."

    batch_sizes = [1, 4, 8, 16, 32, 64, 128]
    max_batch_size = 0

    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()

            # バッチデータ作成
            texts = [sample_text] * batch_size
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.cuda() for k, v in inputs.items()}

            # 推論実行
            with torch.no_grad():
                outputs = model(**inputs)

            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"バッチサイズ {batch_size:3d}: ✅ 成功 (メモリ: {memory_used:.3f} GB)")
            max_batch_size = batch_size

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"バッチサイズ {batch_size:3d}: ❌ メモリ不足")
                break
            else:
                print(f"バッチサイズ {batch_size:3d}: ❌ エラー - {e}")
                break

    print(f"\n推奨最大バッチサイズ: {max_batch_size}")
    torch.cuda.empty_cache()

def main():
    """メイン実行"""
    if not torch.cuda.is_available():
        print("❌ CUDAが利用できません")
        return

    # 1. GPU基本情報
    gpu_info()

    # 2. メモリテスト
    memory_test()

    # 3. NLPモデルロードテスト
    tokenizer, model = nlp_model_test()

    # 4. バッチサイズテスト
    batch_size_test(tokenizer, model)

    print("\n" + "=" * 60)
    print("ベンチマーク完了")
    print("=" * 60)

if __name__ == "__main__":
    main()
