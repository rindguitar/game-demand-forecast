"""
DAPT（ドメイン適応事前学習）の feasibility 確認スクリプト

本格DAPTに着手する前に、GTX 1060 6GB で MLM継続学習が現実的かを測る（GO/NO-GO判断用）。
性能（精度）は測らない。測るのは「実行可能性」＝メモリに載るか・全体で何時間かかるか。

測定:
  - ピークGPUメモリ（6GBに収まるか）
  - 1ステップあたりの時間 → 目標コーパス×エポックに外挿した総所要時間

コーパスは測定用に既存の reviews_10000.csv テキストを流用（新規収集は不要）。
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

CORPUS_CSV = 'data/train/reviews_10000.csv'
BASE_MODEL = 'distilbert-base-uncased'


def main():
    parser = argparse.ArgumentParser(description='DAPT feasibility 測定')
    parser.add_argument('--n-reviews', type=int, default=2000, help='測定に使うレビュー数')
    parser.add_argument('--batch-size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--max-length', type=int, default=128, help='最大トークン長')
    parser.add_argument('--steps', type=int, default=50, help='計測する学習ステップ数')
    parser.add_argument('--warmup', type=int, default=3, help='計測から除くウォームアップ步数')
    # 外挿の目標（本番DAPTの想定規模）
    parser.add_argument('--target-reviews', type=int, default=50000, help='本番DAPTの想定コーパス件数')
    parser.add_argument('--target-epochs', type=int, default=3, help='本番DAPTの想定エポック数')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('⚠️ CUDAが使えません。feasibilityはGPUで測る必要があります。')
        return

    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print('=' * 60)
    print('DAPT feasibility 測定')
    print('=' * 60)
    print(f'  GPU: {gpu_name} ({gpu_total:.1f} GB)')
    print(f'  batch_size={args.batch_size} / max_length={args.max_length}')

    # 1. コーパス読み込み（測定用の小さなサブセット）
    df = pd.read_csv(CORPUS_CSV).dropna(subset=['review_text'])
    texts = df['review_text'].astype(str).tolist()[:args.n_reviews]
    print(f'  測定コーパス: {len(texts)}件')

    # 2. トークナイズ＋MLMコレータ（15%マスク）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    enc = tokenizer(texts, truncation=True, max_length=args.max_length)
    examples = [{'input_ids': ids} for ids in enc['input_ids']]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # 3. MLMモデル＋オプティマイザ
    print(f'  モデル読み込み: {BASE_MODEL} (MaskedLM)')
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    # 4. 学習ループを回して時間・メモリを計測
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    t_step = None
    n_done = 0
    print(f'\n  {args.warmup}ステップのウォームアップ後、{args.steps}ステップを計測...')
    for i, batch in enumerate(loader):
        if i >= args.warmup + args.steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if i == args.warmup:
            torch.cuda.synchronize()
            t_step = time.time()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i >= args.warmup:
            n_done += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t_step
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    sec_per_step = elapsed / max(n_done, 1)
    reviews_per_sec = args.batch_size / sec_per_step

    # 5. 本番規模への外挿
    total_steps = (args.target_reviews / args.batch_size) * args.target_epochs
    est_seconds = total_steps * sec_per_step
    est_hours = est_seconds / 3600

    print('\n' + '=' * 60)
    print('結果')
    print('=' * 60)
    print(f'  計測ステップ数: {n_done}')
    print(f'  1ステップ: {sec_per_step*1000:.0f} ms  (≈ {reviews_per_sec:.0f} reviews/sec)')
    print(f'  ピークGPUメモリ: {peak_mem:.2f} GB / {gpu_total:.1f} GB '
          f'({peak_mem/gpu_total*100:.0f}%)')

    print(f'\n  【本番DAPT想定への外挿】 {args.target_reviews:,}件 × {args.target_epochs}エポック')
    print(f'    総ステップ: 約 {total_steps:,.0f}')
    print(f'    推定所要時間: 約 {est_hours:.1f} 時間')

    # 6. GO/NO-GO の目安
    print('\n  【判断の目安】')
    if peak_mem > gpu_total * 0.95:
        print('    ⚠️ メモリがほぼ上限。batch_size か max_length を下げる必要あり')
    else:
        print(f'    ✓ メモリは収まっている（余裕 {gpu_total - peak_mem:.1f} GB）')
    if est_hours > 12:
        print(f'    ⚠️ {est_hours:.1f}時間は長い。コーパス/エポックを減らすか規模を再検討')
    else:
        print(f'    ✓ {est_hours:.1f}時間なら現実的（一晩〜半日）')


if __name__ == '__main__':
    main()
