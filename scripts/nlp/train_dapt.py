"""
DAPT（ドメイン適応事前学習）本体

distilbert-base-uncased を、未ラベルSteamレビュー（data/dapt/corpus.csv）で
MLM継続学習し、ゲームドメインに適応したDistilBERTを作る。

出力モデル（models/dapt_distilbert/）は、後段のファインチューニングで
SentimentClassifier の base として読み込む（AutoModel.from_pretrained で encoder を再利用）。

クラッシュ対策:
- 一定ステップごとに checkpoint を保存（落ちても最新の適応済みモデルが残る）
- 損失履歴を training_log.json に保存（後で学習曲線を確認できる）

実行（GPU必須）:
    make exec CMD="python scripts/nlp/train_dapt.py"
"""

import os
import sys
import time
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)

CORPUS = 'data/dapt/corpus.csv'
BASE_MODEL = 'distilbert-base-uncased'
OUTPUT_DIR = 'models/dapt_distilbert'


def save_checkpoint(model, tokenizer, output_dir):
    """適応済みモデルとtokenizerを保存（HuggingFace形式）"""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description='DAPT（MLM継続学習）')
    parser.add_argument('--corpus', type=str, default=CORPUS, help='未ラベルコーパスCSV')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR, help='出力モデルディレクトリ')
    parser.add_argument('--epochs', type=int, default=3, help='エポック数')
    parser.add_argument('--batch-size', type=int, default=16, help='バッチサイズ')
    parser.add_argument('--max-length', type=int, default=128, help='最大トークン長')
    parser.add_argument('--lr', type=float, default=5e-5, help='学習率')
    parser.add_argument('--mlm-prob', type=float, default=0.15, help='マスク率')
    parser.add_argument('--limit', type=int, default=0,
                        help='コーパスの先頭N件だけ使う（0=全件・動作確認用）')
    parser.add_argument('--warmup-ratio', type=float, default=0.05, help='warmupステップ比率')
    parser.add_argument('--save-every', type=int, default=500, help='checkpoint保存間隔（ステップ）')
    parser.add_argument('--log-every', type=int, default=50, help='損失ログ間隔（ステップ）')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('⚠️ CUDAが使えません。DAPTはGPUで実行してください。')
        return
    device = 'cuda'

    print('=' * 60)
    print('DAPT（ドメイン適応事前学習・MLM継続学習）')
    print('=' * 60)
    print(f'  base: {BASE_MODEL}  → 出力: {args.output}')
    print(f'  epochs={args.epochs} / batch={args.batch_size} / max_len={args.max_length} / lr={args.lr}')

    # 1. コーパス読み込み
    if not os.path.exists(args.corpus):
        print(f'❌ コーパスが見つかりません: {args.corpus}')
        print('   先に collect_dapt_corpus.py で収集してください。')
        return
    df = pd.read_csv(args.corpus).dropna(subset=['review_text'])
    texts = df['review_text'].astype(str).tolist()
    if args.limit > 0:
        texts = texts[:args.limit]
        print(f'  ⚠️ --limit={args.limit}: 動作確認モード（先頭{len(texts)}件のみ使用）')
    print(f'  コーパス: {len(texts):,}件')

    # 2. トークナイズ＋MLMコレータ（15%マスク）
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    enc = tokenizer(texts, truncation=True, max_length=args.max_length)
    examples = [{'input_ids': ids} for ids in enc['input_ids']]
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=args.mlm_prob)
    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # 3. モデル・オプティマイザ・スケジューラ
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'  総ステップ: {total_steps:,}（warmup {warmup_steps}）')

    # 4. 学習ループ
    model.train()
    history = []
    global_step = 0
    running_loss = 0.0
    start = time.time()
    torch.cuda.reset_peak_memory_stats()

    print('\n  学習開始...')
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            running_loss += loss.item()

            # 損失ログ
            if global_step % args.log_every == 0:
                avg_loss = running_loss / args.log_every
                running_loss = 0.0
                elapsed = time.time() - start
                eta = elapsed / global_step * (total_steps - global_step)
                print(f'  ep{epoch+1} step {global_step:,}/{total_steps:,} '
                      f'loss={avg_loss:.4f} 経過{elapsed/60:.1f}分 残り約{eta/60:.1f}分')
                history.append({'step': global_step, 'epoch': epoch + 1, 'loss': avg_loss})

            # checkpoint保存（クラッシュ対策・最新を上書き）
            if global_step % args.save_every == 0:
                save_checkpoint(model, tokenizer, args.output)
                print(f'    💾 checkpoint保存（step {global_step:,}）')

    # 5. 最終保存＋学習ログ
    save_checkpoint(model, tokenizer, args.output)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    total_min = (time.time() - start) / 60
    with open(os.path.join(args.output, 'dapt_training_log.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'base_model': BASE_MODEL,
            'corpus': args.corpus,
            'n_reviews': len(texts),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'lr': args.lr,
            'total_steps': total_steps,
            'peak_gpu_mem_gb': round(peak_mem, 2),
            'total_minutes': round(total_min, 1),
            'loss_history': history,
        }, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 60)
    print('✅ DAPT完了')
    print('=' * 60)
    print(f'  所要時間: {total_min:.1f}分 / ピークメモリ: {peak_mem:.2f}GB')
    if history:
        print(f'  loss: {history[0]["loss"]:.4f}（最初） → {history[-1]["loss"]:.4f}（最後）')
    print(f'  保存先: {args.output}/')
    print(f'  次：このモデルを base にファインチューニング（③）')


if __name__ == '__main__':
    main()
