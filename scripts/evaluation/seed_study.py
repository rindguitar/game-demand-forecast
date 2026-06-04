"""
多シードでのDAPT効果の頑健性検証（Issue #24・方針A）

各シードで DAPTベース と vanillaベース をペア微調整し、OOD（未知ゲーム）で評価する。
- 学習の乱数はseed固定済み（各シード＝再現可能run）
- モデルは保存せず破棄（再現可能なので、採用シードは後で再学習すれば同一モデルが得られる）
- 結果は1runごとにCSV追記 → 中断しても再開可・後からシード追加も可

実行（GPU・長時間）:
    make exec CMD="python scripts/evaluation/seed_study.py --n-seeds 15"
分析のみ（既存結果を集計）:
    make exec CMD="python scripts/evaluation/seed_study.py --analyze-only"
"""

import os
import sys
import gc
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../nlp'))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from train_sentiment import train_sentiment  # scripts/nlp/train_sentiment.py
from src.nlp.model import load_model
from src.nlp.train import evaluate
from src.nlp.dataset import SteamReviewDataset

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

DATASET = 'data/train/reviews_10000.csv'
OOD = 'data/test/reviews_ood_2000.csv'
DAPT_BASE = 'models/dapt_distilbert'
VANILLA_BASE = 'distilbert-base-uncased'
RESULTS = 'data/experiments/seed_study/results.csv'
TMP_OUT = '/tmp/seed_study_model'  # 毎run上書き・破棄（ディスク節約）

BASES = [('dapt', DAPT_BASE), ('vanilla', VANILLA_BASE)]


def eval_ood(model_dir: str, ood_csv: str, device: str):
    """学習済みモデルをOODセットで評価し、accuracyとFP/FNを返す"""
    model, tokenizer = load_model(model_dir, device=device)
    df = pd.read_csv(ood_csv).dropna(subset=['review_text']).reset_index(drop=True)
    if 'label' not in df.columns and 'sentiment' in df.columns:
        df = df.rename(columns={'sentiment': 'label'})

    ds = SteamReviewDataset(df['review_text'].tolist(), df['label'].tolist(), tokenizer, max_length=128)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds, labels = evaluate(model, loader, device)
    preds, labels = np.array(preds), np.array(labels)

    acc = float((preds == labels).mean())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return acc, fp, fn


def load_done(results_path: str) -> set:
    """既に完了した(seed, base)の集合（再開用）"""
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        return set(zip(df['seed'], df['base']))
    return set()


def append_row(results_path: str, row: dict):
    """1run分を即追記（クラッシュしても結果が残る）"""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    header = not os.path.exists(results_path)
    pd.DataFrame([row]).to_csv(results_path, mode='a', header=header, index=False)


def run_grid(seeds, args, device):
    """seed × {dapt, vanilla} を学習・OOD評価し、結果を追記"""
    done = load_done(args.results)
    total = len(seeds) * len(BASES)
    i = 0
    for seed in seeds:
        for base_name, base_path in BASES:
            i += 1
            if (seed, base_name) in done:
                print(f'[{i}/{total}] seed={seed} base={base_name} → 既存スキップ')
                continue
            print(f'[{i}/{total}] seed={seed} base={base_name} 学習中...', flush=True)
            t0 = time.time()
            try:
                m = train_sentiment(
                    dataset_path=args.dataset, output_dir=args.tmp_out,
                    base_model=base_path, random_seed=seed, verbose=False,
                )
                ood_acc, ood_fp, ood_fn = eval_ood(args.tmp_out, args.ood, device)
                row = {
                    'seed': seed, 'base': base_name,
                    'train_acc': round(m['train_acc'], 2),
                    'val_acc': round(m['val_acc'], 2),
                    'test_acc': round(m['test_acc'], 2),
                    'ood_acc': round(ood_acc * 100, 2),
                    'ood_fp': ood_fp, 'ood_fn': ood_fn,
                    'best_epoch': m['best_epoch'],
                    'sec': round(time.time() - t0, 1), 'error': '',
                }
            except Exception as e:  # 壊れたrunは記録して継続（後で除外）
                row = {'seed': seed, 'base': base_name, 'train_acc': '', 'val_acc': '',
                       'test_acc': '', 'ood_acc': '', 'ood_fp': '', 'ood_fn': '',
                       'best_epoch': '', 'sec': round(time.time() - t0, 1), 'error': str(e)[:200]}
                print(f'   ⚠️ 失敗: {e}')
            append_row(args.results, row)
            print(f'   → ood_acc={row["ood_acc"]} val_acc={row["val_acc"]} ({row["sec"]}s)', flush=True)
            gc.collect()
            torch.cuda.empty_cache()


def analyze(results_path: str):
    """平均±SD・ペア検定・代表モデル候補を表示"""
    if not os.path.exists(results_path):
        print('結果がまだありません。')
        return
    df = pd.read_csv(results_path)
    df = df[df['error'].fillna('') == '']  # 壊れたrunを除外
    if len(df) == 0:
        print('有効な結果がありません。')
        return

    print('\n' + '=' * 64)
    print('多シード検証の分析（Issue #24）')
    print('=' * 64)

    for base in ['dapt', 'vanilla']:
        sub = df[df['base'] == base]
        if len(sub) == 0:
            continue
        sd = sub['ood_acc'].std(ddof=1) if len(sub) > 1 else 0.0
        vsd = sub['val_acc'].std(ddof=1) if len(sub) > 1 else 0.0
        print(f'\n[{base}] n={len(sub)}')
        print(f'  OOD acc: 平均 {sub["ood_acc"].mean():.2f}% ± {sd:.2f}% '
              f'(min {sub["ood_acc"].min():.2f} / max {sub["ood_acc"].max():.2f})')
        print(f'  Val acc: 平均 {sub["val_acc"].mean():.2f}% ± {vsd:.2f}%')

    # ペア比較（共通シードで dapt vs vanilla）
    piv = df.pivot_table(index='seed', columns='base', values='ood_acc')
    if {'dapt', 'vanilla'} <= set(piv.columns):
        piv = piv.dropna(subset=['dapt', 'vanilla'])
        if len(piv) >= 2:
            d, v = piv['dapt'].values, piv['vanilla'].values
            diff = d - v
            print(f'\n[ペア比較] 共通シード {len(piv)}個')
            print(f'  OOD差(dapt − vanilla): 平均 {diff.mean():+.2f}% ± {diff.std(ddof=1):.2f}%')
            print(f'  daptが勝ったシード: {int((diff > 0).sum())}/{len(piv)}')
            if scipy_stats is not None:
                try:
                    tp = scipy_stats.ttest_rel(d, v).pvalue
                    print(f'  paired t-test p={tp:.4g}', end='')
                    if len(set(diff)) > 1:
                        wp = scipy_stats.wilcoxon(d, v).pvalue
                        print(f' / Wilcoxon p={wp:.4g}')
                    else:
                        print()
                except Exception as e:
                    print(f'  検定スキップ: {e}')

    # 代表モデル候補（dapt・Val中央値のシード）
    dapt = df[df['base'] == 'dapt'].sort_values('val_acc').reset_index(drop=True)
    if len(dapt):
        mid = dapt.iloc[len(dapt) // 2]
        print(f'\n[代表モデル候補] dapt・Val中央値 → seed={int(mid["seed"])} '
              f'(val={mid["val_acc"]}%, ood={mid["ood_acc"]}%)')
        print('  本番化（再学習で同一モデルを再現）:')
        print(f'    make exec CMD="python scripts/nlp/train_sentiment.py '
              f'--dataset {DATASET} --output models/best_model '
              f'--base-model {DAPT_BASE} --seed {int(mid["seed"])}"')
    print('=' * 64)


def main():
    p = argparse.ArgumentParser(description='多シードDAPT検証（Issue #24・方針A）')
    p.add_argument('--n-seeds', type=int, default=15, help='シード数（seeds=0..N-1）')
    p.add_argument('--seeds', type=str, default=None, help='明示指定（カンマ区切り）。指定時は--n-seedsより優先')
    p.add_argument('--dataset', default=DATASET)
    p.add_argument('--ood', default=OOD)
    p.add_argument('--results', default=RESULTS)
    p.add_argument('--tmp-out', default=TMP_OUT)
    p.add_argument('--analyze-only', action='store_true', help='学習せず既存結果を集計するだけ')
    args = p.parse_args()

    if args.analyze_only:
        analyze(args.results)
        return

    if not torch.cuda.is_available():
        print('⚠️ GPUが使えません。多シード検証はGPUで実行してください。')
        return
    device = 'cuda'

    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else list(range(args.n_seeds))
    print(f'シード: {seeds}')
    print(f'{len(seeds)}シード × 2モデル(dapt/vanilla) = {len(seeds) * 2} run（既存はスキップ）')
    run_grid(seeds, args, device)
    analyze(args.results)


if __name__ == '__main__':
    main()
