"""
Learning Curve実験: データ量と精度の関係を検証

複数のデータ量（1000, 5000, 10000, 20000件）で各3回ずつ学習を行い、
データ量と精度の関係を定量的に分析する。
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.train_sentiment import train_sentiment


def run_learning_curve_experiment(
    data_sizes: list = [1000, 5000, 10000, 20000],
    seeds: list = [42, 123, 456],
    output_dir: str = 'data/experiments/learning_curve'
):
    """
    Learning Curve実験を実行

    Args:
        data_sizes: テストするデータ量のリスト
        seeds: ランダムシードのリスト（各データ量で使用）
        output_dir: 結果を保存するディレクトリ
    """
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)

    # 実験結果を格納
    results = []

    print("=" * 70)
    print("Learning Curve実験開始")
    print("=" * 70)
    print(f"\nデータ量: {data_sizes}")
    print(f"ランダムシード: {seeds}")
    print(f"総実行回数: {len(data_sizes)} × {len(seeds)} = {len(data_sizes) * len(seeds)}回\n")

    # 各データ量について
    for data_size in data_sizes:
        print(f"\n{'=' * 70}")
        print(f"データ量: {data_size}件")
        print(f"{'=' * 70}\n")

        # データセットパス確認
        dataset_path = f'data/train/reviews_{data_size}.csv'
        if not os.path.exists(dataset_path):
            print(f"⚠️  データセット未存在: {dataset_path}")
            print(f"   → スキップします。先にデータ収集が必要です。\n")
            continue

        # 各シードで学習
        for trial, seed in enumerate(seeds, start=1):
            print(f"\n[Trial {trial}/{len(seeds)}] データ量={data_size}, Seed={seed}")
            print("-" * 70)

            try:
                # モデル保存ディレクトリ
                model_dir = f'models/learning_curve/size_{data_size}/trial_{trial}_seed_{seed}'
                os.makedirs(model_dir, exist_ok=True)

                # 学習実行
                metrics = train_sentiment(
                    dataset_path=dataset_path,
                    output_dir=model_dir,
                    random_seed=seed,
                    batch_size=16,
                    epochs=10,
                    learning_rate=1e-5,
                    patience=3,
                    verbose=True
                )

                # 結果を記録
                result = {
                    'data_size': data_size,
                    'trial': trial,
                    'seed': seed,
                    'train_acc': metrics['train_acc'],
                    'val_acc': metrics['val_acc'],
                    'test_acc': metrics['test_acc'],
                    'best_epoch': metrics['best_epoch'],
                    'model_dir': model_dir,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)

                print(f"\n✅ Trial {trial} 完了")
                print(f"   Train: {metrics['train_acc']:.2f}% | "
                      f"Val: {metrics['val_acc']:.2f}% | "
                      f"Test: {metrics['test_acc']:.2f}%")

            except Exception as e:
                print(f"\n❌ Trial {trial} 失敗: {e}")
                print(f"   → 続行します...")
                continue

    # 結果を保存
    if results:
        # DataFrameに変換
        df_results = pd.DataFrame(results)

        # CSV保存
        csv_path = os.path.join(output_dir, 'results.csv')
        df_results.to_csv(csv_path, index=False)
        print(f"\n✅ 結果をCSVに保存: {csv_path}")

        # JSON保存（メタデータ含む）
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'experiment_info': {
                    'data_sizes': data_sizes,
                    'seeds': seeds,
                    'total_trials': len(results),
                    'timestamp': datetime.now().isoformat()
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ 結果をJSONに保存: {json_path}")

        # 統計サマリー表示
        print_summary(df_results)
    else:
        print("\n⚠️  実験結果なし（すべての試行が失敗またはスキップされました）")

    print("\n" + "=" * 70)
    print("Learning Curve実験完了")
    print("=" * 70)

    return results


def print_summary(df: pd.DataFrame):
    """実験結果のサマリーを表示"""
    print("\n" + "=" * 70)
    print("実験結果サマリー")
    print("=" * 70)

    # データ量ごとの統計
    summary = df.groupby('data_size').agg({
        'train_acc': ['mean', 'std'],
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    }).round(2)

    print("\n【データ量別統計】")
    print(summary)

    # 目標達成状況（Test Accuracy 85%以上）
    print("\n【目標達成状況】（Test Accuracy ≥ 85%）")
    for data_size in df['data_size'].unique():
        subset = df[df['data_size'] == data_size]
        success_count = (subset['test_acc'] >= 85.0).sum()
        total_count = len(subset)
        mean_acc = subset['test_acc'].mean()
        std_acc = subset['test_acc'].std()

        print(f"  {data_size:>5}件: {success_count}/{total_count}回達成 "
              f"(平均: {mean_acc:.2f}% ± {std_acc:.2f}%)")


def main():
    """メイン実行"""
    import argparse

    parser = argparse.ArgumentParser(description='Learning Curve実験')
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000, 20000],
        help='テストするデータ量（例: --sizes 10000 20000）'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help='ランダムシード（例: --seeds 42 123 456）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/experiments/learning_curve',
        help='結果保存ディレクトリ'
    )
    args = parser.parse_args()

    results = run_learning_curve_experiment(
        data_sizes=args.sizes,
        seeds=args.seeds,
        output_dir=args.output_dir
    )

    return results


if __name__ == '__main__':
    main()
