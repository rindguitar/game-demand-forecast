"""
Learning Curve実験結果の分析と可視化

データ量と精度の関係を分析し、最適なデータ量を決定する。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_results(csv_path='data/experiments/learning_curve/results.csv'):
    """実験結果を読み込み"""
    df = pd.read_csv(csv_path)
    return df


def calculate_statistics(df):
    """データ量ごとの統計を計算"""
    stats = df.groupby('data_size').agg({
        'train_acc': ['mean', 'std'],
        'val_acc': ['mean', 'std'],
        'test_acc': ['mean', 'std']
    }).round(2)

    return stats


def print_summary(df):
    """実験結果のサマリーを表示"""
    print("=" * 80)
    print("Learning Curve実験結果サマリー")
    print("=" * 80)

    # データ量ごとの統計
    stats = calculate_statistics(df)
    print("\n【データ量別統計】")
    print(stats)

    # 目標達成状況
    print("\n" + "=" * 80)
    print("【目標達成状況】（Test Accuracy >= 85%）")
    print("=" * 80)

    target = 85.0

    for data_size in sorted(df['data_size'].unique()):
        subset = df[df['data_size'] == data_size]
        success_count = (subset['test_acc'] >= target).sum()
        total_count = len(subset)
        mean_acc = subset['test_acc'].mean()
        std_acc = subset['test_acc'].std()

        status = "✅" if success_count == total_count else "⚠️"

        print(f"\n{data_size:>6}件: {status} {success_count}/{total_count}回達成")
        print(f"         平均: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"         詳細: ", end="")

        for idx, row in subset.iterrows():
            mark = "✅" if row['test_acc'] >= target else "❌"
            print(f"{mark} {row['test_acc']:.2f}%  ", end="")
        print()

    # 最適データ量の決定
    print("\n" + "=" * 80)
    print("【最適データ量の決定】")
    print("=" * 80)

    # 全試行で目標達成したデータ量
    all_success_sizes = []
    for data_size in sorted(df['data_size'].unique()):
        subset = df[df['data_size'] == data_size]
        if (subset['test_acc'] >= target).all():
            all_success_sizes.append(data_size)

    if all_success_sizes:
        optimal_size = min(all_success_sizes)
        print(f"\n✅ 最適データ量: {optimal_size}件")
        print(f"   理由: 全試行で目標（85%）を安定達成した最小データ量")

        optimal_subset = df[df['data_size'] == optimal_size]
        print(f"   平均Test Accuracy: {optimal_subset['test_acc'].mean():.2f}% ± {optimal_subset['test_acc'].std():.2f}%")
    else:
        print("\n⚠️  全試行で目標達成したデータ量なし")
        print("   次善策を検討...")

        # 平均が85%以上のデータ量を探す
        mean_success_sizes = []
        for data_size in sorted(df['data_size'].unique()):
            subset = df[df['data_size'] == data_size]
            if subset['test_acc'].mean() >= target:
                mean_success_sizes.append(data_size)

        if mean_success_sizes:
            optimal_size = min(mean_success_sizes)
            print(f"\n   推奨データ量: {optimal_size}件")
            print(f"   理由: 平均精度が目標を達成した最小データ量")

            optimal_subset = df[df['data_size'] == optimal_size]
            success_rate = (optimal_subset['test_acc'] >= target).sum() / len(optimal_subset)
            print(f"   平均Test Accuracy: {optimal_subset['test_acc'].mean():.2f}% ± {optimal_subset['test_acc'].std():.2f}%")
            print(f"   目標達成率: {success_rate:.1%}")


def plot_learning_curve(df, output_path='data/experiments/learning_curve/learning_curve.png'):
    """Learning Curveをプロット"""

    # データ量ごとの平均と標準偏差を計算
    data_sizes = sorted(df['data_size'].unique())

    train_means = [df[df['data_size'] == size]['train_acc'].mean() for size in data_sizes]
    train_stds = [df[df['data_size'] == size]['train_acc'].std() for size in data_sizes]

    val_means = [df[df['data_size'] == size]['val_acc'].mean() for size in data_sizes]
    val_stds = [df[df['data_size'] == size]['val_acc'].std() for size in data_sizes]

    test_means = [df[df['data_size'] == size]['test_acc'].mean() for size in data_sizes]
    test_stds = [df[df['data_size'] == size]['test_acc'].std() for size in data_sizes]

    # プロット
    fig, ax = plt.subplots(figsize=(12, 7))

    # Train
    ax.plot(data_sizes, train_means, 'o-', label='Train', linewidth=2, markersize=8, color='#2ecc71')
    ax.fill_between(data_sizes,
                     np.array(train_means) - np.array(train_stds),
                     np.array(train_means) + np.array(train_stds),
                     alpha=0.2, color='#2ecc71')

    # Validation
    ax.plot(data_sizes, val_means, 'o-', label='Validation', linewidth=2, markersize=8, color='#3498db')
    ax.fill_between(data_sizes,
                     np.array(val_means) - np.array(val_stds),
                     np.array(val_means) + np.array(val_stds),
                     alpha=0.2, color='#3498db')

    # Test
    ax.plot(data_sizes, test_means, 'o-', label='Test', linewidth=2, markersize=8, color='#e74c3c')
    ax.fill_between(data_sizes,
                     np.array(test_means) - np.array(test_stds),
                     np.array(test_means) + np.array(test_stds),
                     alpha=0.2, color='#e74c3c')

    # 目標ライン（85%）
    ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)', alpha=0.7)

    # 装飾
    ax.set_xlabel('Dataset Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curve: Dataset Size vs Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xscale('log')
    ax.set_xticks(data_sizes)
    ax.set_xticklabels([f'{size}' for size in data_sizes])
    ax.set_ylim(70, 100)

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Learning Curve保存: {output_path}")

    plt.close()


def plot_detailed_results(df, output_path='data/experiments/learning_curve/detailed_results.png'):
    """詳細な結果をプロット（各試行を表示）"""

    data_sizes = sorted(df['data_size'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ['train_acc', 'val_acc', 'test_acc']
    titles = ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]

        # 各データ量での全試行をプロット
        positions = []
        values = []
        colors = []

        for i, data_size in enumerate(data_sizes):
            subset = df[df['data_size'] == data_size]
            for _, row in subset.iterrows():
                positions.append(i)
                values.append(row[metric])
                # Test Accで目標達成かどうかで色分け
                if metric == 'test_acc':
                    colors.append('#2ecc71' if row['test_acc'] >= 85 else '#e74c3c')
                else:
                    colors.append('#3498db')

        # 散布図
        ax.scatter(positions, values, s=100, alpha=0.6, c=colors, edgecolors='black', linewidth=1.5)

        # 平均値を線でプロット
        means = [df[df['data_size'] == size][metric].mean() for size in data_sizes]
        ax.plot(range(len(data_sizes)), means, 'k-', linewidth=2, label='Mean', alpha=0.7)

        # 目標ライン
        if metric == 'test_acc':
            ax.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target (85%)', alpha=0.7)

        # 装飾
        ax.set_xlabel('Dataset Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(data_sizes)))
        ax.set_xticklabels([f'{size}' for size in data_sizes])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(70, 100)
        ax.legend(fontsize=10)

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 詳細結果グラフ保存: {output_path}")

    plt.close()


def main():
    """メイン実行"""

    # 結果読み込み
    df = load_results()

    # サマリー表示
    print_summary(df)

    # グラフ生成
    print("\n" + "=" * 80)
    print("グラフ生成中...")
    print("=" * 80)

    plot_learning_curve(df)
    plot_detailed_results(df)

    print("\n" + "=" * 80)
    print("分析完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
