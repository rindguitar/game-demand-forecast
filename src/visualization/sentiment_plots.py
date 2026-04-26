"""
感情分析結果の可視化

DBから取得した感情分析結果をグラフで可視化する関数を提供します。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional
from src.nlp.sentiment_db import (
    get_sentiment_stats,
    get_overall_stats,
    get_game_list,
    get_sentiment_timeseries
)

# 日本語フォント設定（必要に応じて）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")


def plot_sentiment_distribution(
    app_id: Optional[int] = None,
    save_path: Optional[str] = None,
    db_path: str = 'data/game_demand.db'
) -> None:
    """
    ゲーム別の感情分布を円グラフで表示

    Args:
        app_id: ゲームID（Noneの場合は全体）
        save_path: 保存先パス（Noneの場合は表示のみ）
        db_path: データベースファイルパス
    """
    if app_id is not None:
        # 特定ゲームの統計
        stats = get_sentiment_stats(app_id, db_path)
        if len(stats) == 0:
            print(f"No data for app_id {app_id}")
            return

        game_name = stats.iloc[0]['game_name']
        title = f'{game_name} - Sentiment Distribution'
    else:
        # 全体の統計
        overall = get_overall_stats(db_path)
        stats = pd.DataFrame([
            {'sentiment': 'POSITIVE', 'count': overall['positive']},
            {'sentiment': 'NEGATIVE', 'count': overall['negative']}
        ])
        title = 'Overall Sentiment Distribution'

    # 円グラフ
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#2ecc71', '#e74c3c']  # Green for POSITIVE, Red for NEGATIVE
    labels = stats['sentiment'].tolist()
    sizes = stats['count'].tolist()

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
    ax.set_title(title, fontsize=16, weight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sentiment_by_game(
    save_path: Optional[str] = None,
    db_path: str = 'data/game_demand.db'
) -> None:
    """
    全ゲームの感情比較を棒グラフで表示

    Args:
        save_path: 保存先パス（Noneの場合は表示のみ）
        db_path: データベースファイルパス
    """
    stats = get_sentiment_stats(None, db_path)

    if len(stats) == 0:
        print("No data available")
        return

    # データを整形
    pivot_data = stats.pivot(index='game_name', columns='sentiment', values='percentage')
    pivot_data = pivot_data.fillna(0)

    # 棒グラフ
    fig, ax = plt.subplots(figsize=(14, 8))

    x = range(len(pivot_data))
    width = 0.35

    if 'POSITIVE' in pivot_data.columns:
        ax.barh([i - width/2 for i in x], pivot_data['POSITIVE'],
                width, label='Positive', color='#2ecc71', alpha=0.8)
    if 'NEGATIVE' in pivot_data.columns:
        ax.barh([i + width/2 for i in x], pivot_data['NEGATIVE'],
                width, label='Negative', color='#e74c3c', alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels(pivot_data.index, fontsize=11)
    ax.set_xlabel('Percentage (%)', fontsize=12, weight='bold')
    ax.set_title('Sentiment Distribution by Game', fontsize=16, weight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(axis='x', alpha=0.3)

    # パーセント値を表示
    for i, game in enumerate(pivot_data.index):
        if 'POSITIVE' in pivot_data.columns:
            val = pivot_data.loc[game, 'POSITIVE']
            ax.text(val + 1, i - width/2, f'{val:.1f}%', va='center', fontsize=9)
        if 'NEGATIVE' in pivot_data.columns:
            val = pivot_data.loc[game, 'NEGATIVE']
            ax.text(val + 1, i + width/2, f'{val:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_sentiment_timeseries(
    app_id: Optional[int] = None,
    interval: str = 'day',
    save_path: Optional[str] = None,
    db_path: str = 'data/game_demand.db'
) -> None:
    """
    感情スコアの時系列グラフ

    Args:
        app_id: ゲームID（Noneの場合は全ゲーム）
        interval: 'day', 'week', 'month'
        save_path: 保存先パス（Noneの場合は表示のみ）
        db_path: データベースファイルパス
    """
    timeseries = get_sentiment_timeseries(app_id, interval, db_path)

    if len(timeseries) == 0:
        print("No timeseries data available")
        return

    # タイトル設定
    if app_id is not None:
        games = get_game_list(db_path)
        game = games[games['app_id'] == app_id]
        if len(game) > 0:
            game_name = game.iloc[0]['game_name']
            title = f'{game_name} - Sentiment Over Time ({interval.capitalize()})'
        else:
            title = f'Game {app_id} - Sentiment Over Time ({interval.capitalize()})'
    else:
        title = f'Overall Sentiment Over Time ({interval.capitalize()})'

    # グラフ作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # グラフ1: Positive率の推移
    ax1.plot(timeseries['date'], timeseries['positive_rate'],
             marker='o', linewidth=2, markersize=4, color='#2ecc71', label='Positive Rate')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.fill_between(timeseries['date'], 0, timeseries['positive_rate'],
                      alpha=0.2, color='#2ecc71')

    ax1.set_ylabel('Positive Rate (%)', fontsize=12, weight='bold')
    ax1.set_title(title, fontsize=14, weight='bold', pad=15)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 100)

    # x軸のラベルを回転
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # グラフ2: レビュー数の推移
    ax2.bar(timeseries['date'], timeseries['review_count'],
            color='#3498db', alpha=0.7, label='Review Count')

    ax2.set_xlabel('Date', fontsize=12, weight='bold')
    ax2.set_ylabel('Review Count', fontsize=12, weight='bold')
    ax2.set_title(f'Review Count Over Time', fontsize=14, weight='bold', pad=15)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    # x軸のラベルを回転
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # 動作確認
    print("=" * 60)
    print("可視化テスト")
    print("=" * 60)

    # 出力ディレクトリ作成
    os.makedirs('outputs', exist_ok=True)

    # 全体の感情分布
    print("\n1. 全体の感情分布（円グラフ）")
    plot_sentiment_distribution(save_path='outputs/sentiment_overall.png')

    # ゲーム別感情比較
    print("\n2. ゲーム別感情比較（棒グラフ）")
    plot_sentiment_by_game(save_path='outputs/sentiment_by_game.png')

    # 全ゲームの月別時系列
    print("\n3. 全ゲーム月別時系列")
    plot_sentiment_timeseries(interval='month', save_path='outputs/sentiment_timeseries_monthly.png')

    # 特定ゲームの日別時系列（Grand Theft Auto V）
    print("\n4. Grand Theft Auto V 日別時系列")
    plot_sentiment_timeseries(app_id=271590, interval='day',
                              save_path='outputs/sentiment_timeseries_gta5_daily.png')

    print("\n" + "=" * 60)
    print("✅ All visualizations saved to outputs/")
    print("=" * 60)
