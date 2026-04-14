"""
5000件のbalanced Steam レビューデータセットを収集（Phase 2）

Phase 1（1000件）でTest精度が目標未達（79.33% < 85%）だったため、
データ量を5倍に増やして精度向上を図る。

- 目標: 2500 Positive + 2500 Negative = 5000件
- ゲーム: 7タイトル（Phase 1と同じ）
- 各ゲーム: 約357 Positive + 357 Negative = 714件
"""

import os
import sys
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.steam_collector import collect_balanced_reviews


def main():
    """5000件のbalanced datasetを収集"""

    # 収集対象のゲーム（Phase 1と同じ）
    games = [
        # 新しいゲーム（2020-2023）
        (1086940, "Baldur's Gate 3", 2023),
        (1091500, "Cyberpunk 2077", 2020),

        # 定番・ロングセラー（2010年代）
        (413150, "Stardew Valley", 2016),
        (292030, "The Witcher 3", 2015),
        (271590, "Grand Theft Auto V", 2013),

        # 基幹タイトル
        (730, "Counter-Strike 2", 2023),
        (570, "Dota 2", 2013),
    ]

    # 各ゲームから収集する件数（7ゲームで5000件）
    per_game_positive = 2500 // len(games)  # 357件/ゲーム
    per_game_negative = 2500 // len(games)  # 357件/ゲーム

    # 端数調整（7で割り切れないので最初のゲームで調整）
    remainder = 2500 % len(games)

    all_positive_reviews = []
    all_negative_reviews = []

    print("=" * 60)
    print("Steam レビューデータセット収集開始（5000件・Phase 2）")
    print("=" * 60)
    print(f"\n収集対象: {len(games)}ゲーム")
    print(f"各ゲーム: 約{per_game_positive} Positive + 約{per_game_negative} Negative")
    print("\n【収集対象ゲーム一覧】")
    for app_id, game_name, year in games:
        print(f"  - {game_name} ({year})")

    for idx, (app_id, game_name, year) in enumerate(games):
        # 最初のゲームで端数調整
        n_pos = per_game_positive + (remainder if idx == 0 else 0)
        n_neg = per_game_negative + (remainder if idx == 0 else 0)

        print(f"\n[{idx+1}/{len(games)}] {game_name} からレビュー収集中...")
        print(f"  目標: {n_pos} Positive + {n_neg} Negative")

        try:
            reviews = collect_balanced_reviews(
                app_id=app_id,
                language='english',
                n_positive=n_pos,
                n_negative=n_neg
            )

            # レビューにゲーム情報を追加
            for review in reviews['positive']:
                review['game_id'] = app_id
                review['game_name'] = game_name
                review['game_year'] = year

            for review in reviews['negative']:
                review['game_id'] = app_id
                review['game_name'] = game_name
                review['game_year'] = year

            all_positive_reviews.extend(reviews['positive'])
            all_negative_reviews.extend(reviews['negative'])

            print(f"  ✅ 収集完了: {len(reviews['positive'])} Positive, {len(reviews['negative'])} Negative")

        except Exception as e:
            print(f"  ❌ エラー: {e}")
            print(f"  ⚠️  {game_name}をスキップして続行します...")
            continue

    # DataFrameに変換
    print("\n" + "=" * 60)
    print("データセット統合中...")
    print("=" * 60)

    positive_df = pd.DataFrame(all_positive_reviews)
    positive_df['label'] = 1  # Positive = 1

    negative_df = pd.DataFrame(all_negative_reviews)
    negative_df['label'] = 0  # Negative = 0

    # 統合してシャッフル
    df = pd.concat([positive_df, negative_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # シャッフル

    # 保存
    output_path = 'data/train/reviews_5000.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # 統計表示
    print(f"\n✅ データセット保存完了: {output_path}")
    print(f"\n【統計情報】")
    print(f"  総件数: {len(df)}")
    print(f"  Positive: {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    print(f"  Negative: {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")

    print(f"\n【ゲーム別内訳】")
    game_stats = df.groupby(['game_name', 'label']).size().unstack(fill_value=0)
    game_stats.columns = ['Negative', 'Positive']
    game_stats['Total'] = game_stats['Negative'] + game_stats['Positive']
    print(game_stats)

    print(f"\n【時代別内訳】")
    era_stats = df.groupby(['game_year', 'label']).size().unstack(fill_value=0)
    era_stats.columns = ['Negative', 'Positive']
    print(era_stats)

    print("\n" + "=" * 60)
    print("データ収集完了！")
    print("=" * 60)


if __name__ == '__main__':
    main()
