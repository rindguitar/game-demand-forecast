"""
20000件のbalanced Steam レビューデータセットを収集

Learning Curve実験用に20000件のデータセットを収集する。
- 目標: 10000 Positive + 10000 Negative = 20000件
- ゲーム: 7タイトル（既存と同じ）
- 各ゲーム: 約1428 Positive + 1428 Negative = 2856件
"""

import os
import sys
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.steam_collector import collect_balanced_reviews


def main():
    """20000件のbalanced datasetを収集"""

    # 収集対象のゲーム（既存と同じ7タイトル）
    games = [
        # 新しいゲーム（2020-2023）
        (1086940, "Baldur's Gate 3", 2023),
        (1091500, "Cyberpunk 2077", 2020),

        # 定番・ロングセラー（2010年代）
        (413150, "Stardew Valley", 2016),
        (292030, "The Witcher 3", 2015),
        (271590, "Grand Theft Auto V", 2013),

        # 基幹タイトル（継続的にレビュー蓄積）
        (730, "Counter-Strike 2", 2023),
        (570, "Dota 2", 2013),
    ]

    # 各ゲームから収集する件数（7ゲームで20000件）
    per_game_positive = 10000 // len(games)  # 1428件/ゲーム
    per_game_negative = 10000 // len(games)  # 1428件/ゲーム

    # 端数調整
    remainder = 10000 % len(games)

    all_positive_reviews = []
    all_negative_reviews = []

    print("=" * 60)
    print("Steam レビューデータセット収集開始（20000件）")
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

    df_positive = pd.DataFrame(all_positive_reviews)
    df_negative = pd.DataFrame(all_negative_reviews)

    # ラベル付け
    df_positive['label'] = 1  # POSITIVE
    df_negative['label'] = 0  # NEGATIVE

    # 統合
    df = pd.concat([df_positive, df_negative], ignore_index=True)

    # シャッフル
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✅ データセット作成完了")
    print(f"  Total: {len(df)} reviews")
    print(f"  Positive: {len(df_positive)} ({len(df_positive)/len(df)*100:.1f}%)")
    print(f"  Negative: {len(df_negative)} ({len(df_negative)/len(df)*100:.1f}%)")

    # 保存
    output_path = 'data/train/reviews_20000.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')

    print(f"\n✅ データセット保存: {output_path}")
    print(f"   ファイルサイズ: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("収集完了")
    print("=" * 60)


if __name__ == '__main__':
    main()
