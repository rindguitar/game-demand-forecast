"""
OODテストセット収集スクリプト

学習に使った7ゲーム以外の「レビューが豊富なゲーム」から、条件付きランダムで
テストデータを収集する。自作モデルと既製汎用モデルのOOD性能比較用（Issue #21）。

設計:
- 母集団: Steam公式検索APIのレビュー数順ゲーム（=レビューが豊富なゲーム）
- 学習7ゲームは除外
- ランダムに選び、有効な英語レビューが P50/N50 取れるゲームのみ採用
- 目標: 20ゲーム × (Positive 50 + Negative 50) = 2000件（balanced）

再利用可能な汎用ツールとして設計（他プロジェクトでも流用可）。
"""

import os
import sys
import re
import time
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import requests
import pandas as pd

from src.data.steam_collector import collect_balanced_reviews


# 学習に使った7ゲーム（テストでは除外）
TRAIN_GAME_IDS = {
    1086940,  # Baldur's Gate 3
    1091500,  # Cyberpunk 2077
    413150,   # Stardew Valley
    292030,   # The Witcher 3
    271590,   # Grand Theft Auto V
    730,      # Counter-Strike 2
    570,      # Dota 2
}


def get_popular_games(n_pages: int = 20) -> list:
    """
    Steam公式の検索APIからレビュー数順の人気ゲームを取得（レビューが豊富な母集団）

    Steam公式の検索結果JSONはappidを直接持たず、ロゴ画像URLに埋め込まれているため、
    正規表現で抽出する。

    Args:
        n_pages: 取得ページ数（1ページ約25件、20ページで約500件）

    Returns:
        (app_id, game_name)のリスト（レビュー数の多い順）
    """
    base_url = "https://store.steampowered.com/search/results/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0 Safari/537.36'
    }

    games = []
    seen = set()

    for page in range(n_pages):
        params = {
            'query': '',
            'start': page * 25,
            'count': 25,
            'sort_by': 'Reviews_DESC',  # レビュー数の多い順
            'category1': 998,           # 998 = ゲームのみ（DLC・ツール等を除外）
            'json': 1,
        }
        response = requests.get(base_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        items = data.get('items', [])
        if not items:
            break

        for item in items:
            logo = item.get('logo', '')
            # ロゴURL内の /apps/<appid>/ からappidを抽出
            match = re.search(r'/apps/(\d+)/', logo)
            if not match:
                continue
            app_id = int(match.group(1))
            if app_id in seen:
                continue
            seen.add(app_id)
            games.append((app_id, item.get('name', 'Unknown')))

        time.sleep(0.5)  # Steam APIへのrate limiting

    return games


def main():
    parser = argparse.ArgumentParser(description='OODテストセット収集')
    parser.add_argument('--n-games', type=int, default=20, help='採用するゲーム数')
    parser.add_argument('--n-positive', type=int, default=50, help='1ゲームあたりPositive件数')
    parser.add_argument('--n-negative', type=int, default=50, help='1ゲームあたりNegative件数')
    parser.add_argument('--seed', type=int, default=42, help='ランダムシード')
    parser.add_argument('--output', type=str, default='data/test/reviews_ood_2000.csv',
                        help='出力先CSVパス')
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("OODテストセット収集")
    print("=" * 60)
    print(f"目標: {args.n_games}ゲーム × (P{args.n_positive}+N{args.n_negative})")

    # 1. 母集団取得
    print("\n[1/3] Steam公式検索APIから人気ゲームリスト取得...")
    popular_games = get_popular_games()
    print(f"   取得: {len(popular_games)}ゲーム")

    # 学習7ゲーム除外
    candidates = [(aid, name) for aid, name in popular_games if aid not in TRAIN_GAME_IDS]
    print(f"   学習7ゲーム除外後: {len(candidates)}ゲーム")

    # シャッフル（ランダム化）
    random.shuffle(candidates)

    # 2. 各ゲームから条件付き収集
    print(f"\n[2/3] 条件付き収集（P/N両方が目標数取れたゲームのみ採用）...")
    collected_games = []
    all_reviews = []

    for app_id, game_name in candidates:
        if len(collected_games) >= args.n_games:
            break

        print(f"\n  試行: {game_name} (appid={app_id})")
        try:
            reviews = collect_balanced_reviews(
                app_id=app_id,
                language='english',
                n_positive=args.n_positive,
                n_negative=args.n_negative
            )
        except Exception as e:
            print(f"    ❌ エラー: {e} → スキップ")
            continue

        n_pos = len(reviews['positive'])
        n_neg = len(reviews['negative'])

        # 条件: P/N両方が目標数に達したゲームのみ採用
        if n_pos < args.n_positive or n_neg < args.n_negative:
            print(f"    ⚠️ 件数不足 (P{n_pos}/N{n_neg}) → スキップ")
            continue

        # 採用: メタ情報を付与
        for r in reviews['positive'][:args.n_positive]:
            r['game_id'] = app_id
            r['game_name'] = game_name
            r['sentiment'] = 1
            all_reviews.append(r)
        for r in reviews['negative'][:args.n_negative]:
            r['game_id'] = app_id
            r['game_name'] = game_name
            r['sentiment'] = 0
            all_reviews.append(r)

        collected_games.append((app_id, game_name))
        print(f"    ✅ 採用 ({len(collected_games)}/{args.n_games})")

    # 3. 保存
    print(f"\n[3/3] 保存...")
    df = pd.DataFrame(all_reviews)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 列の整理:
    # - recommended（投稿者の👍/👎・生の値。sentimentの検算用に残す）
    # - sentiment（正解ラベル 1=positive / 0=negative）
    df = df.drop(columns=['votes_up'])
    df = df.rename(columns={'voted_up': 'recommended'})
    col_order = ['review_text', 'sentiment', 'recommended', 'language',
                 'timestamp_created', 'author', 'game_id', 'game_name']
    df = df[col_order]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("✅ 収集完了")
    print("=" * 60)
    print(f"   採用ゲーム: {len(collected_games)}")
    print(f"   総レビュー: {len(df)}")
    if len(df) > 0:
        print(f"   Positive: {(df['sentiment']==1).sum()} / Negative: {(df['sentiment']==0).sum()}")
    print(f"   保存先: {args.output}")
    print(f"\n【採用ゲーム一覧】")
    for aid, name in collected_games:
        print(f"  - {name} ({aid})")


if __name__ == '__main__':
    main()
