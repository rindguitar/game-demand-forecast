"""
OODテストセット収集スクリプト

学習に使った7ゲーム以外の「レビューが豊富なゲーム」から、条件付きランダムで
テストデータを収集する。自作モデルと既製汎用モデルのOOD性能比較用（Issue #21）。

設計:
- 母集団: Steam公式検索APIのレビュー数順ゲーム（=レビューが豊富なゲーム）
- 学習7ゲームは除外
- ランダムに選び、有効な英語レビューが P50/N50 取れるゲームのみ採用
- 目標: 20ゲーム × (Positive 50 + Negative 50) = 2000件（balanced）

ジャンル偏り対策（特定ジャンルがテストセットを占有しないように）:
- ノイズタグ（Indie / Free To Play / Early Access）は判定から除外（遊びの種類を表さないため）
- (a) 除去後のジャンル集合が同一なら --max-per-profile 本までに制限（同一プロファイルの重複防止）
- (b) 1ジャンルあたりの所属数に上限（--max-per-genre）を設ける（メインジャンルの偏り防止）

似たゲーム除外（粗いgenresが取りこぼす細かい被りをユーザータグで検出）:
- (C-2) storeページのユーザータグ上位N個を取得し、ノイズタグ(TAG_NOISE)を除去後、
  採用済みゲームと閾値以上タグが共通する「似たゲーム」を弾く（例: Puzzle系・Roguelike系の重複）

再利用可能な汎用ツールとして設計（他プロジェクトでも流用可）。
"""

import os
import sys
import re
import html
import time
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd

from src.data.steam_collector import (
    collect_balanced_reviews,
    request_with_backoff,
    get_popular_games,
    HEADERS,
)


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

# ジャンル判定から除外するノイズタグ
# 遊びの種類ではなく、開発規模(Indie)・価格モデル(Free To Play)・販売ステータス(Early Access)を表すため
NOISE_TAGS = {'Indie', 'Free To Play', 'Early Access'}

# タグ重なり判定から除外するノイズタグ
# 17本の実データ分析で「無関係なゲーム同士の誤爆」を生んでいたタグを特定（憶測ではなく実例ベース）。
# 気分・機能・開発規模を表すもので、遊びの被りを意味しないため除外する。
TAG_NOISE = {
    'Singleplayer', 'Great Soundtrack', 'Atmospheric', 'Surreal',
    'Free to Play', 'Indie', 'Early Access',
}

def get_game_genres(app_id: int) -> frozenset:
    """
    Steam公式のappdetails APIからゲームのジャンル集合を取得

    Steamのジャンルは粗く順番も重要度順ではないため、「メインジャンル1つ」を
    決めるのではなく集合（frozenset）として扱う。ノイズタグ（Indie等）は呼び出し側で除去する。

    Args:
        app_id: SteamのアプリID

    Returns:
        ジャンル名のfrozenset（取得失敗・ジャンル無しの場合は空のfrozenset）
    """
    url = "https://store.steampowered.com/api/appdetails"
    params = {'appids': app_id, 'filters': 'genres'}
    response = request_with_backoff(url, params=params, headers=HEADERS, timeout=30)
    data = response.json()

    entry = data.get(str(app_id), {})
    if not entry.get('success'):
        return frozenset()
    genres = entry.get('data', {}).get('genres', [])
    return frozenset(g['description'] for g in genres)


def get_game_tags(app_id: int, n_tags: int = 6) -> list:
    """
    Steam storeページからユーザータグ（上位n_tags個）を取得

    Steam公式の `genres` は粗く「Puzzle」「Roguelike」等の実ジャンルを取りこぼすため、
    より粒度の細かいユーザータグを使って「似たゲーム」を検出する。タグはstoreページの
    HTMLに埋め込まれているのでスクレイプする（appdetails APIには含まれない）。

    Args:
        app_id: SteamのアプリID
        n_tags: 取得する上位タグ数

    Returns:
        上位n_tags個のタグ名リスト（取得失敗時は空リスト）
    """
    url = f"https://store.steampowered.com/app/{app_id}/"
    headers = dict(HEADERS)
    headers['Cookie'] = 'birthtime=0; mature_content=1'  # 年齢確認ゲートを回避
    response = request_with_backoff(url, headers=headers, timeout=30)

    # <a class="app_tag" ...>\n\tタグ名\t</a> からタグ名を抽出
    raw = re.findall(r'class="app_tag"[^>]*>\s*([^<]+?)\s*</a>', response.text)
    tags = [html.unescape(t).strip() for t in raw if t.strip()]
    return tags[:n_tags]


def main():
    parser = argparse.ArgumentParser(description='OODテストセット収集')
    parser.add_argument('--n-games', type=int, default=20, help='採用するゲーム数')
    parser.add_argument('--n-positive', type=int, default=50, help='1ゲームあたりPositive件数')
    parser.add_argument('--n-negative', type=int, default=50, help='1ゲームあたりNegative件数')
    parser.add_argument('--max-per-genre', type=int, default=6,
                        help='1ジャンルあたりの最大採用数（ジャンル偏り防止）')
    parser.add_argument('--max-per-profile', type=int, default=2,
                        help='同じジャンル集合（プロファイル）あたりの最大採用数')
    parser.add_argument('--n-tags', type=int, default=6,
                        help='タグ重なり判定で見る上位タグ数')
    parser.add_argument('--tag-overlap-threshold', type=int, default=2,
                        help='採用済みゲームとこの数以上タグが共通したら「似たゲーム」として弾く')
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
    print(f"\n[2/3] 条件付き収集（ジャンル偏り対策＋タグ重なり除外＋P/N両方が目標数取れたゲームのみ採用）...")
    print(f"   ジャンル上限: 1ジャンルあたり最大{args.max_per_genre}本 / "
          f"同一プロファイル最大{args.max_per_profile}本")
    print(f"   タグ重なり: 上位{args.n_tags}タグ中{args.tag_overlap_threshold}個以上共通で弾く")
    collected_games = []      # (app_id, name, sorted_genres, real_tags) のリスト
    all_reviews = []
    genre_counts = {}         # ジャンル名 -> 採用済み本数
    profile_counts = {}       # ジャンル集合(frozenset) -> 採用済み本数（完全一致dedup用）

    for app_id, game_name in candidates:
        if len(collected_games) >= args.n_games:
            break

        print(f"\n  試行: {game_name} (appid={app_id})")

        # --- ジャンル判定（軽い処理を先に。重いレビュー収集の前にフィルタ） ---
        try:
            genres = get_game_genres(app_id)
        except Exception as e:
            print(f"    ❌ ジャンル取得エラー: {e} → スキップ")
            continue
        time.sleep(0.3)  # appdetails APIへのrate limiting

        real_genres = genres - NOISE_TAGS  # ノイズタグ(Indie等)を除去
        if not real_genres:
            print(f"    ⚠️ 有効なジャンルなし → スキップ")
            continue

        # (a) 完全一致dedup: 同じジャンルプロファイルは max_per_profile 本まで
        if profile_counts.get(real_genres, 0) >= args.max_per_profile:
            print(f"    ⚠️ プロファイル上限 {sorted(real_genres)} に到達 → スキップ")
            continue

        # (b) ジャンルキャップ: いずれかのジャンルが上限に達していたら弾く
        over = sorted(g for g in real_genres if genre_counts.get(g, 0) >= args.max_per_genre)
        if over:
            print(f"    ⚠️ ジャンル上限 {over} に到達 → スキップ")
            continue

        # (C-2) タグ重なり除外: 採用済みゲームとタグが多数共通する「似たゲーム」を弾く
        # （genresは粗くPuzzle/Roguelike等の実ジャンルを取りこぼすため、細かいタグで判定）
        try:
            tags = get_game_tags(app_id, n_tags=args.n_tags)
        except Exception as e:
            print(f"    ❌ タグ取得エラー: {e} → スキップ")
            continue
        time.sleep(0.3)  # storeページへのrate limiting

        real_tags = set(tags) - TAG_NOISE  # ノイズタグ(Singleplayer等)を除去
        similar_to = None
        for _aid, prev_name, _g, prev_tags in collected_games:
            common = real_tags & prev_tags
            if len(common) >= args.tag_overlap_threshold:
                similar_to = (prev_name, sorted(common))
                break
        if similar_to:
            print(f"    ⚠️ タグ重なり {similar_to[1]} （{similar_to[0]}と類似）→ スキップ")
            continue

        # --- レビュー収集（重い処理） ---
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

        # 採用記録を更新（ジャンルカウント・プロファイルカウント・タグ集合）
        collected_games.append((app_id, game_name, sorted(real_genres), real_tags))
        profile_counts[real_genres] = profile_counts.get(real_genres, 0) + 1
        for g in real_genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        print(f"    ✅ 採用 ({len(collected_games)}/{args.n_games}) "
              f"ジャンル={sorted(real_genres)} タグ={sorted(real_tags)}")

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
    print(f"   採用ゲーム: {len(collected_games)} / 目標 {args.n_games}")
    if len(collected_games) < args.n_games:
        print(f"   ⚠️ 目標に未達。ジャンル上限(--max-per-genre={args.max_per_genre})が"
              f"厳しすぎる可能性。下のジャンル分布を見て緩めて再走を検討してください。")
    print(f"   総レビュー: {len(df)}")
    if len(df) > 0:
        print(f"   Positive: {(df['sentiment']==1).sum()} / Negative: {(df['sentiment']==0).sum()}")
    print(f"   保存先: {args.output}")

    print(f"\n【採用ゲーム一覧】")
    for aid, name, genres, tags in collected_games:
        print(f"  - {name} ({aid}) ジャンル={genres} タグ={sorted(tags)}")

    print(f"\n【ジャンル分布】（上限 {args.max_per_genre}）")
    for g, c in sorted(genre_counts.items(), key=lambda x: -x[1]):
        print(f"  - {g}: {c}本")


if __name__ == '__main__':
    main()
