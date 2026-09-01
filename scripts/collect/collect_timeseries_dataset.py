"""
時系列予測用のレビュー収集スクリプト（Issue #32）

需要スコアの材料になるレビューを、期間で区切って自然比率のまま集める。
感情分析用（collect_dataset_10k.py）とは必要なデータの性質が正反対なので別スクリプト。

  感情分析用  各ゲーム714件で打ち切り・ポジ/ネガを50:50に強制
  時系列用    期間で区切る・比率は元のまま

理由は Wiki「学習用データと測定用データの違い」を参照。要点は、均衡させると
「何が求められているか」の測定結果が収集方法の産物になってしまうこと。

設計:
- 母集団: get_popular_games()（売上上位＝レビューが豊富）
- 採用条件: 累計レビュー数が閾値以上（実測で1万件未満は 2.12件/日まで落ちる）
- ジャンル偏り対策: collect_ood_testset.py のジャンル判定・タグ重なり除外を流用
- 保存: ゲーム1本ごとに追記。中断しても再開できる（収集済みは自動スキップ）

レビュー本文をそのまま保存する。集約値だけにすると、後から要素の強弱
（2人用か4人用か等）を断面で調べられなくなるため。

使い方は --help を参照。
"""

import os
import sys
import csv
import time
import random
import argparse
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from src.data.steam_collector import (  # noqa: E402
    collect_natural_reviews,
    get_popular_games,
    get_review_summary,
)
from collect_ood_testset import (  # noqa: E402
    TAG_NOISE,
    NOISE_TAGS,
    get_game_genres,
    get_game_tags,
)

FIELDS = [
    'game_id', 'game_name', 'review_text', 'voted_up', 'timestamp_created',
    'timestamp_updated', 'author', 'language', 'votes_up', 'comment_count',
    'weighted_vote_score', 'playtime_at_review', 'playtime_forever',
    'playtime_last_two_weeks', 'steam_purchase', 'received_for_free',
    'refunded', 'written_during_early_access',
]


def load_collected_ids(path: str) -> set:
    """収集済みのapp_idを読む（中断からの再開用）"""
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {int(r['game_id']) for r in csv.DictReader(f) if r.get('game_id')}


def append_rows(path: str, rows: list) -> None:
    """1ゲーム分を追記する。ゲーム単位で書くので中断しても途中まで残る"""
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def select_games(args, already: set) -> list:
    """
    収集対象のゲームを選ぶ。

    実数合算で需要を測る方針のため、集めたレビューの内訳がそのまま需要スコアの
    内訳になる。つまりゲーム選定は収集量の問題ではなく定義の一部であり、
    ジャンルが偏らないよう条件を付ける（詳細は docs/decisions.md）。
    """
    popular = get_popular_games(n_pages=args.pages)
    candidates = [g for g in popular if g[0] not in already]
    random.seed(args.seed)
    random.shuffle(candidates)

    selected = []
    genre_counts, profile_counts = {}, {}

    for app_id, name in candidates:
        if len(selected) >= args.n_games:
            break

        # 1. レビュー数の下限（軽い1リクエストなので最初に弾く）
        summary = get_review_summary(app_id)
        total = summary.get('total_reviews', 0)
        if total < args.min_reviews:
            continue

        # 2. ジャンル判定（ノイズタグを除いた実ジャンルが取れるものだけ）
        genres = get_game_genres(app_id) - NOISE_TAGS
        if not genres:
            continue
        if profile_counts.get(genres, 0) >= args.max_per_profile:
            continue
        if any(genre_counts.get(g, 0) >= args.max_per_genre for g in genres):
            continue

        # 3. タグ重なりで「似たゲーム」を弾く（粗いジャンルが取りこぼす被りを検出）
        tags = set(get_game_tags(app_id, args.n_tags)) - TAG_NOISE
        if any(len(tags & prev) >= args.tag_overlap_threshold for _, _, prev in selected):
            continue

        selected.append((app_id, name, tags))
        profile_counts[genres] = profile_counts.get(genres, 0) + 1
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
        print(f"  採用 [{len(selected):2d}/{args.n_games}] {name[:34]:34s} "
              f"累計{total:>8,d}件  {'/'.join(sorted(genres))[:34]}")
        time.sleep(args.sleep)

    return selected


def main():
    parser = argparse.ArgumentParser(description='時系列予測用のレビュー収集')
    parser.add_argument('--years', type=float, default=3, help='遡る年数')
    parser.add_argument('--n-games', type=int, default=30, help='収集するゲーム数')
    parser.add_argument('--min-reviews', type=int, default=10000,
                        help='採用するゲームの累計レビュー数の下限')
    parser.add_argument('--pages', type=int, default=20,
                        help='母集団取得のページ数（1ページ約25件）')
    parser.add_argument('--max-per-genre', type=int, default=6,
                        help='1ジャンルあたりの最大採用数')
    parser.add_argument('--max-per-profile', type=int, default=2,
                        help='同じジャンル集合あたりの最大採用数')
    parser.add_argument('--n-tags', type=int, default=6, help='タグ重なり判定で見る上位タグ数')
    parser.add_argument('--tag-overlap-threshold', type=int, default=2,
                        help='この数以上タグが共通したら「似たゲーム」として弾く')
    parser.add_argument('--max-reviews-per-game', type=int, default=200000,
                        help='1ゲームあたりの安全弁（期間で足りるはずだが暴走を防ぐ）')
    parser.add_argument('--seed', type=int, default=42, help='ゲーム選定のランダムシード')
    parser.add_argument('--sleep', type=float, default=0.5, help='リクエスト間の待機秒数')
    parser.add_argument('--output', default='data/timeseries/reviews_timeseries.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    since_ts = int(time.time() - args.years * 365 * 86400)
    since_date = dt.datetime.fromtimestamp(since_ts, dt.timezone.utc).date()

    print('=' * 70)
    print('時系列予測用レビュー収集（Issue #32）')
    print('=' * 70)
    print(f'  期間: {since_date} 以降（{args.years}年）')
    print(f'  対象: 累計{args.min_reviews:,}件以上のゲーム {args.n_games}本')
    print(f'  出力: {args.output}')

    # 1. 収集済みを確認（中断からの再開）
    already = load_collected_ids(args.output)
    if already:
        print(f'\n収集済み {len(already)}ゲームをスキップします')

    # 2. ゲームを選ぶ
    print('\n[1/2] ゲーム選定')
    games = select_games(args, already)
    if not games:
        print('  条件を満たすゲームが見つかりませんでした')
        return

    # 3. ゲームごとに期間指定で収集し、1本ずつ追記する
    print(f'\n[2/2] レビュー収集（{len(games)}ゲーム）')
    total_rows = 0
    for i, (app_id, name, _tags) in enumerate(games, 1):
        try:
            reviews = collect_natural_reviews(
                app_id=app_id, since_ts=since_ts,
                max_reviews=args.max_reviews_per_game,
            )
        except Exception as exc:  # 1本失敗しても全体を止めない
            print(f"  [{i:2d}/{len(games)}] {name[:30]:30s} 失敗: {exc}")
            continue

        rows = [dict(r, game_id=app_id, game_name=name) for r in reviews]
        append_rows(args.output, rows)
        total_rows += len(rows)

        pos = sum(1 for r in reviews if r['voted_up'])
        ratio = pos / len(reviews) if reviews else 0
        print(f"  [{i:2d}/{len(games)}] {name[:30]:30s} {len(rows):>6,d}件  "
              f"ポジ率{ratio:>5.1%}  （累計 {total_rows:,}件）")
        time.sleep(args.sleep)

    print(f'\n✓ 完了: {total_rows:,}件 → {args.output}')
    print('  ポジ率が実態（学習7ゲームで88.8%前後）に近ければ、自然比率のまま'
          '集められている')


if __name__ == '__main__':
    main()
