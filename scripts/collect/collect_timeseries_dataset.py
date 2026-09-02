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
- 保存: ゲーム1本ごとに追記。中断しても再開できる
- 網羅性: ゲーム別に「指定期間を全部カバーできたか」を collection_log.csv に記録する。
  再開時にスキップするのはカバーできたゲームだけで、途中で切れたものは収集し直す
  （Steamは連続アクセスに対し空ページを返して黙って打ち切ることがあるため）

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
    STOP_EXHAUSTED,
    STOP_REACHED_SINCE,
    collect_natural_reviews,
    get_popular_games,
    get_release_date,
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


LOG_FIELDS = ['app_id', 'name', 'rows', 'oldest', 'newest', 'coverage',
              'stop_reason', 'collected_at']


def fmt_date(ts: int) -> str:
    """Unix時刻を YYYY-MM-DD にする（0なら空文字）"""
    if not ts:
        return ''
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime('%Y-%m-%d')


def load_collection_log(path: str) -> dict:
    """ゲーム別の収集結果を読む（app_id -> 記録）"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {int(r['app_id']): r for r in csv.DictReader(f) if r.get('app_id')}


def append_log(path: str, row: dict) -> None:
    """収集結果を1ゲーム分追記する"""
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_collected_ids(path: str, log_path: str) -> set:
    """
    再収集をスキップしてよいapp_idを返す（中断からの再開用）

    スキップするのは**期間を全部カバーできたゲームだけ**。途中で切れたゲームを
    「収集済み」として飛ばすと、切れたまま二度と直らない。
    ログが「収集済み」と言っていても本体CSVに行が無ければ収集し直す
    （CSVだけ消して取り直す運用で、何も集まらなくなるのを防ぐ）
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        in_csv = {int(r['game_id']) for r in csv.DictReader(f) if r.get('game_id')}

    log = load_collection_log(log_path)
    if log:
        return {app_id for app_id, r in log.items()
                if r.get('coverage') == 'ok' and app_id in in_csv}

    # ログが無い（旧バージョンで収集した）場合はCSVの有無だけで判断する
    return in_csv


def drop_game_rows(path: str, app_ids: set) -> int:
    """再収集するゲームの行を出力CSVから取り除く（重複追記を防ぐ）"""
    if not app_ids or not os.path.exists(path):
        return 0
    tmp = path + '.tmp'
    removed = 0
    with open(path, encoding='utf-8') as src, \
            open(tmp, 'w', newline='', encoding='utf-8') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        for row in reader:
            if row.get('game_id') and int(row['game_id']) in app_ids:
                removed += 1
                continue
            writer.writerow(row)
    os.replace(tmp, path)
    return removed


def judge_coverage(reason: str, oldest: int, since_ts: int, release_date: str,
                   slack_days: int = 30) -> str:
    """
    期間を全部カバーできたかを判定する

    'ok'      指定期間の先頭まで遡れた／発売がウィンドウ内でAPIを取り切った
    'partial' 途中で止まった。直近しか無いので合算すると偽の成長を作る
    'unknown' 発売日が取れず、'ok' と 'partial' を区別できない
    """
    if reason == STOP_REACHED_SINCE:
        return 'ok'
    if not oldest:
        return 'partial'
    if reason == STOP_EXHAUSTED:
        if not release_date:
            return 'unknown'
        try:
            released = dt.datetime.strptime(release_date, '%Y-%m-%d').replace(
                tzinfo=dt.timezone.utc).timestamp()
        except ValueError:
            return 'unknown'
        # 発売がウィンドウ内なら、それ以上古いレビューは存在しない＝取り切れている
        return 'ok' if oldest <= released + slack_days * 86400 else 'partial'
    return 'partial'


def append_rows(path: str, rows: list) -> None:
    """1ゲーム分を追記する。ゲーム単位で書くので中断しても途中まで残る"""
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


GAME_FIELDS = ['app_id', 'name', 'genres', 'tags', 'total_reviews',
               'total_positive', 'total_negative', 'release_date']


def save_game_master(path: str, games: list) -> None:
    """選んだゲームの台帳を保存する（ジャンル・タグ・累計レビュー数）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=GAME_FIELDS)
        writer.writeheader()
        writer.writerows(games)


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
        time.sleep(args.sleep)  # 弾いた候補でもリクエストは投げているので待つ
        total = summary.get('total_reviews', 0)
        if total < args.min_reviews:
            continue

        # 2. ジャンル判定（ノイズタグを除いた実ジャンルが取れるものだけ）
        genres = get_game_genres(app_id) - NOISE_TAGS
        time.sleep(args.sleep)
        if not genres:
            continue
        if profile_counts.get(genres, 0) >= args.max_per_profile:
            continue
        if any(genre_counts.get(g, 0) >= args.max_per_genre for g in genres):
            continue

        # 3. タグ重なりで「似たゲーム」を弾く（粗いジャンルが取りこぼす被りを検出）
        tags = set(get_game_tags(app_id, args.n_tags)) - TAG_NOISE
        time.sleep(args.sleep)
        if any(len(tags & set(g['tags'].split('|'))) >= args.tag_overlap_threshold
               for g in selected):
            continue

        # 発売日は「収集が途中で切れたのか、発売がその期間内なのか」の判定に使う
        selected.append({
            'app_id': app_id,
            'name': name,
            'genres': '|'.join(sorted(genres)),
            'tags': '|'.join(sorted(tags)),
            'total_reviews': total,
            'total_positive': summary.get('total_positive', 0),
            'total_negative': summary.get('total_negative', 0),
            'release_date': get_release_date(app_id),
        })
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
    parser.add_argument('--max-reviews-per-game', type=int, default=400000,
                        help='1ゲームあたりの安全弁。#31の実測最大 234件/日 × 3年 = 約26万件'
                             'なので余裕を持たせている。当たると期間を全部カバーできない')
    parser.add_argument('--seed', type=int, default=42, help='ゲーム選定のランダムシード')
    parser.add_argument('--sleep', type=float, default=0.5, help='リクエスト間の待機秒数')
    parser.add_argument('--output', default='data/timeseries/reviews_timeseries.csv')
    parser.add_argument('--games-output', default='data/timeseries/games.csv',
                        help='選んだゲームの台帳（ジャンル・タグ・累計レビュー数）')
    parser.add_argument('--log-output', default='data/timeseries/collection_log.csv',
                        help='ゲーム別の収集結果（期間を全部カバーできたかの記録）')
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
    already = load_collected_ids(args.output, args.log_output)
    if already:
        print(f'\n期間を全部カバーできた {len(already)}ゲームをスキップします')

    # 2. ゲームを選ぶ
    print('\n[1/2] ゲーム選定')
    games = select_games(args, already)
    if not games:
        print('  条件を満たすゲームが見つかりませんでした')
        return

    # 3. ゲーム台帳を保存する。ジャンル・タグは選定時にしか手元に無いので、
    #    ここで残さないと後から偏りを分析できない（再取得が必要になる）
    save_game_master(args.games_output, games)
    print(f'  台帳を保存: {args.games_output}')

    # 4. 再収集するゲームの古い行を消す（追記方式なので放置すると二重になる）
    removed = drop_game_rows(args.output, {g['app_id'] for g in games})
    if removed:
        print(f'  再収集のため既存 {removed:,}行を削除')

    # 5. ゲームごとに期間指定で収集し、1本ずつ追記する
    print(f'\n[2/2] レビュー収集（{len(games)}ゲーム）')
    total_rows = 0
    incomplete = []
    for i, game in enumerate(games, 1):
        app_id, name = game['app_id'], game['name']
        log = {'app_id': app_id, 'name': name,
               'collected_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        try:
            reviews, reason = collect_natural_reviews(
                app_id=app_id, since_ts=since_ts,
                max_reviews=args.max_reviews_per_game, sleep=args.sleep,
            )
        except Exception as exc:  # 1本失敗しても全体を止めない
            print(f"  [{i:2d}/{len(games)}] {name[:30]:30s} 失敗: {exc}")
            append_log(args.log_output, dict(log, rows=0, oldest='', newest='',
                                             coverage='partial', stop_reason=f'error: {exc}'))
            incomplete.append((name, f'error: {exc}'))
            continue

        rows = [dict(r, game_id=app_id, game_name=name) for r in reviews]
        append_rows(args.output, rows)
        total_rows += len(rows)

        # 期間を全部カバーできたかを判定する。直近しか取れていないゲームを黙って
        # 混ぜると、合算したときに「参加ゲームが増えただけの偽の成長」を作る
        oldest = min((r['timestamp_created'] for r in reviews), default=0)
        newest = max((r['timestamp_created'] for r in reviews), default=0)
        coverage = judge_coverage(reason, oldest, since_ts, game.get('release_date', ''))
        pos = sum(1 for r in reviews if r['voted_up'])
        ratio = pos / len(reviews) if reviews else 0
        append_log(args.log_output, dict(
            log, rows=len(rows), oldest=fmt_date(oldest), newest=fmt_date(newest),
            coverage=coverage, stop_reason=reason))

        note = ''
        if coverage != 'ok':
            missing = (oldest - since_ts) / 86400 if oldest else 0
            note = f'  ⚠️ 期間未達（{missing:.0f}日分不足・{coverage}/{reason[:24]}）'
            incomplete.append((name, reason))
        print(f"  [{i:2d}/{len(games)}] {name[:30]:30s} {len(rows):>6,d}件  "
              f"ポジ率{ratio:>5.1%}  最古{fmt_date(oldest)}  （累計 {total_rows:,}件）{note}")
        time.sleep(args.sleep)

    print(f'\n✓ 完了: {total_rows:,}件 → {args.output}')
    print(f'  収集結果の記録: {args.log_output}')
    if incomplete:
        print(f'\n⚠️ {len(incomplete)}本が期間を全部カバーできていない:')
        for name, reason in incomplete:
            print(f'    - {name[:40]:40s} {reason[:60]}')
        print('  これらは直近だけに存在するため、合算すると偽の成長を作る。')
        print('  もう一度同じコマンドを実行すると、この分だけ収集し直す。')
    else:
        print('  全ゲームが指定期間を全部カバーできている')
    print(f'\n次: python scripts/collect/inspect_timeseries_dataset.py で偏りを点検する')


if __name__ == '__main__':
    main()
