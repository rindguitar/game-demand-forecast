"""
時系列予測用のレビュー収集スクリプト（Issue #32）

需要スコアの材料になるレビューを、期間で区切って自然比率のまま集める。
感情分析用（collect_dataset_10k.py）とは必要なデータの性質が正反対なので別スクリプト。

  感情分析用  各ゲーム714件で打ち切り・ポジ/ネガを50:50に強制
  時系列用    期間で区切る・比率は元のまま

理由は Wiki「学習用データと測定用データの違い」を参照。要点は、均衡させると
「何が求められているか」の測定結果が収集方法の産物になってしまうこと。

設計:
- 母集団: get_popular_games()（売上上位＝レビューが豊富）→ メタ情報をキャッシュに集める
- 採用条件: 累計レビュー数が閾値以上（実測で1万件未満は 2.12件/日まで落ちる）かつ
  発売から1年以上（履歴が短すぎるゲームは時系列に寄与しない）
- 選定は3つの条件を同時に満たす組を作る（詳細は select_from_pool()）
  1. 各ジャンル最低3本  2. 土台は14本まで  3. タグが2個以上重なるゲームは入れない
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
import json
import time
import random
import argparse
import datetime as dt
from collections import Counter

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

    # ログが無い（旧バージョンで収集した）場合はCSVの有無だけで判断する。
    # 期間を全部カバーできたかは分からないので、その旨を伝える
    if in_csv:
        print(f'⚠️ 収集ログ（{log_path}）が無いため、既存CSVの{len(in_csv)}ゲームは'
              '網羅性を確認できません。')
        print('   途中で切れていても「済み」として飛ばされます。'
              '取り直す場合はCSVを削除してください')
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
                   slack_days: int = 7) -> str:
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
        # 発売がウィンドウ内なら、それ以上古いレビューは存在しない＝取り切れている。
        # slackを詰めているのは、Early Accessのゲームが 1.0 の日付を返すため。
        # 緩めるとEA期間ごと切れたデータを「取り切った」と誤判定する（実測: Hades II）
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
               'total_positive', 'total_negative', 'release_date', 'tier']


def save_game_master(path: str, games: list) -> None:
    """選んだゲームの台帳を保存する（ジャンル・タグ・発売日・層）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=GAME_FIELDS, extrasaction='ignore')
        writer.writeheader()
        for g in games:
            writer.writerow(dict(g,
                                 genres='|'.join(sorted(g['genres'])),
                                 tags='|'.join(sorted(g['tags']))))


def load_game_master(path: str) -> list:
    """保存済みの台帳を読む（選定をやり直さず、同じ24本を収集し直すため）"""
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['app_id'] = int(r['app_id'])
        r['genres'] = set(filter(None, r.get('genres', '').split('|')))
        r['tags'] = set(filter(None, r.get('tags', '').split('|')))
    return rows


def load_pool_cache(path: str) -> dict:
    """母集団のメタ情報を読む（取得済みならAPIを叩かない）"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_pool_cache(path: str, cache: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def build_pool(args) -> list:
    """
    母集団のメタ情報（レビュー数・発売日・ジャンル・タグ）を集める

    ジャンルの下限を満たすには候補全部のジャンルを先に知っている必要があるため、
    1本ずつ即決せず、いったん全件をキャッシュに集めてから選ぶ。取得済みのものは
    APIを叩かないので、条件を変えた選び直しは数秒で終わる。
    """
    cache = load_pool_cache(args.pool_cache)
    if args.refresh_pool or '__order__' not in cache:
        listing = get_popular_games(n_pages=args.pages)
        cache['__order__'] = [[app_id, name] for app_id, name in listing]
        print(f'  母集団を取得: {len(listing)}本')
    listing = cache['__order__']

    pool, fetched, failed = [], 0, 0
    for i, (app_id, name) in enumerate(listing, 1):
        rec = cache.setdefault(str(app_id), {})
        rec.setdefault('name', name)
        try:
            # 1. レビュー数（1リクエスト。ここで6割が落ちるので最初に見る）
            if 'total_reviews' not in rec:
                summary = get_review_summary(app_id)
                rec.update({k: summary.get(k, 0) for k in
                            ('total_reviews', 'total_positive', 'total_negative')})
                fetched += 1
                time.sleep(args.sleep)
            if rec.get('total_reviews', 0) < args.min_reviews:
                continue

            # 2. 発売日 → 3. ジャンル → 4. タグ（下限を通った候補にだけ聞く）
            if 'release_date' not in rec:
                rec['release_date'] = get_release_date(app_id)
                fetched += 1
                time.sleep(args.sleep)
            if 'genres' not in rec:
                rec['genres'] = sorted(get_game_genres(app_id))
                fetched += 1
                time.sleep(args.sleep)
            if 'tags' not in rec:
                rec['tags'] = get_game_tags(app_id, args.n_tags)
                fetched += 1
                time.sleep(args.sleep)
        except Exception as exc:  # 1本の失敗で母集団作りを止めない
            failed += 1
            print(f'  取得失敗 {name[:30]}: {exc}')
            continue
        finally:
            if i % 50 == 0:
                save_pool_cache(args.pool_cache, cache)

        genres = frozenset(rec['genres']) - NOISE_TAGS
        if not genres or not rec.get('release_date'):
            continue
        pool.append({
            'app_id': app_id,
            'name': rec['name'],
            'genres': genres,
            'tags': set(rec.get('tags') or ()) - TAG_NOISE,
            'release_date': rec['release_date'],
            'total_reviews': rec['total_reviews'],
            'total_positive': rec.get('total_positive', 0),
            'total_negative': rec.get('total_negative', 0),
        })

    save_pool_cache(args.pool_cache, cache)
    print(f'  候補 {len(pool)}本（母集団{len(listing)}本 / 新規取得{fetched}件'
          f'{" / 失敗" + str(failed) + "本" if failed else ""}）')
    return pool


def select_from_pool(pool: list, window_start: str, min_history: str, args) -> tuple:
    """
    収集する対象を選ぶ。3つの条件を同時に満たす組を作る

      1. 各ジャンル最低 genre_floor 本
         1本しか入らないジャンルは、その1本の当たり外れがジャンルの結論になる
      2. 土台（発売がウィンドウ開始より前）は max_backbone 本まで
         土台は時系列の線を引く役だが、増やすほど新しい話題が入らなくなる。
         上限にしているので、残り（n_games - max_backbone）が新しい側の下限になる
      3. タグが tag_overlap_threshold 個以上重なるゲームは入れない
         同じ需要を二重に数えないため

    希少なジャンルから先に埋める。頻出ジャンルから埋めると枠を使い切ってしまい、
    Racing や Sports の番が来たときに残りが無くなる。

    Returns:
        (選んだゲームのリスト, ジャンル別の本数)
    """
    candidates = [dict(g, backbone=g['release_date'] <= window_start)
                  for g in pool if g['release_date'] <= min_history]
    random.Random(args.seed).shuffle(candidates)

    selected, genre_counts = [], {}

    def acceptable(game):
        if game['backbone'] and sum(1 for g in selected if g['backbone']) >= args.max_backbone:
            return False
        if any(genre_counts.get(x, 0) >= args.max_per_genre for x in game['genres']):
            return False
        return not any(len(game['tags'] & g['tags']) >= args.tag_overlap_threshold
                       for g in selected)

    def take(game):
        selected.append(game)
        for x in game['genres']:
            genre_counts[x] = genre_counts.get(x, 0) + 1

    # 1. 希少なジャンルから下限を埋める（期間の頭を埋められるのは土台だけなので土台を優先）
    rarity = Counter(x for g in candidates for x in g['genres'])
    for genre in sorted(rarity, key=lambda x: rarity[x]):
        while genre_counts.get(genre, 0) < args.genre_floor and len(selected) < args.n_games:
            pick = (next((g for g in candidates if g not in selected and genre in g['genres']
                          and g['backbone'] and acceptable(g)), None)
                    or next((g for g in candidates if g not in selected and genre in g['genres']
                             and acceptable(g)), None))
            if not pick:
                break
            take(pick)

    # 2. 残りを埋める
    while len(selected) < args.n_games:
        pick = next((g for g in candidates if g not in selected and acceptable(g)), None)
        if not pick:
            break
        take(pick)

    return selected, genre_counts


def report_selection(games: list, genre_counts: dict, pool: list, args) -> None:
    """選定結果の内訳を出す（層とジャンルの配分がそのまま需要スコアの配分になる）"""
    tiers = Counter(g['tier'] for g in games)
    print(f"  {len(games)}本を選定 — 土台{tiers['土台']} / 中間{tiers['中間']} / "
          f"直近{tiers['直近']}")
    for g in sorted(games, key=lambda x: x['release_date']):
        print(f"  {g['tier']} {g['name'][:32]:32s} {g['release_date']} "
              f"{g['total_reviews']:>8,d}件  {'/'.join(sorted(g['genres']))[:30]}")
    print(f"  ジャンル別: {dict(sorted(genre_counts.items(), key=lambda x: -x[1]))}")
    short = sorted(x for x in {y for g in pool for y in g['genres']}
                   if genre_counts.get(x, 0) < args.genre_floor)
    if short:
        print(f"  ⚠️ 下限{args.genre_floor}本に届かなかったジャンル: {short}")


def main():
    parser = argparse.ArgumentParser(description='時系列予測用のレビュー収集')
    parser.add_argument('--years', type=float, default=3, help='遡る年数')
    parser.add_argument('--n-games', type=int, default=24, help='収集するゲーム数')
    parser.add_argument('--min-reviews', type=int, default=10000,
                        help='採用するゲームの累計レビュー数の下限')
    parser.add_argument('--pages', type=int, default=20,
                        help='母集団取得のページ数（1ページ約25件）')
    parser.add_argument('--genre-floor', type=int, default=3,
                        help='1ジャンルあたりの最低採用数。1本だとその1本の当たり外れが'
                             'ジャンルの結論になる')
    parser.add_argument('--max-backbone', type=int, default=14,
                        help='土台（発売がウィンドウ開始より前）の上限。'
                             'n-games から引いた数が「新しい側」の下限になる')
    parser.add_argument('--max-per-genre', type=int, default=12,
                        help='1ジャンルあたりの最大採用数（1ジャンルが全部を占める退化を防ぐ保険）')
    parser.add_argument('--min-history-years', type=float, default=1.0,
                        help='この年数より後に発売したゲームは対象外（履歴が短すぎる）')
    parser.add_argument('--recent-years', type=float, default=2.0,
                        help='この年数以内の発売を「直近」として台帳に記録する')
    parser.add_argument('--pool-cache', default='data/timeseries/pool_cache.json',
                        help='母集団のメタ情報のキャッシュ。2回目以降はAPIを叩かない')
    parser.add_argument('--refresh-pool', action='store_true',
                        help='母集団の一覧を取り直す（売上上位は日々入れ替わる）')
    parser.add_argument('--reselect', action='store_true',
                        help='既存の台帳を捨ててゲームを選び直す')
    parser.add_argument('--dry-run', action='store_true',
                        help='選定だけ行い、収集はしない（条件を変えて顔ぶれを確認する用）')
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

    # 2. 収集対象を決める。台帳があればそれを使う（実行のたびに顔ぶれが変わらないように）
    if os.path.exists(args.games_output) and not args.reselect:
        games = load_game_master(args.games_output)
        print(f'\n[1/2] 既存の台帳を使用: {len(games)}本（選び直すなら --reselect）')
    else:
        print('\n[1/2] ゲーム選定')
        today = dt.datetime.now(dt.timezone.utc).date()
        window_start = since_date.isoformat()
        min_history = (today - dt.timedelta(days=int(365 * args.min_history_years))).isoformat()
        recent_from = (today - dt.timedelta(days=int(365 * args.recent_years))).isoformat()

        pool = build_pool(args)
        games, genre_counts = select_from_pool(pool, window_start, min_history, args)
        if not games:
            print('  条件を満たすゲームが見つかりませんでした')
            return
        for g in games:
            g['tier'] = ('土台' if g['release_date'] <= window_start
                         else '直近' if g['release_date'] > recent_from else '中間')
        report_selection(games, genre_counts, pool, args)
        save_game_master(args.games_output, games)
        print(f'  台帳を保存: {args.games_output}')

    if args.dry_run:
        print('\n--dry-run のため収集は行わない')
        return

    # 3. まだ期間を全部カバーできていないゲームだけを収集する
    targets = [g for g in games if g['app_id'] not in already]
    if not targets:
        print('\n全ゲームが期間を全部カバー済み。収集するものはありません')
        return

    # 4. 再収集するゲームの古い行を消す（追記方式なので放置すると二重になる）
    removed = drop_game_rows(args.output, {g['app_id'] for g in targets})
    if removed:
        print(f'  再収集のため既存 {removed:,}行を削除')

    # 5. ゲームごとに期間指定で収集し、1本ずつ追記する
    print(f'\n[2/2] レビュー収集（{len(targets)}ゲーム）')
    total_rows = 0
    incomplete = []
    for i, game in enumerate(targets, 1):
        app_id, name = game['app_id'], game['name']
        log = {'app_id': app_id, 'name': name,
               'collected_at': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        try:
            reviews, reason = collect_natural_reviews(
                app_id=app_id, since_ts=since_ts,
                max_reviews=args.max_reviews_per_game, sleep=args.sleep,
            )
        except Exception as exc:  # 1本失敗しても全体を止めない
            print(f"  [{i:2d}/{len(targets)}] {name[:30]:30s} 失敗: {exc}")
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
        print(f"  [{i:2d}/{len(targets)}] {name[:30]:30s} {len(rows):>6,d}件  "
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
