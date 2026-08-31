"""
時系列データの feasibility 測定（Issue #31）

時系列用データの収集（Issue #32）を設計する前に、「1ゲームが単位時間あたり何件の
使えるレビューを生むか」を実測する。集計の粒度（日次/週次）・必要ゲーム数・遡る期間は
この値からしか決められない。

測定するもの:
  - 平均レート（全ゲーム）: 有効件数 ÷ 日付範囲
  - 日次分布（--daily-sample の数本）: ゼロの日がどれだけあるか
  - 英語フィルタ通過率: 生件数と is_valid_english_review() 通過後の件数の比
  - 自然なポジ/ネガ比率: query_summary から（追加リクエスト不要）

get_steam_reviews() を使わないのは、同関数がフィルタ通過後のレビューしか返さず、
生件数＝フィルタ通過率が取れないため。

使い方は --help を参照。
"""

import os
import sys
import csv
import time
import argparse
import datetime as dt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.steam_collector import (  # noqa: E402
    HEADERS,
    get_popular_games,
    is_valid_english_review,
    request_with_backoff,
)

REVIEWS_URL = 'https://store.steampowered.com/appreviews/'
TRAIN_CSV = 'data/train/reviews_10000.csv'
OOD_CSV = 'data/test/reviews_ood_2000.csv'


def load_games_from_csv(path: str) -> list:
    """収集済みCSVから (app_id, game_name) のユニークなリストを作る"""
    with open(path, encoding='utf-8') as f:
        rows = {(int(r['game_id']), r['game_name']) for r in csv.DictReader(f)}
    return sorted(rows)


def build_target_games(pages: int, band_size: int) -> list:
    """
    測定対象を組み立てる。

    学習7とOOD20だけでは全てレビューが豊富なゲームに偏り、密度が上限寄りに出る。
    規模による幅を見るため、売上上位の母集団から先頭/中間/末尾を足す。
    """
    games, seen = [], set()

    def add(items, band):
        for app_id, name in items:
            if app_id in seen:
                continue
            seen.add(app_id)
            games.append({'app_id': app_id, 'name': name, 'rank_band': band})

    # 1. 既存の収集済みCSVから読む（app_idの重複定義を避けるため）
    add(load_games_from_csv(TRAIN_CSV), 'train7')
    add(load_games_from_csv(OOD_CSV), 'ood20')

    # 2. 売上上位の母集団から3つの帯を抜く
    popular = get_popular_games(n_pages=pages)
    mid_start = (len(popular) - band_size) // 2
    add(popular[:band_size], 'top')
    add(popular[mid_start:mid_start + band_size], 'mid')
    add(popular[-band_size:], 'low')

    return games


def fetch_review_stats(app_id: int, max_valid: int, max_days: int,
                       max_raw: int = 0, sleep: float = 0.5,
                       max_retries: int = 5) -> dict:
    """
    レビューを新しい順に辿り、生件数・有効件数・日付ごとの件数を数える。

    打ち切りは max_valid / max_days / max_raw のいずれか早い方。どれで止まったかは
    stop_reason に入れる（件数で打ち切った場合、レートは「その値以上」と読む）。
    """
    params = {
        'json': 1,
        'language': 'english',
        'filter': 'recent',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': 100,
    }
    cutoff = time.time() - max_days * 86400
    daily = defaultdict(lambda: {'raw': 0, 'valid': 0})
    raw = valid = 0
    min_ts, max_ts = 0, 0
    summary, cursor, stop = {}, '*', 'exhausted'
    seen_cursors = set()

    while True:
        params['cursor'] = cursor
        response = request_with_backoff(
            f'{REVIEWS_URL}{app_id}', params=params, headers=HEADERS,
            timeout=20, max_retries=max_retries
        )
        data = response.json()

        if data.get('success') != 1:
            stop = 'api_error'
            break

        # query_summary は初回ページにのみ入る（自然比率の取得元）
        if not summary:
            summary = data.get('query_summary', {})

        reviews = data.get('reviews', [])
        if not reviews:
            break

        for review in reviews:
            ts = review.get('timestamp_created', 0)
            date = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
            raw += 1
            daily[date]['raw'] += 1
            min_ts = min(min_ts, ts) if min_ts else ts
            max_ts = max(max_ts, ts)
            if is_valid_english_review(review.get('review', '')):
                valid += 1
                daily[date]['valid'] += 1

        # 打ち切り判定
        if valid >= max_valid:
            stop = 'max_valid'
            break
        if reviews[-1].get('timestamp_created', 0) < cutoff:
            stop = 'max_days'
            break
        if max_raw and raw >= max_raw:
            stop = 'max_raw'
            break

        # cursorが進まない場合があるので無限ループを防ぐ
        next_cursor = data.get('cursor')
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
        time.sleep(sleep)

    return {'raw': raw, 'valid': valid, 'daily': dict(daily),
            'min_ts': min_ts, 'max_ts': max_ts,
            'summary': summary, 'stop_reason': stop}


def summarize_game(game: dict, stats: dict) -> dict:
    """
    1ゲーム分の測定結果を、CSVに書ける1行にまとめる。

    レートは暦日数ではなく実際の時間差で割る。最古日と最新日は取得の途中で切れた
    不完全な日なので、暦日数で割るとレートを過小評価するため。
    """
    dates = sorted(stats['daily'])
    if dates:
        first, last = dt.date.fromisoformat(dates[0]), dt.date.fromisoformat(dates[-1])
        span_days = (last - first).days + 1
    else:
        first = last = None
        span_days = 0

    elapsed_days = (stats['max_ts'] - stats['min_ts']) / 86400
    summary = stats['summary']
    total = summary.get('total_reviews', 0)
    positive = summary.get('total_positive', 0)

    return {
        'app_id': game['app_id'],
        'name': game['name'],
        'rank_band': game['rank_band'],
        'raw_fetched': stats['raw'],
        'valid_fetched': stats['valid'],
        'pass_rate': round(stats['valid'] / stats['raw'], 4) if stats['raw'] else 0,
        'first_date': first.isoformat() if first else '',
        'last_date': last.isoformat() if last else '',
        'span_days': span_days,
        'elapsed_days': round(elapsed_days, 2),
        'valid_per_day': round(stats['valid'] / elapsed_days, 2) if elapsed_days > 0 else 0,
        'stop_reason': stats['stop_reason'],
        'total_reviews': total,
        'total_positive': positive,
        'total_negative': summary.get('total_negative', 0),
        'pos_ratio': round(positive / total, 4) if total else '',
    }


def expand_daily_rows(app_id: int, daily: dict) -> list:
    """
    日付ごとの件数を、欠損日をゼロで埋めた行のリストにする。

    最新日（まだ進行中）と最古日（取得を打ち切った途中）は不完全な観測なので
    is_boundary を立てる。ゼロの日を数えるときに除外しないと、実際には無い
    「空白の日」を数えてしまうため。
    """
    if not daily:
        return []
    dates = sorted(daily)
    first, last = dt.date.fromisoformat(dates[0]), dt.date.fromisoformat(dates[-1])

    rows = []
    day = first
    while day <= last:
        key = day.isoformat()
        counts = daily.get(key, {'raw': 0, 'valid': 0})
        rows.append({'app_id': app_id, 'date': key,
                     'raw': counts['raw'], 'valid': counts['valid'],
                     'is_boundary': 1 if day in (first, last) else 0})
        day += dt.timedelta(days=1)
    return rows


def estimate_granularity(rate: float, n_games: int, outlier_rate: float,
                         topic_counts: list, target_per_cell: int) -> list:
    """
    粒度ごとに「1マスあたり件数」と「1マスに target 件を満たすのに必要なゲーム数」を試算。

        1マスあたり件数 = ゲーム数 × 件/日 × 区切りの日数 × (1 - Outlier率) ÷ トピック数
    """
    rows = []
    usable = rate * (1 - outlier_rate)
    for bucket, days in (('日次', 1), ('週次', 7)):
        for topics in topic_counts:
            per_cell = n_games * usable * days / topics if topics else 0
            need = target_per_cell * topics / (usable * days) if usable else float('inf')
            rows.append({
                'bucket': bucket,
                'topics': topics,
                'per_cell': round(per_cell, 2),
                'need_games': int(need) + 1 if need != float('inf') else -1,
            })
    return rows


def stratify_by_reviews(rate_rows: list, thresholds: list, outlier_rate: float,
                        topic_counts: list, target_per_cell: int) -> list:
    """
    累計レビュー数で層別し、層ごとに必要ゲーム数を試算する。

    rank_band で層別しないのは、帯が検索結果の並び順でしかなく、レビュー数の多寡を
    表さないため。レートはゲームの規模で桁が変わるので、実測した total_reviews で分ける。
    """
    out = []
    for threshold in thresholds:
        sub = [r for r in rate_rows
               if r['total_reviews'] >= threshold and r['valid_per_day'] > 0]
        if not sub:
            continue
        med = median([r['valid_per_day'] for r in sub])
        for est in estimate_granularity(med, len(sub), outlier_rate,
                                        topic_counts, target_per_cell):
            out.append({'threshold': threshold, 'n_games': len(sub),
                        'median_rate': round(med, 2), **est})
    return out


def print_estimates(med: float, est: list) -> None:
    """層別した試算結果を標準出力に表として出す"""
    print(f'  全体中央値 {med:.2f}件/日')
    print(f"  {'累計レビュー':>12} {'該当':>4} {'件/日':>8} {'粒度':<6} "
          f"{'トピック':>7} {'1マス':>8} {'必要ゲーム数':>12}")
    for e in est:
        label = f"{e['threshold']:,}以上" if e['threshold'] else '全て'
        print(f"  {label:>12} {e['n_games']:>4d} {e['median_rate']:>8.2f} "
              f"{e['bucket']:<6} {e['topics']:>7d} {e['per_cell']:>8.2f} "
              f"{e['need_games']:>12d}")


def load_previous_results(out_dir: str) -> tuple:
    """
    保存済みのCSVを読み直す（--summary-only 用）。

    閾値やOutlier率を変えて試算し直すたびにAPIを叩き直すのは無駄なため。
    CSVは全て文字列で返るので、集計に使う列だけ数値へ戻す。
    """
    def read(name):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding='utf-8') as f:
            return list(csv.DictReader(f))

    rate_rows = read('review_rate.csv')
    for row in rate_rows:
        for key in ('raw_fetched', 'valid_fetched', 'span_days', 'total_reviews',
                    'total_positive', 'total_negative'):
            row[key] = int(row[key] or 0)
        for key in ('pass_rate', 'elapsed_days', 'valid_per_day'):
            row[key] = float(row[key] or 0)

    daily_rows = read('daily_counts.csv')
    for row in daily_rows:
        for key in ('raw', 'valid', 'is_boundary'):
            row[key] = int(row[key] or 0)

    return rate_rows, daily_rows


def write_csv(path: str, rows: list, fields: list) -> None:
    """辞書のリストをCSVに書き出す"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(values: list) -> float:
    """中央値（プレイ時間同様、少数の突出値に引きずられないため平均は使わない）"""
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def pick_daily_samples(rows: list, per_band: int = 1) -> list:
    """日次分布を見るゲームを、人気度の帯ごとに選ぶ"""
    picked = []
    for band in ('top', 'mid', 'low', 'train7'):
        in_band = [r for r in rows if r['rank_band'] == band and r['valid_per_day'] > 0]
        picked.extend(in_band[:per_band])
    return picked


def main():
    parser = argparse.ArgumentParser(description='時系列データの feasibility 測定')
    parser.add_argument('--out-dir', default='data/experiments/timeseries_feasibility')
    parser.add_argument('--limit', type=int, default=0,
                        help='先頭N本だけ測る（0=全件・動作確認用）')
    parser.add_argument('--pages', type=int, default=20,
                        help='get_popular_games のページ数（1ページ約25件）')
    parser.add_argument('--band-size', type=int, default=10,
                        help='ランキング上位/中位/下位から抜く本数')
    parser.add_argument('--max-valid', type=int, default=300,
                        help='平均レート測定の打ち切り（有効レビュー件数）')
    parser.add_argument('--max-days', type=int, default=180,
                        help='平均レート測定の打ち切り（遡る日数）')
    parser.add_argument('--daily-max-days', type=int, default=90,
                        help='日次分布の対象期間')
    parser.add_argument('--daily-max-raw', type=int, default=3000,
                        help='日次分布の打ち切り（生レビュー件数）')
    parser.add_argument('--daily-per-band', type=int, default=1,
                        help='日次分布を見るゲーム数（人気度の帯ごと）')
    parser.add_argument('--outlier-rate', type=float, default=0.432,
                        help='トピック分類不能の割合（現行実績 43.2%%）')
    parser.add_argument('--topics', type=int, nargs='+', default=[86, 15],
                        help='試算するトピック数（現行86 / 束ねた場合15）')
    parser.add_argument('--target-per-cell', type=int, default=10,
                        help='1マスあたり確保したい件数')
    parser.add_argument('--review-thresholds', type=int, nargs='+',
                        default=[100000, 10000, 1000, 0],
                        help='累計レビュー数で層別する閾値（0=全件）')
    parser.add_argument('--sleep', type=float, default=0.5,
                        help='リクエスト間の待機秒数')
    parser.add_argument('--summary-only', action='store_true',
                        help='APIを叩かず、保存済みCSVから summary.md だけ作り直す')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    started = dt.datetime.now()

    print('=' * 70)
    print('時系列データの feasibility 測定（Issue #31）')
    print('=' * 70)

    if args.summary_only:
        rate_rows, daily_rows = load_previous_results(args.out_dir)
        if not rate_rows:
            print(f'\n{args.out_dir}/review_rate.csv がありません。まず測定を実行してください。')
            return
        print(f'\n保存済みの {len(rate_rows)}ゲーム分から再計算します')
        rates = [r['valid_per_day'] for r in rate_rows if r['valid_per_day'] > 0]
        med = median(rates)
        est = stratify_by_reviews(rate_rows, args.review_thresholds, args.outlier_rate,
                                  args.topics, args.target_per_cell)
        print_estimates(med, est)
        write_summary(args, started, rate_rows, rate_rows, daily_rows, med, est)
        print(f'\n✓ 更新: {args.out_dir}/summary.md')
        return

    # 1. 測定対象のゲームを組み立てる
    games = build_target_games(args.pages, args.band_size)
    if args.limit:
        games = games[:args.limit]
    bands = defaultdict(int)
    for g in games:
        bands[g['rank_band']] += 1
    print(f'\n対象: {len(games)}ゲーム  ' + ' / '.join(f'{k}={v}' for k, v in bands.items()))

    # 2. 全ゲームの平均レートを測る
    print(f'\n[1/3] 平均レート測定（打ち切り: 有効{args.max_valid}件 or {args.max_days}日）')
    rate_rows = []
    for i, game in enumerate(games, 1):
        stats = fetch_review_stats(game['app_id'], args.max_valid, args.max_days,
                                   sleep=args.sleep)
        row = summarize_game(game, stats)
        rate_rows.append(row)
        print(f"  [{i:3d}/{len(games)}] {row['name'][:28]:28s} "
              f"{row['valid_per_day']:8.2f}件/日  通過率{row['pass_rate']:.0%}  "
              f"({row['stop_reason']})")
        time.sleep(args.sleep)

    # 3. 一部のゲームだけ日次分布を測る（平均ではゼロの日が見えないため）
    samples = pick_daily_samples(rate_rows, args.daily_per_band)
    print(f'\n[2/3] 日次分布測定（{len(samples)}本 × 直近{args.daily_max_days}日）')
    daily_rows = []
    for row in samples:
        stats = fetch_review_stats(row['app_id'], max_valid=10 ** 9,
                                   max_days=args.daily_max_days,
                                   max_raw=args.daily_max_raw, sleep=args.sleep)
        expanded = expand_daily_rows(row['app_id'], stats['daily'])
        daily_rows.extend(expanded)
        complete = [r for r in expanded if not r['is_boundary']]
        zero_days = sum(1 for r in complete if r['valid'] == 0)
        ratio = zero_days / len(complete) if complete else 0
        print(f"  {row['name'][:28]:28s} 完全な{len(complete):4d}日中 "
              f"有効ゼロが{zero_days:4d}日 ({ratio:.0%})  ({stats['stop_reason']})")

    # 4. 累計レビュー数で層別して粒度を試算（レートはゲームの規模で桁が変わるため）
    rates = [r['valid_per_day'] for r in rate_rows if r['valid_per_day'] > 0]
    med = median(rates)
    est = stratify_by_reviews(rate_rows, args.review_thresholds, args.outlier_rate,
                              args.topics, args.target_per_cell)

    print('\n[3/3] 粒度の試算')
    print_estimates(med, est)

    # 5. 書き出し
    write_csv(os.path.join(args.out_dir, 'review_rate.csv'),
              rate_rows, list(rate_rows[0].keys()))
    if daily_rows:
        write_csv(os.path.join(args.out_dir, 'daily_counts.csv'),
                  daily_rows, ['app_id', 'date', 'raw', 'valid', 'is_boundary'])

    write_summary(args, started, games, rate_rows, daily_rows, med, est)
    print(f'\n✓ 出力: {args.out_dir}/')


def write_summary(args, started, games, rate_rows, daily_rows, med, est) -> None:
    """測定条件と結果の要点を summary.md に残す"""
    by_band = defaultdict(list)
    for r in rate_rows:
        if r['valid_per_day'] > 0:
            by_band[r['rank_band']].append(r['valid_per_day'])

    pass_rates = [r['pass_rate'] for r in rate_rows if r['raw_fetched']]
    pos = sum(r['total_positive'] for r in rate_rows)
    tot = sum(r['total_reviews'] for r in rate_rows)

    lines = [
        '# 時系列データの feasibility 測定結果',
        '',
        f'実行日: {started:%Y-%m-%d %H:%M}（測定するのは直近のペースのため、時期に依存する）',
        '',
        '## 測定条件',
        '',
        f'- 対象: {len(games)}ゲーム',
        f'- 平均レート: 有効{args.max_valid}件 または {args.max_days}日で打ち切り',
        f'- 日次分布: 直近{args.daily_max_days}日 または 生{args.daily_max_raw}件で打ち切り',
        '',
        '## 帯ごとの発生レート（件/日・中央値）',
        '',
        '| 帯 | ゲーム数 | 中央値 |',
        '|---|---|---|',
    ]
    for band in ('train7', 'ood20', 'top', 'mid', 'low'):
        if by_band.get(band):
            lines.append(f'| {band} | {len(by_band[band])} | {median(by_band[band]):.2f} |')

    lines += [
        '',
        f'全体の中央値: **{med:.2f} 件/日**',
        '',
        '## 英語フィルタ通過率',
        '',
        f'中央値 {median(pass_rates):.1%}'
        '（`query_summary` の件数から使える件数を概算する際の補正に使う）',
        '',
        '## 自然なポジ率（query_summary 合計）',
        '',
        f'{pos:,} / {tot:,} = **{pos / tot:.1%}**' if tot else '取得できず',
        '',
        '## 粒度の試算（累計レビュー数で層別）',
        '',
        f'1マスあたり件数 = ゲーム数 × 件/日 × 区切り日数 × (1 - {args.outlier_rate}) ÷ トピック数',
        '',
        'レートはゲームの規模で桁が変わるため、`rank_band` ではなく実測した累計レビュー数で層別する。',
        '',
        f'| 累計レビュー | 該当本数 | 件/日 | 粒度 | トピック数 | 1マス | 1マス{args.target_per_cell}件に必要なゲーム数 |',
        '|---|---|---|---|---|---|---|',
    ]
    for e in est:
        label = f"{e['threshold']:,}以上" if e['threshold'] else '全て'
        lines.append(f"| {label} | {e['n_games']} | {e['median_rate']:.2f} | {e['bucket']} "
                     f"| {e['topics']} | {e['per_cell']:.2f} | {e['need_games']} |")

    if daily_rows:
        complete = [r for r in daily_rows if not r['is_boundary']]
        zero = sum(1 for r in complete if r['valid'] == 0)
        lines += [
            '',
            '## 日次分布',
            '',
            f'完全に観測できた{len(complete)}日のうち、有効レビューがゼロの日は **{zero}日'
            f'（{zero / len(complete):.0%}）**。詳細は `daily_counts.csv`',
            '',
            '最新日（進行中）と最古日（取得を打ち切った途中）は不完全な観測のため、'
            '`is_boundary=1` を立てて上の集計から除いている。',
        ]

    lines += [
        '',
        '## 注意',
        '',
        '- **`rank_band` はレビュー数の多寡を表さない**。帯は母集団の並び順でしかないので、'
        '層別には実測した `total_reviews` を使うこと',
        '- 2026-08-31 の測定は `get_popular_games()` の修正前（`sort_by=Reviews_DESC` ＝'
        '評価スコア順）に取得したもの。同関数は売上上位（`filter=globaltopsellers`）を返すよう'
        '修正済みのため、測り直すと母集団が変わり、レートは上振れする見込み',
        '- `stop_reason` が `max_raw` の行は、レートが「その値以上」であることを示す',
        '',
    ]

    with open(os.path.join(args.out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
