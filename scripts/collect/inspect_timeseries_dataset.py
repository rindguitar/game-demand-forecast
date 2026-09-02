"""
時系列用データセットの偏りを点検する（Issue #32・Phase 5「データ品質レポート」）

需要スコアは実数合算で出すため、**集めたレビューの内訳がそのまま需要スコアの内訳に
なる**。つまり収集の偏りは、そのまま「需要の偏り」に化ける。出てきたランキングを
読む前に、その偏りがどれくらいあるかを知っておく必要がある。

点検するもの:
  1. 自然比率が保たれているか（実態は学習7ゲームで88.8%前後）
  2. ゲーム別のレビュー量シェア — 数本の大作が支配していないか
  3. ジャンル別の「本数シェア」と「レビュー量シェア」の乖離
     → 選定は本数で均衡させているが、量では偏りうる。ここが最も見落としやすい
  4. 期間ごとの参加ゲーム数の変化
     → 増えていると絶対数では偽の成長が出る（シェアを主軸にした理由）

APIを叩かず、収集済みCSVだけを読む。収集の途中でも実行できる。

使い方は --help を参照。
"""

import os
import sys
import csv
import argparse
import datetime as dt
from collections import defaultdict

csv.field_size_limit(sys.maxsize)  # レビュー本文が長い行に備える


def load_reviews(path: str) -> list:
    """収集済みレビューを読む（本文は使わないので捨ててメモリを節約）"""
    keep = ('game_id', 'game_name', 'voted_up', 'timestamp_created')
    with open(path, encoding='utf-8') as f:
        return [{k: r[k] for k in keep} for r in csv.DictReader(f)]


def load_games(path: str) -> dict:
    """ゲーム台帳を読む（app_id -> ジャンル等）"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {int(r['app_id']): r for r in csv.DictReader(f)}


def bar(ratio: float, width: int = 30) -> str:
    return '█' * int(ratio * width)


def report_ratio(reviews: list) -> None:
    """自然比率が保たれているかを見る"""
    pos = sum(1 for r in reviews if str(r['voted_up']).lower() in ('true', '1'))
    ratio = pos / len(reviews)
    print('\n## 1. 自然比率')
    print(f'  ポジ {pos:,} / 全体 {len(reviews):,} = {ratio:.1%}')
    if 0.80 <= ratio <= 0.95:
        print('  → 実態（学習7ゲームで88.8%）に近い。均衡させずに集められている')
    else:
        print('  → 実態(88.8%前後)から外れている。収集方法かゲーム構成を確認すること')


def report_game_concentration(reviews: list, top_n: int) -> None:
    """ゲーム別のレビュー量シェア。数本が支配していないかを見る"""
    counts = defaultdict(int)
    for r in reviews:
        counts[r['game_name']] += 1
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda x: -x[1])

    print(f'\n## 2. ゲーム別のレビュー量シェア（{len(counts)}本中 上位{top_n}）')
    for name, n in ranked[:top_n]:
        share = n / total
        print(f'  {name[:30]:30s} {n:>8,d}  {share:>6.1%}  {bar(share)}')

    for k in (1, 3, 5):
        if len(ranked) >= k:
            share = sum(n for _, n in ranked[:k]) / total
            print(f'  上位{k}本で {share:.1%} を占める')


def report_genre_divergence(reviews: list, games: dict) -> None:
    """
    ジャンルの「本数シェア」と「レビュー量シェア」の乖離を見る。

    選定は max_per_genre で本数を均衡させているが、量までは均衡しない。
    1本の大作が入ると、そのジャンルが需要スコアを支配する。
    """
    if not games:
        print('\n## 3. ジャンル別の偏り')
        print('  ゲーム台帳（games.csv）が無いためスキップ')
        return

    volume, titles = defaultdict(int), defaultdict(int)
    for r in reviews:
        info = games.get(int(r['game_id']))
        if not info:
            continue
        for g in info['genres'].split('|'):
            volume[g] += 1
    for info in games.values():
        for g in info['genres'].split('|'):
            titles[g] += 1

    total_vol = sum(volume.values())
    total_titles = sum(titles.values())
    if not total_vol:
        print('\n## 3. ジャンル別の偏り\n  ジャンル情報が突き合わせできなかった')
        return

    print('\n## 3. ジャンル別 — 本数シェア vs レビュー量シェア')
    print(f"  {'ジャンル':<22} {'本数':>4} {'本数比':>7} {'レビュー量':>10} {'量比':>7} {'乖離':>8}")
    rows = sorted(volume.items(), key=lambda x: -x[1])
    for genre, vol in rows:
        t_share = titles[genre] / total_titles
        v_share = vol / total_vol
        gap = v_share - t_share
        flag = '  ←偏り大' if abs(gap) >= 0.15 else ''
        print(f'  {genre[:22]:<22} {titles[genre]:>4d} {t_share:>7.1%} '
              f'{vol:>10,d} {v_share:>7.1%} {gap:>+8.1%}{flag}')
    print('  ※ 本数で均衡させても量で偏る場合、ジャンル均衡は見かけだけになる')


def report_roster_change(reviews: list, bucket_days: int) -> None:
    """
    期間ごとの参加ゲーム数の変化を見る。

    増えていると、絶対数では「ゲームが増えただけ」が成長に見える。
    シェアを主軸にした理由がここにある（docs/decisions.md）。
    """
    buckets = defaultdict(set)
    for r in reviews:
        ts = int(r['timestamp_created'])
        day = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date()
        key = day.toordinal() // bucket_days
        buckets[key].add(r['game_id'])

    if len(buckets) < 2:
        return
    keys = sorted(buckets)
    print(f'\n## 4. 参加ゲーム数の推移（{bucket_days}日ごと）')
    label = lambda k: dt.date.fromordinal(k * bucket_days).isoformat()  # noqa: E731
    print(f'  最古 {label(keys[0])}: {len(buckets[keys[0]])}本')
    print(f'  最新 {label(keys[-1])}: {len(buckets[keys[-1]])}本')
    counts = [len(buckets[k]) for k in keys]
    print(f'  範囲 {min(counts)}〜{max(counts)}本')
    if max(counts) - min(counts) >= 3:
        print('  → 期間で参加ゲーム数が変わっている。絶対数だと偽の成長が出るため、'
              'Y軸はシェアを使うこと')


def main():
    parser = argparse.ArgumentParser(description='時系列用データセットの偏りを点検')
    parser.add_argument('--input', default='data/timeseries/reviews_timeseries.csv')
    parser.add_argument('--games', default='data/timeseries/games.csv')
    parser.add_argument('--top-n', type=int, default=10, help='ゲーム別に表示する本数')
    parser.add_argument('--bucket-days', type=int, default=90,
                        help='参加ゲーム数を数える区切り日数')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'{args.input} がありません。先に収集を実行してください')
        return

    reviews = load_reviews(args.input)
    games = load_games(args.games)

    print('=' * 70)
    print('時系列用データセットの偏り点検')
    print('=' * 70)
    ts = [int(r['timestamp_created']) for r in reviews]
    lo = dt.datetime.fromtimestamp(min(ts), dt.timezone.utc).date()
    hi = dt.datetime.fromtimestamp(max(ts), dt.timezone.utc).date()
    print(f'  {len(reviews):,}件 / {len({r["game_id"] for r in reviews})}ゲーム'
          f' / {lo} 〜 {hi}')

    report_ratio(reviews)
    report_game_concentration(reviews, args.top_n)
    report_genre_divergence(reviews, games)
    report_roster_change(reviews, args.bucket_days)


if __name__ == '__main__':
    main()
