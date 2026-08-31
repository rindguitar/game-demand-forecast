"""
レビュー投稿数の季節性を測る（Issue #32 の遡る期間を決めるため）

時系列予測の学習に何年分必要かは「年周期の繰り返しがあるか」で決まる。繰り返しを
最低2回観測しないと、周期なのか単発の出来事なのか区別できないため。

ここで測るのは「いつ跳ねるか」を人が特定することではない。祝日カレンダーを知る必要は
なく（英語レビューの投稿者の居住国は推定できない）、**毎年同じ形が繰り返されるか**
だけが分かればよい。繰り返しがあれば2年以上、無ければ下限は下がる。

個別ゲームはアップデートやDLCで独自に跳ねるので、複数ゲームを合算して
ゲーム固有の山を打ち消し、共通の季節変動だけを残す。

使い方は --help を参照。
"""

import os
import sys
import csv
import argparse
import statistics as st
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.dirname(__file__))

from timeseries_feasibility import fetch_review_stats  # noqa: E402

# 中規模で長く運営されているゲーム。大規模作を入れないのは、取得上限に達して
# 直近数ヶ月しか埋まらず、合算したときに偽のトレンドを作るため
DEFAULT_GAMES = [
    (216150, 'Super Hexagon'),
    (1366540, 'Dyson Sphere Program'),
    (597170, 'VTOL VR'),
    (597220, 'West of Loathing'),
    (246620, 'Kingdom Rush'),
    (1263850, 'IXION'),
    (1476970, 'IdleOn'),
    (632470, 'Disco Elysium'),
]


def to_monthly(daily: dict) -> dict:
    """日ごとの件数を年月（YYYY-MM）単位に畳む"""
    monthly = defaultdict(lambda: {'raw': 0, 'valid': 0})
    for date, counts in daily.items():
        ym = date[:7]
        monthly[ym]['raw'] += counts['raw']
        monthly[ym]['valid'] += counts['valid']
    return dict(monthly)


def drop_partial_months(monthly: dict) -> dict:
    """最古・最新の月は取得が途中で切れているため除外する"""
    if len(monthly) <= 2:
        return {}
    keys = sorted(monthly)
    return {k: v for k, v in monthly.items() if k not in (keys[0], keys[-1])}


def month_profile(monthly_totals: dict) -> dict:
    """
    年ごとに「その月が年間に占める割合」を出す。

    件数そのものでなく割合にするのは、年によって総量が違う（ゲームが増える・
    人気が変わる）ため。形が繰り返されているかを見たいので水準を揃える。
    """
    by_year = defaultdict(dict)
    for ym, count in monthly_totals.items():
        year, month = ym.split('-')
        by_year[year][int(month)] = count

    profiles = {}
    for year, months in by_year.items():
        # 12ヶ月揃っている年だけを比較対象にする
        if len(months) < 12:
            continue
        total = sum(months.values())
        if total:
            profiles[year] = {m: months[m] / total for m in range(1, 13)}
    return profiles


def correlation(a: list, b: list) -> float:
    """2つの月次プロファイルの相関係数（形が似ているほど1に近い）"""
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / (va * vb) ** 0.5 if va and vb else 0.0


def main():
    parser = argparse.ArgumentParser(description='レビュー投稿数の季節性を測る')
    parser.add_argument('--years', type=int, default=3, help='遡る年数')
    parser.add_argument('--max-raw', type=int, default=20000,
                        help='1ゲームあたりの取得上限（生レビュー件数）')
    parser.add_argument('--sleep', type=float, default=0.5, help='リクエスト間の待機秒数')
    parser.add_argument('--out-dir', default='data/experiments/seasonality')
    parser.add_argument('--games', type=int, nargs='+', default=None,
                        help='対象のapp_id（未指定なら既定リスト）')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    games = [(g, str(g)) for g in args.games] if args.games else DEFAULT_GAMES

    print('=' * 70)
    print(f'レビュー投稿数の季節性測定（直近{args.years}年・{len(games)}ゲーム）')
    print('=' * 70)

    # 1. ゲームごとに取得し、年月単位に畳む
    #    要求期間を埋めきれなかったゲームは合算から外す。取得上限で直近しか埋まらない
    #    ゲームを混ぜると、合算した月次推移に偽のトレンドが乗るため
    required_months = int(args.years * 12 * 0.9)
    rows, totals, used, skipped = [], defaultdict(int), [], []

    for i, (app_id, name) in enumerate(games, 1):
        stats = fetch_review_stats(app_id, max_valid=10 ** 9,
                                   max_days=args.years * 365,
                                   max_raw=args.max_raw, sleep=args.sleep)
        monthly = drop_partial_months(to_monthly(stats['daily']))
        for ym, counts in sorted(monthly.items()):
            rows.append({'app_id': app_id, 'name': name, 'year_month': ym,
                         'raw': counts['raw'], 'valid': counts['valid']})

        qualified = len(monthly) >= required_months
        if qualified:
            used.append(name)
            for ym, counts in monthly.items():
                totals[ym] += counts['raw']
        else:
            skipped.append(name)

        span = f"{min(monthly)}〜{max(monthly)}" if monthly else '取得なし'
        mark = '合算' if qualified else '除外'
        print(f"  [{i}/{len(games)}] {name[:24]:24s} {stats['raw']:>6d}件  "
              f"{len(monthly):>3d}ヶ月  {span}  [{mark}] ({stats['stop_reason']})")

    print(f'\n合算対象 {len(used)}本 / 除外 {len(skipped)}本'
          + (f'（{", ".join(skipped)} は{required_months}ヶ月に届かず）' if skipped else ''))

    if not totals:
        print('\n合算できるゲームがありません（--max-raw を上げるか --years を減らす）')
        return

    # 2. 全ゲーム合算の月次推移（ゲーム固有の山を打ち消す）
    print(f'\n合算した月次件数（{len(totals)}ヶ月）')
    peak = max(totals.values())
    for ym in sorted(totals):
        bar = '█' * int(totals[ym] / peak * 40)
        print(f'  {ym}  {totals[ym]:>6d}  {bar}')

    # 3. 年ごとの「月別シェア」を比べ、形が繰り返されているか見る
    profiles = month_profile(totals)
    print(f'\n12ヶ月揃った年: {sorted(profiles) if profiles else "なし"}')

    years = sorted(profiles)
    if len(years) >= 2:
        print('\n年どうしの月別シェアの相関（形が似ていれば1に近い）')
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                a = [profiles[years[i]][m] for m in range(1, 13)]
                b = [profiles[years[j]][m] for m in range(1, 13)]
                r = correlation(a, b)
                verdict = '形が一致' if r >= 0.7 else ('やや一致' if r >= 0.4 else '一致せず')
                print(f'  {years[i]} vs {years[j]}: r={r:+.2f}  {verdict}')

        print('\n月別シェアの平均（12ヶ月均等なら 8.3%）')
        for m in range(1, 13):
            avg = st.mean(profiles[y][m] for y in years)
            bar = '█' * int(avg * 300)
            print(f'  {m:>2d}月  {avg:>6.1%}  {bar}')
    else:
        print('\n12ヶ月揃った年が2つ未満のため、年どうしの比較はできない')
        print('（--years を増やすか、--max-raw を上げて取得量を増やす）')

    # 4. 保存
    with open(os.path.join(args.out_dir, 'monthly_counts.csv'), 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['app_id', 'name', 'year_month', 'raw', 'valid'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n✓ 出力: {args.out_dir}/monthly_counts.csv')


if __name__ == '__main__':
    main()
