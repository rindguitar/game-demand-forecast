"""
発売後の山の続き方を測る（Issue #63 段階1）

発売直後の週平均に対して、中盤・後半の週平均がどれだけ残ったか（残った割合）を、
週次の件数と、全ゲームの中での割合の両方で測る。発売前からのファン・発売の種類で続き方が
動くので、区分（configs/launch_labels.csv）が同じ発売どうしの中央値からの差（相対値）で見る。
窓は発売週を0週目として、直後 = 0〜7週・中盤 = 8〜25週・後半 = 26週〜最後・発売前 = 8週前〜1週前。

処理の流れ:
  1. レビュー・ゲーム台帳・区分ファイルを読み、対象（発売日の範囲内）の区分を検査する
  2. 週×ゲームの件数表を作り、発売ごとに窓の週平均と残った割合を出す（測れない発売は外す）
  3. 区分から組を決め、組の中央値・相対値・件数版と割合版の食い違いの印を出す
  4. launch_persistence.csv / group_summary.csv / 図を書き、表を画面に出す

使い方:
    make launch-persistence
    make launch-persistence PERSISTENCE_ARGS="--min-group 4"
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.timeseries.persistence import (  # noqa: E402
    IMMEDIATE_WEEKS,
    MID_END,
    MIN_GROUP,
    MISSING_DAYS,
    PRE_WEEKS,
    add_group_comparison,
    measure_persistence,
    persistence_curves,
    validate_labels,
    weekly_counts,
)
from src.visualization.timeseries_plots import plot_persistence_grid  # noqa: E402

DEFAULT_REVIEWS = 'data/timeseries/reviews_timeseries_with_topics_64.csv'
DEFAULT_GAMES = 'data/timeseries/games.csv'
DEFAULT_LABELS = 'configs/launch_labels.csv'
DEFAULT_OUTPUT_DIR = 'data/timeseries/persistence_64'
DEFAULT_RELEASE_FROM = '2023-07-31'
DEFAULT_RELEASE_TO = '2026-03-01'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reviews', default=DEFAULT_REVIEWS)
    parser.add_argument('--games', default=DEFAULT_GAMES)
    parser.add_argument('--labels', default=DEFAULT_LABELS)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--release-from', default=DEFAULT_RELEASE_FROM, help='対象にする発売日の下限')
    parser.add_argument('--release-to', default=DEFAULT_RELEASE_TO, help='対象にする発売日の上限')
    parser.add_argument('--immediate-weeks', type=int, default=IMMEDIATE_WEEKS, help='発売直後の週数')
    parser.add_argument('--mid-end', type=int, default=MID_END, help='中盤の終わり（この週から後半）')
    parser.add_argument('--pre-weeks', type=int, default=PRE_WEEKS, help='発売前の週数')
    parser.add_argument('--min-group', type=int, default=MIN_GROUP, help='中央値を出す組の最小本数')
    parser.add_argument('--missing-days', type=int, default=MISSING_DAYS,
                        help='最古のレビューが発売日よりこの日数以上あとなら測れない')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 読み込みと区分の検査
    games = pd.read_csv(args.games)
    in_range = games[(games['release_date'] >= args.release_from) &
                     (games['release_date'] <= args.release_to)]
    releases = in_range.set_index('name')['release_date']
    labels = validate_labels(pd.read_csv(args.labels), releases.index)
    reviews = pd.read_csv(args.reviews, usecols=['game_name', 'timestamp_created'])

    # 2. 窓ごとの週平均と残った割合
    counts, oldest = weekly_counts(reviews)
    measured, excluded = measure_persistence(
        counts, oldest, releases, args.immediate_weeks, args.mid_end, args.pre_weeks,
        args.missing_days)

    # 3. 組の中央値と相対値
    result, summary = add_group_comparison(measured, labels, args.min_group)

    # 4. 書き出しと表示
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    result.to_csv(os.path.join(args.output_dir, 'launch_persistence.csv'), index=False)
    summary.to_csv(os.path.join(args.output_dir, 'group_summary.csv'), index=False)
    curves = persistence_curves(counts, result, args.immediate_weeks)
    plot_path = plot_persistence_grid(
        curves, 'Review persistence after launch (by group)',
        os.path.join(args.output_dir, 'plots', 'persistence_by_group.png'),
        groups=summary['group'], immediate_weeks=args.immediate_weeks, mid_end=args.mid_end)

    pd.set_option('display.width', 220)
    pd.set_option('display.max_columns', 30)
    print(f'測れた発売: {len(result)}本 / 外した発売: {len(excluded)}本\n')
    print('== 組ごとの本数と中央値 ==')
    print(summary.round(3).to_string(index=False))
    cols = ['name', 'group', 'persist_mid_count', 'persist_late_count', 'persist_mid_share',
            'persist_late_share', 'relative_mid_count', 'relative_late_count',
            'mismatch_mid', 'mismatch_late']
    print('\n== 発売ごとの残った割合と相対値 ==')
    print(result.sort_values(['group', 'name'])[cols].round(3).to_string(index=False))
    print('\n== 外した発売 ==')
    print(excluded.to_string(index=False) if len(excluded) else '無し')
    print(f'\n出力: {args.output_dir}/ （図: {plot_path}）')


if __name__ == '__main__':
    main()
