"""
期間中の発売を選ばずに全部集める処理（pick_launches / release_since_ts）のテスト（#63 段階2）

選定の規則を通さないこと、発売の何週前から集めるかの計算を固定する。
"""

import sys
import os
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/collect'))

from collect_timeseries_dataset import pick_launches, release_since_ts  # noqa: E402

FROM, TO = '2024-01-01', '2024-12-31'


def game(app_id, release, genres=('Action',), tags=('a', 'b', 'c')):
    return {'app_id': app_id, 'name': f'g{app_id}', 'genres': frozenset(genres),
            'tags': list(tags), 'release_date': release, 'total_reviews': 100}


def ids(games):
    return [g['app_id'] for g in games]


def test_範囲の両端は含み前日と翌日は含まない():
    pool = [game(1, '2023-12-31'), game(2, FROM), game(3, TO), game(4, '2025-01-01')]
    assert ids(pick_launches(pool, FROM, TO)) == [2, 3]


def test_除外のゲームは入らない():
    pool = [game(1, '2024-02-01'), game(2, '2024-03-01')]
    assert ids(pick_launches(pool, FROM, TO, exclude_ids={1})) == [2]


def test_発売日とapp_idの順に並ぶ():
    pool = [game(5, '2024-05-01'), game(3, '2024-02-01'), game(2, '2024-05-01')]
    assert ids(pick_launches(pool, FROM, TO)) == [3, 2, 5]


def test_全部にtier発売が付き入力は書き換わらない():
    pool = [game(1, '2024-02-01'), game(2, '2024-03-01')]
    picked = pick_launches(pool, FROM, TO)
    assert all(g['tier'] == '発売' for g in picked)
    assert all('tier' not in g for g in pool)


def test_ジャンルやタグが同じゲームも全部入る():
    pool = [game(i, '2024-02-01', genres=('Action',), tags=('a', 'b', 'c'))
            for i in range(1, 31)]
    assert len(pick_launches(pool, FROM, TO)) == 30


def test_発売の8週前の0時UTCを返す():
    expected = int(dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc).timestamp()) - 56 * 86400
    assert release_since_ts('2024-03-01', 8, default_ts=123) == expected


def test_発売日が空かNoneか読めない形式なら既定を返す():
    assert release_since_ts('', 8, default_ts=123) == 123
    assert release_since_ts(None, 8, default_ts=123) == 123
    assert release_since_ts('2024/03/01', 8, default_ts=123) == 123
