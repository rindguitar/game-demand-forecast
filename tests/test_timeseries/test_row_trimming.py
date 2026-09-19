"""
収集CSVの切り詰め（trim_game_rows）のテスト

追記方式なので、取り直すゲームの古い行は消す必要がある。一方で途中から再開する
ゲームは、進捗に記録された行数までは正しいので残す。記録より先の行は
「追記はしたが進捗を保存する前に落ちた」分で、残すと二重になる。
"""

import csv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/collect'))

from collect_timeseries_dataset import FIELDS, trim_game_rows  # noqa: E402


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def read_ids(path):
    with open(path, encoding='utf-8') as f:
        return [int(r['game_id']) for r in csv.DictReader(f)]


def rows(*specs):
    """(app_id, 件数) から行を作る。本文で順序が分かるようにしておく"""
    out = []
    for app_id, n in specs:
        out += [{'game_id': app_id, 'game_name': f'G{app_id}',
                 'review_text': f'{app_id}-{i}', 'timestamp_created': 1000 + i}
                for i in range(n)]
    return out


def test_keep_zero_drops_all_rows_of_that_game(tmp_path):
    """0を指定したゲームは全削除（先頭から取り直すため）"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 3), (11, 2)))
    assert trim_game_rows(path, {10: 0}) == 3
    assert read_ids(path) == [11, 11]


def test_keep_n_leaves_first_n_rows(tmp_path):
    """記録された行数までは残し、それより先だけ落とす"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 5)))
    assert trim_game_rows(path, {10: 2}) == 3
    with open(path, encoding='utf-8') as f:
        kept = [r['review_text'] for r in csv.DictReader(f)]
    assert kept == ['10-0', '10-1']


def test_untouched_games_are_left_alone(tmp_path):
    """指定していないゲームには触らない"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 2), (11, 3)))
    trim_game_rows(path, {10: 1})
    assert read_ids(path).count(11) == 3


def test_keep_larger_than_actual_removes_nothing(tmp_path):
    """記録より実際の行が少なければ何も消さない（欠けは切り詰めでは直せない）"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 2)))
    assert trim_game_rows(path, {10: 99}) == 0
    assert read_ids(path) == [10, 10]


def test_handles_multiple_games_independently(tmp_path):
    """ゲームごとに別々の件数で切り詰められる"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 4), (11, 4)))
    assert trim_game_rows(path, {10: 1, 11: 0}) == 7
    assert read_ids(path) == [10]


def test_empty_keep_is_noop(tmp_path):
    """対象が無ければ何もしない"""
    path = str(tmp_path / 'r.csv')
    write_csv(path, rows((10, 2)))
    assert trim_game_rows(path, {}) == 0
    assert read_ids(path) == [10, 10]


def test_missing_file_is_noop(tmp_path):
    """CSVがまだ無ければ何もしない（初回収集）"""
    assert trim_game_rows(str(tmp_path / 'none.csv'), {10: 0}) == 0
