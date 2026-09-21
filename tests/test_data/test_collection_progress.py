"""
収集の進捗保存のテスト
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.collection_progress import NEW_ENTRY, advance, load_progress, save_progress


def _page(times, ups=None):
    ups = ups or [True] * len(times)
    return [{'timestamp_created': t, 'voted_up': u} for t, u in zip(times, ups)]


def test_load_returns_empty_when_missing(tmp_path):
    """進捗が無ければ空（＝先頭から取る）"""
    assert load_progress(str(tmp_path / 'none.json')) == {}


def test_save_and_load_roundtrip(tmp_path):
    """保存した進捗をそのまま読み戻せる（キーは整数のapp_id）"""
    path = str(tmp_path / 'p.json')
    save_progress(path, {10: dict(NEW_ENTRY, cursor='c1', rows=5)})
    loaded = load_progress(path)
    assert loaded[10]['cursor'] == 'c1'
    assert loaded[10]['rows'] == 5


def test_load_fills_missing_fields(tmp_path):
    """古い形式で足りない項目があっても既定値で埋める"""
    path = str(tmp_path / 'p.json')
    open(path, 'w', encoding='utf-8').write('{"10": {"cursor": "c1"}}')
    assert load_progress(path)[10] == dict(NEW_ENTRY, cursor='c1')


def test_load_survives_broken_file(tmp_path):
    """壊れた進捗ファイルで落ちない（収集はやり直せるので空を返す）"""
    path = str(tmp_path / 'p.json')
    open(path, 'w', encoding='utf-8').write('{broken')
    assert load_progress(path) == {}


def test_advance_accumulates_rows_and_positives():
    """件数とポジ数はページをまたいで足し上がる"""
    e = advance(NEW_ENTRY, _page([100, 200], [True, False]), 'c1')
    e = advance(e, _page([300], [True]), 'c2')
    assert e['rows'] == 3
    assert e['positives'] == 2
    assert e['cursor'] == 'c2'


def test_advance_tracks_oldest_and_newest():
    """最古・最新を流しながら追う（全件をメモリに残さないため）"""
    e = advance(NEW_ENTRY, _page([500, 900]), 'c1')
    e = advance(e, _page([100, 700]), 'c2')
    assert e['oldest'] == 100
    assert e['newest'] == 900


def test_advance_does_not_treat_unset_oldest_as_zero():
    """初回の最古は0ではなくそのページの最小になる"""
    assert advance(NEW_ENTRY, _page([500]), 'c1')['oldest'] == 500


def test_advance_keeps_cursor_when_none_given():
    """終端で次のcursorが無くても、直前のcursorを保つ"""
    e = advance(dict(NEW_ENTRY, cursor='c9'), _page([100]), None)
    assert e['cursor'] == 'c9'


def test_advance_ignores_empty_page():
    """空ページでは何も進まない"""
    e = dict(NEW_ENTRY, cursor='c1', rows=3, oldest=100, newest=900)
    assert advance(e, [], 'c2') == dict(e, cursor='c2')
