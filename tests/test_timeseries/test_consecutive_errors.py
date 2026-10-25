"""
連続エラーの安全装置（--max-consecutive-errors）のテスト

API の障害やレート制限のとき、全ゲームを失敗で流さないよう、エラーで止まったゲームが
続いたら収集を中断する。エラー以外で止まったゲームが1本挟まれば数え直す。
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/collect'))

import collect_timeseries_dataset as cts  # noqa: E402
from src.data.steam_collector import STOP_ERROR, STOP_EXHAUSTED  # noqa: E402

ERR = f'{STOP_ERROR}: boom'


def run_main(monkeypatch, tmp_path, reasons, extra_args=()):
    """5本の台帳で main() を動かし、iter_natural_reviews が呼ばれた回数を返す。
    reasons は呼び出し順の停止理由（足りなければ最後のものを繰り返す）"""
    games = [{'app_id': 100 + i, 'name': f'game{i}', 'genres': frozenset(['Action']),
              'tags': ['t'], 'total_reviews': 1000, 'total_positive': 0,
              'total_negative': 0, 'release_date': '2020-01-01', 'tier': '土台'}
             for i in range(5)]
    games_csv = tmp_path / 'games.csv'
    cts.save_game_master(str(games_csv), games)

    calls = []

    def fake_iter(**kwargs):
        reason = reasons[min(len(calls), len(reasons) - 1)]
        calls.append(kwargs['app_id'])
        yield [], '*', reason

    monkeypatch.setattr(cts, 'iter_natural_reviews', fake_iter)
    monkeypatch.setattr(cts.time, 'sleep', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', [
        'collect_timeseries_dataset.py',
        '--games-output', str(games_csv),
        '--output', str(tmp_path / 'reviews.csv'),
        '--progress-output', str(tmp_path / 'progress.json'),
        '--log-output', str(tmp_path / 'log.csv'),
        '--sleep', '0', *extra_args,
    ])
    cts.main()
    return len(calls)


def test_default_stops_after_three_errors(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, [ERR]) == 3


def test_threshold_can_be_changed(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, [ERR], ['--max-consecutive-errors', '5']) == 5


def test_success_resets_the_count(monkeypatch, tmp_path):
    """エラー, エラー, 成功, エラー, エラー なら中断せず5本とも集める"""
    reasons = [ERR, ERR, STOP_EXHAUSTED, ERR, ERR]
    assert run_main(monkeypatch, tmp_path, reasons) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
