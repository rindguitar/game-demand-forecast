"""
収集の網羅性判定（judge_coverage）のテスト

「直近しか無いゲーム」と「直近しか取れなかったゲーム」を取り違えると、合算したとき
に参加ゲームが増えただけの偽の成長が出る。この区別を固定する。
"""

import sys
import os
import datetime as dt
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/collect'))

from src.data.steam_collector import (  # noqa: E402
    STOP_EXHAUSTED,
    STOP_REACHED_NUM,
    STOP_REACHED_SINCE,
)
from collect_timeseries_dataset import judge_coverage  # noqa: E402


def ts(date_str: str) -> int:
    return int(dt.datetime.strptime(date_str, '%Y-%m-%d')
               .replace(tzinfo=dt.timezone.utc).timestamp())


SINCE = ts('2023-09-01')


def test_reached_since_is_complete():
    """期間の先頭まで遡れたら完全"""
    assert judge_coverage(STOP_REACHED_SINCE, ts('2023-09-01'), SINCE, '2015-01-01') == 'ok'


def test_exhausted_after_release_in_window_is_complete():
    """発売がウィンドウ内なら、それ以上古いレビューは存在しない"""
    assert judge_coverage(STOP_EXHAUSTED, ts('2025-04-10'), SINCE, '2025-04-10') == 'ok'


def test_exhausted_long_after_release_is_partial():
    """発売がずっと前なのに途中で終わったのは、取り切れていない"""
    assert judge_coverage(STOP_EXHAUSTED, ts('2025-07-23'), SINCE, '2015-05-29') == 'partial'


def test_exhausted_without_release_date_is_unknown():
    """発売日が取れないと区別できないので、人が見る印を付ける"""
    assert judge_coverage(STOP_EXHAUSTED, ts('2025-07-23'), SINCE, '') == 'unknown'


def test_hitting_review_cap_is_partial():
    """件数上限で止まったら期間は埋まっていない"""
    assert judge_coverage(STOP_REACHED_NUM, ts('2026-01-01'), SINCE, '2015-01-01') == 'partial'


def test_no_reviews_is_partial():
    """1件も取れなかった場合も未達として扱う"""
    assert judge_coverage(STOP_EXHAUSTED, 0, SINCE, '2015-01-01') == 'partial'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
