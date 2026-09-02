"""
レビュー収集のページング（_collect_reviews_paged）のテスト

Steamは連続アクセスに対し、エラーではなく空ページを返して黙って打ち切ることがある。
これを終端と解釈して収集を止めると、期間の途中で切れたデータが「それしか無かった」
ものとして混ざる。実際に15本中6本がこれで欠けたため、停止条件を固定するテストを置く。
"""

import sys
import os
import time
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.steam_collector import (  # noqa: E402
    STOP_ERROR,
    STOP_EXHAUSTED,
    STOP_REACHED_NUM,
    STOP_REACHED_SINCE,
    _collect_reviews_paged,
)

NOW = int(time.time())
SINCE = NOW - 30 * 86400  # 直近30日を集める想定

# langdetectを通す必要があるので、実際に英語として判定される本文を使う
TEXT = 'This game is really fun and I enjoyed every single minute of playing it.'


def make_review(days_ago: int) -> dict:
    return {
        'review': TEXT,
        'voted_up': True,
        'votes_up': 1,
        'language': 'english',
        'timestamp_created': NOW - days_ago * 86400,
        'author': {'steamid': '123'},
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def fake_api(monkeypatch, pages):
    """pagesを順に返す偽のAPI。要素が例外ならraiseする"""
    calls = iter(pages)

    def _request(*args, **kwargs):
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    monkeypatch.setattr('src.data.steam_collector.request_with_backoff', _request)


def page(reviews, cursor='next'):
    return {'success': 1, 'reviews': reviews, 'cursor': cursor}


def collect(**kwargs):
    params = {'json': 1}
    defaults = dict(app_id=1, params=params, num=1000, since_ts=SINCE,
                    detailed=False, max_retries=1, sleep=0)
    defaults.update(kwargs)
    return _collect_reviews_paged(**defaults)


def test_empty_page_is_retried_not_treated_as_end(monkeypatch):
    """途中の空ページで止まらない（レート制限を終端と誤認しないこと）"""
    fake_api(monkeypatch, [
        page([make_review(1)], cursor='c1'),
        page([], cursor='c1'),               # レート制限による空ページ
        page([make_review(2)], cursor='c2'),
        page([make_review(40)], cursor='c3'),  # since_tsより古い＝期間の先頭
    ])
    reviews, reason = collect()
    assert len(reviews) == 2
    assert reason == STOP_REACHED_SINCE


def test_empty_after_full_page_is_retried_harder(monkeypatch):
    """満杯のページの直後の空ページは、終端ではなくレート制限として粘る"""
    full = [make_review(1)] * 100
    fake_api(monkeypatch, [
        page(full, cursor='c1'),
        page([], cursor='c1'),
        page([], cursor='c1'),
        page([make_review(40)], cursor='c2'),  # 期間の先頭
    ])
    reviews, reason = collect(sleep=0)
    assert len(reviews) == 100
    assert reason == STOP_REACHED_SINCE


def test_persistent_empty_pages_report_exhausted(monkeypatch):
    """空ページが続く場合は取れた分を返し、理由をexhaustedにする"""
    fake_api(monkeypatch, [page([make_review(1)], cursor='c1')] + [page([], cursor='c1')] * 4)
    reviews, reason = collect()
    assert len(reviews) == 1
    assert reason == STOP_EXHAUSTED


def test_same_cursor_is_treated_as_end(monkeypatch):
    """同じcursorが返ってきたら終端（無限ループを防ぐ）"""
    fake_api(monkeypatch, [
        page([make_review(1)], cursor='c1'),
        page([make_review(2)], cursor='c1'),
    ])
    reviews, reason = collect()
    assert len(reviews) == 2
    assert reason == STOP_EXHAUSTED


def test_request_failure_keeps_partial_results(monkeypatch):
    """途中で失敗しても取れた分は捨てない（捨てると再収集のコストが跳ね上がる）"""
    fake_api(monkeypatch, [
        page([make_review(1)], cursor='c1'),
        requests.exceptions.RequestException('boom'),
    ])
    reviews, reason = collect()
    assert len(reviews) == 1
    assert reason.startswith(STOP_ERROR)


def test_num_limit_stops_collection(monkeypatch):
    """件数上限に達したら理由をreached_numにする"""
    fake_api(monkeypatch, [page([make_review(1), make_review(2)], cursor='c1')])
    reviews, reason = collect(num=2)
    assert len(reviews) == 2
    assert reason == STOP_REACHED_NUM


def test_api_error_response_returns_partial(monkeypatch):
    """success!=1 は例外にせず、理由付きで取れた分を返す"""
    fake_api(monkeypatch, [
        page([make_review(1)], cursor='c1'),
        {'success': 2, 'reviews': []},
    ])
    reviews, reason = collect()
    assert len(reviews) == 1
    assert reason.startswith(STOP_ERROR)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
