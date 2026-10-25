"""
レビュー API（GetAppReviews）への変換と取得のテスト

旧 appreviews は 2026-10-22 に廃止された（Issue #67）。新 API は知らない値を黙って無視するため、
旧形式の引数が正しく変換されること・知らない値は弾かれることを固定する。
"""

import sys
import os
import json
import copy
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.steam_collector import (  # noqa: E402
    REVIEWS_API_URL,
    _natural_review_params,
    fetch_reviews_page,
    get_review_summary,
    to_review_api_input,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def fake_request(monkeypatch, payload):
    """呼ばれた (url, kwargs) を記録して payload を返す偽の request_with_backoff"""
    calls = []

    def _request(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse(payload)

    monkeypatch.setattr('src.data.steam_collector.request_with_backoff', _request)
    return calls


def test_natural_params_are_converted():
    """自然比率の収集の引数が、期待どおりの入力になる（json は消える）"""
    params = dict(_natural_review_params('english'), cursor='*')
    assert to_review_api_input(123, params) == {
        'appid': 123, 'languages': ['english'], 'filter': 1, 'review_type': 0,
        'purchase_type': 1, 'num_per_page': 100, 'cursor': '*',
    }


def test_review_type_and_purchase_type_tables():
    """review_type と purchase_type が enum の数字になる"""
    base = {'language': 'english', 'filter': 'all'}
    positive = to_review_api_input(1, dict(base, review_type='positive', purchase_type='steam'))
    assert (positive['review_type'], positive['purchase_type'], positive['filter']) == (1, 0, 0)
    negative = to_review_api_input(1, dict(base, review_type='negative'))
    assert negative['review_type'] == 2


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        to_review_api_input(1, {'language': 'english', 'foo': 1})


def test_value_not_in_table_raises():
    with pytest.raises(ValueError):
        to_review_api_input(1, {'filter': 'newest'})


def test_input_params_not_modified():
    params = dict(_natural_review_params('english'), cursor='*')
    before = copy.deepcopy(params)
    to_review_api_input(1, params)
    assert params == before


def test_fetch_reviews_page_sends_input_json_and_returns_response(monkeypatch):
    """URL は新 API・params は input_json だけ・返事の response の中身が返る"""
    calls = fake_request(monkeypatch, {'response': {'reviews': [], 'cursor': 'c1'}})
    params = dict(_natural_review_params('english'), cursor='*')
    data = fetch_reviews_page(42, params)
    assert data == {'reviews': [], 'cursor': 'c1'}
    url, kwargs = calls[0]
    assert url == REVIEWS_API_URL
    assert list(kwargs['params']) == ['input_json']
    sent = json.loads(kwargs['params']['input_json'])
    assert sent['appid'] == 42 and sent['languages'] == ['english'] and sent['cursor'] == '*'


@pytest.mark.parametrize('body', [{'error': 'x'}, {'response': 'oops'}, []])
def test_fetch_reviews_page_without_response_raises(monkeypatch, body):
    """response が無い・dict でない返事は、空ページ（終端）と読まず ValueError"""
    fake_request(monkeypatch, body)
    with pytest.raises(ValueError):
        fetch_reviews_page(42, {'language': 'english'})


def test_get_review_summary_returns_query_summary(monkeypatch):
    summary = {'total_reviews': 10, 'total_positive': 8, 'total_negative': 2}
    fake_request(monkeypatch, {'response': {'query_summary': summary, 'reviews': []}})
    assert get_review_summary(42) == summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
