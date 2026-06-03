"""
Steam APIゲームレビュー収集モジュール

Steam APIからゲームレビューを収集する機能を提供します。
"""

import requests
import time
import re
from typing import List, Dict, Optional
from langdetect import detect_langs, LangDetectException
from langdetect import DetectorFactory
DetectorFactory.seed = 0  # 再現性のために固定

# Steam APIアクセス時のブラウザUA（データセンターIPからのブロック回避）
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0 Safari/537.36'
}


def request_with_backoff(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    timeout: int = 10,
    max_retries: int = 5,
    base_wait: float = 1.0,
) -> requests.Response:
    """
    指数バックオフ付きでGETリクエストを実行

    429（レート制限・bot判定）など一時的なエラー時は待ち時間を倍々に伸ばしてリトライする。
    短い間隔でリトライを繰り返してブロックが解けないまま延々失敗し続けるのを防ぐ。

    Args:
        url: リクエストURL
        params: クエリパラメータ
        headers: リクエストヘッダ
        timeout: タイムアウト秒数
        max_retries: 最大リトライ回数
        base_wait: バックオフの基準待ち時間（秒）。attempt回目は base_wait * 2**attempt 秒待つ。
            429の場合はさらに10倍長く待つ（レート制限の解除には時間がかかるため）。

    Returns:
        成功時のrequests.Response

    Raises:
        requests.exceptions.RequestException: 全リトライ失敗時
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            # 最後の試行で失敗したら例外を投げる
            if attempt == max_retries - 1:
                raise
            # HTTPステータスを取得（429=レート制限か判定）
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            wait = base_wait * (2 ** attempt)
            if status == 429:
                wait *= 10  # レート制限は桁違いに長く待つ
            print(f"    ⏳ リクエスト失敗 (status={status}, "
                  f"{attempt + 1}/{max_retries}回目) → {wait:.0f}秒待機してリトライ")
            time.sleep(wait)


def get_popular_games(n_pages: int = 20) -> list:
    """
    Steam公式の検索APIからレビュー数順の人気ゲームを取得（レビューが豊富な母集団）

    Steam公式の検索結果JSONはappidを直接持たず、ロゴ画像URLに埋め込まれているため、
    正規表現で抽出する。

    Args:
        n_pages: 取得ページ数（1ページ約25件、20ページで約500件）

    Returns:
        (app_id, game_name)のリスト（レビュー数の多い順）
    """
    base_url = "https://store.steampowered.com/search/results/"

    games = []
    seen = set()

    for page in range(n_pages):
        params = {
            'query': '',
            'start': page * 25,
            'count': 25,
            'sort_by': 'Reviews_DESC',  # レビュー数の多い順
            'category1': 998,           # 998 = ゲームのみ（DLC・ツール等を除外）
            'json': 1,
        }
        response = request_with_backoff(base_url, params=params, headers=HEADERS, timeout=30)
        data = response.json()

        items = data.get('items', [])
        if not items:
            break

        for item in items:
            logo = item.get('logo', '')
            # ロゴURL内の /apps/<appid>/ からappidを抽出
            match = re.search(r'/apps/(\d+)/', logo)
            if not match:
                continue
            app_id = int(match.group(1))
            if app_id in seen:
                continue
            seen.add(app_id)
            games.append((app_id, item.get('name', 'Unknown')))

        time.sleep(0.5)  # Steam APIへのrate limiting

    return games


def is_valid_english_review(text: str, min_length: int = 20, lang_confidence: float = 0.8) -> bool:
    """
    レビューが有効な英語かどうかを判定

    Args:
        text: レビューテキスト
        min_length: 最小文字数（デフォルト: 20）
        lang_confidence: langdetectの英語信頼スコア閾値（デフォルト: 0.8）
            閾値未満のレビューは除外して再収集することで、
            短文でも確信度の高いレビューのみを採用できる。

    Returns:
        有効な英語レビューならTrue

    判定条件:
        1. ASCII文字のみで構成されている
        2. 最小文字数以上
        3. アルファベットが50%以上含まれる（記号のみを除外）
        4. langdetectの英語信頼スコアが閾値以上
    """
    if not text or not isinstance(text, str):
        return False

    # 1. ASCII文字のみ（中国語、ロシア語、アラビア語等を除外）
    if not re.match(r'^[\x00-\x7F]+$', text):
        return False

    # 2. 最小文字数チェック（超短文・スパムを除外）
    if len(text) < min_length:
        return False

    # 3. アルファベット割合チェック（記号のみ、数字のみを除外）
    alpha_chars = len(re.findall(r'[a-zA-Z]', text))
    if alpha_chars / len(text) < 0.5:
        return False

    # 4. langdetectの信頼スコアチェック（閾値未満は除外して再収集）
    try:
        langs = detect_langs(text)
        en_score = next((l.prob for l in langs if l.lang == 'en'), 0.0)
        if en_score < lang_confidence:
            return False
    except LangDetectException:
        return False

    return True


def get_steam_reviews(
    app_id: int,
    language: str = 'english',
    review_type: str = 'all',
    num: int = 100,
    max_retries: int = 5
) -> List[Dict]:
    """
    Steam APIからレビューを収集

    Args:
        app_id: SteamゲームID（例: 730=CS:GO, 570=Dota 2）
        language: 'english', 'japanese', 'all'のいずれか
        review_type: 'positive', 'negative', 'all'のいずれか
        num: 収集するレビュー数
        max_retries: API retry試行回数の上限

    Returns:
        レビューのdictリスト、各dictは以下を含む:
            - review_text: レビュー本文
            - voted_up: True=おすすめ, False=おすすめしない
            - votes_up: 高評価数
            - language: レビューの言語
            - timestamp_created: レビュー作成時刻
            - author: 投稿者のSteam ID

    Raises:
        ValueError: app_idまたはパラメータが不正な場合
        requests.exceptions.RequestException: APIリクエスト失敗時

    Example:
        >>> reviews = get_steam_reviews(app_id=730, language='english', num=100)
        >>> print(f"Collected {len(reviews)} reviews")
    """
    if app_id <= 0:
        raise ValueError(f"Invalid app_id: {app_id}")

    if language not in ['english', 'japanese', 'all']:
        raise ValueError(f"Invalid language: {language}. Must be 'english', 'japanese', or 'all'")

    if review_type not in ['positive', 'negative', 'all']:
        raise ValueError(f"Invalid review_type: {review_type}. Must be 'positive', 'negative', or 'all'")

    if num <= 0:
        raise ValueError(f"Invalid num: {num}. Must be positive")

    # Steam APIエンドポイント
    base_url = "https://store.steampowered.com/appreviews/"

    # APIパラメータ
    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',  # 最新レビューを取得
        'review_type': review_type,
        'purchase_type': 'all',
        'num_per_page': min(100, num),  # APIの上限は1リクエスト100件
    }

    reviews = []
    cursor = '*'  # 初期cursor

    while len(reviews) < num:
        params['cursor'] = cursor

        # retry付きAPIリクエスト（429レート制限は指数バックオフで待機）
        response = request_with_backoff(
            f"{base_url}{app_id}", params=params, timeout=10, max_retries=max_retries
        )
        data = response.json()

        # APIレスポンス確認
        if data.get('success') != 1:
            raise requests.exceptions.RequestException(
                f"Steam API returned error: {data.get('error', 'Unknown error')}"
            )

        # レビュー抽出
        api_reviews = data.get('reviews', [])
        if not api_reviews:
            break  # これ以上レビューなし

        for review in api_reviews:
            if len(reviews) >= num:
                break

            review_text = review.get('review', '')

            # 有効な英語レビューのみ収集
            if not is_valid_english_review(review_text):
                continue

            reviews.append({
                'review_text': review_text,
                'voted_up': review.get('voted_up', False),
                'votes_up': review.get('votes_up', 0),
                'language': review.get('language', ''),
                'timestamp_created': review.get('timestamp_created', 0),
                'author': review.get('author', {}).get('steamid', ''),
            })

        # 次のcursorを取得
        cursor = data.get('cursor')
        if not cursor:
            break  # ページなし

        # Rate limiting: Steam APIを尊重
        time.sleep(0.5)

    return reviews


def collect_balanced_reviews(
    app_id: int,
    language: str = 'english',
    n_positive: int = 50,
    n_negative: int = 50
) -> Dict[str, List[Dict]]:
    """
    検証用にbalancedなpositive/negativeレビューを収集

    Args:
        app_id: SteamゲームID
        language: 'english'または'japanese'
        n_positive: positiveレビュー数（おすすめ）
        n_negative: negativeレビュー数（おすすめしない）

    Returns:
        'positive'と'negative'をkeyとするdict、各valueはレビューのリスト

    Example:
        >>> reviews = collect_balanced_reviews(app_id=730, language='english')
        >>> print(f"Positive: {len(reviews['positive'])}, Negative: {len(reviews['negative'])}")
    """
    print(f"Collecting {n_positive} positive reviews...")
    positive_reviews = get_steam_reviews(
        app_id=app_id,
        language=language,
        review_type='positive',
        num=n_positive
    )

    print(f"Collecting {n_negative} negative reviews...")
    negative_reviews = get_steam_reviews(
        app_id=app_id,
        language=language,
        review_type='negative',
        num=n_negative
    )

    return {
        'positive': positive_reviews,
        'negative': negative_reviews
    }
