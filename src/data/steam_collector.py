"""
Steam APIゲームレビュー収集モジュール

Steam APIからゲームレビューを収集する機能を提供します。
"""

import requests
import time
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
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
    Steam公式の検索APIから売上上位のゲームを取得（レビューが豊富な母集団）

    Steam公式の検索結果JSONはappidを直接持たず、ロゴ画像URLに埋め込まれているため、
    正規表現で抽出する。未発売タイトル等レビュー0件のゲームも含むので、必要なら
    呼び出し側でレビュー数の下限を確認すること。

    Args:
        n_pages: 取得ページ数（1ページ約25件、20ページで約500件）

    Returns:
        (app_id, game_name)のリスト（売上上位順）
    """
    base_url = "https://store.steampowered.com/search/results/"

    games = []
    seen = set()

    for page in range(n_pages):
        params = {
            'query': '',
            'start': page * 25,
            'count': 25,
            # 売上上位（＝レビューが多く集まる）。sort_by='Reviews_DESC' は
            # レビュー「数」ではなく「評価スコア」順なので使わない
            'filter': 'globaltopsellers',
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


def get_review_summary(app_id: int, language: str = 'english',
                       max_retries: int = 5) -> Dict:
    """
    ゲームのレビュー要約（総数・ポジ/ネガ内訳）を1リクエストで取得

    APIはレビュー本体と一緒に query_summary を返すが、get_steam_reviews() は
    これを読み捨てている。件数だけ知りたい場合に本文まで取るのは無駄なので、
    num_per_page=1 で要約だけ取る。

    Args:
        app_id: SteamゲームID
        language: 'english', 'japanese', 'all'のいずれか
        max_retries: API retry試行回数の上限

    Returns:
        total_reviews / total_positive / total_negative / review_score 等を含むdict
        （取得失敗時は空dict）

    Note:
        件数はSteam API側のフィルタに基づく集計であり、is_valid_english_review()
        を通す前の数字。実際に使える件数はこれより少ない（実測で中央値65%）。
    """
    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': 1,
        'cursor': '*',
    }
    response = request_with_backoff(
        f"https://store.steampowered.com/appreviews/{app_id}",
        params=params, headers=HEADERS, timeout=20, max_retries=max_retries
    )
    data = response.json()
    if data.get('success') != 1:
        return {}
    return data.get('query_summary', {})


def get_release_date(app_id: int, max_retries: int = 5) -> str:
    """
    ゲームの発売日を取得（YYYY-MM-DD、取得・解釈できなければ空文字）

    収集が期間の途中で切れたのか、そもそも発売がその期間内なのかを区別するために使う。
    区別できないと「直近しか無いゲーム」と「直近しか取れなかったゲーム」が混ざる。
    """
    url = "https://store.steampowered.com/api/appdetails"
    params = {'appids': app_id, 'filters': 'release_date', 'l': 'english', 'cc': 'us'}
    try:
        response = request_with_backoff(url, params=params, headers=HEADERS,
                                        timeout=20, max_retries=max_retries)
        entry = response.json().get(str(app_id), {})
    except (requests.exceptions.RequestException, ValueError):
        return ''
    if not isinstance(entry, dict) or not entry.get('success'):
        return ''
    detail = entry.get('data', {})
    if not isinstance(detail, dict):
        return ''

    raw = (detail.get('release_date') or {}).get('date', '')
    # Steamの表記ゆれ（"11 Sep, 2025" / "Sep 11, 2025" 等）に備えて複数書式を試す
    for fmt in ('%d %b, %Y', '%b %d, %Y', '%d %B, %Y', '%B %d, %Y'):
        try:
            return datetime.strptime(raw, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return ''


def _extract_detail_fields(review: Dict) -> Dict:
    """
    需要スコアの材料になる追加フィールドを取り出す

    プレイ時間は意見の信頼度の材料に使う（需要の強さの重みには使わない。
    離脱者の声が消えるため。詳細は docs/decisions.md）。
    """
    author = review.get('author', {})
    return {
        'playtime_at_review': author.get('playtime_at_review', 0),
        'playtime_forever': author.get('playtime_forever', 0),
        'playtime_last_two_weeks': author.get('playtime_last_two_weeks', 0),
        'steam_purchase': review.get('steam_purchase', False),
        'received_for_free': review.get('received_for_free', False),
        'refunded': review.get('refunded', False),
        'written_during_early_access': review.get('written_during_early_access', False),
        'weighted_vote_score': review.get('weighted_vote_score', 0),
        'comment_count': review.get('comment_count', 0),
        'timestamp_updated': review.get('timestamp_updated', 0),
    }


# 収集の停止理由。「期間の先頭まで遡れた」のか「途中で止められた」のかを
# 呼び出し側が区別できないと、切れたデータが「それしか無かった」ものとして混ざる
STOP_REACHED_SINCE = 'reached_since'   # 指定期間の先頭まで到達（正常）
STOP_REACHED_NUM = 'reached_num'       # 件数上限に到達
STOP_EXHAUSTED = 'exhausted'           # APIがこれ以上返さない
STOP_ERROR = 'error'                   # リクエスト失敗が続いた


def _collect_reviews_paged(
    app_id: int,
    params: Dict,
    num: int,
    since_ts: int,
    detailed: bool,
    max_retries: int,
    empty_retries: int = 3,
    sleep: float = 0.5,
) -> Tuple[List[Dict], str]:
    """
    cursorページングでレビューを集め、「なぜ止まったか」も返す

    Steamは連続アクセスに対し、エラーではなく**空ページ**を返して黙って打ち切る
    ことがある。これを終端と解釈すると期間の途中で切れたデータが混ざるため
    （実測で15本中6本が該当）、空ページは待って同じcursorでやり直す。

    Args:
        params: APIクエリパラメータ（cursorは本関数が書き換える）
        empty_retries: 空ページを何回まで待ってやり直すか
        sleep: ページ間の待機秒数

    Returns:
        (レビューのリスト, 停止理由)。停止理由は STOP_* のいずれか。
        途中で失敗した場合も、取れた分は捨てずに返す
    """
    base_url = "https://store.steampowered.com/appreviews/"
    page_size = params.get('num_per_page', 100)
    reviews: List[Dict] = []
    cursor = '*'
    seen_cursors = set()
    empty_streak = 0
    last_page_size = 0

    while len(reviews) < num:
        params['cursor'] = cursor

        # 1. 1ページ取得（429等は request_with_backoff 側でリトライ済み）
        try:
            response = request_with_backoff(
                f"{base_url}{app_id}", params=params, timeout=10, max_retries=max_retries
            )
            data = response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            return reviews, f'{STOP_ERROR}: {exc}'

        if data.get('success') != 1:
            return reviews, f"{STOP_ERROR}: success={data.get('error', data.get('success'))}"

        # 2. 空ページは、待って同じcursorをやり直す。
        #    直前が満杯のページなら「まだ続きがあるのに空が返った」＝レート制限を疑い、
        #    長めに粘る。直前が途中までのページなら本当の終端なので粘らない
        api_reviews = data.get('reviews', [])
        if not api_reviews:
            empty_streak += 1
            limit = empty_retries if last_page_size >= page_size else 1
            if empty_streak > limit:
                return reviews, STOP_EXHAUSTED
            time.sleep(min(45.0, sleep * 10 * (3 ** (empty_streak - 1))))
            continue
        empty_streak = 0
        last_page_size = len(api_reviews)

        # 3. レビューを取り出す。filter=recent は新しい順なので、
        #    since_ts より古いものが出た時点で以降はすべて範囲外
        for review in api_reviews:
            if len(reviews) >= num:
                break

            created = review.get('timestamp_created', 0)
            if since_ts and created < since_ts:
                return reviews, STOP_REACHED_SINCE

            review_text = review.get('review', '')
            if not is_valid_english_review(review_text):
                continue

            record = {
                'review_text': review_text,
                'voted_up': review.get('voted_up', False),
                'votes_up': review.get('votes_up', 0),
                'language': review.get('language', ''),
                'timestamp_created': created,
                'author': review.get('author', {}).get('steamid', ''),
            }
            if detailed:
                record.update(_extract_detail_fields(review))
            reviews.append(record)

        # 4. 次のページへ。cursorが無い・同じcursorが返るのは終端
        cursor = data.get('cursor')
        if not cursor or cursor in seen_cursors:
            return reviews, STOP_EXHAUSTED
        seen_cursors.add(cursor)

        time.sleep(sleep)  # Rate limiting: Steam APIを尊重

    return reviews, STOP_REACHED_NUM


def get_steam_reviews(
    app_id: int,
    language: str = 'english',
    review_type: str = 'all',
    num: int = 100,
    max_retries: int = 5,
    since_ts: int = 0,
    detailed: bool = False
) -> List[Dict]:
    """
    Steam APIからレビューを収集

    Args:
        app_id: SteamゲームID（例: 730=CS:GO, 570=Dota 2）
        language: 'english', 'japanese', 'all'のいずれか
        review_type: 'positive', 'negative', 'all'のいずれか
        num: 収集するレビュー数（since_ts併用時は上限として働く）
        max_retries: API retry試行回数の上限
        since_ts: 指定するとこのUnix時刻より古いレビューに到達した時点で打ち切る。
            件数ではなく期間で区切りたい時系列用（0=期間で区切らない）
        detailed: Trueならプレイ時間・購入経路・返金有無などの追加フィールドも返す

    Returns:
        レビューのdictリスト、各dictは以下を含む:
            - review_text: レビュー本文
            - voted_up: True=おすすめ, False=おすすめしない
            - votes_up: 高評価数
            - language: レビューの言語
            - timestamp_created: レビュー作成時刻
            - author: 投稿者のSteam ID
        detailed=Trueの場合はさらに playtime_at_review / playtime_forever /
        playtime_last_two_weeks / steam_purchase / received_for_free / refunded /
        written_during_early_access / weighted_vote_score / comment_count /
        timestamp_updated を含む

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

    # APIパラメータ
    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',  # 最新レビューを取得
        'review_type': review_type,
        'purchase_type': 'all',
        'num_per_page': min(100, num),  # APIの上限は1リクエスト100件
    }

    reviews, reason = _collect_reviews_paged(
        app_id=app_id, params=params, num=num, since_ts=since_ts,
        detailed=detailed, max_retries=max_retries,
    )
    # 従来の呼び出し側は例外を期待しているので、失敗時はここで投げ直す。
    # 途中経過も含めて受け取りたい場合は _collect_reviews_paged() を直接使う
    if reason.startswith(STOP_ERROR):
        raise requests.exceptions.RequestException(f"Steam API error: {reason}")
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


def collect_natural_reviews(
    app_id: int,
    since_ts: int,
    language: str = 'english',
    max_reviews: int = 200000,
    max_retries: int = 5,
    sleep: float = 0.5,
) -> Tuple[List[Dict], str]:
    """
    指定期間のレビューを自然比率のまま収集（時系列用）

    collect_balanced_reviews() との違いは2点。感情分析の学習用と時系列の測定用で、
    必要なデータの性質が正反対なため別関数にしている（詳細は Wiki「学習用データと
    測定用データの違い」）。

    - ポジ/ネガを均衡させない。review_type='all' で1回だけ呼び、元の比率を保つ
    - 件数ではなく期間で区切る。件数固定だとゲームごとに集まる期間がバラバラになる

    Args:
        app_id: SteamゲームID
        since_ts: この Unix時刻以降のレビューを集める
        language: 'english', 'japanese', 'all'のいずれか
        max_reviews: 安全弁としての上限件数（期間で足りるはずだが暴走を防ぐ）
        max_retries: API retry試行回数の上限

    Returns:
        (レビューのリスト, 停止理由)。停止理由が STOP_REACHED_SINCE 以外の場合、
        指定期間の先頭まで遡れていない（＝そのゲームは直近しか揃っていない）。
        合算すると偽の成長を作るため、呼び出し側で必ず確認すること

    Example:
        >>> import time
        >>> since = int(time.time()) - 3 * 365 * 86400  # 直近3年
        >>> reviews, reason = collect_natural_reviews(app_id=413150, since_ts=since)
    """
    if since_ts <= 0:
        raise ValueError(f"Invalid since_ts: {since_ts}. Must be a positive Unix timestamp")

    params = {
        'json': 1,
        'language': language,
        'filter': 'recent',
        'review_type': 'all',
        'purchase_type': 'all',
        'num_per_page': 100,
    }
    return _collect_reviews_paged(
        app_id=app_id, params=params, num=max_reviews, since_ts=since_ts,
        detailed=True, max_retries=max_retries, sleep=sleep,
    )
