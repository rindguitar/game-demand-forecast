"""
レビュー収集の進捗を保存するモジュール

ゲーム1本を取り切ってから書く方式だと、中断や例外で**進行中のゲームが丸ごと消える**。
大きいゲームでは全レビューがメモリに載るので、件数に比例して膨らみもする。
ページごとに追記し、「どのcursorまで取ったか」をここに残して途中から再開できるようにする。

保存の順序は **追記 → 進捗保存** で固定する。逆にすると、その間に落ちたとき
ページが1つ抜けたまま「取り切った」ことになる。欠けは後から気づけないが、
重複は `rows`（記録済みの行数）まで切り詰めれば正確に直せる。
"""

import json
import os
from typing import Dict, List

# 1本もまだ取っていない状態。cursor の '*' がSteam APIの先頭を指す
NEW_ENTRY = {'cursor': '*', 'rows': 0, 'oldest': 0, 'newest': 0, 'positives': 0}


def load_progress(path: str) -> Dict[int, dict]:
    """進捗を読む。無ければ空（＝全部を先頭から取る）"""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding='utf-8') as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        # 書き込み途中で落ちた等。進捗を失うだけで、収集はやり直せる
        print(f'⚠️ 進捗ファイルを読めませんでした（先頭から収集します）: {path}')
        return {}
    return {int(k): dict(NEW_ENTRY, **v) for k, v in raw.items()}


def save_progress(path: str, progress: Dict[int, dict]) -> None:
    """進捗を書く。一時ファイルに書いてから差し替える（書き込み中の中断に備える）"""
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in progress.items()}, f, ensure_ascii=False)
    os.replace(tmp, path)


def advance(entry: dict, page: List[dict], cursor: str) -> dict:
    """1ページ分だけ進捗を進める

    全レビューをメモリに残さずに済むよう、件数・最古・最新・ポジ数を
    流しながら集計する。カバー率の判定にはこれだけあれば足りる。
    """
    oldest, newest = entry['oldest'], entry['newest']
    times = [r['timestamp_created'] for r in page if r.get('timestamp_created')]
    if times:
        oldest = min(times) if not oldest else min(oldest, min(times))
        newest = max(newest, max(times))
    return {
        'cursor': cursor or entry['cursor'],
        'rows': entry['rows'] + len(page),
        'oldest': oldest,
        'newest': newest,
        'positives': entry['positives'] + sum(1 for r in page if r.get('voted_up')),
    }
