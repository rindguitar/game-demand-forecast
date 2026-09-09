"""
母集団キャッシュ（pool_cache.json）からタグを読むモジュール

`pool_cache.json` は収集時に作った497本のメタ情報。ここにはSteamのユーザータグが
入っており、レビューを1件も集めていないゲームについても「何のゲームか」が分かる。

タグは供給の信号（市場が何を出荷したか）であって需要ではない。
需要（レビューのトピック）と突き合わせる用途に使う（→ Issue #42）。
"""

import json
from typing import Dict, List, Optional

import pandas as pd

# キャッシュのメタ情報が入っている特殊キー（ゲームではない）
ORDER_KEY = '__order__'


def load_pool(path: str) -> Dict[str, dict]:
    """母集団キャッシュを読む（ゲーム以外のキーは除く）"""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != ORDER_KEY}


def load_pool_tags(path: str, min_tags: int = 1,
                   only_games: Optional[List[str]] = None) -> pd.DataFrame:
    """ゲーム × タグ の縦長DataFrameを作る

    タグを持たないゲームは落とす。キャッシュは上位n個で切られているので、
    ここに出るのは「そのゲームを代表するタグ」であり、全タグではない。

    Args:
        min_tags: これ未満しかタグを持たないゲームは落とす
        only_games: 指定するとこのゲーム名だけに絞る（ロスターとの比較用）

    Returns:
        game_name / tag を持つ縦長のDataFrame
    """
    rows = []
    for game in load_pool(path).values():
        name, tags = game.get('name'), game.get('tags') or []
        if not name or len(tags) < min_tags:
            continue
        if only_games is not None and name not in only_games:
            continue
        rows.extend({'game_name': name, 'tag': t} for t in tags)
    return pd.DataFrame(rows, columns=['game_name', 'tag'])
