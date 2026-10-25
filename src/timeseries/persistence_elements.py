"""
山の続き方と、発売が持つ要素の関係を「のぞく」モジュール（Issue #63 段階3）

よく続いた発売と続かなかった発売で、持っている要素が違うかを、要素ごと・2つの組み合わせごとに
比べる。差が偶然でも出るかは、発売と相対値の対応を入れ替える（シャッフル）ことで確かめる。
ファイルの読み書きはしない。
"""

from itertools import combinations
from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd

from src.nlp.tag_semantics import ELEMENT

MIN_COUNT = 3        # 要素を見る最小の発売数
N_SHUFFLE = 2000     # シャッフル回数
SEED = 0             # 乱数の種
TOP_TAGS = 6         # 発売ごとに見る上位タグの数
MAX_DIFF_QUANTILE = 95  # 「全要素の最大差」の線にするパーセンタイル
ELEMENT_CATEGORY = 'element'


def usable_releases(persistence: pd.DataFrame, value_col: str = 'relative_mid_count') -> pd.Series:
    """相対値が空でない発売だけを、名前 → 相対値の Series で返す

    1. value_col が欠損の行を外す → 2. name を index にして返す
    """
    df = persistence.dropna(subset=[value_col])
    return df.set_index('name')[value_col].astype(float)


def recipe_element_sets(recipes: pd.DataFrame, category: str = ELEMENT_CATEGORY) -> Dict[str, Set]:
    """レシピから、ゲームごとの「持つ要素（①の unit）」の集合を作る

    1. category の行だけ残す → 2. ゲーム名ごとに unit を集める
    """
    sub = recipes[recipes['category'] == category]
    return {g: set(grp['unit']) for g, grp in sub.groupby('game_name')}


def unit_keywords(recipes: pd.DataFrame, category: str = ELEMENT_CATEGORY) -> Dict:
    """要素（unit）ごとのキーワード文字列（最初に出てきたもの）"""
    sub = recipes[recipes['category'] == category]
    return sub.drop_duplicates('unit').set_index('unit')['keywords'].to_dict()


def game_tag_sets(games: pd.DataFrame, judgments: Dict[str, str], top_n: int = TOP_TAGS,
                  only: Optional[Iterable[str]] = None) -> Dict[str, Set[str]]:
    """ゲームごとの「持つタグ（上位 top_n のうち①にするもの）」の集合を作る

    1. only があればその発売だけに絞る → 2. tags 列（| 区切り）の先頭 top_n 個を取る
    3. 判定の無いタグがあれば、タグ名とゲーム名を並べて ValueError で止める
    4. 判定が element のタグだけ残す
    """
    if only is not None:
        games = games[games['name'].isin(set(only))]
    result, unjudged = {}, {}
    for name, tags in zip(games['name'], games['tags']):
        top = str(tags).split('|')[:top_n] if isinstance(tags, str) else []
        for t in top:
            if t not in judgments:
                unjudged.setdefault(t, []).append(name)
        result[name] = {t for t in top if judgments.get(t) == ELEMENT}
    if unjudged:
        detail = '; '.join(f'{t}（{", ".join(g)}）' for t, g in sorted(unjudged.items()))
        raise ValueError(f'判定の無いタグがあります: {detail}（判定ファイルに追記すること）')
    return result


def membership_matrix(item_sets: Dict[str, Set], release_names: Iterable[str]) -> pd.DataFrame:
    """発売 × 要素 の真偽表を作る（持っていれば True）

    1. 対象の発売が持つ要素を全部集める → 2. 発売ごとに 持つ/持たない を並べる
    """
    names = list(release_names)
    items = sorted({i for n in names for i in item_sets.get(n, set())}, key=str)
    data = {i: [i in item_sets.get(n, set()) for n in names] for i in items}
    return pd.DataFrame(data, index=names, dtype=bool)


def pair_membership(membership: pd.DataFrame, min_count: int = MIN_COUNT) -> pd.DataFrame:
    """2つの要素の組み合わせ（両方を持つ）の真偽表を作る。両方持つ発売が min_count 本以上のペアだけ

    1. 本数が min_count 以上の要素だけで総当たり → 2. 両方持つ列を作り、本数で絞る
    列名は (a, b) のタプル。
    """
    counts = membership.sum()
    items = [c for c in membership.columns if counts[c] >= min_count]
    data = {}
    for a, b in combinations(items, 2):
        both = membership[a] & membership[b]
        if both.sum() >= min_count:
            data[(a, b)] = both
    return pd.DataFrame(data, index=membership.index, dtype=bool)


def _group_medians(mask: np.ndarray, values: np.ndarray):
    """列ごとに、持つ発売・持たない発売の中央値を返す（mask は 発売 × 要素）"""
    x = values[:, None]
    have = np.nanmedian(np.where(mask, x, np.nan), axis=0)
    lack = np.nanmedian(np.where(~mask, x, np.nan), axis=0)
    return have, lack


def compare_items(membership: pd.DataFrame, values: pd.Series,
                  min_count: int = MIN_COUNT) -> pd.DataFrame:
    """要素ごとに、持つ発売と持たない発売の相対値の中央値と差を出す

    1. 持つ本数が min_count 未満、または持たない発売が無い要素を外す
    2. 持つ/持たない の中央値と差（持つ − 持たない）・持つ発売の名前を出す → 3. 差の大きい順
    """
    membership = membership.loc[values.index]
    n_have = membership.sum()
    keep = [c for c in membership.columns if n_have[c] >= min_count and n_have[c] < len(values)]
    cols = ['item', 'n_have', 'n_lack', 'median_have', 'median_lack', 'diff', 'games']
    if not keep:
        return pd.DataFrame(columns=cols)

    mask = membership[keep].to_numpy()
    have, lack = _group_medians(mask, values.to_numpy())
    games = [', '.join(values.index[mask[:, j]]) for j in range(len(keep))]
    out = pd.DataFrame({
        'item': keep,
        'n_have': mask.sum(axis=0),
        'n_lack': (~mask).sum(axis=0),
        'median_have': have,
        'median_lack': lack,
        'diff': have - lack,
        'games': games,
    })
    return out.sort_values('diff', ascending=False, kind='stable').reset_index(drop=True)


def shuffle_test(membership: pd.DataFrame, values: pd.Series, items: list,
                 n_shuffle: int = N_SHUFFLE, seed: int = SEED,
                 quantile: float = MAX_DIFF_QUANTILE):
    """発売と相対値の対応を入れ替えて、差が偶然でも出るかを見る

    1. 相対値をシャッフルし、同じ要素ごとに差を出す（n_shuffle 回・種固定）
    2. 要素ごとに「シャッフルの |差| が観測の |差| 以上」の割合 = p_shuffle
    3. 毎回の「全要素でいちばん大きい |差|」の quantile パーセンタイル = max_diff_95
    戻り値は (p_shuffle の配列〈items と同じ順〉, max_diff_95)。
    """
    if not items:
        return np.array([]), float('nan')
    mask = membership.loc[values.index, items].to_numpy()
    v = values.to_numpy()
    have, lack = _group_medians(mask, v)
    observed = np.abs(have - lack)

    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(items))
    max_diffs = np.empty(n_shuffle)
    for i in range(n_shuffle):
        h, lk = _group_medians(mask, rng.permutation(v))
        d = np.abs(h - lk)
        exceed += d >= observed - 1e-12
        max_diffs[i] = d.max()
    return exceed / n_shuffle, float(np.percentile(max_diffs, quantile))


def compare_with_shuffle(membership: pd.DataFrame, values: pd.Series,
                         min_count: int = MIN_COUNT, n_shuffle: int = N_SHUFFLE,
                         seed: int = SEED, labels: Optional[Dict] = None):
    """比較表にシャッフルの結果（p_shuffle・線を越えたか）を付ける

    1. compare_items で表を作る → 2. shuffle_test で p と線を出す
    3. 表に p_shuffle と beyond_line（|差| が線を超えた）を足す
    labels があれば、item の表示名（キーワード等）を keywords 列に入れる。
    戻り値は (表, max_diff_95)。
    """
    table = compare_items(membership, values, min_count)
    p, max_diff = shuffle_test(membership, values, list(table['item']), n_shuffle, seed)
    table['p_shuffle'] = p
    table['max_diff_95'] = max_diff
    table['beyond_line'] = table['diff'].abs() > max_diff
    if labels is not None:
        table.insert(1, 'keywords', [labels.get(i, '') for i in table['item']])
    return table, max_diff
