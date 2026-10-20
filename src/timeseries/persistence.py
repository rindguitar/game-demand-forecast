"""
発売後の山の続き方を測るモジュール（Issue #63 段階1）

発売直後の週平均に対して、中盤・後半の週中央値がどれだけ残ったか（残った割合）を、
週次の件数と、全ゲームの中での割合（シェア）の両方で測る。続き方は組み合わせと関係ない理由
（発売前からのファン・発売の種類）でも動くので、区分が同じ発売どうしの中央値からの差（相対値）で見る。
ファイルの読み書きはしない（forecast.py と同じ方針）。詳細は scripts/README.md を参照。
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

IMMEDIATE_WEEKS = 8   # 発売直後 = 0〜7週
MID_END = 26          # 中盤 = 8〜25週、後半 = 26週〜
PRE_WEEKS = 8         # 発売前 = 発売の8週前〜1週前
MIN_GROUP = 3         # 中央値を出す組の最小本数
MISSING_DAYS = 8      # 最古のレビューが発売日よりこの日数以上あとなら測れない

FAN_BASES = ('none', 'source', 'series')
LAUNCH_TYPES = ('first', 'rerelease', 'ea_to_full')

# 残った割合の指標名 = (窓, 版)
METRICS = [('mid', 'count'), ('late', 'count'), ('mid', 'share'), ('late', 'share')]


def validate_labels(labels: pd.DataFrame, names) -> pd.DataFrame:
    """
    区分ファイルを検査して、対象ゲームの行だけを name の索引で返す

    処理の流れ: 1. 対象に区分が無ければ止まる → 2. 区分の値が決められた語以外なら止まる
    """
    names = list(names)
    indexed = labels.drop_duplicates('name', keep='last').set_index('name')
    missing = [n for n in names if n not in indexed.index]
    if missing:
        raise ValueError(f'区分ファイルに無いゲームがあります: {missing}')
    picked = indexed.loc[names]
    bad_fan = sorted(set(picked['fan_base']) - set(FAN_BASES))
    bad_type = sorted(set(picked['launch_type']) - set(LAUNCH_TYPES))
    if bad_fan or bad_type:
        raise ValueError(f'決められた語以外の区分があります: fan_base={bad_fan}, launch_type={bad_type}')
    return picked


def assign_group(fan_base: str, launch_type: str) -> str:
    """比べる組の名前。first はファン別に分け、rerelease / ea_to_full は1組ずつ"""
    return f'first_{fan_base}' if launch_type == 'first' else launch_type


def weekly_counts(reviews: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    レビューから、週×ゲームの件数表と、ゲームごとの最古のレビュー時刻を作る

    処理の流れ: 1. 週（月曜始まり）を付ける → 2. 週×ゲームで数える
    → 3. 全期間の週を欠けなく並べ、レビューの無い週は0にする
    """
    date = pd.to_datetime(reviews['timestamp_created'], unit='s')
    week = date.dt.to_period('W').dt.start_time
    counts = pd.crosstab(week, reviews['game_name'])
    full = pd.date_range(counts.index.min(), counts.index.max(), freq='W-MON')
    counts = counts.reindex(full, fill_value=0)
    oldest = date.groupby(reviews['game_name']).min()
    return counts, oldest


def _window_stat(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, how: str = 'mean'):
    """start〜end の週（両端を含む）の平均（how='mean'）または中央値（how='median'）。週が1つも無ければ NaN"""
    part = series[(series.index >= start) & (series.index <= end)]
    if not len(part):
        return np.nan
    return part.median() if how == 'median' else part.mean()


def measure_persistence(counts: pd.DataFrame, oldest: pd.Series, releases: pd.Series,
                        immediate_weeks: int = IMMEDIATE_WEEKS, mid_end: int = MID_END,
                        pre_weeks: int = PRE_WEEKS, missing_days: int = MISSING_DAYS
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    発売ごとに、窓ごとの週平均・中盤と後半の週中央値・残った割合を出す

    残った割合の分子は窓の中央値（DLC・大会などのくり返す山に引っ張られないため）、
    分母は発売直後の平均（発売直後の山の大きさそのもの）。分子が平均の版は persist_mean_* に参考で残す。

    処理の流れ:
      1. 件数から割合（全ゲームの中でのシェア）の表を作る
      2. 発売ごとに、測れないものを外す（最古のレビューが発売日の missing_days 日以上あと
         / 中盤の窓がデータの最後を越える）
      3. 発売直後・中盤・後半・発売前の週平均と、中盤・後半の週中央値を、件数と割合の両方で出す
      4. 残った割合（中盤・後半の中央値 ÷ 発売直後の平均。平均版も参考）と発売前の比を出す

    Args:
        counts: weekly_counts の週×ゲームの件数表
        oldest: ゲームごとの最古のレビュー時刻
        releases: ゲーム名 → 発売日

    Returns:
        (測れた発売の表, 外した発売の表 [name, reason])
    """
    share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    last_week = counts.index.max()
    step = pd.Timedelta(weeks=1)
    rows: List[Dict] = []
    excluded: List[Dict] = []

    for name, release in releases.items():
        release_dt = pd.Timestamp(release)
        release_week = release_dt.to_period('W').start_time
        if oldest[name] >= release_dt + pd.Timedelta(days=missing_days):
            excluded.append({'name': name, 'reason': '発売直後が欠けている（最古のレビューが発売日よりあと）'})
            continue
        if release_week + (mid_end - 1) * step > last_week:
            excluded.append({'name': name, 'reason': '中盤の窓がデータの最後を越える'})
            continue

        first_week = oldest[name].to_period('W').start_time
        immediate = (release_week, release_week + (immediate_weeks - 1) * step)
        mid = (release_week + immediate_weeks * step, release_week + (mid_end - 1) * step)
        late = (release_week + mid_end * step, last_week)
        pre = (max(first_week, release_week - pre_weeks * step), release_week - step)

        row: Dict = {'name': name, 'release_week': release_week}
        for kind, table in (('count', counts), ('share', share)):
            series = table[name]
            for label, window in (('immediate', immediate), ('mid', mid), ('late', late)):
                row[f'{label}_{kind}'] = _window_stat(series, *window)
            for label, window in (('mid', mid), ('late', late)):
                row[f'{label}_median_{kind}'] = _window_stat(series, *window, how='median')
            row[f'pre_{kind}'] = _window_stat(series, *pre) if first_week < release_week else np.nan
            for label in ('mid', 'late'):
                base = row[f'immediate_{kind}']
                row[f'persist_{label}_{kind}'] = row[f'{label}_median_{kind}'] / base
                row[f'persist_mean_{label}_{kind}'] = row[f'{label}_{kind}'] / base
            row[f'pre_ratio_{kind}'] = row[f'pre_{kind}'] / row[f'immediate_{kind}']
        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(excluded, columns=['name', 'reason'])


def add_group_comparison(measured: pd.DataFrame, labels: pd.DataFrame,
                         min_group: int = MIN_GROUP) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    区分と組を付け、組の中央値からの相対値と、件数版と割合版の食い違いの印を出す

    処理の流れ:
      1. 区分を付けて組を決める
      2. 組ごとに、指標（中盤・後半 × 件数・割合）の値が min_group 本以上あれば中央値を出す
      3. 相対値 = 残った割合 ÷ 組の中央値（中央値が無ければ空）
      4. 件数版と割合版で「中央値より上か下か」が食い違う発売に印を付ける

    Returns:
        (発売ごとの表, 組ごとの本数と中央値の表)
    """
    result = measured.copy()
    result['fan_base'] = result['name'].map(labels['fan_base'])
    result['launch_type'] = result['name'].map(labels['launch_type'])
    result['group'] = [assign_group(f, t)
                       for f, t in zip(result['fan_base'], result['launch_type'])]

    summary_rows = []
    for group, part in result.groupby('group'):
        summary_rows.append({'group': group, 'n': len(part)})
    summary = pd.DataFrame(summary_rows).set_index('group')

    for window, kind in METRICS:
        col = f'persist_{window}_{kind}'
        valid = result.groupby('group')[col].agg(lambda s: s.notna().sum())
        median = result.groupby('group')[col].median().where(valid >= min_group)
        summary[f'median_{window}_{kind}'] = median
        result[f'median_{window}_{kind}'] = result['group'].map(median)
        result[f'relative_{window}_{kind}'] = result[col] / result[f'median_{window}_{kind}']

    for window in ('mid', 'late'):
        rel_count = result[f'relative_{window}_count']
        rel_share = result[f'relative_{window}_share']
        known = rel_count.notna() & rel_share.notna()
        mismatch = ((rel_count > 1) != (rel_share > 1)).astype('boolean')
        result[f'mismatch_{window}'] = mismatch.where(known)

    return result, summary.reset_index()


def persistence_curves(counts: pd.DataFrame, measured: pd.DataFrame,
                       immediate_weeks: int = IMMEDIATE_WEEKS) -> pd.DataFrame:
    """
    図用に、発売ごとの「週次件数 ÷ 発売直後の週平均」を発売からの週で並べる

    Returns:
        name / group / weeks_since_launch / ratio の縦長の表（発売週が0週目）
    """
    rows = []
    for _, r in measured.iterrows():
        series = counts[r['name']]
        after = series[series.index >= r['release_week']]
        weeks = ((after.index - r['release_week']).days // 7).astype(int)
        rows.append(pd.DataFrame({'name': r['name'], 'group': r['group'],
                                  'weeks_since_launch': weeks,
                                  'ratio': after.values / r['immediate_count']}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=['name', 'group', 'weeks_since_launch', 'ratio'])
