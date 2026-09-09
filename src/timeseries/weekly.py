"""
トピックの週次時系列を作るモジュール

需要の「大きさ」（言及数）と「充足度」（ポジ率）を別の軸として持つ
（docs/decisions.md 2026-08-18）。Y軸はシェアを主軸、総言及数を補助線にする
（同 2026-08-31）ため、両方を同じ表に載せる。

実データで見つかった落とし穴を2つ扱う:
  - 収集の端は**部分週**になる（最終週が5日分しかない等）。件数が落ちて見えるので落とす
  - 系列ごとに**初出より前の週がある**（発売前のゲームの要素など）。ここを0で埋めると
    「需要がゼロだった」ことになるが、実際は観測対象が存在していない。欠測のまま残す
"""

from typing import Optional
import pandas as pd


def add_week_column(df: pd.DataFrame, timestamp_column: str = 'timestamp_created',
                    week_column: str = 'week') -> pd.DataFrame:
    """UNIX秒のタイムスタンプ列から、その週の月曜日を指す列を足す"""
    result = df.copy()
    date = pd.to_datetime(result[timestamp_column], unit='s')
    result[week_column] = date.dt.to_period('W').dt.start_time
    result['_date'] = date
    return result


def trim_partial_weeks(df: pd.DataFrame, week_column: str = 'week',
                       date_column: str = '_date') -> pd.DataFrame:
    """
    端の部分週を落とす

    収集期間の最初と最後は7日そろっていないことがある（実測: 最終週が5日分で
    件数が前週の6割に落ちた）。件数の増減として誤読されるので、週として数えない。
    """
    if df.empty:
        return df
    first_week, last_week = df[week_column].min(), df[week_column].max()
    starts_mid_week = df[date_column].min() > first_week
    ends_mid_week = df[date_column].max() < last_week + pd.Timedelta(days=6, hours=23, minutes=59)

    drop = []
    if starts_mid_week:
        drop.append(first_week)
    if ends_mid_week:
        drop.append(last_week)
    return df[~df[week_column].isin(drop)]


def build_weekly_series(df: pd.DataFrame,
                        unit_column: str = 'topic_id',
                        week_column: str = 'week',
                        game_column: str = 'game_name',
                        positive_column: str = 'voted_up',
                        denominator: Optional[pd.Series] = None,
                        min_reviews_for_rate: int = 5) -> pd.DataFrame:
    """
    単位（トピックや束）ごとの週次系列を作る

    処理の流れ:
      1. 単位 × 週で件数・参加ゲーム数・ポジ率を集計する
      2. 全ての週を埋めた形に整える（欠けた週は件数0）
      3. 各単位の初出より前の週は欠測に戻す（需要ゼロではなく観測対象外のため）
      4. その週の総言及数で割ってシェアを出す

    充足度は「実際のポジ率」だけでなく「期待ポジ率」も出す。トピックは特定のゲームに
    偏るので、実際の率だけ見るとそのゲームの評判をそのまま読んでしまう
    （実測: cards は50%だが、中身の93%を占める MTG Arena 自体が53%だった）。
    期待ポジ率は「そのトピック・その週のゲーム構成なら何%になるはずか」で、
    差し引いた `positive_rate_gap` が要素そのものの効き方になる。

    Args:
        denominator: 週ごとの総言及数。省略すると、渡された df 全体の週次件数を使う
        min_reviews_for_rate: ポジ率を出す最小件数。これ未満の週は欠測にする

    Returns:
        week / unit / count / games / share / total /
        positive_rate / expected_positive_rate / positive_rate_gap を持つ縦長のDataFrame
    """
    # 週の軸は、観測された週ではなく最初から最後までの連続した週にする。
    # 1件も無い週を飛ばすと、その週が時間軸から消えて系列がずれる
    weeks = pd.Index(pd.date_range(df[week_column].min(), df[week_column].max(), freq='W-MON'),
                     name=week_column)
    if denominator is None:
        denominator = df.groupby(week_column).size()
    denominator = denominator.reindex(weeks).fillna(0)

    # ゲームごとの全期間のポジ率を、各レビューに貼る。
    # それを平均すると「そのトピック・その週のゲーム構成から期待されるポジ率」になる
    game_rates = df.groupby(game_column)[positive_column].mean()
    working = df.copy()
    working['_game_rate'] = working[game_column].map(game_rates)

    grouped = working.groupby([unit_column, week_column])
    table = pd.DataFrame({
        'count': grouped.size(),
        'games': grouped[game_column].nunique(),
        'positives': grouped[positive_column].sum(),
        'expected_positive_rate': grouped['_game_rate'].mean(),
    })

    # 2. 全週 × 全単位の格子に広げる（欠けた週は0件）
    units = pd.Index(sorted(df[unit_column].unique()), name=unit_column)
    grid = pd.MultiIndex.from_product([units, weeks])
    table = table.reindex(grid, fill_value=0)

    # 3. 初出より前は欠測に戻す
    observed = table['count'] > 0
    first_seen = observed[observed].reset_index().groupby(unit_column)[week_column].min()
    weeks_level = table.index.get_level_values(week_column)
    units_level = table.index.get_level_values(unit_column)
    before_first = weeks_level < units_level.map(first_seen)
    table.loc[before_first, ['count', 'games', 'positives', 'expected_positive_rate']] = pd.NA

    # 4. シェアと充足度
    table = table.reset_index()
    table['total'] = table[week_column].map(denominator)
    table['share'] = table['count'] / table['total'].replace(0, pd.NA)
    enough = table['count'] >= min_reviews_for_rate
    table['positive_rate'] = (table['positives'] / table['count']).where(enough)
    table['expected_positive_rate'] = table['expected_positive_rate'].where(enough)
    table['positive_rate_gap'] = table['positive_rate'] - table['expected_positive_rate']
    return table.drop(columns=['positives']).rename(columns={unit_column: 'unit'})


def weekly_median(df: pd.DataFrame, week_axis: pd.Index,
                  unit_column: str = 'topic_id') -> pd.Series:
    """単位ごとに「週あたり件数の中央値」を出す

    平均だと発売スパイク型が密度十分に見えてしまう
    （実測: metroidvania は平均16.7件/週だが中央値は1.0件）。
    """
    table = df.groupby([unit_column, 'week']).size().unstack(fill_value=0)
    return table.reindex(columns=week_axis, fill_value=0).median(axis=1)


def measure_topic_panels(df: pd.DataFrame, backbone_games,
                         unit_column: str = 'topic_id',
                         game_column: str = 'game_name',
                         outlier_id: int = -1):
    """レビュー本体から、パネルごとの週あたり件数とゲーム集中度を測る

    処理の流れ:
      1. 週の列を足し、端の部分週を落とす
      2. Outlier を除いた分について、単位ごとの件数・参加ゲーム数・集中度を出す
      3. 全24本パネルと土台パネルのそれぞれで週あたり件数の中央値を出す

    粒度を変えて比べるとき、物差しが1つでないと比較が成り立たないため、
    分類スクリプトと粒度比較スクリプトの双方がこの関数を使う。

    Args:
        df: game_column / timestamp_created / unit_column を持つ生のDataFrame
        backbone_games: 土台パネルとして数えるゲーム名の並び

    Returns:
        (単位ごとの指標のDataFrame, 全体の件数などの dict)
    """
    df = trim_partial_weeks(add_week_column(df))
    week_axis = pd.date_range(df['week'].min(), df['week'].max(), freq='W-MON')
    assigned = df[df[unit_column] != outlier_id]

    grouped = assigned.groupby(unit_column)[game_column]
    panels = pd.DataFrame({
        'count_all': assigned[unit_column].value_counts(),
        'games': grouped.nunique(),
        'top1_share': grouped.apply(lambda s: s.value_counts(normalize=True).iloc[0]),
        'top1_game': grouped.apply(lambda s: s.value_counts().index[0]),
    })
    panels['mean_per_week_all'] = panels['count_all'] / len(week_axis)
    panels['per_week_all'] = weekly_median(assigned, week_axis, unit_column)

    bb = assigned[assigned[game_column].isin(backbone_games)]
    panels['count_backbone'] = bb[unit_column].value_counts()
    panels['count_backbone'] = panels['count_backbone'].fillna(0).astype(int)
    panels['per_week_backbone'] = (weekly_median(bb, week_axis, unit_column)
                                   .reindex(panels.index).fillna(0))

    totals = {'all_reviews': len(df), 'assigned': len(assigned),
              'backbone_assigned': len(bb), 'weeks': len(week_axis)}
    return panels, totals
