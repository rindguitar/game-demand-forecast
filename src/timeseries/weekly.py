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

    Args:
        denominator: 週ごとの総言及数。省略すると、渡された df 全体の週次件数を使う
        min_reviews_for_rate: ポジ率を出す最小件数。これ未満の週は欠測にする

    Returns:
        week / unit / count / games / positive_rate / share / total を持つ縦長のDataFrame
    """
    # 週の軸は、観測された週ではなく最初から最後までの連続した週にする。
    # 1件も無い週を飛ばすと、その週が時間軸から消えて系列がずれる
    weeks = pd.Index(pd.date_range(df[week_column].min(), df[week_column].max(), freq='W-MON'),
                     name=week_column)
    if denominator is None:
        denominator = df.groupby(week_column).size()
    denominator = denominator.reindex(weeks).fillna(0)

    grouped = df.groupby([unit_column, week_column])
    table = pd.DataFrame({
        'count': grouped.size(),
        'games': grouped[game_column].nunique(),
        'positives': grouped[positive_column].sum(),
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
    table.loc[before_first, ['count', 'games', 'positives']] = pd.NA

    # 4. シェアと充足度
    table = table.reset_index()
    table['total'] = table[week_column].map(denominator)
    table['share'] = table['count'] / table['total'].replace(0, pd.NA)
    enough = table['count'] >= min_reviews_for_rate
    table['positive_rate'] = (table['positives'] / table['count']).where(enough)
    return table.drop(columns=['positives']).rename(columns={unit_column: 'unit'})
