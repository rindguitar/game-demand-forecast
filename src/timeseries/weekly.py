"""
トピックの週次時系列を作るモジュール

需要の「大きさ」（言及数）と「充足度」（ポジ率）を別の軸として持つ
（docs/decisions.md 2026-08-18）。Y軸はシェアを主軸、総言及数を補助線にする
（同 2026-08-31）ため、両方を同じ表に載せる。

実データで見つかった落とし穴を3つ扱う:
  - ゲームごとに**収集期間がずれる**（ロスターを2回に分けて集めた）。収集の端は部分週になり、
    端を落とすだけでは先に集め終えたゲームの側に「集めていないので0件」の偽の谷が残る。
    全ゲームの収集がそろう**共通の期間**に絞る（common_window / trim_to_window）
  - **発売直後の減り**が土台に入る。期間開始より十分前に発売したゲームだけを土台にする
    （select_backbone）
  - 系列ごとに**初出より前の週がある**（発売前のゲームの要素など）。ここを0で埋めると
    「需要がゼロだった」ことになるが、実際は観測対象が存在していない。欠測のまま残す
"""

from typing import Iterable, List, Optional, Sequence, Tuple
import pandas as pd

# 共通の期間 = (最初の週の月曜, 最後の週の月曜)。どちらの週も含む
Window = Tuple[pd.Timestamp, pd.Timestamp]

# 土台に入れるゲームが、期間開始より何週以上前に発売していること。
# 発売直後の減りは13週前でも見え、18週・20週前では見えなかった。遅れて来る波も考えて
# 余裕を持たせた（docs/decisions.md 2026-10-01）
MIN_WEEKS_SINCE_RELEASE = 26


def add_week_column(df: pd.DataFrame, timestamp_column: str = 'timestamp_created',
                    week_column: str = 'week') -> pd.DataFrame:
    """UNIX秒のタイムスタンプ列から、その週の月曜日を指す列を足す"""
    result = df.copy()
    date = pd.to_datetime(result[timestamp_column], unit='s')
    result[week_column] = date.dt.to_period('W').dt.start_time
    result['_date'] = date
    return result


def common_window(log: pd.DataFrame, names: Iterable[str]) -> Window:
    """
    全ゲームの収集がそろう期間を、収集ログから出す

    処理の流れ:
      1. 対象ゲームの行を収集ログから取る（ログに無い・日付が空のゲームがあれば止まる）
      2. 開始 = oldest のうち最も遅い日。その日を含む週は途中からしか集めていないので使わず、
         次の月曜から始める
      3. 終了 = newest のうち最も早い日。その日を含む週は途中までしか集めていないので使わず、
         前の週の月曜で終える

    発売が期間より後のゲームは、遡る先が無く oldest が発売日になる。期間を狭めて
    しまうので、呼び出し側は対象に入れない（土台のゲームだけを渡す）。

    Args:
        log: 収集ログ（collection_log.csv）を読んだDataFrame。name / oldest / newest 列が必要
        names: 期間の決定に使うゲーム名の並び

    Returns:
        (最初の週の月曜, 最後の週の月曜)。どちらの週も期間に含む
    """
    names = list(names)
    if not names:
        raise ValueError('期間を決めるゲームが1本も渡されていない')

    # 1. 同じゲームの行が複数あれば最後の行を使う（収集側 load_collection_log と同じ規則）
    latest = log.drop_duplicates('name', keep='last').set_index('name')
    missing = [name for name in names if name not in latest.index]
    if missing:
        raise ValueError(f'収集ログに無いゲームがある: {missing}')
    rows = latest.loc[names]
    oldest = pd.to_datetime(rows['oldest'], errors='coerce')
    newest = pd.to_datetime(rows['newest'], errors='coerce')
    empty = rows.index[oldest.isna() | newest.isna()].tolist()
    if empty:
        raise ValueError(f'収集ログの oldest / newest が空のゲームがある: {empty}')

    # 2. 開始と終了は、端の日を含む週を1週ぶん内側へ寄せる
    start = oldest.max().to_period('W').start_time + pd.Timedelta(weeks=1)
    end = newest.min().to_period('W').start_time - pd.Timedelta(weeks=1)
    if start > end:
        raise ValueError(f'全ゲームの収集がそろう週が無い（{start:%Y-%m-%d} > {end:%Y-%m-%d}）')
    return start, end


def trim_to_window(df: pd.DataFrame, window: Window,
                   week_column: str = 'week') -> pd.DataFrame:
    """共通の期間（最初の週の月曜〜最後の週の月曜）に入る週のレビューだけを残す"""
    start, end = window
    return df[(df[week_column] >= start) & (df[week_column] <= end)]


def select_backbone(games: pd.DataFrame, window_start: pd.Timestamp,
                    tier: str = '土台',
                    min_weeks: int = MIN_WEEKS_SINCE_RELEASE,
                    exclude: Optional[Sequence[str]] = None) -> List[str]:
    """
    土台パネルに入れるゲームを選ぶ

    発売直後の件数は大きく減っていくので、その減りが土台に入らないように、
    期間開始の min_weeks 週前までに発売したゲームだけを選ぶ。

    処理の流れ:
      1. tier が一致するゲームに絞る
      2. 発売日が読めないゲームがあれば止まる（黙って含めたり外したりしない）
      3. 発売日が「期間開始 − min_weeks 週」以前のゲームだけ残す
      4. exclude に挙げたゲームを外す（手で外したいときだけ。既定は外さない）

    Args:
        games: ゲーム台帳。name / release_date / tier 列が必要
        window_start: 期間の開始（common_window が返す最初の週の月曜）
        tier: 土台として扱う tier の値
        min_weeks: 発売から期間開始までに必要な週数。ちょうどその週数前の発売は含める
        exclude: 手で外すゲーム名

    Returns:
        ゲーム名のリスト（台帳の並び順）
    """
    cutoff = pd.Timestamp(window_start) - pd.Timedelta(weeks=min_weeks)

    candidates = games[games['tier'] == tier]
    released = pd.to_datetime(candidates['release_date'], errors='coerce')
    unreadable = candidates.loc[released.isna(), 'name'].tolist()
    if unreadable:
        raise ValueError(f'発売日が読めないゲームがある: {unreadable}')

    names = candidates.loc[released <= cutoff, 'name'].tolist()
    excluded = set(exclude or [])
    return [name for name in names if name not in excluded]


def decide_window_and_backbone(games: pd.DataFrame, log: pd.DataFrame,
                               tier: str = '土台',
                               min_weeks: int = MIN_WEEKS_SINCE_RELEASE,
                               exclude: Optional[Sequence[str]] = None
                               ) -> Tuple[Window, List[str], List[str]]:
    """
    共通の期間と、土台パネルのゲームを決める

    build_weekly_series.py / categorize_topics.py / compare_topic_granularity.py の3本が
    同じ入口として呼ぶ。期間と土台の定義が3か所に分かれないようにするため。

    処理の流れ:
      1. 台帳から tier が一致するゲームを取る
      2. そのゲームの収集ログから共通の期間を出す（common_window）
      3. 期間の開始から、土台に入れるゲームを選ぶ（select_backbone）
      4. tier が一致したのに土台から外れたゲームを集める

    Args:
        games: ゲーム台帳。name / release_date / tier 列が必要
        log: 収集ログ（collection_log.csv）を読んだDataFrame
        tier / min_weeks / exclude: select_backbone と同じ

    Returns:
        (期間, 土台のゲーム名, 外れたゲーム名)。期間は (最初の週の月曜, 最後の週の月曜)。
        外れたゲームには、exclude で手で外したものも含む
    """
    tier_games = games.loc[games['tier'] == tier, 'name'].tolist()
    window = common_window(log, tier_games)
    backbone = select_backbone(games, window[0], tier=tier, min_weeks=min_weeks,
                               exclude=exclude)
    left_out = [name for name in tier_games if name not in backbone]
    return window, backbone, left_out


def describe_window_and_backbone(window: Window, backbone: Sequence[str],
                                 left_out: Sequence[str], tier: str, min_weeks: int) -> str:
    """決めた期間と土台を、画面に出す3行の文にする（3本のスクリプトで同じ表示にするため）"""
    weeks = len(pd.date_range(window[0], window[1], freq='W-MON'))
    return (f"期間: {window[0]:%Y-%m-%d} 〜 {window[1]:%Y-%m-%d}"
            f"（{weeks}週・全ゲームの収集がそろう範囲）\n"
            f"土台パネル: {len(backbone)}本（tier={tier} で、発売が期間開始の{min_weeks}週以上前）\n"
            f"  外れたゲーム（{len(left_out)}本）: {', '.join(left_out) or 'なし'}")


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


def measure_topic_panels(df: pd.DataFrame, backbone_games, window: Window,
                         unit_column: str = 'topic_id',
                         game_column: str = 'game_name',
                         outlier_id: int = -1):
    """レビュー本体から、パネルごとの週あたり件数とゲーム集中度を測る

    処理の流れ:
      1. 週の列を足し、共通の期間に絞る
      2. Outlier を除いた分について、単位ごとの件数・参加ゲーム数・集中度を出す
      3. 全ゲームパネルと土台パネルのそれぞれで週あたり件数の中央値を出す

    粒度を変えて比べるとき、物差しが1つでないと比較が成り立たないため、
    分類スクリプトと粒度比較スクリプトの双方がこの関数を使う。

    期間（window）は省略できない。省略できると、収集の端の週が週あたり件数の中央値に
    混ざる古いやり方へ黙って戻るため（docs/decisions.md 2026-09-27 と同じ考え方）。

    Args:
        df: game_column / timestamp_created / unit_column を持つ生のDataFrame
        backbone_games: 土台パネルとして数えるゲーム名の並び
        window: 共通の期間 (最初の週の月曜, 最後の週の月曜)。common_window が返す。
            週の軸もこの期間そのものになる

    Returns:
        (単位ごとの指標のDataFrame, 全体の件数などの dict)
    """
    df = trim_to_window(add_week_column(df), window)
    week_axis = pd.date_range(window[0], window[1], freq='W-MON')
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
