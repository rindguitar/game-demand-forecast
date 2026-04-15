"""
感情分析結果のデータベースクエリ関数

DBに保存された感情分析結果を取得・集計する関数を提供します。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
from typing import Optional, List, Dict
from src.data.database import Database


def get_sentiment_by_game(app_id: int, db_path: str = 'data/game_demand.db') -> pd.DataFrame:
    """
    ゲーム別の感情分析結果を取得

    Args:
        app_id: ゲームID
        db_path: データベースファイルパス

    Returns:
        pd.DataFrame: 感情分析結果のDataFrame
    """
    db = Database(db_path)

    results = db.fetchall("""
        SELECT
            id,
            app_id,
            game_name,
            review_text,
            timestamp_created,
            sentiment,
            sentiment_score,
            analyzed_at
        FROM sentiment_results
        WHERE app_id = ?
        ORDER BY timestamp_created DESC
    """, (app_id,))

    return pd.DataFrame([dict(row) for row in results])


def get_all_sentiments(db_path: str = 'data/game_demand.db') -> pd.DataFrame:
    """
    全ゲームの感情分析結果を取得

    Args:
        db_path: データベースファイルパス

    Returns:
        pd.DataFrame: 全感情分析結果のDataFrame
    """
    db = Database(db_path)

    results = db.fetchall("""
        SELECT
            id,
            app_id,
            game_name,
            review_text,
            timestamp_created,
            sentiment,
            sentiment_score,
            analyzed_at
        FROM sentiment_results
        ORDER BY timestamp_created DESC
    """)

    return pd.DataFrame([dict(row) for row in results])


def get_sentiment_stats(app_id: Optional[int] = None, db_path: str = 'data/game_demand.db') -> pd.DataFrame:
    """
    ゲームの感情統計を取得

    Args:
        app_id: ゲームID（Noneの場合は全ゲーム）
        db_path: データベースファイルパス

    Returns:
        pd.DataFrame: 感情統計のDataFrame（game_name, sentiment, count, percentage）
    """
    db = Database(db_path)

    if app_id is not None:
        # 特定ゲームの統計
        results = db.fetchall("""
            WITH game_totals AS (
                SELECT
                    game_name,
                    sentiment,
                    COUNT(*) as count
                FROM sentiment_results
                WHERE app_id = ?
                GROUP BY game_name, sentiment
            ),
            total_count AS (
                SELECT SUM(count) as total FROM game_totals
            )
            SELECT
                game_name,
                sentiment,
                count,
                ROUND(count * 100.0 / total, 2) as percentage
            FROM game_totals, total_count
            ORDER BY sentiment DESC
        """, (app_id,))
    else:
        # 全ゲームの統計
        results = db.fetchall("""
            WITH game_totals AS (
                SELECT
                    game_name,
                    sentiment,
                    COUNT(*) as count
                FROM sentiment_results
                GROUP BY game_name, sentiment
            ),
            game_counts AS (
                SELECT
                    game_name,
                    SUM(count) as total
                FROM game_totals
                GROUP BY game_name
            )
            SELECT
                gt.game_name,
                gt.sentiment,
                gt.count,
                ROUND(gt.count * 100.0 / gc.total, 2) as percentage
            FROM game_totals gt
            JOIN game_counts gc ON gt.game_name = gc.game_name
            ORDER BY gt.game_name, gt.sentiment DESC
        """)

    return pd.DataFrame([dict(row) for row in results])


def get_overall_stats(db_path: str = 'data/game_demand.db') -> Dict[str, any]:
    """
    全体の感情統計を取得

    Args:
        db_path: データベースファイルパス

    Returns:
        Dict: 統計情報（total, positive, negative, positive_rate）
    """
    db = Database(db_path)

    result = db.fetchone("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative,
            ROUND(SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as positive_rate
        FROM sentiment_results
    """)

    return dict(result)


def get_game_list(db_path: str = 'data/game_demand.db') -> pd.DataFrame:
    """
    ゲーム一覧を取得

    Args:
        db_path: データベースファイルパス

    Returns:
        pd.DataFrame: ゲーム一覧（app_id, game_name, review_count）
    """
    db = Database(db_path)

    results = db.fetchall("""
        SELECT
            app_id,
            game_name,
            COUNT(*) as review_count
        FROM sentiment_results
        GROUP BY app_id, game_name
        ORDER BY review_count DESC
    """)

    return pd.DataFrame([dict(row) for row in results])


def get_sentiment_timeseries(
    app_id: Optional[int] = None,
    interval: str = 'day',
    db_path: str = 'data/game_demand.db'
) -> pd.DataFrame:
    """
    ゲームの感情スコア時系列データを取得

    Args:
        app_id: ゲームID（Noneの場合は全ゲーム）
        interval: 'day', 'week', 'month'のいずれか
        db_path: データベースファイルパス

    Returns:
        pd.DataFrame: 時系列データ（date, positive_count, negative_count, positive_rate, review_count）
    """
    db = Database(db_path)

    # intervalに応じてDATE関数の引数を変更
    date_format_map = {
        'day': "date(timestamp_created, 'unixepoch')",
        'week': "date(timestamp_created, 'unixepoch', 'weekday 0', '-6 days')",  # 週の開始日（月曜日）
        'month': "strftime('%Y-%m', datetime(timestamp_created, 'unixepoch'))"
    }

    if interval not in date_format_map:
        raise ValueError(f"Invalid interval: {interval}. Must be 'day', 'week', or 'month'")

    date_expr = date_format_map[interval]

    if app_id is not None:
        # 特定ゲームの時系列
        query = f"""
            SELECT
                {date_expr} as date,
                SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                ROUND(SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as positive_rate,
                COUNT(*) as review_count
            FROM sentiment_results
            WHERE app_id = ?
            GROUP BY date
            ORDER BY date
        """
        results = db.fetchall(query, (app_id,))
    else:
        # 全ゲームの時系列
        query = f"""
            SELECT
                {date_expr} as date,
                SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as negative_count,
                ROUND(SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as positive_rate,
                COUNT(*) as review_count
            FROM sentiment_results
            GROUP BY date
            ORDER BY date
        """
        results = db.fetchall(query)

    return pd.DataFrame([dict(row) for row in results])


if __name__ == '__main__':
    # 動作確認
    print("=" * 60)
    print("感情分析DBクエリ関数テスト")
    print("=" * 60)

    # 全体統計
    print("\n【全体統計】")
    overall = get_overall_stats()
    print(f"Total: {overall['total']}")
    print(f"Positive: {overall['positive']} ({overall['positive_rate']}%)")
    print(f"Negative: {overall['negative']} ({100 - overall['positive_rate']:.2f}%)")

    # ゲーム一覧
    print("\n【ゲーム一覧】")
    games = get_game_list()
    print(games.to_string(index=False))

    # ゲーム別統計
    print("\n【ゲーム別統計】")
    stats = get_sentiment_stats()
    print(stats.to_string(index=False))

    # 特定ゲームの詳細（最初のゲーム）
    if len(games) > 0:
        first_game_id = games.iloc[0]['app_id']
        first_game_name = games.iloc[0]['game_name']

        print(f"\n【{first_game_name} の統計】")
        game_stats = get_sentiment_stats(first_game_id)
        print(game_stats.to_string(index=False))

        print(f"\n【{first_game_name} のレビュー（最新5件）】")
        reviews = get_sentiment_by_game(first_game_id).head(5)
        if len(reviews) > 0:
            print(reviews[['game_name', 'sentiment', 'sentiment_score']].to_string(index=False))
        else:
            print("No reviews found")

    # 時系列データのテスト
    print("\n" + "=" * 60)
    print("時系列データテスト")
    print("=" * 60)

    # 全ゲームの月別時系列
    print("\n【全ゲーム月別時系列（最新5件）】")
    monthly = get_sentiment_timeseries(interval='month')
    if len(monthly) > 0:
        print(monthly.tail(5).to_string(index=False))
    else:
        print("No data")

    # 特定ゲームの日別時系列
    if len(games) > 0:
        first_game_id = games.iloc[0]['app_id']
        first_game_name = games.iloc[0]['game_name']

        print(f"\n【{first_game_name} 日別時系列（最新10件）】")
        daily = get_sentiment_timeseries(app_id=first_game_id, interval='day')
        if len(daily) > 0:
            print(daily.tail(10).to_string(index=False))
        else:
            print("No data")
