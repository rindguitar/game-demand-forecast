"""
SQLiteデータベース操作モジュール

ゲーム需要予測システムのデータベース操作を提供します。
"""

import sqlite3
from typing import Optional, List, Dict, Any
from pathlib import Path


class Database:
    """SQLiteデータベース操作クラス"""

    def __init__(self, db_path: str = 'data/game_demand.db'):
        """
        データベース初期化

        Args:
            db_path: データベースファイルパス
        """
        self.db_path = db_path
        self._ensure_db_directory()

    def _ensure_db_directory(self):
        """データベースディレクトリが存在することを確認"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def get_connection(self) -> sqlite3.Connection:
        """
        データベース接続を取得

        Returns:
            sqlite3.Connection: データベース接続
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # カラム名でアクセス可能にする
        return conn

    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        """
        SQLクエリを実行

        Args:
            query: SQLクエリ
            params: クエリパラメータ

        Returns:
            sqlite3.Cursor: 実行結果のカーソル
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        conn.close()
        return cursor

    def executemany(self, query: str, params_list: List[tuple]) -> None:
        """
        SQLクエリを複数回実行（バッチ処理）

        Args:
            query: SQLクエリ
            params_list: クエリパラメータのリスト
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        conn.commit()
        conn.close()

    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """
        SQLクエリを実行して全結果を取得

        Args:
            query: SQLクエリ
            params: クエリパラメータ

        Returns:
            List[sqlite3.Row]: クエリ結果のリスト
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        return results

    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """
        SQLクエリを実行して1件の結果を取得

        Args:
            query: SQLクエリ
            params: クエリパラメータ

        Returns:
            Optional[sqlite3.Row]: クエリ結果（存在しない場合はNone）
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        return result

    def create_tables(self):
        """全テーブルを作成"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # sentiment_results テーブル
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                app_id INTEGER NOT NULL,
                game_name TEXT,
                review_text TEXT NOT NULL,
                timestamp_created INTEGER,
                sentiment TEXT NOT NULL,
                sentiment_score REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT DEFAULT 'distilbert-phase2'
            )
        """)

        # インデックス作成
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_app_id
            ON sentiment_results(app_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON sentiment_results(timestamp_created)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sentiment
            ON sentiment_results(sentiment)
        """)

        conn.commit()
        conn.close()
        print("✅ Database tables created successfully")


def init_database(db_path: str = 'data/game_demand.db') -> Database:
    """
    データベースを初期化

    Args:
        db_path: データベースファイルパス

    Returns:
        Database: 初期化されたDatabaseインスタンス
    """
    db = Database(db_path)
    db.create_tables()
    return db


if __name__ == '__main__':
    # データベース初期化テスト
    print("Initializing database...")
    db = init_database()
    print("Database initialized successfully!")
