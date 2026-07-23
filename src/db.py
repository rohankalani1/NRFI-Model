"""SQLite storage for predictions, drill-down features, graded outcomes, and
the historical backtest snapshot. This is the only thing the dashboard
(app.py) reads - it never touches the large raw/processed parquet files.
"""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nrfi.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    game_pk INTEGER PRIMARY KEY,
    game_date TEXT NOT NULL,
    matchup TEXT,
    away_team TEXT,
    home_team TEXT,
    away_pitcher_id INTEGER,
    away_pitcher_name TEXT,
    home_pitcher_id INTEGER,
    home_pitcher_name TEXT,
    p_away_scores_top1 REAL,
    p_home_scores_bot1 REAL,
    p_nrfi REAL,
    p_yrfi REAL,
    pick TEXT,
    confidence_tier TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS game_features (
    game_pk INTEGER NOT NULL,
    side TEXT NOT NULL,
    batting_team TEXT,
    opponent_team TEXT,
    opposing_starter_id INTEGER,
    opposing_starter_name TEXT,
    opposing_starter_inning1_rate REAL,
    opposing_starter_overall_rate REAL,
    batting_team_slot1_4_ops REAL,
    batting_team_own_inning1_rate REAL,
    park_factor REAL,
    PRIMARY KEY (game_pk, side)
);

CREATE TABLE IF NOT EXISTS outcomes (
    game_pk INTEGER PRIMARY KEY,
    actual_away_scored INTEGER,
    actual_home_scored INTEGER,
    actual_nrfi INTEGER,
    correct INTEGER,
    graded_at TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _upsert(df: pd.DataFrame, table: str, conn: sqlite3.Connection) -> None:
    """INSERT OR REPLACE keyed on the table's own PRIMARY KEY - atomic, and
    avoids passing numpy scalar types (from .iterrows()) as bind parameters,
    which silently fail to match SQLite's INTEGER affinity and previously
    caused UNIQUE constraint violations on rerun."""
    if df.empty:
        return
    df = df.astype(object).where(pd.notnull(df), None)
    cols = list(df.columns)
    col_list = ", ".join(cols)
    placeholders = ", ".join("?" for _ in cols)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})",
        df[cols].values.tolist(),
    )
    conn.commit()


def upsert_predictions(df: pd.DataFrame) -> None:
    conn = get_connection()
    try:
        _upsert(df, "predictions", conn)
    finally:
        conn.close()


def upsert_game_features(df: pd.DataFrame) -> None:
    conn = get_connection()
    try:
        _upsert(df, "game_features", conn)
    finally:
        conn.close()


def upsert_outcomes(df: pd.DataFrame) -> None:
    conn = get_connection()
    try:
        _upsert(df, "outcomes", conn)
    finally:
        conn.close()


def write_backtest(df: pd.DataFrame) -> None:
    """Full daily replace - this table is a fresh snapshot of the held-out
    walk-forward evaluation each time the pipeline runs, not an accumulating log."""
    conn = get_connection()
    try:
        df.to_sql("backtest_results", conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def read_table(table: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


def read_predictions_for_date(date: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql("SELECT * FROM predictions WHERE game_date = ?", conn, params=(date,))
    finally:
        conn.close()


def read_live_track_record() -> pd.DataFrame:
    """Predictions joined to graded outcomes - only rows with a real result."""
    conn = get_connection()
    try:
        return pd.read_sql(
            """
            SELECT p.*, o.actual_away_scored, o.actual_home_scored, o.actual_nrfi, o.correct
            FROM predictions p
            JOIN outcomes o ON p.game_pk = o.game_pk
            """,
            conn,
        )
    finally:
        conn.close()
