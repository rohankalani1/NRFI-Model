"""Grade past predictions once the real games have finished. Compares each
un-graded row in the `predictions` table against the real result in
games.parquet (which lags by one day, same as everything else in this
pipeline) and writes the result into `outcomes`.
"""
from pathlib import Path

import pandas as pd

import db

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def grade_pending() -> int:
    db.init_db()
    predictions = db.read_table("predictions")
    if predictions.empty:
        print("no predictions to grade")
        return 0

    existing_outcomes = db.read_table("outcomes")
    graded_pks = set(existing_outcomes["game_pk"]) if not existing_outcomes.empty else set()
    pending = predictions[~predictions["game_pk"].isin(graded_pks)]
    if pending.empty:
        print("no pending predictions to grade")
        return 0

    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    matched = pending.merge(games[["game_pk", "away_scored_top1", "home_scored_bottom1"]], on="game_pk", how="inner")
    if matched.empty:
        print(f"{len(pending)} pending predictions, none have completed games available yet")
        return 0

    matched["actual_away_scored"] = matched["away_scored_top1"]
    matched["actual_home_scored"] = matched["home_scored_bottom1"]
    matched["actual_nrfi"] = (
        1 - ((matched["actual_away_scored"] > 0) | (matched["actual_home_scored"] > 0)).astype(int)
    )
    matched["correct"] = (
        ((matched["pick"] == "NRFI") & (matched["actual_nrfi"] == 1))
        | ((matched["pick"] == "YRFI") & (matched["actual_nrfi"] == 0))
    ).astype(int)
    matched["graded_at"] = pd.Timestamp.now().isoformat()

    outcomes = matched[
        ["game_pk", "actual_away_scored", "actual_home_scored", "actual_nrfi", "correct", "graded_at"]
    ]
    db.upsert_outcomes(outcomes)
    print(f"graded {len(outcomes)} newly-completed games")
    return len(outcomes)


if __name__ == "__main__":
    grade_pending()
