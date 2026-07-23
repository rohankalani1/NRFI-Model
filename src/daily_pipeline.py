"""Orchestrates the full daily run: pull yesterday's new Statcast data,
rebuild the derived tables, refresh the held-out evaluation + refit the
production model on all data, grade any predictions whose games have since
finished, predict today's slate, and write everything to data/nrfi.db.

Run manually with `python src/daily_pipeline.py`, or via
.github/workflows/daily.yml on a schedule. `--date` overrides "today", mainly
for testing/backfilling a specific day.
"""
import argparse
import datetime as dt

import pandas as pd

import build_dataset
import build_features
import db
import fetch_statcast
import grade_outcomes
import predict_today
import train_model


def run(date: str | None = None) -> None:
    if date is None:
        date = dt.date.today().isoformat()

    print(f"=== daily pipeline for {date} ===")
    db.init_db()

    print("\n--- 1. fetch statcast (incremental) ---")
    fetch_statcast.fetch_season(2026)

    print("\n--- 2. rebuild dataset ---")
    build_dataset.main()

    print("\n--- 3. rebuild features ---")
    build_features.main()

    print("\n--- 4. evaluate (held-out backtest) + train production model ---")
    train_df = pd.read_parquet(train_model.PROCESSED_DIR / "model_dataset_train.parquet")
    game_level = train_model.evaluate(train_df)
    train_model.train_production_model(train_df)

    backtest = game_level.reset_index()
    db.write_backtest(backtest)

    print("\n--- 5. grade any newly-completed games ---")
    grade_outcomes.grade_pending()

    print(f"\n--- 6. predict {date}'s slate ---")
    predictions_df, features_df = predict_today.predict_date(date)
    if predictions_df.empty:
        print("no games scheduled today")
    else:
        db.upsert_predictions(predictions_df)
        db.upsert_game_features(features_df)
        print(predictions_df[["matchup", "p_nrfi", "p_yrfi", "pick", "confidence_tier"]].to_string(index=False))

    print(f"\n=== done: {db.DB_PATH} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    run(args.date)
