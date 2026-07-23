"""Train and evaluate the 1st-inning-scoring model.

Single model trained on the pooled long-format dataset (one row per
half-inning, is_home_batting as a feature). Compared: logistic regression
baseline vs. a gradient-boosted tree. Validated with a time-based
(walk-forward) split - train on earlier games, test on the most recent slice -
never a random shuffle, since this is sequential data.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

FEATURES = [
    "batting_team_slot1_4_ops",
    "opposing_starter_inning1_rate",
    "opposing_starter_overall_rate",
    "batting_team_own_inning1_rate",
    "park_factor",
    "is_home_batting",
]
TARGET = "scored"
TEST_FRACTION = 0.2


def time_split(df: pd.DataFrame):
    dates = df["game_date"].drop_duplicates().sort_values()
    cutoff = dates.iloc[int(len(dates) * (1 - TEST_FRACTION))]
    train = df[df["game_date"] < cutoff]
    test = df[df["game_date"] >= cutoff]
    return train, test, cutoff


def calibration_table(y_true, y_pred, n_bins=5):
    bins = pd.qcut(y_pred, n_bins, duplicates="drop")
    table = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "bin": bins})
    return table.groupby("bin", observed=True).agg(
        n=("y_true", "size"), avg_predicted=("y_pred", "mean"), actual_rate=("y_true", "mean")
    )


def print_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"  log-loss:    {log_loss(y_true, y_pred):.4f}")
    print(f"  brier score: {brier_score_loss(y_true, y_pred):.4f}")
    print(f"  roc auc:     {roc_auc_score(y_true, y_pred):.4f}")
    print("  calibration (predicted vs actual, by bucket):")
    print(calibration_table(y_true, y_pred).to_string())


def combine_to_game_level(df: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    tmp = df.copy()
    tmp["pred"] = preds
    game_dates = tmp.groupby("game_pk")["game_date"].first()
    pivoted = tmp.pivot_table(index="game_pk", columns="is_home_batting", values=["pred", "scored"])
    game_level = pd.DataFrame({
        "p_away_scores_top1": pivoted[("pred", 0)],
        "p_home_scores_bot1": pivoted[("pred", 1)],
        "actual_away_scored": pivoted[("scored", 0)],
        "actual_home_scored": pivoted[("scored", 1)],
    }).dropna()
    game_level["game_date"] = game_dates
    game_level["p_nrfi"] = (1 - game_level["p_away_scores_top1"]) * (1 - game_level["p_home_scores_bot1"])
    game_level["p_yrfi"] = 1 - game_level["p_nrfi"]
    game_level["actual_nrfi"] = 1 - (
        (game_level["actual_away_scored"] > 0) | (game_level["actual_home_scored"] > 0)
    ).astype(int)
    return game_level


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based (walk-forward) held-out evaluation. This is the ONLY source
    of truth for historical accuracy claims (e.g. confidence-tier success
    rates) - the models fit here are intentionally discarded and never used
    for live predictions, since they exclude the most recent ~20% of games.
    Writes game_level_backtest.parquet and returns the same dataframe."""
    train, test, cutoff = time_split(df)
    print(f"train: {len(train):,} rows through {train['game_date'].max().date()}")
    print(f"test:  {len(test):,} rows from {cutoff.date()} onward")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    logit = make_pipeline(StandardScaler(), LogisticRegression())
    logit.fit(X_train, y_train)
    logit_preds = logit.predict_proba(X_test)[:, 1]
    print_metrics("Logistic Regression", y_test, logit_preds)

    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict_proba(X_test)[:, 1]
    print_metrics("XGBoost", y_test, xgb_preds)

    print("\nfeature importances (xgboost):")
    for feat, imp in sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    best_preds = logit_preds if log_loss(y_test, logit_preds) < log_loss(y_test, xgb_preds) else xgb_preds
    best_name = "logit" if best_preds is logit_preds else "xgboost"
    print(f"\nusing {best_name} for game-level NRFI backtest (lower test log-loss)")

    game_level = combine_to_game_level(test, best_preds)
    print(f"\ngame-level NRFI backtest ({len(game_level)} games in test period):")
    print(f"  avg predicted NRFI prob: {game_level['p_nrfi'].mean():.3f}")
    print(f"  actual NRFI rate:        {game_level['actual_nrfi'].mean():.3f}")
    print(f"  log-loss (NRFI):         {log_loss(game_level['actual_nrfi'], game_level['p_nrfi']):.4f}")
    print(f"  brier score (NRFI):      {brier_score_loss(game_level['actual_nrfi'], game_level['p_nrfi']):.4f}")

    game_level.to_parquet(PROCESSED_DIR / "game_level_backtest.parquet")
    return game_level


def train_production_model(df: pd.DataFrame):
    """Fits the logistic model on ALL available history (no held-out split)
    so live predictions always use the freshest possible data. Saved
    separately from the evaluation models above."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES], df[TARGET]

    logit = make_pipeline(StandardScaler(), LogisticRegression())
    logit.fit(X, y)
    joblib.dump(logit, MODELS_DIR / "logit_model.joblib")

    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
    )
    xgb.fit(X, y)
    joblib.dump(xgb, MODELS_DIR / "xgb_model.joblib")

    print(f"production models (fit on all {len(df):,} rows) saved to {MODELS_DIR}")
    return logit, xgb


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "model_dataset_train.parquet")
    evaluate(df)
    train_production_model(df)


if __name__ == "__main__":
    main()
