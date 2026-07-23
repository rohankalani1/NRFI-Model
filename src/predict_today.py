"""Predict NRFI/YRFI probability for today's games using the trained model.

Uses probable starters (schedule) and actual confirmed batting order (boxscore,
once available) for identity, and rolling stats computed from all historical
data through the last cached date (i.e. no data from "today" itself is used
for features - only who's playing).
"""
import argparse
import datetime as dt
from pathlib import Path

import joblib
import pandas as pd
import requests

from player_names import get_names

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

ROLLING_WINDOW = 15

FEATURES = [
    "batting_team_slot1_4_ops",
    "opposing_starter_inning1_rate",
    "opposing_starter_overall_rate",
    "batting_team_own_inning1_rate",
    "park_factor",
    "is_home_batting",
]


def team_id_to_code() -> dict:
    r = requests.get("https://statsapi.mlb.com/api/v1/teams", params={"sportId": 1}, timeout=30)
    return {t["id"]: t["abbreviation"] for t in r.json()["teams"]}


def todays_games(date: str) -> list:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date, "gameType": "R", "hydrate": "probablePitcher"},
        timeout=30,
    )
    data = r.json()
    return data["dates"][0]["games"] if data["dates"] else []


def confirmed_lineup(game_pk: int, side: str) -> list:
    """side is 'away' or 'home'. Returns up to 4 batter ids in slots 1-4, or [] if not posted yet."""
    r = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=30)
    data = r.json()
    players = data["teams"][side]["players"]
    slots = []
    for p in players.values():
        order = p.get("battingOrder")
        if order and order.endswith("00"):
            slots.append((int(order), p["person"]["id"]))
    slots.sort()
    return [pid for _, pid in slots[:4]]


def typical_lineup(games: pd.DataFrame, team_code: str) -> list:
    """Fallback when a lineup isn't posted yet: most frequent slot-1-4 batters
    over the team's last 15 games."""
    team_games = games[(games["away_team"] == team_code) | (games["home_team"] == team_code)].sort_values("game_date").tail(ROLLING_WINDOW)
    batter_cols_away = ["away_top1_batter_1", "away_top1_batter_2", "away_top1_batter_3", "away_top1_batter_4"]
    batter_cols_home = ["home_bot1_batter_1", "home_bot1_batter_2", "home_bot1_batter_3", "home_bot1_batter_4"]
    counts = {}
    for _, row in team_games.iterrows():
        cols = batter_cols_away if row["away_team"] == team_code else batter_cols_home
        for slot_idx, col in enumerate(cols):
            pid = row[col]
            if pd.notna(pid):
                counts.setdefault(slot_idx, {})
                counts[slot_idx][pid] = counts[slot_idx].get(pid, 0) + 1
    result = []
    for slot_idx in range(4):
        if slot_idx in counts and counts[slot_idx]:
            best = max(counts[slot_idx].items(), key=lambda kv: kv[1])[0]
            result.append(best)
    return result


def latest_rolling_ops(player_game: pd.DataFrame, batter_id) -> float | None:
    g = player_game[player_game["batter"] == batter_id].sort_values("game_date").tail(ROLLING_WINDOW)
    if g.empty:
        return None
    ab, h, tb, bb, hbp, sf = g["ab"].sum(), g["h"].sum(), g["tb"].sum(), g["bb"].sum(), g["hbp"].sum(), g["sf"].sum()
    denom = ab + bb + hbp + sf
    if denom == 0:
        return None
    obp = (h + bb + hbp) / denom
    slg = tb / ab if ab > 0 else 0
    return obp + slg


def latest_pitcher_rate(pitcher_log: pd.DataFrame, pitcher_id) -> float | None:
    g = pitcher_log[pitcher_log["pitcher"] == pitcher_id].sort_values("game_date").tail(ROLLING_WINDOW)
    if g.empty:
        return None
    bf = g["batters_faced"].sum()
    baserunners = g["hits_allowed"].sum() + g["walks_allowed"].sum() + g["hbp_allowed"].sum()
    if bf == 0:
        return None
    return baserunners / bf


def latest_team_rate(games: pd.DataFrame, team_code: str) -> float | None:
    away_hist = games[games["away_team"] == team_code][["game_date", "away_scored_top1"]].rename(
        columns={"away_scored_top1": "scored"}
    )
    home_hist = games[games["home_team"] == team_code][["game_date", "home_scored_bottom1"]].rename(
        columns={"home_scored_bottom1": "scored"}
    )
    hist = pd.concat([away_hist, home_hist]).sort_values("game_date").tail(ROLLING_WINDOW)
    if hist.empty:
        return None
    return hist["scored"].mean()


def confidence_tier(pick_prob: float) -> str:
    if pick_prob >= 0.55:
        return "55%+"
    if pick_prob >= 0.53:
        return "53%+"
    return "none"


def predict_date(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions_df, features_df) for every game scheduled on `date`.
    predictions_df has one row per game; features_df has one row per side
    (away/home) with the drill-down feature values behind that prediction."""
    player_game = pd.read_parquet(PROCESSED_DIR / "player_game_batting.parquet")
    pitcher_inn1 = pd.read_parquet(PROCESSED_DIR / "pitcher_game_inning1.parquet")
    pitcher_full = pd.read_parquet(PROCESSED_DIR / "pitcher_game_full.parquet")
    games_hist = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    park_factors = pd.read_csv(RAW_DIR / "park_factors_2026.csv")
    model = joblib.load(MODELS_DIR / "logit_model.joblib")

    team_codes = team_id_to_code()
    games_today = todays_games(date)
    print(f"{len(games_today)} games scheduled for {date}\n")

    # fallback values (for players/pitchers with no history yet) come from the trained feature table
    full_features = pd.read_parquet(PROCESSED_DIR / "model_dataset_full.parquet")
    medians = full_features[FEATURES[:-2]].median()

    pred_rows = []
    feature_rows = []
    all_pitcher_ids = []
    for g in games_today:
        game_pk = g["gamePk"]
        away = g["teams"]["away"]
        home = g["teams"]["home"]
        away_code = team_codes.get(away["team"]["id"])
        home_code = team_codes.get(home["team"]["id"])
        away_pitcher = away.get("probablePitcher", {}).get("id")
        home_pitcher = home.get("probablePitcher", {}).get("id")

        away_batters = confirmed_lineup(game_pk, "away")
        home_batters = confirmed_lineup(game_pk, "home")
        if not away_batters:
            away_batters = typical_lineup(games_hist, away_code)
        if not home_batters:
            home_batters = typical_lineup(games_hist, home_code)

        away_ops_vals = [latest_rolling_ops(player_game, b) for b in away_batters]
        away_ops_vals = [v for v in away_ops_vals if v is not None]
        away_slot_ops = sum(away_ops_vals) / len(away_ops_vals) if away_ops_vals else medians["batting_team_slot1_4_ops"]

        home_ops_vals = [latest_rolling_ops(player_game, b) for b in home_batters]
        home_ops_vals = [v for v in home_ops_vals if v is not None]
        home_slot_ops = sum(home_ops_vals) / len(home_ops_vals) if home_ops_vals else medians["batting_team_slot1_4_ops"]

        home_pitcher_inn1 = latest_pitcher_rate(pitcher_inn1, home_pitcher)
        home_pitcher_inn1 = home_pitcher_inn1 if home_pitcher_inn1 is not None else medians["opposing_starter_inning1_rate"]
        away_pitcher_inn1 = latest_pitcher_rate(pitcher_inn1, away_pitcher)
        away_pitcher_inn1 = away_pitcher_inn1 if away_pitcher_inn1 is not None else medians["opposing_starter_inning1_rate"]

        home_pitcher_overall = latest_pitcher_rate(pitcher_full, home_pitcher)
        home_pitcher_overall = home_pitcher_overall if home_pitcher_overall is not None else medians["opposing_starter_overall_rate"]
        away_pitcher_overall = latest_pitcher_rate(pitcher_full, away_pitcher)
        away_pitcher_overall = away_pitcher_overall if away_pitcher_overall is not None else medians["opposing_starter_overall_rate"]

        away_team_rate = latest_team_rate(games_hist, away_code)
        away_team_rate = away_team_rate if away_team_rate is not None else medians["batting_team_own_inning1_rate"]
        home_team_rate = latest_team_rate(games_hist, home_code)
        home_team_rate = home_team_rate if home_team_rate is not None else medians["batting_team_own_inning1_rate"]

        park_row = park_factors[park_factors["team_code"] == home_code]
        park_factor = park_row["park_factor"].iloc[0] if not park_row.empty else 100

        X = pd.DataFrame([
            {
                "batting_team_slot1_4_ops": away_slot_ops,
                "opposing_starter_inning1_rate": home_pitcher_inn1,
                "opposing_starter_overall_rate": home_pitcher_overall,
                "batting_team_own_inning1_rate": away_team_rate,
                "park_factor": park_factor,
                "is_home_batting": 0,
            },
            {
                "batting_team_slot1_4_ops": home_slot_ops,
                "opposing_starter_inning1_rate": away_pitcher_inn1,
                "opposing_starter_overall_rate": away_pitcher_overall,
                "batting_team_own_inning1_rate": home_team_rate,
                "park_factor": park_factor,
                "is_home_batting": 1,
            },
        ])[FEATURES]

        p_away_scores, p_home_scores = model.predict_proba(X)[:, 1]
        p_nrfi = (1 - p_away_scores) * (1 - p_home_scores)
        p_yrfi = 1 - p_nrfi
        pick = "NRFI" if p_nrfi >= 0.5 else "YRFI"

        all_pitcher_ids.extend([away_pitcher, home_pitcher])

        pred_rows.append({
            "game_pk": game_pk,
            "game_date": date,
            "matchup": f"{away_code} @ {home_code}",
            "away_team": away_code,
            "home_team": home_code,
            "away_pitcher_id": away_pitcher,
            "home_pitcher_id": home_pitcher,
            "p_away_scores_top1": round(p_away_scores, 3),
            "p_home_scores_bot1": round(p_home_scores, 3),
            "p_nrfi": round(p_nrfi, 3),
            "p_yrfi": round(p_yrfi, 3),
            "pick": pick,
            "confidence_tier": confidence_tier(max(p_nrfi, p_yrfi)),
            "created_at": dt.datetime.now().isoformat(),
        })

        feature_rows.append({
            "game_pk": game_pk, "side": "away",
            "batting_team": away_code, "opponent_team": home_code,
            "opposing_starter_id": home_pitcher,
            "opposing_starter_inning1_rate": round(home_pitcher_inn1, 3),
            "opposing_starter_overall_rate": round(home_pitcher_overall, 3),
            "batting_team_slot1_4_ops": round(away_slot_ops, 3),
            "batting_team_own_inning1_rate": round(away_team_rate, 3),
            "park_factor": park_factor,
        })
        feature_rows.append({
            "game_pk": game_pk, "side": "home",
            "batting_team": home_code, "opponent_team": away_code,
            "opposing_starter_id": away_pitcher,
            "opposing_starter_inning1_rate": round(away_pitcher_inn1, 3),
            "opposing_starter_overall_rate": round(away_pitcher_overall, 3),
            "batting_team_slot1_4_ops": round(home_slot_ops, 3),
            "batting_team_own_inning1_rate": round(home_team_rate, 3),
            "park_factor": park_factor,
        })

    names = get_names([p for p in all_pitcher_ids if p is not None])

    predictions_df = pd.DataFrame(pred_rows)
    if not predictions_df.empty:
        predictions_df["away_pitcher_name"] = predictions_df["away_pitcher_id"].map(names)
        predictions_df["home_pitcher_name"] = predictions_df["home_pitcher_id"].map(names)
        predictions_df = predictions_df.sort_values("p_nrfi", ascending=False)

    features_df = pd.DataFrame(feature_rows)
    if not features_df.empty:
        features_df["opposing_starter_name"] = features_df["opposing_starter_id"].map(names)

    return predictions_df, features_df


def main(date: str | None = None) -> None:
    if date is None:
        date = dt.date.today().isoformat()

    predictions_df, _ = predict_date(date)
    display_cols = [
        "matchup", "away_pitcher_id", "home_pitcher_id",
        "p_away_scores_top1", "p_home_scores_bot1", "p_nrfi", "p_yrfi",
        "pick", "confidence_tier",
    ]
    print(predictions_df[display_cols].to_string(index=False))
    predictions_df.to_csv(PROCESSED_DIR / f"predictions_{date}.csv", index=False)
    print(f"\nsaved to {PROCESSED_DIR / f'predictions_{date}.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    main(args.date)
