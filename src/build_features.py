"""Build the rolling features and final training table.

For every game, computes - using ONLY data from strictly earlier games -
each team's rolling OPS for whoever occupied batting-order slots 1-4, and each
starter's rolling 1st-inning ERA-equivalent stats. Games from the season's
first week are kept only as history (to seed these rolling stats); they are
excluded from the final training rows since they have no lookback themselves.
"""
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

ROLLING_WINDOW = 15  # games of history for batter/pitcher rolling stats
SEED_WEEKS = 1  # exclude this much history at the start of the EARLIEST season as training rows


def compute_player_ops_asof(player_game: pd.DataFrame) -> pd.DataFrame:
    """For each (player, game_date), compute that player's rolling OPS using
    only their own games strictly before game_date."""
    player_game = player_game.sort_values(["batter", "game_date"]).copy()

    def rolling_ops(group: pd.DataFrame) -> pd.DataFrame:
        shifted = group.shift(1)  # exclude current game itself
        roll = shifted.rolling(window=ROLLING_WINDOW, min_periods=1)
        ab = roll["ab"].sum()
        h = roll["h"].sum()
        tb = roll["tb"].sum()
        bb = roll["bb"].sum()
        hbp = roll["hbp"].sum()
        sf = roll["sf"].sum()

        obp_denom = (ab + bb + hbp + sf).replace(0, pd.NA)
        obp = (h + bb + hbp) / obp_denom
        slg = tb / ab.replace(0, pd.NA)
        ops = (obp.fillna(0) + slg.fillna(0))
        games_seen = shifted["ab"].expanding().count()

        out = group[["game_date", "game_pk"]].copy()
        out["batter"] = group.name
        out["rolling_ops"] = ops.values
        out["games_of_history"] = games_seen.values
        return out

    result = player_game.groupby("batter", group_keys=False).apply(rolling_ops)
    return result


def compute_pitcher_inning1_asof(pitcher_inn1: pd.DataFrame) -> pd.DataFrame:
    """Rolling 1st-inning stats for each pitcher, using only starts strictly
    before game_date. Produces a simple runs-allowed-proxy rate:
    (hits + walks + hbp allowed) per batter faced in the 1st, rolling."""
    pitcher_inn1 = pitcher_inn1.sort_values(["pitcher", "game_date"]).copy()

    def rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
        shifted = group.shift(1)
        roll = shifted.rolling(window=ROLLING_WINDOW, min_periods=1)
        bf = roll["batters_faced"].sum()
        baserunners = roll["hits_allowed"].sum() + roll["walks_allowed"].sum() + roll["hbp_allowed"].sum()

        rate = baserunners / bf.replace(0, pd.NA)
        starts_seen = shifted["batters_faced"].expanding().count()

        out = group[["game_date", "game_pk"]].copy()
        out["pitcher"] = group.name
        out["rolling_inning1_baserunner_rate"] = rate.fillna(rate.mean()).values
        out["starts_of_history"] = starts_seen.values
        return out

    result = pitcher_inn1.groupby("pitcher", group_keys=False).apply(rolling_stats)
    return result


def compute_pitcher_overall_asof(pitcher_full: pd.DataFrame) -> pd.DataFrame:
    """Rolling full-game (all innings) baserunner rate per pitcher, using only
    starts strictly before game_date. Much larger sample per start than the
    1st-inning-only version, used to stabilize that noisier feature."""
    pitcher_full = pitcher_full.sort_values(["pitcher", "game_date"]).copy()

    def rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
        shifted = group.shift(1)
        roll = shifted.rolling(window=ROLLING_WINDOW, min_periods=1)
        bf = roll["batters_faced"].sum()
        baserunners = roll["hits_allowed"].sum() + roll["walks_allowed"].sum() + roll["hbp_allowed"].sum()

        rate = baserunners / bf.replace(0, pd.NA)

        out = group[["game_date", "game_pk"]].copy()
        out["pitcher"] = group.name
        out["rolling_overall_baserunner_rate"] = rate.fillna(rate.mean()).values
        return out

    result = pitcher_full.groupby("pitcher", group_keys=False).apply(rolling_stats)
    return result


def compute_team_inning1_rate_asof(games: pd.DataFrame) -> pd.DataFrame:
    """Each team's own rolling rate of scoring in their half of the 1st inning,
    using only games strictly before game_date - pooled across their games as
    both the away team (top 1st) and home team (bottom 1st)."""
    away_hist = games[["game_pk", "game_date", "away_team", "away_scored_top1"]].rename(
        columns={"away_team": "team", "away_scored_top1": "scored"}
    )
    home_hist = games[["game_pk", "game_date", "home_team", "home_scored_bottom1"]].rename(
        columns={"home_team": "team", "home_scored_bottom1": "scored"}
    )
    team_hist = pd.concat([away_hist, home_hist], ignore_index=True).sort_values(["team", "game_date"])

    def rolling_rate(group: pd.DataFrame) -> pd.DataFrame:
        shifted = group["scored"].shift(1)
        rate = shifted.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

        out = group[["game_date", "game_pk"]].copy()
        out["team"] = group.name
        out["rolling_team_inning1_rate"] = rate.values
        return out

    result = team_hist.groupby("team", group_keys=False).apply(rolling_rate)
    return result


def main() -> None:
    games = pd.read_parquet(PROCESSED_DIR / "games.parquet")
    player_game = pd.read_parquet(PROCESSED_DIR / "player_game_batting.parquet")
    pitcher_inn1 = pd.read_parquet(PROCESSED_DIR / "pitcher_game_inning1.parquet")
    pitcher_full = pd.read_parquet(PROCESSED_DIR / "pitcher_game_full.parquet")
    park_factors = pd.read_csv(RAW_DIR / "park_factors_2026.csv")

    games["game_date"] = pd.to_datetime(games["game_date"])
    player_game["game_date"] = pd.to_datetime(player_game["game_date"])
    pitcher_inn1["game_date"] = pd.to_datetime(pitcher_inn1["game_date"])
    pitcher_full["game_date"] = pd.to_datetime(pitcher_full["game_date"])

    print("computing per-player rolling OPS (as-of each game)...")
    player_ops = compute_player_ops_asof(player_game)

    print("computing per-pitcher rolling 1st-inning rates (as-of each start)...")
    pitcher_rates = compute_pitcher_inning1_asof(pitcher_inn1)

    print("computing per-pitcher rolling overall (full-game) rates (as-of each start)...")
    pitcher_overall_rates = compute_pitcher_overall_asof(pitcher_full)

    print("computing per-team rolling 1st-inning-scored rates (as-of each game)...")
    team_rates = compute_team_inning1_rate_asof(games)

    ops_lookup = player_ops.set_index(["batter", "game_pk"])["rolling_ops"]
    pitcher_lookup = pitcher_rates.set_index(["pitcher", "game_pk"])["rolling_inning1_baserunner_rate"]
    pitcher_overall_lookup = pitcher_overall_rates.set_index(["pitcher", "game_pk"])["rolling_overall_baserunner_rate"]
    team_rate_lookup = team_rates.set_index(["team", "game_pk"])["rolling_team_inning1_rate"]

    def slot_avg_ops(row, batter_cols):
        vals = []
        for col in batter_cols:
            pid = row[col]
            if pd.isna(pid):
                continue
            key = (pid, row["game_pk"])
            if key in ops_lookup.index:
                vals.append(ops_lookup.loc[key])
        return sum(vals) / len(vals) if vals else None

    away_cols = ["away_top1_batter_1", "away_top1_batter_2", "away_top1_batter_3", "away_top1_batter_4"]
    home_cols = ["home_bot1_batter_1", "home_bot1_batter_2", "home_bot1_batter_3", "home_bot1_batter_4"]

    print("assembling feature table (this takes a minute)...")
    games["away_slot1_4_ops"] = games.apply(lambda r: slot_avg_ops(r, away_cols), axis=1)
    games["home_slot1_4_ops"] = games.apply(lambda r: slot_avg_ops(r, home_cols), axis=1)

    def lookup_rate(row, key_col, lookup):
        key = (row[key_col], row["game_pk"])
        return lookup.loc[key] if key in lookup.index else None

    games["home_starter_inning1_rate"] = games.apply(lambda r: lookup_rate(r, "home_starter", pitcher_lookup), axis=1)
    games["away_starter_inning1_rate"] = games.apply(lambda r: lookup_rate(r, "away_starter", pitcher_lookup), axis=1)
    games["home_starter_overall_rate"] = games.apply(lambda r: lookup_rate(r, "home_starter", pitcher_overall_lookup), axis=1)
    games["away_starter_overall_rate"] = games.apply(lambda r: lookup_rate(r, "away_starter", pitcher_overall_lookup), axis=1)

    games = games.merge(
        team_rates.rename(columns={"team": "away_team", "rolling_team_inning1_rate": "away_team_own_inning1_rate"}),
        on=["away_team", "game_pk", "game_date"], how="left",
    )
    games = games.merge(
        team_rates.rename(columns={"team": "home_team", "rolling_team_inning1_rate": "home_team_own_inning1_rate"}),
        on=["home_team", "game_pk", "game_date"], how="left",
    )

    games = games.merge(
        park_factors[["team_code", "park_factor"]].rename(columns={"team_code": "home_team"}),
        on="home_team", how="left",
    )

    # long format: one row per half-inning (batting team vs opposing starter)
    top_rows = games[[
        "game_pk", "game_date", "away_team", "home_team", "park_factor",
        "away_slot1_4_ops", "home_starter_inning1_rate", "home_starter_overall_rate",
        "away_team_own_inning1_rate", "away_scored_top1",
    ]].rename(columns={
        "away_team": "batting_team",
        "home_team": "opponent_team",
        "away_slot1_4_ops": "batting_team_slot1_4_ops",
        "home_starter_inning1_rate": "opposing_starter_inning1_rate",
        "home_starter_overall_rate": "opposing_starter_overall_rate",
        "away_team_own_inning1_rate": "batting_team_own_inning1_rate",
        "away_scored_top1": "scored",
    })
    top_rows["is_home_batting"] = 0

    bot_rows = games[[
        "game_pk", "game_date", "home_team", "away_team", "park_factor",
        "home_slot1_4_ops", "away_starter_inning1_rate", "away_starter_overall_rate",
        "home_team_own_inning1_rate", "home_scored_bottom1",
    ]].rename(columns={
        "home_team": "batting_team",
        "away_team": "opponent_team",
        "home_slot1_4_ops": "batting_team_slot1_4_ops",
        "away_starter_inning1_rate": "opposing_starter_inning1_rate",
        "away_starter_overall_rate": "opposing_starter_overall_rate",
        "home_team_own_inning1_rate": "batting_team_own_inning1_rate",
        "home_scored_bottom1": "scored",
    })
    bot_rows["is_home_batting"] = 1

    long_df = pd.concat([top_rows, bot_rows], ignore_index=True).sort_values("game_date")

    long_df.to_parquet(PROCESSED_DIR / "model_dataset_full.parquet", index=False)

    train_cutoff = long_df["game_date"].min() + pd.Timedelta(weeks=SEED_WEEKS)
    train_df = long_df[long_df["game_date"] >= train_cutoff].dropna(
        subset=[
            "batting_team_slot1_4_ops", "opposing_starter_inning1_rate",
            "opposing_starter_overall_rate", "batting_team_own_inning1_rate", "park_factor",
        ]
    )
    train_df.to_parquet(PROCESSED_DIR / "model_dataset_train.parquet", index=False)

    print(f"full long-format dataset: {len(long_df):,} rows ({games['game_pk'].nunique():,} games)")
    print(f"training dataset (>= {train_cutoff.date()}, complete features): {len(train_df):,} rows")
    print(f"  scored rate in training set: {train_df['scored'].mean():.3f}")
    print(f"  missing feature rows dropped: {len(long_df[long_df['game_date'] >= train_cutoff]) - len(train_df):,}")


if __name__ == "__main__":
    main()
