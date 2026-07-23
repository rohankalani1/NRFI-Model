"""Derive per-game and per-player-game tables from raw Statcast pitch data.

Produces three tables in data/processed/:
  - player_game_batting.parquet : one row per (player, game) with AB/H/BB/HBP/SF/TB
    for that game, used later to build rolling OPS features.
  - pitcher_game_inning1.parquet: one row per (pitcher, game) with that pitcher's
    own 1st-inning line (batters faced, hits, walks, runs allowed).
  - games.parquet               : one row per game with home/away teams, starters,
    slot 1-4 batter ids for each half of the 1st inning, and the two target labels
    (away_scored_top1, home_scored_bottom1).
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

HIT_EVENTS = {"single", "double", "triple", "home_run"}
TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALK_EVENTS = {"walk", "intent_walk"}
HBP_EVENTS = {"hit_by_pitch"}
SAC_FLY_EVENTS = {"sac_fly", "sac_fly_double_play"}
NOT_AT_BAT_EVENTS = WALK_EVENTS | HBP_EVENTS | SAC_FLY_EVENTS | {
    "sac_bunt", "sac_bunt_double_play", "catcher_interf",
}


def load_statcast() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("statcast_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)
    return df


def plate_appearances(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse pitch-level rows to one row per plate appearance (last pitch of each PA)."""
    pa = df[df["events"].notna()].copy()
    pa = pa.sort_values(["game_pk", "at_bat_number"]).reset_index(drop=True)
    return pa


def build_player_game_batting(pa: pd.DataFrame) -> pd.DataFrame:
    pa = pa.copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["total_bases"] = pa["events"].map(TOTAL_BASES).fillna(0).astype(int)
    pa["is_bb"] = pa["events"].isin(WALK_EVENTS).astype(int)
    pa["is_hbp"] = pa["events"].isin(HBP_EVENTS).astype(int)
    pa["is_sf"] = pa["events"].isin(SAC_FLY_EVENTS).astype(int)
    pa["is_ab"] = (~pa["events"].isin(NOT_AT_BAT_EVENTS)).astype(int)

    grouped = (
        pa.groupby(["game_pk", "game_date", "batter"], as_index=False)
        .agg(
            ab=("is_ab", "sum"),
            h=("is_hit", "sum"),
            tb=("total_bases", "sum"),
            bb=("is_bb", "sum"),
            hbp=("is_hbp", "sum"),
            sf=("is_sf", "sum"),
        )
    )
    return grouped


def build_pitcher_game_full(pa: pd.DataFrame) -> pd.DataFrame:
    """Each pitcher's full-game (all innings) line - a much larger sample per
    start than the 1st-inning-only line, used to stabilize the noisy
    1st-inning-specific rolling rate via blending."""
    pa = pa.copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_bb"] = pa["events"].isin(WALK_EVENTS).astype(int)
    pa["is_hbp"] = pa["events"].isin(HBP_EVENTS).astype(int)

    grouped = (
        pa.groupby(["game_pk", "game_date", "pitcher"], as_index=False)
        .agg(
            batters_faced=("events", "count"),
            hits_allowed=("is_hit", "sum"),
            walks_allowed=("is_bb", "sum"),
            hbp_allowed=("is_hbp", "sum"),
        )
    )
    return grouped


def build_pitcher_game_inning1(df: pd.DataFrame) -> pd.DataFrame:
    inn1 = df[df["inning"] == 1]
    pa = plate_appearances(inn1)
    pa = pa.copy()
    pa["is_hit"] = pa["events"].isin(HIT_EVENTS).astype(int)
    pa["is_bb"] = pa["events"].isin(WALK_EVENTS).astype(int)
    pa["is_hbp"] = pa["events"].isin(HBP_EVENTS).astype(int)

    grouped = (
        pa.groupby(["game_pk", "game_date", "pitcher"], as_index=False)
        .agg(
            batters_faced=("events", "count"),
            hits_allowed=("is_hit", "sum"),
            walks_allowed=("is_bb", "sum"),
            hbp_allowed=("is_hbp", "sum"),
        )
    )
    return grouped


def build_games_table(df: pd.DataFrame) -> pd.DataFrame:
    inn1 = df[df["inning"] == 1]
    rows = []
    for game_pk, g in inn1.groupby("game_pk"):
        top = g[g["inning_topbot"] == "Top"].sort_values(["at_bat_number", "pitch_number"])
        bot = g[g["inning_topbot"] == "Bot"].sort_values(["at_bat_number", "pitch_number"])
        if top.empty:
            continue

        home_team = g["home_team"].iloc[0]
        away_team = g["away_team"].iloc[0]
        game_date = g["game_date"].iloc[0]

        home_starter = top["pitcher"].iloc[0]
        away_top1_batters = list(dict.fromkeys(top["batter"].tolist()))[:4]
        away_scored_top1 = int(top["post_away_score"].max() > 0) if len(top) else 0

        if not bot.empty:
            away_starter = bot["pitcher"].iloc[0]
            home_bottom1_batters = list(dict.fromkeys(bot["batter"].tolist()))[:4]
            home_scored_bottom1 = int(bot["post_home_score"].max() > 0)
        else:
            # visiting team made 3 outs and home team never batted in the bottom
            # of the 1st in this slice - shouldn't happen for inning 1, but guard anyway
            away_starter = None
            home_bottom1_batters = []
            home_scored_bottom1 = 0

        rows.append({
            "game_pk": game_pk,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
            "home_starter": home_starter,
            "away_starter": away_starter,
            "away_top1_batter_1": away_top1_batters[0] if len(away_top1_batters) > 0 else None,
            "away_top1_batter_2": away_top1_batters[1] if len(away_top1_batters) > 1 else None,
            "away_top1_batter_3": away_top1_batters[2] if len(away_top1_batters) > 2 else None,
            "away_top1_batter_4": away_top1_batters[3] if len(away_top1_batters) > 3 else None,
            "home_bot1_batter_1": home_bottom1_batters[0] if len(home_bottom1_batters) > 0 else None,
            "home_bot1_batter_2": home_bottom1_batters[1] if len(home_bottom1_batters) > 1 else None,
            "home_bot1_batter_3": home_bottom1_batters[2] if len(home_bottom1_batters) > 2 else None,
            "home_bot1_batter_4": home_bottom1_batters[3] if len(home_bottom1_batters) > 3 else None,
            "away_scored_top1": away_scored_top1,
            "home_scored_bottom1": home_scored_bottom1,
        })

    return pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("loading raw statcast data...")
    df = load_statcast()
    print(f"  {len(df):,} pitch rows, {df['game_pk'].nunique():,} games")

    print("building player-game batting logs...")
    pa_all = plate_appearances(df)
    player_game = build_player_game_batting(pa_all)
    player_game.to_parquet(PROCESSED_DIR / "player_game_batting.parquet", index=False)
    print(f"  {len(player_game):,} player-game rows")

    print("building pitcher 1st-inning game logs...")
    pitcher_inn1 = build_pitcher_game_inning1(df)
    pitcher_inn1.to_parquet(PROCESSED_DIR / "pitcher_game_inning1.parquet", index=False)
    print(f"  {len(pitcher_inn1):,} pitcher-game rows")

    print("building pitcher full-game logs...")
    pitcher_full = build_pitcher_game_full(pa_all)
    pitcher_full.to_parquet(PROCESSED_DIR / "pitcher_game_full.parquet", index=False)
    print(f"  {len(pitcher_full):,} pitcher-game rows")

    print("building game-level table (starters, slots 1-4, labels)...")
    games = build_games_table(df)
    games.to_parquet(PROCESSED_DIR / "games.parquet", index=False)
    print(f"  {len(games):,} games")
    print(f"  away_scored_top1 rate: {games['away_scored_top1'].mean():.3f}")
    print(f"  home_scored_bottom1 rate: {games['home_scored_bottom1'].mean():.3f}")


if __name__ == "__main__":
    main()
