"""Pull and cache raw Statcast pitch-by-pitch data for a given MLB season.

Chunks the pull by month so a single failure doesn't lose the whole season,
and skips months that are already cached on disk.
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def season_date_range(season: int) -> tuple[dt.date, dt.date]:
    r = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "season": season, "gameType": "R",
                "fields": "dates,date,games,gamePk"},
        timeout=30,
    )
    data = r.json()
    dates = [dt.date.fromisoformat(d["date"]) for d in data["dates"]]
    return min(dates), max(dates)


def month_chunks(start: dt.date, end: dt.date):
    chunk_start = start
    while chunk_start <= end:
        next_month = (chunk_start.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month - dt.timedelta(days=1), end)
        yield chunk_start, chunk_end
        chunk_start = next_month


def fetch_season(season: int, end_date: dt.date | None = None) -> None:
    season_start, season_end = season_date_range(season)
    if end_date is None:
        end_date = min(season_end, dt.date.today() - dt.timedelta(days=1))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"season {season}: {season_start} - {season_end} (pulling through {end_date})")

    for chunk_start, chunk_end in month_chunks(season_start, end_date):
        out_path = RAW_DIR / f"statcast_{chunk_start:%Y_%m}.parquet"

        if out_path.exists():
            existing = pd.read_parquet(out_path)
            cached_max = pd.to_datetime(existing["game_date"]).max().date()
            if cached_max >= chunk_end:
                print(f"skip {chunk_start} - {chunk_end} (cached through {cached_max} at {out_path.name})")
                continue

            fetch_start = cached_max + dt.timedelta(days=1)
            print(f"appending {fetch_start} - {chunk_end} to {out_path.name} (cached through {cached_max}) ...")
            new_df = statcast(start_dt=str(fetch_start), end_dt=str(chunk_end))
            if new_df is None or new_df.empty:
                print(f"  no new data returned for {fetch_start} - {chunk_end}")
                continue

            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
            combined.to_parquet(out_path, index=False)
            print(f"  appended {len(new_df):,} new rows -> {out_path.name} ({len(combined):,} total)")
            continue

        print(f"fetching {chunk_start} - {chunk_end} ...")
        df = statcast(start_dt=str(chunk_start), end_dt=str(chunk_end))
        if df is None or df.empty:
            print(f"  no data returned for {chunk_start} - {chunk_end}")
            continue

        df.to_parquet(out_path, index=False)
        print(f"  saved {len(df):,} rows -> {out_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    args = parser.parse_args()
    fetch_season(args.season)
