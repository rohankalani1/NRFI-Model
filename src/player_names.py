"""Small id -> full name lookup cache for pitchers/batters, via the MLB Stats
API, so the dashboard can show "Zac Gallen" instead of a raw player id.
Only fetches ids not already cached.
"""
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_PATH = RAW_DIR / "player_names.csv"

BATCH_SIZE = 100


def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame(columns=["id", "full_name"])


def _fetch_names(ids: list[int]) -> dict:
    names = {}
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/people",
            params={"personIds": ",".join(str(i) for i in batch)},
            timeout=30,
        )
        r.raise_for_status()
        for person in r.json().get("people", []):
            names[person["id"]] = person["fullName"]
    return names


def get_names(ids: list) -> dict:
    """Returns {id: full_name} for every id in `ids`, fetching only the ones
    not already cached on disk."""
    ids = [int(i) for i in ids if pd.notna(i)]
    if not ids:
        return {}

    cache = _load_cache()
    cached_ids = set(cache["id"])
    missing = sorted(set(ids) - cached_ids)

    if missing:
        fetched = _fetch_names(missing)
        if fetched:
            new_rows = pd.DataFrame(
                {"id": list(fetched.keys()), "full_name": list(fetched.values())}
            )
            cache = pd.concat([cache, new_rows], ignore_index=True)
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            cache.to_csv(CACHE_PATH, index=False)

    lookup = dict(zip(cache["id"], cache["full_name"]))
    return {i: lookup.get(i, f"Unknown ({i})") for i in ids}
