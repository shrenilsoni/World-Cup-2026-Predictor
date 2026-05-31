import json
import os
import sys
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import compute_features

# football-data.org names → our dataset names
# Extend this if you see "unmapped" teams in the update log
TEAM_NAME_MAP = {
    "USA":                       "United States",
    "Côte d'Ivoire":             "Ivory Coast",
    "Korea Republic":            "South Korea",
    "Bosnia-Herzegovina":        "Bosnia and Herzegovina",
    "Curaçao":                   "Curaçao",
    "Congo DR":                  "DR Congo",
    "Czech Republic/Czechia":    "Czech Republic",
}


def _normalize(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def fetch_completed_matches(api_key: str) -> list:
    url = "https://api.football-data.org/v4/competitions/WC/matches"
    resp = requests.get(
        url,
        headers={"X-Auth-Token": api_key},
        params={"status": "FINISHED"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def update_results(api_key: str) -> dict:
    """
    Pull completed WC 2026 results from football-data.org,
    write scores back into results.csv, recompute team stats.
    Returns a summary dict.
    """
    print("Fetching completed WC 2026 matches from football-data.org...")
    try:
        api_matches = fetch_completed_matches(api_key)
    except requests.HTTPError as e:
        return {"error": f"API error: {e}", "updated": 0}
    except Exception as e:
        return {"error": str(e), "updated": 0}

    df = pd.read_csv(ROOT / "data" / "raw" / "results.csv")
    updated, skipped, unmapped = 0, 0, []

    for m in api_matches:
        home = _normalize(m["homeTeam"]["name"])
        away = _normalize(m["awayTeam"]["name"])
        score = m.get("score", {}).get("fullTime", {})
        hs, as_ = score.get("home"), score.get("away")

        if hs is None or as_ is None:
            skipped += 1
            continue

        # Case-insensitive match on rows that still have NaN scores
        mask = (
            df["home_team"].str.lower().eq(home.lower())
            & df["away_team"].str.lower().eq(away.lower())
            & df["home_score"].isna()
        )

        if mask.any():
            df.loc[mask, "home_score"] = float(hs)
            df.loc[mask, "away_score"] = float(as_)
            updated += 1
        else:
            unmapped.append(f"{home} vs {away}")

    if updated > 0:
        df.to_csv(ROOT / "data" / "raw" / "results.csv", index=False)
        print(f"  Wrote {updated} results. Recomputing features...")
        df_features, current_stats = compute_features(df)
        df_features.to_csv(ROOT / "data" / "processed" / "matches_with_features.csv", index=False)
        with open(ROOT / "data" / "processed" / "team_stats.json", "w") as f:
            json.dump(current_stats, f, indent=2)
        print("  Team stats updated.")
    else:
        print("  No new results to update.")

    return {
        "updated": updated,
        "skipped": skipped,
        "total_from_api": len(api_matches),
        "unmapped": unmapped,
    }


if __name__ == "__main__":
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        print("Set FOOTBALL_DATA_API_KEY env var")
        sys.exit(1)
    result = update_results(key)
    print(result)
