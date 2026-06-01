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
TEAM_NAME_MAP = {
    "USA":                       "United States",
    "Côte d'Ivoire":             "Ivory Coast",
    "Korea Republic":            "South Korea",
    "Bosnia-Herzegovina":        "Bosnia and Herzegovina",
    "Curaçao":                   "Curaçao",
    "Congo DR":                  "DR Congo",
    "Czech Republic/Czechia":    "Czech Republic",
}

# football-data.org stage → our round identifier
STAGE_TO_ROUND = {
    "GROUP_STAGE":    "group",
    "ROUND_OF_32":    "r32",
    "ROUND_OF_16":    "r16",
    "QUARTER_FINALS": "qf",
    "SEMI_FINALS":    "sf",
    "THIRD_PLACE":    "3rd",
    "FINAL":          "final",
}


def _normalize(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _fetch_wc_matches(api_key: str, status: str) -> list:
    resp = requests.get(
        "https://api.football-data.org/v4/competitions/WC/matches",
        headers={"X-Auth-Token": api_key},
        params={"status": status},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def _sync_fixtures(api_key: str, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fetch scheduled WC 2026 matches from the API and add any that aren't
    in the CSV yet (knockout fixtures that appear after the group stage).
    Returns (updated_df, number_of_new_rows_added).
    """
    try:
        scheduled = _fetch_wc_matches(api_key, "SCHEDULED")
        tbd = _fetch_wc_matches(api_key, "TIMED")
        all_upcoming = scheduled + tbd
    except Exception as e:
        print(f"  Could not fetch scheduled fixtures: {e}")
        return df, 0

    added = 0
    for m in all_upcoming:
        stage = m.get("stage", "")
        round_val = STAGE_TO_ROUND.get(stage)
        if not round_val or round_val == "group":
            continue  # group stage already in CSV

        home = _normalize(m["homeTeam"]["name"])
        away = _normalize(m["awayTeam"]["name"])
        date = m["utcDate"][:10]  # YYYY-MM-DD

        # Skip if TBD placeholders (team name missing or placeholder)
        if not home or not away or home == away:
            continue

        # Skip if already in CSV
        already = (
            df["home_team"].str.lower().eq(home.lower())
            & df["away_team"].str.lower().eq(away.lower())
            & df["date"].eq(date)
        )
        if already.any():
            continue

        new_row = {
            "date":        date,
            "home_team":   home,
            "away_team":   away,
            "home_score":  None,
            "away_score":  None,
            "tournament":  "FIFA World Cup",
            "city":        m.get("venue", {}).get("city", "") if isinstance(m.get("venue"), dict) else "",
            "country":     "USA/Canada/Mexico",
            "neutral":     True,
            "round":       round_val,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  Added {round_val.upper()} fixture: {home} vs {away} on {date}")
        added += 1

    return df, added


def update_results(api_key: str) -> dict:
    """
    1. Pull completed WC 2026 results and write scores into results.csv.
    2. Sync any new knockout fixtures that have been published.
    3. Recompute team stats.
    """
    print("Fetching completed WC 2026 matches...")
    try:
        finished = _fetch_wc_matches(api_key, "FINISHED")
    except requests.HTTPError as e:
        return {"error": f"API error: {e}", "updated": 0}
    except Exception as e:
        return {"error": str(e), "updated": 0}

    df = pd.read_csv(ROOT / "data" / "raw" / "results.csv")

    # Ensure round column exists
    if "round" not in df.columns:
        df["round"] = None
        mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
        df.loc[mask, "round"] = "group"

    updated, skipped, unmapped = 0, 0, []

    for m in finished:
        home = _normalize(m["homeTeam"]["name"])
        away = _normalize(m["awayTeam"]["name"])
        score = m.get("score", {}).get("fullTime", {})
        hs, as_ = score.get("home"), score.get("away")

        if hs is None or as_ is None:
            skipped += 1
            continue

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

    # Sync any new knockout fixtures
    print("Checking for new knockout fixtures...")
    df, new_fixtures = _sync_fixtures(api_key, df)

    if updated > 0 or new_fixtures > 0:
        df.to_csv(ROOT / "data" / "raw" / "results.csv", index=False)
        if updated > 0:
            print(f"  Wrote {updated} results. Recomputing features...")
            df_features, current_stats = compute_features(df)
            df_features.to_csv(ROOT / "data" / "processed" / "matches_with_features.csv", index=False)
            with open(ROOT / "data" / "processed" / "team_stats.json", "w") as f:
                json.dump(current_stats, f, indent=2)
            print("  Team stats updated.")
    else:
        print("  No new results or fixtures.")

    return {
        "updated":       updated,
        "new_fixtures":  new_fixtures,
        "skipped":       skipped,
        "total_from_api": len(finished),
        "unmapped":      unmapped,
    }


if __name__ == "__main__":
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        print("Set FOOTBALL_DATA_API_KEY env var")
        sys.exit(1)
    result = update_results(key)
    print(result)
