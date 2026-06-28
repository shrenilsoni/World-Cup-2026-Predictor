import json
import os
import sys
import difflib
import unicodedata
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
    "Czechia":                   "Czech Republic",
    "Cape Verde Islands":        "Cape Verde",
    "Türkiye":                   "Turkey",
}

# football-data.org stage → our round identifier.
# The API uses LAST_32 / LAST_16 for the first two knockout rounds (not
# ROUND_OF_32 / ROUND_OF_16); both spellings are mapped so a future API change
# either way keeps working.
STAGE_TO_ROUND = {
    "GROUP_STAGE":    "group",
    "LAST_32":        "r32",
    "ROUND_OF_32":    "r32",
    "LAST_16":        "r16",
    "ROUND_OF_16":    "r16",
    "QUARTER_FINALS": "qf",
    "SEMI_FINALS":    "sf",
    "THIRD_PLACE":    "3rd",
    "FINAL":          "final",
}


def _redact(msg: str, secret: str) -> str:
    """Strip a secret out of an error string before it's surfaced to clients."""
    text = str(msg)
    secret = (secret or "").strip()
    if secret and secret in text:
        text = text.replace(secret, "***REDACTED***")
    return text


def _normalize(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


def _strip_accents(s: str) -> str:
    """Lowercase + drop diacritics so 'Curaçao' == 'curacao'."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    ).lower().strip()


def _resolve_team(name: str, known_teams: list[str]) -> tuple[str, bool]:
    """
    Resolve a football-data.org team name to the canonical CSV name.

    Strategy (most reliable first):
      1. Explicit TEAM_NAME_MAP override.
      2. Exact case-insensitive match against the known WC team pool.
      3. Accent-insensitive match (handles diacritic differences).
      4. Fuzzy match for minor spelling variants (strict cutoff).

    Returns (resolved_name, matched) where ``matched`` is False if no entry in
    the known pool could be resolved — the caller should treat that as an
    unmapped failure rather than silently dropping the result.
    """
    mapped = _normalize(name)

    by_lower = {t.lower(): t for t in known_teams}
    if mapped.lower() in by_lower:
        return by_lower[mapped.lower()], True

    by_accent = {_strip_accents(t): t for t in known_teams}
    key = _strip_accents(mapped)
    if key in by_accent:
        return by_accent[key], True

    close = difflib.get_close_matches(key, list(by_accent.keys()), n=1, cutoff=0.85)
    if close:
        return by_accent[close[0]], True

    return mapped, False


def _upsert_to_supabase(home_team: str, away_team: str, match_date: str, home_score: int, away_score: int) -> None:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not url or not key:
        return
    try:
        from supabase import create_client
        sb = create_client(url, key)
        sb.table("match_results").upsert({
            "match_date":  match_date,
            "home_team":   home_team,
            "away_team":   away_team,
            "home_score":  home_score,
            "away_score":  away_score,
        }, on_conflict="match_date,home_team,away_team").execute()
        print(f"  Supabase: persisted {home_team} {home_score}–{away_score} {away_team}")
    except Exception as e:
        print(f"  Supabase upsert warning: {e}")


def _fetch_wc_matches(api_key: str, status: str | None = None) -> list:
    """Fetch WC matches. Omit `status` to get every match in a single call
    (the free tier is rate-limited, so we prefer one call over several)."""
    params = {"status": status} if status else {}
    resp = requests.get(
        "https://api.football-data.org/v4/competitions/WC/matches",
        headers={"X-Auth-Token": api_key.strip()},
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def _sync_fixtures(api_matches: list, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Add any knockout fixtures from the API that aren't in the CSV yet
    (they appear after the group stage). `api_matches` is the prefetched
    full match list. Returns (updated_df, number_of_new_rows_added).
    """
    added = 0
    for m in api_matches:
        stage = m.get("stage", "")
        round_val = STAGE_TO_ROUND.get(stage)
        if not round_val or round_val == "group":
            continue  # group stage already in CSV

        raw_home = m.get("homeTeam", {}).get("name")
        raw_away = m.get("awayTeam", {}).get("name")
        if not raw_home or not raw_away:
            continue  # TBD knockout slot — teams not determined yet
        home = _normalize(raw_home)
        away = _normalize(raw_away)
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
            "kickoff":     m.get("utcDate", ""),  # full ISO UTC timestamp
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  Added {round_val.upper()} fixture: {home} vs {away} on {date}")
        added += 1

    return df, added


def _sync_kickoffs(api_matches: list, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Backfill the per-match `kickoff` (full ISO UTC timestamp) for WC 2026
    fixtures from the prefetched API match list, so each match can lock at
    its own kickoff time. Matches by resolved team names (date-agnostic).
    Returns (df, n_filled).
    """
    if "kickoff" not in df.columns:
        df["kickoff"] = ""

    wc_mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
    known_teams = sorted(
        set(df.loc[wc_mask, "home_team"].tolist() + df.loc[wc_mask, "away_team"].tolist())
    )

    filled = 0
    for m in api_matches:
        kickoff = m.get("utcDate", "")
        raw_home = m.get("homeTeam", {}).get("name")
        raw_away = m.get("awayTeam", {}).get("name")
        if not kickoff or not raw_home or not raw_away:
            continue  # skip TBD knockout fixtures with undetermined teams
        home, _ = _resolve_team(raw_home, known_teams)
        away, _ = _resolve_team(raw_away, known_teams)
        # Order-insensitive: the CSV's home/away can be the reverse of the API's.
        row_mask = wc_mask & (
            (df["home_team"].str.lower().eq(home.lower())
             & df["away_team"].str.lower().eq(away.lower()))
            | (df["home_team"].str.lower().eq(away.lower())
               & df["away_team"].str.lower().eq(home.lower()))
        )
        # Only fill rows whose kickoff is currently missing or different.
        needs = row_mask & (df["kickoff"].isna() | (df["kickoff"].astype(str).str.strip() != kickoff))
        if needs.any():
            df.loc[row_mask, "kickoff"] = kickoff
            filled += int(needs.sum())

    return df, filled


def update_results(api_key: str) -> dict:
    """
    1. Pull completed WC 2026 results and write scores into results.csv.
    2. Sync any new knockout fixtures that have been published.
    3. Recompute team stats.
    """
    print("Fetching completed WC 2026 matches...")
    try:
        # One call for every match (rate-limit friendly); derive the rest.
        all_matches = _fetch_wc_matches(api_key)
    except requests.HTTPError as e:
        return {"error": _redact(f"API error: {e}", api_key), "updated": 0}
    except Exception as e:
        return {"error": _redact(e, api_key), "updated": 0}

    finished = [m for m in all_matches if m.get("status") == "FINISHED"]

    df = pd.read_csv(ROOT / "data" / "raw" / "results.csv")

    # Ensure round column exists
    if "round" not in df.columns:
        df["round"] = None
        mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
        df.loc[mask, "round"] = "group"

    # Pool of canonical WC 2026 team names to resolve API names against.
    wc_mask = (df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")
    known_teams = sorted(
        set(df.loc[wc_mask, "home_team"].tolist() + df.loc[wc_mask, "away_team"].tolist())
    )

    updated, skipped, already = 0, 0, 0
    unmapped = []

    for m in finished:
        raw_home = m["homeTeam"]["name"]
        raw_away = m["awayTeam"]["name"]
        home, home_ok = _resolve_team(raw_home, known_teams)
        away, away_ok = _resolve_team(raw_away, known_teams)
        score = m.get("score", {}).get("fullTime", {})
        hs, as_ = score.get("home"), score.get("away")

        if hs is None or as_ is None:
            skipped += 1
            continue

        # Match the WC 2026 fixture by resolved team names (date is ignored:
        # the API's UTC date can be a day off from the CSV's local date).
        # The CSV's home/away designation for neutral-site games can be the
        # reverse of the API's, so check both orientations and flip the scores
        # to match the CSV row's orientation.
        fwd = (
            df["home_team"].str.lower().eq(home.lower())
            & df["away_team"].str.lower().eq(away.lower())
        )
        rev = (
            df["home_team"].str.lower().eq(away.lower())
            & df["away_team"].str.lower().eq(home.lower())
        )
        team_mask = wc_mask & (fwd | rev)
        open_mask = team_mask & df["home_score"].isna()

        if open_mask.any():
            open_fwd = open_mask & fwd
            open_rev = open_mask & rev
            # Forward rows: scores as-is. Reverse rows: swap home/away scores.
            df.loc[open_fwd, "home_score"] = float(hs)
            df.loc[open_fwd, "away_score"] = float(as_)
            df.loc[open_rev, "home_score"] = float(as_)
            df.loc[open_rev, "away_score"] = float(hs)
            match_date = m["utcDate"][:10]
            # Persist each matched row in its own CSV orientation.
            for _, r in df.loc[open_mask].iterrows():
                _upsert_to_supabase(
                    r["home_team"], r["away_team"], match_date,
                    int(r["home_score"]), int(r["away_score"]),
                )
            updated += 1
        elif team_mask.any():
            already += 1  # result already recorded — benign, not a failure
        else:
            # Could not line this finished match up with any WC 2026 fixture.
            # Surface it loudly so a name/order mismatch never fails silently.
            detail = f"{raw_home} vs {raw_away} ({hs}-{as_})"
            if not (home_ok and away_ok):
                detail += " [unresolved name]"
            print(f"  ⚠️  UNMAPPED finished match — not recorded: {detail}")
            unmapped.append(detail)

    # Sync any new knockout fixtures
    print("Checking for new knockout fixtures...")
    df, new_fixtures = _sync_fixtures(all_matches, df)

    # Backfill per-match kickoff times so matches lock at their own start.
    print("Syncing kickoff times...")
    df, kickoffs_filled = _sync_kickoffs(all_matches, df)

    if updated > 0 or new_fixtures > 0 or kickoffs_filled > 0:
        df.to_csv(ROOT / "data" / "raw" / "results.csv", index=False)
        if kickoffs_filled > 0:
            print(f"  Filled {kickoffs_filled} kickoff time(s).")
        if updated > 0:
            print(f"  Wrote {updated} results. Recomputing features...")
            df_features, current_stats = compute_features(df)
            df_features.to_csv(ROOT / "data" / "processed" / "matches_with_features.csv", index=False)
            with open(ROOT / "data" / "processed" / "team_stats.json", "w") as f:
                json.dump(current_stats, f, indent=2)
            print("  Team stats updated.")
    else:
        print("  No new results or fixtures.")

    if unmapped:
        print(f"  ⚠️  {len(unmapped)} finished match(es) could not be mapped to a fixture.")

    return {
        "updated":         updated,
        "new_fixtures":    new_fixtures,
        "kickoffs_filled": kickoffs_filled,
        "skipped":         skipped,
        "already":         already,
        "total_from_api":  len(finished),
        "unmapped":        unmapped,
    }


if __name__ == "__main__":
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        print("Set FOOTBALL_DATA_API_KEY env var")
        sys.exit(1)
    result = update_results(key)
    print(result)
