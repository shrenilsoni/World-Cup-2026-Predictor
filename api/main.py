import sys
import os
import threading
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.append(str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import json
import datetime
from zoneinfo import ZoneInfo
import pandas as pd

_PDT = ZoneInfo('America/Los_Angeles')

# Group-winner predictions close ~7 days before the final group-stage match
# (last group match is 2026-06-27), giving everyone the same early deadline.
_GROUP_WINNER_LOCK = datetime.datetime(2026, 6, 20, 23, 59, tzinfo=_PDT)
_GROUP_WINNER_POINTS = 70

# ---------------------------------------------------------------------------
# Supabase client (service role — server-side only)
# ---------------------------------------------------------------------------

# .strip() guards against trailing newlines/spaces in dashboard-set env vars,
# which otherwise produce invalid HTTP header values.
_SB_URL = os.environ.get("SUPABASE_URL", "").strip()
_SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
_sb = None
if _SB_URL and _SB_KEY:
    try:
        from supabase import create_client
        _sb = create_client(_SB_URL, _SB_KEY)
    except Exception as _e:
        print(f"Supabase init warning: {_e}")


def _fetch_supabase_results() -> list:
    """Return all WC 2026 results stored in Supabase as a list of dicts."""
    if not _sb:
        return []
    try:
        resp = _sb.table("match_results").select("*").execute()
        return resp.data or []
    except Exception as e:
        print(f"Supabase fetch warning: {e}")
        return []


def _apply_supabase_results() -> None:
    """
    On startup: pull any results stored in Supabase, write them into the
    local results.csv (which starts from the git snapshot), and recompute
    team_stats so the simulation is correct after a dyno restart.
    """
    global team_stats, fixtures
    rows = _fetch_supabase_results()
    if not rows:
        print("  Supabase: no persisted results found.")
        return

    from features import compute_features
    csv_path = Path(__file__).parent.parent / "data" / "raw" / "results.csv"
    df = pd.read_csv(csv_path)

    updated = 0
    for r in rows:
        # Match on teams + unplayed only (not exact date): football-data's UTC
        # date can differ from the CSV's local date by a day, and the live
        # update writes scores by team name alone, so we mirror that here.
        mask = (
            df["home_team"].str.lower().eq(r["home_team"].lower())
            & df["away_team"].str.lower().eq(r["away_team"].lower())
            & df["home_score"].isna()
        )
        if mask.any():
            df.loc[mask, "home_score"] = float(r["home_score"])
            df.loc[mask, "away_score"] = float(r["away_score"])
            updated += 1

    if updated:
        df.to_csv(csv_path, index=False)
        df_feat, stats = compute_features(df)
        df_feat.to_csv(Path(__file__).parent.parent / "data" / "processed" / "matches_with_features.csv", index=False)
        with open(Path(__file__).parent.parent / "data" / "processed" / "team_stats.json", "w") as fh:
            json.dump(stats, fh, indent=2)
        team_stats = stats
        fixtures = df[df["home_score"].isna()].copy()
        print(f"  Supabase: applied {updated} persisted result(s) on startup.")
    else:
        print(f"  Supabase: {len(rows)} result(s) already in CSV, nothing to apply.")


from predict import (
    load_artifacts, extract_groups, predict_match, predict_goals,
    run_monte_carlo, build_group_fixture_lists,
    _assign_thirds_to_slots
)
from live_update import update_results


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    print("Applying persisted results from Supabase…")
    await asyncio.to_thread(_apply_supabase_results)
    print("Warming up simulation cache at startup…")
    await asyncio.to_thread(_get_sim)
    print("Simulation cache ready.")
    yield


app = FastAPI(title="WC 2026 Predictor", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Startup — load everything once
# ---------------------------------------------------------------------------

def _load_all_wc_fixtures():
    """All WC 2026 group stage rows, played or not — used for group extraction and team list."""
    raw = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "results.csv")
    return raw[(raw["tournament"] == "FIFA World Cup") & (raw["date"] >= "2026-01-01")].copy()


artifacts, team_stats, fixtures, goal_model_h, goal_model_a, goal_feat_h, goal_feat_a = load_artifacts()
model        = artifacts["model"]
feature_cols = artifacts["features"]
classes      = artifacts["classes"]
_all_wc      = _load_all_wc_fixtures()
groups       = extract_groups(_all_wc)
wc_teams     = sorted(set(_all_wc["home_team"].tolist() + _all_wc["away_team"].tolist()))

_sim_cache = None  # invalidated on live update
_sim_lock = threading.Lock()


def _get_sim():
    global _sim_cache
    if _sim_cache is not None:
        return _sim_cache
    with _sim_lock:
        if _sim_cache is None:  # double-checked locking
            print("Running Monte Carlo simulation...")
            _sim_cache = run_monte_carlo(
                fixtures, groups, model, feature_cols, classes, team_stats, n=10000
            )
    return _sim_cache


def _invalidate_cache():
    """Called after a live update rewrites team_stats."""
    global _sim_cache, team_stats, fixtures, groups, wc_teams
    global goal_model_h, goal_model_a, goal_feat_h, goal_feat_a
    _sim_cache = None
    # Reload team stats and fixtures from disk
    with open(Path(__file__).parent.parent / "data" / "processed" / "team_stats.json") as f:
        team_stats = json.load(f)
    import pandas as pd
    raw = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "results.csv")
    fixtures = raw[raw["home_score"].isna()].copy()
    _all_wc  = _load_all_wc_fixtures()
    groups   = extract_groups(_all_wc)
    wc_teams = sorted(set(_all_wc["home_team"].tolist() + _all_wc["away_team"].tolist()))
    # Reload goal models from pkl
    new_artifacts, _, _, gm_h, gm_a, gf_h, gf_a = load_artifacts()
    goal_model_h = gm_h
    goal_model_a = gm_a
    goal_feat_h  = gf_h
    goal_feat_a  = gf_a


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    neutral: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/teams")
def get_teams():
    teams_with_elo = [
        {"name": t, "elo": round(team_stats.get(t, {}).get("elo", 1500), 1)}
        for t in wc_teams
    ]
    teams_with_elo.sort(key=lambda x: x["elo"], reverse=True)
    return {"teams": teams_with_elo}


@app.post("/api/predict")
def predict(req: MatchRequest):
    probs = predict_match(
        req.home_team, req.away_team, req.neutral,
        model, feature_cols, classes, team_stats
    )

    goal_data = None
    if goal_model_h and goal_model_a:
        goal_data = predict_goals(
            req.home_team, req.away_team, req.neutral,
            goal_model_h, goal_model_a, goal_feat_h, goal_feat_a, team_stats
        )

    # Use Poisson-derived win/draw/loss in probabilities if available
    if goal_data:
        outcome_probs = {
            "home_win": goal_data["home_win_pct"],
            "draw":     goal_data["draw_pct"],
            "away_win": goal_data["away_win_pct"],
        }
    else:
        outcome_probs = {k: round(v * 100, 1) for k, v in probs.items()}

    return {
        "home_team": req.home_team,
        "away_team": req.away_team,
        "neutral": req.neutral,
        "home_elo": round(team_stats.get(req.home_team, {}).get("elo", 1500), 1),
        "away_elo": round(team_stats.get(req.away_team, {}).get("elo", 1500), 1),
        "probabilities": outcome_probs,
        "goals": goal_data,
    }


@app.get("/api/simulate")
def simulate():
    results = _get_sim()
    return {"simulations": 10000, "results": results}


@app.get("/api/matches")
def get_matches():
    """
    All WC 2026 matches with results and lock status.
    is_locked = match date has passed OR result already in system.
    Used by the My Picks tab to render pick forms and show scores.
    """
    today    = datetime.date.today().isoformat()
    now_pdt  = datetime.datetime.now(_PDT)
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "results.csv")
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-01-01")]
    wc = wc.sort_values("date").reset_index(drop=True)

    result = []
    for idx, row in wc.iterrows():
        has_result = pd.notna(row.get("home_score"))
        round_val = str(row["round"]) if pd.notna(row.get("round")) else "group"
        match_date = datetime.date.fromisoformat(str(row["date"]))
        # Lock each match at its own kickoff (full UTC timestamp from the API).
        # Fall back to 11:59 PM PDT the night before if no kickoff is known yet.
        kickoff = row.get("kickoff")
        if pd.notna(kickoff) and str(kickoff).strip():
            lock_dt = datetime.datetime.fromisoformat(
                str(kickoff).strip().replace("Z", "+00:00")
            )
        else:
            lock_dt = datetime.datetime.combine(
                match_date - datetime.timedelta(days=1), datetime.time(23, 59),
                tzinfo=_PDT
            )
        result.append({
            "match_id":   f"{row['date']}_{row['home_team']}_{row['away_team']}",
            "match_index": int(idx),
            "date":       str(row["date"]),
            "home_team":  row["home_team"],
            "away_team":  row["away_team"],
            "home_score": int(row["home_score"]) if has_result else None,
            "away_score": int(row["away_score"]) if has_result else None,
            "is_locked":  bool(has_result or now_pdt >= lock_dt),
            "lock_at":    lock_dt.astimezone(datetime.timezone.utc).isoformat(),
            "neutral":    bool(row["neutral"]),
            "city":       row.get("city", ""),
            "round":      round_val,
        })
    return {"matches": result, "today": today}


@app.get("/api/bracket")
def get_bracket():
    """
    Build the predicted bracket using official FIFA WC 2026 R32 structure.
    Uses simulation probabilities to pick the most likely team per slot
    and to compute head-to-head win odds at each match.
    """
    sim    = _get_sim()
    by_team = {r['team']: r for r in sim}

    A,B,C,D,E,F,G,H,I,J,K,L = range(12)

    # Predict each group's finishing order — each team appears exactly once
    firsts, seconds, thirds_candidates = [], [], []
    for g_idx, group in enumerate(groups):
        ranked = sorted(group, key=lambda t: by_team.get(t, {}).get('group_1st_pct', 0), reverse=True)
        firsts.append(ranked[0])
        rest = [t for t in group if t != ranked[0]]
        second = max(rest, key=lambda t: by_team.get(t, {}).get('group_2nd_pct', 0))
        seconds.append(second)
        rest2 = [t for t in rest if t != second]
        third = max(rest2, key=lambda t: by_team.get(t, {}).get('group_3rd_pct', 0))
        thirds_candidates.append((third, g_idx))

    thirds_candidates.sort(key=lambda x: by_team.get(x[0], {}).get('group_3rd_pct', 0), reverse=True)
    best_thirds = thirds_candidates[:8]
    slot_assignment = _assign_thirds_to_slots(best_thirds)

    def t3(s): return slot_assignment[s]

    def match(t1, t2):
        if not t1 or not t2:
            return {'team1': t1, 'team2': t2, 'winner': t1 or t2, 'p1': 100, 'p2': 0}
        s1 = max(by_team.get(t1, {}).get('champion_pct', 0), 0.1)
        s2 = max(by_team.get(t2, {}).get('champion_pct', 0), 0.1)
        p1 = round(s1 / (s1 + s2) * 100)
        return {'team1': t1, 'team2': t2, 'winner': t1 if p1 >= 50 else t2, 'p1': p1, 'p2': 100 - p1}

    def advance(matches):
        pairs = [(matches[i]['winner'], matches[i+1]['winner']) for i in range(0, len(matches), 2)]
        return [match(a, b) for a, b in pairs]

    r32 = [
        match(firsts[E],  t3(0)),          # left  — 1E  vs 3ABCDF
        match(firsts[I],  t3(1)),          # left  — 1I  vs 3CDFGH
        match(seconds[A], seconds[B]),     # left  — 2A  vs 2B
        match(firsts[F],  seconds[C]),     # left  — 1F  vs 2C
        match(seconds[K], seconds[L]),     # left  — 2K  vs 2L
        match(firsts[H],  seconds[J]),     # left  — 1H  vs 2J
        match(firsts[D],  t3(2)),          # left  — 1D  vs 3BEFIJ
        match(firsts[G],  t3(3)),          # left  — 1G  vs 3AEHIJ
        match(firsts[C],  seconds[F]),     # right — 1C  vs 2F
        match(seconds[E], seconds[I]),     # right — 2E  vs 2I
        match(firsts[A],  t3(4)),          # right — 1A  vs 3CEFHI
        match(firsts[L],  t3(5)),          # right — 1L  vs 3EHIJK
        match(firsts[J],  seconds[H]),     # right — 1J  vs 2H
        match(seconds[D], seconds[G]),     # right — 2D  vs 2G
        match(firsts[B],  t3(6)),          # right — 1B  vs 3EFGIJ
        match(firsts[K],  t3(7)),          # right — 1K  vs 3DEIJL
    ]

    r16   = advance(r32)
    qf    = advance(r16)
    sf    = advance(qf)
    final = advance(sf)

    return {
        'r32':     r32,
        'r16':     r16,
        'qf':      qf,
        'sf':      sf,
        'final':   final,
        'champion': final[0]['winner'] if final else None,
    }


@app.get("/api/groups")
def get_groups():
    """Return group structure + per-team predicted finish probabilities."""
    sim = _get_sim()
    sim_by_team = {r["team"]: r for r in sim}

    groups_data = []
    for i, group in enumerate(groups):
        teams_data = []
        for team in group:
            s = sim_by_team.get(team, {})
            ts = team_stats.get(team, {})
            teams_data.append({
                "name": team,
                "elo": round(ts.get("elo", 1500), 1),
                "form": round(ts.get("form", 0.5), 2),
                "pred_1st": s.get("group_1st_pct", 0),
                "pred_2nd": s.get("group_2nd_pct", 0),
                "pred_3rd": s.get("group_3rd_pct", 0),
                "pred_4th": s.get("group_4th_pct", 0),
                "advance_pct": round(
                    s.get("group_1st_pct", 0) + s.get("group_2nd_pct", 0), 1
                ),
            })
        # Sort by ELO descending for display
        teams_data.sort(key=lambda x: x["elo"], reverse=True)
        groups_data.append({
            "name": chr(65 + i),
            "teams": teams_data,
        })

    return {"groups": groups_data}


def _compute_group_standings():
    """
    Compute actual group standings from played results. Returns a list (one
    entry per group A–L) with the ranked table and the winner (only once all
    of that group's matches have been played).
    """
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "raw" / "results.csv")
    g = df[
        (df["tournament"] == "FIFA World Cup")
        & (df["date"] >= "2026-01-01")
        & (df["round"].astype(str) == "group")
    ]

    out = []
    for i, group in enumerate(groups):
        members = set(group)
        stats = {
            t: {"team": t, "played": 0, "win": 0, "draw": 0, "loss": 0,
                "gf": 0, "ga": 0, "points": 0}
            for t in group
        }
        played_matches = 0
        for _, r in g.iterrows():
            h, a = r["home_team"], r["away_team"]
            if h not in members or a not in members or pd.isna(r.get("home_score")):
                continue
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            played_matches += 1
            stats[h]["played"] += 1; stats[a]["played"] += 1
            stats[h]["gf"] += hs;    stats[h]["ga"] += as_
            stats[a]["gf"] += as_;   stats[a]["ga"] += hs
            if hs > as_:
                stats[h]["points"] += 3; stats[h]["win"] += 1; stats[a]["loss"] += 1
            elif hs < as_:
                stats[a]["points"] += 3; stats[a]["win"] += 1; stats[h]["loss"] += 1
            else:
                stats[h]["points"] += 1; stats[a]["points"] += 1
                stats[h]["draw"] += 1;   stats[a]["draw"] += 1

        for s in stats.values():
            s["gd"] = s["gf"] - s["ga"]
        # FIFA tiebreakers (simplified): points, goal difference, goals for.
        # Final tiebreak by name keeps the order deterministic.
        ranked = sorted(
            stats.values(),
            key=lambda s: (-s["points"], -s["gd"], -s["gf"], s["team"]),
        )
        n = len(group)
        expected = n * (n - 1) // 2          # round-robin match count
        complete = played_matches >= expected
        out.append({
            "name":     chr(65 + i),
            "index":    i,
            "teams":    list(group),
            "standings": ranked,
            "complete": complete,
            "winner":   ranked[0]["team"] if (complete and ranked) else None,
        })
    return out


@app.get("/api/group-winners")
def get_group_winners():
    """
    Group-winner prediction data: each group's teams, current standings, the
    actual winner once the group is complete, plus the shared lock deadline.
    """
    now_pdt = datetime.datetime.now(_PDT)
    return {
        "groups":    _compute_group_standings(),
        "lock_at":   _GROUP_WINNER_LOCK.astimezone(datetime.timezone.utc).isoformat(),
        "is_locked": now_pdt >= _GROUP_WINNER_LOCK,
        "points":    _GROUP_WINNER_POINTS,
    }


@app.post("/api/update")
def live_update(x_api_key: str = Header(default=None)):
    """Pull latest WC results from football-data.org and invalidate sim cache."""
    api_key = (x_api_key or os.environ.get("FOOTBALL_DATA_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="FOOTBALL_DATA_API_KEY not set")

    try:
        result = update_results(api_key)
    except Exception:
        # Never surface a raw exception — it can contain the API key.
        print("Live update failed with an unexpected error.")
        raise HTTPException(status_code=502, detail="Live update failed")

    if result.get("error"):
        # update_results already redacts the key, but redact again defensively
        # in case the key ever leaks into the message via another path.
        detail = result["error"].replace(api_key, "***REDACTED***") if api_key else result["error"]
        raise HTTPException(status_code=502, detail=detail)

    if result["updated"] > 0:
        _invalidate_cache()

    return result


# ---------------------------------------------------------------------------
# Privacy policy
# ---------------------------------------------------------------------------

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Privacy Policy — World Cup 2026 Predictor</title>
  <style>
    body{font-family:system-ui,sans-serif;max-width:700px;margin:60px auto;padding:0 2rem;color:#222;line-height:1.7}
    h1{font-size:1.6rem;margin-bottom:.25rem}
    h2{font-size:1.1rem;margin-top:2rem}
    p,li{font-size:.95rem;color:#444}
    a{color:#4f8ef7}
  </style>
</head>
<body>
  <h1>Privacy Policy</h1>
  <p><strong>World Cup 2026 Predictor</strong> &mdash; Last updated: June 1, 2026</p>

  <h2>What we collect</h2>
  <p>We collect your email address and name when you sign in with Google. This is used solely to identify your account and display your username on the leaderboard.</p>

  <h2>What we don't do</h2>
  <p>We do not sell, share, or rent your personal information to any third party. We do not use your data for advertising.</p>

  <h2>How your data is stored</h2>
  <p>Your account information and score predictions are stored securely via Supabase. We do not store your Google password.</p>

  <h2>Third-party services</h2>
  <p>This app uses Google OAuth for authentication. <a href="https://policies.google.com/privacy">Google's privacy policy</a> applies to the sign-in process.</p>

  <h2>Contact</h2>
  <p>If you have any questions, contact <a href="mailto:shrenilsoni@gmail.com">shrenilsoni@gmail.com</a>.</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Frontend — must come last so API routes take priority
# ---------------------------------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)
