import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.stats import poisson as _scipy_poisson

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts():
    with open(ROOT / 'models' / 'model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    with open(ROOT / 'data' / 'processed' / 'team_stats.json') as f:
        team_stats = json.load(f)
    fixtures = pd.read_csv(ROOT / 'data' / 'raw' / 'results.csv')
    fixtures = fixtures[fixtures['home_score'].isna()].copy()
    # Extract goal model artifacts gracefully (absent in old model.pkl)
    goal_model_home    = artifacts.get('goal_model_home')
    goal_model_away    = artifacts.get('goal_model_away')
    goal_features_home = artifacts.get('goal_features_home', [])
    goal_features_away = artifacts.get('goal_features_away', [])
    return artifacts, team_stats, fixtures, goal_model_home, goal_model_away, goal_features_home, goal_features_away


# ---------------------------------------------------------------------------
# Poisson goal prediction
# ---------------------------------------------------------------------------

def _poisson_pmf(k, lam):
    return float(_scipy_poisson.pmf(k, max(float(lam), 0.01)))


def predict_goals(home, away, neutral, goal_model_h, goal_model_a, feat_h, feat_a, team_stats):
    """Returns expected goals, most likely score, top-5 scorelines, and Poisson-derived outcome probs."""
    hs = team_stats.get(home, {})
    as_ = team_stats.get(away, {})
    elo_diff = hs.get('elo', 1500) - as_.get('elo', 1500)
    n = int(neutral)

    X_h = pd.DataFrame([{
        'home_avg_scored':   hs.get('avg_scored', 1.3),
        'away_avg_conceded': as_.get('avg_conceded', 1.3),
        'elo_diff': elo_diff,
        'neutral':  n,
    }])[feat_h]

    X_a = pd.DataFrame([{
        'away_avg_scored':   as_.get('avg_scored', 1.3),
        'home_avg_conceded': hs.get('avg_conceded', 1.3),
        'elo_diff': elo_diff,
        'neutral':  n,
    }])[feat_a]

    lam_h = max(float(goal_model_h.predict(X_h)[0]), 0.1)
    lam_a = max(float(goal_model_a.predict(X_a)[0]), 0.1)

    # Build 9×9 score probability matrix
    max_g = 9
    score_probs = {}
    for i in range(max_g):
        for j in range(max_g):
            score_probs[(i, j)] = _poisson_pmf(i, lam_h) * _poisson_pmf(j, lam_a)

    total = sum(score_probs.values())
    score_probs = {k: v / total for k, v in score_probs.items()}

    top5 = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    most_likely_score, most_likely_pct = top5[0]

    p_hw = sum(p for (i, j), p in score_probs.items() if i > j)
    p_d  = sum(p for (i, j), p in score_probs.items() if i == j)
    p_aw = sum(p for (i, j), p in score_probs.items() if i < j)

    pred_h = round(lam_h)
    pred_a = round(lam_a)
    pred_pct = score_probs.get((pred_h, pred_a), 0)
    return {
        'home_xg':   round(lam_h, 2),
        'away_xg':   round(lam_a, 2),
        'most_likely_score': f"{pred_h}–{pred_a}",
        'most_likely_pct':   round(pred_pct * 100, 1),
        'top_scores': [
            {'score': f"{i}–{j}", 'pct': round(p * 100, 1)}
            for (i, j), p in top5
        ],
        'home_win_pct': round(p_hw * 100, 1),
        'draw_pct':     round(p_d  * 100, 1),
        'away_win_pct': round(p_aw * 100, 1),
    }


# ---------------------------------------------------------------------------
# Single match prediction
# ---------------------------------------------------------------------------

def _team_stats(team, team_stats):
    s = team_stats.get(team, {})
    return s.get('elo', 1500.0), s.get('form', 0.5), s.get('gd', 0.0)


def predict_match(home, away, neutral, model, feature_cols, classes, team_stats):
    h_elo, h_form, h_gd = _team_stats(home, team_stats)
    a_elo, a_form, a_gd = _team_stats(away, team_stats)
    X = pd.DataFrame([{
        'home_elo': h_elo, 'away_elo': a_elo, 'elo_diff': h_elo - a_elo,
        'home_form': h_form, 'away_form': a_form,
        'home_gd': h_gd, 'away_gd': a_gd, 'neutral': int(neutral),
    }])[feature_cols]
    probs = model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(classes, probs)}


def precompute_probs(fixtures, wc_teams, model, feature_cols, classes, team_stats):
    """
    Build a single batch of all matchups, call predict_proba once, cache results.
    Group stage: 72 fixed matchups.
    Knockout stage: all C(48,2)=1128 possible pairings at neutral=True.
    """
    rows, keys = [], []

    for _, row in fixtures.iterrows():
        h, a, n = row['home_team'], row['away_team'], bool(row['neutral'])
        h_elo, h_form, h_gd = _team_stats(h, team_stats)
        a_elo, a_form, a_gd = _team_stats(a, team_stats)
        rows.append([h_elo, a_elo, h_elo - a_elo, h_form, a_form, h_gd, a_gd, int(n)])
        keys.append((h, a, n, False))

    teams = sorted(wc_teams)
    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            h_elo, h_form, h_gd = _team_stats(t1, team_stats)
            a_elo, a_form, a_gd = _team_stats(t2, team_stats)
            rows.append([h_elo, a_elo, h_elo - a_elo, h_form, a_form, h_gd, a_gd, 1])
            keys.append((t1, t2, True, True))

    X = pd.DataFrame(rows, columns=feature_cols)
    all_probs = model.predict_proba(X)

    cache = {}
    for (h, a, n, is_ko_pair), probs in zip(keys, all_probs):
        p = {cls: float(v) for cls, v in zip(classes, probs)}
        cache[(h, a, n)] = p
        if is_ko_pair:
            cache[(a, h, True)] = {'home_win': p['away_win'], 'draw': p['draw'], 'away_win': p['home_win']}

    return cache


def _sample_result(probs):
    r = np.random.random()
    if r < probs['home_win']:
        return 'home_win'
    elif r < probs['home_win'] + probs['draw']:
        return 'draw'
    return 'away_win'


def _knockout_winner(team1, team2, prob_cache, team_stats):
    """Knockout match — no draws. Draws resolved by ELO-weighted coin flip."""
    probs = prob_cache[(team1, team2, True)]
    result = _sample_result(probs)
    if result == 'home_win':
        return team1
    if result == 'away_win':
        return team2
    h_elo = team_stats.get(team1, {}).get('elo', 1500)
    a_elo = team_stats.get(team2, {}).get('elo', 1500)
    et_prob = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
    return team1 if np.random.random() < et_prob else team2


# ---------------------------------------------------------------------------
# Group extraction from fixture schedule
# ---------------------------------------------------------------------------

def extract_groups(fixtures):
    """
    Reconstruct WC groups by finding 4-team cliques in the fixture graph,
    then reorder to match FIFA's official Group A-L labelling.
    """
    # Official FIFA WC 2026 group order (December 5, 2025 draw, Kennedy Center, Washington D.C.)
    FIFA_ORDER = [
        {'Mexico', 'South Africa', 'South Korea', 'Czech Republic'},       # A
        {'Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'},       # B
        {'Brazil', 'Morocco', 'Haiti', 'Scotland'},                         # C
        {'United States', 'Paraguay', 'Australia', 'Turkey'},               # D
        {'Germany', 'Curaçao', 'Ivory Coast', 'Ecuador'},                   # E
        {'Netherlands', 'Japan', 'Sweden', 'Tunisia'},                      # F
        {'Belgium', 'Egypt', 'Iran', 'New Zealand'},                        # G
        {'Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'},                 # H
        {'France', 'Senegal', 'Iraq', 'Norway'},                            # I
        {'Argentina', 'Algeria', 'Austria', 'Jordan'},                      # J
        {'Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'},                 # K
        {'England', 'Croatia', 'Ghana', 'Panama'},                          # L
    ]

    match_pairs = set()
    all_teams = set()
    for _, row in fixtures.iterrows():
        h, a = row['home_team'], row['away_team']
        match_pairs.add((min(h, a), max(h, a)))
        all_teams.add(h)
        all_teams.add(a)

    all_teams = sorted(all_teams)
    found = []
    assigned = set()

    for combo in combinations(all_teams, 4):
        if any(t in assigned for t in combo):
            continue
        if all((min(a, b), max(a, b)) in match_pairs for a, b in combinations(combo, 2)):
            found.append(set(combo))
            assigned.update(combo)

    # Sort extracted groups into FIFA's official A-L order
    ordered = []
    for official_set in FIFA_ORDER:
        for g in found:
            if g == official_set:
                ordered.append(sorted(g))
                break
    # Append any unmatched groups at the end (safety net)
    matched = {frozenset(g) for g in ordered}
    for g in found:
        if g not in matched:
            ordered.append(sorted(g))

    return ordered


# ---------------------------------------------------------------------------
# Group stage simulation
# ---------------------------------------------------------------------------

def _sim_goals(result, h_elo, a_elo):
    """Rough goal counts consistent with the result outcome."""
    base = max(0.5, 1 + (h_elo - a_elo) / 800)
    if result == 'home_win':
        hg = max(1, int(np.random.poisson(base + 0.5)))
        ag = int(np.random.poisson(max(0.3, base - 0.8)))
        ag = min(ag, hg - 1)
    elif result == 'draw':
        hg = int(np.random.poisson(base))
        ag = hg
    else:
        ag = max(1, int(np.random.poisson(base + 0.5)))
        hg = int(np.random.poisson(max(0.3, base - 0.8)))
        hg = min(hg, ag - 1)
    return max(0, hg), max(0, ag)


def build_group_fixture_lists(fixtures, groups):
    """Pre-extract group fixtures as plain lists of tuples — done once before the simulation loop."""
    group_fixture_lists = {}
    for group in groups:
        group_set = set(group)
        gf = fixtures[fixtures['home_team'].isin(group_set) & fixtures['away_team'].isin(group_set)]
        group_fixture_lists[tuple(group)] = [
            (row['home_team'], row['away_team'], bool(row['neutral']))
            for _, row in gf.iterrows()
        ]
    return group_fixture_lists


def simulate_group_stage(groups, group_fixture_lists, prob_cache, team_stats):
    standings = {}

    for group in groups:
        pts = defaultdict(int)
        gd = defaultdict(int)
        gf = defaultdict(int)

        for home, away, neutral in group_fixture_lists[tuple(group)]:
            probs = prob_cache[(home, away, neutral)]
            result = _sample_result(probs)

            h_elo = team_stats.get(home, {}).get('elo', 1500)
            a_elo = team_stats.get(away, {}).get('elo', 1500)
            hg, ag = _sim_goals(result, h_elo, a_elo)

            if result == 'home_win':
                pts[home] += 3
            elif result == 'draw':
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3

            gd[home] += hg - ag
            gd[away] += ag - hg
            gf[home] += hg
            gf[away] += ag

        ranked = sorted(group, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        standings[tuple(group)] = {
            'ranked': ranked,
            'pts': dict(pts),
            'gd': dict(gd),
            'gf': dict(gf),
        }

    return standings


def get_advancing_teams(standings, groups):
    """Top 2 per group + 8 best third-place teams (with group index for bracket slotting)."""
    firsts, seconds, thirds_data = [], [], []

    for group_idx, group in enumerate(groups):
        key = tuple(group)
        ranked = standings[key]['ranked']
        firsts.append(ranked[0])
        seconds.append(ranked[1])
        thirds_data.append((
            ranked[2],
            group_idx,   # which FIFA group (A=0 … L=11) — needed for bracket slot assignment
            standings[key]['pts'].get(ranked[2], 0),
            standings[key]['gd'].get(ranked[2], 0),
            standings[key]['gf'].get(ranked[2], 0),
        ))

    thirds_sorted = sorted(thirds_data, key=lambda x: (x[2], x[3], x[4]), reverse=True)
    # Return as (team, group_idx) pairs — group_idx tells the bracket which slot this team can fill
    best_thirds = [(t[0], t[1]) for t in thirds_sorted[:8]]

    return firsts, seconds, best_thirds


def _assign_thirds_to_slots(thirds_with_groups):
    """
    Assign the 8 best 3rd-place teams to the 8 bracket slots using backtracking.
    Each slot only accepts 3rd-place teams from specific groups (FIFA rule).

    Slot → valid group indices (A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11):
      0 (vs 1E): A B C D F     → {0,1,2,3,5}
      1 (vs 1I): C D F G H     → {2,3,5,6,7}
      2 (vs 1D): B E F I J     → {1,4,5,8,9}
      3 (vs 1G): A E H I J     → {0,4,7,8,9}
      4 (vs 1A): C E F H I     → {2,4,5,7,8}
      5 (vs 1L): E H I J K     → {4,7,8,9,10}
      6 (vs 1B): E F G I J     → {4,5,6,8,9}
      7 (vs 1K): D E I J L     → {3,4,8,9,11}
    """
    SLOT_GROUPS = [
        {0,1,2,3,5},
        {2,3,5,6,7},
        {1,4,5,8,9},
        {0,4,7,8,9},
        {2,4,5,7,8},
        {4,7,8,9,10},
        {4,5,6,8,9},
        {3,4,8,9,11},
    ]

    assignment = [None] * 8
    used_teams = [False] * 8

    def backtrack(slot):
        if slot == 8:
            return True
        for i, (team, group_idx) in enumerate(thirds_with_groups):
            if not used_teams[i] and group_idx in SLOT_GROUPS[slot]:
                assignment[slot] = team
                used_teams[i] = True
                if backtrack(slot + 1):
                    return True
                assignment[slot] = None
                used_teams[i] = False
        return False

    if not backtrack(0):
        # Fallback: fill slots in order regardless of group (shouldn't happen with valid data)
        for i, (team, _) in enumerate(thirds_with_groups):
            assignment[i] = team

    return assignment


# ---------------------------------------------------------------------------
# Knockout bracket simulation
# ---------------------------------------------------------------------------

def simulate_bracket(firsts, seconds, thirds_with_groups, prob_cache, team_stats):
    """
    Official FIFA WC 2026 R32 bracket (from the published draw graphic).

    Group index mapping: A=0 B=1 C=2 D=3 E=4 F=5 G=6 H=7 I=8 J=9 K=10 L=11

    Left side R32:
      1E vs 3[ABCDF]  |  1I vs 3[CDFGH]  |  2A vs 2B  |  1F vs 2C
      2K vs 2L        |  1H vs 2J         |  1D vs 3[BEFIJ]  |  1G vs 3[AEHIJ]

    Right side R32:
      1C vs 2F  |  2E vs 2I  |  1A vs 3[CEFHI]  |  1L vs 3[EHIJK]
      1J vs 2H  |  2D vs 2G  |  1B vs 3[EFGIJ]  |  1K vs 3[DEIJL]
    """
    A,B,C,D,E,F,G,H,I,J,K,L = range(12)

    t3 = _assign_thirds_to_slots(thirds_with_groups)

    r32_matchups = [
        # Left side
        (firsts[E],   t3[0]),          # 1E  vs 3ABCDF
        (firsts[I],   t3[1]),          # 1I  vs 3CDFGH
        (seconds[A],  seconds[B]),     # 2A  vs 2B
        (firsts[F],   seconds[C]),     # 1F  vs 2C
        (seconds[K],  seconds[L]),     # 2K  vs 2L
        (firsts[H],   seconds[J]),     # 1H  vs 2J
        (firsts[D],   t3[2]),          # 1D  vs 3BEFIJ
        (firsts[G],   t3[3]),          # 1G  vs 3AEHIJ
        # Right side
        (firsts[C],   seconds[F]),     # 1C  vs 2F
        (seconds[E],  seconds[I]),     # 2E  vs 2I
        (firsts[A],   t3[4]),          # 1A  vs 3CEFHI
        (firsts[L],   t3[5]),          # 1L  vs 3EHIJK
        (firsts[J],   seconds[H]),     # 1J  vs 2H
        (seconds[D],  seconds[G]),     # 2D  vs 2G
        (firsts[B],   t3[6]),          # 1B  vs 3EFGIJ
        (firsts[K],   t3[7]),          # 1K  vs 3DEIJL
    ]

    def play_round(matchups):
        results = []
        for t1, t2 in matchups:
            if t1 is None or t2 is None:
                winner = t1 or t2
                results.append({'team1': t1, 'team2': t2, 'winner': winner})
            else:
                w = _knockout_winner(t1, t2, prob_cache, team_stats)
                results.append({'team1': t1, 'team2': t2, 'winner': w})
        return results

    def winners_of(round_results):
        return [r['winner'] for r in round_results]

    bracket = {}
    bracket['r32'] = play_round(r32_matchups)

    r16 = [(winners_of(bracket['r32'])[i], winners_of(bracket['r32'])[i + 1])
           for i in range(0, 16, 2)]
    bracket['r16'] = play_round(r16)

    qf = [(winners_of(bracket['r16'])[i], winners_of(bracket['r16'])[i + 1])
          for i in range(0, 8, 2)]
    bracket['qf'] = play_round(qf)

    sf = [(winners_of(bracket['qf'])[i], winners_of(bracket['qf'])[i + 1])
          for i in range(0, 4, 2)]
    bracket['sf'] = play_round(sf)

    final = [(winners_of(bracket['sf'])[0], winners_of(bracket['sf'])[1])]
    bracket['final'] = play_round(final)

    return bracket


# ---------------------------------------------------------------------------
# Monte Carlo tournament simulator
# ---------------------------------------------------------------------------

def run_monte_carlo(fixtures, groups, model, feature_cols, classes, team_stats, n=10000):
    all_wc_teams = sorted(set(fixtures['home_team'].tolist() + fixtures['away_team'].tolist()))

    np.random.seed(42)
    print("  Pre-computing match probabilities...")
    prob_cache = precompute_probs(fixtures, all_wc_teams, model, feature_cols, classes, team_stats)
    group_fixture_lists = build_group_fixture_lists(fixtures, groups)
    print(f"  Cached {len(prob_cache)} matchup probabilities. Running {n} simulations...")

    champion_counts = defaultdict(int)
    reach = {team: defaultdict(int) for team in all_wc_teams}
    group_pos = {team: defaultdict(int) for team in all_wc_teams}

    for i in range(n):
        if i % 2000 == 0:
            print(f"  {i}/{n} simulations...")

        standings = simulate_group_stage(groups, group_fixture_lists, prob_cache, team_stats)
        firsts, seconds, thirds = get_advancing_teams(standings, groups)

        # Track group finishing positions
        for group in groups:
            ranked = standings[tuple(group)]['ranked']
            for pos, team in enumerate(ranked):
                group_pos[team][pos + 1] += 1

        thirds_teams = [t for t, _ in thirds]
        advancing = set(firsts + seconds + thirds_teams)
        for team in advancing:
            reach[team]['r32'] += 1

        bracket = simulate_bracket(firsts, seconds, thirds, prob_cache, team_stats)

        for stage in ('r16', 'qf', 'sf', 'final'):
            for result in bracket.get(stage, []):
                reach[result['winner']][stage] += 1

        champion = bracket['final'][0]['winner']
        champion_counts[champion] += 1

    results = []
    for team in all_wc_teams:
        results.append({
            'team': team,
            'elo': round(team_stats.get(team, {}).get('elo', 1500), 1),
            'champion_pct': round(champion_counts[team] / n * 100, 1),
            'final_pct':    round(reach[team]['final'] / n * 100, 1),
            'sf_pct':       round(reach[team]['sf']    / n * 100, 1),
            'qf_pct':       round(reach[team]['qf']    / n * 100, 1),
            'r16_pct':      round(reach[team]['r16']   / n * 100, 1),
            'r32_pct':      round(reach[team]['r32']   / n * 100, 1),
            'group_1st_pct': round(group_pos[team][1] / n * 100, 1),
            'group_2nd_pct': round(group_pos[team][2] / n * 100, 1),
            'group_3rd_pct': round(group_pos[team][3] / n * 100, 1),
            'group_4th_pct': round(group_pos[team][4] / n * 100, 1),
        })

    results.sort(key=lambda x: x['champion_pct'], reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    artifacts, team_stats, fixtures, goal_model_home, goal_model_away, goal_feat_h, goal_feat_a = load_artifacts()
    model       = artifacts['model']
    feature_cols = artifacts['features']
    classes     = artifacts['classes']

    groups = extract_groups(fixtures)
    print(f"Reconstructed {len(groups)} groups from fixture schedule:")
    for i, g in enumerate(groups):
        print(f"  Group {chr(65 + i)}: {', '.join(g)}")

    print("\nRunning 10,000 Monte Carlo simulations...")
    results = run_monte_carlo(fixtures, groups, model, feature_cols, classes, team_stats, n=10000)

    print("\n{'Team':<25} {'Champion%':>10} {'Final%':>8} {'SF%':>6} {'QF%':>6}")
    print("-" * 60)
    for r in results[:20]:
        print(f"  {r['team']:<23} {r['champion_pct']:>9.1f}% {r['final_pct']:>7.1f}% "
              f"{r['sf_pct']:>5.1f}% {r['qf_pct']:>5.1f}%")
