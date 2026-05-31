import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent


def compute_features(df):
    df = df.copy().sort_values('date').reset_index(drop=True)

    elos = {}
    form = defaultdict(list)
    gd_history = defaultdict(list)
    scored_history = defaultdict(list)
    conceded_history = defaultdict(list)
    K = 32

    home_elos, away_elos = [], []
    home_forms, away_forms = [], []
    home_gds, away_gds = [], []
    home_avg_scored_list, away_avg_scored_list = [], []
    home_avg_conceded_list, away_avg_conceded_list = [], []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']

        if home not in elos:
            elos[home] = 1500
        if away not in elos:
            elos[away] = 1500

        # Snapshot current stats BEFORE updating
        home_elos.append(elos[home])
        away_elos.append(elos[away])

        h_form = form[home]
        a_form = form[away]
        home_forms.append(h_form.count(1) / len(h_form) if h_form else 0.5)
        away_forms.append(a_form.count(1) / len(a_form) if a_form else 0.5)

        h_gd = gd_history[home]
        a_gd = gd_history[away]
        home_gds.append(float(np.mean(h_gd)) if h_gd else 0.0)
        away_gds.append(float(np.mean(a_gd)) if a_gd else 0.0)

        h_scored = scored_history[home]
        a_scored = scored_history[away]
        h_conceded = conceded_history[home]
        a_conceded = conceded_history[away]
        home_avg_scored_list.append(float(np.mean(h_scored)) if h_scored else 1.3)
        away_avg_scored_list.append(float(np.mean(a_scored)) if a_scored else 1.3)
        home_avg_conceded_list.append(float(np.mean(h_conceded)) if h_conceded else 1.3)
        away_avg_conceded_list.append(float(np.mean(a_conceded)) if a_conceded else 1.3)

        # Only update dicts if this match has a known result
        if pd.notna(row['home_score']):
            hs, as_ = row['home_score'], row['away_score']

            if hs > as_:
                actual, home_result, away_result = 1, 1, 0
            elif hs == as_:
                actual, home_result, away_result = 0.5, 0.5, 0.5
            else:
                actual, home_result, away_result = 0, 0, 1

            expected_home = 1 / (1 + 10 ** ((elos[away] - elos[home]) / 400))
            elos[home] += K * (actual - expected_home)
            elos[away] += K * ((1 - actual) - (1 - expected_home))

            form[home].append(home_result)
            form[away].append(away_result)
            if len(form[home]) > 10:
                form[home].pop(0)
            if len(form[away]) > 10:
                form[away].pop(0)

            gd_history[home].append(hs - as_)
            gd_history[away].append(as_ - hs)
            if len(gd_history[home]) > 10:
                gd_history[home].pop(0)
            if len(gd_history[away]) > 10:
                gd_history[away].pop(0)

            scored_history[home].append(hs)
            scored_history[away].append(as_)
            if len(scored_history[home]) > 10:
                scored_history[home].pop(0)
            if len(scored_history[away]) > 10:
                scored_history[away].pop(0)

            conceded_history[home].append(as_)
            conceded_history[away].append(hs)
            if len(conceded_history[home]) > 10:
                conceded_history[home].pop(0)
            if len(conceded_history[away]) > 10:
                conceded_history[away].pop(0)

    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['home_form'] = home_forms
    df['away_form'] = away_forms
    df['home_gd'] = home_gds
    df['away_gd'] = away_gds
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    df['home_avg_scored']   = home_avg_scored_list
    df['away_avg_scored']   = away_avg_scored_list
    df['home_avg_conceded'] = home_avg_conceded_list
    df['away_avg_conceded'] = away_avg_conceded_list

    # Current team stats as of their most recent match — used for WC predictions
    current_stats = {
        team: {
            'elo': round(elos[team], 4),
            'form': round(form[team].count(1) / len(form[team]) if form[team] else 0.5, 4),
            'gd': round(float(np.mean(gd_history[team])) if gd_history[team] else 0.0, 4),
            'avg_scored':   round(float(np.mean(scored_history[team]))   if scored_history[team]   else 1.3, 4),
            'avg_conceded': round(float(np.mean(conceded_history[team])) if conceded_history[team] else 1.3, 4),
        }
        for team in elos
    }

    return df, current_stats


if __name__ == '__main__':
    df = pd.read_csv(ROOT / 'data' / 'raw' / 'results.csv')
    print(f"Loaded {len(df)} rows. Computing features (this takes ~60s)...")

    df_features, current_stats = compute_features(df)

    (ROOT / 'data' / 'processed').mkdir(parents=True, exist_ok=True)
    df_features.to_csv(ROOT / 'data' / 'processed' / 'matches_with_features.csv', index=False)

    with open(ROOT / 'data' / 'processed' / 'team_stats.json', 'w') as f:
        json.dump(current_stats, f, indent=2)

    print(f"Saved matches_with_features.csv  shape: {df_features.shape}")
    print(f"Saved team_stats.json  {len(current_stats)} teams\n")

    print("Top 15 teams by ELO:")
    top = sorted(current_stats.items(), key=lambda x: x[1]['elo'], reverse=True)[:15]
    for team, stats in top:
        print(f"  {team:25s}  ELO: {stats['elo']:.1f}  Form: {stats['form']:.2f}  AvgGD: {stats['gd']:+.2f}")

    wc = df_features[df_features['home_score'].isna()]
    print(f"\nWC 2026 fixture rows with features: {len(wc)}")
