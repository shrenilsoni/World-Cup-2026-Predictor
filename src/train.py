import pandas as pd
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent

FEATURES = ['home_elo', 'away_elo', 'elo_diff', 'home_form', 'away_form', 'home_gd', 'away_gd', 'neutral']

GOAL_FEATURES_HOME = ['home_avg_scored', 'away_avg_conceded', 'elo_diff', 'neutral']
GOAL_FEATURES_AWAY = ['away_avg_scored', 'home_avg_conceded', 'elo_diff', 'neutral']


def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'home_win'
    elif row['home_score'] < row['away_score']:
        return 'away_win'
    return 'draw'


def prepare_data(df):
    df = df[df['home_score'].notna()].copy()
    df = df[df['date'] >= '2016-01-01'].copy()
    df['result'] = df.apply(get_result, axis=1)
    df['neutral'] = df['neutral'].astype(int)
    return df.sort_values('date').reset_index(drop=True)


def train():
    df = pd.read_csv(ROOT / 'data' / 'processed' / 'matches_with_features.csv')
    df = prepare_data(df)

    # Time-based split — never random, or future data leaks into training
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    X_train, y_train = train_df[FEATURES], train_df['result']
    X_test, y_test = test_df[FEATURES], test_df['result']

    print(f"Train: {len(train_df)} matches ({train_df['date'].min()} → {train_df['date'].max()})")
    print(f"Test:  {len(test_df)} matches ({test_df['date'].min()} → {test_df['date'].max()})\n")

    candidates = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    }

    best_model, best_loss, best_name = None, float('inf'), None

    for name, base_model in candidates.items():
        # CalibratedClassifierCV wraps the model to ensure probabilities are well-calibrated
        model = CalibratedClassifierCV(base_model, cv=5, method='isotonic')
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)
        loss = log_loss(y_test, probs)
        acc = accuracy_score(y_test, model.predict(X_test))

        print(f"{name}:")
        print(f"  Log-loss: {loss:.4f}  (lower = better calibrated probabilities)")
        print(f"  Accuracy: {acc:.3f}  (% of exact result predicted correctly)\n")

        if loss < best_loss:
            best_loss, best_model, best_name = loss, model, name

    print(f"Winner: {best_name}  (log-loss {best_loss:.4f})")

    # ── Poisson goal models ──────────────────────────────────────────────────
    # StandardScaler is essential: elo_diff has variance ~33,000 vs avg_scored ~0.5
    # Without scaling, PoissonRegressor's gradient for small features rounds to zero.
    print("\nTraining Poisson goal models...")
    goal_train = train_df.copy()
    goal_test  = test_df.copy()

    goal_model_home = Pipeline([
        ('scaler', StandardScaler()),
        ('poisson', PoissonRegressor(alpha=0.001, max_iter=500)),
    ])
    goal_model_home.fit(goal_train[GOAL_FEATURES_HOME], goal_train['home_score'])

    goal_model_away = Pipeline([
        ('scaler', StandardScaler()),
        ('poisson', PoissonRegressor(alpha=0.001, max_iter=500)),
    ])
    goal_model_away.fit(goal_train[GOAL_FEATURES_AWAY], goal_train['away_score'])

    mae_home = mean_absolute_error(goal_test['home_score'], goal_model_home.predict(goal_test[GOAL_FEATURES_HOME]))
    mae_away = mean_absolute_error(goal_test['away_score'], goal_model_away.predict(goal_test[GOAL_FEATURES_AWAY]))
    print(f"  Poisson home goals MAE: {mae_home:.4f}")
    print(f"  Poisson away goals MAE: {mae_away:.4f}")

    (ROOT / 'models').mkdir(parents=True, exist_ok=True)
    with open(ROOT / 'models' / 'model.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'features': FEATURES,
            'classes': best_model.classes_.tolist(),
            'goal_model_home': goal_model_home,
            'goal_model_away': goal_model_away,
            'goal_features_home': GOAL_FEATURES_HOME,
            'goal_features_away': GOAL_FEATURES_AWAY,
        }, f)

    print("Model saved → models/model.pkl")
    return best_model


if __name__ == '__main__':
    train()
