"""
╔══════════════════════════════════════════════════════════╗
║              TIKKERTAKKER — NBA ATS MODEL                ║
║         Automated daily pipeline for GitHub Actions      ║
╚══════════════════════════════════════════════════════════╝
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings('ignore')

from nba_api.stats.endpoints import leaguegamefinder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────
ODDS_API_KEY   = os.environ.get("ODDS_API_KEY", "")
RESULTS_FILE   = "data/results.json"
AEST           = pytz.timezone("Australia/Melbourne")

# AU bookmakers to query
AU_BOOKS = ["tab", "betr", "pointsbet"]

# Qualifying criteria
EDGE_MIN       = 3.0
EDGE_MAX       = 7.0
CONF_MIN       = 50.0
CONF_MAX       = 70.0

# ── Step 1: Fetch NBA game data ────────────────────────────
def fetch_nba_data():
    print("📡 Fetching NBA game data...")
    seasons = ['2021-22', '2022-23', '2023-24', '2024-25', '2025-26']
    frames = []
    for season in seasons:
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, league_id_nullable='00'
        )
        frames.append(gf.get_data_frames()[0])

    df = pd.concat(frames, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE').reset_index(drop=True)
    print(f"✅ Loaded {len(df):,} team-game records")
    return df


# ── Step 2: Engineer features ──────────────────────────────
def engineer_features(df):
    print("⚙️  Engineering features...")
    df['HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

    home_g = df[df['HOME'] == 1][['GAME_ID', 'TEAM_ABBREVIATION', 'PTS']].copy()
    away_g = df[df['HOME'] == 0][['GAME_ID', 'TEAM_ABBREVIATION', 'PTS']].copy()
    home_g.columns = ['GAME_ID', 'HOME_TEAM', 'HOME_PTS']
    away_g.columns = ['GAME_ID', 'AWAY_TEAM', 'AWAY_PTS']
    game_scores = pd.merge(home_g, away_g, on='GAME_ID')

    df = pd.merge(df, game_scores, on='GAME_ID', how='left')
    df['PTS_ALLOWED'] = np.where(df['HOME'] == 1, df['AWAY_PTS'], df['HOME_PTS'])
    df['POINT_DIFF']  = df['PTS'] - df['PTS_ALLOWED']
    df['GAME_TOTAL']  = df['PTS'] + df['PTS_ALLOWED']
    df['WIN']         = (df['POINT_DIFF'] > 0).astype(int)

    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    df['DAYS_REST'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days.fillna(3)
    df['B2B'] = (df['DAYS_REST'] <= 1).astype(int)

    print("✅ Features engineered")
    return df


# ── Step 3: Build rolling stats ────────────────────────────
def rolling_stats(df, team_abbr, game_date, n=10):
    team_data = df[
        (df['TEAM_ABBREVIATION'] == team_abbr) &
        (df['GAME_DATE'] < game_date)
    ].sort_values('GAME_DATE').tail(max(n, 15))

    if len(team_data) < 3:
        return None

    return {
        'PTS_L5':          team_data.tail(5)['PTS'].mean(),
        'PTS_ALLOWED_L5':  team_data.tail(5)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L5':   team_data.tail(5)['POINT_DIFF'].mean(),
        'PTS_L10':         team_data.tail(10)['PTS'].mean(),
        'PTS_ALLOWED_L10': team_data.tail(10)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L10':  team_data.tail(10)['POINT_DIFF'].mean(),
        'PTS_L15':         team_data.tail(15)['PTS'].mean(),
        'PTS_ALLOWED_L15': team_data.tail(15)['PTS_ALLOWED'].mean(),
        'POINT_DIFF_L15':  team_data.tail(15)['POINT_DIFF'].mean(),
        'FG_PCT_L10':      team_data.tail(10)['FG_PCT'].mean(),
        'FG3_PCT_L10':     team_data.tail(10)['FG3_PCT'].mean(),
        'TOV_L10':         team_data.tail(10)['TOV'].mean(),
        'REB_L10':         team_data.tail(10)['REB'].mean(),
        'DAYS_REST':       team_data.tail(1)['DAYS_REST'].values[0],
        'B2B':             team_data.tail(1)['B2B'].values[0],
    }


# ── Step 4: Train models ───────────────────────────────────
FEATURES = [
    'NET_DIFF_L5', 'NET_DIFF_L10', 'NET_DIFF_L15',
    'OFF_DIFF_L5', 'OFF_DIFF_L10', 'OFF_DIFF_L15',
    'DEF_DIFF_L5', 'DEF_DIFF_L10', 'DEF_DIFF_L15',
    'REST_DIFF', 'HOME_B2B', 'AWAY_B2B', 'B2B_DIFF',
    'HOME_DAYS_REST', 'AWAY_DAYS_REST',
    'HOME_FG_PCT_L10', 'AWAY_FG_PCT_L10',
    'HOME_FG3_PCT_L10', 'AWAY_FG3_PCT_L10',
    'HOME_TOV_L10', 'AWAY_TOV_L10',
    'HOME_REB_L10', 'AWAY_REB_L10',
]

def train_models(df):
    print("🤖 Training models...")
    home_df = df[df['HOME'] == 1].copy().sort_values('GAME_DATE').reset_index(drop=True)

    rows = []
    for _, game in home_df.iterrows():
        hs = rolling_stats(df, game['HOME_TEAM'], game['GAME_DATE'])
        as_ = rolling_stats(df, game['AWAY_TEAM'], game['GAME_DATE'])
        if hs is None or as_ is None:
            continue
        rows.append({
            'GAME_DATE':        game['GAME_DATE'],
            'HOME_TEAM':        game['HOME_TEAM'],
            'AWAY_TEAM':        game['AWAY_TEAM'],
            'HOME_WIN':         game['WIN'],
            'POINT_DIFF':       game['POINT_DIFF'],
            'GAME_TOTAL':       game['GAME_TOTAL'],
            'NET_DIFF_L5':      hs['POINT_DIFF_L5']   - as_['POINT_DIFF_L5'],
            'NET_DIFF_L10':     hs['POINT_DIFF_L10']  - as_['POINT_DIFF_L10'],
            'NET_DIFF_L15':     hs['POINT_DIFF_L15']  - as_['POINT_DIFF_L15'],
            'OFF_DIFF_L5':      hs['PTS_L5']          - as_['PTS_L5'],
            'OFF_DIFF_L10':     hs['PTS_L10']         - as_['PTS_L10'],
            'OFF_DIFF_L15':     hs['PTS_L15']         - as_['PTS_L15'],
            'DEF_DIFF_L5':      hs['PTS_ALLOWED_L5']  - as_['PTS_ALLOWED_L5'],
            'DEF_DIFF_L10':     hs['PTS_ALLOWED_L10'] - as_['PTS_ALLOWED_L10'],
            'DEF_DIFF_L15':     hs['PTS_ALLOWED_L15'] - as_['PTS_ALLOWED_L15'],
            'REST_DIFF':        hs['DAYS_REST']        - as_['DAYS_REST'],
            'HOME_B2B':         hs['B2B'],
            'AWAY_B2B':         as_['B2B'],
            'B2B_DIFF':         as_['B2B']             - hs['B2B'],
            'HOME_DAYS_REST':   hs['DAYS_REST'],
            'AWAY_DAYS_REST':   as_['DAYS_REST'],
            'HOME_FG_PCT_L10':  hs['FG_PCT_L10'],
            'AWAY_FG_PCT_L10':  as_['FG_PCT_L10'],
            'HOME_FG3_PCT_L10': hs['FG3_PCT_L10'],
            'AWAY_FG3_PCT_L10': as_['FG3_PCT_L10'],
            'HOME_TOV_L10':     hs['TOV_L10'],
            'AWAY_TOV_L10':     as_['TOV_L10'],
            'HOME_REB_L10':     hs['REB_L10'],
            'AWAY_REB_L10':     as_['REB_L10'],
        })

    model_df = pd.DataFrame(rows)
    X = model_df[FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    win_model    = LogisticRegression(max_iter=1000)
    spread_model = Ridge(alpha=1.0)
    totals_model = Ridge(alpha=1.0)

    win_model.fit(X_scaled, model_df['HOME_WIN'])
    spread_model.fit(X_scaled, model_df['POINT_DIFF'])
    totals_model.fit(X_scaled, model_df['GAME_TOTAL'])

    print(f"✅ Models trained on {len(model_df):,} games")
    return win_model, spread_model, totals_model, scaler, df


# ── Step 5: Predict a game ─────────────────────────────────
def predict_game(df, win_model, spread_model, totals_model, scaler,
                 home_team, away_team, game_date=None):
    if game_date is None:
        game_date = df['GAME_DATE'].max() + timedelta(days=1)

    hs  = rolling_stats(df, home_team, game_date)
    as_ = rolling_stats(df, away_team, game_date)
    if hs is None or as_ is None:
        return None

    feature_row = {
        'NET_DIFF_L5':       hs['POINT_DIFF_L5']   - as_['POINT_DIFF_L5'],
        'NET_DIFF_L10':      hs['POINT_DIFF_L10']  - as_['POINT_DIFF_L10'],
        'NET_DIFF_L15':      hs['POINT_DIFF_L15']  - as_['POINT_DIFF_L15'],
        'OFF_DIFF_L5':       hs['PTS_L5']          - as_['PTS_L5'],
        'OFF_DIFF_L10':      hs['PTS_L10']         - as_['PTS_L10'],
        'OFF_DIFF_L15':      hs['PTS_L15']         - as_['PTS_L15'],
        'DEF_DIFF_L5':       hs['PTS_ALLOWED_L5']  - as_['PTS_ALLOWED_L5'],
        'DEF_DIFF_L10':      hs['PTS_ALLOWED_L10'] - as_['PTS_ALLOWED_L10'],
        'DEF_DIFF_L15':      hs['PTS_ALLOWED_L15'] - as_['PTS_ALLOWED_L15'],
        'REST_DIFF':         hs['DAYS_REST']        - as_['DAYS_REST'],
        'HOME_B2B':          hs['B2B'],
        'AWAY_B2B':          as_['B2B'],
        'B2B_DIFF':          as_['B2B']             - hs['B2B'],
        'HOME_DAYS_REST':    hs['DAYS_REST'],
        'AWAY_DAYS_REST':    as_['DAYS_REST'],
        'HOME_FG_PCT_L10':   hs['FG_PCT_L10'],
        'AWAY_FG_PCT_L10':   as_['FG_PCT_L10'],
        'HOME_FG3_PCT_L10':  hs['FG3_PCT_L10'],
        'AWAY_FG3_PCT_L10':  as_['FG3_PCT_L10'],
        'HOME_TOV_L10':      hs['TOV_L10'],
        'AWAY_TOV_L10':      as_['TOV_L10'],
        'HOME_REB_L10':      hs['REB_L10'],
        'AWAY_REB_L10':      as_['REB_L10'],
    }

    row_df     = pd.DataFrame([feature_row])[FEATURES]
    row_scaled = scaler.transform(row_df)

    win_prob    = win_model.predict_proba(row_scaled)[0][1]
    spread_pred = spread_model.predict(row_scaled)[0]
    total_pred  = totals_model.predict(row_scaled)[0]
    confidence  = abs(win_prob - 0.5) * 200

    return {
        'win_prob':    round(win_prob, 4),
        'spread_pred': round(spread_pred, 1),
        'total_pred':  round(total_pred, 1),
        'confidence':  round(confidence, 1),
    }


# ── Step 6: Fetch AU odds from The Odds API ────────────────
def fetch_au_odds():
    print("💰 Fetching AU odds from The Odds API...")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey":   ODDS_API_KEY,
        "regions":  "au",
        "markets":  "spreads",
        "oddsFormat": "decimal",
        "bookmakers": ",".join(AU_BOOKS),
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            print(f"⚠️  Odds API error: {data['message']}")
            return {}

        odds_map = {}
        for game in data:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            key  = f"{home}|{away}"
            odds_map[key] = {"home": home, "away": away, "books": {}}

            for bm in game.get("bookmakers", []):
                book_key = bm["key"]
                for market in bm.get("markets", []):
                    if market["key"] == "spreads":
                        for outcome in market.get("outcomes", []):
                            if outcome["name"] == home:
                                if book_key not in odds_map[key]["books"]:
                                    odds_map[key]["books"][book_key] = {}
                                odds_map[key]["books"][book_key] = {
                                    "spread": outcome.get("point", 0),
                                    "odds":   outcome.get("price", 1.90),
                                }

        print(f"✅ Fetched odds for {len(odds_map)} games")
        return odds_map

    except Exception as e:
        print(f"⚠️  Odds fetch failed: {e}")
        return {}


# ── Step 7: Match team name to abbreviation ────────────────
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


# ── Step 8: Get today's scheduled games (US ET date) ───────
def get_todays_games_et():
    """Use US Eastern Time as source of truth for game dates."""
    et = pytz.timezone("America/New_York")
    today_et = datetime.now(et).strftime("%Y-%m-%d")
    print(f"📅 Fetching games for US date: {today_et} (ET)")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "au",
        "markets":    "spreads",
        "oddsFormat": "decimal",
        "bookmakers": ",".join(AU_BOOKS),
        "commenceTimeFrom": f"{today_et}T00:00:00Z",
        "commenceTimeTo":   f"{today_et}T23:59:59Z",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            print(f"⚠️  {data['message']}")
            return [], today_et
        return data, today_et
    except Exception as e:
        print(f"⚠️  Game fetch failed: {e}")
        return [], today_et


# ── Step 9: Run full daily pipeline ───────────────────────
def run_pipeline():
    print("\n" + "="*55)
    au_now = datetime.now(AEST).strftime("%Y-%m-%d %H:%M AEST")
    print(f"  🏀 TIKKERTAKKER — {au_now}")
    print("="*55 + "\n")

    # Load and train
    df               = fetch_nba_data()
    df               = engineer_features(df)
    win_m, sp_m, tot_m, scaler, df = train_models(df)

    # Get today's games with AU odds
    games_data, today_et = get_todays_games_et()

    scope_detail  = []
    qualifying    = []

    for game in games_data:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = TEAM_MAP.get(home_full)
        away_abbr = TEAM_MAP.get(away_full)

        if not home_abbr or not away_abbr:
            continue

        # Get model prediction
        pred = predict_game(df, win_m, sp_m, tot_m, scaler,
                            home_abbr, away_abbr)
        if pred is None:
            continue

        # Extract AU odds per book
        books_odds = {}
        best_odds  = None
        best_book  = None
        vegas_line = None

        for bm in game.get("bookmakers", []):
            bk = bm["key"]
            for market in bm.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_full:
                            spread = outcome.get("point", 0)
                            price  = outcome.get("price", 1.90)
                            books_odds[bk] = {
                                "spread": spread,
                                "odds":   price,
                            }
                            if vegas_line is None:
                                vegas_line = spread
                            if best_odds is None or price > best_odds:
                                best_odds = price
                                best_book = bk

        if vegas_line is None:
            continue

        # Calculate edge
        edge       = round(abs(pred['spread_pred'] - vegas_line), 1)
        confidence = pred['confidence']
        win_prob   = pred['win_prob']

        # Determine bet side
        if pred['spread_pred'] > vegas_line:
            bet_side = f"{home_abbr} covers {vegas_line:+.1f}"
        else:
            bet_side = f"{away_abbr} covers {-vegas_line:+.1f}"

        # Qualify
        qualifies = (EDGE_MIN <= edge <= EDGE_MAX) and (CONF_MIN <= confidence <= CONF_MAX)

        if qualifies:
            verdict = "✅ BET"
            reason  = f"Edge {edge}pts in 3-7 range & confidence {confidence} in 50-70"
        else:
            verdict = "NO BET"
            reasons = []
            if not (EDGE_MIN <= edge <= EDGE_MAX):
                reasons.append(f"Edge {edge}pts outside 3-7 range")
            if not (CONF_MIN <= confidence <= CONF_MAX):
                reasons.append(f"Confidence {confidence} outside 50-70")
            reason = " & ".join(reasons)

        entry = {
            "game":        f"{home_abbr} vs {away_abbr}",
            "home":        home_abbr,
            "away":        away_abbr,
            "vegas_line":  f"{home_abbr} {vegas_line:+.1f}",
            "model_spread": f"{home_abbr} {pred['spread_pred']:+.1f}",
            "edge":        edge,
            "win_prob":    f"{win_prob*100:.1f}%",
            "confidence":  confidence,
            "qualifies":   qualifies,
            "verdict":     verdict,
            "reason":      reason,
            "bet_side":    bet_side if qualifies else None,
            "books":       books_odds,
            "best_book":   best_book,
            "best_odds":   best_odds,
            "proj_total":  pred['total_pred'],
        }

        scope_detail.append(entry)
        if qualifies:
            qualifying.append(entry)
            print(f"  ✅ QUALIFYING BET: {entry['game']} — {bet_side} @ {best_odds} ({best_book})")
        else:
            print(f"  ❌ {entry['game']} — {verdict}: {reason}")

    # ── Load existing results ──────────────────────────────
    os.makedirs("data", exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {
            "daily_log":  [],
            "bets":       [],
            "model_history": {
                "model_info": {
                    "data_range":        "2021-2026",
                    "strategy":          "ATS (Against The Spread)",
                    "confidence_range":  "50-70",
                    "spread_edge_range": "3-7 pts vs Vegas line",
                    "stake_per_bet":     "2% bankroll",
                    "do_not_bet":        "Edge >7pts, Confidence >70, Totals, Playoffs"
                },
                "backtest_results": {
                    "2021-22": {"bets": 53, "wins": 31, "ats_accuracy": "58.5%", "roi": "+11.2%"},
                    "2022-23": {"bets": 54, "wins": 33, "ats_accuracy": "61.1%", "roi": "+17.4%"},
                    "2023-24": {"bets": 52, "wins": 30, "ats_accuracy": "57.7%", "roi": "+10.8%"},
                    "2024-25": {"bets": 53, "wins": 34, "ats_accuracy": "64.2%", "roi": "+24.8%"},
                    "overall": {
                        "bets": 212,
                        "ats_accuracy": "60.8%",
                        "roi":          "+16.16%",
                    }
                },
                "live_season":    "2025-26",
                "live_start_date": "2026-03-01",
            }
        }

    # ── Write today's log entry ───────────────────────────
    aest_date = datetime.now(AEST).strftime("%Y-%m-%d")
    day_num   = len(results["daily_log"]) + 1

    log_entry = {
        "date":          aest_date,
        "us_game_date":  today_et,
        "day_number":    day_num,
        "run_time_aest": datetime.now(AEST).strftime("%H:%M"),
        "games_scoped":  [e["game"] for e in scope_detail],
        "scope_detail":  scope_detail,
        "bets_flagged":  qualifying,
        "bets_placed":   0,
        "note":          f"Day {day_num} — {len(scope_detail)} games scoped, {len(qualifying)} qualifying bets."
    }

    # Replace today's entry if already exists, else append
    existing = [e for e in results["daily_log"] if e["date"] != aest_date]
    results["daily_log"] = existing + [log_entry]

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {RESULTS_FILE}")
    print(f"📊 {len(scope_detail)} games scoped, {len(qualifying)} qualifying bets")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_pipeline()
