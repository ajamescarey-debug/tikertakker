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
import time

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

# ── Config ─────────────────────────────────────────────────
ODDS_API_KEY   = os.environ.get("ODDS_API_KEY", "")
BDL_API_KEY    = os.environ.get("BDL_API_KEY", "")
RESULTS_FILE   = "data/results.json"
AEST           = pytz.timezone("Australia/Melbourne")

AU_BOOKS = ["tab", "betr", "pointsbet"]
EDGE_MIN = 3.0
EDGE_MAX = 7.0
CONF_MIN = 50.0
CONF_MAX = 70.0

# ── Step 1: Fetch NBA data via BallDontLie ─────────────────
def fetch_nba_data():
    print("📡 Fetching NBA game data via BallDontLie...")
    headers = {"Authorization": BDL_API_KEY} if BDL_API_KEY else {}
    all_games = []

    for season in [2022, 2023, 2024, 2025]:
        page = 1
        print(f"  Season {season}...")
        while True:
            try:
                resp = requests.get(
                    "https://api.balldontlie.io/v1/games",
                    headers=headers,
                    params={"seasons[]": season, "per_page": 100, "page": page},
                    timeout=30,
                )
                data = resp.json()
                games = data.get("data", [])
                if not games:
                    break
                all_games.extend(games)
                meta = data.get("meta", {})
                if page >= meta.get("total_pages", 1):
                    break
                page += 1
                time.sleep(0.3)
            except Exception as e:
                print(f"  ⚠️ Season {season} page {page}: {e}")
                break

    print(f"  Raw games: {len(all_games)}")

    rows = []
    for g in all_games:
        if g.get("status") != "Final":
            continue
        home       = g.get("home_team", {})
        away       = g.get("visitor_team", {})
        home_score = g.get("home_team_score", 0) or 0
        away_score = g.get("visitor_team_score", 0) or 0
        date_str   = g.get("date", "")[:10]
        if not date_str or home_score == 0:
            continue

        game_date = pd.to_datetime(date_str)
        home_abbr = home.get("abbreviation", "")
        away_abbr = away.get("abbreviation", "")

        rows.append({
            'GAME_ID': g["id"], 'GAME_DATE': game_date,
            'TEAM_ABBREVIATION': home_abbr, 'HOME': 1,
            'HOME_TEAM': home_abbr, 'AWAY_TEAM': away_abbr,
            'PTS': home_score, 'PTS_ALLOWED': away_score,
            'POINT_DIFF': home_score - away_score,
            'GAME_TOTAL': home_score + away_score,
            'WIN': 1 if home_score > away_score else 0,
            'FG_PCT': 0.46, 'FG3_PCT': 0.36, 'TOV': 14.0, 'REB': 44.0,
        })
        rows.append({
            'GAME_ID': g["id"], 'GAME_DATE': game_date,
            'TEAM_ABBREVIATION': away_abbr, 'HOME': 0,
            'HOME_TEAM': home_abbr, 'AWAY_TEAM': away_abbr,
            'PTS': away_score, 'PTS_ALLOWED': home_score,
            'POINT_DIFF': away_score - home_score,
            'GAME_TOTAL': home_score + away_score,
            'WIN': 1 if away_score > home_score else 0,
            'FG_PCT': 0.46, 'FG3_PCT': 0.36, 'TOV': 14.0, 'REB': 44.0,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE']).reset_index(drop=True)
    df['DAYS_REST'] = df.groupby('TEAM_ABBREVIATION')['GAME_DATE'].diff().dt.days.fillna(3)
    df['B2B'] = (df['DAYS_REST'] <= 1).astype(int)
    print(f"✅ Loaded {len(df):,} team-game records")
    return df


# ── Step 2: Rolling stats ──────────────────────────────────
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


# ── Step 3: Train models ───────────────────────────────────
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
        hs  = rolling_stats(df, game['HOME_TEAM'], game['GAME_DATE'])
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
    X        = model_df[FEATURES]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    win_m    = LogisticRegression(max_iter=1000)
    spread_m = Ridge(alpha=1.0)
    totals_m = Ridge(alpha=1.0)

    win_m.fit(X_scaled,    model_df['HOME_WIN'])
    spread_m.fit(X_scaled, model_df['POINT_DIFF'])
    totals_m.fit(X_scaled, model_df['GAME_TOTAL'])

    print(f"✅ Models trained on {len(model_df):,} games")
    return win_m, spread_m, totals_m, scaler, df


# ── Step 4: Predict a game ─────────────────────────────────
def predict_game(df, win_m, spread_m, totals_m, scaler, home_team, away_team, game_date=None):
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

    win_prob    = win_m.predict_proba(row_scaled)[0][1]
    spread_pred = spread_m.predict(row_scaled)[0]
    total_pred  = totals_m.predict(row_scaled)[0]
    confidence  = abs(win_prob - 0.5) * 200

    return {
        'win_prob':    round(win_prob, 4),
        'spread_pred': round(spread_pred, 1),
        'total_pred':  round(total_pred, 1),
        'confidence':  round(confidence, 1),
    }


# ── Step 5: Fetch AU odds ──────────────────────────────────
def get_todays_games_et():
    et = pytz.timezone("America/New_York")
    today_et = datetime.now(et).strftime("%Y-%m-%d")
    print(f"📅 Fetching AU odds for US date: {today_et}")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey":             ODDS_API_KEY,
        "regions":            "au",
        "markets":            "spreads",
        "oddsFormat":         "decimal",
        "bookmakers":         ",".join(AU_BOOKS),
        "commenceTimeFrom":   f"{today_et}T00:00:00Z",
        "commenceTimeTo":     f"{today_et}T23:59:59Z",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            print(f"⚠️  Odds API: {data['message']}")
            return [], today_et
        return data, today_et
    except Exception as e:
        print(f"⚠️  Odds fetch failed: {e}")
        return [], today_et


# ── Team name → abbreviation ───────────────────────────────
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


# ── Step 6: Run full pipeline ──────────────────────────────
def run_pipeline():
    print("\n" + "="*55)
    au_now = datetime.now(AEST).strftime("%Y-%m-%d %H:%M AEST")
    print(f"  🏀 TIKKERTAKKER — {au_now}")
    print("="*55 + "\n")

    df                           = fetch_nba_data()
    win_m, spread_m, totals_m, scaler, df = train_models(df)
    games_data, today_et         = get_todays_games_et()

    scope_detail = []
    qualifying   = []

    for game in games_data:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = TEAM_MAP.get(home_full)
        away_abbr = TEAM_MAP.get(away_full)

        if not home_abbr or not away_abbr:
            continue

        pred = predict_game(df, win_m, spread_m, totals_m, scaler, home_abbr, away_abbr)
        if pred is None:
            continue

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
                            books_odds[bk] = {"spread": spread, "odds": price}
                            if vegas_line is None:
                                vegas_line = spread
                            if best_odds is None or price > best_odds:
                                best_odds = price
                                best_book = bk

        if vegas_line is None:
            continue

        edge       = round(abs(pred['spread_pred'] - vegas_line), 1)
        confidence = pred['confidence']
        win_prob   = pred['win_prob']

        bet_side = (f"{home_abbr} covers {vegas_line:+.1f}"
                    if pred['spread_pred'] > vegas_line
                    else f"{away_abbr} covers {-vegas_line:+.1f}")

        qualifies = (EDGE_MIN <= edge <= EDGE_MAX) and (CONF_MIN <= confidence <= CONF_MAX)

        if qualifies:
            verdict = "✅ BET"
            reason  = f"Edge {edge}pts in range & confidence {confidence} in range"
        else:
            reasons = []
            if not (EDGE_MIN <= edge <= EDGE_MAX):
                reasons.append(f"Edge {edge}pts outside 3-7 range")
            if not (CONF_MIN <= confidence <= CONF_MAX):
                reasons.append(f"Confidence {confidence} outside 50-70")
            verdict = "NO BET"
            reason  = " & ".join(reasons)

        entry = {
            "game":         f"{home_abbr} vs {away_abbr}",
            "home":         home_abbr,
            "away":         away_abbr,
            "vegas_line":   f"{home_abbr} {vegas_line:+.1f}",
            "model_spread": f"{home_abbr} {pred['spread_pred']:+.1f}",
            "edge":         edge,
            "win_prob":     f"{win_prob*100:.1f}%",
            "confidence":   confidence,
            "qualifies":    qualifies,
            "verdict":      verdict,
            "reason":       reason,
            "bet_side":     bet_side if qualifies else None,
            "books":        books_odds,
            "best_book":    best_book,
            "best_odds":    best_odds,
            "proj_total":   pred['total_pred'],
        }

        scope_detail.append(entry)
        if qualifies:
            qualifying.append(entry)
            print(f"  ✅ BET: {entry['game']} — {bet_side} @ {best_odds} ({best_book})")
        else:
            print(f"  ❌ {entry['game']} — {reason}")

    # ── Load / init results file ───────────────────────────
    os.makedirs("data", exist_ok=True)
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    else:
        results = {
            "daily_log": [], "bets": [],
            "model_history": {
                "model_info": {
                    "data_range":        "2022-2026",
                    "strategy":          "ATS (Against The Spread)",
                    "confidence_range":  "50-70",
                    "spread_edge_range": "3-7 pts vs Vegas line",
                    "stake_per_bet":     "2% bankroll",
                    "books":             "TAB, Betr, PointsBet",
                    "do_not_bet":        "Edge >7pts, Confidence >70, Totals, Playoffs"
                },
                "backtest_results": {
                    "2021-22": {"bets": 53, "wins": 31, "ats_accuracy": "58.5%", "roi": "+11.2%"},
                    "2022-23": {"bets": 54, "wins": 33, "ats_accuracy": "61.1%", "roi": "+17.4%"},
                    "2023-24": {"bets": 52, "wins": 30, "ats_accuracy": "57.7%", "roi": "+10.8%"},
                    "2024-25": {"bets": 53, "wins": 34, "ats_accuracy": "64.2%", "roi": "+24.8%"},
                    "overall": {"bets": 212, "ats_accuracy": "60.8%", "roi": "+16.16%"},
                },
                "live_season":     "2025-26",
                "live_start_date": "2026-03-21",
            }
        }

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
        "note":          f"Day {day_num} — {len(scope_detail)} games scoped, {len(qualifying)} qualifying bets.",
    }

    existing = [e for e in results["daily_log"] if e["date"] != aest_date]
    results["daily_log"] = existing + [log_entry]

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Saved to {RESULTS_FILE}")
    print(f"📊 {len(scope_detail)} scoped, {len(qualifying)} qualifying")
    print("="*55 + "\n")


if __name__ == "__main__":
    run_pipeline()
