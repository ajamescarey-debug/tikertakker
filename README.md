# 🏀 TIKKERTAKKER — NBA ATS Model

Automated NBA Against-The-Spread intelligence system. Runs daily via GitHub Actions, publishes to GitHub Pages.

## How it works

1. Pulls NBA game data (2021–2026) via `nba_api`
2. Trains 3 models: win probability, spread prediction, totals prediction
3. Fetches live AU odds from TAB, Betr and PointsBet via The Odds API
4. Identifies qualifying bets: **edge 3–7pts AND confidence 50–70%**
5. Shows best odds across all three books
6. Updates dashboard automatically at 6:15am AEST daily

## Qualifying criteria

| Criteria | Range |
|----------|-------|
| Edge (model vs Vegas) | 3–7 points |
| Confidence | 50–70 / 100 |
| Both conditions | Required simultaneously |

## Setup

### 1. Add secret
Go to repo **Settings → Secrets → Actions → New repository secret**
- Name: `ODDS_API_KEY`
- Value: your key from [the-odds-api.com](https://the-odds-api.com)

### 2. Enable GitHub Pages
Go to repo **Settings → Pages**
- Source: **Deploy from a branch**
- Branch: `main` / root

### 3. Run manually first
Go to **Actions → TikkerTakker Daily Pipeline → Run workflow**

## Dashboard
Live at: `https://[your-username].github.io/tikkertakker`

## Do NOT bet
- Edge > 7pts
- Confidence > 70
- Totals / over-under
- Playoffs
- B2B situation alone (must meet both criteria)

---
*Statistical modelling tool — not financial advice. Gamble responsibly. 18+*
