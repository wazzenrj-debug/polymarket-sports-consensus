# Polymarket Sports Consensus Dashboard 🎯

**Live Dashboard:** [View Here](https://YOUR_USERNAME.github.io/polymarket-sports-consensus/)

> See where the **top 200 most profitable Polymarket traders** are placing their sports bets.

![Dashboard Preview](https://img.shields.io/badge/Updates-Every%206%20Hours-green)

## What This Does

1. **Pulls leaderboard** - Gets top 200 traders by profit (last 30 days)
2. **Scans positions** - Finds their sports bets ≥$100 on events in next 7 days
3. **Aggregates consensus** - Shows which outcomes the smart money favors
4. **Auto-updates** - Refreshes every 6 hours via GitHub Actions

## Example Output

```
Lakers vs Celtics
├── Lakers to win: 78% of traders (23 traders, $125,000)
└── Celtics to win: 22% of traders (6 traders, $34,000)
[Strong Consensus] [View on Polymarket →]
```

## How It Works

- GitHub Actions runs `polymarket_sports_consensus.py` every 6 hours
- Generates a fresh `index.html` dashboard
- Publishes to GitHub Pages automatically

## Run Locally

```bash
pip install httpx[http2]
python polymarket_sports_consensus.py
```

## Configuration

Edit `CONFIG` in the Python script:

```python
CONFIG = {
    "leaderboard_window": "30d",      # Time window for profit ranking
    "leaderboard_limit": 200,          # Number of top traders
    "min_position_size": 100,          # Minimum $USD position
    "min_traders_per_match": 5,        # Minimum traders to show match
    "days_ahead": 7,                   # How far to look ahead
}
```

## Disclaimer

This is for informational purposes only. Not financial advice. Do your own research.

---

Built for tracking smart money on Polymarket sports markets.
