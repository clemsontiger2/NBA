# 🏀 NBA Stats Analyzer

A Streamlit website that uses the official NBA stats API endpoints (via [`nba_api`](https://github.com/swar/nba_api)) to pull live NBA.com data and run interactive analysis.

## Features

- Pulls player and team stats for any NBA season.
- Supports Regular Season and Playoffs data.
- Allows multiple per-mode views (Per Game, Totals, Per 36, etc.).
- Visualizes:
  - Top players by a selected metric.
  - Usage vs scoring bubble chart.
  - Team offense/defense/net rating table.
- Includes raw data tables for deeper inspection.

## Run locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

3. Open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Notes

- Data is fetched from NBA.com stats endpoints through `nba_api`.
- Requests are cached for one hour in Streamlit to speed up reloads.
