from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NBA Stats Analyzer", layout="wide")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_stats(season: str, season_type: str, per_mode: str) -> pd.DataFrame:
    """Fetch league-wide player stats from NBA Stats API."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
    )
    return response.get_data_frames()[0]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_stats(season: str, season_type: str, per_mode: str) -> pd.DataFrame:
    """Fetch league-wide team stats from NBA Stats API."""
    from nba_api.stats.endpoints import leaguedashteamstats

    response = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
    )
    return response.get_data_frames()[0]


def infer_current_season() -> str:
    today = dt.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year}"


st.title("🏀 NBA Stats Analyzer")
st.caption("Pull live league data from NBA.com via `nba_api` and explore key trends.")

with st.sidebar:
    st.header("Filters")
    season = st.text_input("Season (YYYY-YY)", infer_current_season())
    season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"])
    per_mode = st.selectbox(
        "Per Mode",
        ["PerGame", "Totals", "Per36", "Per48", "Per100Possessions"],
        index=0,
    )
    top_n = st.slider("Top N players by selected metric", 5, 50, 15)

st.subheader("League Data")

try:
    player_df = fetch_player_stats(season, season_type, per_mode)
    team_df = fetch_team_stats(season, season_type, per_mode)
except Exception as exc:  # noqa: BLE001
    st.error(
        "Could not load data from NBA stats API. Ensure `nba_api` is installed and try again."
    )
    st.exception(exc)
    st.stop()

numeric_cols = [col for col in player_df.columns if pd.api.types.is_numeric_dtype(player_df[col])]
default_metric = "PTS" if "PTS" in numeric_cols else numeric_cols[0]
metric = st.selectbox("Analysis metric", numeric_cols, index=numeric_cols.index(default_metric))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players", f"{player_df.shape[0]:,}")
col2.metric("Teams", f"{team_df.shape[0]:,}")
col3.metric("Avg player PTS", f"{player_df['PTS'].mean():.1f}" if "PTS" in player_df else "N/A")
col4.metric("Avg team PTS", f"{team_df['PTS'].mean():.1f}" if "PTS" in team_df else "N/A")

st.subheader(f"Top {top_n} Players by {metric}")
top_players = (
    player_df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", metric]]
    .sort_values(metric, ascending=False)
    .head(top_n)
)
st.bar_chart(top_players.set_index("PLAYER_NAME")[metric], use_container_width=True)
st.dataframe(top_players, use_container_width=True)

st.subheader("Shot Creation vs Scoring")
if all(col in player_df.columns for col in ["PTS", "AST", "USG_PCT", "PLAYER_NAME"]):
    scatter = player_df[["PLAYER_NAME", "USG_PCT", "PTS", "AST"]].copy()
    st.scatter_chart(scatter, x="USG_PCT", y="PTS", size="AST", color=None)
    st.caption("X=Usage %, Y=Points, bubble size=Assists")
else:
    st.info("Required columns for scatter plot are not available in this dataset.")

st.subheader("Team Efficiency Snapshot")
if all(col in team_df.columns for col in ["TEAM_NAME", "OFF_RATING", "DEF_RATING", "NET_RATING"]):
    efficiency = team_df[["TEAM_NAME", "W", "L", "OFF_RATING", "DEF_RATING", "NET_RATING"]].sort_values(
        "NET_RATING", ascending=False
    )
    st.dataframe(efficiency, use_container_width=True)
else:
    st.dataframe(team_df.head(20), use_container_width=True)

with st.expander("Raw data"):
    st.write("Player stats")
    st.dataframe(player_df, use_container_width=True)
    st.write("Team stats")
    st.dataframe(team_df, use_container_width=True)
