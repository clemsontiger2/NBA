from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NBA Stats Analyzer", layout="wide", page_icon="\U0001f3c0")

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stat-leader-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .stat-leader-card h3 {
        color: #e94560;
        margin: 0 0 4px 0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stat-leader-card .value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 4px 0;
    }
    .stat-leader-card .player {
        color: #a8a8b3;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 12px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data-fetching helpers (cached 1 hour)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_stats(season: str, season_type: str, per_mode: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerstats

    resp = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
    )
    return resp.get_data_frames()[0]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_team_stats(season: str, season_type: str, per_mode: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashteamstats

    resp = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
    )
    return resp.get_data_frames()[0]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_career(player_id: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import playercareerstats

    resp = playercareerstats.PlayerCareerStats(player_id=str(player_id))
    return resp.get_data_frames()[0]


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_player_game_log(player_id: int, season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelog

    resp = playergamelog.PlayerGameLog(player_id=str(player_id), season=season)
    return resp.get_data_frames()[0]


def infer_current_season() -> str:
    today = dt.date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    end_year = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year}"


def leader_card(label: str, player: str, value: str) -> str:
    return (
        f'<div class="stat-leader-card">'
        f'<h3>{label}</h3>'
        f'<div class="value">{value}</div>'
        f'<div class="player">{player}</div>'
        f"</div>"
    )


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize to 0-100 for radar charts."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([50.0] * len(s), index=s.index)
    return ((s - mn) / (mx - mn) * 100).round(1)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("\U0001f3c0 NBA Stats Analyzer")
st.caption("Live league data from NBA.com \u2014 explore stats, compare players, and analyze teams.")

with st.sidebar:
    st.header("\u2699\ufe0f Filters")
    season = st.text_input("Season (YYYY-YY)", infer_current_season())
    season_type = st.selectbox("Season Type", ["Regular Season", "Playoffs"])
    per_mode = st.selectbox(
        "Per Mode",
        ["PerGame", "Totals", "Per36", "Per48", "Per100Possessions"],
        index=0,
    )
    min_gp = st.slider("Minimum Games Played", 0, 82, 10)
    top_n = st.slider("Top N players", 5, 50, 15)

    st.divider()
    st.markdown(
        "**Data source:** [nba_api](https://github.com/swar/nba_api) \u2022 "
        "Cached for 1 hour"
    )

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
with st.spinner("Loading NBA data\u2026"):
    try:
        player_df = fetch_player_stats(season, season_type, per_mode)
        team_df = fetch_team_stats(season, season_type, per_mode)
    except Exception as exc:
        st.error("Could not load data from the NBA stats API. Check your connection and `nba_api` installation.")
        st.exception(exc)
        st.stop()

# Apply minimum-games filter
filtered_df = player_df[player_df["GP"] >= min_gp].copy()

numeric_cols = [c for c in filtered_df.columns if pd.api.types.is_numeric_dtype(filtered_df[c])]

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_players, tab_teams, tab_h2h = st.tabs(
    ["\U0001f4ca League Overview", "\U0001f50d Player Explorer", "\U0001f3c6 Team Analysis", "\u2694\ufe0f Head-to-Head"]
)

# ========================== TAB 1 – LEAGUE OVERVIEW ========================
with tab_overview:

    # --- KPI row ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Players", f"{filtered_df.shape[0]:,}")
    k2.metric("Teams", f"{team_df.shape[0]:,}")
    k3.metric(
        "Avg PTS",
        f"{filtered_df['PTS'].mean():.1f}" if "PTS" in filtered_df else "N/A",
    )
    k4.metric(
        "Avg REB",
        f"{filtered_df['REB'].mean():.1f}" if "REB" in filtered_df else "N/A",
    )

    st.divider()

    # --- Statistical leaders ---
    st.subheader("Statistical Leaders")
    leader_cats = {
        "Points": "PTS",
        "Rebounds": "REB",
        "Assists": "AST",
        "Steals": "STL",
        "Blocks": "BLK",
        "3-Pointers": "FG3M",
    }

    cols = st.columns(len(leader_cats))
    for col, (label, stat) in zip(cols, leader_cats.items()):
        if stat in filtered_df.columns and not filtered_df.empty:
            idx = filtered_df[stat].idxmax()
            row = filtered_df.loc[idx]
            col.markdown(
                leader_card(label, row["PLAYER_NAME"], f"{row[stat]:.1f}"),
                unsafe_allow_html=True,
            )

    st.divider()

    # --- Top N players chart ---
    st.subheader(f"Top {top_n} Players")
    default_metric = "PTS" if "PTS" in numeric_cols else numeric_cols[0]
    metric = st.selectbox("Metric", numeric_cols, index=numeric_cols.index(default_metric), key="overview_metric")

    top_players = (
        filtered_df[["PLAYER_NAME", "TEAM_ABBREVIATION", "GP", metric]]
        .sort_values(metric, ascending=False)
        .head(top_n)
    )

    fig_bar = px.bar(
        top_players,
        x=metric,
        y="PLAYER_NAME",
        orientation="h",
        color=metric,
        color_continuous_scale="Sunset",
        hover_data=["TEAM_ABBREVIATION", "GP"],
        text=metric,
    )
    fig_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, top_n * 28),
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_showscale=False,
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Shot creation vs scoring ---
    st.subheader("Shot Creation vs Scoring")
    if all(c in filtered_df.columns for c in ["PTS", "AST", "USG_PCT", "PLAYER_NAME", "TEAM_ABBREVIATION"]):
        fig_scatter = px.scatter(
            filtered_df,
            x="USG_PCT",
            y="PTS",
            size="AST",
            color="TEAM_ABBREVIATION",
            hover_name="PLAYER_NAME",
            hover_data={"USG_PCT": ":.1f", "PTS": ":.1f", "AST": ":.1f"},
            labels={"USG_PCT": "Usage %", "PTS": "Points", "AST": "Assists"},
            size_max=18,
        )
        fig_scatter.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Required columns for the scatter plot are not available in this mode.")

    # --- Distribution ---
    st.subheader("Scoring Distribution")
    if "PTS" in filtered_df.columns:
        fig_hist = px.histogram(
            filtered_df,
            x="PTS",
            nbins=30,
            color_discrete_sequence=["#e94560"],
            labels={"PTS": "Points"},
        )
        fig_hist.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=10, b=0),
            bargap=0.05,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Raw player data"):
        st.dataframe(filtered_df, use_container_width=True)


# ========================== TAB 2 – PLAYER EXPLORER ========================
with tab_players:
    st.subheader("Player Explorer")

    if filtered_df.empty:
        st.warning("No players match the current filters.")
    else:
        player_names = sorted(filtered_df["PLAYER_NAME"].unique())
        selected_player = st.selectbox("Search for a player", player_names, index=0, key="player_search")

        p_row = filtered_df[filtered_df["PLAYER_NAME"] == selected_player].iloc[0]
        player_id = int(p_row["PLAYER_ID"])

        # --- Profile header ---
        st.markdown(f"### {selected_player} \u2014 {p_row.get('TEAM_ABBREVIATION', 'N/A')}")

        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        stat_labels = [("GP", "Games"), ("PTS", "Points"), ("REB", "Rebounds"),
                       ("AST", "Assists"), ("STL", "Steals"), ("BLK", "Blocks")]
        for col, (stat, lbl) in zip([pc1, pc2, pc3, pc4, pc5, pc6], stat_labels):
            val = p_row.get(stat, "N/A")
            col.metric(lbl, f"{val:.1f}" if isinstance(val, (int, float)) else val)

        st.divider()

        # --- Shooting splits ---
        st.subheader("Shooting Splits")
        shoot_cols_map = {
            "FG%": "FG_PCT",
            "3P%": "FG3_PCT",
            "FT%": "FT_PCT",
            "eFG%": "EFG_PCT",
            "TS%": "TS_PCT",
        }
        available_shoot = {k: v for k, v in shoot_cols_map.items() if v in filtered_df.columns}
        if available_shoot:
            sc = st.columns(len(available_shoot))
            for col, (lbl, stat) in zip(sc, available_shoot.items()):
                val = p_row.get(stat, 0)
                col.metric(lbl, f"{val * 100:.1f}%" if val and val <= 1 else f"{val:.1f}%")

        st.divider()

        # --- Game log ---
        st.subheader(f"Game Log ({season})")
        try:
            game_log = fetch_player_game_log(player_id, season)
            if game_log.empty:
                st.info("No game log data available for this season.")
            else:
                display_cols = [c for c in ["GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"] if c in game_log.columns]
                st.dataframe(game_log[display_cols], use_container_width=True)

                # Points trend line
                if "PTS" in game_log.columns and "GAME_DATE" in game_log.columns:
                    gl = game_log[["GAME_DATE", "PTS"]].copy()
                    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
                    gl = gl.sort_values("GAME_DATE")
                    fig_gl = px.line(
                        gl,
                        x="GAME_DATE",
                        y="PTS",
                        markers=True,
                        labels={"GAME_DATE": "Date", "PTS": "Points"},
                        color_discrete_sequence=["#e94560"],
                    )
                    fig_gl.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_gl, use_container_width=True)
        except Exception:
            st.info("Game log data is not available for this player/season.")

        st.divider()

        # --- Career stats ---
        st.subheader("Career Stats")
        try:
            career = fetch_player_career(player_id)
            if career.empty:
                st.info("No career data available.")
            else:
                career_display = [c for c in ["SEASON_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"] if c in career.columns]
                st.dataframe(career[career_display], use_container_width=True)

                # Career scoring trend
                if "PTS" in career.columns and "SEASON_ID" in career.columns:
                    fig_career = px.line(
                        career,
                        x="SEASON_ID",
                        y="PTS",
                        markers=True,
                        labels={"SEASON_ID": "Season", "PTS": "Points"},
                        color_discrete_sequence=["#0f3460"],
                    )
                    fig_career.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_career, use_container_width=True)
        except Exception:
            st.info("Career data is not available for this player.")


# ========================== TAB 3 – TEAM ANALYSIS =========================
with tab_teams:
    st.subheader("Team Analysis")

    # --- Team efficiency table ---
    eff_cols = ["TEAM_NAME", "W", "L", "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING"]
    if all(c in team_df.columns for c in eff_cols):
        efficiency = team_df[eff_cols].sort_values("NET_RATING", ascending=False).reset_index(drop=True)
        efficiency.index = efficiency.index + 1  # 1-based rank

        st.dataframe(
            efficiency.style.background_gradient(subset=["NET_RATING"], cmap="RdYlGn")
            .background_gradient(subset=["OFF_RATING"], cmap="YlOrRd")
            .background_gradient(subset=["DEF_RATING"], cmap="YlOrRd_r")
            .format({"W_PCT": "{:.3f}", "OFF_RATING": "{:.1f}", "DEF_RATING": "{:.1f}", "NET_RATING": "{:.1f}"}),
            use_container_width=True,
        )

        st.divider()

        # --- Off vs Def rating scatter ---
        st.subheader("Offensive vs Defensive Rating")
        fig_team = px.scatter(
            team_df,
            x="OFF_RATING",
            y="DEF_RATING",
            text="TEAM_NAME",
            hover_name="TEAM_NAME",
            hover_data={"NET_RATING": ":.1f", "W": True, "L": True},
            color="NET_RATING",
            color_continuous_scale="RdYlGn",
            labels={"OFF_RATING": "Offensive Rating", "DEF_RATING": "Defensive Rating (lower is better)"},
        )
        fig_team.update_traces(textposition="top center", textfont_size=9)
        fig_team.update_layout(height=600, margin=dict(l=0, r=0, t=10, b=0))
        # Flip y-axis so lower (better) DEF_RATING is at top
        fig_team.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_team, use_container_width=True)
        st.caption("Top-right quadrant = best teams (high offense, low defense).")
    else:
        st.dataframe(team_df, use_container_width=True)

    st.divider()

    # --- Team comparison ---
    st.subheader("Compare Two Teams")
    team_names = sorted(team_df["TEAM_NAME"].unique()) if "TEAM_NAME" in team_df.columns else []
    if len(team_names) >= 2:
        tc1, tc2 = st.columns(2)
        team_a = tc1.selectbox("Team A", team_names, index=0, key="team_a")
        team_b = tc2.selectbox("Team B", team_names, index=min(1, len(team_names) - 1), key="team_b")

        compare_stats = ["W", "L", "W_PCT", "PTS", "REB", "AST", "STL", "BLK", "OFF_RATING", "DEF_RATING", "NET_RATING"]
        available_stats = [s for s in compare_stats if s in team_df.columns]

        row_a = team_df[team_df["TEAM_NAME"] == team_a].iloc[0]
        row_b = team_df[team_df["TEAM_NAME"] == team_b].iloc[0]

        comparison = pd.DataFrame({
            "Stat": available_stats,
            team_a: [row_a[s] for s in available_stats],
            team_b: [row_b[s] for s in available_stats],
        })
        st.dataframe(comparison.set_index("Stat"), use_container_width=True)

        # Grouped bar chart
        radar_stats = [s for s in ["PTS", "REB", "AST", "STL", "BLK"] if s in team_df.columns]
        if radar_stats:
            comp_data = pd.DataFrame({
                "Stat": radar_stats * 2,
                "Value": [row_a[s] for s in radar_stats] + [row_b[s] for s in radar_stats],
                "Team": [team_a] * len(radar_stats) + [team_b] * len(radar_stats),
            })
            fig_comp = px.bar(
                comp_data,
                x="Stat",
                y="Value",
                color="Team",
                barmode="group",
                color_discrete_sequence=["#e94560", "#0f3460"],
            )
            fig_comp.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("Full team data"):
        st.dataframe(team_df, use_container_width=True)


# ========================== TAB 4 – HEAD-TO-HEAD ==========================
with tab_h2h:
    st.subheader("Player Head-to-Head Comparison")

    if filtered_df.empty:
        st.warning("No players match the current filters.")
    else:
        player_names_h2h = sorted(filtered_df["PLAYER_NAME"].unique())

        h1, h2 = st.columns(2)
        player_a = h1.selectbox("Player A", player_names_h2h, index=0, key="h2h_a")
        default_b = min(1, len(player_names_h2h) - 1)
        player_b = h2.selectbox("Player B", player_names_h2h, index=default_b, key="h2h_b")

        row_a = filtered_df[filtered_df["PLAYER_NAME"] == player_a].iloc[0]
        row_b = filtered_df[filtered_df["PLAYER_NAME"] == player_b].iloc[0]

        # --- Side-by-side metrics ---
        compare_stats = ["GP", "PTS", "REB", "AST", "STL", "BLK", "FG_PCT", "FG3_PCT", "FT_PCT"]
        available = [s for s in compare_stats if s in filtered_df.columns]

        comp_df = pd.DataFrame({
            "Stat": available,
            player_a: [row_a[s] for s in available],
            player_b: [row_b[s] for s in available],
        })

        def highlight_winner(row):
            """Bold the higher value in each stat row."""
            a_val = row[player_a]
            b_val = row[player_b]
            if pd.isna(a_val) or pd.isna(b_val):
                return ["", "", ""]
            if a_val > b_val:
                return ["", "background-color: #d4edda", ""]
            elif b_val > a_val:
                return ["", "", "background-color: #d4edda"]
            return ["", "", ""]

        st.dataframe(
            comp_df.set_index("Stat").style.apply(highlight_winner, axis=1),
            use_container_width=True,
        )

        st.divider()

        # --- Radar chart ---
        st.subheader("Skill Comparison Radar")
        radar_cats = [s for s in ["PTS", "REB", "AST", "STL", "BLK"] if s in filtered_df.columns]

        if radar_cats:
            # Normalize across the filtered dataset so the radar is meaningful
            norm_data = {}
            for cat in radar_cats:
                norm_data[cat] = normalize_series(filtered_df[cat])

            vals_a = [norm_data[cat].loc[row_a.name] for cat in radar_cats]
            vals_b = [norm_data[cat].loc[row_b.name] for cat in radar_cats]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_a + [vals_a[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself",
                name=player_a,
                line_color="#e94560",
                fillcolor="rgba(233, 69, 96, 0.25)",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_b + [vals_b[0]],
                theta=radar_cats + [radar_cats[0]],
                fill="toself",
                name=player_b,
                line_color="#0f3460",
                fillcolor="rgba(15, 52, 96, 0.25)",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("Values are percentile-normalized across all qualifying players (0\u2013100 scale).")

        st.divider()

        # --- Per-game bar comparison ---
        st.subheader("Stat-by-Stat Comparison")
        bar_stats = [s for s in ["PTS", "REB", "AST", "STL", "BLK", "FG3M", "TOV"] if s in filtered_df.columns]
        if bar_stats:
            bar_data = pd.DataFrame({
                "Stat": bar_stats * 2,
                "Value": [row_a[s] for s in bar_stats] + [row_b[s] for s in bar_stats],
                "Player": [player_a] * len(bar_stats) + [player_b] * len(bar_stats),
            })
            fig_h2h = px.bar(
                bar_data,
                x="Stat",
                y="Value",
                color="Player",
                barmode="group",
                color_discrete_sequence=["#e94560", "#0f3460"],
                text="Value",
            )
            fig_h2h.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_h2h.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_h2h, use_container_width=True)
