"""NRFI/YRFI dashboard - reads only from data/nrfi.db (never the large raw/
processed parquet files, those are a build-time cache refreshed by
src/daily_pipeline.py). Three views: today's slate, track record (historical
backtest + the live day-by-day record since this system launched), and a
team/pitcher trend explorer.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import db  # noqa: E402

st.set_page_config(page_title="NRFI Model", layout="wide")


@st.cache_data(ttl=600)
def load_table(name: str) -> pd.DataFrame:
    return db.read_table(name)


@st.cache_data(ttl=600)
def load_live_track_record() -> pd.DataFrame:
    return db.read_live_track_record()


def tier_stats(df: pd.DataFrame, threshold: float) -> tuple[int, float | None]:
    """df must have p_nrfi, p_yrfi, actual_nrfi columns. Returns (n_games, success_rate)."""
    pick_prob = df[["p_nrfi", "p_yrfi"]].max(axis=1)
    confident = df[pick_prob >= threshold]
    if confident.empty:
        return 0, None
    confident_is_nrfi_pick = confident["p_nrfi"] >= 0.5
    correct = (confident_is_nrfi_pick & (confident["actual_nrfi"] == 1)) | (
        ~confident_is_nrfi_pick & (confident["actual_nrfi"] == 0)
    )
    return len(confident), correct.mean()


def render_slate() -> None:
    predictions = load_table("predictions")
    if predictions.empty:
        st.info("No predictions in the database yet. Run `python src/daily_pipeline.py` to generate today's slate.")
        return

    dates_available = sorted(predictions["game_date"].unique(), reverse=True)
    selected_date = st.selectbox("Slate date", dates_available, index=0)

    slate = predictions[predictions["game_date"] == selected_date].sort_values("p_nrfi", ascending=False)
    features = load_table("game_features")

    display = slate[[
        "matchup", "away_pitcher_name", "home_pitcher_name",
        "p_nrfi", "p_yrfi", "pick", "confidence_tier",
    ]].rename(columns={
        "away_pitcher_name": "Away SP", "home_pitcher_name": "Home SP",
        "p_nrfi": "P(NRFI)", "p_yrfi": "P(YRFI)", "pick": "Pick", "confidence_tier": "Confidence",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.caption("Confidence: \"53%+\"/\"55%+\" mark games where the model's pick cleared that probability threshold "
               "(the levels validated as profitable in backtesting).")

    st.subheader("Game details")
    for _, game in slate.iterrows():
        with st.expander(f"{game['matchup']} — P(NRFI) {game['p_nrfi']:.1%}"):
            game_feats = features[features["game_pk"] == game["game_pk"]]
            away_feat = game_feats[game_feats["side"] == "away"]
            home_feat = game_feats[game_feats["side"] == "home"]
            col1, col2 = st.columns(2)
            for col, feat, label in [(col1, away_feat, "Away"), (col2, home_feat, "Home")]:
                with col:
                    st.markdown(f"**{label} batting**")
                    if feat.empty:
                        st.write("no feature detail stored")
                        continue
                    f = feat.iloc[0]
                    st.write(f"Slot 1-4 rolling OPS: {f['batting_team_slot1_4_ops']:.3f}")
                    st.write(f"Team's own inning-1 scoring rate: {f['batting_team_own_inning1_rate']:.1%}")
                    st.write(f"Opposing starter ({f['opposing_starter_name']}):")
                    st.write(f"  - 1st-inning baserunner rate: {f['opposing_starter_inning1_rate']:.1%}")
                    st.write(f"  - overall (full-game) baserunner rate: {f['opposing_starter_overall_rate']:.1%}")
                    st.write(f"Park factor (home park): {f['park_factor']:.0f}")


def render_track_record() -> None:
    st.subheader("Historical backtest")
    st.caption(
        "Held-out, walk-forward evaluation - the model used here never saw these games during training. "
        "This is the honest source of truth for \"how accurate has this actually been.\""
    )
    backtest = load_table("backtest_results")
    if backtest.empty:
        st.info("No backtest data yet. Run `python src/daily_pipeline.py` at least once.")
    else:
        backtest["game_date"] = pd.to_datetime(backtest["game_date"])
        max_date = backtest["game_date"].max()

        window = st.selectbox(
            "Window", ["Last 2 weeks", "Last month", "Last 2 months", "Full backtest"], index=1
        )
        if window == "Last 2 weeks":
            cutoff = max_date - pd.Timedelta(days=14)
        elif window == "Last month":
            cutoff = max_date - pd.Timedelta(days=30)
        elif window == "Last 2 months":
            cutoff = max_date - pd.Timedelta(days=60)
        else:
            cutoff = backtest["game_date"].min()

        windowed = backtest[backtest["game_date"] >= cutoff]
        is_nrfi_pick = windowed["p_nrfi"] >= 0.5
        overall_correct = (is_nrfi_pick & (windowed["actual_nrfi"] == 1)) | (
            ~is_nrfi_pick & (windowed["actual_nrfi"] == 0)
        )

        n_53, rate_53 = tier_stats(windowed, 0.53)
        n_55, rate_55 = tier_stats(windowed, 0.55)

        c1, c2, c3 = st.columns(3)
        c1.metric("All games", f"{overall_correct.mean():.1%}", help=f"{len(windowed)} games")
        c2.metric("53%+ confidence picks", f"{rate_53:.1%}" if rate_53 is not None else "n/a", help=f"{n_53} games")
        c3.metric("55%+ confidence picks", f"{rate_55:.1%}" if rate_55 is not None else "n/a", help=f"{n_55} games")

    st.divider()
    st.subheader("Live track record")
    st.caption(
        "The actual day-by-day picks this deployed system made, graded once games finished. "
        "Starts small and grows from the day this system launched - zero lookahead by construction."
    )
    live = load_live_track_record()
    if live.empty:
        st.info("No graded predictions yet - check back once today's (or earlier) games have finished.")
    else:
        live_is_nrfi_pick = live["p_nrfi"] >= 0.5
        live_correct = (live_is_nrfi_pick & (live["actual_nrfi"] == 1)) | (~live_is_nrfi_pick & (live["actual_nrfi"] == 0))
        st.metric("Overall success rate", f"{live_correct.mean():.1%}", help=f"{len(live)} graded games")
        st.dataframe(
            live[["game_date", "matchup", "p_nrfi", "p_yrfi", "pick", "confidence_tier", "correct"]]
            .sort_values("game_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_explorer() -> None:
    predictions = load_table("predictions")
    features = load_table("game_features")
    if features.empty:
        st.info("No feature history yet - this builds up day by day as the pipeline runs.")
        return

    features = features.merge(predictions[["game_pk", "game_date"]], on="game_pk", how="left")
    features["game_date"] = pd.to_datetime(features["game_date"])

    mode = st.radio("Explore by", ["Team", "Starting pitcher"], horizontal=True)

    if mode == "Team":
        teams = sorted(features["batting_team"].dropna().unique())
        team = st.selectbox("Team", teams)
        team_hist = features[features["batting_team"] == team].sort_values("game_date")
        if team_hist.empty:
            st.info("No history for this team yet.")
            return
        st.line_chart(team_hist.set_index("game_date")[["batting_team_own_inning1_rate", "batting_team_slot1_4_ops"]])
        st.dataframe(
            team_hist[["game_date", "opponent_team", "batting_team_own_inning1_rate", "batting_team_slot1_4_ops"]],
            use_container_width=True, hide_index=True,
        )
    else:
        pitchers = sorted(features["opposing_starter_name"].dropna().unique())
        pitcher = st.selectbox("Starting pitcher", pitchers)
        p_hist = features[features["opposing_starter_name"] == pitcher].sort_values("game_date")
        if p_hist.empty:
            st.info("No history for this pitcher yet.")
            return
        st.line_chart(p_hist.set_index("game_date")[["opposing_starter_inning1_rate", "opposing_starter_overall_rate"]])
        st.dataframe(
            p_hist[["game_date", "batting_team", "opposing_starter_inning1_rate", "opposing_starter_overall_rate"]],
            use_container_width=True, hide_index=True,
        )


st.title("NRFI / YRFI Model")

tab1, tab2, tab3 = st.tabs(["Today's Slate", "Track Record", "Team / Pitcher Explorer"])
with tab1:
    render_slate()
with tab2:
    render_track_record()
with tab3:
    render_explorer()
