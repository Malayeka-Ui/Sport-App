import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import urllib.parse
import random
import time
import sqlite3
from datetime import datetime
from openai import OpenAI

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Badminton Pro-Hub Master",
    page_icon="🏸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .court-box {
        border: 2px dashed #0d6efd;
        border-radius: 8px;
        padding: 12px;
        background-color: #f0f7ff;
        font-weight: 600;
        text-align: center;
    }
    .stButton>button {
        border-radius: 6px;
    }
    .streak-banner {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SQLITE PERSISTENT H2H DATABASE SETUP
# ==============================================================================
DB_PATH = "badminton_h2h.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date TEXT,
            player1 TEXT,
            player2 TEXT,
            discipline TEXT,
            winner TEXT,
            score_summary TEXT,
            total_duration_sec REAL,
            total_rallies INTEGER,
            p1_smash_winners INTEGER,
            p2_smash_winners INTEGER
        )
        """
    )
    conn.commit()
    conn.close()

def save_match_to_db(p1, p2, disc, winner, summary, dur_sec, rallies, p1_smashes, p2_smashes):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO matches (match_date, player1, player2, discipline, winner, score_summary, total_duration_sec, total_rallies, p1_smash_winners, p2_smash_winners)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().strftime("%Y-%m-%d %H:%M"), p1, p2, disc, winner, summary, dur_sec, rallies, p1_smashes, p2_smashes),
    )
    conn.commit()
    conn.close()

def get_h2h_records():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM matches ORDER BY id DESC", conn)
    conn.close()
    return df

init_db()

# ==============================================================================
# AUDIO SYNTHESIS ENGINE
# ==============================================================================
def format_bwf_score_announcement(server_score, receiver_score, is_service_over=False):
    def to_word(s):
        return "Love" if s == 0 else str(s)

    srv = to_word(server_score)
    rcv = to_word(receiver_score)

    if server_score == receiver_score:
        score_phrase = f"{srv} All"
    else:
        score_phrase = f"{srv}, {rcv}"

    if is_service_over:
        return f"Service over. {score_phrase}"
    return score_phrase

def trigger_speech(text):
    if st.session_state.get("enable_audio", True):
        safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
        html_code = f"""
        <script>
            (function() {{
                try {{
                    const synth = window.parent.speechSynthesis || window.speechSynthesis;
                    if (synth) {{
                        synth.cancel();
                        synth.resume();
                        const utter = new SpeechSynthesisUtterance("{safe_text}");
                        utter.rate = 1.0;
                        utter.pitch = 1.0;
                        utter.lang = 'en-US';
                        synth.speak(utter);
                    }}
                }} catch (e) {{
                    console.error("Audio synthesis error:", e);
                }}
            }})();
        </script>
        """
        st.components.v1.html(html_code, height=0, width=0)

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
DEFAULT_SCHEDULE = [
    {"day": "Monday", "category": "Footwork & Agility", "drill": "6-Corner Shadow Footwork (4x20 reps)", "completed": False},
    {"day": "Tuesday", "category": "Gym & Strength", "drill": "Squats, Lunges & Shoulder Rotator Cuff stability", "completed": False},
    {"day": "Wednesday", "category": "On-Court Drills", "drill": "Net Kills, Drops & Half-Smash variations", "completed": False},
    {"day": "Thursday", "category": "Cardio", "drill": "HIIT Court Sprints (30s on / 30s off)", "completed": False},
    {"day": "Friday", "category": "Match Play", "drill": "Full competitive best-of-3 sets", "completed": False},
]

defaults = {
    "match_mode": "Singles",
    "score_p1": 0,
    "score_p2": 0,
    "games_p1": 0,
    "games_p2": 0,
    "current_game_num": 1,
    "completed_games": [],
    "server": "Player 1",
    "match_history": [],
    "p1_name": "Player 1",
    "p2_name": "Player 2",
    "p1_partner": "Partner 1",
    "p2_partner": "Partner 2",
    "game_over": False,
    "match_over": False,
    "match_winner": "",
    "match_start_time": None,
    "rally_start_time": time.time(),
    "chat_messages": [
        {
            "role": "assistant",
            "content": "Welcome to Badminton Pro-Hub! Ask me anything regarding official BWF rules, doubles rotations, string tension, or footwork drills.",
        }
    ],
    "training_schedule": [dict(item) for item in DEFAULT_SCHEDULE],
    "tournament_players": "1: Viktor Axelsen\n2: Shi Yuqi\n3: Lee Zii Jia\n4: Anthony Ginting\nKodai Naraoka\nChou Tien Chen\nLoh Kean Yew\nPrannoy H. S.",
    "enable_audio": True,
    "court_allocations": {"Court 1": "Available", "Court 2": "Available", "Court 3": "Available"},
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SECTOR_MAP = {
    "P1 Front-Left (Net)": (1.5, 5.7),
    "P1 Front-Right (Net)": (4.5, 5.7),
    "P1 Mid-Left (Drive)": (1.5, 3.5),
    "P1 Mid-Right (Drive)": (4.5, 3.5),
    "P1 Back-Left (Baseline)": (1.5, 1.2),
    "P1 Back-Right (Baseline)": (4.5, 1.2),
    "P2 Front-Left (Net)": (4.5, 7.7),
    "P2 Front-Right (Net)": (1.5, 7.7),
    "P2 Mid-Left (Drive)": (4.5, 9.9),
    "P2 Mid-Right (Drive)": (1.5, 9.9),
    "P2 Back-Left (Baseline)": (4.5, 12.2),
    "P2 Back-Right (Baseline)": (1.5, 12.2),
}

# ==============================================================================
# COMPUTATIONAL & ENGINE HELPERS
# ==============================================================================
def check_game_status(s1, s2):
    if s1 >= 21 or s2 >= 21:
        diff = abs(s1 - s2)
        if diff >= 2:
            return True, st.session_state.p1_name if s1 > s2 else st.session_state.p2_name
        if s1 == 30:
            return True, st.session_state.p1_name
        if s2 == 30:
            return True, st.session_state.p2_name
    return False, ""

def run_monte_carlo(s1, s2, server, n_simulations=1000):
    if s1 == s2 == 0:
        return 50.0, 50.0
    total_rallies = len(st.session_state.match_history)
    base_p1_prob = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p1_name) / total_rallies if total_rallies > 5 else 0.50

    p1_wins = 0
    for _ in range(n_simulations):
        sim_s1, sim_s2, sim_srv = s1, s2, server
        while True:
            prob = max(0.15, min(0.85, base_p1_prob + (0.03 if sim_srv == "Player 1" else -0.03)))
            if np.random.rand() < prob:
                sim_s1 += 1
                sim_srv = "Player 1"
            else:
                sim_s2 += 1
                sim_srv = "Player 2"

            if sim_s1 >= 21 or sim_s2 >= 21:
                if abs(sim_s1 - sim_s2) >= 2 or sim_s1 == 30 or sim_s2 == 30:
                    if sim_s1 > sim_s2:
                        p1_wins += 1
                    break
    p1_pct = round((p1_wins / n_simulations) * 100, 1)
    return p1_pct, round(100.0 - p1_pct, 1)

def get_current_streak():
    if not st.session_state.match_history:
        return None, 0
    recent_winner = st.session_state.match_history[-1]["winner"]
    count = 0
    for r in reversed(st.session_state.match_history):
        if r["winner"] == recent_winner:
            count += 1
        else:
            break
    return recent_winner, count

# ==============================================================================
# COURT RENDERING FUNCTION
# ==============================================================================
def draw_base_court_figure():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=6.1, y1=13.4, line=dict(color="#ffffff", width=3), fillcolor="#1e824c", opacity=0.35)
    fig.add_shape(type="line", x0=-0.3, y0=6.7, x1=6.4, y1=6.7, line=dict(color="#ffffff", width=4, dash="dash"))
    fig.add_shape(type="line", x0=0.46, y0=0, x1=0.46, y1=13.4, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=5.64, y0=0, x1=5.64, y1=13.4, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=0, y0=4.72, x1=6.1, y1=4.72, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=0, y0=8.68, x1=6.1, y1=8.68, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=0, y0=0.76, x1=6.1, y1=0.76, line=dict(color="#dcdde1", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=0, y0=12.64, x1=6.1, y1=12.64, line=dict(color="#dcdde1", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=3.05, y0=0, x1=3.05, y1=4.72, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=3.05, y0=8.68, x1=3.05, y1=13.4, line=dict(color="#ffffff", width=2))
    return fig

def draw_tactical_court(server_label, p1_name, p2_name, is_even, is_doubles=False, p1_sub="", p2_sub=""):
    fig = draw_base_court_figure()
    min_x = 0.0 if is_doubles else 0.46
    max_x = 6.1 if is_doubles else 5.64

    if server_label == "Player 1":
        serving_display = f"{p1_name} ({p1_sub})" if is_doubles else p1_name
        receiving_display = f"{p2_name} ({p2_sub})" if is_doubles else p2_name
        if is_even:
            srv_x0, srv_x1, srv_y0, srv_y1 = 3.05, max_x, 0, 4.72
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = min_x, 3.05, 8.68, 12.64 if is_doubles else 13.4
            side_str = "Right Court (Even)"
        else:
            srv_x0, srv_x1, srv_y0, srv_y1 = min_x, 3.05, 0, 4.72
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 3.05, max_x, 8.68, 12.64 if is_doubles else 13.4
            side_str = "Left Court (Odd)"
    else:
        serving_display = f"{p2_name} ({p2_sub})" if is_doubles else p2_name
        receiving_display = f"{p1_name} ({p1_sub})" if is_doubles else p1_name
        if is_even:
            srv_x0, srv_x1, srv_y0, srv_y1 = min_x, 3.05, 8.68, 13.4
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 3.05, max_x, 0, 12.64 if is_doubles else 4.72
            side_str = "Right Court (Even)"
        else:
            srv_x0, srv_x1, srv_y0, srv_y1 = 3.05, max_x, 8.68, 13.4
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = min_x, 3.05, 0, 12.64 if is_doubles else 4.72
            side_str = "Left Court (Odd)"

    fig.add_shape(type="rect", x0=srv_x0, y0=srv_y0, x1=srv_x1, y1=srv_y1, fillcolor="#f39c12", opacity=0.7, line=dict(color="#d35400", width=3))
    fig.add_annotation(x=(srv_x0+srv_x1)/2, y=(srv_y0+srv_y1)/2, text=f"🏸 SERVING<br><b>{serving_display}</b>", showarrow=False, font=dict(color="#fff", size=10, family="Arial Black"))

    fig.add_shape(type="rect", x0=rcv_x0, y0=rcv_y0, x1=rcv_x1, y1=rcv_y1, fillcolor="#2980b9", opacity=0.6, line=dict(color="#1abc9c", width=2, dash="dot"))
    fig.add_annotation(x=(rcv_x0+rcv_x1)/2, y=(rcv_y0+rcv_y1)/2, text=f"🎯 RECEIVING<br><b>{receiving_display}</b>", showarrow=False, font=dict(color="#fff", size=10, family="Arial Black"))

    fig.add_annotation(x=3.05, y=-0.5, text=f"🔻 {p1_name} End", showarrow=False, font=dict(color="#ecf0f1", size=11))
    fig.add_annotation(x=3.05, y=13.9, text=f"🔺 {p2_name} End", showarrow=False, font=dict(color="#ecf0f1", size=11))

    fig.update_layout(
        title=f"Court Radar ({'Doubles' if is_doubles else 'Singles'}): {side_str}",
        xaxis=dict(range=[-0.6, 6.7], visible=False),
        yaxis=dict(range=[-1.0, 14.4], visible=False),
        width=380,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#1e272e",
    )
    return fig

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.title("🏸 Badminton Pro-Hub")
st.sidebar.caption("Tactical Scorer, AI Coach & Training Suite")

nav_choice = st.sidebar.radio(
    "Modules",
    [
        "🏸 Live Tactical Scorer",
        "🎞️ Match Point Replay",
        "📋 Tactical Debrief Report",
        "⏱️ Court Timers & Intervals",
        "🗺️ Shot Placement Heatmap",
        "🏃 Movement & Mileage Tracker",
        "🧘 Kinematic Warm-Up Routine",
        "📊 Head-to-Head (H2H) Database",
        "🧵 String Tension Degradation",
        "⚖️ Racket Balance & Swing Weight",
        "🧤 Grip Wrap Replacement Log",
        "🤖 AI Rules & Tactical Coach",
        "🏃 Interactive Footwork Pacer",
        "🌡️ Shuttlecock Speed Optimizer",
        "🏆 Tournament Generator",
        "🏟️ Multi-Court Allocation Board",
        "🛍️ Local Equipment Finder",
        "📅 Training Schedule Planner",
        "❤️ Health, Fitness & Fatigue",
        "⚙️ Racket & Tension Optimizer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Audible Score Announcements:**")
st.session_state.enable_audio = st.sidebar.checkbox("🔊 Voice Umpire Enabled", value=st.session_state.enable_audio)

# ==============================================================================
# MODULE 1: LIVE TACTICAL SCORER (WITH TIMERS, STREAKS & VOICE)
# ==============================================================================
if nav_choice == "🏸 Live Tactical Scorer":
    st.header("🏸 Live Tactical Scorer & Multi-Game Manager")

    # Match Config
    cfg1, cfg2, cfg3 = st.columns([1.5, 2, 2])
    with cfg1:
        st.session_state.match_mode = st.selectbox("Discipline", ["Singles", "Doubles"], index=0 if st.session_state.match_mode == "Singles" else 1)
    with cfg2:
        st.session_state.p1_name = st.text_input("Side 1 / Team", value=st.session_state.p1_name)
        if st.session_state.match_mode == "Doubles":
            st.session_state.p1_partner = st.text_input("Team 1 Partner", value=st.session_state.p1_partner)
    with cfg3:
        st.session_state.p2_name = st.text_input("Side 2 / Team", value=st.session_state.p2_name)
        if st.session_state.match_mode == "Doubles":
            st.session_state.p2_partner = st.text_input("Team 2 Partner", value=st.session_state.p2_partner)

    # Master Timers Bar
    now_ts = time.time()
    if st.session_state.match_start_time is None and len(st.session_state.match_history) > 0:
        st.session_state.match_start_time = now_ts

    total_elapsed_sec = int(now_ts - st.session_state.match_start_time) if st.session_state.match_start_time else 0
    mins, secs = divmod(total_elapsed_sec, 60)
    hrs, mins = divmod(mins, 60)
    match_clock_str = f"{hrs:02d}:{mins:02d}:{secs:02d}"

    st.markdown("---")
    tbar1, tbar2, tbar3 = st.columns(3)
    tbar1.metric("⏱️ Overall Match Duration", match_clock_str)
    cur_rally_dur = round(max(1.0, now_ts - st.session_state.rally_start_time), 1)
    tbar2.metric("⏳ Current Rally Clock", f"{cur_rally_dur}s")

    # Momentum Banner
    streak_holder, streak_count = get_current_streak()
    if streak_count >= 3:
        tbar3.markdown(f'<div class="streak-banner">🔥 MOMENTUM: {streak_holder} on a {streak_count}-Point Run!</div>', unsafe_allow_html=True)
    else:
        tbar3.info("Momentum balanced. Service turnover active.")

    # Standings & Scoring
    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns([1.5, 1.5, 2, 1])
    with sc_col1:
        st.metric(f"{st.session_state.p1_name} (Games: {st.session_state.games_p1})", st.session_state.score_p1)
    with sc_col2:
        st.metric(f"{st.session_state.p2_name} (Games: {st.session_state.games_p2})", st.session_state.score_p2)
    with sc_col3:
        current_srv = st.session_state.p1_name if st.session_state.server == "Player 1" else st.session_state.p2_name
        srv_score = st.session_state.score_p1 if st.session_state.server == "Player 1" else st.session_state.score_p2
        is_even = (srv_score % 2 == 0)
        box_side = "RIGHT" if is_even else "LEFT"
        st.markdown(
            f"""
            <div class="court-box">
                Game {st.session_state.current_game_num} of 3<br>
                🏸 Serving: <b>{current_srv}</b> ({box_side} Court)
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sc_col4:
        st.write("")
        if st.button("↩️ Undo Point", use_container_width=True, disabled=len(st.session_state.match_history) == 0):
            last = st.session_state.match_history.pop()
            st.session_state.score_p1 = last["prev_s1"]
            st.session_state.score_p2 = last["prev_s2"]
            st.session_state.server = last["prev_server"]
            st.session_state.game_over = False
            trigger_speech("Point Undone")
            st.rerun()

        if st.button("🔄 Reset Match", use_container_width=True):
            st.session_state.score_p1 = 0
            st.session_state.score_p2 = 0
            st.session_state.games_p1 = 0
            st.session_state.games_p2 = 0
            st.session_state.current_game_num = 1
            st.session_state.completed_games = []
            st.session_state.server = "Player 1"
            st.session_state.match_history = []
            st.session_state.game_over = False
            st.session_state.match_over = False
            st.session_state.match_start_time = None
            st.session_state.rally_start_time = time.time()
            st.rerun()

    # Change ends notification
    if st.session_state.current_game_num == 3 and (st.session_state.score_p1 == 11 or st.session_state.score_p2 == 11):
        st.warning("⚠️ **BWF Change of Ends:** 11 points in deciding set. Players switch sides!")

    # Game / Match transition
    if st.session_state.game_over and not st.session_state.match_over:
        st.success(f"🏆 Game {st.session_state.current_game_num} Winner: **{st.session_state.last_game_winner}**!")
        if st.button("▶️ Advance to Next Game"):
            st.session_state.completed_games.append({
                "game": st.session_state.current_game_num,
                "p1_score": st.session_state.score_p1,
                "p2_score": st.session_state.score_p2,
                "winner": st.session_state.last_game_winner
            })
            if st.session_state.score_p1 > st.session_state.score_p2:
                st.session_state.games_p1 += 1
            else:
                st.session_state.games_p2 += 1

            if st.session_state.games_p1 == 2 or st.session_state.games_p2 == 2:
                st.session_state.match_over = True
                st.session_state.match_winner = st.session_state.p1_name if st.session_state.games_p1 == 2 else st.session_state.p2_name
                trigger_speech(f"Match won by {st.session_state.match_winner}")

                p1_s = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p1_name and "Smash" in r["shot"])
                p2_s = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p2_name and "Smash" in r["shot"])
                summary_str = f"{st.session_state.games_p1}-{st.session_state.games_p2}"
                save_match_to_db(
                    st.session_state.p1_name, st.session_state.p2_name,
                    st.session_state.match_mode, st.session_state.match_winner,
                    summary_str, total_elapsed_sec, len(st.session_state.match_history), p1_s, p2_s
                )
            else:
                st.session_state.current_game_num += 1
                st.session_state.score_p1 = 0
                st.session_state.score_p2 = 0
                st.session_state.game_over = False
                trigger_speech(f"Game {st.session_state.current_game_num}. Love all, play.")
            st.rerun()

    if st.session_state.match_over:
        st.balloons()
        st.success(f"🥇 **MATCH CHAMPION: {st.session_state.match_winner}** (Saved to Career Database)")

    # Scoring Buttons
    st.markdown("---")
    st.subheader("Point Logger & Coordinate Placement")

    p1_sectors = ["P2 Front-Left (Net)", "P2 Front-Right (Net)", "P2 Mid-Left (Drive)", "P2 Mid-Right (Drive)", "P2 Back-Left (Baseline)", "P2 Back-Right (Baseline)"]
    p2_sectors = ["P1 Front-Left (Net)", "P1 Front-Right (Net)", "P1 Mid-Left (Drive)", "P1 Mid-Right (Drive)", "P1 Back-Left (Baseline)", "P1 Back-Right (Baseline)"]
    shot_types = ["Smash Winner", "Drop Winner", "Net Kill", "Opponent Out", "Opponent Net Error", "Service Fault"]

    b1, b2 = st.columns(2)
    with b1:
        st.markdown(f"**Award to: {st.session_state.p1_name}**")
        p1_shot = st.selectbox("Point Outcome (Side 1)", shot_types, key="p1_st")
        p1_sec = st.selectbox("Landing Sector on Opponent Court", p1_sectors, key="p1_sec")
        if st.button(f"➕ Score for {st.session_state.p1_name}", disabled=st.session_state.game_over or st.session_state.match_over, use_container_width=True):
            if st.session_state.match_start_time is None:
                st.session_state.match_start_time = time.time()

            point_duration = round(max(1.0, time.time() - st.session_state.rally_start_time), 1)
            st.session_state.rally_start_time = time.time()

            prev_s1, prev_s2 = st.session_state.score_p1, st.session_state.score_p2
            prev_srv = st.session_state.server

            st.session_state.score_p1 += 1
            st.session_state.server = "Player 1"
            service_changed = (prev_srv != "Player 1")

            origin_coords = (4.35, 2.36) if (prev_s1 % 2 == 0) else (1.75, 2.36)
            target_coords = SECTOR_MAP.get(p1_sec, (3.05, 9.9))

            st.session_state.match_history.append({
                "game": st.session_state.current_game_num,
                "rally": len(st.session_state.match_history) + 1,
                "winner": st.session_state.p1_name,
                "loser": st.session_state.p2_name,
                "shot": p1_shot,
                "sector": p1_sec,
                "duration_sec": point_duration,
                "origin_coords": origin_coords,
                "target_coords": target_coords,
                "p1_score": st.session_state.score_p1,
                "p2_score": st.session_state.score_p2,
                "prev_s1": prev_s1,
                "prev_s2": prev_s2,
                "prev_server": prev_srv,
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.last_game_winner = win
                trigger_speech(f"Game to {win}. {st.session_state.score_p1}, {st.session_state.score_p2}")
            else:
                phrase = format_bwf_score_announcement(st.session_state.score_p1, st.session_state.score_p2, service_changed)
                trigger_speech(phrase)
            st.rerun()

    with b2:
        st.markdown(f"**Award to: {st.session_state.p2_name}**")
        p2_shot = st.selectbox("Point Outcome (Side 2)", shot_types, key="p2_st")
        p2_sec = st.selectbox("Landing Sector on Opponent Court", p2_sectors, key="p2_sec")
        if st.button(f"➕ Score for {st.session_state.p2_name}", disabled=st.session_state.game_over or st.session_state.match_over, use_container_width=True):
            if st.session_state.match_start_time is None:
                st.session_state.match_start_time = time.time()

            point_duration = round(max(1.0, time.time() - st.session_state.rally_start_time), 1)
            st.session_state.rally_start_time = time.time()

            prev_s1, prev_s2 = st.session_state.score_p1, st.session_state.score_p2
            prev_srv = st.session_state.server

            st.session_state.score_p2 += 1
            st.session_state.server = "Player 2"
            service_changed = (prev_srv != "Player 2")

            origin_coords = (1.75, 11.04) if (prev_s2 % 2 == 0) else (4.35, 11.04)
            target_coords = SECTOR_MAP.get(p2_sec, (3.05, 3.5))

            st.session_state.match_history.append({
                "game": st.session_state.current_game_num,
                "rally": len(st.session_state.match_history) + 1,
                "winner": st.session_state.p2_name,
                "loser": st.session_state.p1_name,
                "shot": p2_shot,
                "sector": p2_sec,
                "duration_sec": point_duration,
                "origin_coords": origin_coords,
                "target_coords": target_coords,
                "p1_score": st.session_state.score_p1,
                "p2_score": st.session_state.score_p2,
                "prev_s1": prev_s1,
                "prev_s2": prev_s2,
                "prev_server": prev_srv,
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.last_game_winner = win
                trigger_speech(f"Game to {win}. {st.session_state.score_p2}, {st.session_state.score_p1}")
            else:
                phrase = format_bwf_score_announcement(st.session_state.score_p2, st.session_state.score_p1, service_changed)
                trigger_speech(phrase)
            st.rerun()

    st.markdown("---")
    mc_col, crt_col = st.columns([3, 2])
    with mc_col:
        st.subheader("🎲 Monte Carlo Win Forecaster")
        p1_prob, p2_prob = run_monte_carlo(st.session_state.score_p1, st.session_state.score_p2, st.session_state.server)
        m1, m2 = st.columns(2)
        m1.metric(f"{st.session_state.p1_name} Win Rate", f"{p1_prob}%")
        m2.metric(f"{st.session_state.p2_name} Win Rate", f"{p2_prob}%")
        st.progress(int(p1_prob))

    with crt_col:
        st.plotly_chart(
            draw_tactical_court(
                server_label=st.session_state.server,
                p1_name=st.session_state.p1_name,
                p2_name=st.session_state.p2_name,
                is_even=is_even,
                is_doubles=(st.session_state.match_mode == "Doubles"),
                p1_sub=st.session_state.p1_partner,
                p2_sub=st.session_state.p2_partner,
            ),
            use_container_width=True,
        )

# ==============================================================================
# MODULE 2: POINT REPLAY VISUALIZER (FULL TRAJECTORY & TIMER REVIEW)
# ==============================================================================
elif nav_choice == "🎞️ Match Point Replay":
    st.header("🎞️ Match Point Replay Visualizer & Flight Log")
    st.caption("Inspect the trajectory, duration, and landing sector of every rally in the match.")

    if not st.session_state.match_history:
        st.info("No rallies recorded yet. Score points in Live Tactical Scorer to review them.")
    else:
        total_pts = len(st.session_state.match_history)
        selected_idx = st.slider("Select Rally Number to Inspect:", min_value=1, max_value=total_pts, value=total_pts) - 1
        rally_data = st.session_state.match_history[selected_idx]

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Game #", rally_data["game"])
        r2.metric("Point Winner", f"{rally_data['winner']} ({rally_data['shot']})")
        r3.metric("Rally Duration", f"{rally_data.get('duration_sec', 0.0)}s")
        r4.metric("Score After Point", f"{rally_data['p1_score']} - {rally_data['p2_score']}")

        fig_rep = draw_base_court_figure()
        ox, oy = rally_data.get("origin_coords", (3.05, 2.36))
        tx, ty = rally_data.get("target_coords", (3.05, 9.9))

        fig_rep.add_annotation(
            x=tx, y=ty, ax=ox, ay=oy, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3,
            arrowcolor="#e74c3c" if "Winner" in rally_data["shot"] else "#f39c12",
        )
        fig_rep.add_trace(go.Scatter(x=[ox], y=[oy], mode="markers+text", marker=dict(size=14, color="#3498db"), text=["🏸 Serve Origin"], textposition="top center"))
        fig_rep.add_trace(go.Scatter(x=[tx], y=[ty], mode="markers+text", marker=dict(size=16, color="#2ecc71" if "Winner" in rally_data["shot"] else "#e74c3c", symbol="x"), text=[f"🎯 {rally_data['shot']}"], textposition="bottom center"))

        fig_rep.update_layout(title=f"Rally {selected_idx+1} Trajectory ({rally_data.get('duration_sec', 0.0)}s exchange)", xaxis=dict(range=[-0.5, 6.6], visible=False), yaxis=dict(range=[-0.5, 13.9], visible=False), width=500, height=650, plot_bgcolor="#1e272e", showlegend=False)
        st.plotly_chart(fig_rep, use_container_width=True)

# ==============================================================================
# MODULE 3: TACTICAL DEBRIEF REPORT
# ==============================================================================
elif nav_choice == "📋 Tactical Debrief Report":
    st.header("📋 Post-Match Tactical Debrief & Performance Report")
    st.caption("In-depth statistical breakdown of conversion rates, rally tempos, and error patterns.")

    if not st.session_state.match_history:
        st.info("Log rallies in the scorer to view comprehensive match analytics.")
    else:
        df_hist = pd.DataFrame(st.session_state.match_history)
        avg_dur = df_hist["duration_sec"].mean() if "duration_sec" in df_hist else 0.0

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Rallies Played", len(df_hist))
        d2.metric("Average Rally Pace", f"{round(avg_dur, 1)}s")
        p1_winners = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p1_name and "Winner" in r["shot"])
        p2_winners = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p2_name and "Winner" in r["shot"])
        d3.metric(f"{st.session_state.p1_name} Direct Winners", p1_winners)
        d4.metric(f"{st.session_state.p2_name} Direct Winners", p2_winners)

        st.markdown("---")
        c_ch1, c_ch2 = st.columns(2)
        with c_ch1:
            fig_shots = px.histogram(df_hist, x="shot", color="winner", barmode="group", title="Points Won by Shot Category")
            st.plotly_chart(fig_shots, use_container_width=True)
        with c_ch2:
            fig_dur = px.box(df_hist, x="winner", y="duration_sec", points="all", title="Rally Duration Distribution by Point Winner")
            st.plotly_chart(fig_dur, use_container_width=True)

# ==============================================================================
# MODULE 4: COURT TIMERS & INTERVALS
# ==============================================================================
elif nav_choice == "⏱️ Court Timers & Intervals":
    st.header("⏱️ Official BWF Court Timers & Interval Alarms")
    st.caption("Automated countdown timers adhering to BWF Law 16.")

    tm1, tm2, tm3 = st.columns(3)
    with tm1:
        st.subheader("60-Second Interval")
        st.caption("Called when the leading score reaches 11 points.")
        if st.button("Start 60s Interval Timer", use_container_width=True):
            trigger_speech("60 second interval begins.")
            p = st.progress(0)
            txt = st.empty()
            for s in range(60, -1, -1):
                txt.markdown(f"### ⏳ Remaining: **{s}s**")
                p.progress((60 - s) / 60)
                if s == 20:
                    trigger_speech("20 seconds remaining.")
                time.sleep(1)
            trigger_speech("Court. Time.")
            st.success("Interval completed!")

    with tm2:
        st.subheader("120-Second Game Break")
        st.caption("Between Game 1, Game 2, and Game 3.")
        if st.button("Start 120s Game Interval", use_container_width=True):
            trigger_speech("Two minute interval.")
            p = st.progress(0)
            txt = st.empty()
            for s in range(120, -1, -1):
                txt.markdown(f"### ⏳ Remaining: **{s}s**")
                p.progress((120 - s) / 120)
                if s == 30:
                    trigger_speech("30 seconds remaining.")
                time.sleep(1)
            trigger_speech("Court. Time.")
            st.success("Game break complete!")

    with tm3:
        st.subheader("Pre-Match Warm-up")
        st.caption("Official 2-minute knock-up timer.")
        if st.button("Start 2-Min Warm-up", use_container_width=True):
            trigger_speech("Two minutes knock up play.")
            p = st.progress(0)
            txt = st.empty()
            for s in range(120, -1, -1):
                txt.markdown(f"### ⏳ Remaining: **{s}s**")
                p.progress((120 - s) / 120)
                time.sleep(1)
            trigger_speech("Ready to play.")
            st.success("Warm-up complete!")

# ==============================================================================
# MODULE 5: SHOT PLACEMENT HEATMAP
# ==============================================================================
elif nav_choice == "🗺️ Shot Placement Heatmap":
    st.header("🗺️ Interactive Shot Placement Heatmap")
    st.caption("Analyzes rally finishing zones, smash distributions, and tactical court vulnerabilities.")

    if not st.session_state.match_history:
        st.info("No rallies recorded yet. Play and log shots in the Tactical Scorer to generate heatmaps.")
    else:
        df_shots = pd.DataFrame(st.session_state.match_history)
        df_shots["x"] = df_shots["target_coords"].map(lambda c: c[0] + np.random.uniform(-0.25, 0.25))
        df_shots["y"] = df_shots["target_coords"].map(lambda c: c[1] + np.random.uniform(-0.25, 0.25))

        fig_heat = px.density_contour(
            df_shots, x="x", y="y", color="winner",
            range_x=[0, 6.1], range_y=[0, 13.4],
            title="Landing Density by Player (Full Court Perspective)"
        )
        fig_heat.add_trace(go.Scatter(x=df_shots["x"], y=df_shots["y"], mode="markers", marker=dict(size=9, color="#e74c3c"), text=df_shots["shot"] + " by " + df_shots["winner"]))
        fig_heat.update_layout(plot_bgcolor="#1e272e", width=700, height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================================
# MODULE 6: MOVEMENT & MILEAGE TRACKER
# ==============================================================================
elif nav_choice == "🏃 Movement & Mileage Tracker":
    st.header("🏃 Court Movement & Mileage Tracker")
    st.caption("Estimates physical distance covered based on rally counts, tempo, and court dimensions.")

    rallies_count = len(st.session_state.match_history)
    # Competitive badminton averages ~6.4m of court coverage per rally
    est_meters = rallies_count * 6.4
    km_covered = round(est_meters / 1000.0, 2)
    est_lunges = int(rallies_count * 2.8)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Distance Covered", f"{km_covered} km", f"{int(est_meters)} meters")
    m2.metric("Explosive Lunges Executed", f"~{est_lunges} reps")
    m3.metric("High-Impact Decelerations", f"~{int(rallies_count * 3.5)} stops")

    st.info("💡 **Physiological Note:** Badminton requires more explosive multi-directional direction-changes per minute than tennis or squash.")

# ==============================================================================
# MODULE 7: KINEMATIC WARM-UP ROUTINE
# ==============================================================================
elif nav_choice == "🧘 Kinematic Warm-Up Routine":
    st.header("🧘 Badminton Kinematic Warm-Up Guide")
    st.caption("Pre-match dynamic stretching and tendon activation sequence.")

    warmup_steps = [
        ("Wrist & Forearm Dynamic Rotations", "30 seconds", "Prepares extensor tendons for rapid racket pronation/supination."),
        ("Multi-Directional Lunges (Groin Activation)", "45 seconds", "Deep corner lunges to activate adductors and hamstrings safely."),
        ("Rotator Cuff External Band Rotations", "45 seconds", "Stabilizes posterior shoulder to support maximum-velocity smashes."),
        ("Ankle Plantar & Dorsiflexion Hops", "30 seconds", "Primes Achilles tendon elasticity for explosive split-steps."),
    ]

    for title, dur, desc in warmup_steps:
        st.markdown(f"#### 🏸 {title} (`{dur}`)")
        st.write(desc)
        st.markdown("---")

# ==============================================================================
# MODULE 8: HEAD-TO-HEAD (H2H) DATABASE
# ==============================================================================
elif nav_choice == "📊 Head-to-Head (H2H) Database":
    st.header("📊 Head-to-Head (H2H) Match History & Career Records")
    st.caption("Persistent career records saved in local SQLite storage.")

    df_h2h = get_h2h_records()
    if df_h2h.empty:
        st.info("No finished matches saved in database yet. Finish a match in Live Tactical Scorer to record records.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Matches Recorded", len(df_h2h))
        top_winner = df_h2h["winner"].mode()[0] if not df_h2h.empty else "N/A"
        h2.metric("Most Dominant Player", top_winner)
        total_smashes = df_h2h["p1_smash_winners"].sum() + df_h2h["p2_smash_winners"].sum()
        h3.metric("Career Smash Winners Logged", int(total_smashes))

        st.markdown("---")
        st.subheader("Match History Log")
        st.dataframe(df_h2h, use_container_width=True, hide_index=True)

# ==============================================================================
# MODULE 9: STRING TENSION DEGRADATION
# ==============================================================================
elif nav_choice == "🧵 String Tension Degradation":
    st.header("🧵 Racket String Tension Degradation Calculator")
    st.caption("Calculates tension loss (initial 24h drop + play wear) and recommends restringing intervals.")

    st1, st2, st3 = st.columns(3)
    with st1:
        initial_tension = st.number_input("Strung Tension (lbs)", min_value=18, max_value=35, value=26)
    with st2:
        hours_played = st.number_input("Total Court Hours Played", min_value=0.0, max_value=150.0, value=12.0, step=1.0)
    with st3:
        string_type = st.selectbox("String Model", ["Thin High-Repulsion (0.65mm - BG66UM/Li-Ning No.1)", "Durable Gauge (0.70mm - BG65/VBS-70)"])

    hourly_rate = 0.32 if "Thin" in string_type else 0.22
    estimated_drop = 1.5 + (hours_played * hourly_rate)
    current_tension = max(14.0, round(initial_tension - estimated_drop, 1))

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Current Tension", f"{current_tension} lbs", delta=f"-{round(estimated_drop, 1)} lbs")
    c2.metric("Remaining Elastic Sweet Spot", f"{max(10, int(100 - (estimated_drop * 8)))}%")
    status_alert = "🟢 Optimal Performance" if current_tension >= (initial_tension - 3) else ("🟡 Noticeable Tension Loss" if current_tension >= (initial_tension - 5) else "🔴 Restring Recommended")
    c3.metric("String Bed Health", status_alert)

# ==============================================================================
# MODULE 10: RACKET BALANCE & SWING WEIGHT
# ==============================================================================
elif nav_choice == "⚖️ Racket Balance & Swing Weight":
    st.header("⚖️ Racket Balance & Swing Weight Calculator")
    st.caption("Determine precise racket balance category based on measured balance point from handle butt cap.")

    col_w, col_bp = st.columns(2)
    with col_w:
        weight_g = st.number_input("Racket Weight (grams, unstrung or strung)", min_value=70.0, max_value=110.0, value=83.0, step=0.5)
    with col_bp:
        bp_mm = st.number_input("Measured Balance Point (mm from handle base)", min_value=270, max_value=330, value=300)

    if bp_mm > 300:
        balance_class = "Head-Heavy (Power & Smash Offensive)"
    elif bp_mm >= 290:
        balance_class = "Even-Balance (All-Round & Driving)"
    else:
        balance_class = "Head-Light (Speed & Fast Defense)"

    # Approximate swing weight index
    sw_index = round((weight_g * (bp_mm / 100.0) ** 2) / 1000.0, 1)

    st.markdown("---")
    b1, b2 = st.columns(2)
    b1.metric("Racket Classification", balance_class.split()[0])
    b2.metric("Calculated Swing Weight Index", f"{sw_index} kg·cm²")

# ==============================================================================
# MODULE 11: GRIP WRAP REPLACEMENT LOG
# ==============================================================================
elif nav_choice == "🧤 Grip Wrap Replacement Log":
    st.header("🧤 Grip Wrap & Overgrip Wear Tracker")
    st.caption("Tracks hours on grip tape to prevent racket slippage and loss of bevel tactile feel.")

    grip_hours = st.number_input("Hours on Current Grip Wrap", min_value=0.0, max_value=60.0, value=14.0, step=1.0)
    grip_material = st.selectbox("Grip Type", ["Polyurethane Overgrip (Yonex Super Grap)", "Towel Grip (Cotton)"])

    max_hours = 12.0 if "Towel" in grip_material else 20.0
    wear_pct = min(100, int((grip_hours / max_hours) * 100))

    st.progress(wear_pct / 100.0)
    if wear_pct >= 85:
        st.warning("⚠️ **Grip replacement recommended!** Sweat saturation reduces racket control.")
    else:
        st.success(f"Grip condition acceptable (~{int(max_hours - grip_hours)} court hours remaining).")

# ==============================================================================
# MODULE 12: AI RULES ARBITER & COACH (GROQ OPENAI SDK)
# ==============================================================================
elif nav_choice == "🤖 AI Rules & Tactical Coach":
    st.header("🤖 AI Rules Arbiter & Tactical Coach")
    st.caption("Powered by OpenAI Models on Groq LPU Architecture & BWF Regulations Engine")

    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if groq_api_key:
        st.success("🟢 AI Coach Engine Online (Connected to OpenAI Models via Groq)")
    else:
        st.info("⚪ Offline BWF Knowledge Base Active (Add GROQ_API_KEY to secrets to enable generative AI)")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about rules, tactics, shot selection, or drills...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("AI Coach analyzing with Groq..."):
            response_text = get_groq_openai_response(user_input, groq_api_key)

        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

# ==============================================================================
# MODULE 13: INTERACTIVE FOOTWORK PACER
# ==============================================================================
elif nav_choice == "🏃 Interactive Footwork Pacer":
    st.header("🏃 Interactive Shadow Footwork Pacer & Audio Metronome")
    st.caption("Solo reaction trainer calling out random corners to sharpen multi-directional split-steps.")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        interval_speed = st.slider("Corner Call Interval (Seconds)", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    with col_p2:
        drill_reps = st.number_input("Total Shots", min_value=10, max_value=60, value=20)
    with col_p3:
        include_smash = st.checkbox("Include Jumping Smashes", value=True)

    corners = ["Front Left Net", "Front Right Net", "Midcourt Left Drive", "Midcourt Right Drive", "Rear Left Clear", "Rear Right Smash"]
    if not include_smash:
        corners.remove("Rear Right Smash")

    if st.button("🚀 Start Footwork Drill", use_container_width=True):
        st.info("Drill in progress! Follow vocal prompts and visual cues:")
        prog = st.progress(0)
        status_placeholder = st.empty()

        for i in range(drill_reps):
            target = random.choice(corners)
            trigger_speech(target)
            status_placeholder.markdown(
                f"""
                <div style="background-color:#0984e3; padding:35px; border-radius:10px; text-align:center; color:white;">
                    <h2>SHOT {i+1} / {drill_reps}</h2>
                    <h1>🏸 {target.upper()}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )
            prog.progress((i + 1) / drill_reps)
            time.sleep(interval_speed)

        st.success("🎉 Drill Complete! Recover and log your sets in the Training Planner.")

# ==============================================================================
# MODULE 14: SHUTTLECOCK SPEED & ELEVATION OPTIMIZER
# ==============================================================================
elif nav_choice == "🌡️ Shuttlecock Speed Optimizer":
    st.header("🌡️ Shuttlecock Speed & Altitude Optimizer")
    st.caption("Calculate exact BWF shuttlecock speed ratings based on ambient temperature, humidity, and elevation.")

    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Hall Temperature (°C)", min_value=10.0, max_value=42.0, value=24.0)
    with c2:
        elevation = st.number_input("Court Altitude (Meters above sea level)", min_value=0, max_value=3000, value=350)
    with c3:
        humidity = st.slider("Relative Humidity (%)", min_value=10, max_value=95, value=50)

    speed_score = (temp * 0.5) + (elevation * 0.008) - (humidity * 0.05)
    if speed_score < 14:
        rec_speed = "Speed 78 (Fast / Heavy)"
        tip_action = "No tipping required. Ideal for cold halls."
    elif speed_score < 18:
        rec_speed = "Speed 77 (Standard / Moderate)"
        tip_action = "Standard factory flight. Test with 2 baseline clears."
    else:
        rec_speed = "Speed 76 (Slow / Light)"
        tip_action = "Shuttle flies fast. Tip every second feather slightly outward to slow flight speed."

    st.markdown("---")
    s1, s2 = st.columns(2)
    s1.metric("Recommended Shuttlecock Rating", rec_speed)
    s2.info(f"**Feather Tipping Instruction:** {tip_action}")

# ==============================================================================
# MODULE 15: TOURNAMENT GENERATOR (SEEDED BRACKET & ROUND ROBIN)
# ==============================================================================
elif nav_choice == "🏆 Tournament Generator":
    st.header("🏆 Tournament Bracket & Round-Robin Generator")
    st.caption("Features seeded single-elimination brackets and round-robin league standings with Points Differential (Δ).")

    t1, t2 = st.columns([2, 1])
    with t1:
        player_list_str = st.text_area("Enter Player Names (Format '1: Name' for seeds)", value=st.session_state.tournament_players, height=140)
        st.session_state.tournament_players = player_list_str
    with t2:
        tourn_mode = st.radio("Tournament Format", ["Seeded Single Elimination Bracket", "Round-Robin League Standings"])

    raw_players = [p.strip() for p in player_list_str.split("\n") if p.strip()]

    if st.button("⚡ Generate Draw / Fixtures"):
        if len(raw_players) < 2:
            st.warning("Please enter at least 2 players.")
        elif tourn_mode == "Seeded Single Elimination Bracket":
            st.subheader("Tournament Knockout Bracket (Seed Separated)")
            seeds = [p for p in raw_players if ":" in p]
            unseeded = [p for p in raw_players if ":" not in p]
            random.shuffle(unseeded)

            draw = []
            if len(seeds) >= 2:
                draw.append(seeds[0])  # Top seed at top
                draw.extend(unseeded[: len(unseeded)//2])
                draw.extend(unseeded[len(unseeded)//2 :])
                draw.append(seeds[1])  # Seed 2 at bottom
            else:
                draw = raw_players[:]
                random.shuffle(draw)

            matches = []
            for i in range(0, len(draw) - 1, 2):
                matches.append((draw[i], draw[i+1]))
            if len(draw) % 2 != 0:
                matches.append((draw[-1], "BYE (Advances to Round 2)"))

            for idx, m in enumerate(matches):
                st.markdown(f"**Match {idx+1}:** 🏸 `{m[0]}` **vs** `{m[1]}`")
        else:
            st.subheader("Round-Robin League Schedule & Point Differential Table")
            clean_names = [p.split(":")[-1].strip() for p in raw_players]
            standings = {p: {"Matches": len(clean_names)-1, "Won": 0, "Lost": 0, "Pts For": 0, "Pts Against": 0, "Diff (Δ)": 0} for p in clean_names}

            fixtures = []
            for i in range(len(clean_names)):
                for j in range(i + 1, len(clean_names)):
                    fixtures.append((clean_names[i], clean_names[j]))

            df_standings = pd.DataFrame.from_dict(standings, orient="index").reset_index()
            df_standings.rename(columns={"index": "Player / Team"}, inplace=True)
            st.dataframe(df_standings, use_container_width=True)

            st.write("**Scheduled Fixtures:**")
            df_fix = pd.DataFrame(fixtures, columns=["Team 1", "Team 2"])
            df_fix["Status"] = "Pending"
            st.dataframe(df_fix, use_container_width=True)

# ==============================================================================
# MODULE 16: MULTI-COURT ALLOCATION BOARD
# ==============================================================================
elif nav_choice == "🏟️ Multi-Court Allocation Board":
    st.header("🏟️ Multi-Court Allocation Board")
    st.caption("Manage match flow and court assignments across up to 3 tournament courts.")

    c_cols = st.columns(3)
    for idx, (court_name, status) in enumerate(st.session_state.court_allocations.items()):
        with c_cols[idx]:
            st.markdown(f"### {court_name}")
            new_status = st.selectbox(f"Status for {court_name}", ["Available", "Warm-up", "Match in Progress", "Maintenance"], index=["Available", "Warm-up", "Match in Progress", "Maintenance"].index(status), key=f"court_{idx}")
            st.session_state.court_allocations[court_name] = new_status
            if new_status == "Available":
                st.success("🟢 Ready for Next Match")
            elif new_status == "Match in Progress":
                st.error("🔴 Court Busy")
            else:
                st.warning(f"🟡 {new_status}")

# ==============================================================================
# MODULE 17: LOCAL SPORTS GEAR & EQUIPMENT FINDER (GEO-AWARE)
# ==============================================================================
elif nav_choice == "🛍️ Local Equipment Finder":
    st.header("🛍️ Local Sports Gear & Equipment Finder")
    st.caption("Live geo-query powered by OpenStreetMap Nominatim and Overpass APIs (No API keys required).")

    col_country, col_city, col_btn = st.columns([2, 2, 1])
    with col_country:
        country_input = st.text_input("Country", value="Pakistan")
    with col_city:
        city_input = st.text_input("City", value="Peshawar")
    with col_btn:
        st.write("")
        st.write("")
        search_clicked = st.button("🔍 Find Stores", use_container_width=True)

    if search_clicked:
        with st.spinner(f"Locating sports & badminton gear stores in {city_input}, {country_input}..."):
            try:
                nominatim_url = f"https://nominatim.openstreetmap.org/search?city={urllib.parse.quote(city_input)}&country={urllib.parse.quote(country_input)}&format=json&limit=1"
                headers = {"User-Agent": "BadmintonProHub/1.0 (Educational sports tool)"}
                geo_resp = requests.get(nominatim_url, headers=headers, timeout=10).json()

                if geo_resp:
                    lat = float(geo_resp[0]["lat"])
                    lon = float(geo_resp[0]["lon"])

                    overpass_query = f"""
                    [out:json][timeout:15];
                    (
                      node["shop"="sports"](around:15000,{lat},{lon});
                      way["shop"="sports"](around:15000,{lat},{lon});
                      node["sport"="badminton"](around:15000,{lat},{lon});
                    );
                    out center tags;
                    """
                    overpass_url = "https://overpass-api.de/api/interpreter"
                    shop_resp = requests.post(overpass_url, data={"data": overpass_query}, timeout=15).json()

                    elements = shop_resp.get("elements", [])
                    stores_data = []

                    for el in elements:
                        tags = el.get("tags", {})
                        name = tags.get("name", "Local Sports / Badminton Outlet")
                        street = tags.get("addr:street", tags.get("addr:full", "City Center / Main Bazaar"))
                        shop_type = tags.get("shop", "Sports Shop")
                        store_query = urllib.parse.quote(f"{name} {city_input}")
                        gmaps_link = f"https://www.google.com/maps/search/?api=1&query={store_query}"

                        stores_data.append({
                            "Store Name": name,
                            "Category": shop_type.capitalize(),
                            "Location / Area": street,
                            "Google Maps Directions": gmaps_link,
                        })

                    if stores_data:
                        st.success(f"Found {len(stores_data)} sports & equipment outlets near {city_input}!")
                        df_stores = pd.DataFrame(stores_data)
                        st.dataframe(df_stores, column_config={"Google Maps Directions": st.column_config.LinkColumn("View on Google Maps")}, use_container_width=True, hide_index=True)
                    else:
                        direct_url = f"https://www.google.com/maps/search/badminton+sports+gear+store+in+{urllib.parse.quote(city_input)}"
                        st.markdown(f"👉 **[Search Badminton Stores in {city_input} on Google Maps]({direct_url})**")
                else:
                    st.error("Could not resolve location coordinates. Please verify city and country spelling.")
            except Exception:
                fallback_url = f"https://www.google.com/maps/search/badminton+shops+{urllib.parse.quote(city_input)}"
                st.markdown(f"👉 **[Click here to view Badminton Stores in {city_input}]({fallback_url})**")

# ==============================================================================
# MODULE 18: TRAINING SCHEDULE & SESSION ORGANIZER (PRESETS & RESETS)
# ==============================================================================
elif nav_choice == "📅 Training Schedule Planner":
    st.header("📅 Training Schedule & Session Organizer")
    st.caption("Plan, track, reset, and export customized badminton training routines.")

    st.subheader("⚡ Quick Routine Presets & Reset Controls")
    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    with ctl1:
        if st.button("🔄 Reset to Default Schedule", use_container_width=True):
            st.session_state.training_schedule = [dict(item) for item in DEFAULT_SCHEDULE]
            st.rerun()
    with ctl2:
        if st.button("🏆 Load Tournament Prep", use_container_width=True):
            st.session_state.training_schedule = [dict(item) for item in PRESET_TOURNAMENT_PREP]
            st.rerun()
    with ctl3:
        if st.button("⚡ Load Footwork Focus", use_container_width=True):
            st.session_state.training_schedule = [dict(item) for item in PRESET_FOOTWORK_FOCUS]
            st.rerun()
    with ctl4:
        if st.button("🗑️ Clear Entire Schedule", use_container_width=True):
            st.session_state.training_schedule = []
            st.rerun()

    with st.expander("➕ Add Custom Training Session"):
        with st.form("new_drill_form"):
            form_day = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            form_cat = st.selectbox("Category", ["On-Court Drills", "Match Play", "Gym & Strength", "Footwork & Agility", "Cardio / Conditioning"])
            form_drill = st.text_input("Drill Description", value="15 min Shadow footwork + 30 min smashing defense")
            submitted = st.form_submit_button("Add Session")
            if submitted and form_drill:
                st.session_state.training_schedule.append({"day": form_day, "category": form_cat, "drill": form_drill, "completed": False})
                st.rerun()

    st.markdown("---")
    st.subheader("Current Weekly Sessions")

    if not st.session_state.training_schedule:
        st.info("Your schedule is currently empty. Click one of the preset buttons above or add a drill.")
    else:
        indices_to_delete = []
        for idx, session in enumerate(st.session_state.training_schedule):
            c1, c2, c3, c4, c5 = st.columns([1.5, 2, 4.5, 1, 0.8])
            c1.markdown(f"**{session['day']}**")
            c2.markdown(f"`{session['category']}`")
            c3.markdown(session["drill"])
            completed_val = c4.checkbox("Done", value=session["completed"], key=f"drill_{idx}")
            st.session_state.training_schedule[idx]["completed"] = completed_val
            if c5.button("✖", key=f"del_{idx}", help="Remove this drill"):
                indices_to_delete.append(idx)

        if indices_to_delete:
            for i in sorted(indices_to_delete, reverse=True):
                st.session_state.training_schedule.pop(i)
            st.rerun()

    st.markdown("---")
    completed_count = sum(1 for s in st.session_state.training_schedule if s["completed"])
    total_count = len(st.session_state.training_schedule)
    progress_val = completed_count / total_count if total_count > 0 else 0
    st.write(f"**Weekly Completion Rate:** {completed_count}/{total_count} sessions ({int(progress_val*100)}%)")
    st.progress(progress_val)

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        json_schedule = json.dumps(st.session_state.training_schedule, indent=2)
        st.download_button("📥 Export Schedule as JSON", data=json_schedule, file_name="badminton_training_plan.json", mime="application/json", use_container_width=True)
    with exp_col2:
        if st.session_state.training_schedule:
            df_sched = pd.DataFrame(st.session_state.training_schedule)
            csv_sched = df_sched.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Schedule as CSV", data=csv_sched, file_name="badminton_training_plan.csv", mime="text/csv", use_container_width=True)

# ==============================================================================
# MODULE 19: HEALTH, FITNESS & FATIGUE TRACKER
# ==============================================================================
elif nav_choice == "❤️ Health, Fitness & Fatigue":
    st.header("❤️ Health, Fitness & Fatigue Tracker")
    st.caption("Personalized conditioning metrics, heart-rate zones, and injury prevention recovery protocols.")

    col_bio1, col_bio2, col_bio3 = st.columns(3)
    with col_bio1:
        user_age = st.number_input("Age (years)", min_value=12, max_value=85, value=25)
    with col_bio2:
        user_weight = st.number_input("Weight (kg)", min_value=35.0, max_value=160.0, value=70.0, step=0.5)
    with col_bio3:
        resting_hr = st.number_input("Resting Heart Rate (bpm)", min_value=40, max_value=100, value=65)

    max_hr = 220 - user_age
    hrr = max_hr - resting_hr

    st.subheader("🎯 Heart Rate Training Zones (Karvonen Method)")
    hz1, hz2, hz3 = st.columns(3)
    hz1.metric("Max Heart Rate", f"{max_hr} bpm")
    hz2.metric("Aerobic Base (70%)", f"{int(resting_hr + 0.70 * hrr)} bpm")
    hz3.metric("Anaerobic Threshold (85%)", f"{int(resting_hr + 0.85 * hrr)} bpm")

    st.markdown("---")
    st.subheader("🔥 Match Energy Expenditure & Hydration")

    col_dur, col_intense = st.columns(2)
    with col_dur:
        duration_mins = st.slider("Session Duration (minutes)", min_value=15, max_value=180, value=60, step=15)
    with col_intense:
        intensity_level = st.select_slider(
            "Session Intensity",
            options=["Light (Casual Rally)", "Moderate (Drills / Doubles)", "High (Competitive Singles Match)", "Extreme (Tournament Match Play)"],
            value="High (Competitive Singles Match)",
        )

    met_map = {
        "Light (Casual Rally)": 5.5,
        "Moderate (Drills / Doubles)": 7.0,
        "High (Competitive Singles Match)": 8.5,
        "Extreme (Tournament Match Play)": 10.0,
    }
    met = met_map[intensity_level]

    duration_hours = duration_mins / 60.0
    calories_burned = int(met * user_weight * duration_hours)
    hydration_ml = int(duration_hours * 800)

    c_box1, c_box2 = st.columns(2)
    c_box1.metric("Estimated Calories Burned", f"{calories_burned} kcal")
    c_box2.metric("Recommended Fluid Intake", f"{hydration_ml} ml")

    st.markdown("---")
    st.subheader("🛡️ Fatigue Recovery & Injury Prevention Protocol")

    fatigue_score = (met / 10.0) * (duration_mins / 60.0) * 10
    if fatigue_score < 4.0:
        recovery_time = "12 - 18 Hours"
        advice = "Low muscle strain. Standard cool-down stretching is sufficient."
    elif fatigue_score < 8.0:
        recovery_time = "24 Hours"
        advice = "Moderate muscle fatigue. Ensure adequate protein synthesis and 7-8 hours sleep."
    else:
        recovery_time = "36 - 48 Hours"
        advice = "High neuromuscular fatigue. Strict active recovery recommended to protect tendons."

    rf1, rf2 = st.columns(2)
    rf1.metric("Recommended Rest Interval", recovery_time)
    rf2.info(advice)

# ==============================================================================
# MODULE 20: RACKET & TENSION OPTIMIZER
# ==============================================================================
elif nav_choice == "⚙️ Racket & Tension Optimizer":
    st.header("⚙️ Racket & Tension Optimizer")
    st.caption("Calculate optimal string tension, string gauge, and racket balance for your playing style.")

    col_lvl, col_style, col_shuttle = st.columns(3)
    with col_lvl:
        player_exp = st.selectbox("Player Experience Level", ["Beginner (0-1 yr)", "Intermediate (1-3 yrs)", "Advanced / Club Player", "Professional / Elite"])
    with col_style:
        play_style = st.selectbox("Playing Style", ["Aggressive Attacker / Smasher", "Control & Tactical Net Play", "All-Round / Defensive Counter"])
    with col_shuttle:
        shuttle_type = st.selectbox("Shuttlecock Type", ["Feather Shuttlecocks (Goose/Duck)", "Nylon / Plastic Shuttles (Mavis)"])

    base_tensions = {
        "Beginner (0-1 yr)": (20, 22),
        "Intermediate (1-3 yrs)": (23, 25),
        "Advanced / Club Player": (26, 28),
        "Professional / Elite": (29, 32),
    }
    low_t, high_t = base_tensions[player_exp]

    if shuttle_type.startswith("Nylon"):
        low_t -= 2
        high_t -= 2
        shuttle_note = "Plastic shuttles are heavier and stiffer; lower tension protects your racket frame."
    else:
        shuttle_note = "Feather shuttles provide clean aerodynamic trajectory suited for crisp string beds."

    racket_balance = "Head-Heavy (e.g., Yonex Astrox / Victor Thruster)" if "Attacker" in play_style else ("Even-Balance (e.g., Yonex Arcsaber)" if "Control" in play_style else "Head-Light (e.g., Yonex Nanoflare)")
    rec_string = "Yonex BG66 Ultimax (0.65mm) / Li-Ning No.1" if "Advanced" in player_exp or "Attacker" in play_style else "Yonex BG65 (0.70mm) / Victor VBS-70 (Ultra Durable)"

    st.markdown("---")
    st.subheader("📋 Personalized Equipment Recommendation")

    t1, t2, t3 = st.columns(3)
    t1.metric("Recommended Tension Range", f"{low_t} - {high_t} lbs")
    t2.metric("Optimal Racket Balance", racket_balance.split()[0])
    t3.metric("Recommended String Gauge", rec_string.split()[1])

    st.info(f"**Shuttlecock Factor:** {shuttle_note}")
