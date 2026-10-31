import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import urllib.parse
from datetime import datetime
from openai import OpenAI

# ==============================================================================
# PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="Badminton Pro-Hub",
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
        padding: 14px;
        background-color: #f0f7ff;
        font-weight: 600;
        text-align: center;
    }
    .stButton>button {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# DEFAULT TRAINING PRESETS
# ==============================================================================
DEFAULT_SCHEDULE = [
    {"day": "Monday", "category": "Footwork & Agility", "drill": "6-Corner Shadow Footwork (4x20 reps)", "completed": False},
    {"day": "Tuesday", "category": "Gym & Strength", "drill": "Squats, Lunges & Shoulder Rotator Cuff stability", "completed": False},
    {"day": "Wednesday", "category": "On-Court Drills", "drill": "Net Kills, Drops & Half-Smash variations", "completed": False},
    {"day": "Thursday", "category": "Cardio", "drill": "HIIT Court Sprints (30s on / 30s off)", "completed": False},
    {"day": "Friday", "category": "Match Play", "drill": "Full competitive best-of-3 sets", "completed": False},
]

PRESET_TOURNAMENT_PREP = [
    {"day": "Monday", "category": "Match Play", "drill": "High-intensity simulation match + tactical debrief", "completed": False},
    {"day": "Tuesday", "category": "On-Court Drills", "drill": "Smash-defense transition + cross-court blocks (40 mins)", "completed": False},
    {"day": "Wednesday", "category": "Footwork & Agility", "drill": "Deceptive split-step reactions + multi-shuttle feeds", "completed": False},
    {"day": "Thursday", "category": "Cardio", "drill": "200m interval sprints x 6 + active stretch", "completed": False},
    {"day": "Friday", "category": "On-Court Drills", "drill": "Service precision drill (50 short serves, 50 flick serves)", "completed": False},
    {"day": "Saturday", "category": "Match Play", "drill": "Full 3-set test match with tournament line calls", "completed": False},
]

PRESET_FOOTWORK_FOCUS = [
    {"day": "Monday", "category": "Footwork & Agility", "drill": "Corner lunging stability with 3-second hold (4x15)", "completed": False},
    {"day": "Tuesday", "category": "Cardio", "drill": "Agility ladder drills + skipping rope (2,000 double unders)", "completed": False},
    {"day": "Wednesday", "category": "Footwork & Agility", "drill": "Fast backward scissor kicks from front to baseline", "completed": False},
    {"day": "Thursday", "category": "Gym & Strength", "drill": "Calf raises, hamstring curls, and explosive box jumps", "completed": False},
    {"day": "Friday", "category": "Footwork & Agility", "drill": "Blind reaction drill with coach finger signals", "completed": False},
]

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "score_p1" not in st.session_state:
    st.session_state.score_p1 = 0
if "score_p2" not in st.session_state:
    st.session_state.score_p2 = 0
if "server" not in st.session_state:
    st.session_state.server = "Player 1"
if "match_history" not in st.session_state:
    st.session_state.match_history = []
if "p1_name" not in st.session_state:
    st.session_state.p1_name = "Player 1"
if "p2_name" not in st.session_state:
    st.session_state.p2_name = "Player 2"
if "game_over" not in st.session_state:
    st.session_state.game_over = False
if "winner" not in st.session_state:
    st.session_state.winner = ""

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Welcome to Badminton Pro-Hub! Ask me anything regarding official BWF rules, singles/doubles positioning, tactical setups, shot choices, or conditioning.",
        }
    ]

if "training_schedule" not in st.session_state:
    st.session_state.training_schedule = [dict(item) for item in DEFAULT_SCHEDULE]

# ==============================================================================
# ENGINE & CALCULATION HELPERS
# ==============================================================================
def check_game_status(s1, s2):
    """Enforces BWF 21-pt rules, deuce past 20-20, and 30-pt sudden death cap."""
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
    """Simulates match outcomes factoring in serving momentum."""
    if s1 == s2 == 0:
        return 50.0, 50.0

    total_rallies = len(st.session_state.match_history)
    if total_rallies > 5:
        p1_rallies_won = sum(1 for r in st.session_state.match_history if r["winner"] == st.session_state.p1_name)
        base_p1_prob = p1_rallies_won / total_rallies
    else:
        base_p1_prob = 0.50

    p1_wins = 0
    for _ in range(n_simulations):
        sim_s1, sim_s2 = s1, s2
        sim_srv = server

        while True:
            current_prob = base_p1_prob + (0.03 if sim_srv == "Player 1" else -0.03)
            current_prob = max(0.15, min(0.85, current_prob))

            if np.random.rand() < current_prob:
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
    p2_pct = round(100.0 - p1_pct, 1)
    return p1_pct, p2_pct

RULE_KNOWLEDGE_BASE = {
    "service height": "Under BWF Law 9.1.6, the whole shuttle must be below 1.15 meters (3.77 feet) from the court surface at the instant of racket impact.",
    "fault": "Common BWF faults include: shuttle landing outside lines, shuttle passing through or under the net, touching the net during play, striking the shuttle before crossing the net, and double-hitting.",
    "let": "A 'Let' halts play and replays the rally: called when an unforeseen hazard occurs, the shuttle breaks mid-rally, the receiver was not ready, or a stray shuttle enters the court.",
    "interval": "A 60-second break occurs when the leading score reaches 11 points in each set. A 120-second (2-minute) break is granted between consecutive sets.",
    "doubles positioning": "Attacking setup: Front-and-back formation (rear player smashes/drops, front player intercepts net returns). Defensive setup: Side-by-side midcourt to cover all baseline smash angles.",
    "singles positioning": "Recover to the central T base (1-2 steps behind the service line intersection) with a synchronized split-step at the opponent's racket impact.",
}

def match_internal_rules(query_lower):
    for key, answer in RULE_KNOWLEDGE_BASE.items():
        if key in query_lower:
            return f"**BWF Official Rule Directive:**\n\n{answer}"
    if "serve" in query_lower:
        return "BWF Law 9: Both feet must stay stationary inside the service court. The racket must hit the shuttle base below 1.15m in an uninterrupted forward stroke."
    elif "net" in query_lower:
        return "BWF Law 13.4: Touching the net with your racket, body, or clothing while the shuttle is live is an immediate fault."
    return (
        "**BWF Rule Engine (Offline Mode):**\n\n"
        "Ask about *'service height'*, *'touching net'*, *'let calls'*, *'intervals'*, or *'doubles positioning'*. "
        "Set `GROQ_API_KEY` in Streamlit Secrets to enable open-ended AI tactical advice."
    )

def get_groq_openai_response(user_query, api_key):
    """Queries OpenAI models hosted on Groq using the OpenAI Python SDK."""
    if not api_key:
        return match_internal_rules(user_query.lower())

    openai_models_on_groq = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    ]

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        system_prompt = (
            "You are an elite BWF Certified High-Performance Badminton Coach and International Umpire. "
            "Provide clear, tactical, rules-accurate, and concise advice. "
            "Address footwork, racket tension, singles/doubles strategy, or BWF laws authoritatively."
        )

        for model_id in openai_models_on_groq:
            try:
                chat_completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                    ],
                    temperature=0.3,
                    max_tokens=600,
                )
                return chat_completion.choices[0].message.content
            except Exception as inner_e:
                if "404" in str(inner_e) or "model_not_found" in str(inner_e):
                    continue
                raise inner_e

    except Exception as e:
        return f"*(Groq Service Notice: {e}. Defaulting to offline knowledge base)*\n\n" + match_internal_rules(user_query.lower())

    return match_internal_rules(user_query.lower())

# ==============================================================================
# ACCURATE 2D BADMINTON COURT RADAR
# ==============================================================================
def draw_tactical_court(server_label, p1_name, p2_name, is_even):
    """
    Renders accurate 2D BWF court map dynamically switching:
    - Player 1 serving from bottom half
    - Player 2 serving from top half
    - Right Box (Even score) vs Left Box (Odd score)
    - Diagonally opposite receiver target box
    """
    fig = go.Figure()

    # Full Court Green Surface (6.10m wide x 13.40m long)
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=6.1, y1=13.4,
        line=dict(color="#ffffff", width=3),
        fillcolor="#1e824c",
        opacity=0.35,
    )

    # Net (Midcourt y = 6.70m)
    fig.add_shape(type="line", x0=-0.3, y0=6.7, x1=6.4, y1=6.7, line=dict(color="#ffffff", width=4, dash="dash"))

    # Singles inner side tramlines (0.46m from boundary)
    fig.add_shape(type="line", x0=0.46, y0=0, x1=0.46, y1=13.4, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=5.64, y0=0, x1=5.64, y1=13.4, line=dict(color="#ffffff", width=2))

    # Short service lines (1.98m from net -> y = 4.72m and y = 8.68m)
    fig.add_shape(type="line", x0=0, y0=4.72, x1=6.1, y1=4.72, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=0, y0=8.68, x1=6.1, y1=8.68, line=dict(color="#ffffff", width=2))

    # Doubles long service lines (0.76m inside baselines)
    fig.add_shape(type="line", x0=0, y0=0.76, x1=6.1, y1=0.76, line=dict(color="#dcdde1", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0, y0=12.64, x1=6.1, y1=12.64, line=dict(color="#dcdde1", width=1, dash="dot"))

    # Center service lines (x = 3.05m)
    fig.add_shape(type="line", x0=3.05, y0=0, x1=3.05, y1=4.72, line=dict(color="#ffffff", width=2))
    fig.add_shape(type="line", x0=3.05, y0=8.68, x1=3.05, y1=13.4, line=dict(color="#ffffff", width=2))

    # Compute server & diagonal target receiver boxes
    if server_label == "Player 1":
        serving_player_name = p1_name
        receiving_player_name = p2_name
        if is_even:
            # Bottom-Right server -> Top-Left receiver (viewer perspective)
            srv_x0, srv_x1, srv_y0, srv_y1 = 3.05, 5.64, 0, 4.72
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 0.46, 3.05, 8.68, 13.4
            side_str = "Right Court (Even)"
        else:
            # Bottom-Left server -> Top-Right receiver
            srv_x0, srv_x1, srv_y0, srv_y1 = 0.46, 3.05, 0, 4.72
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 3.05, 5.64, 8.68, 13.4
            side_str = "Left Court (Odd)"
    else:
        serving_player_name = p2_name
        receiving_player_name = p1_name
        if is_even:
            # Top-Left server -> Bottom-Right receiver
            srv_x0, srv_x1, srv_y0, srv_y1 = 0.46, 3.05, 8.68, 13.4
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 3.05, 5.64, 0, 4.72
            side_str = "Right Court (Even)"
        else:
            # Top-Right server -> Bottom-Left receiver
            srv_x0, srv_x1, srv_y0, srv_y1 = 3.05, 5.64, 8.68, 13.4
            rcv_x0, rcv_x1, rcv_y0, rcv_y1 = 0.46, 3.05, 0, 4.72
            side_str = "Left Court (Odd)"

    # Highlight Active Serving Box
    fig.add_shape(
        type="rect",
        x0=srv_x0, y0=srv_y0, x1=srv_x1, y1=srv_y1,
        fillcolor="#f39c12", opacity=0.7,
        line=dict(color="#d35400", width=3),
    )
    fig.add_annotation(
        x=(srv_x0 + srv_x1) / 2, y=(srv_y0 + srv_y1) / 2,
        text=f"🏸 SERVING<br><b>{serving_player_name}</b>",
        showarrow=False,
        font=dict(color="#ffffff", size=10, family="Arial Black"),
    )

    # Highlight Target Receiver Box
    fig.add_shape(
        type="rect",
        x0=rcv_x0, y0=rcv_y0, x1=rcv_x1, y1=rcv_y1,
        fillcolor="#2980b9", opacity=0.6,
        line=dict(color="#1abc9c", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=(rcv_x0 + rcv_x1) / 2, y=(rcv_y0 + rcv_y1) / 2,
        text=f"🎯 RECEIVING<br><b>{receiving_player_name}</b>",
        showarrow=False,
        font=dict(color="#ffffff", size=10, family="Arial Black"),
    )

    fig.add_annotation(x=3.05, y=-0.5, text=f"🔻 {p1_name} Side", showarrow=False, font=dict(color="#ecf0f1", size=11))
    fig.add_annotation(x=3.05, y=13.9, text=f"🔺 {p2_name} Side", showarrow=False, font=dict(color="#ecf0f1", size=11))

    fig.update_layout(
        title=f"Court Radar: {serving_player_name} serving from {side_str}",
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
        "🤖 AI Rules & Tactical Coach",
        "🛍️ Local Equipment Finder",
        "📅 Training Schedule Planner",
        "❤️ Health, Fitness & Fatigue",
        "⚙️ Racket & Tension Optimizer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**BWF Rule Reference:**")
st.sidebar.caption("• 21 pts per game (rally point system)\n• 2-pt lead needed past 20-20\n• 30-pt sudden death cap\n• Even score = Right court, Odd = Left")

# ==============================================================================
# MODULE 1: LIVE TACTICAL SCORER & WIN FORECASTER
# ==============================================================================
if nav_choice == "🏸 Live Tactical Scorer":
    st.header("🏸 Live Tactical Scorer & Win Forecaster")

    col_name1, col_name2, col_reset = st.columns([2, 2, 1])
    with col_name1:
        st.session_state.p1_name = st.text_input("Player / Pair 1", value=st.session_state.p1_name)
    with col_name2:
        st.session_state.p2_name = st.text_input("Player / Pair 2", value=st.session_state.p2_name)
    with col_reset:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Match", use_container_width=True):
            st.session_state.score_p1 = 0
            st.session_state.score_p2 = 0
            st.session_state.server = "Player 1"
            st.session_state.match_history = []
            st.session_state.game_over = False
            st.session_state.winner = ""
            st.rerun()

    st.markdown("---")

    server_name = st.session_state.p1_name if st.session_state.server == "Player 1" else st.session_state.p2_name
    server_score = st.session_state.score_p1 if st.session_state.server == "Player 1" else st.session_state.score_p2
    is_even = (server_score % 2 == 0)
    court_side = "RIGHT" if is_even else "LEFT"

    top_c1, top_c2, top_c3 = st.columns([1.5, 1.5, 2])
    with top_c1:
        st.metric(label=f"{st.session_state.p1_name} Score", value=st.session_state.score_p1)
    with top_c2:
        st.metric(label=f"{st.session_state.p2_name} Score", value=st.session_state.score_p2)
    with top_c3:
        st.markdown(
            f"""
            <div class="court-box">
                🏸 Serving: <strong>{server_name}</strong><br>
                Court Position: <strong>{court_side} Box ({'Even' if is_even else 'Odd'} Score)</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.game_over:
        st.success(f"🏆 Match Finished! Winner: **{st.session_state.winner}**")

    st.subheader("Point Logger")
    shot_types = ["Smash Winner", "Drop Winner", "Net Kill", "Opponent Out", "Opponent Net Error", "Service Fault"]

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.markdown(f"**Award Point to: {st.session_state.p1_name}**")
        p1_shot = st.selectbox("Point Outcome (P1)", shot_types, key="p1_shot_select")
        if st.button(f"➕ Score for {st.session_state.p1_name}", disabled=st.session_state.game_over, use_container_width=True):
            st.session_state.score_p1 += 1
            st.session_state.server = "Player 1"
            st.session_state.match_history.append({
                "rally": len(st.session_state.match_history) + 1,
                "winner": st.session_state.p1_name,
                "shot": p1_shot,
                "p1_score": st.session_state.score_p1,
                "p2_score": st.session_state.score_p2,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.winner = win
            st.rerun()

    with btn_col2:
        st.markdown(f"**Award Point to: {st.session_state.p2_name}**")
        p2_shot = st.selectbox("Point Outcome (P2)", shot_types, key="p2_shot_select")
        if st.button(f"➕ Score for {st.session_state.p2_name}", disabled=st.session_state.game_over, use_container_width=True):
            st.session_state.score_p2 += 1
            st.session_state.server = "Player 2"
            st.session_state.match_history.append({
                "rally": len(st.session_state.match_history) + 1,
                "winner": st.session_state.p2_name,
                "shot": p2_shot,
                "p1_score": st.session_state.score_p1,
                "p2_score": st.session_state.score_p2,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.winner = win
            st.rerun()

    st.markdown("---")
    sim_col, court_col = st.columns([3, 2])
    with sim_col:
        st.subheader("🎲 Monte Carlo Win Forecaster (1,000 Simulations)")
        if not st.session_state.game_over:
            p1_prob, p2_prob = run_monte_carlo(
                st.session_state.score_p1, st.session_state.score_p2, st.session_state.server
            )
        else:
            p1_prob = 100.0 if st.session_state.winner == st.session_state.p1_name else 0.0
            p2_prob = 100.0 - p1_prob

        m1, m2 = st.columns(2)
        m1.metric(f"{st.session_state.p1_name} Win Prob.", f"{p1_prob}%")
        m2.metric(f"{st.session_state.p2_name} Win Prob.", f"{p2_prob}%")
        st.progress(int(p1_prob))

    with court_col:
        st.plotly_chart(
            draw_tactical_court(
                server_label=st.session_state.server,
                p1_name=st.session_state.p1_name,
                p2_name=st.session_state.p2_name,
                is_even=is_even,
            ),
            use_container_width=True,
        )

    if st.session_state.match_history:
        st.markdown("---")
        st.subheader("📊 Match Progression & Shot Breakdown")
        df_history = pd.DataFrame(st.session_state.match_history)

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(x=df_history["rally"], y=df_history["p1_score"], mode="lines+markers", name=st.session_state.p1_name, line=dict(color="#1f77b4", width=3)))
            fig_timeline.add_trace(go.Scatter(x=df_history["rally"], y=df_history["p2_score"], mode="lines+markers", name=st.session_state.p2_name, line=dict(color="#ff7f0e", width=3)))
            fig_timeline.update_layout(title="Rally Progression Timeline", xaxis_title="Rally Number", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_timeline, use_container_width=True)

        with graph_col2:
            shot_counts = df_history.groupby(["winner", "shot"]).size().reset_index(name="count")
            fig_shots = px.bar(shot_counts, x="shot", y="count", color="winner", barmode="group", title="Points Won by Shot Type", template="plotly_white")
            st.plotly_chart(fig_shots, use_container_width=True)

        csv_data = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Full Match Log (CSV)", data=csv_data, file_name="badminton_match_log.csv", mime="text/csv")

# ==============================================================================
# MODULE 2: AI RULES ARBITER & TACTICAL COACH (GROQ OPENAI MODELS)
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
# MODULE 3: LOCAL SPORTS GEAR & EQUIPMENT FINDER (GEO-AWARE)
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
                        st.dataframe(
                            df_stores,
                            column_config={
                                "Google Maps Directions": st.column_config.LinkColumn("View on Google Maps")
                            },
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.warning(f"No specific sports shop nodes mapped on OpenStreetMap for {city_input}. Direct query generated below:")
                        direct_url = f"https://www.google.com/maps/search/badminton+sports+gear+store+in+{urllib.parse.quote(city_input)}"
                        st.markdown(f"👉 **[Search Badminton Stores in {city_input} on Google Maps]({direct_url})**")
                else:
                    st.error("Could not resolve location coordinates. Please verify city and country spelling.")
            except Exception:
                st.info("Direct OSM connection timed out. You can browse local shops directly via Google Maps:")
                fallback_url = f"https://www.google.com/maps/search/badminton+shops+{urllib.parse.quote(city_input)}"
                st.markdown(f"👉 **[Click here to view Badminton Stores in {city_input}]({fallback_url})**")

    st.markdown("---")
    st.subheader("🌐 Global & Regional Equipment Portals")
    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.markdown("**Yonex Official**\n[yonex.com](https://www.yonex.com)")
    with col_r2:
        st.markdown("**Victor Rackets**\n[victorsport.com](https://www.victorsport.com)")
    with col_r3:
        st.markdown("**Li-Ning Badminton**\n[lining.com](https://www.lining.com)")
    with col_r4:
        st.markdown("**Local / Global Retail**\n[Amazon](https://www.amazon.com) | [Daraz](https://www.daraz.pk) | [Decathlon](https://www.decathlon.com)")

# ==============================================================================
# MODULE 4: TRAINING SCHEDULE & SESSION ORGANIZER (PRESETS & RESETS)
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
                st.session_state.training_schedule.append({
                    "day": form_day,
                    "category": form_cat,
                    "drill": form_drill,
                    "completed": False,
                })
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
        st.download_button(
            label="📥 Export Schedule as JSON",
            data=json_schedule,
            file_name="badminton_training_plan.json",
            mime="application/json",
            use_container_width=True,
        )
    with exp_col2:
        if st.session_state.training_schedule:
            df_sched = pd.DataFrame(st.session_state.training_schedule)
            csv_sched = df_sched.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Schedule as CSV",
                data=csv_sched,
                file_name="badminton_training_plan.csv",
                mime="text/csv",
                use_container_width=True,
            )

# ==============================================================================
# MODULE 5: HEALTH, FITNESS & FATIGUE TRACKER
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
        advice = "Moderate muscle fatigue. Ensure adequate protein synthesis and 7-8 hours sleep before next on-court session."
    else:
        recovery_time = "36 - 48 Hours"
        advice = "High neuromuscular fatigue. Strict active recovery recommended to protect tendons."

    rf1, rf2 = st.columns(2)
    rf1.metric("Recommended Rest Interval", recovery_time)
    rf2.info(advice)

    with st.expander("🧘 Essential Badminton Injury Prevention Protocols"):
        st.markdown(
            """
            - **Achilles Tendon & Calf:** Dynamic eccentric heel drops prior to matches; static wall calf stretch for 45s post-match.
            - **Patellar Tendon (Jumper's Knee):** Foam roll quadriceps and IT band; incorporate isometric Spanish squats.
            - **Shoulder Rotator Cuff:** Internal & external resistance band rotations (3x15) to maintain posterior shoulder balance against explosive smash velocity.
            - **Ankle Sprain Mitigation:** High-friction non-marking badminton shoes and lateral hopping drills to reinforce stabilizing ankle ligaments.
            """
        )

# ==============================================================================
# MODULE 6: RACKET & TENSION OPTIMIZER
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
        shuttle_note = "Plastic shuttles are heavier and stiffer; lower tension protects your racket frame and prevents golfer's elbow."
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

    with st.expander("🔬 Physics of String Tension (Repulsion vs. Control)"):
        st.markdown(
            """
            - **Lower Tension (20-23 lbs):** Generates a larger trampoline effect with a huge sweet spot. Produces effortless depth for clears and protects the forearm tendons from vibrations.
            - **Higher Tension (26-30+ lbs):** Smaller sweet spot requiring precise hitting technique. Produces crisp feedback, fast shuttle repulsion off the racket face, and superior spin control on net slices.
            """
        )
