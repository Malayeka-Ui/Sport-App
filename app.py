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
# PAGE CONFIGURATION & STYLING
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
        border-radius: 6px;
        padding: 12px;
        background-color: #f0f7ff;
        font-weight: 600;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    st.session_state.training_schedule = [
        {"day": "Monday", "category": "Footwork & Agility", "drill": "6-Corner Shadow Footwork (4x20 reps)", "completed": False},
        {"day": "Tuesday", "category": "Gym & Strength", "drill": "Squats, Lunges & Shoulder Rotator Cuff stability", "completed": False},
        {"day": "Wednesday", "category": "On-Court Drills", "drill": "Net Kills, Drops & Half-Smash variations", "completed": False},
        {"day": "Thursday", "category": "Cardio", "drill": "HIIT Court Sprints (30s on / 30s off)", "completed": False},
        {"day": "Friday", "category": "Match Play", "drill": "Full competitive best-of-3 sets", "completed": False},
    ]

# ==============================================================================
# HELPER FUNCTIONS: RULES, SIMULATION & GROQ OPENAI INTEGRATION
# ==============================================================================
def check_game_status(s1, s2):
    """Evaluates standard BWF 21-point rules, deuce past 20-20, and sudden-death cap at 30."""
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
    """Simulates match outcomes using serving momentum advantage."""
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
            return f"**BWF Rule Directive:**\n\n{answer}"
    if "serve" in query_lower:
        return "BWF Law 9: Both feet must stay stationary inside the service court. The racket must hit the shuttle base below 1.15m in an uninterrupted forward stroke."
    elif "net" in query_lower:
        return "BWF Law 13.4: Touching the net with your racket, body, or clothing while the shuttle is live is an immediate fault."
    return (
        "**BWF Rule Engine (Offline Mode):**\n\n"
        "Ask about *'service height'*, *'touching net'*, *'let calls'*, *'intervals'*, or *'doubles positioning'*. "
        "Enter your free Groq API key in the configuration panel above to unlock open-ended tactics and interactive advice."
    )

def get_groq_openai_response(user_query, api_key, model_name):
    """Executes chat completions via Groq's OpenAI-compatible API endpoint."""
    if api_key:
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
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"*(Groq API notice: {e}. Defaulting to offline knowledge base)*\n\n" + match_internal_rules(user_query.lower())
    return match_internal_rules(user_query.lower())

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
    court_side = "RIGHT Court (Even Score)" if server_score % 2 == 0 else "LEFT Court (Odd Score)"

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric(label=f"{st.session_state.p1_name} Score", value=st.session_state.score_p1)
    with sc2:
        st.metric(label=f"{st.session_state.p2_name} Score", value=st.session_state.score_p2)
    with sc3:
        st.markdown(
            f"""
            <div class="court-box">
                🏸 Serving: {server_name}<br>
                Service Box: {court_side}
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state.game_over:
        st.success(f"🏆 Game Over! Winner: **{st.session_state.winner}**")

    st.subheader("Point Logger")
    shot_types = ["Smash Winner", "Drop Winner", "Net Kill", "Opponent Out", "Opponent Net Error", "Service Fault"]

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
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
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.winner = win
            st.rerun()

    with col_btn2:
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
            })
            ended, win = check_game_status(st.session_state.score_p1, st.session_state.score_p2)
            if ended:
                st.session_state.game_over = True
                st.session_state.winner = win
            st.rerun()

    st.markdown("---")
    st.subheader("🎲 Monte Carlo Win Forecaster (1,000 Simulations)")
    if not st.session_state.game_over:
        p1_prob, p2_prob = run_monte_carlo(
            st.session_state.score_p1, st.session_state.score_p2, st.session_state.server
        )
    else:
        p1_prob = 100.0 if st.session_state.winner == st.session_state.p1_name else 0.0
        p2_prob = 100.0 - p1_prob

    m1, m2 = st.columns(2)
    m1.metric(f"{st.session_state.p1_name} Projected Win Rate", f"{p1_prob}%")
    m2.metric(f"{st.session_state.p2_name} Projected Win Rate", f"{p2_prob}%")
    st.progress(int(p1_prob))

    if st.session_state.match_history:
        st.markdown("---")
        st.subheader("📊 Match Progression & Shot Analytics")
        df_history = pd.DataFrame(st.session_state.match_history)

        graph_col1, graph_col2 = st.columns(2)
        with graph_col1:
            fig_timeline = go.Figure()
            fig_timeline.add_trace(go.Scatter(x=df_history["rally"], y=df_history["p1_score"], mode="lines+markers", name=st.session_state.p1_name, line=dict(color="#1f77b4", width=3)))
            fig_timeline.add_trace(go.Scatter(x=df_history["rally"], y=df_history["p2_score"], mode="lines+markers", name=st.session_state.p2_name, line=dict(color="#ff7f0e", width=3)))
            fig_timeline.update_layout(title="Rally Score Progression", xaxis_title="Rally #", yaxis_title="Score", template="plotly_white")
            st.plotly_chart(fig_timeline, use_container_width=True)

        with graph_col2:
            shot_counts = df_history.groupby(["winner", "shot"]).size().reset_index(name="count")
            fig_shots = px.bar(shot_counts, x="shot", y="count", color="winner", barmode="group", title="Points Won by Shot Type", template="plotly_white")
            st.plotly_chart(fig_shots, use_container_width=True)

# ==============================================================================
# MODULE 2: AI RULES ARBITER & TACTICAL COACH (GROQ VIA OPENAI SDK)
# ==============================================================================
elif nav_choice == "🤖 AI Rules & Tactical Coach":
    st.header("🤖 AI Rules Arbiter & Tactical Coach")
    st.caption("Powered by Groq Inference via the OpenAI Python Client & BWF Regulations Engine")

    default_groq_key = st.secrets.get("GROQ_API_KEY", "")

    with st.expander("🔑 Configure Groq OpenAI-Compatible Settings"):
        groq_api_key = st.text_input(
            "Groq API Key (starts with gsk_):",
            value=default_groq_key,
            type="password",
            help="Free key from console.groq.com. If omitted, the app uses built-in BWF rules.",
        )
        model_choice = st.selectbox(
            "Select Groq Model:",
            ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            index=0,
            help="70B offers the deepest tactical reasoning; 8B offers maximum generation speed.",
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about rules, tactics, shot selection, or drills...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("AI Coach analyzing with Groq..."):
            response_text = get_groq_openai_response(user_input, groq_api_key, model_choice)

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
# MODULE 4: TRAINING SCHEDULE & SESSION ORGANIZER
# ==============================================================================
elif nav_choice == "📅 Training Schedule Planner":
    st.header("📅 Training Schedule & Session Organizer")
    st.caption("Plan, track, and export your weekly badminton training program.")

    with st.expander("➕ Add New Training Session"):
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

    st.subheader("Your Weekly Sessions")
    for idx, session in enumerate(st.session_state.training_schedule):
        c1, c2, c3, c4 = st.columns([1.5, 2, 4, 1])
        c1.markdown(f"**{session['day']}**")
        c2.markdown(f"`{session['category']}`")
        c3.markdown(session["drill"])
        completed_val = c4.checkbox("Done", value=session["completed"], key=f"drill_{idx}")
        st.session_state.training_schedule[idx]["completed"] = completed_val

    st.markdown("---")
    completed_count = sum(1 for s in st.session_state.training_schedule if s["completed"])
    total_count = len(st.session_state.training_schedule)
    progress_val = completed_count / total_count if total_count > 0 else 0
    st.write(f"**Weekly Completion Progress:** {completed_count}/{total_count} sessions ({int(progress_val*100)}%)")
    st.progress(progress_val)

    json_schedule = json.dumps(st.session_state.training_schedule, indent=2)
    st.download_button(
        label="📥 Export Schedule as JSON",
        data=json_schedule,
        file_name="badminton_training_plan.json",
        mime="application/json",
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
