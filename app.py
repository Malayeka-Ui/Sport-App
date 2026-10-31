import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import urllib.parse
from datetime import datetime

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
        padding: 10px;
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
            "content": "Welcome to Badminton Pro-Hub! Ask me anything regarding BWF rules, singles/doubles positioning, tactical setups, or fitness advice.",
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
# HELPER FUNCTIONS: MONTE CARLO & RULES
# ==============================================================================
def check_game_status(s1, s2):
    """Evaluates BWF 21-point rules, deuce past 20-20, and sudden-death cap at 30."""
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
            # Server receives a 3% offensive momentum edge
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

    # Match Setup
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

    # Service & Court Positioning
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

    # Shot-by-Shot Logger
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

    # Monte Carlo Win Forecaster
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

    # Match Analytics & Visualizations
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
# MODULE 2: AI RULES ARBITER & TACTICAL COACH
# ==============================================================================
elif nav_choice == "🤖 AI Rules & Tactical Coach":
    st.header("🤖 AI Rules Arbiter & Tactical Coach")
    st.caption("Ask questions about BWF Laws of Badminton, doubles rotations, shot choices, and fitness conditioning.")

    with st.expander("🔑 Optional: Configure Free Google Gemini API Key for Open-Ended AI"):
        gemini_api_key = st.text_input("Gemini API Key (optional - built-in rules engine activates if empty):", type="password")

    # Built-in Knowledge Base fallback
    RULE_KNOWLEDGE_BASE = {
        "service height": "Under BWF Law 9.1.6, the whole shuttle must be below 1.15 meters (3.77 feet) from the surface of the court at the instant of being hit by the server's racket.",
        "fault": "Common BWF faults include: shuttle landing outside court boundaries, passing through or under net, player or racket touching net during play, striking shuttle before it crosses the net, and hitting the shuttle twice successively.",
        "let": "A 'Let' is called by the umpire to halt play when an unforeseen accident occurs, shuttle catches on net after passing over, server serves before receiver is ready, or shuttle disintegrates mid-rally. The rally is replayed with no point awarded.",
        "interval": "A 60-second interval is permitted in every game when the leading score reaches 11 points. A 120-second (2-minute) interval is permitted between consecutive games.",
        "doubles positioning": "Attacking formation: Front-to-back (rear player executes smashes/drops while front player controls net kills and interceptions). Defensive formation: Side-by-side along mid-court to cover full baseline smash angles.",
        "singles positioning": "Singles court recovery revolves around the 'central T base' located roughly 1-2 steps behind the service line intersection. Recover here after high clears or lifts to cover all 4 corners efficiently.",
    }

    def get_coach_response(user_query, api_key):
        query_lower = user_query.lower()
        if api_key:
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                prompt = (
                    "You are an elite BWF Certified Badminton Coach and International Umpire. "
                    "Provide clear, concise, rules-accurate, and tactically sound badminton advice.\n\n"
                    f"User: {user_query}"
                )
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                return f"*(Gemini API returned an error: {e}. Falling back to internal engine)*\n\n" + match_internal_rules(query_lower)
        return match_internal_rules(query_lower)

    def match_internal_rules(query_lower):
        for key, answer in RULE_KNOWLEDGE_BASE.items():
            if key in query_lower:
                return answer
        if "serve" in query_lower:
            return "Serving Rules (BWF Law 9): Server must stand inside service court without touching boundary lines. Racket head must contact the shuttle base below 1.15m height with a continuous forward stroke."
        elif "net" in query_lower:
            return "Net Rules (BWF Law 13.4): Touching the net with racket, body, or clothing while shuttle is in play is an immediate fault. Striking the shuttle before it crosses the net plane is also a fault."
        return "I can answer specific questions on BWF regulations (service height, lets, faults, intervals) or tactical advice (attacking doubles rotation, singles central base recovery). Please type your specific question or provide an API key above."

    # Render Chat History
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a badminton rule or tactical question...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        response_text = get_coach_response(user_input, gemini_api_key)
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
                # 1. Geocode city coordinates using OpenStreetMap Nominatim
                nominatim_url = f"https://nominatim.openstreetmap.org/search?city={urllib.parse.quote(city_input)}&country={urllib.parse.quote(country_country if 'country_country' in locals() else country_input)}&format=json&limit=1"
                headers = {"User-Agent": "BadmintonProHub/1.0 (Educational sports tool)"}
                geo_resp = requests.get(nominatim_url, headers=headers, timeout=10).json()

                if geo_resp:
                    lat = float(geo_resp[0]["lat"])
                    lon = float(geo_resp[0]["lon"])

                    # 2. Overpass API query for sports shops within a 15km bounding radius
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

                        # Build direct Google Maps search link
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
                        st.warning(f"No specific sports shop nodes mapped on OpenStreetMap for {city_input}. Browse major sports districts via Google Maps below.")
                        direct_url = f"https://www.google.com/maps/search/badminton+sports+gear+store+in+{urllib.parse.quote(city_input)}"
                        st.markdown(f"👉 **[Search Badminton Stores in {city_input} on Google Maps]({direct_url})**")
                else:
                    st.error("Could not resolve location coordinates. Please verify city and country spelling.")
            except Exception as ex:
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

    # Export Schedule
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

    # Heart Rate Training Zones (Karvonen Formula)
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

    # MET values: Casual = 5.5, Moderate = 7.0, High = 8.5, Extreme = 10.0
    met_map = {
        "Light (Casual Rally)": 5.5,
        "Moderate (Drills / Doubles)": 7.0,
        "High (Competitive Singles Match)": 8.5,
        "Extreme (Tournament Match Play)": 10.0,
    }
    met = met_map[intensity_level]

    # Calories = MET * Weight (kg) * Duration (hours)
    duration_hours = duration_mins / 60.0
    calories_burned = int(met * user_weight * duration_hours)
    # Fluid requirement: ~700-1000ml per hour of high-intensity badminton
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
