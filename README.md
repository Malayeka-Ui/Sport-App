# 🏸 Badminton Pro-Hub

**Badminton Pro-Hub: Tactical Scorer, AI Coach & Training Suite** is a comprehensive, production-grade web application built in Python and Streamlit for badminton players, coaches, and umpires.

---

## 🚀 Key Modules

1. **🏸 Live Tactical Scorer & Win Forecaster**
   - Official BWF 21-point rally scoring engine with deuce, advantage, and 30-point sudden death rules.
   - Dynamic service side indication (even score = right court, odd = left court).
   - Real-time 1,000-iteration Monte Carlo win probability simulator factoring in serving momentum.
   - Shot-by-shot logger with Plotly score progression timelines and shot distribution charts.

2. **🤖 AI Rules Arbiter & Tactical Coach**
   - Built-in rule engine covering BWF Laws (1.15m service height rule, faults, lets, interval rules, singles and doubles positioning).
   - Optional free Google Gemini API integration for open-ended coaching conversations.

3. **🛍️ Local Sports Gear & Equipment Finder**
   - Geo-aware lookup powered by OpenStreetMap Nominatim and Overpass APIs (free, zero API key required).
   - Queries live badminton and sports outlets for any input city and country worldwide.
   - Displays store addresses with direct Google Maps navigation links.

4. **📅 Training Schedule & Session Organizer**
   - Interactive weekly workout planner categorized into Drills, Footwork, Match Play, and Gym.
   - Session checklist with progress tracking and JSON export capability.

5. **❤️ Health, Fitness & Fatigue Tracker**
   - Heart rate training zone calculator using the Karvonen formula (Aerobic vs. Anaerobic threshold).
   - MET-based calorie burn and hydration requirement estimator.
   - Rest period calculator and injury prevention protocols targeting Achilles tendons, jumper's knee, and rotator cuffs.

---

## 🛠️ Free Deployment to Streamlit Community Cloud

1. **Fork or Push to GitHub**:
   - Create a new GitHub repository (e.g., `badminton-pro-hub`).
   - Push `app.py`, `requirements.txt`, and `README.md` to your repository.

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **New app**.
   - Select your repository, branch (`main`), and set the main file path to `app.py`.
   - Click **Deploy**!
