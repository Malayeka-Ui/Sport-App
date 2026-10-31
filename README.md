# 🏸 Badminton Pro-Hub

# 🏸 Badminton Pro-Hub Ultimate: Tactical Scorer, AI Coach & Training Suite

**Badminton Pro-Hub Ultimate** is a production-grade, all-in-one badminton analytics, training, and match management web platform built using Python, Streamlit, Plotly, and Groq's high-speed inference engine. 

It is engineered for competitive players, club organizers, coaches, and international umpires—designed to deploy seamlessly for free on **Streamlit Community Cloud**.

---

## 🚀 Key Modules & Architecture

### 1. 🏸 Live Tactical Scorer & Multi-Game Manager
- **BWF 21-Point Rally Scoring Engine:** Manages deuce/advantage scenarios (2-point lead required) with a hard sudden-death cap at 30 points.
- **Best-of-3 Sets Engine:** Tracks multi-game match transitions, cumulative set standings, and automated court-end changeover alerts at 11 points in the deciding 3rd set.
- **Singles & Doubles Rotation Engine:** Supports distinct singles and doubles court geometry with partner tracking and legal service court switching.
- **↩️ One-Click Undo Rally:** Instantly reverts the last rally point, restoring prior scores, server tracking, and game states without resetting the match.
- **🎲 Monte Carlo Win Forecaster:** Simulates 1,000 match completions in real time using serving momentum to calculate live winning probability percentages.
- **Accurate 2D Court Radar:** Dynamically renders the active server half (Player 1 bottom vs. Player 2 top), even/odd court boxes, and diagonally opposing receiver target boxes.
- **🔊 Voice Umpire (Text-to-Speech):** Speaks scorelines and official umpire calls (*"Love-All, Play"*, *"Game Point"*, *"Match Won By..."*) via the browser Web Speech API.
- **📋 Printable BWF Scoresheet:** Generates a formatted HTML scoresheet ready for club tournament logging and PDF exports.

### 2. 🗺️ Interactive Shot Placement Heatmap
- Records landing sectors (*Net, Drive, Baseline*) for every logged winner and unforced error.
- Plots a Plotly 2D density contour map to identify an opponent's court vulnerabilities, smash clusters, and defensive gaps.

### 3. 🤖 AI Rules Arbiter & Tactical Coach
- **Zero-Exposure Security:** Connects to Groq using the OpenAI Python SDK strictly via backend Streamlit Secrets (`st.secrets["GROQ_API_KEY"]`). No key input fields in the user interface.
- **Fast Reasoning Models:** Powered by Groq-hosted open-weights models (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`).
- **Built-In Offline Engine:** Includes a full fallback knowledge base covering BWF Law 9 (1.15m service height rule), Law 13 (faults), Law 14 (lets), and interval timings if no API key is present.

### 4. 🏃 Interactive Footwork Pacer & Metronome
- Solo reaction training tool that calls out randomized corner targets (*Front-Left Net*, *Rear-Right Smash*, etc.) at adjustable interval cadences (1.0s to 4.0s).
- Uses auditory voice cues and visual prompt cards to sharpen split-step recovery.

### 5. 🌡️ Shuttlecock Speed & Elevation Optimizer
- Environmental physics calculator computing exact shuttlecock ratings (**Speed 76, 77, or 78**) based on ambient hall temperature, altitude, and relative humidity.
- Advises players on feather tipping and bending techniques to regulate flight speed in extreme hall conditions.

### 6. 🏆 Tournament Bracket & Round-Robin Generator
- Instant competition scheduler supporting **Single-Elimination Knockout Brackets** (with automated BYE handling) and **Round-Robin League Schedules**.

### 7. 🛍️ Local Sports Gear & Equipment Finder
- Free, zero-API-key geo-query utilizing **OpenStreetMap Nominatim** and **Overpass API**.
- Searches physical sports shops, badminton specialists, and racket stringing outlets in any city/country worldwide with direct Google Maps search links.

### 8. 📅 Training Schedule Planner
- Interactive weekly workout planner with instant presets (**Tournament Prep**, **Footwork Focus**, and **Default Weekly**).
- Includes one-click schedule resets, individual drill deletion, session completion checklists, and **JSON/CSV export options**.

### 9. ❤️ Health, Fitness & Fatigue Tracker
- Calculates personalized aerobic and anaerobic heart rate zones using the **Karvonen Formula**.
- Computes match caloric burn and fluid intake requirements via MET values.
- Recommends neuromuscular recovery rest intervals and injury prevention protocols for Achilles tendons, jumper's knee, and rotator cuffs.

### 10. ⚙️ Racket & Tension Optimizer
- Computes ideal string tension ranges (lbs), racket balance points (*Head-Heavy, Even-Balance, Head-Light*), and string gauges (*0.65mm vs 0.70mm*) tailored to player experience and shuttlecock type (Feather vs. Nylon).

---

## 📦 Project Structure

```text
badminton-pro-hub/
├── .streamlit/
│   └── secrets.toml          # Secure local secrets (DO NOT commit to GitHub)
├── app.py                    # Complete multi-module Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
