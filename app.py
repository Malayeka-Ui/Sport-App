import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except ImportError:
    GROQ_SDK_AVAILABLE = False


# ==============================================================================
# 1. PAGE CONFIGURATION & FUTURISTIC DARK SPORTS-LAB CSS
# ==============================================================================
st.set_page_config(
    page_title="AERO-KINETIX AI | Sports Science & Scouting Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css(accent_hex: str = "#10B981") -> None:
    """Injects futuristic dark glassmorphic styling with dynamic athlete accent colors."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

        :root {{
            --bg-obsidian: #0B0F19;
            --bg-card: rgba(17, 24, 39, 0.78);
            --bg-card-hover: rgba(31, 41, 55, 0.88);
            --accent-primary: #10B981;
            --accent-cyan: #06B6D4;
            --accent-athlete: {accent_hex};
            --text-main: #F9FAFB;
            --text-muted: #9CA3AF;
            --border-glass: rgba(255, 255, 255, 0.09);
        }}

        .stApp {{
            background: radial-gradient(circle at 12% 10%, rgba(16, 185, 129, 0.08), transparent 32%),
                        radial-gradient(circle at 88% 15%, rgba(6, 182, 212, 0.08), transparent 35%),
                        linear-gradient(180deg, #070A12 0%, #0B0F19 50%, #090D16 100%);
            color: var(--text-main);
            font-family: 'Inter', sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0D1322 0%, #090D18 100%) !important;
            border-right: 1px solid rgba(6, 182, 212, 0.18);
        }}

        /* Top Hero Banner */
        .hero-banner {{
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.92) 0%, rgba(11, 15, 25, 0.95) 100%);
            border: 1px solid rgba(16, 185, 129, 0.28);
            border-left: 5px solid var(--accent-athlete);
            border-radius: 16px;
            padding: 1.35rem 1.75rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45), 0 0 24px rgba(16, 185, 129, 0.06);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        .hero-title {{
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #FFFFFF 0%, #10B981 55%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }}

        .hero-subtitle {{
            color: #9CA3AF;
            font-size: 0.88rem;
            margin-top: 0.25rem;
        }}

        /* Glassmorphic Telemetry Cards */
        .glass-card {{
            background: var(--bg-card);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid var(--border-glass);
            border-radius: 14px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .glass-card:hover {{
            border-color: rgba(6, 182, 212, 0.35);
        }}

        .metric-label {{
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: #9CA3AF;
            font-weight: 600;
            margin-bottom: 0.3rem;
        }}

        .metric-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.85rem;
            font-weight: 700;
            color: #F9FAFB;
            line-height: 1.15;
        }}

        .metric-sub {{
            font-size: 0.78rem;
            margin-top: 0.35rem;
            font-weight: 500;
        }}

        /* Status Badges */
        .status-badge {{
            display: inline-block;
            padding: 0.22rem 0.68rem;
            border-radius: 999px;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            font-family: 'JetBrains Mono', monospace;
        }}

        .badge-optimal {{
            background: rgba(16, 185, 129, 0.16);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .badge-caution {{
            background: rgba(245, 158, 11, 0.16);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.18);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.45);
        }}

        .badge-info {{
            background: rgba(6, 182, 212, 0.16);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.4);
        }}

        /* AI Coach Terminal Box */
        .coach-terminal {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.10) 0%, rgba(6, 182, 212, 0.08) 100%);
            border: 1px solid rgba(16, 185, 129, 0.45);
            border-left: 4px solid #10B981;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
            margin: 0.85rem 0 1.25rem 0;
            box-shadow: 0 8px 28px rgba(16, 185, 129, 0.08);
        }}

        .coach-header {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #10B981;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }}

        .coach-body {{
            font-size: 0.96rem;
            line-height: 1.62;
            color: #E5E7EB;
        }}

        /* Drill Prescription Row */
        .drill-card {{
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-left: 3px solid var(--accent-cyan);
            border-radius: 10px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.65rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}

        /* Tabs Styling */
        button[data-baseweb="tab"] {{
            background-color: rgba(17, 24, 39, 0.65) !important;
            border-radius: 10px 10px 0 0 !important;
            margin-right: 6px !important;
            padding: 10px 20px !important;
            color: #9CA3AF !important;
            font-weight: 600 !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.22), rgba(6, 182, 212, 0.18)) !important;
            color: #FFFFFF !important;
            border-bottom: 2px solid #10B981 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. EMBEDDED MULTI-SPORT DATABASE & 28-DAY HISTORICAL TRAINING GENERATOR
# ==============================================================================
def get_pro_athletes_db() -> pd.DataFrame:
    """Returns a comprehensive multi-sport database of elite world athletes."""
    athletes = [
        # BADMINTON
        {
            "Sport": "Badminton",
            "Team_League": "BWF World Tour (Denmark)",
            "Athlete": "Viktor Axelsen",
            "Position": "Men's Singles Attacking Baseline",
            "Pace": 91,
            "Endurance": 94,
            "Agility": 96,
            "Power": 95,
            "Tactical_IQ": 98,
            "Recovery": 93,
            "OVR": 95,
            "Jersey_Color": "#EF4444",
            "Signature_Style": "Steep-angle overhead jump smashes & impenetrable court coverage",
        },
        {
            "Sport": "Badminton",
            "Team_League": "BWF Legends (China)",
            "Athlete": "Lin Dan",
            "Position": "Men's Singles Counter-Attacking",
            "Pace": 95,
            "Endurance": 97,
            "Agility": 99,
            "Power": 93,
            "Tactical_IQ": 99,
            "Recovery": 95,
            "OVR": 97,
            "Jersey_Color": "#F59E0B",
            "Signature_Style": "Deceptive wrist flicks, explosive net lunges & sixth-sense anticipation",
        },
        {
            "Sport": "Badminton",
            "Team_League": "BWF World Tour (South Korea)",
            "Athlete": "An Se-young",
            "Position": "Women's Singles Defensive Retriever",
            "Pace": 93,
            "Endurance": 99,
            "Agility": 97,
            "Power": 88,
            "Tactical_IQ": 96,
            "Recovery": 98,
            "OVR": 95,
            "Jersey_Color": "#3B82F6",
            "Signature_Style": "Relentless rally attrition, elastic split-steps & baseline consistency",
        },
        # FOOTBALL
        {
            "Sport": "Football",
            "Team_League": "La Liga (Real Madrid)",
            "Athlete": "Kylian Mbappé",
            "Position": "Inside Forward / Striker",
            "Pace": 99,
            "Endurance": 89,
            "Agility": 95,
            "Power": 88,
            "Tactical_IQ": 92,
            "Recovery": 91,
            "OVR": 94,
            "Jersey_Color": "#06B6D4",
            "Signature_Style": "Supersonic transition sprints & clinical half-space finishing",
        },
        {
            "Sport": "Football",
            "Team_League": "Premier League (Manchester City)",
            "Athlete": "Erling Haaland",
            "Position": "Target No. 9 Striker",
            "Pace": 94,
            "Endurance": 88,
            "Agility": 85,
            "Power": 99,
            "Tactical_IQ": 94,
            "Recovery": 93,
            "OVR": 94,
            "Jersey_Color": "#38BDF8",
            "Signature_Style": "Penalty-box predator, high-impulse vertical leaps & physical shielding",
        },
        {
            "Sport": "Football",
            "Team_League": "La Liga (Real Madrid)",
            "Athlete": "Jude Bellingham",
            "Position": "Box-to-Box Attacking Midfielder",
            "Pace": 88,
            "Endurance": 97,
            "Agility": 91,
            "Power": 90,
            "Tactical_IQ": 95,
            "Recovery": 94,
            "OVR": 93,
            "Jersey_Color": "#A855F7",
            "Signature_Style": "Late box arrivals, high-volume pressing & spatial tempo control",
        },
        # BASKETBALL
        {
            "Sport": "Basketball",
            "Team_League": "NBA (Los Angeles Lakers)",
            "Athlete": "LeBron James",
            "Position": "Point Forward (SF/PF)",
            "Pace": 91,
            "Endurance": 95,
            "Agility": 90,
            "Power": 98,
            "Tactical_IQ": 99,
            "Recovery": 99,
            "OVR": 96,
            "Jersey_Color": "#FACC15",
            "Signature_Style": "Downhill freight-train drives, elite court vision & cryo-bio longevity",
        },
        {
            "Sport": "Basketball",
            "Team_League": "NBA (Golden State Warriors)",
            "Athlete": "Stephen Curry",
            "Position": "Combo Guard (PG)",
            "Pace": 92,
            "Endurance": 99,
            "Agility": 97,
            "Power": 82,
            "Tactical_IQ": 98,
            "Recovery": 94,
            "OVR": 95,
            "Jersey_Color": "#2563EB",
            "Signature_Style": "Off-ball marathon relocations, sub-0.4s release & deep-range gravity",
        },
        {
            "Sport": "Basketball",
            "Team_League": "NBA (Denver Nuggets)",
            "Athlete": "Nikola Jokić",
            "Position": "Point Center (C)",
            "Pace": 78,
            "Endurance": 95,
            "Agility": 86,
            "Power": 96,
            "Tactical_IQ": 99,
            "Recovery": 95,
            "OVR": 96,
            "Jersey_Color": "#F97316",
            "Signature_Style": "High-post orchestration, Sombor shuffle touch & elite rebound positioning",
        },
        # CRICKET
        {
            "Sport": "Cricket",
            "Team_League": "ICC / PSL (Pakistan National Team)",
            "Athlete": "Babar Azam",
            "Position": "Top-Order Batter (No. 3)",
            "Pace": 87,
            "Endurance": 94,
            "Agility": 93,
            "Power": 88,
            "Tactical_IQ": 96,
            "Recovery": 92,
            "OVR": 93,
            "Jersey_Color": "#10B981",
            "Signature_Style": "High-elbow biomechanics, late-contact cover drives & crease balance",
        },
        {
            "Sport": "Cricket",
            "Team_League": "ICC / IPL (India National Team)",
            "Athlete": "Virat Kohli",
            "Position": "Top-Order Chase Master (No. 3)",
            "Pace": 94,
            "Endurance": 98,
            "Agility": 95,
            "Power": 91,
            "Tactical_IQ": 98,
            "Recovery": 96,
            "OVR": 96,
            "Jersey_Color": "#0EA5E9",
            "Signature_Style": "Explosive 2-run conversion sprints, wrists-driven flicks & mental grit",
        },
        {
            "Sport": "Cricket",
            "Team_League": "ICC / PSL (Pakistan National Team)",
            "Athlete": "Shaheen Shah Afridi",
            "Position": "Left-Arm Fast Bowler",
            "Pace": 96,
            "Endurance": 91,
            "Agility": 89,
            "Power": 95,
            "Tactical_IQ": 92,
            "Recovery": 90,
            "OVR": 93,
            "Jersey_Color": "#14B8A6",
            "Signature_Style": "145+ kph late-inswinging yorkers & high-release front-leg brace",
        },
    ]
    return pd.DataFrame(athletes)


def generate_28_day_seed_log() -> pd.DataFrame:
    """Generates a realistic 28-day periodized multi-sport training dataset."""
    rng = np.random.default_rng(42)
    today = datetime.now().date()
    sports_cycle = ["Badminton", "Football", "Basketball", "Cricket"]

    records = []
    for i in range(28, 0, -1):
        log_date = today - timedelta(days=i)
        weekday = log_date.weekday()
        sport = sports_cycle[i % len(sports_cycle)]

        if weekday == 6:
            duration = int(rng.integers(35, 50))
            rpe = int(rng.integers(3, 5))
            planned_sets = 8
        elif weekday in (1, 3, 4):
            surge = 1.15 if i <= 6 else 1.0
            duration = int(rng.integers(75, 105) * surge)
            rpe = int(min(10, rng.integers(7, 9) * surge))
            planned_sets = int(rng.integers(16, 22))
        else:
            duration = int(rng.integers(60, 85))
            rpe = int(rng.integers(5, 8))
            planned_sets = int(rng.integers(12, 16))

        sleep_hrs = round(float(rng.uniform(6.2, 8.8)), 1)
        soreness = int(np.clip(round(rpe * 0.65 + rng.uniform(-1.5, 1.8)), 1, 9))
        energy = int(np.clip(round(sleep_hrs * 1.1 - soreness * 0.35 + rng.uniform(0.5, 2.0)), 2, 10))

        s_norm = min(100.0, (sleep_hrs / 8.5) * 100.0)
        m_norm = ((10.0 - soreness) / 9.0) * 100.0
        e_norm = ((energy - 1.0) / 9.0) * 100.0
        readiness = round(float(np.clip(0.40 * s_norm + 0.35 * m_norm + 0.25 * e_norm, 25.0, 98.0)), 1)

        compliance_factor = rng.uniform(0.85, 1.05) if readiness >= 65 else rng.uniform(0.68, 0.88)
        completed_sets = int(max(4, round(planned_sets * compliance_factor)))
        workload = int(duration * rpe)

        records.append(
            {
                "Date": pd.to_datetime(log_date),
                "Sport": sport,
                "Duration_Min": duration,
                "RPE": rpe,
                "Daily_Workload": workload,
                "Sleep_Hours": sleep_hrs,
                "Soreness": soreness,
                "Energy": energy,
                "Readiness_Score": readiness,
                "Planned_Sets": planned_sets,
                "Completed_Sets": completed_sets,
            }
        )

    df = pd.DataFrame(records).sort_values("Date").reset_index(drop=True)
    return recompute_rolling_metrics(df)


def recompute_rolling_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes 7-day Acute Load, 28-day Chronic Load, and ACWR across the training log."""
    df = df.sort_values("Date").reset_index(drop=True).copy()
    df["Acute_Load_7D"] = df["Daily_Workload"].rolling(window=7, min_periods=1).mean().round(1)
    df["Chronic_Load_28D"] = df["Daily_Workload"].rolling(window=28, min_periods=1).mean().round(1)
    df["ACWR"] = (df["Acute_Load_7D"] / df["Chronic_Load_28D"].replace(0, 1)).round(2)
    return df


# ==============================================================================
# 3. DOMAIN SCIENCE: READINESS, ACWR & ADAPTIVE SCHEDULE ENGINE
# ==============================================================================
SPORT_WEEKLY_ROUTINES: Dict[str, Dict[str, List[Dict[str, object]]]] = {
    "Badminton": {
        "Monday: Shadow Footwork & Split-Step Mechanics": [
            {"Drill": "6-Corner Explosive Shadow Footwork", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Court Coverage & Split-Step Timing"},
            {"Drill": "Rear-Court Scissor Jump to Net Lunge", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Posterior-Anterior Transition"},
            {"Drill": "Low Isometric Lunge Hold + Racket Tracking", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Patellar Tendon & Quad Isometric"},
        ],
        "Tuesday: Linear & Multi-Directional Agility": [
            {"Drill": "Forehand/Backhand Side-to-Side Defense Shuttles", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Lateral Hip Abduction Drive"},
            {"Drill": "Reactive Light/Audio Split-Step Sprint Matrix", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Visual-Motor Reaction Latency"},
            {"Drill": "Plyometric Hurdle Hops + Deep Lunge Recovery", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Stretch-Shortening Cycle (SSC)"},
        ],
        "Wednesday: High-Velocity Court Drills": [
            {"Drill": "Multi-Shuttle Rapid Net Kill & Lift Sequence", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Forearm Pronation & Net Dominance"},
            {"Drill": "2-vs-1 Flat Drive & Half-Smash Counter Drill", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "High-Tempo Tactical Defense"},
            {"Drill": "Deceptive Cross-Court Drop & Clear Precision", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Fine Motor Control Under Fatigue"},
        ],
        "Thursday: Anaerobic Alactic Conditioning": [
            {"Drill": "40s On / 20s Off Multi-Shuttle Survival Feed", "Base_Sets": 8, "Base_Rest_Sec": 40, "Focus": "Lactate Clearance & VO2 Peak"},
            {"Drill": "Double-Jump Smash Repeatability Protocol", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Alactic Power Maintenance"},
            {"Drill": "Rotator Cuff Eccentric Deceleration Band Work", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Glenohumeral Joint Integrity"},
        ],
        "Friday: Tactical Game Simulation": [
            {"Drill": "21-Point Pressure Situation Match Play (14-14 Start)", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Clutch Decision Making"},
            {"Drill": "Service-Return 3rd-Shot Attack Patterns", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "First-3-Shots Rally Control"},
            {"Drill": "Post-Match Parasympathetic Breathwork & Foam Roll", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Vagal Tone Restoration"},
        ],
    },
    "Football": {
        "Monday: Shadow Footwork & Positional Scanning": [
            {"Drill": "180° Half-Turn Scanning & First-Touch Orientation", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Proprioceptive Spatial Awareness"},
            {"Drill": "Close-Control Slalom Weave & Deceleration", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Eccentric Braking Mechanics"},
            {"Drill": "Copenhagen Adductor & Nordic Hamstring Prep", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Posterior Chain Injury Shield"},
        ],
        "Tuesday: Linear Footwork & Max-Velocity Sprints": [
            {"Drill": "30m Fly Sprints with Curved Run-In", "Base_Sets": 6, "Base_Rest_Sec": 90, "Focus": "Top-End Horizontal Force"},
            {"Drill": "5-10-5 Pro Agility Box Transition", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Change of Direction Deficit (CODD)"},
            {"Drill": "1v1 Channel Recovery Defending Sprints", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Hip-Turn & Acceleration"},
        ],
        "Wednesday: Pitch Drills & Rondo Pressing": [
            {"Drill": "6v3 High-Intensity Transition Rondo", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "One-Touch Cognitive Speed"},
            {"Drill": "Overlap/Underlap Final-Third Cutback Finishing", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "High-Heart-Rate Composure"},
            {"Drill": "Long-Diagonal Switch & Cushioned Volley Control", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Hip Mobility & Ball Striking"},
        ],
        "Thursday: Metabolic Conditioning (RSA)": [
            {"Drill": "Repeated Sprint Ability (6 x 40m Shuttle)", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Phosphocreatine Resynthesis"},
            {"Drill": "Box-to-Box High-Speed Running Intervals (15s/15s)", "Base_Sets": 8, "Base_Rest_Sec": 30, "Focus": "Aerobic Power Ceiling"},
            {"Drill": "Core Anti-Rotation Pallof Press & Plank Matrix", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Pelvic Stability in Duels"},
        ],
        "Friday: Game Simulation & Tactical Phases": [
            {"Drill": "11v11 Zonal High-Press vs Low-Block Phase Play", "Base_Sets": 4, "Base_Rest_Sec": 120, "Focus": "Collective Tactical Compactness"},
            {"Drill": "Set-Piece Delivery & Near-Post Attacking Runs", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Dead-Ball Execution"},
            {"Drill": "Lower-Limb Contrast Mobility & Hydro-Recovery", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Inflammatory Marker Clearance"},
        ],
    },
    "Basketball": {
        "Monday: Shadow Footwork & Closeout Mechanics": [
            {"Drill": "Defensive Slide-Drop-Contest Closeout Triangle", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Lateral Hip Stiffness & Balance"},
            {"Drill": "Triple-Threat Jab-Step & Split-Footwork Series", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "First-Step Separation"},
            {"Drill": "Ankle Eversion/Inversion Isometric Balance", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Lateral Ligament Protection"},
        ],
        "Tuesday: Linear Footwork & Transition Lanes": [
            {"Drill": "Baseline-to-Baseline 3-Man Weave Euro-Step Finish", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Deceleration & Stride Length"},
            {"Drill": "Lane Agility Cone Sprint & Backpedal Matrix", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Multi-Planar Transition Speed"},
            {"Drill": "Approach Vertical Jump & 2-Foot Landing Mechanics", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Impulse Absorption"},
        ],
        "Wednesday: Court Drills & Pick-and-Roll Reads": [
            {"Drill": "Spain Pick-and-Roll Pocket Pass / Pull-Up 3PT", "Base_Sets": 6, "Base_Rest_Sec": 55, "Focus": "2-Man Game Manipulation"},
            {"Drill": "Off-Ball Pin-Down Relocations & Catch-and-Shoot", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Shot Preparation Footwork"},
            {"Drill": "Contested Finishing Through Contact Pad", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Mid-Air Core Bracing"},
        ],
        "Thursday: Hardwood Conditioning & Suicides": [
            {"Drill": "17s Sideline-to-Sideline Conditioning Shuttles", "Base_Sets": 7, "Base_Rest_Sec": 45, "Focus": "Glycolytic Buffering Capacity"},
            {"Drill": "Fatigued Free-Throw Pressure Protocol (Swish = -1 Set)", "Base_Sets": 5, "Base_Rest_Sec": 35, "Focus": "Autonomic Heart Rate Control"},
            {"Drill": "Eccentric Single-Leg Squat on Decline Board", "Base_Sets": 4, "Base_Rest_Sec": 50, "Focus": "Jumper's Knee Prevention"},
        ],
        "Friday: 5v5 Half/Full-Court Game Simulation": [
            {"Drill": "Late-Clock (8s Shot Clock) Half-Court Execution", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Crunch-Time Spacing"},
            {"Drill": "Full-Court Press Break & Secondary Transition", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Ball Security Under Trap"},
            {"Drill": "Thoracic Spine & Hip Flexor Decompression Flow", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Postural Reset"},
        ],
    },
    "Cricket": {
        "Monday: Shadow Footwork & Crease Alignment": [
            {"Drill": "Front-Foot Stride & Back-Foot Trigger Movement Shadow", "Base_Sets": 6, "Base_Rest_Sec": 40, "Focus": "Head Stillness & Center of Mass"},
            {"Drill": "Slip Cordon Reactive Low-Catch Footwork", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "Pre-Delivery Split-Step"},
            {"Drill": "Thoracic Rotation & Scapular Retraction Mobility", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Bowling/Throwing Shoulder Health"},
        ],
        "Tuesday: Linear Footwork & 22-Yard Sprint Turning": [
            {"Drill": "Explosive 2s & 3s Crease-Turning Shuttle With Bat", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Low Center-of-Gravity Braking"},
            {"Drill": "Bowler Run-Up Rhythm & Bound Alignment", "Base_Sets": 6, "Base_Rest_Sec": 70, "Focus": "Front-Foot Block Impulse"},
            {"Drill": "In-Field Swoop, Pick-Up & Off-Balance Direct Hit", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Rotational Core Transfer"},
        ],
        "Wednesday: Pitch / Net Drills & Ball-Striking": [
            {"Drill": "140+ kph Sidearm Short-Pitched Pull & Duck Session", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Visual Tracking & Weight Transfer"},
            {"Drill": "Spin Sweep / Advance Down-the-Track Footwork Matrix", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Pitch Reading & Wrist Work"},
            {"Drill": "Death-Overs Yorker & Slower-Ball Spot Bowling", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Release Point Consistency"},
        ],
        "Thursday: Spell Endurance & Heat Conditioning": [
            {"Drill": "6-Over Simulated Fast Bowling Spell / 100-Ball Batting", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Neuromuscular Fatigue Resistance"},
            {"Drill": "Boundary Rope Sliding Stop & Relay Throw Sprints", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Repeat High-Intensity Efforts"},
            {"Drill": "Lumbar Multifidus & Oblique Isometric Stabilization", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "L4-L5 Pars Stress Protection"},
        ],
        "Friday: Match Scenario Simulation": [
            {"Drill": "Powerplay / Death-Overs Required-Run-Rate Chase (12 RPO)", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Gap Manipulation Under Pressure"},
            {"Drill": "Fielding Captaincy Ring-Setting vs 360° Hitter", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Tactical Matchup Execution"},
            {"Drill": "Posterior Shoulder & Hamstring PNF Stretching", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Myofascial Length Restoration"},
        ],
    },
}


def compute_readiness_and_acwr(
    sleep_hours: float,
    soreness: int,
    energy: int,
    df_log: pd.DataFrame,
    planned_workload_today: int,
) -> Dict[str, object]:
    """Calculates Daily Readiness Index (0-100%), live ACWR, and adaptive scaling multipliers."""
    sleep_score = min(100.0, (sleep_hours / 8.5) * 100.0)
    soreness_score = ((10.0 - float(soreness)) / 9.0) * 100.0
    energy_score = ((float(energy) - 1.0) / 9.0) * 100.0

    raw_readiness = 0.40 * sleep_score + 0.35 * soreness_score + 0.25 * energy_score

    recent_6d = df_log["Daily_Workload"].tail(6).tolist()
    recent_27d = df_log["Daily_Workload"].tail(27).tolist()

    acute_7d = float(np.mean(recent_6d + [planned_workload_today]))
    chronic_28d = float(np.mean(recent_27d + [planned_workload_today]))
    acwr = round(acute_7d / max(1.0, chronic_28d), 2)

    acwr_penalty = 0.0
    if acwr > 1.50:
        acwr_penalty = min(22.0, (acwr - 1.50) * 35.0 + 10.0)
    elif acwr > 1.30:
        acwr_penalty = (acwr - 1.30) * 25.0

    readiness_index = round(float(np.clip(raw_readiness - acwr_penalty, 5.0, 100.0)), 1)

    if acwr < 0.80:
        acwr_zone = "Under-Training (<0.8)"
        acwr_badge = "badge-info"
    elif acwr <= 1.30:
        acwr_zone = "Sweet Spot (0.8–1.3)"
        acwr_badge = "badge-optimal"
    elif acwr <= 1.50:
        acwr_zone = "Caution Overreach (1.3–1.5)"
        acwr_badge = "badge-caution"
    else:
        acwr_zone = "Danger Spike (>1.5)"
        acwr_badge = "badge-danger"

    if readiness_index >= 80 and acwr <= 1.30:
        set_multiplier = 1.15
        rest_multiplier = 0.90
        intensity_tag = "PEAK OVERLOAD (90–95% Max Velocity)"
        readiness_badge = "badge-optimal"
        readiness_status = "CNS Prime / Green Light"
    elif readiness_index >= 62 and acwr <= 1.45:
        set_multiplier = 1.00
        rest_multiplier = 1.00
        intensity_tag = "MAINTENANCE TEMPO (80–85% Sub-Max)"
        readiness_badge = "badge-info"
        readiness_status = "Controlled Adaptation"
    elif readiness_index >= 45 and acwr <= 1.55:
        set_multiplier = 0.72
        rest_multiplier = 1.30
        intensity_tag = "AUTOREGULATED DELOAD (65–75% Technical)"
        readiness_badge = "badge-caution"
        readiness_status = "Autonomic Fatigue Detected"
    else:
        set_multiplier = 0.50
        rest_multiplier = 1.60
        intensity_tag = "NEUROMUSCULAR TAPER (50% Mobility & Flush)"
        readiness_badge = "badge-danger"
        readiness_status = "Red Alert / High Injury Risk"

    return {
        "readiness_index": readiness_index,
        "sleep_score": round(sleep_score, 1),
        "soreness_score": round(soreness_score, 1),
        "energy_score": round(energy_score, 1),
        "acute_7d": round(acute_7d, 1),
        "chronic_28d": round(chronic_28d, 1),
        "acwr": acwr,
        "acwr_zone": acwr_zone,
        "acwr_badge": acwr_badge,
        "set_multiplier": set_multiplier,
        "rest_multiplier": rest_multiplier,
        "intensity_tag": intensity_tag,
        "readiness_badge": readiness_badge,
        "readiness_status": readiness_status,
    }


def build_adaptive_schedule(
    sport: str,
    day_key: str,
    metrics: Dict[str, object],
    muscle_focus: str,
) -> pd.DataFrame:
    """Rescales sets, rest intervals, and intensity based on daily physiological telemetry."""
    base_drills = SPORT_WEEKLY_ROUTINES[sport][day_key]
    scaled_rows = []

    for item in base_drills:
        base_sets = int(item["Base_Sets"])
        base_rest = int(item["Base_Rest_Sec"])
        adj_sets = max(2, int(round(base_sets * float(metrics["set_multiplier"]))))
        adj_rest = int(round((base_rest * float(metrics["rest_multiplier"])) / 5.0) * 5)
        delta_sets = adj_sets - base_sets
        delta_str = f"+{delta_sets}" if delta_sets > 0 else str(delta_sets)

        scaled_rows.append(
            {
                "Drill_Name": item["Drill"],
                "Baseline_Sets": base_sets,
                "Adaptive_Sets": adj_sets,
                "Set_Delta": delta_str,
                "Adaptive_Rest_Sec": f"{adj_rest}s",
                "Target_Intensity": metrics["intensity_tag"],
                "Biomechanical_Focus": f"{item['Focus']} (Protect: {muscle_focus})",
            }
        )
    return pd.DataFrame(scaled_rows)


# ==============================================================================
# 4. DUAL-KEY LLM AI COACH ORCHESTRATION (GROQ + OPENROUTER + HEURISTIC)
# ==============================================================================
def get_secret_api_key(key_name: str) -> str:
    """Safely retrieves API keys from environment variables or st.secrets without raising errors."""
    env_val = os.environ.get(key_name, "").strip()
    if env_val:
        return env_val
    try:
        secret_val = st.secrets.get(key_name, "")
        return str(secret_val).strip() if secret_val else ""
    except Exception:
        return ""


def generate_heuristic_coaching_prescription(
    sport: str,
    athlete_name: str,
    day_key: str,
    sleep_hours: float,
    soreness: int,
    muscle_focus: str,
    energy: int,
    metrics: Dict[str, object],
    total_base_sets: int,
    total_adj_sets: int,
) -> str:
    """Deterministic 3-sentence sports science fallback when offline or without an API key."""
    readiness = metrics["readiness_index"]
    acwr = metrics["acwr"]
    pct_change = int(round(((total_adj_sets - total_base_sets) / max(1, total_base_sets)) * 100))
    direction = (
        f"scaled up by +{pct_change}%"
        if pct_change > 0
        else (f"reduced by {abs(pct_change)}%" if pct_change < 0 else "maintained at baseline volume")
    )

    sentence_1 = (
        f"With a Daily Readiness Index of {readiness}% ({sleep_hours}h sleep, {soreness}/10 soreness in the {muscle_focus}, {energy}/10 CNS drive) "
        f"and an ACWR of {acwr} ({metrics['acwr_zone']}), today's {sport} workload ({day_key.split(':')[0]}) has been {direction} "
        f"from {total_base_sets} to {total_adj_sets} total working sets."
    )

    if readiness < 60 or acwr > 1.35:
        sentence_2 = (
            f"Extending inter-set rest intervals by {int(round((float(metrics['rest_multiplier']) - 1.0) * 100))}% prevents excessive eccentric strain on the {muscle_focus} "
            f"and buffers central nervous system fatigue while preserving clean sub-maximal movement kinematics inspired by {athlete_name}."
        )
    else:
        sentence_2 = (
            f"Your optimal acute-to-chronic load balance permits high-velocity neural recruitment, allowing you to mirror {athlete_name}'s "
            f"explosive kinetic chain efficiency without exceeding tissue tolerance thresholds in the {muscle_focus}."
        )

    recovery_map = {
        "Quads / Hamstrings": "10 minutes of 90/90 hamstring flossing, low-cadence spin flush at <110 bpm, and eccentric Nordic sliders",
        "Calves / Achilles": "bent-knee soleus isometric holds (4x45s), plantar fascia Lacrosse ball release, and ankle dorsiflexion mobilizations",
        "Lower Back / Core": "supine diaphragmatic 90/90 decompression breathing, McGill Big-3 core activation, and thoracic cat-cow extensions",
        "Shoulders / Rotator Cuff": "sleeper-stretch posterior capsule mobilization, light theraband external rotations, and scapular wall slides",
        "Knees / Patella": "Spanish squat isometric holds (4x30s at 70° flexion), VMO foam rolling, and terminal knee extensions",
        "Full Body Systemic": "12 minutes of parasympathetic box breathing (4-4-6-2 cadence), contrast hydrotherapy, and full-chain myofascial rolling",
    }
    drill_rec = recovery_map.get(muscle_focus, recovery_map["Full Body Systemic"])
    sentence_3 = f"Conclude today's session with {drill_rec} to accelerate parasympathetic vagal recovery prior to tomorrow's microcycle block."

    return f"{sentence_1} {sentence_2} {sentence_3}"


def _call_groq(api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Groq API via official SDK or REST endpoint if SDK is missing."""
    if GROQ_SDK_AVAILABLE:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.35,
            max_tokens=260,
        )
        return completion.choices[0].message.content.strip()

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
        "max_tokens": 260,
    }
    resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=12)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_openrouter(api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Calls OpenRouter chat completions REST endpoint."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.35,
        "max_tokens": 260,
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=12)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def query_ai_coach(
    routing_mode: str,
    groq_key: str,
    groq_model: str,
    openrouter_key: str,
    openrouter_model: str,
    sport: str,
    athlete_name: str,
    day_key: str,
    sleep_hours: float,
    soreness: int,
    muscle_focus: str,
    energy: int,
    metrics: Dict[str, object],
    total_base_sets: int,
    total_adj_sets: int,
) -> Tuple[str, str]:
    """
    Supports both Groq and OpenRouter keys simultaneously, single-key operation,
    or offline fallback depending on user preference.
    """
    fallback_text = generate_heuristic_coaching_prescription(
        sport, athlete_name, day_key, sleep_hours, soreness, muscle_focus, energy, metrics, total_base_sets, total_adj_sets
    )

    groq_clean = groq_key.strip()
    or_clean = openrouter_key.strip()

    if routing_mode == "Force Offline Heuristic Engine" or (not groq_clean and not or_clean):
        return fallback_text, "Sports Science Heuristic Engine (Offline / No Key)"

    system_prompt = (
        "You are an elite PhD Sports Scientist and Head of Athletic Performance. "
        "Write EXACTLY 3 concise, high-impact sentences of tactical and physiological instruction for the athlete. "
        "Sentence 1: Justify why today's total sets and rest intervals were adjusted based on their Readiness Index and ACWR. "
        "Sentence 2: Provide a biomechanical execution cue for today's sport routine referencing the pro athlete benchmark. "
        "Sentence 3: Prescribe a specific active recovery protocol targeting their sore muscle group."
    )

    user_prompt = (
        f"Sport: {sport} | Pro Benchmark: {athlete_name} | Session Focus: {day_key}\n"
        f"Physiology: Sleep={sleep_hours}h, Soreness={soreness}/10 ({muscle_focus}), Energy={energy}/10\n"
        f"Telemetry: Readiness={metrics['readiness_index']}%, ACWR={metrics['acwr']} ({metrics['acwr_zone']}), "
        f"Sets Adjusted={total_base_sets} -> {total_adj_sets}, Intensity={metrics['intensity_tag']}."
    )

    # Build ordered provider queue based on user's routing choice and available keys
    attempt_queue: List[Tuple[str, str, str]] = []
    if routing_mode == "Auto-Detect (Groq ➔ OpenRouter ➔ Heuristic)":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Use Groq API (Failover to OpenRouter if set)":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Use OpenRouter API (Failover to Groq if set)":
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))

    last_error = ""
    for provider_label, key_val, model_val in attempt_queue:
        try:
            if provider_label == "Groq":
                text_out = _call_groq(key_val, model_val, system_prompt, user_prompt)
                return text_out, f"Groq Live LLM ({model_val})"
            else:
                text_out = _call_openrouter(key_val, model_val, system_prompt, user_prompt)
                return text_out, f"OpenRouter Live LLM ({model_val})"
        except Exception as exc:
            last_error = f"{provider_label}: {type(exc).__name__}"
            continue

    return fallback_text, f"Heuristic Fallback ({last_error or 'No Valid Key for Selected Mode'})"


# ==============================================================================
# 5. INTERACTIVE 3D WEBGL BIOMECHANICAL AVATAR (PLOTLY SCATTER3D + MESH3D)
# ==============================================================================
def build_3d_biomechanical_avatar(
    athlete_row: pd.Series,
    muscle_focus: str,
    soreness_level: int,
    readiness_index: float,
) -> go.Figure:
    """
    Constructs a 360-degree mouse-rotatable 3D WebGL athletic wireframe & volumetric core avatar.
    Syncs with the athlete's jersey color and highlights fatigued anatomical regions.
    """
    theme_hex = str(athlete_row["Jersey_Color"])
    strain_hex = "#EF4444" if soreness_level >= 7 else ("#F59E0B" if soreness_level >= 4 else "#22D3EE")

    joints: Dict[str, Tuple[float, float, float]] = {
        "Head_Top": (0.0, 0.0, 184.0),
        "Head_Center": (0.0, 2.0, 174.0),
        "Neck_C7": (0.0, 0.0, 158.0),
        "Sternum": (0.0, 6.0, 142.0),
        "Thoracic_T8": (0.0, -3.0, 132.0),
        "Lumbar_L3": (0.0, -1.0, 112.0),
        "Pelvis_Center": (0.0, 0.0, 98.0),
        "L_Shoulder": (-23.0, 0.0, 153.0),
        "L_Elbow": (-33.0, 10.0, 126.0),
        "L_Wrist": (-26.0, 24.0, 108.0),
        "L_Hand": (-23.0, 29.0, 104.0),
        "R_Shoulder": (23.0, 0.0, 153.0),
        "R_Elbow": (35.0, 14.0, 134.0),
        "R_Wrist": (30.0, 28.0, 148.0),
        "R_Hand": (27.0, 32.0, 155.0),
        "L_Hip": (-13.0, 0.0, 96.0),
        "L_Knee": (-18.0, 10.0, 54.0),
        "L_Ankle": (-20.0, -2.0, 12.0),
        "L_Heel": (-20.0, -6.0, 3.0),
        "L_Toe": (-22.0, 11.0, 3.0),
        "R_Hip": (13.0, 0.0, 96.0),
        "R_Knee": (18.0, 12.0, 54.0),
        "R_Ankle": (20.0, 2.0, 12.0),
        "R_Heel": (20.0, -3.0, 3.0),
        "R_Toe": (22.0, 15.0, 3.0),
        "L_Rib": (-16.0, 5.0, 126.0),
        "R_Rib": (16.0, 5.0, 126.0),
    }

    segments = [
        ("Head_Top", "Head_Center", "Core"),
        ("Head_Center", "Neck_C7", "Core"),
        ("Neck_C7", "Sternum", "Lower Back / Core"),
        ("Neck_C7", "Thoracic_T8", "Lower Back / Core"),
        ("Thoracic_T8", "Lumbar_L3", "Lower Back / Core"),
        ("Lumbar_L3", "Pelvis_Center", "Lower Back / Core"),
        ("Sternum", "L_Rib", "Lower Back / Core"),
        ("Sternum", "R_Rib", "Lower Back / Core"),
        ("L_Rib", "Lumbar_L3", "Lower Back / Core"),
        ("R_Rib", "Lumbar_L3", "Lower Back / Core"),
        ("Neck_C7", "L_Shoulder", "Shoulders / Rotator Cuff"),
        ("Neck_C7", "R_Shoulder", "Shoulders / Rotator Cuff"),
        ("Sternum", "L_Shoulder", "Shoulders / Rotator Cuff"),
        ("Sternum", "R_Shoulder", "Shoulders / Rotator Cuff"),
        ("L_Rib", "L_Shoulder", "Shoulders / Rotator Cuff"),
        ("R_Rib", "R_Shoulder", "Shoulders / Rotator Cuff"),
        ("L_Shoulder", "L_Elbow", "Shoulders / Rotator Cuff"),
        ("R_Shoulder", "R_Elbow", "Shoulders / Rotator Cuff"),
        ("L_Elbow", "L_Wrist", "Arms"),
        ("L_Wrist", "L_Hand", "Arms"),
        ("R_Elbow", "R_Wrist", "Arms"),
        ("R_Wrist", "R_Hand", "Arms"),
        ("Pelvis_Center", "L_Hip", "Lower Back / Core"),
        ("Pelvis_Center", "R_Hip", "Lower Back / Core"),
        ("L_Hip", "R_Hip", "Lower Back / Core"),
        ("L_Rib", "L_Hip", "Lower Back / Core"),
        ("R_Rib", "R_Hip", "Lower Back / Core"),
        ("L_Hip", "L_Knee", "Quads / Hamstrings"),
        ("R_Hip", "R_Knee", "Quads / Hamstrings"),
        ("L_Knee", "L_Ankle", "Calves / Achilles"),
        ("R_Knee", "R_Ankle", "Calves / Achilles"),
        ("L_Ankle", "L_Heel", "Calves / Achilles"),
        ("R_Ankle", "R_Heel", "Calves / Achilles"),
        ("L_Ankle", "L_Toe", "Calves / Achilles"),
        ("R_Ankle", "R_Toe", "Calves / Achilles"),
        ("L_Heel", "L_Toe", "Calves / Achilles"),
        ("R_Heel", "R_Toe", "Calves / Achilles"),
    ]

    fig = go.Figure()

    torso_keys = ["Neck_C7", "L_Shoulder", "R_Shoulder", "Sternum", "Thoracic_T8", "L_Rib", "R_Rib", "Lumbar_L3", "L_Hip", "R_Hip", "Pelvis_Center"]
    tx = [joints[k][0] for k in torso_keys]
    ty = [joints[k][1] for k in torso_keys]
    tz = [joints[k][2] for k in torso_keys]

    fig.add_trace(
        go.Mesh3d(
            x=tx,
            y=ty,
            z=tz,
            alphahull=0,
            opacity=0.22,
            color=theme_hex,
            name="Biomechanical Core Mesh",
            hoverinfo="skip",
        )
    )

    norm_x, norm_y, norm_z = [], [], []
    sore_x, sore_y, sore_z = [], [], []

    for j1, j2, region in segments:
        x1, y1, z1 = joints[j1]
        x2, y2, z2 = joints[j2]
        is_sore_region = (
            muscle_focus == "Full Body Systemic"
            or region == muscle_focus
            or (muscle_focus == "Knees / Patella" and ("Knee" in j1 or "Knee" in j2))
        )
        if is_sore_region and soreness_level >= 4:
            sore_x.extend([x1, x2, None])
            sore_y.extend([y1, y2, None])
            sore_z.extend([z1, z2, None])
        else:
            norm_x.extend([x1, x2, None])
            norm_y.extend([y1, y2, None])
            norm_z.extend([z1, z2, None])

    fig.add_trace(
        go.Scatter3d(
            x=norm_x,
            y=norm_y,
            z=norm_z,
            mode="lines",
            line=dict(color=theme_hex, width=7),
            name=f"{athlete_row['Athlete']} Kinetic Frame",
            hoverinfo="skip",
        )
    )

    if sore_x:
        fig.add_trace(
            go.Scatter3d(
                x=sore_x,
                y=sore_y,
                z=sore_z,
                mode="lines",
                line=dict(color=strain_hex, width=10),
                name=f"Strain Focus: {muscle_focus} ({soreness_level}/10)",
                hoverinfo="skip",
            )
        )

    theta = np.linspace(0, 2 * np.pi, 30)
    hx = 7.5 * np.cos(theta)
    hy = 2.0 + 7.5 * np.sin(theta)
    hz = np.full_like(theta, 174.0)
    fig.add_trace(
        go.Scatter3d(
            x=hx,
            y=hy,
            z=hz,
            mode="lines",
            line=dict(color="#22D3EE", width=4),
            name="Cranial Telemetry Halo",
            hoverinfo="skip",
        )
    )

    jx, jy, jz, j_names, j_colors, j_sizes = [], [], [], [], [], []
    for name, (x, y, z) in joints.items():
        jx.append(x)
        jy.append(y)
        jz.append(z)
        clean_name = name.replace("_", " ")
        j_names.append(f"<b>Joint Node:</b> {clean_name}<br><b>Elevation:</b> {z:.0f} cm<br><b>Readiness:</b> {readiness_index}%")

        highlight = False
        if muscle_focus == "Knees / Patella" and "Knee" in name:
            highlight = True
        elif muscle_focus == "Shoulders / Rotator Cuff" and ("Shoulder" in name or "Elbow" in name):
            highlight = True
        elif muscle_focus == "Quads / Hamstrings" and ("Hip" in name or "Knee" in name):
            highlight = True
        elif muscle_focus == "Calves / Achilles" and ("Ankle" in name or "Heel" in name or "Toe" in name):
            highlight = True
        elif muscle_focus == "Lower Back / Core" and ("Lumbar" in name or "Pelvis" in name or "Thoracic" in name):
            highlight = True
        elif muscle_focus == "Full Body Systemic" and soreness_level >= 6:
            highlight = True

        if highlight:
            j_colors.append(strain_hex)
            j_sizes.append(9 if name != "Head_Center" else 14)
        else:
            j_colors.append("#FFFFFF" if name != "Head_Center" else theme_hex)
            j_sizes.append(6 if name != "Head_Center" else 13)

    fig.add_trace(
        go.Scatter3d(
            x=jx,
            y=jy,
            z=jz,
            mode="markers",
            marker=dict(
                size=j_sizes,
                color=j_colors,
                line=dict(color=theme_hex, width=2),
                opacity=0.95,
            ),
            text=j_names,
            hovertemplate="%{text}<extra></extra>",
            name="Biomechanical Joint Sensors",
        )
    )

    for radius, opacity_val in [(28.0, 0.35), (42.0, 0.65)]:
        fx = radius * np.cos(theta)
        fy = radius * np.sin(theta)
        fz = np.zeros_like(theta)
        fig.add_trace(
            go.Scatter3d(
                x=fx,
                y=fy,
                z=fz,
                mode="lines",
                line=dict(color=theme_hex, width=3 if radius == 42.0 else 2),
                opacity=opacity_val,
                name="Force Plate Ring" if radius == 42.0 else "Inner Balance Ring",
                hoverinfo="skip",
                showlegend=(radius == 42.0),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=560,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(11, 15, 25, 0.75)",
            font=dict(size=11, color="#E5E7EB"),
        ),
        scene=dict(
            dragmode="turntable",
            xaxis=dict(
                title="",
                range=[-65, 65],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                showticklabels=False,
                showbackground=False,
            ),
            yaxis=dict(
                title="",
                range=[-65, 65],
                showgrid=True,
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
                showticklabels=False,
                showbackground=False,
            ),
            zaxis=dict(
                title="Height (cm)",
                range=[0, 195],
                showgrid=True,
                gridcolor="rgba(6, 182, 212, 0.12)",
                zeroline=False,
                showbackground=True,
                backgroundcolor="rgba(11, 15, 25, 0.45)",
            ),
            camera=dict(
                eye=dict(x=1.45, y=1.55, z=0.65),
                center=dict(x=0, y=0, z=-0.05),
            ),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.45),
        ),
    )
    return fig


# ==============================================================================
# 6. ANALYTICS & TELEMETRY CHARTS (TAB 2 & TAB 3)
# ==============================================================================
def build_acwr_progression_chart(df: pd.DataFrame) -> go.Figure:
    """Dual-axis Plotly chart plotting 28-day Acute vs Chronic load and ACWR safety bands."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Daily_Workload"],
            name="Daily Workload (AU)",
            marker=dict(color="rgba(6, 182, 212, 0.28)", line=dict(color="rgba(6, 182, 212, 0.65)", width=1)),
            hovertemplate="<b>%{x|%b %d}</b><br>Daily Load: %{y} AU<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Acute_Load_7D"],
            mode="lines+markers",
            name="7-Day Acute Load",
            line=dict(color="#10B981", width=3),
            marker=dict(size=6),
            hovertemplate="Acute Load (7d): %{y:.0f} AU<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Chronic_Load_28D"],
            mode="lines",
            name="28-Day Chronic Baseline",
            line=dict(color="#9CA3AF", width=2.5, dash="dash"),
            hovertemplate="Chronic Load (28d): %{y:.0f} AU<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["ACWR"],
            mode="lines+markers",
            name="ACWR Ratio",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=7, symbol="diamond", color="#FBBF24"),
            hovertemplate="<b>ACWR:</b> %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_hrect(
        y0=0.80,
        y1=1.30,
        line_width=0,
        fillcolor="rgba(16, 185, 129, 0.11)",
        annotation_text="Optimal Sweet Spot (0.8–1.3)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#34D399"),
        secondary_y=True,
    )
    fig.add_hline(
        y=1.50,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Danger Spike Threshold (1.50)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#F87171"),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        height=400,
        margin=dict(l=15, r=15, t=35, b=15),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Training Load (Arbitrary Units)", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="ACWR Ratio", range=[0.4, 2.0], secondary_y=True, showgrid=False)
    return fig


def build_volume_completion_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing Planned Sets vs Completed Sets over the past 7 sessions."""
    last_7 = df.tail(7).copy()
    last_7["Date_Label"] = last_7["Date"].dt.strftime("%b %d")
    last_7["Compliance_Pct"] = ((last_7["Completed_Sets"] / last_7["Planned_Sets"].replace(0, 1)) * 100).round(0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=last_7["Date_Label"],
            y=last_7["Planned_Sets"],
            name="Planned Sets",
            marker_color="rgba(156, 163, 175, 0.45)",
            text=last_7["Planned_Sets"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=last_7["Date_Label"],
            y=last_7["Completed_Sets"],
            name="Completed Sets",
            marker_color="#10B981",
            text=[f"{c} ({p:.0f}%)" for c, p in zip(last_7["Completed_Sets"], last_7["Compliance_Pct"])],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        barmode="group",
        height=360,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Working Sets", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(title="Last 7 Logged Sessions"),
    )
    return fig


def build_readiness_vs_fatigue_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot with numpy linear regression trendline correlating Sleep, Soreness, and Readiness."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Sleep_Hours"],
            y=df["Readiness_Score"],
            mode="markers",
            name="Session Telemetry",
            marker=dict(
                size=np.clip(df["Daily_Workload"] / 35.0, 9, 24),
                color=df["Soreness"],
                colorscale="Tealrose",
                showscale=True,
                colorbar=dict(title="Soreness"),
                line=dict(width=1, color="#FFFFFF"),
            ),
            text=df["Sport"],
            customdata=np.stack((df["Soreness"], df["Daily_Workload"], df["Date"].dt.strftime("%b %d")), axis=-1),
            hovertemplate=(
                "<b>%{customdata[2]} (%{text})</b><br>"
                "Sleep: %{x} hrs<br>"
                "Readiness: %{y}%<br>"
                "Soreness: %{customdata[0]}/10<br>"
                "Workload: %{customdata[1]} AU<extra></extra>"
            ),
        )
    )

    if len(df) >= 2:
        x_vals = df["Sleep_Hours"].to_numpy(dtype=float)
        y_vals = df["Readiness_Score"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 40)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Recovery Trend (+{slope:.1f}% / hr)",
                line=dict(color="#06B6D4", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        height=360,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Sleep Duration (Hours)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Daily Readiness Score (%)", gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


def build_dual_radar_chart(
    athlete_row: pd.Series,
    compare_user: bool,
    user_traits: Dict[str, int],
) -> go.Figure:
    """Builds a 6-axis Spider/Radar chart comparing the Pro Athlete vs User Benchmarks."""
    categories = ["Pace", "Endurance", "Agility", "Power", "Tactical IQ", "Recovery"]
    pro_vals = [
        int(athlete_row["Pace"]),
        int(athlete_row["Endurance"]),
        int(athlete_row["Agility"]),
        int(athlete_row["Power"]),
        int(athlete_row["Tactical_IQ"]),
        int(athlete_row["Recovery"]),
    ]
    cat_closed = categories + [categories[0]]
    pro_closed = pro_vals + [pro_vals[0]]

    theme_hex = str(athlete_row["Jersey_Color"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=pro_closed,
            theta=cat_closed,
            fill="toself",
            fillcolor=f"rgba({int(theme_hex[1:3], 16)}, {int(theme_hex[3:5], 16)}, {int(theme_hex[5:7], 16)}, 0.28)",
            line=dict(color=theme_hex, width=3),
            marker=dict(size=8, color=theme_hex),
            name=f"{athlete_row['Athlete']} (OVR {athlete_row['OVR']})",
        )
    )

    if compare_user:
        u_vals = [user_traits[c] for c in categories]
        u_closed = u_vals + [u_vals[0]]
        u_ovr = int(round(np.mean(u_vals)))
        fig.add_trace(
            go.Scatterpolar(
                r=u_closed,
                theta=cat_closed,
                fill="toself",
                fillcolor="rgba(6, 182, 212, 0.24)",
                line=dict(color="#06B6D4", width=3, dash="dot"),
                marker=dict(size=7, color="#22D3EE"),
                name=f"User Benchmark (OVR {u_ovr})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=40, r=40, t=35, b=35),
        polar=dict(
            bgcolor="rgba(17, 24, 39, 0.65)",
            radialaxis=dict(
                visible=True,
                range=[40, 100],
                gridcolor="rgba(255, 255, 255, 0.10)",
                tickfont=dict(size=10, color="#9CA3AF"),
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.12)",
                tickfont=dict(size=12, color="#F9FAFB"),
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


# ==============================================================================
# 7. MAIN STREAMLIT APPLICATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    if "athletes_df" not in st.session_state:
        st.session_state.athletes_df = get_pro_athletes_db()

    if "training_log" not in st.session_state:
        st.session_state.training_log = generate_28_day_seed_log()

    athletes_df: pd.DataFrame = st.session_state.athletes_df
    df_log: pd.DataFrame = st.session_state.training_log

    # --------------------------------------------------------------------------
    # SIDEBAR: BIOMETRICS, SCOUTING SELECTOR & DUAL-KEY LLM CONFIGURATION
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚡ AERO-KINETIX LAB")
        st.caption("Biomechanical Telemetry & AI Conditioning")
        st.markdown("---")

        st.markdown("### 🏆 Active Sport & Pro Benchmark")
        sports_list = athletes_df["Sport"].unique().tolist()
        selected_sport = st.selectbox("Select Primary Sport", sports_list, index=0)

        sport_subset = athletes_df[athletes_df["Sport"] == selected_sport]
        leagues_list = sport_subset["Team_League"].unique().tolist()
        selected_league = st.selectbox("Team / League Filter", ["All Leagues"] + leagues_list, index=0)

        if selected_league != "All Leagues":
            athlete_subset = sport_subset[sport_subset["Team_League"] == selected_league]
        else:
            athlete_subset = sport_subset

        selected_athlete_name = st.selectbox(
            "Pro Athlete Telemetry Profile",
            athlete_subset["Athlete"].tolist(),
            index=0,
        )
        active_athlete = athlete_subset[athlete_subset["Athlete"] == selected_athlete_name].iloc[0]

        st.markdown("---")
        st.markdown("### 🧬 Daily Physiological Inputs")
        sleep_hours = st.slider("Hours Slept (Last Night)", min_value=3.5, max_value=11.0, value=7.8, step=0.1)
        soreness_level = st.slider("Muscle Soreness (1 = Fresh, 10 = Severe)", min_value=1, max_value=10, value=3)
        muscle_focus = st.selectbox(
            "Primary Muscle Soreness Focus",
            [
                "Quads / Hamstrings",
                "Calves / Achilles",
                "Knees / Patella",
                "Shoulders / Rotator Cuff",
                "Lower Back / Core",
                "Full Body Systemic",
            ],
            index=0,
        )
        energy_level = st.slider("CNS Energy & Motivation (1–10)", min_value=1, max_value=10, value=8)

        st.markdown("---")
        st.markdown("### 📅 Microcycle Session Load")
        day_options = list(SPORT_WEEKLY_ROUTINES[selected_sport].keys())
        selected_day_routine = st.selectbox("Weekly Training Schedule Day", day_options, index=0)
        planned_duration = st.slider("Planned Duration (Minutes)", min_value=30, max_value=150, value=80, step=5)
        planned_rpe = st.slider("Target Session RPE (1–10)", min_value=1, max_value=10, value=7)
        planned_workload_today = planned_duration * planned_rpe

        st.markdown("---")
        st.markdown("### 🤖 Dual-Key AI Coach Engine")
        st.caption("Paste both keys, only one key, or leave both empty for offline heuristics.")

        routing_mode = st.selectbox(
            "Active Routing Mode",
            [
                "Auto-Detect (Groq ➔ OpenRouter ➔ Heuristic)",
                "Use Groq API (Failover to OpenRouter if set)",
                "Use OpenRouter API (Failover to Groq if set)",
                "Force Offline Heuristic Engine",
            ],
            index=0,
        )

        groq_api_key = st.text_input(
            "1. Groq API Key (gsk_...)",
            value=get_secret_api_key("GROQ_API_KEY"),
            type="password",
            help="Paste your Groq key here, or leave blank if using OpenRouter or Offline mode.",
        )
        groq_model = st.selectbox(
            "Groq Model",
            ["llama-3.3-70b-versatile", "llama3-8b-8192"],
            index=0,
        )

        openrouter_api_key = st.text_input(
            "2. OpenRouter API Key (sk-or-...)",
            value=get_secret_api_key("OPENROUTER_API_KEY"),
            type="password",
            help="Paste your OpenRouter key here, or leave blank if using Groq or Offline mode.",
        )
        openrouter_model = st.selectbox(
            "OpenRouter Model",
            ["meta-llama/llama-3.3-70b-instruct", "meta-llama/llama-3.1-8b-instruct"],
            index=0,
        )

        # Live Key Status Pill in Sidebar
        g_ready = "✅ Ready" if groq_api_key.strip() else "⚪ Empty"
        or_ready = "✅ Ready" if openrouter_api_key.strip() else "⚪ Empty"
        st.caption(f"**Key Status:** Groq: {g_ready} | OpenRouter: {or_ready}")

    inject_custom_css(str(active_athlete["Jersey_Color"]))

    telemetry = compute_readiness_and_acwr(
        sleep_hours=sleep_hours,
        soreness=soreness_level,
        energy=energy_level,
        df_log=df_log,
        planned_workload_today=planned_workload_today,
    )

    # --------------------------------------------------------------------------
    # TOP HERO BANNER
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="hero-banner">
            <div>
                <h1 class="hero-title">⚡ AERO-KINETIX AI PERFORMANCE LAB</h1>
                <div class="hero-subtitle">
                    Active Protocol: <b>{selected_sport}</b> &nbsp;|&nbsp; Pro Archetype: <b>{active_athlete['Athlete']}</b> ({active_athlete['Position']})
                    &nbsp;|&nbsp; Soreness Telemetry: <b>{muscle_focus} ({soreness_level}/10)</b>
                </div>
            </div>
            <div style="text-align: right;">
                <span class="status-badge {telemetry['readiness_badge']}">{telemetry['readiness_status']}</span>
                <span class="status-badge {telemetry['acwr_badge']}" style="margin-left: 6px;">ACWR: {telemetry['acwr']} ({telemetry['acwr_zone']})</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # TOP 4 TELEMETRY KPI CARDS
    # --------------------------------------------------------------------------
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Daily Readiness Index</div>
                <div class="metric-value" style="color: #10B981;">{telemetry['readiness_index']}%</div>
                <div class="metric-sub">Sleep {telemetry['sleep_score']}% · Muscle {telemetry['soreness_score']}% · CNS {telemetry['energy_score']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Real-Time ACWR Ratio</div>
                <div class="metric-value" style="color: #06B6D4;">{telemetry['acwr']:.2f}</div>
                <div class="metric-sub">Acute 7D: {telemetry['acute_7d']:.0f} AU vs Chronic 28D: {telemetry['chronic_28d']:.0f} AU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        mult_pct = int(round(float(telemetry["set_multiplier"]) * 100))
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Autoregulated Set Scaling</div>
                <div class="metric-value" style="color: #FBBF24;">{mult_pct}% Volume</div>
                <div class="metric-sub">Rest Multiplier: {telemetry['rest_multiplier']:.2f}x · Planned: {planned_workload_today} AU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Pro Archetype OVR Rating</div>
                <div class="metric-value" style="color: {active_athlete['Jersey_Color']};">{active_athlete['OVR']} OVR</div>
                <div class="metric-sub">{active_athlete['Athlete']} · {active_athlete['Team_League']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------------------------
    # 3-TAB NAVIGATION ARCHITECTURE
    # --------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "🏋️ Daily Adaptive Training & Coach",
            "📈 Performance Trends & History Graphs",
            "🌟 Pro Scouting & 3D Avatar Studio",
        ]
    )

    # ==========================================================================
    # TAB 1: DAILY ADAPTIVE TRAINING & AI COACH
    # ==========================================================================
    with tab1:
        schedule_df = build_adaptive_schedule(
            sport=selected_sport,
            day_key=selected_day_routine,
            metrics=telemetry,
            muscle_focus=muscle_focus,
        )
        total_base_sets = int(schedule_df["Baseline_Sets"].sum())
        total_adj_sets = int(schedule_df["Adaptive_Sets"].sum())

        coach_text, engine_badge = query_ai_coach(
            routing_mode=routing_mode,
            groq_key=groq_api_key,
            groq_model=groq_model,
            openrouter_key=openrouter_api_key,
            openrouter_model=openrouter_model,
            sport=selected_sport,
            athlete_name=str(active_athlete["Athlete"]),
            day_key=selected_day_routine,
            sleep_hours=sleep_hours,
            soreness=soreness_level,
            muscle_focus=muscle_focus,
            energy=energy_level,
            metrics=telemetry,
            total_base_sets=total_base_sets,
            total_adj_sets=total_adj_sets,
        )

        st.markdown(
            f"""
            <div class="coach-terminal">
                <div class="coach-header">🧠 AI Tactical & Physiological Prescription &nbsp;|&nbsp; Source: {engine_badge}</div>
                <div class="coach-body">{coach_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_sched, col_summary = st.columns([1.85, 1.0], gap="large")

        with col_sched:
            st.markdown(f"#### 📋 Adaptive Microcycle Prescription — `{selected_day_routine}`")
            for _, row in schedule_df.iterrows():
                st.markdown(
                    f"""
                    <div class="drill-card">
                        <div>
                            <div style="font-weight: 700; font-size: 0.98rem; color: #F9FAFB;">{row['Drill_Name']}</div>
                            <div style="font-size: 0.80rem; color: #9CA3AF; margin-top: 2px;">
                                🎯 {row['Biomechanical_Focus']} &nbsp;|&nbsp; ⚡ {row['Target_Intensity']}
                            </div>
                        </div>
                        <div style="text-align: right; font-family: 'JetBrains Mono', monospace;">
                            <span class="status-badge badge-optimal">{row['Adaptive_Sets']} Sets ({row['Set_Delta']})</span>
                            <span class="status-badge badge-info" style="margin-left: 6px;">Rest: {row['Adaptive_Rest_Sec']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
            )

        with col_summary:
            st.markdown("#### ⚖️ Autoregulation Breakdown")
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">Session Volume Rescaling</div>
                    <div style="font-size: 1.35rem; font-weight: 700; color: #F9FAFB; margin: 0.3rem 0;">
                        {total_base_sets} Baseline Sets ➔ <span style="color: #10B981;">{total_adj_sets} Adaptive Sets</span>
                    </div>
                    <p style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 0.6rem;">
                        Target Intensity Band: <b style="color: #22D3EE;">{telemetry['intensity_tag']}</b>
                    </p>
                    <hr style="border-color: rgba(255,255,255,0.08);">
                    <div style="font-size: 0.83rem; color: #D1D5DB; line-height: 1.6;">
                        • <b>Sleep Restoration Contribution:</b> {telemetry['sleep_score']}% (Weight: 40%)<br>
                        • <b>Neuromuscular Freshness:</b> {telemetry['soreness_score']}% (Weight: 35%)<br>
                        • <b>Subjective CNS Readiness:</b> {telemetry['energy_score']}% (Weight: 25%)<br>
                        • <b>Localized Tissue Guard:</b> {muscle_focus}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"📅 View Full 5-Day {selected_sport} Periodization Template", expanded=False):
                for d_name, d_list in SPORT_WEEKLY_ROUTINES[selected_sport].items():
                    drills_joined = ", ".join([str(d["Drill"]) for d in d_list])
                    st.markdown(f"**{d_name}**  \n<span style='color:#9CA3AF;font-size:0.84rem;'>{drills_joined}</span>", unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: PERFORMANCE TRENDS, HISTORY GRAPHS & SESSION LOGGER
    # ==========================================================================
    with tab2:
        st.markdown("#### 📊 28-Day Acute:Chronic Workload Ratio (ACWR) & Load Progression")
        acwr_fig = build_acwr_progression_chart(df_log)
        st.plotly_chart(acwr_fig, use_container_width=True)

        g_col1, g_col2 = st.columns(2, gap="large")
        with g_col1:
            st.markdown("#### 🏋️ Volume Completion (Planned vs. Completed Sets — Last 7 Sessions)")
            vol_fig = build_volume_completion_chart(df_log)
            st.plotly_chart(vol_fig, use_container_width=True)

        with g_col2:
            st.markdown("#### 🛌 Readiness vs. Sleep & Soreness Correlation Trend")
            scatter_fig = build_readiness_vs_fatigue_scatter(df_log)
            st.plotly_chart(scatter_fig, use_container_width=True)

        st.markdown("---")
        form_col, table_col = st.columns([1.05, 1.65], gap="large")

        with form_col:
            st.markdown("#### 📝 Log Today's Completed Session")
            with st.form("log_session_form", clear_on_submit=False):
                log_date_input = st.date_input("Session Date", value=datetime.now().date())
                log_sport_input = st.selectbox("Sport Logged", sports_list, index=sports_list.index(selected_sport))
                c1, c2 = st.columns(2)
                with c1:
                    log_duration = st.number_input("Duration (Min)", min_value=15, max_value=240, value=int(planned_duration), step=5)
                    log_planned_sets = st.number_input("Planned Sets", min_value=1, max_value=40, value=int(total_adj_sets))
                with c2:
                    log_rpe = st.number_input("Session RPE (1–10)", min_value=1, max_value=10, value=int(planned_rpe))
                    log_completed_sets = st.number_input("Completed Sets", min_value=1, max_value=40, value=int(total_adj_sets))

                submitted = st.form_submit_button("⚡ Commit Session to Telemetry Database", use_container_width=True)
                if submitted:
                    new_entry = pd.DataFrame(
                        [
                            {
                                "Date": pd.to_datetime(log_date_input),
                                "Sport": log_sport_input,
                                "Duration_Min": int(log_duration),
                                "RPE": int(log_rpe),
                                "Daily_Workload": int(log_duration * log_rpe),
                                "Sleep_Hours": float(sleep_hours),
                                "Soreness": int(soreness_level),
                                "Energy": int(energy_level),
                                "Readiness_Score": float(telemetry["readiness_index"]),
                                "Planned_Sets": int(log_planned_sets),
                                "Completed_Sets": int(log_completed_sets),
                            }
                        ]
                    )
                    updated_df = pd.concat([st.session_state.training_log, new_entry], ignore_index=True)
                    st.session_state.training_log = recompute_rolling_metrics(updated_df)
                    st.success(
                        f"Logged {log_sport_input} session ({int(log_duration * log_rpe)} AU, {log_completed_sets}/{log_planned_sets} sets). Charts updated in real time!"
                    )
                    st.rerun()

        with table_col:
            st.markdown("#### 🗄️ Historical Telemetry Ledger (Latest 10 Sessions)")
            display_df = st.session_state.training_log.tail(10).sort_values("Date", ascending=False).copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(
                display_df[
                    [
                        "Date",
                        "Sport",
                        "Duration_Min",
                        "RPE",
                        "Daily_Workload",
                        "Readiness_Score",
                        "Planned_Sets",
                        "Completed_Sets",
                        "ACWR",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    # ==========================================================================
    # TAB 3: PRO SCOUTING & 3D AVATAR STUDIO
    # ==========================================================================
    with tab3:
        st.markdown("#### 🎛️ Cascading Pro Scouting Filter & Telemetry Benchmark")
        f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.3, 1.3, 1.1])

        with f_col1:
            scout_sport = st.selectbox("1. Sport Discipline", sports_list, index=sports_list.index(selected_sport), key="scout_sport")
        scout_sport_df = athletes_df[athletes_df["Sport"] == scout_sport]

        with f_col2:
            scout_leagues = scout_sport_df["Team_League"].unique().tolist()
            scout_league = st.selectbox("2. Team / League", ["All Leagues"] + scout_leagues, index=0, key="scout_league")
        if scout_league != "All Leagues":
            scout_athlete_df = scout_sport_df[scout_sport_df["Team_League"] == scout_league]
        else:
            scout_athlete_df = scout_sport_df

        with f_col3:
            scout_athlete_name = st.selectbox("3. Pro Athlete", scout_athlete_df["Athlete"].tolist(), index=0, key="scout_athlete")
        scout_row = scout_athlete_df[scout_athlete_df["Athlete"] == scout_athlete_name].iloc[0]

        with f_col4:
            st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
            compare_user_toggle = st.toggle("Compare Athlete vs. User", value=True)

        default_compliance_bonus = int(
            round((df_log["Completed_Sets"].tail(7).sum() / max(1, df_log["Planned_Sets"].tail(7).sum())) * 10)
        )
        user_traits = {
            "Pace": min(99, 74 + default_compliance_bonus),
            "Endurance": min(99, int(round(telemetry["readiness_index"] * 0.45 + 42))),
            "Agility": min(99, 76 + default_compliance_bonus),
            "Power": min(99, 72 + default_compliance_bonus),
            "Tactical IQ": 82,
            "Recovery": min(99, int(round(telemetry["readiness_index"] * 0.55 + 35))),
        }

        if compare_user_toggle:
            with st.expander("🎚️ Customize Your Personal Athletic Telemetry Benchmarks (0–99)", expanded=False):
                u1, u2, u3, u4, u5, u6 = st.columns(6)
                user_traits["Pace"] = u1.slider("Your Pace", 40, 99, user_traits["Pace"])
                user_traits["Endurance"] = u2.slider("Your Endurance", 40, 99, user_traits["Endurance"])
                user_traits["Agility"] = u3.slider("Your Agility", 40, 99, user_traits["Agility"])
                user_traits["Power"] = u4.slider("Your Power", 40, 99, user_traits["Power"])
                user_traits["Tactical IQ"] = u5.slider("Your Tactical IQ", 40, 99, user_traits["Tactical IQ"])
                user_traits["Recovery"] = u6.slider("Your Recovery", 40, 99, user_traits["Recovery"])

        avatar_col, radar_col = st.columns([1.15, 1.0], gap="large")

        with avatar_col:
            st.markdown(
                f"#### 🧍 Interactive 3D Biomechanical Avatar — `{scout_row['Athlete']}` "
                f"<span class='status-badge badge-info' style='margin-left:8px;'>Click & Drag 360° · Scroll to Zoom</span>",
                unsafe_allow_html=True,
            )
            avatar_fig = build_3d_biomechanical_avatar(
                athlete_row=scout_row,
                muscle_focus=muscle_focus,
                soreness_level=soreness_level,
                readiness_index=float(telemetry["readiness_index"]),
            )
            st.plotly_chart(avatar_fig, use_container_width=True)

        with radar_col:
            st.markdown(f"#### 🕸️ 6-Trait Dual Radar Telemetry — `{scout_row['Athlete']}`")
            radar_fig = build_dual_radar_chart(
                athlete_row=scout_row,
                compare_user=compare_user_toggle,
                user_traits=user_traits,
            )
            st.plotly_chart(radar_fig, use_container_width=True)

            trait_keys = ["Pace", "Endurance", "Agility", "Power", "Tactical_IQ", "Recovery"]
            pro_vec = np.array([float(scout_row[k]) for k in trait_keys])
            user_vec = np.array(
                [
                    float(user_traits["Pace"]),
                    float(user_traits["Endurance"]),
                    float(user_traits["Agility"]),
                    float(user_traits["Power"]),
                    float(user_traits["Tactical IQ"]),
                    float(user_traits["Recovery"]),
                ]
            )
            similarity_pct = round(float(100.0 - np.mean(np.abs(pro_vec - user_vec))), 1)

            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 4px solid {scout_row['Jersey_Color']};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="metric-label">{scout_row['Sport']} · {scout_row['Team_League']}</div>
                            <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">
                                {scout_row['Athlete']} <span style="color:{scout_row['Jersey_Color']};">({scout_row['OVR']} OVR)</span>
                            </div>
                        </div>
                        <span class="status-badge badge-optimal">Archetype Match: {similarity_pct}%</span>
                    </div>
                    <div style="font-size: 0.85rem; color: #9CA3AF; margin-top: 0.45rem;">
                        <b>Primary Position:</b> {scout_row['Position']}<br>
                        <b>Biomechanical Signature:</b> {scout_row['Signature_Style']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
