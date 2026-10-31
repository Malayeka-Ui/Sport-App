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
# 1. PAGE CONFIGURATION & USER-FRIENDLY DARK THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="AERO-KINETIX AI | Smart Sports Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_custom_css(accent_hex: str = "#10B981") -> None:
    """Injects clean, modern, user-friendly dark styling with styled sidebar navigation."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {{
            --bg-main: #0B0F19;
            --bg-card: rgba(20, 29, 47, 0.85);
            --accent-green: #10B981;
            --accent-cyan: #06B6D4;
            --accent-athlete: {accent_hex};
            --text-light: #F9FAFB;
            --text-dim: #9CA3AF;
        }}

        .stApp {{
            background: radial-gradient(circle at 15% 10%, rgba(16, 185, 129, 0.08), transparent 35%),
                        radial-gradient(circle at 85% 15%, rgba(6, 182, 212, 0.08), transparent 35%),
                        #0B0F19;
            color: var(--text-light);
            font-family: 'Inter', sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background: #0E1525 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        /* Style Sidebar Radio Buttons as a Modern Navigation Menu */
        div[role="radiogroup"] > label {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 10px 14px !important;
            margin-bottom: 6px !important;
            transition: all 0.2s ease;
            cursor: pointer;
        }}

        div[role="radiogroup"] > label:hover {{
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.4);
        }}

        /* Friendly Welcome Header */
        .welcome-banner {{
            background: linear-gradient(135deg, rgba(20, 29, 47, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-left: 6px solid var(--accent-athlete);
            border-radius: 16px;
            padding: 1.25rem 1.6rem;
            margin-bottom: 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }}

        .welcome-title {{
            font-size: 1.55rem;
            font-weight: 800;
            margin: 0;
            color: #FFFFFF;
        }}

        .welcome-sub {{
            color: #9CA3AF;
            font-size: 0.92rem;
            margin-top: 4px;
        }}

        /* Friendly Summary Cards */
        .friendly-card {{
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-radius: 16px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
        }}

        .card-label {{
            font-size: 0.82rem;
            font-weight: 600;
            color: #9CA3AF;
            margin-bottom: 0.35rem;
        }}

        .card-big-value {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
            line-height: 1.2;
        }}

        .card-helper {{
            font-size: 0.82rem;
            color: #D1D5DB;
            margin-top: 0.45rem;
        }}

        /* Progress Bar inside Cards */
        .progress-track {{
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 8px;
        }}

        /* Pills / Badges */
        .pill {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
        }}

        .pill-green {{
            background: rgba(16, 185, 129, 0.18);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .pill-yellow {{
            background: rgba(245, 158, 11, 0.18);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.4);
        }}

        .pill-red {{
            background: rgba(239, 68, 68, 0.18);
            color: #F87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}

        .pill-blue {{
            background: rgba(6, 182, 212, 0.18);
            color: #22D3EE;
            border: 1px solid rgba(6, 182, 212, 0.4);
        }}

        /* AI Coach Message Bubble */
        .coach-box {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(6, 182, 212, 0.09) 100%);
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin: 0.5rem 0 1.25rem 0;
        }}

        .coach-title {{
            font-size: 0.88rem;
            font-weight: 700;
            color: #34D399;
            margin-bottom: 0.4rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .coach-message {{
            font-size: 1.0rem;
            line-height: 1.65;
            color: #F3F4F6;
        }}

        /* Workout Step Card */
        .workout-step {{
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 4px solid #06B6D4;
            border-radius: 12px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. MULTI-SPORT PRO ATHLETE DATABASE & 28-DAY HISTORY GENERATOR
# ==============================================================================
def get_pro_athletes_db() -> pd.DataFrame:
    """Returns multi-sport athlete profiles with traits and jersey colors."""
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
            "Signature_Style": "Steep jump smashes & wide court reach",
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
            "Signature_Style": "Explosive net lunges & sixth-sense anticipation",
        },
        {
            "Sport": "Badminton",
            "Team_League": "BWF World Tour (South Korea)",
            "Athlete": "An Se-young",
            "Position": "Women's Singles Defensive Specialist",
            "Pace": 93,
            "Endurance": 99,
            "Agility": 97,
            "Power": 88,
            "Tactical_IQ": 96,
            "Recovery": 98,
            "OVR": 95,
            "Jersey_Color": "#3B82F6",
            "Signature_Style": "Relentless rally stamina & elastic split-steps",
        },
        # FOOTBALL
        {
            "Sport": "Football",
            "Team_League": "La Liga (Real Madrid)",
            "Athlete": "Kylian Mbappé",
            "Position": "Forward / Striker",
            "Pace": 99,
            "Endurance": 89,
            "Agility": 95,
            "Power": 88,
            "Tactical_IQ": 92,
            "Recovery": 91,
            "OVR": 94,
            "Jersey_Color": "#06B6D4",
            "Signature_Style": "Lightning counter-attack sprints & clinical finishing",
        },
        {
            "Sport": "Football",
            "Team_League": "Premier League (Manchester City)",
            "Athlete": "Erling Haaland",
            "Position": "Center Striker (No. 9)",
            "Pace": 94,
            "Endurance": 88,
            "Agility": 85,
            "Power": 99,
            "Tactical_IQ": 94,
            "Recovery": 93,
            "OVR": 94,
            "Jersey_Color": "#38BDF8",
            "Signature_Style": "Explosive box power & aerial vertical leaps",
        },
        {
            "Sport": "Football",
            "Team_League": "La Liga (Real Madrid)",
            "Athlete": "Jude Bellingham",
            "Position": "Box-to-Box Midfielder",
            "Pace": 88,
            "Endurance": 97,
            "Agility": 91,
            "Power": 90,
            "Tactical_IQ": 95,
            "Recovery": 94,
            "OVR": 93,
            "Jersey_Color": "#A855F7",
            "Signature_Style": "All-pitch stamina, pressing & late box runs",
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
            "Signature_Style": "Power rim drives, elite playmaking & peak recovery",
        },
        {
            "Sport": "Basketball",
            "Team_League": "NBA (Golden State Warriors)",
            "Athlete": "Stephen Curry",
            "Position": "Point Guard (PG)",
            "Pace": 92,
            "Endurance": 99,
            "Agility": 97,
            "Power": 82,
            "Tactical_IQ": 98,
            "Recovery": 94,
            "OVR": 95,
            "Jersey_Color": "#2563EB",
            "Signature_Style": "Constant off-ball movement & quick-release shooting",
        },
        {
            "Sport": "Basketball",
            "Team_League": "NBA (Denver Nuggets)",
            "Athlete": "Nikola Jokić",
            "Position": "Center (C)",
            "Pace": 78,
            "Endurance": 95,
            "Agility": 86,
            "Power": 96,
            "Tactical_IQ": 99,
            "Recovery": 95,
            "OVR": 96,
            "Jersey_Color": "#F97316",
            "Signature_Style": "Full-court passing vision, post balance & touch",
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
            "Signature_Style": "Smooth cover drives, head balance & crease footwork",
        },
        {
            "Sport": "Cricket",
            "Team_League": "ICC / IPL (India National Team)",
            "Athlete": "Virat Kohli",
            "Position": "Top-Order Batter (No. 3)",
            "Pace": 94,
            "Endurance": 98,
            "Agility": 95,
            "Power": 91,
            "Tactical_IQ": 98,
            "Recovery": 96,
            "OVR": 96,
            "Jersey_Color": "#0EA5E9",
            "Signature_Style": "Fast running between wickets & high-pressure chases",
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
            "Signature_Style": "145+ kph swinging yorkers & explosive bowling action",
        },
    ]
    return pd.DataFrame(athletes)


def generate_28_day_seed_log() -> pd.DataFrame:
    """Generates 28 days of realistic training history so charts work immediately on launch."""
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
# 3. SPORT WORKOUT SCHEDULES & READINESS / ACWR CALCULATOR
# ==============================================================================
SPORT_WEEKLY_ROUTINES: Dict[str, Dict[str, List[Dict[str, object]]]] = {
    "Badminton": {
        "Monday: Shadow Footwork": [
            {"Drill": "6-Corner Shadow Footwork", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Fast split-steps & court recovery"},
            {"Drill": "Rear-Court Jump to Net Lunge", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Back-to-front court transition"},
            {"Drill": "Low Lunge Hold & Racket Balance", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Knee stability & leg strength"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Side-to-Side Defense Shuttles", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Quick lateral push-off"},
            {"Drill": "Reactive Split-Step Sprints", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Reaction speed & first step"},
            {"Drill": "Hurdle Hops to Deep Lunge", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Explosive landing control"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "Rapid Net Kill & Lift Multi-Shuttle", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Front-court wrist speed"},
            {"Drill": "2-vs-1 Flat Drive & Smash Defense", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Fast reflex blocks"},
            {"Drill": "Cross-Court Drop & Clear Accuracy", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Shot placement under fatigue"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "40s Work / 20s Rest Multi-Shuttle Rally", "Base_Sets": 8, "Base_Rest_Sec": 40, "Focus": "Stamina & heart-rate recovery"},
            {"Drill": "Repeat Jump-Smash Power Drill", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Overhead power endurance"},
            {"Drill": "Shoulder Resistance Band Care", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Shoulder joint protection"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "Clutch 21-Point Match Play (Start 14-14)", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Winning tight end-game points"},
            {"Drill": "Serve & 3rd-Shot Attack Patterns", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "Controlling the start of rallies"},
            {"Drill": "Cool-Down Stretching & Foam Roll", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Full-body muscle relaxation"},
        ],
    },
    "Football": {
        "Monday: Shadow Footwork": [
            {"Drill": "Shoulder-Check Scanning & First Touch", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Awareness before receiving the ball"},
            {"Drill": "Close-Control Cone Weave & Stop", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Sharp braking & ball control"},
            {"Drill": "Hamstring & Groin Injury-Prevention Prep", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Protecting leg muscles"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "30m Explosive Speed Sprints", "Base_Sets": 6, "Base_Rest_Sec": 90, "Focus": "Top sprinting speed"},
            {"Drill": "5-10-5 Agility Shuttle Turn", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Sharp direction changes"},
            {"Drill": "1v1 Recovery Defending Sprints", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Acceleration & hip turns"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "6v3 One-Touch Pressing Rondo", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Fast passing under pressure"},
            {"Drill": "Wing Overlap & Cutback Finishing", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Composed goal finishing"},
            {"Drill": "Long Diagonal Pass & First-Touch Cushion", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Switching play cleanly"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "Repeated Sprint Shuttles (6 x 40m)", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Late-match sprint stamina"},
            {"Drill": "Box-to-Box High-Tempo Runs (15s On/15s Off)", "Base_Sets": 8, "Base_Rest_Sec": 30, "Focus": "Engine & lung capacity"},
            {"Drill": "Core Stability Plank & Rotation Series", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Balance in physical duels"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "11v11 High-Press vs Low-Block Scrimmage", "Base_Sets": 4, "Base_Rest_Sec": 120, "Focus": "Team shape & tactical discipline"},
            {"Drill": "Corner & Free-Kick Attacking Routines", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Set-piece execution"},
            {"Drill": "Post-Match Mobility & Leg Flush", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Clearing muscle fatigue"},
        ],
    },
    "Basketball": {
        "Monday: Shadow Footwork": [
            {"Drill": "Defensive Slide & Closeout Footwork", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Staying in front of ball handlers"},
            {"Drill": "Triple-Threat Jab-Step & Separation", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "Creating space for your shot"},
            {"Drill": "Single-Leg Ankle Balance Holds", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Ankle stability & safety"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Full-Court Fast-Break Euro-Step Finish", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Controlled finishing at top speed"},
            {"Drill": "Key-to-Perimeter Agility Cone Sprints", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Forward-to-backward speed"},
            {"Drill": "Vertical Jump & Soft 2-Foot Landings", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Explosive bounce & knee protection"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "Pick-and-Roll Pocket Pass or Pull-Up 3PT", "Base_Sets": 6, "Base_Rest_Sec": 55, "Focus": "Reading the defense"},
            {"Drill": "Off-Screen Relocation Catch-and-Shoot", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Quick shooting footwork"},
            {"Drill": "Contact Layup Finishing Drill", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Scoring through physical contact"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "17-Second Sideline Conditioning Shuttles", "Base_Sets": 7, "Base_Rest_Sec": 45, "Focus": "4th-quarter stamina"},
            {"Drill": "Tired-Legs Free-Throw Challenge", "Base_Sets": 5, "Base_Rest_Sec": 35, "Focus": "Calm shooting with high heart rate"},
            {"Drill": "Slow Single-Leg Squat Lowering", "Base_Sets": 4, "Base_Rest_Sec": 50, "Focus": "Healthy patellar tendons"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "Late-Clock (8s Remaining) Half-Court Play", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Clutch shot creation"},
            {"Drill": "Full-Court Press Break Scrimmage", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Protecting the ball under pressure"},
            {"Drill": "Hip & Upper-Back Stretching Flow", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Loosening tight muscles"},
        ],
    },
    "Cricket": {
        "Monday: Shadow Footwork": [
            {"Drill": "Front-Foot & Back-Foot Batting Trigger Shadow", "Base_Sets": 6, "Base_Rest_Sec": 40, "Focus": "Head stillness & balanced stance"},
            {"Drill": "Slip Fielding Low-Catch Split-Steps", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "Sharp reflex catching"},
            {"Drill": "Shoulder & Upper-Back Rotation Mobility", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Smooth throwing & bowling arm"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Quick 2-Run Turning Sprints With Bat", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Fast crease turns & acceleration"},
            {"Drill": "Bowler Run-Up Rhythm & Gather Stride", "Base_Sets": 6, "Base_Rest_Sec": 70, "Focus": "Consistent front-foot landing"},
            {"Drill": "In-Field Pick-Up & Direct-Hit Throws", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Balanced throwing on the move"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "Fast Short-Ball Pull & Leave Net Session", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Picking length early"},
            {"Drill": "Spin Footwork: Sweep & Step Down the Pitch", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Rotating strike against spin"},
            {"Drill": "Death-Overs Yorker Target Bowling", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Hitting the blockhole under pressure"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "Simulated 6-Over Bowling / 100-Ball Batting", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Long-spell concentration & stamina"},
            {"Drill": "Boundary Dive Stop & Relay Sprints", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Repeat sprinting in the field"},
            {"Drill": "Lower-Back & Core Stability Holds", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Protecting the lower spine"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "Powerplay / Death-Overs Run-Chase Scenario", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Finding boundaries under pressure"},
            {"Drill": "Tactical Field-Setting vs 360° Batter", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Smart match-up strategy"},
            {"Drill": "Hamstring & Shoulder Cool-Down Stretch", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Post-match recovery"},
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
    """Calculates Daily Readiness (0-100%), live ACWR, and workout scaling in plain English."""
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
        acwr_zone = "Light Load (<0.8)"
        acwr_pill = "pill-blue"
        acwr_explain = "Your recent training is lighter than your 4-week average."
    elif acwr <= 1.30:
        acwr_zone = "Sweet Spot (0.8–1.3)"
        acwr_pill = "pill-green"
        acwr_explain = "Optimal balance! Fitness is building safely with low injury risk."
    elif acwr <= 1.50:
        acwr_zone = "High Load (1.3–1.5)"
        acwr_pill = "pill-yellow"
        acwr_explain = "Training load is climbing fast—watch for extra fatigue."
    else:
        acwr_zone = "Danger Spike (>1.5)"
        acwr_pill = "pill-red"
        acwr_explain = "Warning: Weekly load jumped too high. Extra rest recommended."

    if readiness_index >= 80 and acwr <= 1.30:
        set_multiplier = 1.15
        rest_multiplier = 0.90
        intensity_tag = "High Intensity (Push Day 🔥)"
        readiness_pill = "pill-green"
        readiness_status = "Ready to Push"
        readiness_color = "#10B981"
    elif readiness_index >= 62 and acwr <= 1.45:
        set_multiplier = 1.00
        rest_multiplier = 1.00
        intensity_tag = "Normal Pace (Balanced ✅)"
        readiness_pill = "pill-blue"
        readiness_status = "Good to Train"
        readiness_color = "#06B6D4"
    elif readiness_index >= 45 and acwr <= 1.55:
        set_multiplier = 0.72
        rest_multiplier = 1.30
        intensity_tag = "Lighter Pace (Recovery Focus ⚠️)"
        readiness_pill = "pill-yellow"
        readiness_status = "Moderate Fatigue"
        readiness_color = "#F59E0B"
    else:
        set_multiplier = 0.50
        rest_multiplier = 1.60
        intensity_tag = "Easy Active Recovery (Rest & Protect 🛑)"
        readiness_pill = "pill-red"
        readiness_status = "Needs Recovery"
        readiness_color = "#EF4444"

    return {
        "readiness_index": readiness_index,
        "readiness_color": readiness_color,
        "sleep_score": round(sleep_score, 1),
        "soreness_score": round(soreness_score, 1),
        "energy_score": round(energy_score, 1),
        "acute_7d": round(acute_7d, 1),
        "chronic_28d": round(chronic_28d, 1),
        "acwr": acwr,
        "acwr_zone": acwr_zone,
        "acwr_pill": acwr_pill,
        "acwr_explain": acwr_explain,
        "set_multiplier": set_multiplier,
        "rest_multiplier": rest_multiplier,
        "intensity_tag": intensity_tag,
        "readiness_pill": readiness_pill,
        "readiness_status": readiness_status,
    }


def build_adaptive_schedule(
    sport: str,
    day_key: str,
    metrics: Dict[str, object],
    muscle_focus: str,
) -> pd.DataFrame:
    """Adjusts sets and rest times automatically based on today's readiness."""
    base_drills = SPORT_WEEKLY_ROUTINES[sport][day_key]
    scaled_rows = []

    for item in base_drills:
        base_sets = int(item["Base_Sets"])
        base_rest = int(item["Base_Rest_Sec"])
        adj_sets = max(2, int(round(base_sets * float(metrics["set_multiplier"]))))
        adj_rest = int(round((base_rest * float(metrics["rest_multiplier"])) / 5.0) * 5)
        delta_sets = adj_sets - base_sets
        delta_str = f"+{delta_sets} bonus" if delta_sets > 0 else (f"{delta_sets} reduced" if delta_sets < 0 else "standard")

        scaled_rows.append(
            {
                "Drill_Name": item["Drill"],
                "Standard_Sets": base_sets,
                "Recommended_Sets": adj_sets,
                "Adjustment": delta_str,
                "Rest_Between_Sets": f"{adj_rest} sec",
                "Pace": metrics["intensity_tag"],
                "Coaching_Focus": f"{item['Focus']} (Care for: {muscle_focus})",
            }
        )
    return pd.DataFrame(scaled_rows)


# ==============================================================================
# 4. HIDDEN DUAL-KEY AI COACH ENGINE (GROQ OPENAI / LLAMA + OPENROUTER)
# ==============================================================================
def get_secret_api_key(key_name: str) -> str:
    """Silently loads API keys from environment variables or Streamlit secrets."""
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
    """Friendly, clear 3-sentence coaching advice when no API key is entered."""
    readiness = metrics["readiness_index"]
    acwr = metrics["acwr"]

    if total_adj_sets > total_base_sets:
        change_phrase = f"boosted from {total_base_sets} to {total_adj_sets} sets because your readiness is strong ({readiness}%)"
    elif total_adj_sets < total_base_sets:
        change_phrase = f"lightened from {total_base_sets} to {total_adj_sets} sets to protect your body ({readiness}% readiness)"
    else:
        change_phrase = f"kept steady at {total_adj_sets} sets ({readiness}% readiness)"

    sentence_1 = (
        f"Based on your {sleep_hours} hours of sleep, {energy}/10 energy, and {acwr} workload ratio ({metrics['acwr_zone']}), "
        f"today's {sport} workout ({day_key}) has been {change_phrase}."
    )

    if readiness < 60 or acwr > 1.35:
        sentence_2 = (
            f"We added extra rest time between sets so you can practice smooth, controlled technique like {athlete_name} "
            f"without overloading your {muscle_focus}."
        )
    else:
        sentence_2 = (
            f"Your body is fresh and in the safe training zone, so focus on crisp footwork and explosive speed inspired by {athlete_name}."
        )

    recovery_map = {
        "Quads / Hamstrings": "10 minutes of gentle hamstring stretching, foam rolling, and easy leg swings",
        "Calves / Achilles": "slow calf stretches, ankle mobility circles, and a 5-minute foot massage with a ball",
        "Lower Back / Core": "deep belly breathing on your back, gentle cat-cow spine stretches, and light core activation",
        "Shoulders / Rotator Cuff": "light resistance-band shoulder rotations, chest openers, and gentle arm circles",
        "Knees / Patella": "3 sets of 30-second wall-sit holds, quad foam rolling, and gentle knee bends",
        "Full Body Systemic": "10 minutes of slow deep breathing, full-body stretching, and plenty of hydration",
    }
    drill_rec = recovery_map.get(muscle_focus, recovery_map["Full Body Systemic"])
    sentence_3 = f"Finish your session with {drill_rec} to help your {muscle_focus} recover for tomorrow."

    return f"{sentence_1} {sentence_2} {sentence_3}"


def _call_groq(api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
    """Calls Groq API supporting both OpenAI GPT-OSS models and Llama models."""
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
    """Routes to Groq (OpenAI/Llama), OpenRouter, or the built-in Smart Coach engine."""
    fallback_text = generate_heuristic_coaching_prescription(
        sport, athlete_name, day_key, sleep_hours, soreness, muscle_focus, energy, metrics, total_base_sets, total_adj_sets
    )

    groq_clean = groq_key.strip()
    or_clean = openrouter_key.strip()

    if routing_mode == "Offline Smart Coach (No API Key Needed)" or (not groq_clean and not or_clean):
        return fallback_text, "Built-in Sports Science Coach"

    system_prompt = (
        "You are a friendly, encouraging elite Sports Science Coach. "
        "Write EXACTLY 3 clear, easy-to-understand sentences for the athlete. "
        "Sentence 1: Explain why today's sets and rest times were adjusted based on their Readiness Score and ACWR. "
        "Sentence 2: Give one practical movement tip for today's workout inspired by the pro athlete. "
        "Sentence 3: Recommend a simple active recovery stretch or drill for their sore muscle area."
    )

    user_prompt = (
        f"Sport: {sport} | Pro Role Model: {athlete_name} | Today's Focus: {day_key}\n"
        f"Check-in: Sleep={sleep_hours}h, Soreness={soreness}/10 ({muscle_focus}), Energy={energy}/10\n"
        f"Stats: Readiness={metrics['readiness_index']}%, ACWR={metrics['acwr']} ({metrics['acwr_zone']}), "
        f"Sets Adjusted={total_base_sets} -> {total_adj_sets}, Pace={metrics['intensity_tag']}."
    )

    attempt_queue: List[Tuple[str, str, str]] = []
    if routing_mode == "Auto-Detect (Uses Whichever Key You Paste)":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Prefer Groq API":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Prefer OpenRouter API":
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))

    for provider_label, key_val, model_val in attempt_queue:
        try:
            if provider_label == "Groq":
                text_out = _call_groq(key_val, model_val, system_prompt, user_prompt)
                return text_out, f"Groq AI ({model_val})"
            else:
                text_out = _call_openrouter(key_val, model_val, system_prompt, user_prompt)
                return text_out, f"OpenRouter AI ({model_val})"
        except Exception:
            continue

    return fallback_text, "Built-in Sports Science Coach (Fallback)"


# ==============================================================================
# 5. INTERACTIVE 3D MOUSE-ROTATABLE AVATAR (PLOTLY WEBGL)
# ==============================================================================
def build_3d_biomechanical_avatar(
    athlete_row: pd.Series,
    muscle_focus: str,
    soreness_level: int,
    readiness_index: float,
) -> go.Figure:
    """Renders a 360-degree rotatable 3D athlete avatar synced to jersey colors and sore muscles."""
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
            name="Core Body Volume",
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
            name=f"{athlete_row['Athlete']} Stance",
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
                name=f"Sore Area: {muscle_focus}",
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
            name="Head Tracking Halo",
            hoverinfo="skip",
        )
    )

    jx, jy, jz, j_names, j_colors, j_sizes = [], [], [], [], [], []
    for name, (x, y, z) in joints.items():
        jx.append(x)
        jy.append(y)
        jz.append(z)
        clean_name = name.replace("_", " ")
        j_names.append(f"<b>Joint:</b> {clean_name}<br><b>Height:</b> {z:.0f} cm<br><b>Readiness:</b> {readiness_index}%")

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
            name="Joint Nodes",
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
                name="Floor Balance Ring" if radius == 42.0 else "Inner Ring",
                hoverinfo="skip",
                showlegend=(radius == 42.0),
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=20, b=0),
        height=520,
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
            xaxis=dict(title="", range=[-65, 65], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(title="", range=[-65, 65], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(title="Height (cm)", range=[0, 195], showgrid=True, gridcolor="rgba(6, 182, 212, 0.12)", zeroline=False, showbackground=True, backgroundcolor="rgba(11, 15, 25, 0.45)"),
            camera=dict(eye=dict(x=1.45, y=1.55, z=0.65), center=dict(x=0, y=0, z=-0.05)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.45),
        ),
    )
    return fig


# ==============================================================================
# 6. ANALYTICS & PROGRESS CHARTS
# ==============================================================================
def build_acwr_progression_chart(df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart plotting 28-day Acute vs Chronic load and the Safe Sweet Spot."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Daily_Workload"],
            name="Daily Effort Score",
            marker=dict(color="rgba(6, 182, 212, 0.25)", line=dict(color="rgba(6, 182, 212, 0.6)", width=1)),
            hovertemplate="<b>%{x|%b %d}</b><br>Daily Effort: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Acute_Load_7D"],
            mode="lines+markers",
            name="7-Day Fatigue (Acute)",
            line=dict(color="#10B981", width=3),
            marker=dict(size=6),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Chronic_Load_28D"],
            mode="lines",
            name="28-Day Fitness Base (Chronic)",
            line=dict(color="#9CA3AF", width=2.5, dash="dash"),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["ACWR"],
            mode="lines+markers",
            name="Injury Risk Ratio (ACWR)",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=7, symbol="diamond", color="#FBBF24"),
        ),
        secondary_y=True,
    )

    fig.add_hrect(
        y0=0.80,
        y1=1.30,
        line_width=0,
        fillcolor="rgba(16, 185, 129, 0.11)",
        annotation_text="Safe Sweet Spot (0.8–1.3)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#34D399"),
        secondary_y=True,
    )
    fig.add_hline(
        y=1.50,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="High Injury Risk (>1.50)",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#F87171"),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        height=390,
        margin=dict(l=15, r=15, t=35, b=15),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Training Load", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="ACWR Ratio", range=[0.4, 2.0], secondary_y=True, showgrid=False)
    return fig


def build_volume_completion_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing Planned Sets vs Completed Sets over the past 7 sessions."""
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
        height=350,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Sets", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(title="Last 7 Workouts"),
    )
    return fig


def build_readiness_vs_fatigue_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot showing how more sleep and lower soreness improve readiness."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Sleep_Hours"],
            y=df["Readiness_Score"],
            mode="markers",
            name="Past Workouts",
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
                "Soreness: %{customdata[0]}/10<extra></extra>"
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
                name=f"Sleep Trend (+{slope:.1f}% Readiness per hr)",
                line=dict(color="#06B6D4", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.45)",
        height=350,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Hours Slept", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Readiness Score (%)", gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


def build_dual_radar_chart(
    athlete_row: pd.Series,
    compare_user: bool,
    user_traits: Dict[str, int],
) -> go.Figure:
    """6-trait Spider/Radar chart comparing the Pro Athlete vs User stats."""
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
                name=f"Your Stats (OVR {u_ovr})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=40, r=40, t=30, b=30),
        polar=dict(
            bgcolor="rgba(17, 24, 39, 0.65)",
            radialaxis=dict(visible=True, range=[40, 100], gridcolor="rgba(255, 255, 255, 0.10)"),
            angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.12)", tickfont=dict(size=12, color="#F9FAFB")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


# ==============================================================================
# 7. MODULAR PAGE RENDERERS (FOR CLEAN SIDEBAR NAVIGATION OR TABS)
# ==============================================================================
def render_daily_training_view(
    selected_sport: str,
    selected_day_routine: str,
    telemetry: Dict[str, object],
    muscle_focus: str,
    routing_mode: str,
    groq_api_key: str,
    groq_model: str,
    openrouter_api_key: str,
    openrouter_model: str,
    active_athlete: pd.Series,
    sleep_hours: float,
    soreness_level: int,
    energy_level: int,
    planned_duration: int,
    planned_rpe: int,
) -> Tuple[int, int]:
    """Renders the friendly Daily Workout & AI Coach page."""
    schedule_df = build_adaptive_schedule(
        sport=selected_sport,
        day_key=selected_day_routine,
        metrics=telemetry,
        muscle_focus=muscle_focus,
    )
    total_base_sets = int(schedule_df["Standard_Sets"].sum())
    total_adj_sets = int(schedule_df["Recommended_Sets"].sum())

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
        <div class="coach-box">
            <div class="coach-title">
                <span>🤖 Personal AI Coach Advice</span>
                <span class="pill pill-green">{engine_badge}</span>
            </div>
            <div class="coach-message">{coach_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.75, 1.0], gap="large")

    with col_left:
        st.markdown(f"### 📋 Today's Workout Plan — `{selected_day_routine}`")
        for idx, row in schedule_df.iterrows():
            st.markdown(
                f"""
                <div class="workout-step">
                    <div>
                        <div style="font-weight: 700; font-size: 1.02rem; color: #FFFFFF;">
                            {idx + 1}. {row['Drill_Name']}
                        </div>
                        <div style="font-size: 0.84rem; color: #9CA3AF; margin-top: 3px;">
                            🎯 {row['Coaching_Focus']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="pill pill-green">{row['Recommended_Sets']} Sets ({row['Adjustment']})</span>
                        <span class="pill pill-blue" style="margin-left: 6px;">Rest: {row['Rest_Between_Sets']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("📊 View Detailed Schedule Table", expanded=False):
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### ⚡ Quick Actions & Breakdown")
        st.markdown(
            f"""
            <div class="friendly-card">
                <div class="card-label">Why Your Workout Was Adjusted</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.35rem 0;">
                    {total_base_sets} Standard Sets ➔ <span style="color: #10B981;">{total_adj_sets} Today's Sets</span>
                </div>
                <div style="font-size: 0.86rem; color: #D1D5DB; line-height: 1.65; margin-top: 0.5rem;">
                    • <b>Recommended Pace:</b> {telemetry['intensity_tag']}<br>
                    • <b>Sleep Recovery Score:</b> {telemetry['sleep_score']}%<br>
                    • <b>Muscle Freshness:</b> {telemetry['soreness_score']}% ({muscle_focus})<br>
                    • <b>Energy Level:</b> {telemetry['energy_score']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("✅ Complete & Log Today's Workout in 1-Click", use_container_width=True, type="primary"):
            new_entry = pd.DataFrame(
                [
                    {
                        "Date": pd.to_datetime(datetime.now().date()),
                        "Sport": selected_sport,
                        "Duration_Min": int(planned_duration),
                        "RPE": int(planned_rpe),
                        "Daily_Workload": int(planned_duration * planned_rpe),
                        "Sleep_Hours": float(sleep_hours),
                        "Soreness": int(soreness_level),
                        "Energy": int(energy_level),
                        "Readiness_Score": float(telemetry["readiness_index"]),
                        "Planned_Sets": int(total_adj_sets),
                        "Completed_Sets": int(total_adj_sets),
                    }
                ]
            )
            updated_df = pd.concat([st.session_state.training_log, new_entry], ignore_index=True)
            st.session_state.training_log = recompute_rolling_metrics(updated_df)
            st.success("🎉 Awesome job! Today's workout has been saved to your Progress Charts.")

        with st.expander(f"📅 See Full 5-Day {selected_sport} Weekly Plan", expanded=False):
            for d_name, d_list in SPORT_WEEKLY_ROUTINES[selected_sport].items():
                drills_joined = ", ".join([str(d["Drill"]) for d in d_list])
                st.markdown(f"**{d_name}**  \n<span style='color:#9CA3AF;font-size:0.84rem;'>{drills_joined}</span>", unsafe_allow_html=True)

    return total_base_sets, total_adj_sets


def render_progress_charts_view(
    df_log: pd.DataFrame,
    sports_list: List[str],
    selected_sport: str,
    planned_duration: int,
    planned_rpe: int,
    total_adj_sets: int,
    sleep_hours: float,
    soreness_level: int,
    energy_level: int,
    telemetry: Dict[str, object],
) -> None:
    """Renders the 28-day performance charts and custom session logger."""
    st.markdown("### 📊 28-Day Training Load & Injury Safety Zone (ACWR)")
    st.caption("Keep the yellow diamond line inside the green shaded Sweet Spot (0.8–1.3) to build fitness safely.")
    st.plotly_chart(build_acwr_progression_chart(df_log), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🏋️ Sets Planned vs. Completed (Last 7 Workouts)")
        st.plotly_chart(build_volume_completion_chart(df_log), use_container_width=True)
    with c2:
        st.markdown("#### 🛌 How Sleep Affects Your Daily Readiness")
        st.plotly_chart(build_readiness_vs_fatigue_scatter(df_log), use_container_width=True)

    st.markdown("---")
    form_col, table_col = st.columns([1.05, 1.65], gap="large")

    with form_col:
        st.markdown("#### 📝 Log a Custom Workout Session")
        with st.form("custom_log_form", clear_on_submit=False):
            log_date_input = st.date_input("Date", value=datetime.now().date())
            log_sport_input = st.selectbox("Sport", sports_list, index=sports_list.index(selected_sport))
            f1, f2 = st.columns(2)
            with f1:
                log_duration = st.number_input("Minutes Trained", min_value=15, max_value=240, value=int(planned_duration), step=5)
                log_planned_sets = st.number_input("Planned Sets", min_value=1, max_value=40, value=int(total_adj_sets))
            with f2:
                log_rpe = st.number_input("Effort (1=Easy, 10=Max)", min_value=1, max_value=10, value=int(planned_rpe))
                log_completed_sets = st.number_input("Completed Sets", min_value=1, max_value=40, value=int(total_adj_sets))

            if st.form_submit_button("💾 Save Workout to History", use_container_width=True):
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
                st.success("Workout logged! All charts have been updated.")
                st.rerun()

    with table_col:
        st.markdown("#### 🗄️ Recent Workout History (Latest 10 Sessions)")
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


def render_pro_scouting_view(
    athletes_df: pd.DataFrame,
    sports_list: List[str],
    selected_sport: str,
    df_log: pd.DataFrame,
    telemetry: Dict[str, object],
    muscle_focus: str,
    soreness_level: int,
) -> None:
    """Renders the Pro Scouting dropdowns, 3D rotatable avatar, and dual radar chart."""
    st.markdown("### 🌟 Compare Yourself with World-Class Pro Athletes")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.3, 1.3, 1.1])

    with f_col1:
        scout_sport = st.selectbox("1. Choose Sport", sports_list, index=sports_list.index(selected_sport), key="scout_sport")
    scout_sport_df = athletes_df[athletes_df["Sport"] == scout_sport]

    with f_col2:
        scout_leagues = scout_sport_df["Team_League"].unique().tolist()
        scout_league = st.selectbox("2. Choose League / Team", ["All Leagues"] + scout_leagues, index=0, key="scout_league")
    if scout_league != "All Leagues":
        scout_athlete_df = scout_sport_df[scout_sport_df["Team_League"] == scout_league]
    else:
        scout_athlete_df = scout_sport_df

    with f_col3:
        scout_athlete_name = st.selectbox("3. Choose Pro Athlete", scout_athlete_df["Athlete"].tolist(), index=0, key="scout_athlete")
    scout_row = scout_athlete_df[scout_athlete_df["Athlete"] == scout_athlete_name].iloc[0]

    with f_col4:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        compare_user_toggle = st.toggle("Overlay My Stats", value=True)

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
        with st.expander("🎚️ Adjust Your Personal Player Ratings (40–99)", expanded=False):
            u1, u2, u3, u4, u5, u6 = st.columns(6)
            user_traits["Pace"] = u1.slider("Pace", 40, 99, user_traits["Pace"])
            user_traits["Endurance"] = u2.slider("Endurance", 40, 99, user_traits["Endurance"])
            user_traits["Agility"] = u3.slider("Agility", 40, 99, user_traits["Agility"])
            user_traits["Power"] = u4.slider("Power", 40, 99, user_traits["Power"])
            user_traits["Tactical IQ"] = u5.slider("Tactical IQ", 40, 99, user_traits["Tactical IQ"])
            user_traits["Recovery"] = u6.slider("Recovery", 40, 99, user_traits["Recovery"])

    avatar_col, radar_col = st.columns([1.15, 1.0], gap="large")

    with avatar_col:
        st.markdown(
            f"#### 🧍 Interactive 3D Athlete Model — `{scout_row['Athlete']}` "
            f"<span class='pill pill-blue' style='margin-left:8px;'>Click & Drag to Rotate 360°</span>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_3d_biomechanical_avatar(
                athlete_row=scout_row,
                muscle_focus=muscle_focus,
                soreness_level=soreness_level,
                readiness_index=float(telemetry["readiness_index"]),
            ),
            use_container_width=True,
        )

    with radar_col:
        st.markdown(f"#### 🕸️ 6-Trait Skill Radar — `{scout_row['Athlete']}`")
        st.plotly_chart(
            build_dual_radar_chart(
                athlete_row=scout_row,
                compare_user=compare_user_toggle,
                user_traits=user_traits,
            ),
            use_container_width=True,
        )

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
            <div class="friendly-card" style="border-left: 5px solid {scout_row['Jersey_Color']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="card-label">{scout_row['Sport']} · {scout_row['Team_League']}</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">
                            {scout_row['Athlete']} <span style="color:{scout_row['Jersey_Color']};">({scout_row['OVR']} OVR)</span>
                        </div>
                    </div>
                    <span class="pill pill-green">Playstyle Match: {similarity_pct}%</span>
                </div>
                <div style="font-size: 0.86rem; color: #D1D5DB; margin-top: 0.5rem;">
                    <b>Position:</b> {scout_row['Position']}<br>
                    <b>Signature Style:</b> {scout_row['Signature_Style']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# 8. MAIN APPLICATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    if "athletes_df" not in st.session_state:
        st.session_state.athletes_df = get_pro_athletes_db()

    if "training_log" not in st.session_state:
        st.session_state.training_log = generate_28_day_seed_log()

    athletes_df: pd.DataFrame = st.session_state.athletes_df
    df_log: pd.DataFrame = st.session_state.training_log
    sports_list = athletes_df["Sport"].unique().tolist()

    # --------------------------------------------------------------------------
    # CLEAN, USER-FRIENDLY SIDEBAR WITH NAVIGATION MENU & HIDDEN API KEYS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## ⚡ AERO-KINETIX AI")
        st.caption("Smart Conditioning & Pro Scouting")

        # 1. SIDEBAR NAVIGATION MENU
        st.markdown("### 🧭 Menu")
        nav_selection = st.radio(
            "Navigate to:",
            [
                "🏋️ Today's Workout & AI Coach",
                "📈 Progress & History Charts",
                "🌟 Pro Scouting & 3D Avatar",
                "🗂️ All-in-One Tabbed View",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 2. SPORT & ATHLETE SELECTION
        st.markdown("### 🏆 Your Sport & Role Model")
        selected_sport = st.selectbox("Sport", sports_list, index=0)
        sport_subset = athletes_df[athletes_df["Sport"] == selected_sport]
        selected_athlete_name = st.selectbox("Pro Inspiration", sport_subset["Athlete"].tolist(), index=0)
        active_athlete = sport_subset[sport_subset["Athlete"] == selected_athlete_name].iloc[0]

        day_options = list(SPORT_WEEKLY_ROUTINES[selected_sport].keys())
        selected_day_routine = st.selectbox("Workout Day", day_options, index=0)

        st.markdown("---")

        # 3. SIMPLE DAILY BODY CHECK-IN (3 EASY SLIDERS)
        st.markdown("### ❤️ Daily Body Check-In")
        sleep_hours = st.slider("🛌 Hours Slept Last Night", min_value=4.0, max_value=11.0, value=7.8, step=0.5)
        energy_level = st.slider("⚡ Energy & Motivation (1–10)", min_value=1, max_value=10, value=8)
        soreness_level = st.slider("🩹 Muscle Soreness (1=Fresh, 10=Sore)", min_value=1, max_value=10, value=3)
        muscle_focus = st.selectbox(
            "Area Feeling Most Sore",
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

        # 4. OPTIONAL WORKOUT DURATION / EFFORT (COLLAPSED FOR SIMPLICITY)
        with st.expander("⏱️ Adjust Workout Duration & Effort", expanded=False):
            planned_duration = st.slider("Duration (Minutes)", min_value=30, max_value=150, value=75, step=5)
            planned_rpe = st.slider("Target Effort RPE (1–10)", min_value=1, max_value=10, value=7)
        planned_workload_today = planned_duration * planned_rpe

        # 5. HIDDEN API KEY VAULT (COLLAPSED & MASKED BY DEFAULT)
        with st.expander("🔐 AI Model & API Keys (Hidden)", expanded=False):
            st.caption("Keys are masked & hidden. Paste Groq, OpenRouter, both, or leave empty for Offline Coach.")
            routing_mode = st.selectbox(
                "AI Connection Mode",
                [
                    "Auto-Detect (Uses Whichever Key You Paste)",
                    "Prefer Groq API",
                    "Prefer OpenRouter API",
                    "Offline Smart Coach (No API Key Needed)",
                ],
                index=0,
            )

            groq_api_key = st.text_input(
                "Groq API Key (Hidden)",
                value=get_secret_api_key("GROQ_API_KEY"),
                type="password",
                placeholder="gsk_...",
            )
            groq_model = st.selectbox(
                "Groq Model (Includes OpenAI & Llama)",
                [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "llama3-8b-8192",
                ],
                index=0,
            )

            openrouter_api_key = st.text_input(
                "OpenRouter API Key (Hidden)",
                value=get_secret_api_key("OPENROUTER_API_KEY"),
                type="password",
                placeholder="sk-or-...",
            )
            openrouter_model = st.selectbox(
                "OpenRouter Model",
                [
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "meta-llama/llama-3.3-70b-instruct",
                    "meta-llama/llama-3.1-8b-instruct",
                ],
                index=0,
            )

            g_stat = "🟢 Key Saved" if groq_api_key.strip() else "⚪ Not Set"
            or_stat = "🟢 Key Saved" if openrouter_api_key.strip() else "⚪ Not Set"
            st.caption(f"Status — Groq: {g_stat} | OpenRouter: {or_stat}")

    # Inject CSS synced to the active pro athlete's jersey color
    inject_custom_css(str(active_athlete["Jersey_Color"]))

    # Calculate Readiness & ACWR
    telemetry = compute_readiness_and_acwr(
        sleep_hours=sleep_hours,
        soreness=soreness_level,
        energy=energy_level,
        df_log=df_log,
        planned_workload_today=planned_workload_today,
    )

    # --------------------------------------------------------------------------
    # FRIENDLY TOP WELCOME HEADER
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="welcome-banner">
            <div>
                <h1 class="welcome-title">⚡ {selected_sport} Conditioning & Scouting Hub</h1>
                <div class="welcome-sub">
                    Role Model: <b>{active_athlete['Athlete']}</b> ({active_athlete['Position']}) &nbsp;•&nbsp;
                    Protecting: <b>{muscle_focus}</b>
                </div>
            </div>
            <div>
                <span class="pill {telemetry['readiness_pill']}">{telemetry['readiness_status']} ({telemetry['readiness_index']}%)</span>
                <span class="pill {telemetry['acwr_pill']}" style="margin-left: 8px;">{telemetry['acwr_zone']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 3 EASY-TO-READ SUMMARY CARDS AT TOP
    # --------------------------------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="friendly-card">
                <div class="card-label">1. YOUR DAILY READINESS</div>
                <div class="card-big-value" style="color: {telemetry['readiness_color']};">
                    {telemetry['readiness_index']}% — {telemetry['readiness_status']}
                </div>
                <div class="progress-track">
                    <div style="width: {telemetry['readiness_index']}%; height: 100%; background: {telemetry['readiness_color']};"></div>
                </div>
                <div class="card-helper">Based on {sleep_hours}h sleep, {energy_level}/10 energy & {soreness_level}/10 soreness.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="friendly-card">
                <div class="card-label">2. INJURY RISK & WORKLOAD (ACWR)</div>
                <div class="card-big-value" style="color: #06B6D4;">
                    {telemetry['acwr']:.2f} — {telemetry['acwr_zone']}
                </div>
                <div class="card-helper">{telemetry['acwr_explain']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        mult_pct = int(round(float(telemetry["set_multiplier"]) * 100))
        st.markdown(
            f"""
            <div class="friendly-card">
                <div class="card-label">3. TODAY'S WORKOUT PACE</div>
                <div class="card-big-value" style="color: #FBBF24;">
                    {mult_pct}% Volume
                </div>
                <div class="card-helper">{telemetry['intensity_tag']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Compute default adjusted sets for session logging
    temp_sched = build_adaptive_schedule(selected_sport, selected_day_routine, telemetry, muscle_focus)
    default_adj_sets = int(temp_sched["Recommended_Sets"].sum())

    # --------------------------------------------------------------------------
    # RENDER SELECTED VIEW FROM SIDEBAR MENU
    # --------------------------------------------------------------------------
    if nav_selection == "🏋️ Today's Workout & AI Coach":
        render_daily_training_view(
            selected_sport,
            selected_day_routine,
            telemetry,
            muscle_focus,
            routing_mode,
            groq_api_key,
            groq_model,
            openrouter_api_key,
            openrouter_model,
            active_athlete,
            sleep_hours,
            soreness_level,
            energy_level,
            planned_duration,
            planned_rpe,
        )

    elif nav_selection == "📈 Progress & History Charts":
        render_progress_charts_view(
            df_log,
            sports_list,
            selected_sport,
            planned_duration,
            planned_rpe,
            default_adj_sets,
            sleep_hours,
            soreness_level,
            energy_level,
            telemetry,
        )

    elif nav_selection == "🌟 Pro Scouting & 3D Avatar":
        render_pro_scouting_view(
            athletes_df,
            sports_list,
            selected_sport,
            df_log,
            telemetry,
            muscle_focus,
            soreness_level,
        )

    else:
        # All-in-One Tabbed View
        tab1, tab2, tab3 = st.tabs(
            [
                "🏋️ Daily Adaptive Training & Coach",
                "📈 Performance Trends & History Graphs",
                "🌟 Pro Scouting & 3D Avatar Studio",
            ]
        )
        with tab1:
            render_daily_training_view(
                selected_sport,
                selected_day_routine,
                telemetry,
                muscle_focus,
                routing_mode,
                groq_api_key,
                groq_model,
                openrouter_api_key,
                openrouter_model,
                active_athlete,
                sleep_hours,
                soreness_level,
                energy_level,
                planned_duration,
                planned_rpe,
            )
        with tab2:
            render_progress_charts_view(
                df_log,
                sports_list,
                selected_sport,
                planned_duration,
                planned_rpe,
                default_adj_sets,
                sleep_hours,
                soreness_level,
                energy_level,
                telemetry,
            )
        with tab3:
            render_pro_scouting_view(
                athletes_df,
                sports_list,
                selected_sport,
                df_log,
                telemetry,
                muscle_focus,
                soreness_level,
            )


if __name__ == "__main__":
    main()
