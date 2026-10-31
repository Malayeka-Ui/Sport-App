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
    page_title="AERO-KINETIX AI | Smart Sports Coach & 3D Mannequin Lab",
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

        .progress-track {{
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            overflow: hidden;
            margin-top: 8px;
        }}

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
# 2. INTERNATIONAL & LEAGUE PRO ATHLETE DATABASE (80+ WORLD-CLASS PLAYERS)
# ==============================================================================
@st.cache_data
def get_pro_athletes_db() -> pd.DataFrame:
    """
    Returns an extensive roster of 80+ international & league-level athletes across
    Badminton, Football, Basketball, and Cricket.
    Tuple schema:
    (Sport, Team_League, Athlete, Position, Pace, Endurance, Agility, Power, Tactical_IQ, Recovery, OVR, Jersey_Color, Signature_Style)
    """
    raw_roster = [
        # ---------------- BADMINTON (BWF INTERNATIONAL & NATIONAL TEAMS) ----------------
        ("Badminton", "BWF World Tour (Denmark)", "Viktor Axelsen", "Men's Singles Attacking Baseline", 91, 94, 96, 95, 98, 93, 95, "#EF4444", "Steep overhead jump smashes & commanding court reach"),
        ("Badminton", "BWF World Tour (Denmark)", "Anders Antonsen", "Men's Singles All-Court Tactician", 90, 93, 94, 91, 96, 92, 93, "#DC2626", "Deceptive net exchanges & counter-attacking variations"),
        ("Badminton", "BWF Legends (China)", "Lin Dan", "Men's Singles Counter-Attacking", 95, 97, 99, 93, 99, 95, 97, "#F59E0B", "Explosive net lunges, wrist flicks & sixth-sense anticipation"),
        ("Badminton", "BWF World Tour (China)", "Shi Yuqi", "Men's Singles Offensive Specialist", 93, 92, 95, 96, 95, 91, 94, "#EF4444", "Whip-like cross-court smashes & razor-sharp net kills"),
        ("Badminton", "BWF World Tour (China)", "Chen Yufei", "Women's Singles Defensive Anchor", 92, 97, 96, 89, 97, 96, 95, "#F97316", "Unbreakable baseline retrieval & rally patience"),
        ("Badminton", "BWF World Tour (China)", "Liang Weikeng", "Men's Doubles Rear-Court Power", 92, 90, 93, 99, 91, 90, 93, "#E11D48", "Thunderous back-court jump smashes & rotational speed"),
        ("Badminton", "BWF World Tour (South Korea)", "An Se-young", "Women's Singles Defensive Specialist", 94, 99, 98, 89, 97, 98, 96, "#3B82F6", "Relentless rally stamina, elastic split-steps & court coverage"),
        ("Badminton", "BWF World Tour (South Korea)", "Seo Seung-jae", "Men's/Mixed Doubles All-Rounder", 94, 96, 96, 93, 97, 95, 95, "#2563EB", "Left-handed angles, flat drive supremacy & multi-event stamina"),
        ("Badminton", "BWF Legends (Malaysia)", "Lee Chong Wei", "Men's Singles Speed & Counter", 99, 96, 99, 92, 96, 95, 96, "#FACC15", "Supersonic defensive recovery & mid-court jump smashes"),
        ("Badminton", "BWF World Tour (Malaysia)", "Lee Zii Jia", "Men's Singles Power Attacker", 93, 90, 94, 98, 90, 91, 93, "#EAB308", "Backhand whip smashes & explosive baseline power"),
        ("Badminton", "BWF World Tour (Malaysia)", "Aaron Chia", "Men's Doubles Front/Mid-Court", 93, 92, 96, 90, 95, 92, 93, "#F59E0B", "Flat-exchange reflexes & defensive block placement"),
        ("Badminton", "BWF World Tour (India)", "Lakshya Sen", "Men's Singles High-Pace Retriever", 95, 95, 96, 90, 92, 94, 93, "#0EA5E9", "Diving defensive saves & rapid net-to-baseline transitions"),
        ("Badminton", "BWF World Tour (India)", "PV Sindhu", "Women's Singles Attacking Tall", 90, 92, 91, 95, 93, 92, 92, "#38BDF8", "High-point steep cross smashes & big-tournament mentality"),
        ("Badminton", "BWF World Tour (India)", "Satwiksairaj Rankireddy", "Men's Doubles Power Smasher", 91, 91, 92, 99, 92, 91, 93, "#0284C7", "World-record 565 kph smash velocity & rear-court dominance"),
        ("Badminton", "BWF World Tour (India)", "Chirag Shetty", "Men's Doubles Net Interceptor", 95, 93, 97, 90, 95, 93, 94, "#06B6D4", "Lightning front-court interceptions & aggressive net pressure"),
        ("Badminton", "BWF World Tour (Indonesia)", "Anthony Sinisuka Ginting", "Men's Singles Explosive Pace", 97, 90, 98, 93, 91, 90, 93, "#EF4444", "Deceptive net slices & ultra-fast attacking footwork"),
        ("Badminton", "BWF World Tour (Indonesia)", "Jonatan Christie", "Men's Singles Rally & Counter", 93, 95, 94, 92, 94, 94, 94, "#DC2626", "Physical baseline attrition & clinical straight smashes"),
        ("Badminton", "BWF World Tour (Japan)", "Kodai Naraoka", "Men's Singles Marathon Defender", 92, 99, 97, 86, 95, 98, 94, "#EC4899", "90-minute match endurance & low-error defensive walls"),
        ("Badminton", "BWF World Tour (Japan)", "Akane Yamaguchi", "Women's Singles Explosive Retriever", 95, 97, 98, 91, 95, 96, 95, "#DB2777", "Compact footwork, deceptive clears & springy jump transitions"),
        ("Badminton", "BWF World Tour (Chinese Taipei)", "Tai Tzu-ying", "Women's Singles Deception Artist", 93, 91, 98, 90, 99, 92, 95, "#8B5CF6", "Disguised drop shots, late wrist holds & court geometry"),
        ("Badminton", "BWF World Tour (Thailand)", "Kunlavut Vitidsarn", "Men's Singles World Champion Tactician", 93, 96, 96, 91, 97, 95, 95, "#14B8A6", "Counter-punching accuracy & elite defensive absorption"),

        # ---------------- FOOTBALL (UEFA CHAMPIONS LEAGUE & INTERNATIONAL) ----------------
        ("Football", "La Liga (Real Madrid / France)", "Kylian Mbappé", "Inside Forward / Striker", 99, 89, 95, 88, 92, 91, 94, "#06B6D4", "Supersonic transition sprints & clinical half-space finishing"),
        ("Football", "La Liga (Real Madrid / England)", "Jude Bellingham", "Box-to-Box Attacking Midfielder", 88, 97, 91, 90, 95, 94, 93, "#A855F7", "All-pitch stamina, high-volume pressing & late box arrivals"),
        ("Football", "La Liga (Real Madrid / Brazil)", "Vinícius Júnior", "Left Winger", 98, 91, 98, 82, 90, 92, 93, "#FACC15", "Relentless 1v1 touchline dribbling & explosive acceleration"),
        ("Football", "La Liga (Real Madrid / Uruguay)", "Federico Valverde", "Box-to-Box Engine Midfielder", 94, 99, 89, 93, 93, 97, 94, "#38BDF8", "Lung-bursting recovery runs & 30-yard thunderbolt strikes"),
        ("Football", "La Liga (FC Barcelona / Spain)", "Lamine Yamal", "Right Wing Playmaker", 93, 88, 97, 78, 95, 90, 91, "#EF4444", "Tight-space ball mastery, outside-boot crosses & curling finishes"),
        ("Football", "La Liga (FC Barcelona / Spain)", "Pedri", "Central Tempo Controller", 84, 93, 95, 76, 98, 89, 91, "#2563EB", "360-degree press resistance & defence-splitting through balls"),
        ("Football", "La Liga (FC Barcelona / Poland)", "Robert Lewandowski", "Complete No. 9 Striker", 83, 88, 86, 94, 97, 93, 92, "#DC2626", "Penalty-box positioning, first-time volleys & hold-up play"),
        ("Football", "Premier League (Man City / Norway)", "Erling Haaland", "Target No. 9 Striker", 94, 88, 85, 99, 94, 93, 94, "#38BDF8", "Explosive box power, aerial vertical leaps & physical shielding"),
        ("Football", "Premier League (Man City / Spain)", "Rodri", "Defensive Midfield Anchor", 81, 97, 86, 92, 99, 95, 95, "#0EA5E9", "Metronomic passing tempo, tactical interceptions & edge-of-box goals"),
        ("Football", "Premier League (Man City / Belgium)", "Kevin De Bruyne", "Attacking Midfield Architect", 83, 89, 87, 91, 99, 88, 93, "#06B6D4", "Whipped early crosses, visionary passing lanes & two-footed strikes"),
        ("Football", "Premier League (Arsenal / England)", "Bukayo Saka", "Right Inverted Winger", 91, 95, 94, 87, 93, 95, 92, "#EF4444", "Low-center-of-gravity shielding, cut-inside curls & durability"),
        ("Football", "Premier League (Arsenal / Norway)", "Martin Ødegaard", "Right-Half-Space Playmaker", 84, 96, 93, 80, 97, 93, 92, "#E11D48", "High-intensity pressing trigger & disguised reverse passes"),
        ("Football", "Premier League (Arsenal / England)", "Declan Rice", "Ball-Winning Midfielder", 87, 98, 86, 93, 94, 96, 92, "#DC2626", "Long-stride ball carrying & recovery tackle timing"),
        ("Football", "Premier League (Liverpool / Egypt)", "Mohamed Salah", "Inside Right Forward", 93, 93, 94, 89, 95, 96, 94, "#EF4444", "explosive inside channels, core strength & curlers to the far post"),
        ("Football", "Premier League (Liverpool / Netherlands)", "Virgil van Dijk", "Commanding Center-Back", 88, 90, 82, 97, 96, 93, 93, "#F97316", "Aerial dominance, 60-yard diagonal switches & 1v1 composure"),
        ("Football", "Bundesliga (Bayern Munich / England)", "Harry Kane", "Deep-Lying Complete Striker", 80, 91, 83, 94, 98, 92, 93, "#DC2626", "Playmaking No. 9 drops, lethal penalty-box finishing & long passing"),
        ("Football", "Bundesliga (Bayern Munich / Germany)", "Jamal Musiala", "Slalom Attacking Midfielder", 91, 90, 99, 79, 93, 90, 92, "#E11D48", "Elastic ankle agility through crowded midfield blocks"),
        ("Football", "Bundesliga (Leverkusen / Germany)", "Florian Wirtz", "Creative No. 10", 89, 94, 95, 81, 96, 92, 92, "#EF4444", "One-touch combination play & half-space incisiveness"),
        ("Football", "Serie A (Inter Milan / Argentina)", "Lautaro Martínez", "Pressing Striker", 88, 93, 91, 90, 93, 92, 91, "#2563EB", "Aggressive front-foot pressing & near-post darting finishes"),
        ("Football", "MLS (Inter Miami / Argentina)", "Lionel Messi", "Free-Role Playmaker / False 9", 85, 84, 97, 83, 99, 89, 94, "#EC4899", "Unmatched spatial scanning, low-gravity slaloms & pinpoint free kicks"),
        ("Football", "Saudi Pro League (Al-Nassr / Portugal)", "Cristiano Ronaldo", "Penalty-Box Poacher", 87, 92, 87, 95, 96, 97, 92, "#FACC15", "2.9m vertical header hangs, two-footed power & elite conditioning"),

        # ---------------- BASKETBALL (NBA & FIBA INTERNATIONAL) ----------------
        ("Basketball", "NBA (Los Angeles Lakers / USA)", "LeBron James", "Point Forward (SF/PF)", 91, 95, 90, 98, 99, 99, 96, "#FACC15", "Downhill freight-train rim drives, full-court IQ & peak longevity"),
        ("Basketball", "NBA (Los Angeles Lakers / USA)", "Anthony Davis", "Two-Way Center / PF", 85, 90, 88, 95, 94, 89, 93, "#EAB308", " Paint-protecting shot blocks, lob finishing & mid-range face-ups"),
        ("Basketball", "NBA (Golden State Warriors / USA)", "Stephen Curry", "Combo Guard (PG)", 92, 99, 97, 82, 98, 94, 95, "#2563EB", "Constant off-ball marathon movement & sub-0.4s 30-foot release"),
        ("Basketball", "NBA (Denver Nuggets / Serbia)", "Nikola Jokić", "Point Center (C)", 78, 96, 86, 96, 99, 96, 97, "#F97316", "Full-court outlet vision, Sombor shuffle touch & rebound control"),
        ("Basketball", "NBA (Dallas Mavericks / Slovenia)", "Luka Dončić", "Heliocentric Point Guard", 84, 92, 92, 93, 99, 91, 96, "#0EA5E9", "Step-back 3PT separation, deceleration pace & skip-pass mastery"),
        ("Basketball", "NBA (Milwaukee Bucks / Greece)", "Giannis Antetokounmpo", "Slashing Power Forward", 96, 96, 93, 99, 92, 96, 96, "#10B981", "3-dribble coast-to-coast Euro-steps & weak-side rim protection"),
        ("Basketball", "NBA (Boston Celtics / USA)", "Jayson Tatum", "Two-Way Wing (SF/PF)", 89, 95, 91, 91, 94, 95, 94, "#22C55E", "Sidestep 3PT creation, switchable wing defense & durability"),
        ("Basketball", "NBA (Boston Celtics / USA)", "Jaylen Brown", "Slasher Two-Guard (SG/SF)", 94, 93, 93, 92, 89, 93, 92, "#16A34A", "Explosive transition dunks, mid-post footwork & on-ball pressure"),
        ("Basketball", "NBA (OKC Thunder / Canada)", "Shai Gilgeous-Alexander", "Slashing Lead Guard (PG)", 93, 95, 98, 86, 96, 94, 96, "#38BDF8", "Unstoppable start-stop deceleration drives & mid-range efficiency"),
        ("Basketball", "NBA (San Antonio Spurs / France)", "Victor Wembanyama", "Unicorn Rim Protector (C/PF)", 88, 90, 91, 89, 93, 91, 94, "#9CA3AF", "8-foot wingspan shot-blocking, step-back 3s & fluid ball handling"),
        ("Basketball", "NBA (Minnesota Timberwolves / USA)", "Anthony Edwards", "Explosive Scoring Guard (SG)", 97, 94, 96, 95, 90, 94, 94, "#3B82F6", "Hang-time poster dunks, point-of-attack defense & pull-up shooting"),
        ("Basketball", "NBA (Phoenix Suns / USA)", "Kevin Durant", "7-Foot Perimeter Scorer (PF/SF)", 87, 92, 91, 87, 97, 91, 94, "#F59E0B", "Unblockable high-release mid-range jumpers & effortless 3-level scoring"),
        ("Basketball", "NBA (Phoenix Suns / USA)", "Devin Booker", "Three-Level Scoring Guard (SG)", 89, 93, 92, 86, 94, 92, 92, "#F97316", "Pin-down curl footwork, mid-range pull-ups & playmaking"),
        ("Basketball", "NBA (Philadelphia 76ers / USA)", "Joel Embiid", "Dominant Scoring Center (C)", 80, 87, 85, 99, 94, 86, 94, "#2563EB", "Foul-drawing post footwork, nail jumper touch & rim deterrence"),
        ("Basketball", "NBA (New York Knicks / USA)", "Jalen Brunson", "Crafty Lead Guard (PG)", 88, 95, 95, 87, 96, 93, 93, "#F97316", "Pivot-foot paint mastery, clutch shot-making & low turnovers"),
        ("Basketball", "NBA (Sacramento Kings / USA)", "De'Aaron Fox", "Speed Point Guard (PG)", 99, 93, 97, 84, 91, 92, 91, "#A855F7", "End-to-end blur speed, clutch 4th-quarter floaters & steals"),

        # ---------------- CRICKET (ICC INTERNATIONAL & T20 LEAGUES: PSL / IPL / BBL) ----------------
        ("Cricket", "ICC / PSL (Pakistan / Peshawar Zalmi)", "Babar Azam", "Top-Order Batter (No. 3)", 87, 94, 93, 88, 96, 92, 93, "#10B981", "High-elbow biomechanics, late-contact cover drives & crease balance"),
        ("Cricket", "ICC / PSL (Pakistan / Lahore Qalandars)", "Shaheen Shah Afridi", "Left-Arm Fast Bowler", 96, 91, 89, 95, 92, 90, 93, "#14B8A6", "145+ kph late-inswinging yorkers & high-release front-leg brace"),
        ("Cricket", "ICC / PSL (Pakistan / Multan Sultans)", "Mohammad Rizwan", "Wicketkeeper Batter", 92, 98, 96, 87, 95, 96, 93, "#22C55E", "Tireless crouched glovework, leg-side sweeps & busy running"),
        ("Cricket", "ICC / PSL (Pakistan / Islamabad United)", "Naseem Shah", "Right-Arm Express Pace", 95, 90, 90, 92, 91, 89, 91, "#EF4444", "High-arm seam presentation, late outswing & fiery spells"),
        ("Cricket", "ICC / PSL (Pakistan / Peshawar Zalmi)", "Saim Ayub", "Aggressive Opening Batter", 90, 88, 94, 90, 89, 90, 89, "#FACC15", "No-look pick-up flicks, carrom-ball spin & powerplay intent"),
        ("Cricket", "ICC / PSL (Pakistan / Lahore Qalandars)", "Fakhar Zaman", "Explosive Left-Hand Opener", 89, 91, 88, 96, 89, 90, 90, "#10B981", "Clean slog-sweeps over cow corner & big-match century temperament"),
        ("Cricket", "ICC / IPL (India / RCB)", "Virat Kohli", "Top-Order Chase Master (No. 3)", 94, 98, 95, 91, 98, 96, 96, "#0EA5E9", "Explosive 2-run conversion sprints, wrists-driven flicks & mental grit"),
        ("Cricket", "ICC / IPL (India / Mumbai Indians)", "Jasprit Bumrah", "All-Phase Spearhead Fast Bowler", 95, 94, 93, 95, 99, 94, 96, "#2563EB", "Hyper-extension sling release, unplayable yorkers & deceptive slower balls"),
        ("Cricket", "ICC / IPL (India / Mumbai Indians)", "Rohit Sharma", "Opening Batter & Captain", 84, 90, 88, 97, 97, 90, 93, "#3B82F6", "Effortless front-foot pull shots & powerplay field manipulation"),
        ("Cricket", "ICC / IPL (India / Mumbai Indians)", "Suryakumar Yadav", "360-Degree T20 Batter", 91, 92, 97, 93, 96, 92, 94, "#06B6D4", "Fine-leg ramps, wrists-generated supla shots & extreme strike rates"),
        ("Cricket", "ICC / IPL (India / CSK)", "Ravindra Jadeja", "Three-Dimensional All-Rounder", 95, 96, 98, 90, 94, 96, 94, "#FACC15", "Bullet-arm direct hits, rapid left-arm orthodox overs & lower-order power"),
        ("Cricket", "ICC / BBL (Australia / Sunrisers)", "Pat Cummins", "Fast-Bowling Captain", 92, 96, 90, 94, 98, 95, 95, "#F59E0B", "Relentless top-of-off-stump seam accuracy & lower-order clutch hitting"),
        ("Cricket", "ICC / IPL (Australia / Sunrisers)", "Travis Head", " Ultra-Attacking Opener", 89, 91, 92, 97, 92, 91, 93, "#EAB308", "Fearless square-of-the-wicket slashing & finals match-winning hundreds"),
        ("Cricket", "ICC (Australia)", "Steve Smith", "Unorthodox Middle-Order Genius", 86, 97, 94, 86, 99, 95, 94, "#FACC15", "Back-and-across trigger shuffle, hand-eye wizardry & 8-hour concentration"),
        ("Cricket", "ICC / IPL (Australia / KKR)", "Mitchell Starc", "Left-Arm Express Swing", 96, 90, 88, 97, 91, 90, 93, "#10B981", "150 kph toe-crushing inswinging yorkers with the new ball"),
        ("Cricket", "ICC (England)", "Joe Root", "Classical Test & ODI Anchor", 87, 96, 93, 85, 98, 95, 94, "#EF4444", "Late dabs to third man, reverse-scoops & golden-arm off-spin"),
        ("Cricket", "ICC (England)", "Ben Stokes", "Match-Winning All-Rounder", 88, 95, 90, 97, 97, 92, 94, "#DC2626", "Miracle run-chases, bouncer-barrage spells & slip-cordon acrobatics"),
        ("Cricket", "ICC / IPL (England / Rajasthan)", "Jos Buttler", "Explosive White-Ball Opener", 91, 92, 94, 96, 95, 92, 94, "#EC4899", "Deep-crease bat speed, ramp shots & death-overs carnage"),
        ("Cricket", "ICC (New Zealand)", "Kane Williamson", "Technical No. 3 Batter", 86, 95, 92, 84, 99, 94, 93, "#14B8A6", "Under-the-eyes late steers, calm leadership & soft-hands placement"),
        ("Cricket", "ICC / IPL (Afghanistan / GT)", "Rashid Khan", "T20 Leg-Spin Superstar", 92, 94, 96, 91, 97, 95, 95, "#0EA5E9", "100 kph quick-arm googlies, economy-rate strangle & snake-shot sixes"),
        ("Cricket", "ICC / IPL (South Africa / SRH)", "Heinrich Klaasen", "Middle-Overs Spin Destroyer", 88, 91, 91, 98, 93, 91, 93, "#22C55E", "Back-foot pull-shot power against spin & 180+ strike-rate acceleration"),
    ]

    columns = [
        "Sport",
        "Team_League",
        "Athlete",
        "Position",
        "Pace",
        "Endurance",
        "Agility",
        "Power",
        "Tactical_IQ",
        "Recovery",
        "OVR",
        "Jersey_Color",
        "Signature_Style",
    ]
    return pd.DataFrame(raw_roster, columns=columns)


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
# 5. VOLUMETRIC 3D HUMAN MANNEQUIN ENGINE (SMOOTH SHADED SURFACES)
# ==============================================================================
def _create_mannequin_limb_surface(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    r0: Tuple[float, float],
    r1: Tuple[float, float],
    color_hex: str,
    label: str,
    bulge: float = 0.14,
    n_theta: int = 20,
    n_z: int = 10,
    opacity: float = 0.96,
) -> go.Surface:
    """
    Generates a smooth 3D sculpted anatomical limb or torso segment between 3D points p0 and p1
    with elliptical cross-sections (rx, ry) and a natural mid-segment muscle belly bulge.
    """
    p0_arr = np.array(p0, dtype=float)
    p1_arr = np.array(p1, dtype=float)
    v = p1_arr - p0_arr
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        v_hat = np.array([0.0, 0.0, 1.0])
    else:
        v_hat = v / length

    # Construct orthonormal local coordinate frame (e1: lateral, e2: anterior-posterior)
    ref = np.array([0.0, 1.0, 0.0]) if abs(v_hat[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(v_hat, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)
    e2 /= np.linalg.norm(e2)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    t = np.linspace(0.0, 1.0, n_z)
    theta_grid, t_grid = np.meshgrid(theta, t)

    # Muscle belly contour curve using sine arch
    contour = 1.0 + bulge * np.sin(np.pi * t_grid)
    rx = ((1.0 - t_grid) * r0[0] + t_grid * r1[0]) * contour
    ry = ((1.0 - t_grid) * r0[1] + t_grid * r1[1]) * contour

    center_x = p0_arr[0] + t_grid * v[0]
    center_y = p0_arr[1] + t_grid * v[1]
    center_z = p0_arr[2] + t_grid * v[2]

    x_surf = center_x + rx * np.cos(theta_grid) * e1[0] + ry * np.sin(theta_grid) * e2[0]
    y_surf = center_y + rx * np.cos(theta_grid) * e1[1] + ry * np.sin(theta_grid) * e2[1]
    z_surf = center_z + rx * np.cos(theta_grid) * e1[2] + ry * np.sin(theta_grid) * e2[2]

    return go.Surface(
        x=x_surf,
        y=y_surf,
        z=z_surf,
        surfacecolor=np.zeros_like(z_surf),
        colorscale=[[0, color_hex], [1, color_hex]],
        showscale=False,
        opacity=opacity,
        name=label,
        hovertemplate=f"<b>{label}</b><extra></extra>",
        lighting=dict(ambient=0.58, diffuse=0.78, specular=0.45, roughness=0.32, fresnel=0.25),
        lightposition=dict(x=120, y=220, z=280),
    )


def _create_mannequin_ellipsoid(
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
    color_hex: str,
    label: str,
    n_theta: int = 20,
    n_phi: int = 14,
    opacity: float = 0.98,
) -> go.Surface:
    """Generates a smooth 3D ellipsoid for the head, shoulder deltoids, knees, elbows, and hands."""
    cx, cy, cz = center
    rx, ry, rz = radii
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    phi = np.linspace(0.0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x_surf = cx + rx * np.sin(phi_grid) * np.cos(theta_grid)
    y_surf = cy + ry * np.sin(phi_grid) * np.sin(theta_grid)
    z_surf = cz + rz * np.cos(phi_grid)

    return go.Surface(
        x=x_surf,
        y=y_surf,
        z=z_surf,
        surfacecolor=np.zeros_like(z_surf),
        colorscale=[[0, color_hex], [1, color_hex]],
        showscale=False,
        opacity=opacity,
        name=label,
        hovertemplate=f"<b>{label}</b><extra></extra>",
        lighting=dict(ambient=0.58, diffuse=0.78, specular=0.45, roughness=0.32, fresnel=0.25),
        lightposition=dict(x=120, y=220, z=280),
    )


def build_3d_biomechanical_avatar(
    athlete_row: pd.Series,
    muscle_focus: str,
    soreness_level: int,
    readiness_index: float,
) -> go.Figure:
    """
    Constructs a full 3D Volumetric Human Mannequin with smooth anatomical body surfaces,
    athlete jersey colors, and dynamic muscle soreness heatmap highlighting.
    """
    jersey_hex = str(athlete_row["Jersey_Color"])
    mannequin_skin = "#CBD5E1"   # Sculpted metallic mannequin finish
    shorts_hex = "#1E293B"       # Athletic shorts slate
    shoe_hex = "#06B6D4"         # Cyan performance footwear

    # Determine soreness highlight color
    if soreness_level >= 7:
        sore_hex = "#EF4444"     # High soreness crimson
    elif soreness_level >= 4:
        sore_hex = "#F59E0B"     # Moderate soreness amber
    else:
        sore_hex = None

    def pick_color(region_tag: str, default_hex: str) -> str:
        if sore_hex is None:
            return default_hex
        if muscle_focus == "Full Body Systemic":
            return sore_hex
        if muscle_focus == region_tag:
            return sore_hex
        return default_hex

    fig = go.Figure()

    # 1. HEAD, CRANIUM & NECK
    fig.add_trace(
        _create_mannequin_ellipsoid(
            center=(0.0, 1.0, 173.5),
            radii=(8.8, 10.2, 12.2),
            color_hex=mannequin_skin,
            label=f"{athlete_row['Athlete']} — Cranium & Head (Readiness: {readiness_index}%)",
        )
    )
    # Athletic Visor / Eye Shield Strip on Face
    fig.add_trace(
        _create_mannequin_ellipsoid(
            center=(0.0, 9.6, 175.0),
            radii=(7.2, 2.2, 2.4),
            color_hex=jersey_hex,
            label=f"{athlete_row['Athlete']} — Pro Visor",
        )
    )
    # Muscular Neck
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(0.0, 0.0, 154.0),
            p1=(0.0, 0.5, 163.5),
            r0=(6.2, 5.8),
            r1=(5.4, 5.2),
            color_hex=mannequin_skin,
            label="Cervical Neck",
            bulge=0.05,
        )
    )

    # 2. UPPER TORSO (V-TAPER CHEST / JERSEY), ABDOMEN & PELVIS
    chest_color = pick_color("Lower Back / Core", jersey_hex)
    abdomen_color = pick_color("Lower Back / Core", jersey_hex)
    pelvis_color = pick_color("Lower Back / Core", shorts_hex)

    # Upper Chest & Lats (V-Taper)
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(0.0, 0.0, 126.0),
            p1=(0.0, 0.5, 154.0),
            r0=(14.8, 9.6),
            r1=(20.5, 11.8),
            color_hex=chest_color,
            label=f"Upper Torso / Jersey ({athlete_row['Team_League']})",
            bulge=0.10,
        )
    )
    # Waist / Core / Obliques
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(0.0, 0.0, 104.0),
            p1=(0.0, 0.0, 126.5),
            r0=(14.5, 9.5),
            r1=(14.8, 9.6),
            color_hex=abdomen_color,
            label=f"Abdominal Core & Lumbar (Focus: {muscle_focus})",
            bulge=0.06,
        )
    )
    # Pelvis / Hips (Athletic Shorts Base)
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(0.0, 0.0, 91.0),
            p1=(0.0, 0.0, 105.0),
            r0=(15.2, 10.2),
            r1=(15.0, 10.0),
            color_hex=pelvis_color,
            label="Pelvic Girdle & Glutes",
            bulge=0.08,
        )
    )

    # 3. SHOULDERS, ARMS, ELBOWS, FOREARMS & HANDS (ATHLETIC READY POSE)
    shoulder_color = pick_color("Shoulders / Rotator Cuff", jersey_hex)
    upper_arm_color = pick_color("Shoulders / Rotator Cuff", mannequin_skin)

    # Left Shoulder Deltoid Cap, Upper Arm, Elbow, Forearm, Hand
    fig.add_trace(_create_mannequin_ellipsoid((-22.5, 0.0, 150.0), (6.8, 6.2, 6.5), shoulder_color, "Left Deltoid / Shoulder"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(-22.5, 0.0, 150.0),
            p1=(-30.0, 6.0, 122.0),
            r0=(5.3, 5.0),
            r1=(4.3, 4.1),
            color_hex=upper_arm_color,
            label="Left Bicep / Tricep",
            bulge=0.18,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((-30.0, 6.0, 122.0), (4.2, 4.2, 4.2), mannequin_skin, "Left Elbow Joint"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(-30.0, 6.0, 122.0),
            p1=(-25.0, 22.0, 104.0),
            r0=(4.2, 4.0),
            r1=(3.1, 2.9),
            color_hex=mannequin_skin,
            label="Left Forearm",
            bulge=0.15,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((-24.0, 25.0, 101.0), (3.5, 4.2, 3.2), jersey_hex, "Left Hand / Wristband"))

    # Right Shoulder Deltoid Cap, Upper Arm, Elbow, Forearm, Hand (Slightly Elevated Action Stance)
    fig.add_trace(_create_mannequin_ellipsoid((22.5, 0.0, 150.0), (6.8, 6.2, 6.5), shoulder_color, "Right Deltoid / Shoulder"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(22.5, 0.0, 150.0),
            p1=(32.0, 8.0, 126.0),
            r0=(5.3, 5.0),
            r1=(4.3, 4.1),
            color_hex=upper_arm_color,
            label="Right Bicep / Tricep",
            bulge=0.18,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((32.0, 8.0, 126.0), (4.2, 4.2, 4.2), mannequin_skin, "Right Elbow Joint"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(32.0, 8.0, 126.0),
            p1=(26.0, 25.0, 138.0),
            r0=(4.2, 4.0),
            r1=(3.1, 2.9),
            color_hex=mannequin_skin,
            label="Right Forearm",
            bulge=0.15,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((25.0, 28.0, 141.0), (3.6, 4.2, 3.4), jersey_hex, "Right Hand / Action Grip"))

    # 4. THIGHS (QUADS / HAMSTRINGS), KNEES, CALVES & FEET
    thigh_color = pick_color("Quads / Hamstrings", shorts_hex if soreness_level < 4 else sore_hex)
    lower_thigh_color = pick_color("Quads / Hamstrings", mannequin_skin)
    knee_color = pick_color("Knees / Patella", mannequin_skin)
    calf_color = pick_color("Calves / Achilles", mannequin_skin)

    # Left Leg
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(-10.5, 0.0, 92.0),
            p1=(-13.5, 4.0, 52.0),
            r0=(8.4, 8.2),
            r1=(5.8, 5.6),
            color_hex=thigh_color if muscle_focus == "Quads / Hamstrings" and soreness_level >= 4 else lower_thigh_color,
            label="Left Quadriceps & Hamstrings",
            bulge=0.20,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((-13.5, 4.0, 52.0), (5.6, 5.8, 5.2), knee_color, "Left Knee / Patella"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(-13.5, 4.0, 52.0),
            p1=(-14.5, -1.0, 10.0),
            r0=(5.4, 5.4),
            r1=(3.4, 3.4),
            color_hex=calf_color,
            label="Left Calf (Gastrocnemius & Achilles)",
            bulge=0.22,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((-14.5, 5.5, 5.0), (4.6, 9.5, 4.2), shoe_hex, "Left Athletic Shoe"))

    # Right Leg
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(10.5, 0.0, 92.0),
            p1=(13.5, 5.0, 52.0),
            r0=(8.4, 8.2),
            r1=(5.8, 5.6),
            color_hex=thigh_color if muscle_focus == "Quads / Hamstrings" and soreness_level >= 4 else lower_thigh_color,
            label="Right Quadriceps & Hamstrings",
            bulge=0.20,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((13.5, 5.0, 52.0), (5.6, 5.8, 5.2), knee_color, "Right Knee / Patella"))
    fig.add_trace(
        _create_mannequin_limb_surface(
            p0=(13.5, 5.0, 52.0),
            p1=(14.5, 0.0, 10.0),
            r0=(5.4, 5.4),
            r1=(3.4, 3.4),
            color_hex=calf_color,
            label="Right Calf (Gastrocnemius & Achilles)",
            bulge=0.22,
        )
    )
    fig.add_trace(_create_mannequin_ellipsoid((14.5, 6.5, 5.0), (4.6, 9.5, 4.2), shoe_hex, "Right Athletic Shoe"))

    # 5. FORCE-PLATE TELEMETRY PLATFORM RINGS ON FLOOR
    theta_ring = np.linspace(0, 2 * np.pi, 50)
    for radius, width_val in [(28.0, 3), (42.0, 5)]:
        fig.add_trace(
            go.Scatter3d(
                x=radius * np.cos(theta_ring),
                y=radius * np.sin(theta_ring),
                z=np.zeros_like(theta_ring),
                mode="lines",
                line=dict(color=jersey_hex, width=width_val),
                name="Force Plate Platform",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=540,
        showlegend=False,
        scene=dict(
            dragmode="turntable",
            xaxis=dict(title="", range=[-65, 65], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(title="", range=[-65, 65], showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(title="Height (cm)", range=[0, 195], showgrid=True, gridcolor="rgba(6, 182, 212, 0.12)", zeroline=False, showbackground=True, backgroundcolor="rgba(11, 15, 25, 0.45)"),
            camera=dict(eye=dict(x=1.35, y=1.55, z=0.55), center=dict(x=0, y=0, z=-0.03)),
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
    """Renders the Pro Scouting dropdowns, 3D rotatable human mannequin, and dual radar chart."""
    st.markdown("### 🌟 Compare Yourself with 80+ International & League Stars")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.1, 1.35, 1.35, 1.0])

    with f_col1:
        scout_sport = st.selectbox("1. Choose Sport", sports_list, index=sports_list.index(selected_sport), key="scout_sport")
    scout_sport_df = athletes_df[athletes_df["Sport"] == scout_sport]

    with f_col2:
        scout_leagues = scout_sport_df["Team_League"].unique().tolist()
        scout_league = st.selectbox("2. National Team / League", ["All Teams & Leagues"] + scout_leagues, index=0, key="scout_league")
    if scout_league != "All Teams & Leagues":
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
            f"#### 🧍 3D Human Mannequin — `{scout_row['Athlete']}` "
            f"<span class='pill pill-blue' style='margin-left:8px;'>Click & Drag to Rotate 360°</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Jersey synced to {scout_row['Team_League']} • Soreness heatmap active on: **{muscle_focus}**")
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

    with st.expander(f"📋 Browse Complete {scout_sport} International & League Roster ({len(scout_sport_df)} Players)", expanded=False):
        st.dataframe(
            scout_sport_df[
                ["Athlete", "Team_League", "Position", "OVR", "Pace", "Endurance", "Agility", "Power", "Tactical_IQ", "Recovery"]
            ].sort_values("OVR", ascending=False),
            use_container_width=True,
            hide_index=True,
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
        st.caption(f"Smart Conditioning & {len(athletes_df)}+ Pro Athletes")

        # 1. SIDEBAR NAVIGATION MENU
        st.markdown("### 🧭 Menu")
        nav_selection = st.radio(
            "Navigate to:",
            [
                "🏋️ Today's Workout & AI Coach",
                "📈 Progress & History Charts",
                "🌟 Pro Scouting & 3D Mannequin",
                "🗂️ All-in-One Tabbed View",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 2. SPORT, LEAGUE & ATHLETE SELECTION
        st.markdown("### 🏆 Your Sport & Role Model")
        selected_sport = st.selectbox("Sport", sports_list, index=0)
        sport_subset = athletes_df[athletes_df["Sport"] == selected_sport]

        sidebar_leagues = ["All Teams & Leagues"] + sport_subset["Team_League"].unique().tolist()
        selected_sidebar_league = st.selectbox("Team / League Filter", sidebar_leagues, index=0)
        if selected_sidebar_league != "All Teams & Leagues":
            filtered_athletes = sport_subset[sport_subset["Team_League"] == selected_sidebar_league]
        else:
            filtered_athletes = sport_subset

        selected_athlete_name = st.selectbox("Pro Inspiration", filtered_athletes["Athlete"].tolist(), index=0)
        active_athlete = filtered_athletes[filtered_athletes["Athlete"] == selected_athlete_name].iloc[0]

        day_options = list(SPORT_WEEKLY_ROUTINES[selected_sport].keys())
        selected_day_routine = st.selectbox("Workout Day", day_options, index=0)

        st.markdown("---")

        # 3. SIMPLE DAILY BODY CHECK-IN (3 EASY SLIDERS)
        st.markdown("### ❤️ Daily Body Check-In")
        sleep_hours = st.slider("🛌 Hours Slept Last Night", min_value=4.0, max_value=11.0, value=7.8, step=0.5)
        energy_level = st.slider("⚡ Energy & Motivation (1–10)", min_value=1, max_value=10, value=8)
        soreness_level = st.slider("🩹 Muscle Soreness (1=Fresh, 10=Sore)", min_value=1, max_value=10, value=4)
        muscle_focus = st.selectbox(
            "Area Feeling Most Sore (Highlights on 3D Mannequin)",
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

        # 4. OPTIONAL WORKOUT DURATION / EFFORT
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

    inject_custom_css(str(active_athlete["Jersey_Color"]))

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
                    Role Model: <b>{active_athlete['Athlete']}</b> ({active_athlete['Team_League']} • {active_athlete['Position']}) &nbsp;•&nbsp;
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

    elif nav_selection == "🌟 Pro Scouting & 3D Mannequin":
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
        tab1, tab2, tab3 = st.tabs(
            [
                "🏋️ Daily Adaptive Training & Coach",
                "📈 Performance Trends & History Graphs",
                "🌟 Pro Scouting & 3D Mannequin Studio",
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
