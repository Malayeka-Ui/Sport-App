import os
import time
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
# 1. PAGE CONFIGURATION & AI LAB WORKBENCH CSS
# ==============================================================================
st.set_page_config(
    page_title="AI-SPORTS-LAB // Biomechanical & Conditioning Telemetry",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_lab_css(accent_hex: str = "#10B981") -> None:
    """Injects an academic AI Research Lab dark telemetry interface."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        :root {{
            --bg-lab: #070B14;
            --bg-panel: #0E1526;
            --border-grid: rgba(6, 182, 212, 0.22);
            --accent-emerald: #10B981;
            --accent-cyan: #06B6D4;
            --accent-subject: {accent_hex};
            --text-primary: #F3F4F6;
            --text-secondary: #9CA3AF;
        }}

        .stApp {{
            background-color: var(--bg-lab);
            background-image: 
                linear-gradient(rgba(6, 182, 212, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(6, 182, 212, 0.03) 1px, transparent 1px);
            background-size: 28px 28px;
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
        }}

        section[data-testid="stSidebar"] {{
            background: #0A0F1D !important;
            border-right: 1px solid rgba(6, 182, 212, 0.25);
        }}

        /* Sidebar Navigation Radio Modules */
        div[role="radiogroup"] > label {{
            background: rgba(14, 21, 38, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-left: 3px solid transparent;
            border-radius: 6px;
            padding: 8px 12px !important;
            margin-bottom: 5px !important;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem !important;
            transition: all 0.15s ease;
        }}

        div[role="radiogroup"] > label:hover {{
            border-left: 3px solid var(--accent-cyan);
            background: rgba(6, 182, 212, 0.10);
        }}

        /* Top AI Lab Status Header */
        .lab-header {{
            background: linear-gradient(90deg, #0D1424 0%, #090E1A 100%);
            border: 1px solid var(--border-grid);
            border-left: 5px solid var(--accent-subject);
            border-radius: 8px;
            padding: 1rem 1.4rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .lab-title {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            color: #FFFFFF;
            margin: 0;
            letter-spacing: -0.02em;
        }}

        .lab-meta {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: #9CA3AF;
            margin-top: 4px;
        }}

        /* Telemetry Readout Panels */
        .lab-panel {{
            background: rgba(14, 21, 38, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.09);
            border-top: 2px solid var(--accent-cyan);
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.9rem;
        }}

        .panel-tag {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #9CA3AF;
            margin-bottom: 0.25rem;
        }}

        .panel-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F9FAFB;
        }}

        .panel-sub {{
            font-size: 0.78rem;
            color: #9CA3AF;
            margin-top: 0.3rem;
        }}

        /* Lab Badges */
        .lab-badge {{
            display: inline-block;
            padding: 3px 9px;
            border-radius: 4px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 600;
        }}

        .b-optimal {{ background: rgba(16, 185, 129, 0.16); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.45); }}
        .b-caution {{ background: rgba(245, 158, 11, 0.16); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.45); }}
        .b-danger  {{ background: rgba(239, 68, 68, 0.18);  color: #F87171; border: 1px solid rgba(239, 68, 68, 0.45); }}
        .b-cyan    {{ background: rgba(6, 182, 212, 0.16);  color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.45); }}

        /* LLM Inference Console */
        .inference-console {{
            background: #080D1A;
            border: 1px solid rgba(16, 185, 129, 0.45);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.15rem;
            font-family: 'Inter', sans-serif;
        }}

        .console-header {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.76rem;
            color: #10B981;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.45rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            padding-bottom: 0.35rem;
        }}

        .console-output {{
            font-size: 0.94rem;
            line-height: 1.6;
            color: #E5E7EB;
        }}

        /* Protocol Row */
        .protocol-row {{
            background: rgba(14, 21, 38, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-left: 3px solid #06B6D4;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.55rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. INTERNATIONAL & LEAGUE ATHLETE DATABASE (215+ PLAYERS ACROSS 4 SPORTS)
# ==============================================================================
@st.cache_data
def get_pro_athletes_db() -> pd.DataFrame:
    """
    Constructs a comprehensive multi-sport dataset of 215+ international national team
    and major professional league athletes with 6-axis biomechanical feature vectors.
    """
    # Schema: Sport -> List of (Team_League, Jersey_Color, List of (Name, Pos, Pace, End, Agi, Pow, IQ, Rec, Signature))
    db_spec = {
        "Football": [
            ("Real Madrid (La Liga / UCL)", "#06B6D4", [
                ("Kylian Mbappé", "Striker / Inside Forward", 99, 90, 96, 89, 93, 92, "Supersonic transition sprints & clinical half-space finishing"),
                ("Vinícius Júnior", "Left Winger", 98, 92, 98, 83, 91, 93, "Explosive 1v1 touchline dribbling & repeated sprint output"),
                ("Jude Bellingham", "Box-to-Box Attacking Midfielder", 88, 97, 91, 91, 96, 95, "High-volume pressing, aerial box arrivals & tempo control"),
                ("Federico Valverde", "Box-to-Box Engine Midfielder", 94, 99, 89, 94, 93, 98, "Sub-maximal aerobic capacity & 30m ball-striking impulse"),
                ("Rodrygo Goes", "Right Winger / Forward", 93, 90, 95, 82, 92, 90, "Tight-space combination play & high-pressure conversion"),
                ("Luka Modrić", "Deep-Lying Playmaker", 78, 90, 92, 76, 99, 91, "360-degree spatial scanning & trivela pass distribution"),
                ("Eduardo Camavinga", "Dynamic Midfielder", 89, 94, 93, 88, 90, 93, "Lateral recovery tackles & press-resistant ball carrying"),
                ("Aurélien Tchouaméni", "Defensive Midfield Anchor", 84, 93, 85, 94, 93, 92, "Central zone interceptions & aerial duel dominance"),
                ("Antonio Rüdiger", "Aggressive Center-Back", 91, 92, 83, 96, 90, 93, "High-line recovery speed & physical striker neutralization"),
                ("Dani Carvajal", "Attacking Right-Back", 86, 95, 88, 86, 94, 93, "Overlapping flank endurance & defensive duel tenacity"),
                ("Thibaut Courtois", "Goalkeeper", 82, 88, 91, 92, 96, 93, "2.0m wingspan reflex extensions & high-claim command"),
                ("Arda Güler", "Creative Attacking Midfielder", 85, 86, 94, 78, 93, 87, "Left-footed half-space playmaking & set-piece precision"),
            ]),
            ("FC Barcelona (La Liga / Spain)", "#EF4444", [
                ("Lamine Yamal", "Right Wing Playmaker", 93, 89, 98, 79, 95, 90, "Low-gravity 1v1 deceleration & curling far-post delivery"),
                ("Pedri", "Central Tempo Controller", 84, 93, 95, 77, 98, 89, "High-frequency scanning & line-breaking through balls"),
                ("Robert Lewandowski", "Complete No. 9 Striker", 83, 88, 86, 95, 97, 93, "Penalty-box target biomechanics & first-time finishing"),
                ("Raphinha", "High-Workrate Winger", 93, 96, 93, 85, 91, 94, "Counter-pressing sprint volume & diagonal runs"),
                ("Gavi", "Aggressive Pressing Midfielder", 86, 97, 92, 85, 91, 94, "High-intensity counter-pressing & ground duel aggression"),
                ("Frenkie de Jong", "Progressive Ball Carrier", 88, 93, 92, 84, 95, 91, "Midfield press evasion & progressive slalom carries"),
                ("Dani Olmo", "Half-Space No. 10", 86, 88, 95, 80, 95, 87, "One-touch box combinations & tight-angle finishing"),
                ("Pau Cubarsí", "Ball-Playing Center-Back", 82, 89, 84, 86, 94, 90, "Line-breaking vertical passes & composed defensive reading"),
                ("Jules Koundé", "Defensive Right-Back / CB", 90, 93, 89, 88, 92, 93, "1v1 wide channel lock-down & vertical leap power"),
                ("Marc-André ter Stegen", "Sweeper Keeper", 80, 88, 90, 89, 95, 92, "1v1 spread-block reflexes & two-footed build-up passing"),
            ]),
            ("Manchester City (Premier League)", "#38BDF8", [
                ("Erling Haaland", "Target No. 9 Striker", 95, 88, 85, 99, 94, 93, "Explosive penalty-box impulse & aerial vertical leap"),
                ("Rodri", "Defensive Midfield Anchor", 81, 97, 86, 93, 99, 95, "Metronomic passing tempo & second-ball zone control"),
                ("Kevin De Bruyne", "Attacking Midfield Architect", 83, 89, 87, 91, 99, 88, "High-velocity early crosses & transitional vision"),
                ("Phil Foden", "Half-Space Creator", 90, 93, 97, 81, 95, 92, "Instantaneous half-turn acceleration & edge-of-box strikes"),
                ("Bernardo Silva", "Pressing Playmaker", 85, 99, 96, 78, 97, 96, "13km/match aerobic engine & close-control ball retention"),
                ("Rúben Dias", "Defensive Leader CB", 82, 93, 81, 95, 96, 94, "Shot-blocking biomechanics & defensive line organization"),
                ("Joško Gvardiol", "Inverted Left-Back / CB", 89, 93, 88, 93, 92, 93, "Two-footed progressive carries & overlapping box threat"),
                ("Savinho", "Direct Dribbling Winger", 95, 89, 96, 78, 88, 89, "High-tempo 1v1 take-ons & byline cutbacks"),
                ("Ederson", "Sweeper Goalkeeper", 84, 88, 88, 94, 95, 91, "70-yard flat goal-kick trajectories & high-line sweeping"),
            ]),
            ("Arsenal (Premier League)", "#E11D48", [
                ("Bukayo Saka", "Right Inverted Winger", 91, 95, 94, 88, 94, 95, "Contact-balance shielding & near/far-post curlers"),
                ("Martin Ødegaard", "Right-Half-Space Captain", 84, 96, 94, 80, 97, 93, "High-press trigger coordination & disguised through balls"),
                ("Declan Rice", "Ball-Winning Midfielder", 88, 98, 86, 94, 94, 96, "Long-stride interception coverage & set-piece delivery"),
                ("William Saliba", "Recovery Center-Back", 90, 92, 86, 95, 95, 93, "Calm 1v1 sweeping & high-line physical dominance"),
                ("Gabriel Magalhães", "Aggressive Center-Back", 84, 92, 82, 96, 92, 93, "Attacking corner aerial power & box defending"),
                ("Kai Havertz", "Pressing False 9 / Target", 87, 95, 87, 89, 93, 93, "Off-ball space manipulation & aerial hold-up play"),
                ("Gabriel Martinelli", "Transition Left Winger", 97, 94, 93, 82, 88, 92, "Coast-to-coast sprint velocity & pressing intensity"),
            ]),
            ("Liverpool (Premier League)", "#DC2626", [
                ("Mohamed Salah", "Inside Right Forward", 93, 94, 94, 89, 96, 97, "Core-stabilized inside cuts & elite playmaking/scoring"),
                ("Virgil van Dijk", "Commanding Center-Back", 88, 91, 82, 97, 97, 94, "Aerial duel supremacy & 60-yard diagonal switches"),
                ("Trent Alexander-Arnold", "Playmaking Right-Back", 83, 90, 85, 86, 98, 89, "Whipped half-space crosses & long-range passing vision"),
                ("Alexis Mac Allister", "Complete Central Midfielder", 81, 94, 89, 85, 96, 93, "Counter-pressing tempo & edge-of-box shot creation"),
                ("Dominik Szoboszlai", "High-Power Midfielder", 90, 96, 88, 93, 90, 94, "High-velocity ball striking & box-to-box running volume"),
                ("Luis Díaz", "Direct Left Winger", 94, 95, 95, 83, 89, 93, "Relentless ball-carrying & defensive workrate"),
                ("Alisson Becker", "Goalkeeper", 85, 89, 91, 91, 97, 93, "1v1 smothering angles & calm footwork under pressure"),
            ]),
            ("Bayern Munich & Bundesliga", "#F43F5E", [
                ("Harry Kane", "Deep-Lying Complete Striker", 80, 91, 83, 94, 98, 92, "Playmaking No. 9 drops & lethal penalty-box finishing"),
                ("Jamal Musiala", "Slalom Attacking Midfielder", 91, 90, 99, 79, 94, 90, "Elastic ankle agility through crowded defensive blocks"),
                ("Florian Wirtz", "Creative No. 10 (Leverkusen)", 89, 94, 96, 81, 97, 93, "One-touch combination tempo & final-third efficiency"),
                ("Joshua Kimmich", "Deep-Lying Orchestrator", 80, 96, 86, 84, 97, 95, "Chip-pass precision & relentless midfield mentality"),
                ("Alphonso Davies", "Overlapping Left-Back", 99, 93, 94, 86, 87, 92, "Track-sprinter recovery speed & overlapping dribbles"),
                ("Michael Olise", "Inverted Right Playmaker", 89, 89, 95, 82, 93, 89, "Deceptive body feints & whipped inswinging deliveries"),
            ]),
            ("Serie A / Ligue 1 / Global Icons", "#A855F7", [
                ("Lionel Messi", "Free-Role Playmaker (Inter Miami / ARG)", 85, 84, 97, 83, 99, 89, "Unmatched spatial scanning, low-gravity slaloms & free kicks"),
                ("Cristiano Ronaldo", "Penalty-Box Poacher (Al-Nassr / POR)", 87, 92, 87, 96, 96, 97, "2.9m vertical header hangs, two-footed power & elite conditioning"),
                ("Lautaro Martínez", "Pressing Striker (Inter Milan / ARG)", 88, 94, 91, 90, 94, 93, "Front-foot pressing & near-post darting finishes"),
                ("Nicolò Barella", "Box-to-Box Midfielder (Inter / ITA)", 88, 98, 92, 84, 94, 96, "Tireless midfield transitions & half-volley strikes"),
                ("Khvicha Kvaratskhelia", "Two-Footed Winger (Napoli / GEO)", 91, 91, 96, 85, 92, 91, "Unpredictable double-cuts & long-range curlers"),
                ("Rafael Leão", "Power Left Winger (AC Milan / POR)", 97, 86, 94, 91, 88, 88, "Long-stride touchline acceleration & solo counter goals"),
                ("Achraf Hakimi", "Wing-Back Sprinter (PSG / MAR)", 98, 96, 92, 86, 90, 95, "Repeat high-speed flank overlaps & transition sprints"),
                ("Ousmane Dembélé", "Ambidextrous Winger (PSG / FRA)", 96, 87, 98, 80, 89, 86, "Equal left/right-foot fake shots & explosive separation"),
                ("Antoine Griezmann", "Complete Forward (Atletico / FRA)", 84, 96, 91, 83, 98, 95, "All-phase link-up play, defensive tracking & clutch goals"),
                ("Son Heung-min", "Two-Footed Finisher (Spurs / KOR)", 94, 92, 92, 86, 93, 93, "Lethal left/right curlers from edge of the box"),
                ("Bruno Fernandes", "High-Volume Creator (Man Utd / POR)", 83, 98, 87, 85, 96, 97, "High-risk through balls & unmatched match availability"),
                ("Cole Palmer", "Half-Space Playmaker (Chelsea / ENG)", 86, 90, 93, 83, 96, 91, "Sub-zero penalty composure & disguised reverse passes"),
            ]),
        ],
        "Cricket": [
            ("Pakistan National Team & PSL", "#10B981", [
                ("Babar Azam", "Top-Order Batter (No. 3 / Zalmi)", 87, 94, 93, 88, 96, 92, "High-elbow biomechanics, late-contact cover drives & balance"),
                ("Shaheen Shah Afridi", "Left-Arm Fast Bowler (Qalandars)", 96, 91, 89, 95, 92, 90, "145+ kph late-inswinging yorkers & high-release front-leg brace"),
                ("Mohammad Rizwan", "Wicketkeeper Batter (Sultans)", 92, 98, 96, 87, 95, 96, "Tireless crouched glovework, leg-side sweeps & busy running"),
                ("Naseem Shah", "Right-Arm Express Pace (United)", 95, 90, 90, 92, 91, 89, "High-arm seam presentation, late outswing & fiery spells"),
                ("Saim Ayub", "Aggressive Opening Batter (Zalmi)", 90, 88, 94, 90, 89, 90, "No-look pick-up flicks, carrom-ball spin & powerplay intent"),
                ("Fakhar Zaman", "Explosive Left-Hand Opener (Qalandars)", 89, 91, 88, 96, 89, 90, "Clean slog-sweeps over cow corner & big-match temperament"),
                ("Shadab Khan", "Leg-Spin All-Rounder (United)", 91, 92, 96, 88, 94, 91, "Point fielding acrobatics, googly variations & middle-order hits"),
                ("Haris Rauf", "Express Death Bowler (Qalandars)", 98, 90, 91, 94, 89, 90, "150+ kph hard-length skiddy pace & boundary-riding catches"),
                ("Salman Ali Agha", "Spin-Hitting All-Rounder", 87, 94, 91, 87, 94, 93, " Nimble footwork against spin & off-break partnership breakers"),
                ("Abrar Ahmed", "Mystery Spinner (Gladiators)", 82, 89, 87, 83, 93, 89, "Quick-finger carrom & leg-break release disguise"),
            ]),
            ("India National Team & IPL", "#0EA5E9", [
                ("Virat Kohli", "Top-Order Chase Master (RCB)", 94, 98, 95, 91, 98, 96, "Explosive 2-run conversion sprints, wrists-driven flicks & grit"),
                ("Jasprit Bumrah", "All-Phase Spearhead Pace (MI)", 95, 94, 93, 95, 99, 94, "Hyper-extension sling release, unplayable yorkers & slower balls"),
                ("Rohit Sharma", "Opening Batter & Captain (MI)", 84, 90, 88, 97, 97, 90, "Effortless front-foot pull shots & powerplay field manipulation"),
                ("Suryakumar Yadav", "360-Degree T20 Batter (MI)", 91, 92, 97, 93, 96, 92, "Fine-leg ramps, wrists-generated supla shots & extreme SR"),
                ("Ravindra Jadeja", "Three-Dimensional All-Rounder (CSK)", 95, 96, 98, 90, 94, 96, "Bullet-arm direct hits, rapid left-arm spin & lower-order power"),
                ("Hardik Pandya", "Pace-Bowling All-Rounder (MI)", 91, 90, 92, 96, 93, 89, "Heavy-ball seamers & flat-bat death-overs six hitting"),
                ("Rishabh Pant", "Counter-Attacking Keeper (DC)", 89, 91, 93, 96, 92, 90, "One-handed falling sixes & fearless reverse-laps vs pace"),
                ("Yashasvi Jaiswal", "Aggressive Left-Hand Opener (RR)", 91, 92, 93, 94, 91, 92, "Upper-cut boundary clearing & marathon Test double-centuries"),
                ("Shubman Gill", "Classical Top-Order Batter (GT)", 90, 93, 92, 90, 94, 93, "Short-arm jab pulls & high-tempo strike rotation"),
                ("Kuldeep Yadav", "Left-Arm Wrist Spinner (DC)", 84, 90, 89, 84, 95, 90, "Looping drift, sharp wrong'uns & middle-overs wicket bursts"),
                ("Mohammed Shami", "Seam-Position Fast Bowler (GT)", 91, 91, 87, 93, 94, 90, "Laser-straight upright seam & lethal late inward movement"),
            ]),
            ("Australia National Team & BBL/IPL", "#FACC15", [
                ("Pat Cummins", "Fast-Bowling Captain (SRH)", 92, 96, 90, 94, 98, 95, "Relentless top-of-off seam accuracy & clutch lower-order hits"),
                ("Travis Head", "Ultra-Attacking Opener (SRH)", 89, 91, 92, 97, 92, 91, "Fearless square-of-the-wicket slashing & finals hundreds"),
                ("Steve Smith", "Unorthodox Middle-Order Genius", 86, 97, 94, 86, 99, 95, "Back-and-across trigger shuffle & 8-hour concentration"),
                ("Mitchell Starc", "Left-Arm Express Swing (KKR)", 96, 90, 88, 97, 91, 90, "150 kph toe-crushing inswinging yorkers with the new ball"),
                ("Glenn Maxwell", "360-Degree Power All-Rounder (RCB)", 90, 89, 96, 98, 93, 89, "No-footwork switch hits, reverse-sweeps & double-hundred grit"),
                ("Josh Hazlewood", "Line-and-Length Metronome (RCB)", 90, 94, 87, 92, 96, 93, "Test-match corridor accuracy & steep bounce extraction"),
                ("Marnus Labuschagne", "High-Energy No. 3 Batter", 90, 96, 95, 86, 95, 95, "Diving infield saves & relentless crease occupation"),
                ("Adam Zampa", "White-Ball Leg-Spinner", 85, 91, 89, 83, 95, 91, "Over-the-top dip & death-overs big-hitter dismissals"),
            ]),
            ("England National Team & Hundred/IPL", "#EF4444", [
                ("Joe Root", "Classical Test & ODI Anchor", 87, 96, 93, 85, 98, 95, "Late dabs to third man, reverse-scoops & golden-arm spin"),
                ("Ben Stokes", "Match-Winning All-Rounder", 88, 95, 90, 97, 97, 92, "Miracle run-chases, bouncer-barrage spells & slip catches"),
                ("Jos Buttler", "Explosive White-Ball Opener (RR)", 91, 92, 94, 96, 95, 92, "Deep-crease bat speed, ramp shots & death-overs carnage"),
                ("Harry Brook", "High-Backlift Middle-Order Batter", 90, 92, 92, 95, 92, 92, "Still-head pace destruction & 360-degree boundary range"),
                ("Phil Salt", "Powerplay Smasher (KKR)", 91, 90, 92, 96, 90, 91, "First-ball charge down the track & flat-bat pulls"),
                ("Jofra Archer", "Express Fast Bowler", 97, 86, 91, 95, 91, 85, "Effortless 150 kph knuckleballs & chin-music bouncers"),
                ("Adil Rashid", "Crafty Leg-Spin Veteran", 83, 90, 88, 82, 96, 90, "Four distinct googly speeds & flighted lures"),
            ]),
            ("South Africa / NZ / WI / AFG Stars", "#14B8A6", [
                ("Rashid Khan", "T20 Leg-Spin Superstar (AFG / GT)", 92, 94, 96, 91, 97, 95, "100 kph quick-arm googlies & snake-shot lower-order sixes"),
                ("Kane Williamson", "Technical No. 3 Batter (NZ)", 86, 95, 92, 84, 99, 94, "Under-the-eyes late steers & calm tactical captaincy"),
                ("Trent Boult", "New-Ball Swing Master (NZ / RR)", 92, 92, 93, 91, 95, 92, "First-over banana inswing & one-handed boundary catches"),
                ("Rachin Ravindra", "Fluent Left-Hand All-Rounder (NZ / CSK)", 90, 93, 92, 89, 92, 92, " flowing cover drives &back-foot pulls"),
                ("Heinrich Klaasen", "Middle-Overs Spin Destroyer (SA / SRH)", 88, 91, 91, 98, 93, 91, "Back-foot pull-shot power against spin & 185+ strike rate"),
                ("Kagiso Rabada", "Express Strike Bowler (SA / PBKS)", 96, 93, 90, 96, 94, 93, "Effortless biomechanics, heavy seam & high strike rate"),
                ("Aiden Markram", "Classical Batter & Captain (SA)", 90, 92, 93, 90, 94, 92, "High-elbow off-side drives & athletic outfield coverage"),
                ("Nicholas Pooran", "Six-Hitting Keeper-Batter (WI / LSG)", 89, 90, 92, 99, 92, 90, "Zero-backlift bat swing & 105-meter stadium-roof sixes"),
                ("Andre Russell", "Muscle-Power All-Rounder (WI / KKR)", 91, 86, 90, 99, 90, 86, "200+ strike-rate death hitting & heavy-bouncer spells"),
                ("Wanindu Hasaranga", "Googly All-Rounder (SL)", 90, 92, 94, 89, 93, 91, "Stump-to-stump wrong'uns & floating powerplay batting"),
            ]),
        ],
        "Basketball": [
            ("NBA Western Conference Contenders", "#FACC15", [
                ("LeBron James", "Point Forward (LA Lakers / USA)", 91, 95, 90, 98, 99, 99, "Downhill freight-train rim drives, full-court IQ & longevity"),
                ("Anthony Davis", "Two-Way Center (LA Lakers / USA)", 85, 90, 88, 95, 94, 89, "Paint-protecting shot blocks, lob finishing & face-up jumpers"),
                ("Stephen Curry", "Combo Guard (Warriors / USA)", 92, 99, 97, 82, 98, 94, "Constant off-ball marathon movement & sub-0.4s 30ft release"),
                ("Nikola Jokić", "Point Center (Nuggets / Serbia)", 78, 96, 86, 96, 99, 96, "Full-court outlet vision, Sombor shuffle touch & rebounding"),
                ("Jamal Murray", "Clutch Combo Guard (Nuggets / CAN)", 89, 92, 93, 86, 94, 90, "Two-man pick-and-roll pocket game & playoff shot-making"),
                ("Luka Dončić", "Heliocentric Guard (Mavericks / SLO)", 84, 92, 92, 93, 99, 91, "Step-back 3PT separation, deceleration pace & skip passes"),
                ("Kyrie Irving", "Ball-Handling Wizard (Mavericks / USA)", 91, 91, 99, 82, 95, 90, "All-time handle package & high-glass English layup finishes"),
                ("Shai Gilgeous-Alexander", "Slashing Lead Guard (Thunder / CAN)", 93, 95, 98, 86, 96, 94, "Unstoppable start-stop deceleration drives & mid-range craft"),
                ("Chet Holmgren", "Stretch Rim Protector (Thunder / USA)", 84, 89, 87, 86, 92, 89, "Weak-side block timing & trail 3-point shooting"),
            ]),
            ("NBA Western Athletic Superstars", "#3B82F6", [
                ("Anthony Edwards", "Explosive Scoring Guard (Wolves / USA)", 97, 94, 96, 95, 90, 94, "Hang-time poster dunks, point-of-attack defense & pull-ups"),
                ("Victor Wembanyama", "Unicorn Center (Spurs / France)", 88, 90, 91, 89, 93, 91, "8-foot wingspan shot-blocking, step-back 3s & fluid handle"),
                ("Kevin Durant", "7-Foot Scorer (Suns / USA)", 87, 92, 91, 87, 97, 91, "Unblockable high-release mid-range jumpers & 3-level scoring"),
                ("Devin Booker", "Three-Level Guard (Suns / USA)", 89, 93, 92, 86, 94, 92, "Pin-down curl footwork, mid-range pull-ups & playmaking"),
                ("Ja Morant", "Acrobatic Point Guard (Grizzlies / USA)", 99, 90, 99, 91, 91, 89, "44-inch vertical hang-time switches & blur transition speed"),
                ("De'Aaron Fox", "Speed Point Guard (Kings / USA)", 99, 93, 97, 84, 91, 92, "End-to-end blur speed, clutch 4th-quarter floaters & steals"),
                ("Domantas Sabonis", "Hub Rebounding Center (Kings / LTU)", 81, 96, 85, 95, 95, 95, "Dribble-handoff orchestration & league-leading rebounding"),
                ("Kawhi Leonard", "Two-Way Wing (Clippers / USA)", 86, 87, 89, 95, 95, 85, "Claw on-ball steals & robotic mid-range spot elevation"),
            ]),
            ("NBA Eastern Conference Contenders", "#10B981", [
                ("Giannis Antetokounmpo", "Slashing Power Forward (Bucks / GRE)", 96, 96, 93, 99, 92, 96, "3-dribble coast-to-coast Euro-steps & weak-side rim blocks"),
                ("Damian Lillard", "Deep-Range Point Guard (Bucks / USA)", 90, 91, 93, 85, 95, 90, "35-foot logo pull-ups & Dame-Time clutch isolation"),
                ("Jayson Tatum", "Two-Way Wing (Celtics / USA)", 89, 95, 91, 91, 94, 95, "Sidestep 3PT creation, switchable wing defense & durability"),
                ("Jaylen Brown", "Slasher Two-Guard (Celtics / USA)", 94, 93, 93, 92, 89, 93, "Explosive transition dunks, mid-post footwork & ball pressure"),
                ("Joel Embiid", "Dominant Scoring Center (76ers / USA)", 80, 87, 85, 99, 94, 86, "Foul-drawing post footwork, nail jumper touch & rim defense"),
                ("Tyrese Maxey", "Speed Combo Guard (76ers / USA)", 98, 95, 96, 82, 91, 94, " downhill transition bursts, deep catch-and-shoot & floaters"),
                ("Jalen Brunson", "Crafty Lead Guard (Knicks / USA)", 88, 95, 95, 87, 96, 93, "Pivot-foot paint mastery, clutch shot-making & low turnovers"),
                ("Donovan Mitchell", "Explosive Combo Guard (Cavs / USA)", 95, 93, 95, 91, 92, 92, "Euro-step paint splits, long wingspan steals & pull-up 3s"),
                ("Tyrese Haliburton", "Pass-First Point Guard (Pacers / USA)", 91, 94, 93, 81, 98, 92, "Full-court hit-ahead skips & high-tempo transition offense"),
                ("Bam Adebayo", "Switchable Defensive Center (Heat / USA)", 88, 95, 91, 93, 94, 95, "1-through-5 perimeter switching & high-post playmaking"),
            ]),
        ],
        "Badminton": [
            ("BWF Europe & Denmark National Team", "#EF4444", [
                ("Viktor Axelsen", "Men's Singles Attacking Baseline (DEN)", 91, 94, 96, 95, 98, 93, "Steep overhead jump smashes & commanding court reach"),
                ("Anders Antonsen", "Men's Singles All-Court Tactician (DEN)", 90, 93, 94, 91, 96, 92, "Deceptive net exchanges & counter-attacking variations"),
                ("Kim Astrup", "Men's Doubles Tactical Creator (DEN)", 90, 92, 94, 89, 97, 92, "Disguised service flicks & mid-court interception angles"),
                ("Carolina Marín", "Women's Singles High-Pace Attacker (ESP)", 95, 94, 95, 94, 95, 91, "Left-handed rapid pace injection & relentless vocal pressure"),
                ("Toma Junior Popov", "Men's Singles All-Rounder (FRA)", 91, 92, 92, 91, 91, 91, "Explosive net coverage & singles/doubles versatility"),
            ]),
            ("BWF China & Legends Squad", "#F59E0B", [
                ("Lin Dan", "Men's Singles GOAT Counter-Attacker (CHN)", 95, 97, 99, 93, 99, 95, "Explosive net lunges, wrist flicks & sixth-sense anticipation"),
                ("Shi Yuqi", "Men's Singles Offensive Specialist (CHN)", 93, 92, 95, 96, 95, 91, "Whip-like cross-court smashes & razor-sharp net kills"),
                ("Li Shifeng", "Men's Singles Rally & Counter (CHN)", 92, 96, 94, 92, 93, 95, "Elastic defensive retrievals & second-game stamina surges"),
                ("Chen Yufei", "Women's Singles Defensive Anchor (CHN)", 92, 97, 96, 89, 97, 96, "Unbreakable baseline retrieval & championship rally patience"),
                ("Wang Zhiyi", "Women's Singles Four-Corner Mover (CHN)", 91, 95, 94, 89, 94, 94, "High-clear precision & patient counter-attack setups"),
                ("Liang Weikeng", "Men's Doubles Rear-Court Power (CHN)", 92, 90, 93, 99, 91, 90, "Thunderous back-court jump smashes & rotational speed"),
                ("Wang Chang", "Men's Doubles Front-Court Wizard (CHN)", 96, 91, 98, 88, 96, 92, "Instinctive net reflexes & behind-the-back interceptions"),
                ("Zheng Siwei", "Mixed Doubles Rear-Court Dominator (CHN)", 96, 95, 96, 96, 97, 94, "Relentless jump-smash barrage & side-to-side coverage"),
                ("Huang Yaqiong", "Mixed Doubles Net Queen (CHN)", 94, 93, 98, 86, 99, 94, "Unmatched front-court tape reading & cross-net kills"),
            ]),
            ("BWF Korea / Japan / Taipei", "#3B82F6", [
                ("An Se-young", "Women's Singles Defensive Titan (KOR)", 94, 99, 98, 89, 97, 98, "Relentless rally stamina, elastic split-steps & coverage"),
                ("Seo Seung-jae", "Men's/Mixed Doubles Ironman (KOR)", 94, 96, 96, 93, 97, 95, "Left-handed angles, flat drive supremacy & multi-event stamina"),
                ("Kodai Naraoka", "Men's Singles Marathon Defender (JPN)", 92, 99, 97, 86, 95, 98, "90-minute match endurance & low-error defensive walls"),
                ("Akane Yamaguchi", "Women's Singles Explosive Retriever (JPN)", 95, 97, 98, 91, 95, 96, "Compact footwork, deceptive clears & springy jump transitions"),
                ("Yuta Watanabe", "Mixed/Men's Doubles Deception (JPN)", 96, 94, 99, 89, 97, 94, "Drop-shot disguise from rear court & diving defensive saves"),
                ("Tai Tzu-ying", "Women's Singles Deception Artist (TPE)", 93, 91, 98, 90, 99, 92, "Disguised drop shots, late wrist holds & court geometry"),
                ("Chou Tien-chen", "Men's Singles Ironman Veteran (TPE)", 90, 98, 93, 91, 94, 97, "3-game physicality & net-rush counter attacks"),
            ]),
            ("BWF Malaysia / India / Indonesia / Thailand", "#10B981", [
                ("Lee Chong Wei", "Men's Singles Speed Legend (MAS)", 99, 96, 99, 92, 96, 95, "Supersonic defensive recovery & mid-court jump smashes"),
                ("Lee Zii Jia", "Men's Singles Power Attacker (MAS)", 93, 90, 94, 98, 90, 91, "Backhand whip smashes & explosive baseline power"),
                ("Aaron Chia", "Men's Doubles Flat-Drive Specialist (MAS)", 93, 92, 96, 90, 95, 92, "Flat-exchange reflexes & defensive block placement"),
                ("Loh Kean Yew", "Men's Singles Speed Sprinter (SGP)", 98, 91, 97, 93, 90, 92, "Lightning court-to-court bursts & steep overhead winners"),
                ("Kunlavut Vitidsarn", "Men's Singles World Champion (THA)", 93, 96, 96, 91, 97, 95, "Counter-punching accuracy & elite defensive absorption"),
                ("Lakshya Sen", "Men's Singles High-Pace Retriever (IND)", 95, 95, 96, 90, 92, 94, "Diving defensive saves & rapid net-to-baseline transitions"),
                ("PV Sindhu", "Women's Singles Attacking Tall (IND)", 90, 92, 91, 95, 93, 92, "High-point steep cross smashes & big-tournament mentality"),
                ("Satwiksairaj Rankireddy", "Men's Doubles Power Smasher (IND)", 91, 91, 92, 99, 92, 91, "World-record 565 kph smash velocity & rear-court dominance"),
                ("Chirag Shetty", "Men's Doubles Net Interceptor (IND)", 95, 93, 97, 90, 95, 93, "Lightning front-court interceptions & aggressive net pressure"),
                ("HS Prannoy", "Men's Singles Clutch Giant-Killer (IND)", 90, 94, 92, 93, 95, 93, "Backhand straight smashes & decider-set mental composure"),
                ("Anthony Sinisuka Ginting", "Men's Singles Explosive Pace (INA)", 97, 90, 98, 93, 91, 90, "Deceptive net slices & ultra-fast attacking footwork"),
                ("Jonatan Christie", "Men's Singles Rally & Counter (INA)", 93, 95, 94, 92, 94, 94, "Physical baseline attrition & clinical straight smashes"),
                ("Gregoria Mariska Tunjung", "Women's Singles Stroke Maker (INA)", 91, 91, 94, 90, 94, 91, "Silky cross-court drops & crisp overhead technique"),
            ]),
        ],
    }

    rows = []
    for sport, groups in db_spec.items():
        for team_league, color, players in groups:
            for p_name, pos, pace, end, agi, pow_, iq, rec, sig in players:
                ovr = int(round((pace + end + agi + pow_ + iq + rec) / 6.0))
                rows.append(
                    {
                        "Sport": sport,
                        "Team_League": team_league,
                        "Athlete": p_name,
                        "Position": pos,
                        "Pace": pace,
                        "Endurance": end,
                        "Agility": agi,
                        "Power": pow_,
                        "Tactical_IQ": iq,
                        "Recovery": rec,
                        "OVR": ovr,
                        "Jersey_Color": color,
                        "Signature_Style": sig,
                    }
                )
    return pd.DataFrame(rows)


def generate_28_day_seed_log() -> pd.DataFrame:
    """Generates a 28-day periodized multi-sport training dataset in session state."""
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
        "Monday: Shadow Footwork": [
            {"Drill": "6-Corner Explosive Shadow Footwork", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Split-step timing & court recovery"},
            {"Drill": "Rear-Court Scissor Jump to Net Lunge", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Posterior-anterior kinetic chain"},
            {"Drill": "Low Isometric Lunge Hold + Racket Tracking", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Patellar tendon & quad isometric"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Forehand/Backhand Defense Shuttles", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Lateral hip abduction drive"},
            {"Drill": "Reactive Visual Split-Step Sprint Matrix", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Visual-motor reaction latency"},
            {"Drill": "Plyometric Hurdle Hops to Deep Lunge", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Stretch-shortening cycle (SSC)"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "Multi-Shuttle Rapid Net Kill & Lift", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Forearm pronation & net dominance"},
            {"Drill": "2-vs-1 Flat Drive & Smash Defense", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "High-tempo reflex blocks"},
            {"Drill": "Cross-Court Drop & Clear Precision", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Fine motor control under fatigue"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "40s Work / 20s Rest Multi-Shuttle Rally", "Base_Sets": 8, "Base_Rest_Sec": 40, "Focus": "Lactate clearance & VO2 peak"},
            {"Drill": "Repeat Jump-Smash Power Protocol", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Alactic power maintenance"},
            {"Drill": "Rotator Cuff Eccentric Band Deceleration", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Glenohumeral joint integrity"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "21-Point Clutch Match Play (Start 14-14)", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "High-pressure tactical execution"},
            {"Drill": "Service-Return 3rd-Shot Attack Patterns", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "First-3-shots rally control"},
            {"Drill": "Parasympathetic Breathwork & Myofascial Roll", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Vagal tone restoration"},
        ],
    },
    "Football": {
        "Monday: Shadow Footwork": [
            {"Drill": "180° Half-Turn Scanning & First Touch", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Proprioceptive spatial scanning"},
            {"Drill": "Close-Control Cone Slalom & Deceleration", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Eccentric braking mechanics"},
            {"Drill": "Copenhagen Adductor & Nordic Hamstring", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Posterior chain injury shield"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "30m Max-Velocity Fly Sprints", "Base_Sets": 6, "Base_Rest_Sec": 90, "Focus": "Top-end horizontal force"},
            {"Drill": "5-10-5 Pro Agility Shuttle Turn", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "Change of direction deficit (CODD)"},
            {"Drill": "1v1 Channel Recovery Defending Sprints", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Hip-turn & acceleration"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "6v3 High-Intensity One-Touch Rondo", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Cognitive processing speed"},
            {"Drill": "Overlap/Underlap Cutback Finishing", "Base_Sets": 5, "Base_Rest_Sec": 60, "Focus": "High-heart-rate composure"},
            {"Drill": "60m Diagonal Switch & First-Touch Cushion", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Hip mobility & ball striking"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "Repeated Sprint Ability (6 x 40m Shuttle)", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Phosphocreatine resynthesis"},
            {"Drill": "Box-to-Box High-Speed Intervals (15s/15s)", "Base_Sets": 8, "Base_Rest_Sec": 30, "Focus": "Aerobic power ceiling"},
            {"Drill": "Anti-Rotation Pallof Press & Core Matrix", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "Pelvic stability in duels"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "11v11 High-Press vs Low-Block Phase Play", "Base_Sets": 4, "Base_Rest_Sec": 120, "Focus": "Collective tactical compactness"},
            {"Drill": "Set-Piece Delivery & Near-Post Runs", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Dead-ball execution"},
            {"Drill": "Lower-Limb Contrast Mobility & Flush", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Inflammatory marker clearance"},
        ],
    },
    "Basketball": {
        "Monday: Shadow Footwork": [
            {"Drill": "Defensive Slide-Drop-Contest Triangle", "Base_Sets": 6, "Base_Rest_Sec": 45, "Focus": "Lateral hip stiffness & balance"},
            {"Drill": "Triple-Threat Jab-Step & Split Series", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "First-step separation"},
            {"Drill": "Single-Leg Ankle Isometric Balance", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Lateral ligament protection"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Baseline-to-Baseline Euro-Step Finish", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Deceleration & stride length"},
            {"Drill": "Lane Agility Cone Sprint & Backpedal", "Base_Sets": 5, "Base_Rest_Sec": 65, "Focus": "Multi-planar transition speed"},
            {"Drill": "Approach Vertical Jump & 2-Foot Landing", "Base_Sets": 4, "Base_Rest_Sec": 75, "Focus": "Impulse absorption"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "Spain Pick-and-Roll Pocket Pass / Pull-Up", "Base_Sets": 6, "Base_Rest_Sec": 55, "Focus": "2-man game manipulation"},
            {"Drill": "Off-Screen Relocation Catch-and-Shoot", "Base_Sets": 6, "Base_Rest_Sec": 50, "Focus": "Shot preparation footwork"},
            {"Drill": "Contested Finishing Through Contact Pad", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Mid-air core bracing"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "17s Sideline Conditioning Shuttles", "Base_Sets": 7, "Base_Rest_Sec": 45, "Focus": "Glycolytic buffering capacity"},
            {"Drill": "Fatigued Free-Throw Pressure Protocol", "Base_Sets": 5, "Base_Rest_Sec": 35, "Focus": "Autonomic heart rate control"},
            {"Drill": "Eccentric Single-Leg Decline Squat", "Base_Sets": 4, "Base_Rest_Sec": 50, "Focus": "Patellar tendon remodeling"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "Late-Clock (8s Shot Clock) Half-Court Play", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Crunch-time spacing"},
            {"Drill": "Full-Court Press Break & Transition", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Ball security under trap"},
            {"Drill": "Thoracic & Hip Flexor Decompression Flow", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Postural reset"},
        ],
    },
    "Cricket": {
        "Monday: Shadow Footwork": [
            {"Drill": "Front/Back-Foot Batting Trigger Shadow", "Base_Sets": 6, "Base_Rest_Sec": 40, "Focus": "Head stillness & center of mass"},
            {"Drill": "Slip Cordon Reactive Low-Catch Split-Steps", "Base_Sets": 5, "Base_Rest_Sec": 45, "Focus": "Pre-delivery split-step"},
            {"Drill": "Thoracic Rotation & Scapular Mobility", "Base_Sets": 4, "Base_Rest_Sec": 40, "Focus": "Bowling/throwing shoulder health"},
        ],
        "Tuesday: Linear Footwork": [
            {"Drill": "Explosive 2s & 3s Crease-Turning Shuttle", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Low center-of-gravity braking"},
            {"Drill": "Bowler Run-Up Rhythm & Bound Alignment", "Base_Sets": 6, "Base_Rest_Sec": 70, "Focus": "Front-foot block impulse"},
            {"Drill": "In-Field Swoop & Off-Balance Direct Hit", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Rotational core transfer"},
        ],
        "Wednesday: Court Drills": [
            {"Drill": "140+ kph Short-Ball Pull & Duck Session", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Visual tracking & weight transfer"},
            {"Drill": "Spin Footwork: Sweep & Advance Matrix", "Base_Sets": 5, "Base_Rest_Sec": 50, "Focus": "Pitch reading & wrist work"},
            {"Drill": "Death-Overs Yorker & Slower-Ball Spot", "Base_Sets": 5, "Base_Rest_Sec": 75, "Focus": "Release point consistency"},
        ],
        "Thursday: Conditioning": [
            {"Drill": "Simulated 6-Over Spell / 100-Ball Batting", "Base_Sets": 6, "Base_Rest_Sec": 60, "Focus": "Neuromuscular fatigue resistance"},
            {"Drill": "Boundary Rope Dive & Relay Throw Sprints", "Base_Sets": 5, "Base_Rest_Sec": 55, "Focus": "Repeat high-intensity efforts"},
            {"Drill": "Lumbar Multifidus & Oblique Stabilization", "Base_Sets": 4, "Base_Rest_Sec": 45, "Focus": "L4-L5 pars stress protection"},
        ],
        "Friday: Game Simulation": [
            {"Drill": "Powerplay / Death Run-Chase Scenario (12 RPO)", "Base_Sets": 5, "Base_Rest_Sec": 90, "Focus": "Gap manipulation under pressure"},
            {"Drill": "Field-Setting Simulation vs 360° Batter", "Base_Sets": 4, "Base_Rest_Sec": 60, "Focus": "Tactical matchup execution"},
            {"Drill": "Posterior Shoulder & Hamstring PNF Stretch", "Base_Sets": 3, "Base_Rest_Sec": 30, "Focus": "Myofascial length restoration"},
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
    """Calculates Daily Readiness Index (0-100%), live ACWR, and workout scaling parameters."""
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
        acwr_zone = "UNDER-LOAD (<0.8)"
        acwr_badge = "b-cyan"
        acwr_explain = "Acute load below chronic baseline (Detraining risk)"
    elif acwr <= 1.30:
        acwr_zone = "SWEET SPOT (0.8–1.3)"
        acwr_badge = "b-optimal"
        acwr_explain = "Optimal progressive overload window (Low injury risk)"
    elif acwr <= 1.50:
        acwr_zone = "OVERREACH (1.3–1.5)"
        acwr_badge = "b-caution"
        acwr_explain = "Elevated acute fatigue accumulation (Monitor closely)"
    else:
        acwr_zone = "DANGER SPIKE (>1.5)"
        acwr_badge = "b-danger"
        acwr_explain = "Critical workload spike detected (High soft-tissue risk)"

    if readiness_index >= 80 and acwr <= 1.30:
        set_multiplier = 1.15
        rest_multiplier = 0.90
        intensity_tag = "PEAK OVERLOAD (90–95% Vmax)"
        readiness_badge = "b-optimal"
        readiness_status = "CNS PRIME // GREEN"
        readiness_color = "#10B981"
    elif readiness_index >= 62 and acwr <= 1.45:
        set_multiplier = 1.00
        rest_multiplier = 1.00
        intensity_tag = "SUB-MAXIMAL TEMPO (80–85%)"
        readiness_badge = "b-cyan"
        readiness_status = "NOMINAL // STABLE"
        readiness_color = "#06B6D4"
    elif readiness_index >= 45 and acwr <= 1.55:
        set_multiplier = 0.72
        rest_multiplier = 1.30
        intensity_tag = "AUTOREGULATED DELOAD (65–75%)"
        readiness_badge = "b-caution"
        readiness_status = "CNS FATIGUE // CAUTION"
        readiness_color = "#F59E0B"
    else:
        set_multiplier = 0.50
        rest_multiplier = 1.60
        intensity_tag = "ACTIVE RECOVERY TAPER (50%)"
        readiness_badge = "b-danger"
        readiness_status = "CRITICAL // DELOAD"
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
        "acwr_badge": acwr_badge,
        "acwr_explain": acwr_explain,
        "set_multiplier": set_multiplier,
        "rest_multiplier": rest_multiplier,
        "intensity_tag": intensity_tag,
        "readiness_badge": readiness_badge,
        "readiness_status": readiness_status,
        "acwr_penalty": round(acwr_penalty, 1),
    }


def build_adaptive_schedule(
    sport: str,
    day_key: str,
    metrics: Dict[str, object],
    muscle_focus: str,
) -> pd.DataFrame:
    """Rescales sets, rest intervals, and drill intensity based on daily telemetry."""
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
                "Drill_Protocol": item["Drill"],
                "Base_Sets": base_sets,
                "Scaled_Sets": adj_sets,
                "Delta": delta_str,
                "Rest_Interval": f"{adj_rest}s",
                "Target_Intensity": metrics["intensity_tag"],
                "Biomechanical_Focus": f"{item['Focus']} [Guard: {muscle_focus}]",
            }
        )
    return pd.DataFrame(scaled_rows)


# ==============================================================================
# 4. HIDDEN DUAL-KEY LLM INFERENCE ENGINE (GROQ OPENAI / LLAMA + OPENROUTER)
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
    """Deterministic 3-sentence sports science heuristic engine when offline."""
    readiness = metrics["readiness_index"]
    acwr = metrics["acwr"]
    pct_change = int(round(((total_adj_sets - total_base_sets) / max(1, total_base_sets)) * 100))
    direction = (
        f"up-regulated by +{pct_change}% ({total_base_sets} -> {total_adj_sets} sets)"
        if pct_change > 0
        else (f"down-regulated by {abs(pct_change)}% ({total_base_sets} -> {total_adj_sets} sets)" if pct_change < 0 else f"maintained at {total_adj_sets} baseline sets")
    )

    sentence_1 = (
        f"[TELEMETRY SYNTHESIS]: Given a Daily Readiness Index of {readiness}% ({sleep_hours}h sleep, {soreness}/10 soreness localized in {muscle_focus}, {energy}/10 CNS drive) "
        f"and an ACWR of {acwr} ({metrics['acwr_zone']}), today's {sport} protocol ({day_key}) has been {direction}."
    )

    if readiness < 60 or acwr > 1.35:
        sentence_2 = (
            f"Inter-set rest intervals are extended by {int(round((float(metrics['rest_multiplier']) - 1.0) * 100))}% to buffer central nervous system fatigue "
            f"and prevent eccentric overload on the {muscle_focus} while maintaining sub-maximal movement kinematics modeled on {athlete_name}."
        )
    else:
        sentence_2 = (
            f"Your optimal acute-to-chronic workload ratio supports high-velocity motor unit recruitment, enabling you to replicate {athlete_name}'s "
            f"explosive kinetic chain mechanics without exceeding tissue strain thresholds in the {muscle_focus}."
        )

    recovery_map = {
        "Quads / Hamstrings": "10 minutes of 90/90 hamstring neural flossing, low-cadence spin flush (<110 bpm), and eccentric Nordic sliders",
        "Calves / Achilles": "bent-knee soleus isometric holds (4x45s), plantar fascia myofascial release, and ankle dorsiflexion mobilizations",
        "Lower Back / Core": "supine 90/90 diaphragmatic decompression breathing, McGill Big-3 core stabilization, and thoracic cat-cow extensions",
        "Shoulders / Rotator Cuff": "posterior capsule sleeper stretches, theraband external rotations, and scapular wall slides",
        "Knees / Patella": "Spanish squat isometric holds (4x30s at 70° flexion), VMO foam rolling, and terminal knee extensions",
        "Full Body Systemic": "12 minutes of parasympathetic box breathing (4-4-6-2 cadence), contrast hydrotherapy, and full-chain myofascial rolling",
    }
    drill_rec = recovery_map.get(muscle_focus, recovery_map["Full Body Systemic"])
    sentence_3 = f"Post-session active recovery mandate: execute {drill_rec} to accelerate parasympathetic vagal tone restoration."

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
            temperature=0.3,
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
        "temperature": 0.3,
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
        "temperature": 0.3,
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
) -> Tuple[str, str, float]:
    """Routes to Groq (OpenAI/Llama), OpenRouter, or the deterministic Heuristic Engine and returns latency (ms)."""
    t0 = time.perf_counter()
    fallback_text = generate_heuristic_coaching_prescription(
        sport, athlete_name, day_key, sleep_hours, soreness, muscle_focus, energy, metrics, total_base_sets, total_adj_sets
    )

    groq_clean = groq_key.strip()
    or_clean = openrouter_key.strip()

    if routing_mode == "Offline Heuristic Engine (No API Key)" or (not groq_clean and not or_clean):
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return fallback_text, "HEURISTIC-ENGINE-v2.4 (LOCAL)", latency_ms

    system_prompt = (
        "You are an elite Sports Science AI Lab Director. "
        "Write EXACTLY 3 concise, technical yet clear sentences for the athlete. "
        "Sentence 1: Justify why today's working sets and rest intervals were scaled based on Readiness Index and ACWR. "
        "Sentence 2: Provide a biomechanical execution cue for today's routine referencing the pro athlete benchmark. "
        "Sentence 3: Prescribe a targeted active recovery protocol for their sore muscle region."
    )

    user_prompt = (
        f"Sport: {sport} | Benchmark Subject: {athlete_name} | Routine: {day_key}\n"
        f"Biometrics: Sleep={sleep_hours}h, Soreness={soreness}/10 ({muscle_focus}), CNS_Energy={energy}/10\n"
        f"Calculated Telemetry: Readiness={metrics['readiness_index']}%, ACWR={metrics['acwr']} ({metrics['acwr_zone']}), "
        f"Set_Scaling={total_base_sets} -> {total_adj_sets}, Target_Intensity={metrics['intensity_tag']}."
    )

    attempt_queue: List[Tuple[str, str, str]] = []
    if routing_mode == "Auto-Detect (Groq ➔ OpenRouter ➔ Heuristic)":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Force Groq API":
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
    elif routing_mode == "Force OpenRouter API":
        if or_clean:
            attempt_queue.append(("OpenRouter", or_clean, openrouter_model))
        if groq_clean:
            attempt_queue.append(("Groq", groq_clean, groq_model))

    for provider_label, key_val, model_val in attempt_queue:
        try:
            if provider_label == "Groq":
                text_out = _call_groq(key_val, model_val, system_prompt, user_prompt)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return text_out, f"GROQ-API // {model_val}", latency_ms
            else:
                text_out = _call_openrouter(key_val, model_val, system_prompt, user_prompt)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                return text_out, f"OPENROUTER-API // {model_val}", latency_ms
        except Exception:
            continue

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return fallback_text, "HEURISTIC-FALLBACK (API-TIMEOUT)", latency_ms


# ==============================================================================
# 5. CONTINUOUS 3D VOLUMETRIC HUMAN MANNEQUIN ENGINE (PLOTLY SURFACE LOFTING)
# ==============================================================================
def _loft_anatomical_tube(
     stations: List[Tuple[float, float, float, float, float]],
    color_hex: str,
    label: str,
    n_theta: int = 26,
    opacity: float = 0.98,
) -> go.Surface:
    """
    Lofts a continuous, smooth-shaded 3D anatomical skin across multiple cross-sectional stations.
    Each station is (cx, cy, cz, rx, ry), producing a single connected human torso or full limb.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    n_stations = len(stations)

    x_grid = np.zeros((n_stations, n_theta))
    y_grid = np.zeros((n_stations, n_theta))
    z_grid = np.zeros((n_stations, n_theta))

    pts = np.array([[s[0], s[1], s[2]] for s in stations], dtype=float)

    for i in range(n_stations):
        cx, cy, cz, rx, ry = stations[i]
        # Compute smooth tangent vector along anatomical centerline
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n_stations - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]

        norm_t = float(np.linalg.norm(tangent))
        t_hat = tangent / norm_t if norm_t > 1e-6 else np.array([0.0, 0.0, 1.0])

        # Stable local cross-section axes (e1: lateral X, e2: anterior-posterior Y)
        ref = np.array([0.0, 1.0, 0.0]) if abs(t_hat[1]) < 0.88 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(ref, t_hat)
        e1 /= max(1e-6, float(np.linalg.norm(e1)))
        e2 = np.cross(t_hat, e1)
        e2 /= max(1e-6, float(np.linalg.norm(e2)))

        x_grid[i, :] = cx + rx * np.cos(theta) * e1[0] + ry * np.sin(theta) * e2[0]
        y_grid[i, :] = cy + rx * np.cos(theta) * e1[1] + ry * np.sin(theta) * e2[1]
        z_grid[i, :] = cz + rx * np.cos(theta) * e1[2] + ry * np.sin(theta) * e2[2]

    return go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        surfacecolor=np.zeros_like(z_grid),
        colorscale=[[0, color_hex], [1, color_hex]],
        showscale=False,
        opacity=opacity,
        name=label,
        hovertemplate=f"<b>{label}</b><extra></extra>",
        lighting=dict(ambient=0.55, diffuse=0.82, specular=0.55, roughness=0.28, fresnel=0.30),
        lightposition=dict(x=140, y=240, z=260),
    )


def build_3d_biomechanical_avatar(
    athlete_row: pd.Series,
    muscle_focus: str,
    soreness_level: int,
    readiness_index: float,
) -> go.Figure:
    """
    Renders a continuous, seamless 3D Volumetric Human Mannequin with sculpted anatomy,
    team jersey colors, and localized muscle soreness heatmap highlighting.
    """
    jersey_hex = str(athlete_row["Jersey_Color"])
    mannequin_metal = "#CBD5E1"   # Sculpted studio mannequin alloy
    shorts_hex = "#1E293B"        # Compression shorts slate
    shoe_hex = "#06B6D4"          # Telemetry footwear cyan

    if soreness_level >= 7:
        sore_hex = "#EF4444"      # Severe strain crimson
    elif soreness_level >= 4:
        sore_hex = "#F59E0B"      # Moderate strain amber
    else:
        sore_hex = None

    def region_color(tag: str, default_c: str) -> str:
        if sore_hex is None:
            return default_c
        if muscle_focus in ("Full Body Systemic", tag):
            return sore_hex
        return default_c

    fig = go.Figure()

    # 1. CONTINUOUS SCULPTED HEAD, JAW, NECK & TRAPEZIUS MANNEQUIN MESH
    head_neck_stations = [
        (0.0, 0.0, 152.0, 8.5, 6.2),   # Base of neck / Traps collar
        (0.0, 0.2, 157.0, 5.6, 5.4),   # Mid cervical neck
        (0.0, 0.5, 162.5, 5.3, 5.2),   # Upper neck / Under jaw
        (0.0, 1.5, 166.5, 7.6, 8.8),   # Jawline & chin contour
        (0.0, 1.8, 172.5, 8.8, 10.4),  # Cheekbones & cranium temple
        (0.0, 1.2, 178.5, 8.4, 10.0),  # Forehead dome
        (0.0, 0.5, 183.0, 5.2, 6.2),   # Upper crown
        (0.0, 0.0, 184.8, 0.8, 0.8),   # Cranial apex cap
    ]
    fig.add_trace(
        _loft_anatomical_tube(
            head_neck_stations,
            color_hex=mannequin_metal,
            label=f"Mannequin Cranium & Cervical Spine // {athlete_row['Athlete']}",
        )
    )

    # Cybernetic Visor Strip across Eyes
    visor_stations = [
        (0.0, 9.2, 173.2, 7.6, 1.8),
        (0.0, 9.8, 174.8, 7.8, 2.0),
        (0.0, 9.2, 176.2, 7.4, 1.8),
    ]
    fig.add_trace(_loft_anatomical_tube(visor_stations, color_hex=jersey_hex, label="Biomechanical Telemetry Visor"))

    # 2. CONTINUOUS ANATOMICAL TORSO (COLLARBONE -> PECTORALS -> LATS -> WAIST -> PELVIS)
    torso_upper_color = region_color("Lower Back / Core", jersey_hex)
    torso_Stations = [
        (0.0, 0.0, 153.0, 10.0, 6.8),  # Collarbone / Clavicle
        (0.0, 0.4, 149.0, 21.0, 11.2), # Broad Shoulders / Upper Pectorals
        (0.0, 0.8, 141.0, 19.8, 11.8), # Mid Chest / Pectoralis Major Belly
        (0.0, 0.3, 131.0, 17.2, 10.4), # Lower Ribcage / Latissimus Taper
        (0.0, 0.0, 120.0, 14.8, 9.4),  # Upper Abdomen / Rectus Abdominis
        (0.0, 0.2, 110.0, 14.2, 9.2),  # Narrow Waist / Obliques
        (0.0, -0.3, 102.0, 15.8, 10.4),# Iliac Crest / Hips
    ]
    fig.add_trace(
        _loft_anatomical_tube(
            torso_Stations,
            color_hex=torso_upper_color,
            label=f"Anatomical Torso & Core [{athlete_row['Team_League']}]",
        )
    )

    # Pelvic / Gluteal Shorts Block
    pelvis_stations = [
        (0.0, -0.3, 102.5, 15.9, 10.5),
        (0.0, -0.8, 95.5, 16.4, 11.2), # Gluteal contour
        (0.0, -0.4, 89.5, 15.2, 10.0), # Pelvis base
    ]
    fig.add_trace(
        _loft_anatomical_tube(
            pelvis_stations,
            color_hex=region_color("Lower Back / Core", shorts_hex),
            label="Pelvic Girdle & Gluteus Maximus",
        )
    )

    # 3. CONTINUOUS SCULPTED LEFT & RIGHT ARMS (DELTOID -> BICEP -> ELBOW -> FOREARM -> HAND)
    arm_color = region_color("Shoulders / Rotator Cuff", mannequin_metal)
    delt_color = region_color("Shoulders / Rotator Cuff", jersey_hex)

    # Left Deltoid Cap (Jersey Sleeve)
    l_delt = [
        (-19.5, 0.0, 151.5, 5.2, 5.2),
        (-23.0, 0.2, 147.0, 6.6, 6.2),
        (-25.0, 1.2, 140.5, 5.5, 5.2),
    ]
    fig.add_trace(_loft_anatomical_tube(l_delt, color_hex=delt_color, label="Left Deltoid / Shoulder Cap"))

    # Left Continuous Arm (Bicep -> Elbow -> Forearm -> Wrist -> Hand)
    l_arm_stations = [
        (-25.0, 1.2, 140.5, 5.2, 5.0), # Upper Bicep/Tricep
        (-27.5, 3.5, 131.0, 5.5, 5.2), # Bicep Peak Belly
        (-29.5, 6.0, 122.0, 4.2, 4.1), # Elbow Joint
        (-28.0, 11.5, 115.0, 4.7, 4.4),# Forearm Brachioradialis Belly
        (-25.5, 19.5, 106.5, 3.3, 3.1),# Wrist
        (-24.2, 23.5, 102.5, 3.8, 2.4),# Palm / Hand
        (-23.5, 26.0, 100.5, 2.0, 1.4),# Fingers Tip
    ]
    fig.add_trace(_loft_anatomical_tube(l_arm_stations, color_hex=arm_color, label="Left Upper Limb (Bicep & Forearm)"))

    # Right Deltoid Cap (Jersey Sleeve)
    r_delt = [
        (19.5, 0.0, 151.5, 5.2, 5.2),
        (23.0, 0.4, 147.0, 6.6, 6.2),
        (25.5, 2.0, 141.0, 5.5, 5.2),
    ]
    fig.add_trace(_loft_anatomical_tube(r_delt, color_hex=delt_color, label="Right Deltoid / Shoulder Cap"))

    # Right Continuous Arm (Athletic Ready Stance)
    r_arm_stations = [
        (25.5, 2.0, 141.0, 5.2, 5.0),  # Upper Bicep/Tricep
        (29.0, 5.0, 132.5, 5.5, 5.2),  # Bicep Peak Belly
        (31.5, 8.5, 125.0, 4.2, 4.1),  # Elbow Joint
        (29.5, 15.5, 130.0, 4.7, 4.4), # Forearm Belly
        (26.5, 23.5, 136.5, 3.3, 3.1), # Wrist
        (25.2, 27.0, 139.5, 3.8, 2.5), # Hand
        (24.5, 29.5, 141.0, 2.0, 1.4), # Fingers Tip
    ]
    fig.add_trace(_loft_anatomical_tube(r_arm_stations, color_hex=arm_color, label="Right Upper Limb (Bicep & Forearm)"))

    # 4. CONTINUOUS SCULPTED LEGS (HIP/THIGH -> QUAD BELLY -> KNEE -> CALF BELLY -> ANKLE)
    quad_color = region_color("Quads / Hamstrings", mannequin_metal)
    knee_c = region_color("Knees / Patella", mannequin_metal)
    calf_c = region_color("Calves / Achilles", mannequin_metal)
    short_leg_color = region_color("Quads / Hamstrings", shorts_hex)

    for side_sign, side_name in [(-1.0, "Left"), (1.0, "Right")]:
        # Compression Shorts Thigh Sleeve
        shorts_sleeve = [
            (side_sign * 9.8, -0.2, 91.0, 8.2, 8.2),
            (side_sign * 10.8, 0.8, 79.0, 8.0, 7.8),
            (side_sign * 11.6, 1.8, 69.0, 7.2, 7.0),
        ]
        fig.add_trace(_loft_anatomical_tube(shorts_sleeve, color_hex=short_leg_color, label=f"{side_name} Upper Quadriceps & Hamstring"))

        # Lower Quad to Knee Joint
        quad_to_knee = [
            (side_sign * 11.6, 1.8, 69.0, 7.0, 6.8),
            (side_sign * 12.4, 2.8, 60.0, 6.5, 6.3), # VMO / Quadsweep
            (side_sign * 13.0, 3.5, 51.5, 5.2, 5.4), # Patella / Knee Joint
        ]
        fig.add_trace(
            _loft_anatomical_tube(
                quad_to_knee,
                color_hex=knee_c if muscle_focus == "Knees / Patella" else quad_color,
                label=f"{side_name} Distal Quadriceps & Patella",
            )
        )

        # Calf (Gastrocnemius & Soleus) to Ankle
        calf_stations = [
            (side_sign * 13.0, 3.5, 51.5, 5.1, 5.3), # Knee base
            (side_sign * 13.5, 1.5, 39.0, 6.1, 6.3), # Gastrocnemius muscle belly peak
            (side_sign * 13.8, 0.5, 26.0, 4.6, 4.8), # Soleus taper
            (side_sign * 14.0, -0.2, 12.0, 3.3, 3.5),# Achilles & Ankle Malleolus
            (side_sign * 14.0, 0.0, 6.5, 3.6, 3.8),  # Heel base
        ]
        fig.add_trace(_loft_anatomical_tube(calf_stations, color_hex=calf_c, label=f"{side_name} Gastrocnemius Calf & Achilles"))

        # Sculpted Athletic Shoe
        foot_stations = [
            (side_sign * 14.0, -3.5, 4.2, 4.0, 3.8), # Heel counter
            (side_sign * 14.2, 2.5, 4.0, 4.5, 3.8),  # Midfoot arch
            (side_sign * 14.5, 9.5, 3.2, 4.8, 2.8),  # Forefoot / Metatarsal
            (side_sign * 14.5, 14.5, 2.5, 3.2, 1.6), # Toe box
        ]
        fig.add_trace(_loft_anatomical_tube(foot_stations, color_hex=shoe_hex, label=f"{side_name} Telemetry Footwear"))

    # 5. BIOMECHANICAL FORCE-PLATE RINGS & VERTICAL CALIBRATION AXIS
    theta_ring = np.linspace(0, 2 * np.pi, 60)
    for radius, width_val in [(26.0, 2), (40.0, 5)]:
        fig.add_trace(
            go.Scatter3d(
                x=radius * np.cos(theta_ring),
                y=radius * np.sin(theta_ring),
                z=np.zeros_like(theta_ring),
                mode="lines",
                line=dict(color=jersey_hex, width=width_val),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        height=550,
        showlegend=False,
        scene=dict(
            dragmode="turntable",
            xaxis=dict(title="X (cm)", range=[-60, 60], showgrid=True, gridcolor="rgba(6, 182, 212, 0.10)", zeroline=False, showticklabels=False, showbackground=False),
            yaxis=dict(title="Y (cm)", range=[-60, 60], showgrid=True, gridcolor="rgba(6, 182, 212, 0.10)", zeroline=False, showticklabels=False, showbackground=False),
            zaxis=dict(title="Stature Z (cm)", range=[0, 192], showgrid=True, gridcolor="rgba(6, 182, 212, 0.16)", zeroline=False, showbackground=True, backgroundcolor="rgba(7, 11, 20, 0.65)"),
            camera=dict(eye=dict(x=1.32, y=1.52, z=0.52), center=dict(x=0, y=0, z=-0.03)),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=1.48),
        ),
    )
    return fig


# ==============================================================================
# 6. ANALYTICS & TELEMETRY GRAPHS
# ==============================================================================
def build_acwr_progression_chart(df: pd.DataFrame) -> go.Figure:
    """Dual-axis Plotly chart plotting 28-day Acute vs Chronic load and ACWR safety zone."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Daily_Workload"],
            name="Daily sRPE Load (AU)",
            marker=dict(color="rgba(6, 182, 212, 0.24)", line=dict(color="rgba(6, 182, 212, 0.6)", width=1)),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Acute_Load_7D"],
            mode="lines+markers",
            name="Acute Load (7D EWMA)",
            line=dict(color="#10B981", width=2.8),
            marker=dict(size=5),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Chronic_Load_28D"],
            mode="lines",
            name="Chronic Baseline (28D)",
            line=dict(color="#9CA3AF", width=2.2, dash="dash"),
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
        ),
        secondary_y=True,
    )

    fig.add_hrect(
        y0=0.80,
        y1=1.30,
        line_width=0,
        fillcolor="rgba(16, 185, 129, 0.11)",
        annotation_text="OPTIMAL SWEET SPOT (0.80–1.30)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#34D399", family="JetBrains Mono"),
        secondary_y=True,
    )
    fig.add_hline(
        y=1.50,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="CRITICAL SPIKE THRESHOLD (1.50)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#F87171", family="JetBrains Mono"),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 21, 38, 0.65)",
        height=390,
        margin=dict(l=15, r=15, t=35, b=15),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(title_text="Workload (Arbitrary Units)", secondary_y=False, gridcolor="rgba(255,255,255,0.06)")
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
            name="Prescribed Sets",
            marker_color="rgba(156, 163, 175, 0.40)",
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
        plot_bgcolor="rgba(14, 21, 38, 0.65)",
        barmode="group",
        height=350,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis=dict(title="Working Sets", gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(title="Last 7 Microcycle Sessions"),
    )
    return fig


def build_readiness_vs_fatigue_scatter(df: pd.DataFrame) -> go.Figure:
    """OLS Regression Scatter correlating Sleep Duration, Soreness, and Readiness Output."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Sleep_Hours"],
            y=df["Readiness_Score"],
            mode="markers",
            name="Logged Telemetry",
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
                "Sleep: %{x}h | Readiness: %{y}%<br>"
                "Soreness: %{customdata[0]}/10 | Load: %{customdata[1]} AU<extra></extra>"
            ),
        )
    )

    if len(df) >= 2:
        x_vals = df["Sleep_Hours"].to_numpy(dtype=float)
        y_vals = df["Readiness_Score"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        r_corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 40)
        y_line = slope * x_line + intercept

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name=f"OLS Fit (+{slope:.1f}%/hr, r={r_corr:.2f})",
                line=dict(color="#06B6D4", width=2.5, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14, 21, 38, 0.65)",
        height=350,
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Sleep Duration (Hours)", gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title="Readiness Index (%)", gridcolor="rgba(255,255,255,0.06)"),
    )
    return fig


def build_dual_radar_chart(
    athlete_row: pd.Series,
    compare_user: bool,
    user_traits: Dict[str, int],
) -> go.Figure:
    """6-axis Spider/Radar chart comparing Pro Feature Vector vs User Telemetry."""
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
                name=f"Subject Telemetry (OVR {u_ovr})",
            )
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=40, r=40, t=30, b=30),
        polar=dict(
            bgcolor="rgba(14, 21, 38, 0.72)",
            radialaxis=dict(visible=True, range=[40, 100], gridcolor="rgba(255, 255, 255, 0.10)"),
            angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.12)", tickfont=dict(size=12, color="#F9FAFB", family="JetBrains Mono")),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


# ==============================================================================
# 7. MODULAR LAB WORKBENCH VIEWS
# ==============================================================================
def render_lab_module_1(
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
    """Module 01: AI Conditioning Prescription, Autoregulation & Mathematical Model Inspector."""
    schedule_df = build_adaptive_schedule(
        sport=selected_sport,
        day_key=selected_day_routine,
        metrics=telemetry,
        muscle_focus=muscle_focus,
    )
    total_base_sets = int(schedule_df["Base_Sets"].sum())
    total_adj_sets = int(schedule_df["Scaled_Sets"].sum())

    coach_text, engine_badge, latency_ms = query_ai_coach(
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
        <div class="inference-console">
            <div class="console-header">
                <span>🧠 LLM INFERENCE & HEURISTIC TERMINAL // ENGINE: {engine_badge}</span>
                <span>LATENCY: {latency_ms} ms | TEMP: 0.30</span>
            </div>
            <div class="console-output">{coach_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1.75, 1.05], gap="large")

    with col_left:
        st.markdown(f"#### 📋 Autoregulated Microcycle Protocol — `{selected_day_routine}`")
        for idx, row in schedule_df.iterrows():
            st.markdown(
                f"""
                <div class="protocol-row">
                    <div>
                        <div style="font-weight: 700; font-size: 0.96rem; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                            [0{idx + 1}] {row['Drill_Protocol']}
                        </div>
                        <div style="font-size: 0.82rem; color: #9CA3AF; margin-top: 2px;">
                            {row['Biomechanical_Focus']} &nbsp;|&nbsp; {row['Target_Intensity']}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span class="lab-badge b-optimal">{row['Scaled_Sets']} Sets ({row['Delta']})</span>
                        <span class="lab-badge b-cyan" style="margin-left: 6px;">Rest: {row['Rest_Interval']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.dataframe(schedule_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### 🔬 Mathematical Model & Feature Weights")
        st.markdown(
            f"""
            <div class="lab-panel">
                <div class="panel-tag">VOLUME SCALING TRANSFORMATION</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 0.3rem 0;">
                    {total_base_sets} Base Sets ➔ <span style="color: #10B981;">{total_adj_sets} Scaled Sets</span> (x{telemetry['set_multiplier']})
                </div>
                <div style="font-size: 0.83rem; color: #D1D5DB; line-height: 1.65; margin-top: 0.45rem;">
                    • <b>Sleep Restoration ($w_1=0.40$):</b> {telemetry['sleep_score']}%<br>
                    • <b>Myofascial Freshness ($w_2=0.35$):</b> {telemetry['soreness_score']}%<br>
                    • <b>CNS Neural Drive ($w_3=0.25$):</b> {telemetry['energy_score']}%<br>
                    • <b>ACWR Penalty ($\Phi$):</b> -{telemetry['acwr_penalty']}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📐 Inspect Governing Sports Science Equations (LaTeX)", expanded=False):
            st.latex(r"R_{\text{raw}} = 0.40 \cdot S_{\text{norm}} + 0.35 \cdot M_{\text{norm}} + 0.25 \cdot E_{\text{norm}}")
            st.latex(r"\text{ACWR} = \frac{\text{Acute Load}_{7\text{d}}}{\text{Chronic Load}_{28\text{d}}}, \quad R_{\text{index}} = \text{clip}(R_{\text{raw}} - \Phi(\text{ACWR}), 5, 100)")

        if st.button("⚡ Commit Today's Telemetry Session to Lab Ledger", use_container_width=True, type="primary"):
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
            st.success("Session appended to `st.session_state.training_log` and rolling ACWR recomputed.")

    return total_base_sets, total_adj_sets


def render_lab_module_2(
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
    """Module 02: 28-Day Workload Time-Series, Set Compliance & OLS Regression."""
    st.markdown("#### 📈 28-Day Acute:Chronic Workload Ratio (ACWR) & Load Time-Series")
    st.plotly_chart(build_acwr_progression_chart(df_log), use_container_width=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 📊 Microcycle Volume Compliance (Last 7 Sessions)")
        st.plotly_chart(build_volume_completion_chart(df_log), use_container_width=True)
    with c2:
        st.markdown("#### 🔬 OLS Regression: Sleep & Soreness vs. Readiness")
        st.plotly_chart(build_readiness_vs_fatigue_scatter(df_log), use_container_width=True)

    st.markdown("---")
    form_col, table_col = st.columns([1.05, 1.65], gap="large")

    with form_col:
        st.markdown("#### 📝 Manual Telemetry Session Appender")
        with st.form("manual_lab_log_form", clear_on_submit=False):
            log_date_input = st.date_input("Session Timestamp", value=datetime.now().date())
            log_sport_input = st.selectbox("Sport Discipline", sports_list, index=sports_list.index(selected_sport))
            f1, f2 = st.columns(2)
            with f1:
                log_duration = st.number_input("Duration (Min)", min_value=15, max_value=240, value=int(planned_duration), step=5)
                log_planned_sets = st.number_input("Prescribed Sets", min_value=1, max_value=40, value=int(total_adj_sets))
            with f2:
                log_rpe = st.number_input("Session RPE (1–10)", min_value=1, max_value=10, value=int(planned_rpe))
                log_completed_sets = st.number_input("Completed Sets", min_value=1, max_value=40, value=int(total_adj_sets))

            if st.form_submit_button("💾 Append Record & Recompute Rolling EWMA", use_container_width=True):
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
                st.success("Record committed! Time-series graphs updated.")
                st.rerun()

    with table_col:
        st.markdown("#### 🗄️ Raw Telemetry DataFrame (`st.session_state.training_log`)")
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


def render_lab_module_3(
    athletes_df: pd.DataFrame,
    sports_list: List[str],
    selected_sport: str,
    df_log: pd.DataFrame,
    telemetry: Dict[str, object],
    muscle_focus: str,
    soreness_level: int,
) -> None:
    """Module 03: 3D Volumetric Mannequin Biomechanics & 215+ Pro Scouting Radar."""
    st.markdown(f"#### 🌟 Multi-Sport Pro Scouting & 3D Volumetric Mannequin (`{len(athletes_df)}` Indexed Athletes)")
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.05, 1.4, 1.4, 1.0])

    with f_col1:
        scout_sport = st.selectbox("1. Sport Discipline", sports_list, index=sports_list.index(selected_sport), key="scout_sport")
    scout_sport_df = athletes_df[athletes_df["Sport"] == scout_sport]

    with f_col2:
        scout_leagues = scout_sport_df["Team_League"].unique().tolist()
        scout_league = st.selectbox("2. National Team / League", ["All Teams & Leagues"] + scout_leagues, index=0, key="scout_league")
    if scout_league != "All Teams & Leagues":
        scout_athlete_df = scout_sport_df[scout_sport_df["Team_League"] == scout_league]
    else:
        scout_athlete_df = scout_sport_df

    with f_col3:
        scout_athlete_name = st.selectbox("3. Select Pro Benchmark", scout_athlete_df["Athlete"].tolist(), index=0, key="scout_athlete")
    scout_row = scout_athlete_df[scout_athlete_df["Athlete"] == scout_athlete_name].iloc[0]

    with f_col4:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        compare_user_toggle = st.toggle("Overlay Subject Vector", value=True)

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
        with st.expander("🎚️ Calibrate Subject 6-Trait Feature Vector (40–99)", expanded=False):
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
            f"##### 🧍 3D Volumetric Human Mannequin — `{scout_row['Athlete']}` "
            f"<span class='lab-badge b-cyan' style='margin-left:8px;'>WebGL Orbital 360°</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Surface Lofting: Active | Team Shader: `{scout_row['Jersey_Color']}` | Strain Heatmap: **{muscle_focus} ({soreness_level}/10)**")
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
        st.markdown(f"##### 🕸️ 6-Axis Feature Vector Overlay — `{scout_row['Athlete']}`")
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
        cosine_sim = float(np.dot(pro_vec, user_vec) / (np.linalg.norm(pro_vec) * np.linalg.norm(user_vec))) * 100.0
        l1_similarity = round(float(100.0 - np.mean(np.abs(pro_vec - user_vec))), 1)

        st.markdown(
            f"""
            <div class="lab-panel" style="border-top: 3px solid {scout_row['Jersey_Color']};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="panel-tag">{scout_row['Sport']} // {scout_row['Team_League']}</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; font-family: 'JetBrains Mono', monospace;">
                            {scout_row['Athlete']} <span style="color:{scout_row['Jersey_Color']};">[{scout_row['OVR']} OVR]</span>
                        </div>
                    </div>
                    <span class="lab-badge b-optimal">L1 Match: {l1_similarity}% | CosSim: {cosine_sim:.1f}%</span>
                </div>
                <div style="font-size: 0.84rem; color: #D1D5DB; margin-top: 0.45rem;">
                    <b>Primary Role:</b> {scout_row['Position']}<br>
                    <b>Biomechanical Signature:</b> {scout_row['Signature_Style']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f"🗄️ Inspect Complete {scout_sport} International & League Database ({len(scout_sport_df)} Athletes)", expanded=False):
        st.dataframe(
            scout_sport_df[
                ["Athlete", "Team_League", "Position", "OVR", "Pace", "Endurance", "Agility", "Power", "Tactical_IQ", "Recovery", "Signature_Style"]
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
    # SIDEBAR: AI LAB CONTROL PANEL & HIDDEN API VAULT
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("## 🧪 AI-SPORTS-LAB v4.0")
        st.caption(f"Biomechanical Telemetry • {len(athletes_df)} Pro Profiles")

        # 1. LAB MODULE NAVIGATION MENU
        st.markdown("### 🧭 Lab Modules")
        nav_selection = st.radio(
            "Select Lab Module:",
            [
                "01 // AI Conditioning & Coach",
                "02 // Workload & ACWR Analytics",
                "03 // 3D Mannequin & Pro Scouting",
                "04 // Unified Lab Master View (Tabs)",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 2. SUBJECT & BENCHMARK SELECTION
        st.markdown("### 🏆 Benchmark Subject")
        selected_sport = st.selectbox("Sport Discipline", sports_list, index=0)
        sport_subset = athletes_df[athletes_df["Sport"] == selected_sport]

        sidebar_leagues = ["All Teams & Leagues"] + sport_subset["Team_League"].unique().tolist()
        selected_sidebar_league = st.selectbox("League / National Squad", sidebar_leagues, index=0)
        if selected_sidebar_league != "All Teams & Leagues":
            filtered_athletes = sport_subset[sport_subset["Team_League"] == selected_sidebar_league]
        else:
            filtered_athletes = sport_subset

        selected_athlete_name = st.selectbox("Pro Athlete Reference", filtered_athletes["Athlete"].tolist(), index=0)
        active_athlete = filtered_athletes[filtered_athletes["Athlete"] == selected_athlete_name].iloc[0]

        day_options = list(SPORT_WEEKLY_ROUTINES[selected_sport].keys())
        selected_day_routine = st.selectbox("Microcycle Day Protocol", day_options, index=0)

        # CUSTOM PLAYER SYNTHESIZER (ADD ANY PLAYER IN THE WORLD)
        with st.expander("➕ Synthesize Any Custom Player", expanded=False):
            c_name = st.text_input("Player Name", placeholder="e.g., Arshad Nadeem / Iga Świątek")
            c_team = st.text_input("Team / Country", value=f"{selected_sport} International")
            c_pos = st.text_input("Position / Role", value="All-Rounder / Specialist")
            c_ovr = st.slider("Target OVR Rating", 75, 99, 92)
            c_color = st.color_picker("Jersey Hex Color", "#10B981")
            if st.button("Inject Player into Lab DB", use_container_width=True) and c_name.strip():
                new_p = pd.DataFrame([{
                    "Sport": selected_sport,
                    "Team_League": c_team,
                    "Athlete": c_name.strip(),
                    "Position": c_pos,
                    "Pace": c_ovr,
                    "Endurance": max(70, c_ovr - 1),
                    "Agility": min(99, c_ovr + 1),
                    "Power": c_ovr,
                    "Tactical_IQ": min(99, c_ovr + 2),
                    "Recovery": c_ovr,
                    "OVR": c_ovr,
                    "Jersey_Color": c_color,
                    "Signature_Style": "Custom synthesized international/league telemetry profile",
                }])
                st.session_state.athletes_df = pd.concat([st.session_state.athletes_df, new_p], ignore_index=True)
                st.success(f"Added {c_name}!")
                st.rerun()

        st.markdown("---")

        # 3. PHYSIOLOGICAL BIOMETRIC INPUTS
        st.markdown("### 🧬 Biometric Telemetry Inputs")
        sleep_hours = st.slider("Sleep Duration (Hours)", min_value=4.0, max_value=11.0, value=7.8, step=0.2)
        energy_level = st.slider("CNS Drive & Motivation (1–10)", min_value=1, max_value=10, value=8)
        soreness_level = st.slider("DOMS Muscle Soreness (1–10)", min_value=1, max_value=10, value=4)
        muscle_focus = st.selectbox(
            "Localized Soreness Region (3D Heatmap)",
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

        with st.expander("⏱️ Session sRPE Load Parameters", expanded=False):
            planned_duration = st.slider("Planned Duration (Min)", min_value=30, max_value=150, value=75, step=5)
            planned_rpe = st.slider("Target RPE (1–10)", min_value=1, max_value=10, value=7)
        planned_workload_today = planned_duration * planned_rpe

        # 4. HIDDEN DUAL-KEY LLM VAULT (COLLAPSED & PASSWORD-MASKED)
        with st.expander("🔐 LLM Inference Keys (Hidden Vault)", expanded=False):
            st.caption("Keys are password-masked. Paste Groq, OpenRouter, both, or leave blank for Offline Heuristics.")
            routing_mode = st.selectbox(
                "Inference Routing Mode",
                [
                    "Auto-Detect (Groq ➔ OpenRouter ➔ Heuristic)",
                    "Force Groq API",
                    "Force OpenRouter API",
                    "Offline Heuristic Engine (No API Key)",
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
                "Groq Model (OpenAI & Llama)",
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

            g_stat = "🟢 ACTIVE" if groq_api_key.strip() else "⚪ EMPTY"
            or_stat = "🟢 ACTIVE" if openrouter_api_key.strip() else "⚪ EMPTY"
            st.caption(f"Vault Status — Groq: {g_stat} | OpenRouter: {or_stat}")

    inject_lab_css(str(active_athlete["Jersey_Color"]))

    telemetry = compute_readiness_and_acwr(
        sleep_hours=sleep_hours,
        soreness=soreness_level,
        energy=energy_level,
        df_log=df_log,
        planned_workload_today=planned_workload_today,
    )

    # --------------------------------------------------------------------------
    # TOP AI LAB TELEMETRY HEADER
    # --------------------------------------------------------------------------
    st.markdown(
        f"""
        <div class="lab-header">
            <div>
                <h1 class="lab-title">🧪 AI-SPORTS-LAB // BIOMECHANICAL & CONDITIONING WORKBENCH</h1>
                <div class="lab-meta">
                    DISCIPLINE: <b>{selected_sport.upper()}</b> &nbsp;|&nbsp;
                    SUBJECT ARCHETYPE: <b>{active_athlete['Athlete']}</b> ({active_athlete['Team_League']}) &nbsp;|&nbsp;
                    STRAIN FOCUS: <b>{muscle_focus.upper()} ({soreness_level}/10)</b>
                </div>
            </div>
            <div>
                <span class="lab-badge {telemetry['readiness_badge']}">{telemetry['readiness_status']} ({telemetry['readiness_index']}%)</span>
                <span class="lab-badge {telemetry['acwr_badge']}" style="margin-left: 6px;">ACWR: {telemetry['acwr']:.2f} // {telemetry['acwr_zone']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------------------------
    # 4 TELEMETRY READOUT PANELS
    # --------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="lab-panel" style="border-top-color: {telemetry['readiness_color']};">
                <div class="panel-tag">01 // DAILY READINESS INDEX</div>
                <div class="panel-val" style="color: {telemetry['readiness_color']};">{telemetry['readiness_index']}%</div>
                <div class="panel-sub">Sleep: {telemetry['sleep_score']}% · Soreness: {telemetry['soreness_score']}% · CNS: {telemetry['energy_score']}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="lab-panel" style="border-top-color: #06B6D4;">
                <div class="panel-tag">02 // ACUTE:CHRONIC RATIO (ACWR)</div>
                <div class="panel-val" style="color: #06B6D4;">{telemetry['acwr']:.2f}</div>
                <div class="panel-sub">Acute 7D: {telemetry['acute_7d']:.0f} AU / Chronic 28D: {telemetry['chronic_28d']:.0f} AU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        mult_pct = int(round(float(telemetry["set_multiplier"]) * 100))
        st.markdown(
            f"""
            <div class="lab-panel" style="border-top-color: #FBBF24;">
                <div class="panel-tag">03 // AUTOREGULATED SCALING</div>
                <div class="panel-val" style="color: #FBBF24;">{mult_pct}% Volume</div>
                <div class="panel-sub">Rest Factor: {telemetry['rest_multiplier']:.2f}x · Load: {planned_workload_today} AU</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="lab-panel" style="border-top-color: {active_athlete['Jersey_Color']};">
                <div class="panel-tag">04 // PRO ARCHETYPE RATING</div>
                <div class="panel-val" style="color: {active_athlete['Jersey_Color']};">{active_athlete['OVR']} OVR</div>
                <div class="panel-sub">{active_athlete['Athlete']} · {active_athlete['Position']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    temp_sched = build_adaptive_schedule(selected_sport, selected_day_routine, telemetry, muscle_focus)
    default_adj_sets = int(temp_sched["Scaled_Sets"].sum())

    # --------------------------------------------------------------------------
    # RENDER ACTIVE LAB MODULE
    # --------------------------------------------------------------------------
    if nav_selection == "01 // AI Conditioning & Coach":
        render_lab_module_1(
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

    elif nav_selection == "02 // Workload & ACWR Analytics":
        render_lab_module_2(
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

    elif nav_selection == "03 // 3D Mannequin & Pro Scouting":
        render_lab_module_3(
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
            render_lab_module_1(
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
            render_lab_module_2(
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
            render_lab_module_3(
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
