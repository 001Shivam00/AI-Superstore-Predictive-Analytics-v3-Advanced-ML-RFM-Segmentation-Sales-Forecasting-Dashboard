# dashboard_ai_pro_v3_fixed.py
# ============================================================
# AI-Powered Superstore Predictive Analytics Dashboard v3
# Author: Shivam Yadav  |  v3 — Major Upgrade
# ============================================================
# FIXED in this version:
#   - All hex+opacity colors converted to rgba()
#   - Drag-and-drop file uploader CSS fixed
#   - label_visibility="collapsed" for proper drag-drop
#   - !important added to all uploader CSS rules
#   - pointer-events: auto added for drag-drop to work
# ============================================================

import os, io, warnings, datetime, base64, time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, IsolationForest, GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from prophet import Prophet

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors as rl_colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import os
from google import genai
import anthropic
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# ── API Keys Load ──────────────────────────────
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")

# ── Validation ────────────────────────────────
if not GOOGLE_KEY:
    print("❌ GOOGLE_API_KEY missing in .env")
else:
    print("✅ Google API Key loaded")

if not ANTHROPIC_KEY:
    print("❌ ANTHROPIC_API_KEY missing in .env")
else:
    print("✅ Anthropic API Key loaded")

# ── Gemini Setup ──────────────────────────────
try:
    genai.configure(api_key=GOOGLE_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    response = gemini_model.generate_content("Say hello in one word")
    print(f"✅ Gemini working: {response.text.strip()}")
except Exception as e:
    print(f"❌ Gemini error: {e}")

# ── Claude Setup ──────────────────────────────
try:
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    message = claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say hello in one word"}]
    )
    print(f"✅ Claude working: {message.content[0].text.strip()}")
except Exception as e:
    print(f"❌ Claude error: {e}")

# ─────────────────────────────────────────────
# 🛠️  GLOBAL HELPER — hex + opacity → rgba()
# ─────────────────────────────────────────────

def rgba(hex_color: str, opacity: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{round(opacity, 3)})"


# ─────────────────────────────────────────────
# 🎨  PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AI Superstore Analytics v3",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠",
)


# ══ FULL WIDTH + CLEAN UI CSS ══
st.markdown("""
<style>
div[data-testid="stAppViewContainer"] > section.main > div.block-container {
    padding-top: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
    width: 100% !important;
}
div[data-testid="stDecoration"] { display:none !important; }
div[data-testid="stHeader"]     { display:none !important; }
div[data-testid="stToolbar"]    { display:none !important; }
.appview-container .main .block-container { max-width:100% !important; }

/* ✅ Sidebar hamesha visible aur collapse na ho */
[data-testid="collapsedControl"] { display:none !important; }
section[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 21rem !important;
    max-width: 21rem !important;
    visibility: visible !important;
    display: block !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    transform: none !important;
    margin-left: 0 !important;
    min-width: 21rem !important;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# 🎨  THEME SYSTEM
# ─────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════╗
#   ULTRA PREMIUM THEME SYSTEM  ·  Luxury Edition v3.0
#   7 Cinematic Palettes · Extended Tokens · Full CSS Injection
# ╚══════════════════════════════════════════════════════════════╝

THEMES = {

    "◆ Noir Royale": {
        "label":         "Sovereign Dark",
        "mood":          "dark",
        "bg_primary":    "#08090f",
        "bg_secondary":  "#0f0d1a",
        "bg_card":       "#13101f",
        "bg_elevated":   "#1a1628",
        "bg_glass":      "rgba(167,139,250,0.06)",
        "bg_sidebar":    "linear-gradient(158deg,#08090f 0%,#0f0d1a 55%,#120b1e 100%)",

        "accent1":       "#a78bfa",   # amethyst violet
        "accent2":       "#38bdf8",   # sapphire sky
        "accent3":       "#34d399",   # emerald
        "accent4":       "#fbbf24",   # amber
        "accent5":       "#fb7185",   # rose coral

        "accent1_deep":  "#7c3aed",
        "accent1_glow":  "rgba(167,139,250,0.22)",
        "accent1_border":"rgba(167,139,250,0.16)",
        "accent1_fill":  "rgba(167,139,250,0.08)",

        "text_primary":  "#ece8ff",
        "text_secondary":"#9d94c0",
        "text_muted":    "#3d3560",
        "text_inverse":  "#ede9fe",

        "border":        "rgba(167,139,250,0.14)",
        "border_subtle": "rgba(255,255,255,0.04)",
        "border_strong": "rgba(167,139,250,0.30)",

        "plotly_tpl":    "plotly_dark",
        "particle_color":"#a78bfa",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "◈ Atelier Or": {
        "label":         "Goldleaf Studio",
        "mood":          "light",
        "bg_primary":    "#faf8f3",
        "bg_secondary":  "#f3ead8",
        "bg_card":       "#ffffff",
        "bg_elevated":   "#fffef9",
        "bg_glass":      "rgba(196,160,90,0.07)",
        "bg_sidebar":    "linear-gradient(158deg,#f3ead8 0%,#faf8f3 55%,#ede0c4 100%)",

        "accent1":       "#b8860b",   # dark goldenrod
        "accent2":       "#c0392b",   # lacquer red
        "accent3":       "#1a6b3a",   # forest
        "accent4":       "#1a535c",   # deep teal
        "accent5":       "#6b3a8e",   # plum

        "accent1_deep":  "#78570a",
        "accent1_glow":  "rgba(184,134,11,0.15)",
        "accent1_border":"rgba(196,160,90,0.30)",
        "accent1_fill":  "rgba(196,160,90,0.09)",

        "text_primary":  "#1c140a",
        "text_secondary":"#5a4535",
        "text_muted":    "#9a8060",
        "text_inverse":  "#faf8f3",

        "border":        "rgba(196,160,90,0.25)",
        "border_subtle": "rgba(28,20,10,0.06)",
        "border_strong": "rgba(184,134,11,0.40)",

        "plotly_tpl":    "plotly_white",
        "particle_color":"#b8860b",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "◉ Carbon Reef": {
        "label":         "Eclipse Carbon",
        "mood":          "dark",
        "bg_primary":    "#060608",
        "bg_secondary":  "#0d0d10",
        "bg_card":       "#101018",
        "bg_elevated":   "#14141e",
        "bg_glass":      "rgba(34,211,238,0.05)",
        "bg_sidebar":    "linear-gradient(158deg,#060608 0%,#0d0d10 55%,#101018 100%)",

        "accent1":       "#22d3ee",   # glacier cyan
        "accent2":       "#818cf8",   # periwinkle
        "accent3":       "#4ade80",   # lime
        "accent4":       "#fb923c",   # burnt sienna
        "accent5":       "#e879f9",   # fuchsia

        "accent1_deep":  "#0891b2",
        "accent1_glow":  "rgba(34,211,238,0.18)",
        "accent1_border":"rgba(34,211,238,0.13)",
        "accent1_fill":  "rgba(34,211,238,0.06)",

        "text_primary":  "#e4f6ff",
        "text_secondary":"#7db8cc",
        "text_muted":    "#1e3d4a",
        "text_inverse":  "#e0f7ff",

        "border":        "rgba(34,211,238,0.11)",
        "border_subtle": "rgba(255,255,255,0.04)",
        "border_strong": "rgba(34,211,238,0.26)",

        "plotly_tpl":    "plotly_dark",
        "particle_color":"#22d3ee",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "◇ Onyx Ember": {
        "label":         "Volcanic Dark",
        "mood":          "dark",
        "bg_primary":    "#090604",
        "bg_secondary":  "#160e08",
        "bg_card":       "#1c1108",
        "bg_elevated":   "#22150a",
        "bg_glass":      "rgba(251,146,60,0.06)",
        "bg_sidebar":    "linear-gradient(158deg,#090604 0%,#160e08 55%,#1e0e05 100%)",

        "accent1":       "#fb923c",   # volcanic orange
        "accent2":       "#fbbf24",   # molten amber
        "accent3":       "#f43f5e",   # ember red
        "accent4":       "#a78bfa",   # smoke violet
        "accent5":       "#34d399",   # cool contrast

        "accent1_deep":  "#c2410c",
        "accent1_glow":  "rgba(251,146,60,0.20)",
        "accent1_border":"rgba(251,146,60,0.15)",
        "accent1_fill":  "rgba(251,146,60,0.07)",

        "text_primary":  "#fff3e8",
        "text_secondary":"#c4906a",
        "text_muted":    "#4a2a10",
        "text_inverse":  "#fff3e8",

        "border":        "rgba(251,146,60,0.13)",
        "border_subtle": "rgba(255,255,255,0.04)",
        "border_strong": "rgba(251,146,60,0.28)",

        "plotly_tpl":    "plotly_dark",
        "particle_color":"#fb923c",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "○ Marble Light": {
        "label":         "Porcelain Studio",
        "mood":          "light",
        "bg_primary":    "#f9f9f7",
        "bg_secondary":  "#f0eeea",
        "bg_card":       "#ffffff",
        "bg_elevated":   "#fdfdfb",
        "bg_glass":      "rgba(99,102,241,0.05)",
        "bg_sidebar":    "linear-gradient(158deg,#eeedf0 0%,#f9f9f7 55%,#e8e8f2 100%)",

        "accent1":       "#4f46e5",   # indigo ink
        "accent2":       "#0891b2",   # prussian blue
        "accent3":       "#059669",   # malachite
        "accent4":       "#d97706",   # umber gold
        "accent5":       "#be185d",   # raspberry

        "accent1_deep":  "#3730a3",
        "accent1_glow":  "rgba(79,70,229,0.12)",
        "accent1_border":"rgba(99,102,241,0.22)",
        "accent1_fill":  "rgba(99,102,241,0.06)",

        "text_primary":  "#18181b",
        "text_secondary":"#52525b",
        "text_muted":    "#a1a1aa",
        "text_inverse":  "#ffffff",

        "border":        "rgba(99,102,241,0.14)",
        "border_subtle": "rgba(0,0,0,0.05)",
        "border_strong": "rgba(79,70,229,0.28)",

        "plotly_tpl":    "plotly_white",
        "particle_color":"#4f46e5",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "▲ Jade Sanctum": {
        "label":         "Botanical Dark",
        "mood":          "dark",
        "bg_primary":    "#030a06",
        "bg_secondary":  "#071410",
        "bg_card":       "#0c1e16",
        "bg_elevated":   "#112618",
        "bg_glass":      "rgba(52,211,153,0.05)",
        "bg_sidebar":    "linear-gradient(158deg,#030a06 0%,#071410 55%,#052010 100%)",

        "accent1":       "#34d399",   # jade
        "accent2":       "#86efac",   # pale jade
        "accent3":       "#fbbf24",   # saffron
        "accent4":       "#60a5fa",   # sky
        "accent5":       "#c084fc",   # wisteria

        "accent1_deep":  "#059669",
        "accent1_glow":  "rgba(52,211,153,0.18)",
        "accent1_border":"rgba(52,211,153,0.13)",
        "accent1_fill":  "rgba(52,211,153,0.06)",

        "text_primary":  "#e0fef0",
        "text_secondary":"#6dbf95",
        "text_muted":    "#183825",
        "text_inverse":  "#e0fef0",

        "border":        "rgba(52,211,153,0.11)",
        "border_subtle": "rgba(255,255,255,0.04)",
        "border_strong": "rgba(52,211,153,0.26)",

        "plotly_tpl":    "plotly_dark",
        "particle_color":"#34d399",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },

    "▽ Rose Dust": {
        "label":         "Blossom Light",
        "mood":          "light",
        "bg_primary":    "#fdf6f8",
        "bg_secondary":  "#f8edf0",
        "bg_card":       "#ffffff",
        "bg_elevated":   "#fffafc",
        "bg_glass":      "rgba(219,39,119,0.05)",
        "bg_sidebar":    "linear-gradient(158deg,#f8edf0 0%,#fdf6f8 55%,#fce7ef 100%)",

        "accent1":       "#db2777",   # deep rose
        "accent2":       "#f472b6",   # blush
        "accent3":       "#7c3aed",   # plum
        "accent4":       "#0891b2",   # teal contrast
        "accent5":       "#d97706",   # warm amber

        "accent1_deep":  "#9d174d",
        "accent1_glow":  "rgba(219,39,119,0.13)",
        "accent1_border":"rgba(236,72,153,0.24)",
        "accent1_fill":  "rgba(219,39,119,0.06)",

        "text_primary":  "#1a0810",
        "text_secondary":"#6b3050",
        "text_muted":    "#c4809a",
        "text_inverse":  "#fdf6f8",

        "border":        "rgba(236,72,153,0.16)",
        "border_subtle": "rgba(26,8,16,0.05)",
        "border_strong": "rgba(219,39,119,0.30)",

        "plotly_tpl":    "plotly_white",
        "particle_color":"#db2777",
        "font_display":  "'Playfair Display', Georgia, serif",
        "font_body":     "'Outfit', system-ui, sans-serif",
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GOOGLE FONTS LOADER  (ek baar inject karo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  THEME SELECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THEME_KEYS = list(THEMES.keys())
DEFAULT_THEME = "◆ Noir Royale"

if "theme" not in st.session_state:
    st.session_state.theme = DEFAULT_THEME

st.sidebar.markdown("""
<p style="font-size:9px;letter-spacing:0.2em;text-transform:uppercase;
color:#6b7280;margin:0 0 6px;font-family:'Outfit',sans-serif;font-weight:500">
Dashboard Theme</p>""", unsafe_allow_html=True)

selected_theme = st.sidebar.selectbox(
    label="theme_picker",
    options=THEME_KEYS,
    index=THEME_KEYS.index(st.session_state.theme),
    format_func=lambda k: f"{k}  ·  {THEMES[k]['label']}",
    label_visibility="collapsed",
)

if selected_theme != st.session_state.theme:
    st.session_state.theme = selected_theme
    st.rerun()

T = THEMES[st.session_state.theme]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DERIVED GLOBALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCENT_COLORS   = [T["accent1"], T["accent2"], T["accent3"],
                   T["accent4"], T["accent5"], "#c084fc"]
PLOTLY_TEMPLATE = T["plotly_tpl"]
IS_DARK         = T["mood"] == "dark"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GLOBAL CSS INJECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(f"""
<style>
  /* ── Fonts ── */
  html, body, [class*="css"] {{
      font-family: {T['font_body']};
  }}
  h1, h2 {{
      font-family: {T['font_display']};
      font-weight: 700;
      letter-spacing: -0.02em;
      color: {T['text_primary']};
  }}
  h3, h4 {{
      font-family: {T['font_body']};
      font-size: 0.65rem;
      font-weight: 500;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: {T['text_muted']};
  }}

  /* ── Root surfaces ── */
  .stApp {{
      background: {T['bg_primary']};
      color: {T['text_primary']};
  }}
  .block-container {{
      padding-top: 2rem;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] > div:first-child {{
      background: {T['bg_sidebar']};
      border-right: 1px solid {T['border']};
  }}
  [data-testid="stSidebar"] * {{
      color: {T['text_secondary']} !important;
  }}

  /* ── Metric cards ── */
  [data-testid="stMetric"] {{
      background: {T['bg_card']};
      border: 0.5px solid {T['border']};
      border-radius: 16px;
      padding: 1.25rem 1.5rem;
      transition: border-color 0.2s ease, transform 0.2s ease;
  }}
  [data-testid="stMetric"]:hover {{
      border-color: {T['border_strong']};
      transform: translateY(-2px);
  }}
  [data-testid="stMetricLabel"] p {{
      font-size: 9px !important;
      letter-spacing: 0.2em !important;
      text-transform: uppercase !important;
      color: {T['text_muted']} !important;
      font-weight: 500 !important;
  }}
  [data-testid="stMetricValue"] {{
      font-family: {T['font_display']};
      font-size: 2rem !important;
      font-weight: 700 !important;
      color: {T['text_primary']} !important;
  }}
  [data-testid="stMetricDelta"] {{
      font-size: 12px !important;
      font-weight: 500 !important;
  }}

  /* ── Cards / expanders ── */
  [data-testid="stExpander"] {{
      background: {T['bg_card']};
      border: 0.5px solid {T['border']} !important;
      border-radius: 16px !important;
  }}

  /* ── Columns ── */
  div[data-testid="column"] > div {{
      background: {T['bg_card']};
      border: 0.5px solid {T['border']};
      border-radius: 16px;
      padding: 1.25rem 1.5rem;
  }}

  /* ── Inputs ── */
  .stSelectbox > div > div,
  .stTextInput > div > div > input,
  .stNumberInput > div > div > input {{
      background: {T['bg_secondary']} !important;
      border: 0.5px solid {T['border']} !important;
      border-radius: 10px !important;
      color: {T['text_primary']} !important;
  }}

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {{
      font-family: {T['font_body']};
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      background: {T['accent1_deep']};
      border: none;
      border-radius: 10px;
      color: {T['text_inverse']};
      padding: 0.6rem 1.5rem;
      transition: box-shadow 0.25s ease, transform 0.2s ease;
  }}
  .stButton > button[kind="primary"]:hover {{
      box-shadow: 0 0 24px {T['accent1_glow']};
      transform: translateY(-1px);
  }}

  /* ── Secondary button ── */
  .stButton > button[kind="secondary"] {{
      font-family: {T['font_body']};
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      background: transparent;
      border: 0.5px solid {T['accent1_border']};
      border-radius: 10px;
      color: {T['text_secondary']};
  }}
  .stButton > button[kind="secondary"]:hover {{
      background: {T['accent1_fill']};
      border-color: {T['border_strong']};
  }}

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {{
      background: {T['bg_secondary']};
      border-radius: 12px;
      padding: 4px;
      gap: 4px;
      border: 0.5px solid {T['border']};
  }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 8px;
      font-family: {T['font_body']};
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      color: {T['text_muted']};
  }}
  .stTabs [aria-selected="true"] {{
      background: {T['accent1_fill']};
      color: {T['accent1']};
  }}

  /* ── Dividers ── */
  hr {{
      border: none;
      border-top: 0.5px solid {T['border_subtle']};
      margin: 1.5rem 0;
  }}

  /* ── Dataframes ── */
  [data-testid="stDataFrame"] {{
      border: 0.5px solid {T['border']};
      border-radius: 12px;
      overflow: hidden;
  }}

  /* ── Progress bars ── */
  .stProgress > div > div {{
      background: {T['accent1']} !important;
      border-radius: 4px;
  }}
  .stProgress > div {{
      background: {T['accent1_fill']} !important;
      border-radius: 4px;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{width: 4px}}
  ::-webkit-scrollbar-track {{background: transparent}}
  ::-webkit-scrollbar-thumb {{
      background: {T['border_strong']};
      border-radius: 2px;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 🌌  BACKGROUND & CSS INJECTION
# ─────────────────────────────────────────────

PARTICLE_SVG = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%"
     style="position:fixed;top:0;left:0;z-index:-1;pointer-events:none;opacity:0.18;">
  <defs>
    <radialGradient id="glow" cx="50%" cy="50%" r="50%">
      <stop offset="0%"   stop-color="{T['accent1']}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{T['bg_primary']}" stop-opacity="0"/>
    </radialGradient>
    <filter id="blur"><feGaussianBlur stdDeviation="2"/></filter>
    <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
      <path d="M 60 0 L 0 0 0 60" fill="none"
            stroke="{T['accent1']}" stroke-width="0.4" opacity="0.4"/>
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#grid)"/>
  <circle cx="15%"  cy="25%" r="200" fill="{T['accent1']}" opacity="0.04" filter="url(#blur)"/>
  <circle cx="80%"  cy="60%" r="280" fill="{T['accent2']}" opacity="0.04" filter="url(#blur)"/>
  <circle cx="50%"  cy="90%" r="180" fill="{T['accent3']}" opacity="0.03" filter="url(#blur)"/>
  {''.join([
      f'<circle cx="{np.random.randint(2,98)}%" cy="{np.random.randint(2,98)}%"'
      f' r="{np.random.choice([1,1.5,2,2.5])}"'
      f' fill="{np.random.choice([T["accent1"],T["accent2"],T["accent3"]])}"'
      f' opacity="{np.random.uniform(0.3,0.7):.2f}">'
      f'<animate attributeName="opacity" values="0.1;0.8;0.1"'
      f' dur="{np.random.uniform(3,8):.1f}s" repeatCount="indefinite"/>'
      f'</circle>'
      for _ in range(60)
  ])}
  <line x1="-5%" y1="30%" x2="30%" y2="-5%"
        stroke="{T['accent1']}" stroke-width="0.5" opacity="0.2"/>
  <line x1="70%" y1="105%" x2="105%" y2="70%"
        stroke="{T['accent2']}" stroke-width="0.5" opacity="0.2"/>
  <line x1="0%" y1="70%" x2="40%" y2="105%"
        stroke="{T['accent3']}" stroke-width="0.5" opacity="0.15"/>
  <path d="M 30 10 L 10 10 L 10 30" fill="none"
        stroke="{T['accent1']}" stroke-width="2" opacity="0.5"/>
</svg>
"""

st.markdown(PARTICLE_SVG, unsafe_allow_html=True)

# ── Main CSS ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: {T['bg_primary']} !important;
    color: {T['text_primary']};
}}
#MainMenu, footer, header, [data-testid="manage-app-button"] {{ display: none !important; }}
.block-container {{ padding-top: 0.5rem !important; max-width: 100% !important; width: 100% !important; padding-left: 1rem !important; padding-right: 1rem !important; }}
div[data-testid="column"] {{ padding-left: 4px !important; padding-right: 4px !important; }}

section[data-testid="stSidebar"] {{
    background: {T['bg_sidebar']} !important;
    border-right: 1px solid {T['border']};
}}
section[data-testid="stSidebar"] * {{ color: {T['text_primary']} !important; }}
section[data-testid="stSidebar"] label {{ color: {T['text_muted']} !important; font-size:0.75rem; }}

.kpi-card {{
    background: {T['bg_card']};
    border: 1px solid {T['border']};
    border-radius: 20px;
    padding: 14px 8px 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
    min-height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    width: 100%;
}}
.kpi-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {T['accent1']}, {T['accent2']});
    border-radius: 20px 20px 0 0;
}}
.kpi-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 16px 40px {rgba(T['accent1'], 0.15)};
}}
.kpi-icon  {{ font-size: 1.4rem; margin-bottom: 4px; }}
.kpi-label {{ font-size: 0.72rem; color: {T['text_muted']}; letter-spacing:0.1em; text-transform:uppercase; }}
.kpi-value {{ font-family:'Syne',sans-serif; font-size:1.25rem; font-weight:800; color:{T['text_primary']}; line-height:1.2; word-break:break-word; overflow-wrap:anywhere; }}
.kpi-delta {{ font-size:0.78rem; margin-top:6px; font-family:'IBM Plex Mono',monospace; }}
.kpi-delta.up   {{ color: {T['accent3']}; }}
.kpi-delta.down {{ color: #f87171; }}
.kpi-sparkline  {{ margin-top: 8px; opacity: 0.7; }}

.alert-danger  {{ background:{T['bg_card']}; border-left:3px solid #ef4444; padding:11px 16px; border-radius:10px; color:#fca5a5; margin:5px 0; font-size:0.85rem; }}
.alert-warn    {{ background:{T['bg_card']}; border-left:3px solid #f59e0b; padding:11px 16px; border-radius:10px; color:#fcd34d; margin:5px 0; font-size:0.85rem; }}
.alert-success {{ background:{T['bg_card']}; border-left:3px solid {T['accent3']}; padding:11px 16px; border-radius:10px; color:{T['accent3']}; margin:5px 0; font-size:0.85rem; }}
.alert-info    {{ background:{T['bg_card']}; border-left:3px solid {T['accent2']}; padding:11px 16px; border-radius:10px; color:{T['accent2']}; margin:5px 0; font-size:0.85rem; }}

.section-title {{
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem; font-weight: 800;
    color: {T['text_primary']};
    border-left: 3px solid {T['accent1']};
    padding-left: 14px;
    margin: 28px 0 14px 0;
    display: flex; align-items: center; gap: 8px;
}}

.metric-badge {{
    display: inline-block;
    background: {rgba(T['accent1'], 0.13)};
    color: {T['accent1']};
    border: 1px solid {rgba(T['accent1'], 0.27)};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}}

.score-bar-wrap {{
    background: {T['border']};
    border-radius: 8px; height: 8px; overflow: hidden; margin-top: 4px;
}}
.score-bar-fill {{
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg, {T['accent1']}, {T['accent2']});
    transition: width 1s ease;
}}

.lb-row {{
    display: flex; align-items: center; gap: 14px;
    background: {T['bg_card']};
    border: 1px solid {T['border']};
    border-radius: 12px;
    padding: 12px 18px;
    margin: 6px 0;
    transition: all 0.2s;
}}
.lb-row:hover {{ border-color: {T['accent1']}; transform: translateX(4px); }}
.lb-rank {{ font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:{T['accent1']}; min-width:32px; }}
.lb-name {{ flex:1; font-weight:500; color:{T['text_primary']}; }}
.lb-val  {{ font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:{T['accent3']}; }}

/* ✅ FIXED: File uploader drag-and-drop CSS */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
    background: {T['bg_secondary']} !important;
    border: 2px dashed {T['accent1']} !important;
    border-radius: 12px !important;
    padding: 20px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {{
    background: {rgba(T['accent1'], 0.06)} !important;
    border-color: {T['accent2']} !important;
    box-shadow: 0 0 14px {rgba(T['accent1'], 0.18)} !important;
}}
/* ✅ CRITICAL: pointer-events must be auto so drag-drop works */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
    pointer-events: auto !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {{
    pointer-events: auto !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
    background: {T['accent1']} !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 10px 16px !important;
    margin-top: 8px !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease !important;
    width: 100% !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {{
    opacity: 0.88 !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] p {{
    color: {T['text_muted']} !important;
    font-size: 0.82rem !important;
    text-align: center !important;
}}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small {{
    color: {T['text_muted']} !important;
    font-size: 0.75rem !important;
}}

button[data-baseweb="tab"] {{
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}}

.stPlotlyChart {{
    border-radius: 16px !important;
    overflow: hidden !important;
    background: {T['bg_card']} !important;
    border: 1px solid {T['border']} !important;
}}

.rfm-cell-high   {{ background: {rgba(T['accent3'], 0.13)} !important; color:{T['accent3']} !important; }}
.rfm-cell-medium {{ background: {rgba(T['accent4'], 0.13)} !important; color:{T['accent4']} !important; }}
.rfm-cell-low    {{ background: rgba(239,68,68,0.13) !important; color:#f87171 !important; }}

.discount-result {{
    background: {T['bg_card']};
    border: 1px solid {T['accent1']};
    border-radius: 16px; padding: 20px;
    text-align: center;
}}
.discount-result .big {{ font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:{T['accent1']}; }}

::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {T['bg_primary']}; }}
::-webkit-scrollbar-thumb {{ background: {rgba(T['accent1'], 0.33)}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {rgba(T['accent1'], 0.60)}; }}

.refresh-badge {{
    display:inline-flex; align-items:center; gap:6px;
    background:{rgba(T['accent3'], 0.13)}; border:1px solid {rgba(T['accent3'], 0.27)};
    color:{T['accent3']}; border-radius:20px; padding:4px 14px;
    font-size:0.78rem; font-family:'IBM Plex Mono',monospace; font-weight:600;
}}
.pulse-dot {{
    width:7px; height:7px; border-radius:50%;
    background:{T['accent3']};
    animation: pulse 1.4s infinite;
}}
@keyframes pulse {{
    0%,100% {{ opacity:1; transform:scale(1); }}
    50%      {{ opacity:0.3; transform:scale(1.5); }}
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 🔑  GEMINI SETUP
# ─────────────────────────────────────────────

gemini_model = None
GEN_API_KEY  = os.getenv("GOOGLE_API_KEY")
if GEN_API_KEY:
    try:
        genai.configure(api_key=GEN_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.sidebar.warning(f"Gemini init failed: {e}")


 # ─────────────────────────────────────────────
# 📂  DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data(path_or_file) -> pd.DataFrame:
    if hasattr(path_or_file, "name"):
        fname = path_or_file.name.lower()
    else:
        fname = str(path_or_file).lower()

    try:
        if fname.endswith(".csv"):
            df = pd.read_csv(path_or_file, encoding="latin1")
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path_or_file, engine="openpyxl" if fname.endswith(".xlsx") else "xlrd")
        elif fname.endswith(".json"):
            df = pd.read_json(path_or_file)
        else:
            df = pd.read_csv(path_or_file, encoding="latin1")
    except Exception as e:
        st.error(f"❌ File load error: {e}")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Order Date" in df.columns:
        df = df.dropna(subset=["Order Date"])
        df["Year"]    = df["Order Date"].dt.year
        df["Month"]   = df["Order Date"].dt.month
        df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)
        df["Week"]    = df["Order Date"].dt.isocalendar().week.astype(int)

    if {"Sales", "Profit"}.issubset(df.columns):
        df = df.dropna(subset=["Sales", "Profit"])
        df["Profit Margin"]   = df["Profit"] / df["Sales"].replace({0: np.nan})
        df["Profit Margin %"] = df["Profit Margin"] * 100

    df = df.drop_duplicates()

    if "Order Date" in df.columns and "Customer ID" in df.columns and "Sales" in df.columns:
        max_date = df["Order Date"].max()
        rfm_r    = df.groupby("Customer ID")["Order Date"].max().apply(lambda x: (max_date - x).days)
        rfm_f    = df.groupby("Customer ID")["Order ID"].nunique() if "Order ID" in df.columns else df.groupby("Customer ID").size()
        rfm_m    = df.groupby("Customer ID")["Sales"].sum()
        rfm_df   = pd.DataFrame({"Recency": rfm_r, "Frequency": rfm_f, "Monetary": rfm_m})
        rfm_df["R_Score"]   = pd.qcut(rfm_df["Recency"],   4, labels=[4,3,2,1]).astype(int)
        rfm_df["F_Score"]   = pd.qcut(rfm_df["Frequency"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
        rfm_df["M_Score"]   = pd.qcut(rfm_df["Monetary"],  4, labels=[1,2,3,4]).astype(int)
        rfm_df["RFM_Score"] = rfm_df["R_Score"] + rfm_df["F_Score"] + rfm_df["M_Score"]
        df = df.merge(rfm_df[["RFM_Score","R_Score","F_Score","M_Score"]].reset_index(), on="Customer ID", how="left")

    return df

#  ─────────────────────────────────────────────
# ── Sidebar: data source
# ─────────────────────────────────────────────

st.sidebar.markdown(f"""
<div style="background:{T['bg_secondary']};border:1px solid {rgba(T['accent1'], 0.27)};
            border-radius:12px;padding:14px 16px;margin-bottom:12px;">
  <div style="color:{T['accent1']};font-size:0.72rem;letter-spacing:0.12em;
              text-transform:uppercase;font-weight:700;margin-bottom:6px;">
    📂 Data Source
  </div>
  <div style="color:{T['text_primary']};font-size:0.85rem;line-height:1.5;">
    🖱️ <b>Drag & drop</b> your file below, or click <b>Browse files</b><br>
    <span style="color:{T['text_muted']};font-size:0.78rem;">✅ CSV &nbsp;|&nbsp; Excel (.xlsx/.xls) &nbsp;|&nbsp; JSON</span>
  </div>
</div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader(
    label="📁 Upload Data File",
    type=["csv", "xlsx", "xls", "json"],
    label_visibility="collapsed",
    accept_multiple_files=False,
    help="CSV, Excel (.xlsx/.xls), ya JSON file upload karein"
)

DEFAULT_PATH = r"C:\Users\shiva\OneDrive\Desktop\AI_Predictive_Analytics\data\superstore_sales.csv"

if uploaded:
    df = load_data(uploaded)
    if df.empty:
        st.error("❌ File load hua but data empty hai. File check karein.")
        st.stop()
    st.sidebar.markdown(f'<div class="alert-success">✅ Loaded: <b>{uploaded.name}</b> ({len(df):,} rows)</div>', unsafe_allow_html=True)
elif os.path.exists(DEFAULT_PATH):
    df = load_data(DEFAULT_PATH)
    st.sidebar.markdown('<div class="alert-info">📁 Using local dataset</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="alert-danger">❌ No data found — upload a file above</div>', unsafe_allow_html=True)
    st.error("❌ No data found. Please upload CSV, Excel, or JSON file from the sidebar.")
    st.stop()
# ─────────────────────────────────────────────
# 🔍  SIDEBAR FILTERS
# ─────────────────────────────────────────────

st.sidebar.markdown("## 🔍 Filters")

def _opts(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

region_opts   = _opts("Region")
category_opts = _opts("Category of Goods")
year_opts     = sorted(df["Year"].dropna().astype(int).unique().tolist()) if "Year" in df.columns else []

region_sel   = st.sidebar.multiselect("Region",   region_opts,   default=region_opts)
category_sel = st.sidebar.multiselect("Category", category_opts, default=category_opts)
year_sel     = st.sidebar.multiselect("Year",     year_opts,     default=year_opts)
search_term  = st.sidebar.text_input("🔍 Product name contains")

# ── Auto-refresh ──
st.sidebar.markdown("---")
auto_refresh = st.sidebar.toggle("🔁 Auto-refresh (30s)", value=False)
if auto_refresh:
    st.sidebar.markdown("""
    <div class="refresh-badge">
        <div class="pulse-dot"></div> LIVE
    </div>
    """, unsafe_allow_html=True)
    time.sleep(30)
    st.rerun()

# ── Apply filters ──
filtered = df.copy()
if region_sel   and "Region"            in df.columns: filtered = filtered[filtered["Region"].isin(region_sel)]
if category_sel and "Category of Goods" in df.columns: filtered = filtered[filtered["Category of Goods"].isin(category_sel)]
if year_sel     and "Year"              in df.columns: filtered = filtered[filtered["Year"].isin(year_sel)]
if search_term  and "Product Name"      in df.columns:
    filtered = filtered[filtered["Product Name"].str.contains(search_term, case=False, na=False)]

if filtered.empty:
    st.warning("⚠️ No data matches current filters.")
    st.stop()

st.sidebar.markdown(f"""
<div style="background:{T['bg_secondary']};border:1px solid {T['border']};
            border-radius:10px;padding:12px 14px;margin-top:14px;font-size:0.82rem;">
  <div style="color:{T['accent1']};font-weight:700;margin-bottom:6px;">📊 Filter Summary</div>
  <div>📦 Rows: <b style="color:{T['accent3']}">{len(filtered):,}</b></div>
  <div>📅 Years: <b style="color:{T['accent3']}">{', '.join(map(str,year_sel)) if year_sel else 'All'}</b></div>
  <div>🗺️ Regions: <b style="color:{T['accent3']}">{len(region_sel)} selected</b></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 🧮  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def build_monthly(df_in: pd.DataFrame) -> pd.DataFrame:
    if "Order Date" not in df_in.columns:
        return pd.DataFrame(columns=["Order Date", "Sales", "Profit"])
    m = (df_in.groupby(df_in["Order Date"].dt.to_period("M"))[["Sales", "Profit"]]
         .sum().reset_index())
    m["Order Date"] = m["Order Date"].dt.to_timestamp()
    m["Month"] = m["Order Date"].dt.month
    m["Year"]  = m["Order Date"].dt.year
    return m

monthly = build_monthly(filtered)


def rolling_avg(monthly_df: pd.DataFrame, window: int) -> pd.Series:
    return monthly_df["Sales"].rolling(window, min_periods=1).mean()


def compute_alerts(df_in: pd.DataFrame) -> list[dict]:
    alerts = []
    if "Profit Margin %" in df_in.columns:
        avg_m = df_in["Profit Margin %"].mean()
        if avg_m < 5:
            alerts.append({"level":"danger",  "msg":f"🔴 Critical: Avg margin only {avg_m:.1f}%! Act now."})
        elif avg_m < 12:
            alerts.append({"level":"warn",    "msg":f"🟡 Warning: Avg margin {avg_m:.1f}% — below 15% threshold."})
        else:
            alerts.append({"level":"success", "msg":f"🟢 Healthy margin: {avg_m:.1f}%"})
    if "Discount" in df_in.columns:
        high_disc = (df_in["Discount"] > 0.3).mean() * 100
        if high_disc > 25:
            alerts.append({"level":"warn","msg":f"🟡 {high_disc:.0f}% orders have >30% discount — margin risk."})
    if "Profit" in df_in.columns:
        loss_pct = (df_in["Profit"] < 0).mean() * 100
        if loss_pct > 15:
            alerts.append({"level":"danger","msg":f"🔴 {loss_pct:.1f}% orders are loss-making!"})
        elif loss_pct > 5:
            alerts.append({"level":"warn",  "msg":f"🟡 {loss_pct:.1f}% orders are loss-making."})
    if "Sub-Category" in df_in.columns and "Profit" in df_in.columns:
        worst_val = df_in.groupby("Sub-Category")["Profit"].sum().min()
        worst     = df_in.groupby("Sub-Category")["Profit"].sum().idxmin()
        if worst_val < 0:
            alerts.append({"level":"warn","msg":f"🟡 '{worst}' sub-category is loss-making (₹{worst_val:,.0f})."})
    if not monthly.empty and len(monthly) >= 2:
        last_two = monthly.tail(2)["Sales"].tolist()
        mom = (last_two[-1] - last_two[-2]) / max(last_two[-2], 1) * 100
        if mom < -10:
            alerts.append({"level":"danger", "msg":f"🔴 Sales dropped {abs(mom):.1f}% vs last month!"})
        elif mom > 15:
            alerts.append({"level":"success","msg":f"🟢 Sales grew {mom:.1f}% vs last month!"})
    return alerts

alerts = compute_alerts(filtered)


def train_models(monthly_df: pd.DataFrame):
    if len(monthly_df) < 6:
        return pd.DataFrame(), {}, None, None, None
    X = monthly_df[["Month","Year"]]
    y = monthly_df["Sales"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_zoo = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, random_state=42),
        "XGBoost":            XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0),
    }
    results, fitted = [], {}
    for name, m in model_zoo.items():
        try:
            m.fit(X_train, y_train)
            yp   = m.predict(X_test)
            r2   = r2_score(y_test, yp)
            rmse = np.sqrt(mean_squared_error(y_test, yp))
            mae  = np.mean(np.abs(y_test.values - yp))
            results.append({"Model":name,"R² Score":round(r2,4),"RMSE":round(rmse,2),"MAE":round(mae,2)})
            fitted[name] = m
        except Exception:
            results.append({"Model":name,"R² Score":np.nan,"RMSE":np.nan,"MAE":np.nan})
    res_df    = pd.DataFrame(results).sort_values("R² Score", ascending=False)
    best_name = res_df.iloc[0]["Model"] if not res_df.empty else None
    best_m    = fitted.get(best_name)
    if best_m:
        best_m.fit(X, y)
    return res_df, fitted, best_name, best_m, (X_train, X_test, y_train, y_test)


def forecast_next_n(model, monthly_df, n=6):
    latest_year  = int(monthly_df["Year"].max())
    latest_month = int(monthly_df[monthly_df["Year"]==latest_year]["Month"].max())
    rows = []
    for i in range(1, n+1):
        raw_m = latest_month + i
        nm = (raw_m-1) % 12 + 1
        ny = latest_year + (raw_m-1) // 12
        val = float(model.predict([[nm, ny]])[0])
        rows.append({"Month-Year": f"{nm:02d}/{ny}", "Predicted Sales": max(val, 0)})
    return pd.DataFrame(rows)


def arima_forecast(monthly_df, steps=6):
    if not ARIMA_AVAILABLE or len(monthly_df) < 12:
        return None
    try:
        m    = ARIMA(monthly_df["Sales"].values, order=(2,1,2)).fit()
        pred = m.forecast(steps=steps)
        ci   = m.get_forecast(steps=steps).conf_int()
        last = monthly_df["Order Date"].max()
        dates = [last + pd.DateOffset(months=i) for i in range(1, steps+1)]
        return pd.DataFrame({
            "Date":          dates,
            "ARIMA Forecast": np.maximum(pred, 0),
            "Lower CI":       np.maximum(ci.iloc[:,0].values, 0),
            "Upper CI":       np.maximum(ci.iloc[:,1].values, 0),
        })
    except Exception:
        return None


def export_excel(df_in):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df_in.to_excel(writer, sheet_name="Filtered Data", index=False)
        wb  = writer.book
        ws  = writer.sheets["Filtered Data"]
        hfmt = wb.add_format({"bold":True,"bg_color":"#1e293b","font_color":"#f1f5f9","border":1})
        for ci, cn in enumerate(df_in.columns):
            ws.write(0, ci, cn, hfmt)
            ws.set_column(ci, ci, max(len(str(cn))+4, 14))
        ws2 = wb.add_worksheet("KPI Summary")
        kpis = [
            ("Total Sales",  df_in["Sales"].sum()  if "Sales"  in df_in.columns else 0),
            ("Total Profit", df_in["Profit"].sum() if "Profit" in df_in.columns else 0),
            ("Avg Margin %", df_in["Profit Margin %"].mean() if "Profit Margin %" in df_in.columns else 0),
            ("Total Orders", df_in["Order ID"].nunique() if "Order ID" in df_in.columns else len(df_in)),
        ]
        ws2.write(0,0,"Metric",hfmt); ws2.write(0,1,"Value",hfmt)
        vfmt = wb.add_format({"num_format":"#,##0.00"})
        for i,(k,v) in enumerate(kpis,1):
            ws2.write(i,0,k); ws2.write(i,1,v,vfmt)
        if not monthly.empty:
            monthly.to_excel(writer, sheet_name="Monthly Trend", index=False)
    return buf.getvalue()


def build_summary(df_in, max_prod=8):
    lines = []
    if "Sales"  in df_in.columns: lines.append(f"Total Sales: {df_in['Sales'].sum():,.2f}")
    if "Profit" in df_in.columns: lines.append(f"Total Profit: {df_in['Profit'].sum():,.2f}")
    if "Profit Margin %" in df_in.columns:
        lines.append(f"Avg Profit Margin: {df_in['Profit Margin %'].mean():.2f}%")
    if "Region" in df_in.columns:
        lines.append(f"Top Region: {df_in.groupby('Region')['Sales'].sum().idxmax()}")
    if "Category of Goods" in df_in.columns:
        tc = df_in.groupby("Category of Goods")["Sales"].sum().nlargest(5)
        lines.append("Top Categories: "+", ".join(f"{i}({int(v)})" for i,v in tc.items()))
    if "Product Name" in df_in.columns and "Profit" in df_in.columns:
        tp = df_in.groupby("Product Name")["Profit"].sum().nlargest(max_prod)
        lines.append("Top Products: "+"; ".join(f"{i}({int(v)})" for i,v in tp.items()))
    if "Order Date" in df_in.columns:
        lines.append(f"Date Range: {df_in['Order Date'].min().date()} to {df_in['Order Date'].max().date()}")
    for a in alerts:
        lines.append(f"Alert: {a['msg']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 🏠  HEADER
# ─────────────────────────────────────────────

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, {T['bg_secondary']} 0%, {T['bg_card']} 60%, {rgba(T['accent1'], 0.09)} 100%);
    border-radius: 24px; padding: 28px 40px; margin-bottom: 20px;
    border: 1px solid {T['border']};
    position: relative; overflow: hidden;
">
  <svg style="position:absolute;top:0;right:0;width:200px;height:120px;opacity:0.15" viewBox="0 0 200 120">
    <polyline points="200,0 200,120 80,120" fill="none" stroke="{T['accent2']}" stroke-width="1"/>
    <polyline points="200,40 160,40 160,120" fill="none" stroke="{T['accent1']}" stroke-width="0.7"/>
  </svg>
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:2.6rem;">🧠</div>
    <div>
      <h1 style="font-family:'Syne',sans-serif;color:{T['text_primary']};margin:0;font-size:2rem;font-weight:800;line-height:1.1;">
        AI Superstore Analytics
        <span style="color:{T['accent1']};font-size:1rem;font-weight:600;
                     background:{rgba(T['accent1'], 0.13)};border:1px solid {rgba(T['accent1'], 0.27)};
                     padding:2px 10px;border-radius:12px;vertical-align:middle;margin-left:8px;">v3</span>
      </h1>
      <p style="color:{T['text_muted']};margin:6px 0 0;font-size:0.92rem;">
        Advanced ML · RFM Scoring · YoY Analysis · Discount Analyser · Animated Insights
      </p>
    </div>
    <div style="margin-left:auto;text-align:right;font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:{T['text_muted']};">
      <div>{datetime.datetime.now().strftime('%d %b %Y')}</div>
      <div style="color:{T['accent3']};margin-top:2px;">{len(filtered):,} records</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 🚨  ALERTS BANNER
# ─────────────────────────────────────────────

st.markdown('<div class="section-title">🔔 Smart Business Alerts</div>', unsafe_allow_html=True)
col_a1, col_a2 = st.columns(2)
for i, a in enumerate(alerts):
    with (col_a1 if i % 2 == 0 else col_a2):
        st.markdown(f'<div class="alert-{a["level"]}">{a["msg"]}</div>', unsafe_allow_html=True)
if not alerts:
    st.markdown('<div class="alert-success">✅ No critical issues detected.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 📊  KPI CARDS WITH SPARKLINES
# ─────────────────────────────────────────────

st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)


def fmt_val(num, prefix="₹", suffix=""):
    try:
        n = float(num)
    except (TypeError, ValueError):
        return str(num)
    if abs(n) >= 1_000_000:
        return f"{prefix}{n/1_000_000:.1f}M{suffix}"
    elif abs(n) >= 1_000:
        return f"{prefix}{n/1_000:.1f}K{suffix}"
    else:
        return f"{prefix}{n:.1f}{suffix}"


total_sales   = filtered["Sales"].sum()           if "Sales"           in filtered.columns else 0
total_profit  = filtered["Profit"].sum()          if "Profit"          in filtered.columns else 0
avg_margin    = filtered["Profit Margin %"].mean() if "Profit Margin %" in filtered.columns else 0
total_orders  = filtered["Order ID"].nunique()    if "Order ID"        in filtered.columns else len(filtered)
total_qty     = filtered["Quantity"].sum()         if "Quantity"        in filtered.columns else 0
avg_discount  = filtered["Discount"].mean()*100   if "Discount"        in filtered.columns else 0

delta_sales  = ((total_sales  / df["Sales"].sum()  - 1)*100) if df["Sales"].sum()  else 0
delta_profit = ((total_profit / df["Profit"].sum() - 1)*100) if df["Profit"].sum() else 0


def make_sparkline(values, color, width=90, height=28):
    if len(values) < 2:
        return ""
    vals   = np.array(values, dtype=float)
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return ""
    normed = (vals - mn) / (mx - mn) * (height - 4) + 2
    step   = width / (len(vals) - 1)
    pts    = " ".join(f"{i*step:.1f},{height - n:.1f}" for i, n in enumerate(normed))
    return (f'<svg width="{width}" height="{height}" style="display:block;margin:auto;">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
            f'</svg>')


monthly_sales_spark = monthly["Sales"].tail(12).tolist()  if not monthly.empty else []
monthly_prof_spark  = monthly["Profit"].tail(12).tolist() if not monthly.empty else []

c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (c1, "💰", "Total Sales",  fmt_val(total_sales),                          f"{delta_sales:+.1f}% vs all",  "up" if delta_sales>=0  else "down", monthly_sales_spark, T["accent1"]),
    (c2, "📈", "Total Profit", fmt_val(total_profit),                        f"{delta_profit:+.1f}% vs all", "up" if delta_profit>=0 else "down", monthly_prof_spark,  T["accent3"]),
    (c3, "💹", "Avg Margin",   f"{avg_margin:.1f}%",                          "Profit / Sales",               "up" if avg_margin>10   else "down", [],                  T["accent2"]),
    (c4, "📦", "Orders",       fmt_val(total_orders, prefix="", suffix=""),   "Unique orders",                "up",                                [],                  T["accent4"]),
    (c5, "🛒", "Units Sold",   fmt_val(total_qty,    prefix="", suffix=""),   "Total quantity",               "up",                                [],                  T["accent5"]),
    (c6, "🏷️", "Avg Discount", f"{avg_discount:.1f}%",                        "Mean discount %",              "up" if avg_discount<20 else "down", [],                  "#a78bfa"),
]
for col, icon, label, val, delta_txt, direction, sparkdata, clr in kpis:
    spark_svg = make_sparkline(sparkdata, clr) if sparkdata else ""
    with col:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta {direction}">{delta_txt}</div>
          <div class="kpi-sparkline">{spark_svg}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 📑  TABBED LAYOUT
# ─────────────────────────────────────────────

tabs = st.tabs([
    "📈 Trends",
    "🗺️ Geo & Category",
    "🤖 ML & Forecast",
    "🎯 RFM & Customers",
    "🏅 Leaderboard",
    "🧩 Discount Lab",
    "🤖 Gemini & Claude",
    "📤 Export",
])


# ══════════════════════════════════════════
# TAB 1 — TRENDS
# ══════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">📈 Monthly Sales & Profit</div>', unsafe_allow_html=True)

    if not monthly.empty:
        ma_window    = st.select_slider("Rolling Average Window", options=[3, 6, 12], value=3)
        monthly_plot = monthly.copy()
        monthly_plot[f"{ma_window}M MA"] = rolling_avg(monthly_plot, ma_window)

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_plot["Order Date"], y=monthly_plot["Sales"],
            name="Sales", mode="lines+markers",
            line={"color": T["accent1"], "width": 2},
            marker={"size": 5}
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_plot["Order Date"], y=monthly_plot["Profit"],
            name="Profit", mode="lines+markers",
            line={"color": T["accent3"], "width": 2},
            marker={"size": 5}
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_plot["Order Date"], y=monthly_plot[f"{ma_window}M MA"],
            name=f"{ma_window}M Moving Avg",
            mode="lines", line={"color": T["accent4"], "width": 2, "dash": "dot"}
        ))
        fig_trend.update_layout(
            template=PLOTLY_TEMPLATE, hovermode="x unified",
            legend_title_text="Metric", title="Monthly Sales · Profit · Rolling Average"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('<div class="section-title">📅 Year-over-Year Sales Comparison</div>', unsafe_allow_html=True)
    if "Year" in filtered.columns and "Month" in filtered.columns:
        yoy = filtered.groupby(["Year","Month"])["Sales"].sum().reset_index()
        fig_yoy = px.line(
            yoy, x="Month", y="Sales", color="Year",
            markers=True, template=PLOTLY_TEMPLATE,
            color_discrete_sequence=ACCENT_COLORS,
            title="YoY Monthly Sales",
            labels={"Month":"Month","Sales":"Sales (₹)"}
        )
        fig_yoy.update_xaxes(
            tickmode="array", tickvals=list(range(1,13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        )
        st.plotly_chart(fig_yoy, use_container_width=True)

    st.markdown('<div class="section-title">📊 Quarterly Performance</div>', unsafe_allow_html=True)
    if "Quarter" in filtered.columns:
        q_agg = filtered.groupby("Quarter")[["Sales","Profit"]].sum().reset_index()
        fig_q = px.bar(q_agg, x="Quarter", y=["Sales","Profit"],
                       barmode="group", template=PLOTLY_TEMPLATE,
                       color_discrete_sequence=ACCENT_COLORS,
                       title="Quarterly Sales vs Profit")
        st.plotly_chart(fig_q, use_container_width=True)

    st.markdown('<div class="section-title">🌊 Profit Waterfall (Last 12 Months)</div>', unsafe_allow_html=True)
    if not monthly.empty:
        wf = monthly.tail(12).copy()
        fig_wf = go.Figure(go.Waterfall(
            name="Profit", orientation="v",
            x=wf["Order Date"].dt.strftime("%b %y"),
            y=wf["Profit"].tolist(),
            connector={"line":{"color": T["border"]}},
            increasing={"marker":{"color": T["accent3"]}},
            decreasing={"marker":{"color": "#f87171"}},
            totals   ={"marker":{"color": T["accent1"]}},
        ))
        fig_wf.update_layout(template=PLOTLY_TEMPLATE, title="Monthly Profit Waterfall")
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown('<div class="section-title">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    corr_cols = [c for c in ["Sales","Profit","Discount","Quantity","Profit Margin %"] if c in filtered.columns]
    if corr_cols:
        corr  = filtered[corr_cols].corr()
        fig_h, ax = plt.subplots(figsize=(7,4))
        bg = T["bg_card"]
        fig_h.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax,
                    linewidths=0.5, annot_kws={"color":"white"})
        ax.tick_params(colors=T["text_primary"])
        st.pyplot(fig_h)


# ══════════════════════════════════════════
# TAB 2 — GEO & CATEGORY
# ══════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">🗺️ Sales by State</div>', unsafe_allow_html=True)
    if "State" in filtered.columns:
        state_sales = filtered.groupby("State")[["Sales","Profit"]].sum().reset_index()
        fig_map = px.choropleth(
            state_sales, locations="State", locationmode="USA-states",
            color="Sales", hover_name="State", hover_data={"Profit":True},
            color_continuous_scale="Blues", scope="usa",
            template=PLOTLY_TEMPLATE, title="Sales by US State"
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("'State' column not found.")

    st.markdown('<div class="section-title">☀️ Sunburst: Region → Category → Sub-Category</div>', unsafe_allow_html=True)
    sb_path = [c for c in ["Region","Category of Goods","Sub-Category"] if c in filtered.columns]
    if sb_path and "Sales" in filtered.columns:
        fig_sun = px.sunburst(
            filtered, path=sb_path, values="Sales",
            color="Sales", color_continuous_scale="Blues",
            template=PLOTLY_TEMPLATE, title="Sales Hierarchy"
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    st.markdown('<div class="section-title">🗂️ Treemap: Profit by Sub-Category</div>', unsafe_allow_html=True)
    if sb_path and "Profit" in filtered.columns:
        td = filtered.groupby(sb_path)["Profit"].sum().reset_index()
        fig_tree = px.treemap(td, path=sb_path, values="Profit",
                              color="Profit", color_continuous_scale="RdYlGn",
                              template=PLOTLY_TEMPLATE, title="Profit Treemap")
        st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown('<div class="section-title">🔻 Top Sub-Categories Funnel</div>', unsafe_allow_html=True)
    if "Sub-Category" in filtered.columns:
        sc_sales = filtered.groupby("Sub-Category")["Sales"].sum().nlargest(10).reset_index()
        fig_fun  = px.funnel(sc_sales, y="Sub-Category", x="Sales",
                             template=PLOTLY_TEMPLATE, title="Top 10 Sub-Categories",
                             color_discrete_sequence=ACCENT_COLORS)
        st.plotly_chart(fig_fun, use_container_width=True)

    st.markdown('<div class="section-title">🏆 Top 10 Profitable Products</div>', unsafe_allow_html=True)
    if "Product Name" in filtered.columns:
        prod = filtered.groupby("Product Name")["Profit"].sum().nlargest(10).reset_index()
        fig_p = px.bar(prod, x="Profit", y="Product Name", orientation="h",
                       color="Profit", color_continuous_scale="Greens",
                       template=PLOTLY_TEMPLATE, title="Top 10 Products by Profit")
        fig_p.update_layout(yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════
# TAB 3 — ML & FORECASTING
# ══════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">🤖 Model Comparison</div>', unsafe_allow_html=True)
    with st.spinner("Training 5 ML models…"):
        res_df, fitted_models, best_name, best_model, split_data = train_models(monthly)

    if not res_df.empty:
        st.dataframe(
            res_df.style
                .background_gradient(cmap="Greens",  subset=["R² Score"])
                .background_gradient(cmap="Reds_r",  subset=["RMSE","MAE"])
                .format({"R² Score":"{:.4f}","RMSE":"{:,.0f}","MAE":"{:,.0f}"}),
            use_container_width=True
        )
        st.markdown(f"""
        <div class="alert-success">
            🏆 Best model: <b>{best_name}</b>
            &nbsp;|&nbsp; R² = {res_df.iloc[0]['R² Score']:.4f}
            &nbsp;|&nbsp; RMSE = ₹{res_df.iloc[0]['RMSE']:,.0f}
        </div>""", unsafe_allow_html=True)

        X_train, X_test, y_train, y_test = split_data
        y_pred_best = best_model.predict(X_test)
        compare_df  = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_best})
        fig_comp = px.scatter(compare_df, x="Actual", y="Predicted",
                              trendline="ols", template=PLOTLY_TEMPLATE,
                              color_discrete_sequence=[T["accent1"]],
                              title=f"Actual vs Predicted — {best_name}")
        st.plotly_chart(fig_comp, use_container_width=True)

        if best_name in ["Random Forest","Gradient Boosting","XGBoost"]:
            st.markdown('<div class="section-title">📌 Feature Importance</div>', unsafe_allow_html=True)
            fi_df = pd.DataFrame({"Feature":["Month","Year"],
                                  "Importance": best_model.feature_importances_})
            fig_fi = px.bar(fi_df, x="Feature", y="Importance",
                            template=PLOTLY_TEMPLATE, color="Importance",
                            color_continuous_scale="Blues", title="Feature Importance")
            st.plotly_chart(fig_fi, use_container_width=True)

    n_months = st.slider("Forecast horizon (months)", 3, 12, 6, key="fc_slider")

    st.markdown('<div class="section-title">🔮 ML Forecast</div>', unsafe_allow_html=True)
    if best_model and not monthly.empty:
        fc_df = forecast_next_n(best_model, monthly, n=n_months)
        fig_fc = px.bar(fc_df, x="Month-Year", y="Predicted Sales",
                        text_auto=".2s", template=PLOTLY_TEMPLATE,
                        color="Predicted Sales", color_continuous_scale="Blues",
                        title=f"{best_name} — Next {n_months}-Month Forecast")
        st.plotly_chart(fig_fc, use_container_width=True)

    # ── ARIMA ──
    st.markdown('<div class="section-title">📉 ARIMA with Confidence Intervals</div>', unsafe_allow_html=True)
    if ARIMA_AVAILABLE:
        arima_df = arima_forecast(monthly, steps=n_months)
        if arima_df is not None:
            fig_ar = go.Figure()
            fig_ar.add_trace(go.Scatter(
                x=monthly["Order Date"], y=monthly["Sales"],
                name="Historical", line={"color": T["accent1"]}
            ))
            fig_ar.add_trace(go.Scatter(
                x=arima_df["Date"], y=arima_df["ARIMA Forecast"],
                name="ARIMA Forecast",
                line={"color": T["accent3"], "dash": "dash"}
            ))
            fig_ar.add_trace(go.Scatter(
                x=pd.concat([arima_df["Date"], arima_df["Date"][::-1]]),
                y=pd.concat([arima_df["Upper CI"], arima_df["Lower CI"][::-1]]),
                fill="toself",
                fillcolor=rgba(T["accent3"], 0.13),
                line={"color": "rgba(0,0,0,0)"},
                name="95% CI"
            ))
            fig_ar.update_layout(template=PLOTLY_TEMPLATE, title="ARIMA Forecast with 95% CI")
            st.plotly_chart(fig_ar, use_container_width=True)
        else:
            st.info("Need ≥12 months of data for ARIMA.")
    else:
        st.info("Install statsmodels: `pip install statsmodels`")

    # ── Prophet ──
    st.markdown('<div class="section-title">🔮 Prophet Forecast + Components</div>', unsafe_allow_html=True)
    if not monthly.empty:
        prophet_df = monthly.rename(columns={"Order Date":"ds","Sales":"y"})[["ds","y"]]
        try:
            with st.spinner("Running Prophet…"):
                pm = Prophet(yearly_seasonality=True, daily_seasonality=False)
                pm.fit(prophet_df)
                future   = pm.make_future_dataframe(periods=n_months*30)
                forecast = pm.predict(future)

            fig_ph = go.Figure()
            fig_ph.add_trace(go.Scatter(
                x=prophet_df["ds"], y=prophet_df["y"],
                name="Actual", line={"color": T["accent1"]}
            ))
            fig_ph.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["yhat"],
                name="Forecast", line={"color": T["accent3"], "dash": "dash"}
            ))
            fig_ph.add_trace(go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor=rgba(T["accent3"], 0.09),
                line={"color": "rgba(0,0,0,0)"},
                name="Uncertainty"
            ))
            fig_ph.update_layout(template=PLOTLY_TEMPLATE, title="Prophet Sales Forecast")
            st.plotly_chart(fig_ph, use_container_width=True)

            fig_comp2 = make_subplots(rows=2, cols=1, subplot_titles=["Trend","Yearly Seasonality"])
            fig_comp2.add_trace(go.Scatter(
                x=forecast["ds"], y=forecast["trend"],
                line={"color": T["accent1"]}, name="Trend"), row=1, col=1
            )
            if "yearly" in forecast.columns:
                fig_comp2.add_trace(go.Scatter(
                    x=forecast["ds"], y=forecast["yearly"],
                    line={"color": T["accent2"]}, name="Yearly"), row=2, col=1
                )
            fig_comp2.update_layout(
                template=PLOTLY_TEMPLATE, title="Prophet Decomposition",
                height=500, showlegend=True
            )
            st.plotly_chart(fig_comp2, use_container_width=True)

        except Exception as e:
            st.warning(f"Prophet error: {e}")


# ══════════════════════════════════════════
# TAB 4 — RFM & CUSTOMERS
# ══════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">🎯 RFM Score Board</div>', unsafe_allow_html=True)

    if "RFM_Score" in filtered.columns and "Customer ID" in filtered.columns:
        rfm_agg = (filtered.groupby("Customer ID")
                   .agg(
                       RFM_Score   =("RFM_Score","first"),
                       R_Score     =("R_Score","first"),
                       F_Score     =("F_Score","first"),
                       M_Score     =("M_Score","first"),
                       Total_Sales =("Sales","sum"),
                       Orders      =("Order ID","nunique") if "Order ID" in filtered.columns else ("Sales","count")
                   )
                   .reset_index())

        def rfm_segment(score):
            if   score >= 10: return "🥇 Champions"
            elif score >= 8:  return "🥈 Loyal"
            elif score >= 6:  return "👍 Potential"
            elif score >= 4:  return "⚠️ At Risk"
            else:             return "😴 Lost"

        rfm_agg["Segment"] = rfm_agg["RFM_Score"].apply(rfm_segment)
        seg_dist = rfm_agg["Segment"].value_counts().reset_index()
        seg_dist.columns = ["Segment","Count"]

        c_rfm1, c_rfm2 = st.columns([2,1])
        with c_rfm1:
            fig_rfm = px.scatter(
                rfm_agg, x="R_Score", y="M_Score",
                size="Total_Sales", color="Segment",
                hover_data=["Customer ID","RFM_Score"],
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=ACCENT_COLORS,
                title="RFM Scatter: Recency vs Monetary (bubble = Sales)"
            )
            st.plotly_chart(fig_rfm, use_container_width=True)
        with c_rfm2:
            fig_seg = px.pie(
                seg_dist, names="Segment", values="Count",
                hole=0.45, template=PLOTLY_TEMPLATE,
                color_discrete_sequence=ACCENT_COLORS,
                title="Segment Distribution"
            )
            st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown('<div class="section-title">🌟 Top 20 Customers by RFM</div>', unsafe_allow_html=True)
        top_cust = rfm_agg.nlargest(20, "RFM_Score")[
            ["Customer ID","RFM_Score","R_Score","F_Score","M_Score","Total_Sales","Segment"]
        ]
        st.dataframe(
            top_cust.style
                .background_gradient(cmap="Greens", subset=["RFM_Score","Total_Sales"])
                .format({"Total_Sales":"₹{:,.0f}"}),
            use_container_width=True
        )
    else:
        st.info("RFM scoring requires Customer ID, Order ID, Order Date, and Sales columns.")

    st.markdown('<div class="section-title">👥 KMeans Customer Clusters</div>', unsafe_allow_html=True)
    if "Customer ID" in filtered.columns:
        cust_cols = [c for c in ["Sales","Profit","Quantity"] if c in filtered.columns]
        cust = filtered.groupby("Customer ID")[cust_cols].sum().reset_index().dropna()
        if len(cust) >= 3:
            k      = st.slider("Number of clusters", 2, 6, 3, key="km_slider")
            scaler = StandardScaler()
            cs     = scaler.fit_transform(cust[cust_cols])
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            cust["Cluster"] = km.fit_predict(cs).astype(str)
            labels    = ["🥇 Champions","🥈 Loyal","🥉 At Risk","😴 Hibernating","🆕 New","💤 Lost"]
            cl_summary = cust.groupby("Cluster")["Sales"].mean().sort_values(ascending=False)
            cust["Segment"] = cust["Cluster"].map(
                {c: labels[i] for i,c in enumerate(cl_summary.index) if i < len(labels)}
            )
            fig_cl = px.scatter(
                cust, x="Sales", y="Profit", color="Segment",
                size="Quantity" if "Quantity" in cust.columns else None,
                hover_data=["Customer ID"],
                template=PLOTLY_TEMPLATE, title="Customer Clusters",
                color_discrete_sequence=ACCENT_COLORS
            )
            st.plotly_chart(fig_cl, use_container_width=True)

    st.markdown('<div class="section-title">🚨 Sales Anomaly Detection</div>', unsafe_allow_html=True)
    if not monthly.empty:
        iso = IsolationForest(contamination=0.08, random_state=42)
        monthly["Anomaly"] = iso.fit_predict(monthly[["Sales"]].fillna(0))
        monthly["Flag"]    = monthly["Anomaly"].map({1:"Normal", -1:"🚨 Anomaly"})
        fig_an = px.scatter(
            monthly, x="Order Date", y="Sales", color="Flag",
            template=PLOTLY_TEMPLATE, size="Sales",
            color_discrete_map={"Normal": T["accent1"], "🚨 Anomaly": "#ef4444"},
            title="Monthly Sales Anomaly Detection"
        )
        st.plotly_chart(fig_an, use_container_width=True)


# ══════════════════════════════════════════
# TAB 5 — LEADERBOARD
# ══════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">🏅 Leaderboard</div>', unsafe_allow_html=True)

    lb_metric = st.radio("Rank by", ["Sales","Profit"], horizontal=True, key="lb_metric")
    lb_top    = st.slider("Top N", 5, 20, 10, key="lb_n")

    col_lb1, col_lb2, col_lb3 = st.columns(3)

    with col_lb1:
        st.markdown(f"#### 📦 Top {lb_top} Products")
        if "Product Name" in filtered.columns and lb_metric in filtered.columns:
            prod_lb = filtered.groupby("Product Name")[lb_metric].sum().nlargest(lb_top).reset_index()
            for i, row in prod_lb.iterrows():
                rank  = i + 1
                medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
                st.markdown(f"""
                <div class="lb-row">
                  <div class="lb-rank">{medal}</div>
                  <div class="lb-name" style="font-size:0.78rem">{row['Product Name'][:28]}</div>
                  <div class="lb-val">₹{row[lb_metric]:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_lb2:
        st.markdown("#### 🗺️ Top Regions")
        if "Region" in filtered.columns and lb_metric in filtered.columns:
            reg_lb  = filtered.groupby("Region")[lb_metric].sum().sort_values(ascending=False).reset_index()
            max_val = reg_lb[lb_metric].max()
            for i, row in reg_lb.iterrows():
                rank  = i + 1
                medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
                pct   = int(row[lb_metric] / max_val * 100)
                st.markdown(f"""
                <div class="lb-row">
                  <div class="lb-rank">{medal}</div>
                  <div class="lb-name">{row['Region']}</div>
                  <div>
                    <div class="lb-val">₹{row[lb_metric]:,.0f}</div>
                    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%"></div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with col_lb3:
        st.markdown("#### 🏷️ Top Sub-Categories")
        if "Sub-Category" in filtered.columns and lb_metric in filtered.columns:
            sc_lb   = filtered.groupby("Sub-Category")[lb_metric].sum().nlargest(lb_top).reset_index()
            max_val = sc_lb[lb_metric].max()
            for i, row in sc_lb.iterrows():
                rank  = i + 1
                medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f"{rank}."
                pct   = int(row[lb_metric] / max_val * 100)
                st.markdown(f"""
                <div class="lb-row">
                  <div class="lb-rank">{medal}</div>
                  <div class="lb-name">{row['Sub-Category']}</div>
                  <div>
                    <div class="lb-val">₹{row[lb_metric]:,.0f}</div>
                    <div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%"></div></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🕸️ Category Radar (Sales · Profit · Qty)</div>', unsafe_allow_html=True)
    if "Category of Goods" in filtered.columns:
        rad_cols = [c for c in ["Sales","Profit","Quantity"] if c in filtered.columns]
        rad      = filtered.groupby("Category of Goods")[rad_cols].sum()
        rad_norm = (rad - rad.min()) / (rad.max() - rad.min() + 1e-9)
        fig_radar = go.Figure()
        for i, cat in enumerate(rad_norm.index):
            vals = rad_norm.loc[cat].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]], theta=rad_cols + [rad_cols[0]],
                fill="toself", name=cat,
                line={"color": ACCENT_COLORS[i % len(ACCENT_COLORS)]}
            ))
        fig_radar.update_layout(
            template=PLOTLY_TEMPLATE, title="Category Performance Radar",
            polar={"radialaxis":{"visible":True}}
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════
# TAB 6 — DISCOUNT LAB
# ══════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">🧩 Discount Analyser & Breakeven Calculator</div>', unsafe_allow_html=True)

    if "Discount" in filtered.columns and "Profit" in filtered.columns:

        sample = filtered.sample(min(2500, len(filtered)), random_state=42)
        fig_disc = px.scatter(
            sample, x="Discount", y="Profit",
            color="Category of Goods" if "Category of Goods" in sample.columns else None,
            trendline="lowess", template=PLOTLY_TEMPLATE,
            color_discrete_sequence=ACCENT_COLORS,
            title="Discount vs Profit (sampled, with LOWESS trend)"
        )
        st.plotly_chart(fig_disc, use_container_width=True)

        st.markdown('<div class="section-title">📊 Profit by Discount Bucket</div>', unsafe_allow_html=True)
        disc_df = filtered.copy()
        disc_df["Discount Bucket"] = pd.cut(
            disc_df["Discount"],
            bins=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
            labels=["0-5%","5-10%","10-20%","20-30%","30-50%","50%+"],
            right=False
        )
        bucket_agg = disc_df.groupby("Discount Bucket", observed=True).agg(
            Avg_Profit  =("Profit","mean"),
            Total_Sales =("Sales","sum"),
            Count       =("Sales","count")
        ).reset_index()
        fig_bkt = px.bar(
            bucket_agg, x="Discount Bucket", y="Avg_Profit",
            color="Avg_Profit", color_continuous_scale="RdYlGn",
            template=PLOTLY_TEMPLATE,
            title="Avg Profit by Discount Bucket",
            text="Count"
        )
        st.plotly_chart(fig_bkt, use_container_width=True)

        st.markdown('<div class="section-title">🧮 Breakeven Discount Calculator</div>', unsafe_allow_html=True)
        st.markdown("Enter your product details to find the max discount before going loss:")
        col_be1, col_be2, col_be3 = st.columns(3)
        with col_be1:
            cost_price = st.number_input("Cost Price (₹)",          min_value=1.0,  value=500.0,  step=10.0)
        with col_be2:
            sell_price = st.number_input("Selling Price (₹)",       min_value=1.0,  value=800.0,  step=10.0)
        with col_be3:
            fixed_cost = st.number_input("Fixed Cost per Unit (₹)", min_value=0.0,  value=50.0,   step=5.0)

        if sell_price > 0:
            breakeven_disc = max(0, (sell_price - cost_price - fixed_cost) / sell_price * 100)
            profit_at_0    = sell_price - cost_price - fixed_cost
            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.markdown(f"""
                <div class="discount-result">
                  <div style="color:{T['text_muted']};font-size:0.78rem;text-transform:uppercase;letter-spacing:.1em;">Max Discount</div>
                  <div class="big">{breakeven_disc:.1f}%</div>
                  <div style="color:{T['text_muted']};font-size:0.8rem;">before break-even</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r2:
                st.markdown(f"""
                <div class="discount-result">
                  <div style="color:{T['text_muted']};font-size:0.78rem;text-transform:uppercase;letter-spacing:.1em;">Profit at 0% Disc.</div>
                  <div class="big" style="color:{T['accent3']}">₹{profit_at_0:,.0f}</div>
                  <div style="color:{T['text_muted']};font-size:0.8rem;">per unit</div>
                </div>
                """, unsafe_allow_html=True)
            with col_r3:
                margin_pct = (profit_at_0 / sell_price * 100) if sell_price else 0
                st.markdown(f"""
                <div class="discount-result">
                  <div style="color:{T['text_muted']};font-size:0.78rem;text-transform:uppercase;letter-spacing:.1em;">Gross Margin</div>
                  <div class="big" style="color:{T['accent4']}">{margin_pct:.1f}%</div>
                  <div style="color:{T['text_muted']};font-size:0.8rem;">at current price</div>
                </div>
                """, unsafe_allow_html=True)

            disc_range   = np.linspace(0, min(breakeven_disc + 20, 80), 60)
            profit_curve = [(sell_price * (1 - d/100) - cost_price - fixed_cost) for d in disc_range]

            fig_be = go.Figure()
            fig_be.add_trace(go.Scatter(
                x=disc_range, y=profit_curve,
                mode="lines", name="Profit",
                line={"color": T["accent3"], "width": 3},
                fill="tozeroy",
                fillcolor=rgba(T["accent3"], 0.13),
            ))
            fig_be.add_vline(
                x=breakeven_disc, line_dash="dot", line_color="#ef4444",
                annotation_text=f"Breakeven: {breakeven_disc:.1f}%",
                annotation_font_color="#ef4444"
            )
            fig_be.add_hline(y=0, line_dash="dash", line_color=T["text_muted"])
            fig_be.update_layout(
                template=PLOTLY_TEMPLATE,
                title="Profit vs Discount Curve",
                xaxis_title="Discount %",
                yaxis_title="Profit per Unit (₹)"
            )
            st.plotly_chart(fig_be, use_container_width=True)

    else:
        st.info("Discount and Profit columns required for this tab.")


# TAB 7 — AI Q&A (Gemini + Claude)
# ══════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">🤖 Ask AI From Your Data — Gemini vs Claude</div>', unsafe_allow_html=True)

    # ── Shared prompt builder ──
    def build_prompt(question: str, df_in: pd.DataFrame) -> str:
        summary = build_summary(df_in)
        return f"""You are a senior data analyst for a retail superstore.
Use the data summary below to answer the question clearly and concisely.
Include specific numbers where relevant. Suggest business actions if appropriate.

DATA SUMMARY:
{summary}

QUESTION: {question}
"""

    # ── Gemini function ──
    def ask_gemini(question: str, df_in: pd.DataFrame) -> str:
        if gemini_model is None:
            return "❌ Gemini not initialized. Add GOOGLE_API_KEY to .env"
        try:
            resp = gemini_model.generate_content(build_prompt(question, df_in))
            return resp.text if hasattr(resp, "text") else str(resp)
        except Exception as e:
            return f"❌ Gemini error: {e}"

    # ── Claude function ──
    def ask_claude(question: str, df_in: pd.DataFrame) -> str:
        if not ANTHROPIC_KEY:
            return "❌ Claude not initialized. Add ANTHROPIC_API_KEY to .env"
        try:
            message = claude_client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": build_prompt(question, df_in)}]
            )
            return message.content[0].text
        except Exception as e:
            return f"❌ Claude error: {e}"

    # ── Mode selector ──
    ai_mode = st.radio(
        "Select AI Mode",
        ["🔵 Gemini Only", "🟠 Claude Only", "⚡ Compare Both"],
        horizontal=True,
        key="ai_mode"
    )

    # ── Quick questions ──
    quick_qs = [
        "Which region has the lowest profit margin?",
        "What are the top 5 products by profit?",
        "Where are we losing the most money?",
        "Which sub-categories should we discontinue?",
        "What discount strategy would maximize profit?",
        "What does the RFM analysis tell us?",
    ]
    st.markdown("**💡 Quick Questions:**")
    q_cols     = st.columns(3)
    selected_q = ""
    for i, q in enumerate(quick_qs):
        with q_cols[i % 3]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                selected_q = q

    user_q = st.text_input(
        "Or type your own question:",
        value=selected_q,
        placeholder="e.g., Why is the West region underperforming?",
        key="ai_question"
    )

    if st.button("🔍 Get AI Answer", type="primary", key="ai_submit"):
        if not user_q:
            st.warning("Please enter a question first.")
        else:
            # ── Gemini Only ──
            if ai_mode == "🔵 Gemini Only":
                with st.spinner("Gemini is thinking…"):
                    ans = ask_gemini(user_q, filtered)
                st.markdown(f"""
                <div style="background:{T['bg_card']};border-radius:16px;padding:22px;
                            border:1px solid {T['border']};color:{T['text_primary']};
                            line-height:1.75;font-size:0.9rem;">
                <b style="color:{T['accent2']}">🔵 Gemini Answer:</b><br><br>
                {ans.replace(chr(10),'<br>')}
                </div>
                """, unsafe_allow_html=True)

            # ── Claude Only ──
            elif ai_mode == "🟠 Claude Only":
                with st.spinner("Claude is thinking…"):
                    ans = ask_claude(user_q, filtered)
                st.markdown(f"""
                <div style="background:{T['bg_card']};border-radius:16px;padding:22px;
                            border:1px solid {T['border']};color:{T['text_primary']};
                            line-height:1.75;font-size:0.9rem;">
                <b style="color:{T['accent4']}">🟠 Claude Answer:</b><br><br>
                {ans.replace(chr(10),'<br>')}
                </div>
                """, unsafe_allow_html=True)

            # ── Compare Both ──
            else:
                col_g, col_c = st.columns(2)
                with col_g:
                    with st.spinner("Gemini is thinking…"):
                        ans_g = ask_gemini(user_q, filtered)
                    st.markdown(f"""
                    <div style="background:{T['bg_card']};border-radius:16px;padding:22px;
                                border:1px solid {T['accent2']};color:{T['text_primary']};
                                line-height:1.75;font-size:0.88rem;height:100%;">
                    <b style="color:{T['accent2']}">🔵 Gemini</b><br><br>
                    {ans_g.replace(chr(10),'<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    with st.spinner("Claude is thinking…"):
                        ans_c = ask_claude(user_q, filtered)
                    st.markdown(f"""
                    <div style="background:{T['bg_card']};border-radius:16px;padding:22px;
                                border:1px solid {T['accent4']};color:{T['text_primary']};
                                line-height:1.75;font-size:0.88rem;height:100%;">
                    <b style="color:{T['accent4']}">🟠 Claude</b><br><br>
                    {ans_c.replace(chr(10),'<br>')}
                    </div>
                    """, unsafe_allow_html=True)
# ══════════════════════════════════════════
# TAB 8 — EXPORT
# ══════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-title">📤 Export & Reports</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 📄 CSV")
        st.write("Download filtered data as CSV.")
        csv_bytes = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes,
                           file_name=f"superstore_{datetime.date.today()}.csv",
                           mime="text/csv", use_container_width=True)

    with col_b:
        st.markdown("### 📊 Excel")
        st.write("Multi-sheet: data + KPIs + monthly trend.")
        if st.button("⚙️ Generate Excel", use_container_width=True):
            with st.spinner("Building Excel…"):
                xl = export_excel(filtered)
            st.download_button("📥 Download Excel", data=xl,
                               file_name=f"superstore_{datetime.date.today()}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    with col_c:
        st.markdown("### 📑 PDF")
        if REPORTLAB_AVAILABLE:
            st.write("Professional PDF with KPIs, alerts & top products.")
            if st.button("⚙️ Generate PDF", use_container_width=True):
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib import colors as rl_colors
                buf  = io.BytesIO()
                doc  = SimpleDocTemplate(buf, pagesize=A4)
                styl = getSampleStyleSheet()
                story = []
                story.append(Paragraph("AI Superstore Analytics v3 — Report", styl["Title"]))
                story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}", styl["Normal"]))
                story.append(Spacer(1,16))
                story.append(Paragraph("Key Metrics", styl["Heading2"]))
                kpi_data = [["Metric","Value"]]
                if "Sales"  in filtered.columns: kpi_data.append(["Total Sales",  f"₹{filtered['Sales'].sum():,.0f}"])
                if "Profit" in filtered.columns: kpi_data.append(["Total Profit", f"₹{filtered['Profit'].sum():,.0f}"])
                if "Profit Margin %" in filtered.columns:
                    kpi_data.append(["Avg Margin", f"{filtered['Profit Margin %'].mean():.2f}%"])
                tbl = Table(kpi_data, colWidths=[220,220])
                tbl.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0), rl_colors.HexColor("#6366f1")),
                    ("TEXTCOLOR", (0,0),(-1,0), rl_colors.white),
                    ("FONTNAME",  (0,0),(-1,0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[rl_colors.whitesmoke,rl_colors.white]),
                    ("GRID",     (0,0),(-1,-1), 0.5, rl_colors.lightgrey),
                    ("PADDING",  (0,0),(-1,-1), 8),
                ]))
                story.append(tbl)
                story.append(Spacer(1,12))
                story.append(Paragraph("Smart Alerts", styl["Heading2"]))
                for a in alerts:
                    story.append(Paragraph(f"• {a['msg']}", styl["Normal"]))
                    story.append(Spacer(1,4))
                doc.build(story)
                st.download_button("📥 Download PDF", data=buf.getvalue(),
                                   file_name=f"superstore_{datetime.date.today()}.pdf",
                                   mime="application/pdf", use_container_width=True)
        else:
            st.warning("Install reportlab: `pip install reportlab`")

    st.markdown("---")
    st.markdown(f"### 🔎 Filtered Data Preview ({len(filtered):,} rows)")
    fmt_dict = {}
    if "Sales"           in filtered.columns: fmt_dict["Sales"]           = "₹{:,.0f}"
    if "Profit"          in filtered.columns: fmt_dict["Profit"]          = "₹{:,.0f}"
    if "Profit Margin %" in filtered.columns: fmt_dict["Profit Margin %"] = "{:.1f}%"
    st.dataframe(
        filtered.head(200).style
            .background_gradient(cmap="Blues", subset=["Sales"] if "Sales" in filtered.columns else [])
            .format(fmt_dict, na_rep="—"),
        use_container_width=True, height=400
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center;color:{T['text_muted']};font-size:0.8rem;
            padding:16px 0;border-top:1px solid {T['border']};margin-top:24px;">
    🧠 AI Superstore Predictive Analytics <b>v3</b>
    &nbsp;|&nbsp; Developed by <b style="color:{T['accent1']}">Shivam Yadav</b>
    &nbsp;|&nbsp; Powered by Streamlit · Plotly · XGBoost · Prophet · Gemini · RFM
    &nbsp;|&nbsp; Theme: {selected_theme}
</div>
""", unsafe_allow_html=True)
