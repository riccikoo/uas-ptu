import streamlit as st
import os
import numpy as np
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import io
import asyncio
import edge_tts
import plotly.graph_objects as go
import time
from datetime import datetime

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AI Voice Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# PREMIUM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg-primary: #0B0B0D;
    --bg-secondary: #111113;
    --bg-tertiary: #18181B;
    --bg-card: #141416;
    --accent-gold: #C8A96B;
    --accent-light: #E7D3A7;
    --accent-highlight: #F5E6C8;
    --accent-muted: #8B7355;
    --text-primary: #FFFFFF;
    --text-secondary: #D1D5DB;
    --text-muted: #9CA3AF;
    --border: rgba(200, 169, 107, 0.15);
    --border-hover: rgba(200, 169, 107, 0.35);
    --glass: rgba(255, 255, 255, 0.03);
    --glass-hover: rgba(255, 255, 255, 0.06);
    --shadow-gold: 0 0 40px rgba(200, 169, 107, 0.08);
    --shadow-deep: 0 24px 64px rgba(0, 0, 0, 0.6);
    --font-display: 'Syne', sans-serif;
    --font-body: 'DM Sans', sans-serif;
    --radius: 16px;
    --radius-sm: 10px;
    --radius-lg: 24px;
}

/* GLOBAL RESET */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(200, 169, 107, 0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 80% 80%, rgba(200, 169, 107, 0.03) 0%, transparent 50%) !important;
    min-height: 100vh;
}

/* HIDE STREAMLIT BRANDING */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* MAIN CONTAINER */
.main .block-container {
    max-width: 1100px !important;
    padding: 2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* SCROLLBAR */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-muted); border-radius: 4px; }

/* ======= HERO SECTION ======= */
.hero-wrapper {
    text-align: center;
    padding: 4rem 2rem 3rem;
    position: relative;
    overflow: hidden;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--glass);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 2rem;
    animation: fadeSlideUp 0.6s ease both;
}

.hero-eyebrow::before {
    content: '';
    width: 6px; height: 6px;
    background: var(--accent-gold);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

.hero-title {
    font-family: var(--font-display) !important;
    font-size: clamp(3rem, 7vw, 5.5rem) !important;
    font-weight: 800 !important;
    line-height: 1.0 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 1.5rem !important;
    animation: fadeSlideUp 0.7s ease 0.1s both;
    background: linear-gradient(135deg, #FFFFFF 0%, var(--accent-light) 50%, var(--accent-gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    font-size: 1.05rem !important;
    color: var(--text-muted) !important;
    max-width: 520px;
    margin: 0 auto 2.5rem !important;
    line-height: 1.75 !important;
    font-weight: 300 !important;
    animation: fadeSlideUp 0.8s ease 0.2s both;
}

.hero-divider {
    width: 60px;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
    margin: 0 auto 3rem;
    animation: fadeSlideUp 0.9s ease 0.3s both;
}

/* ======= NAVIGATION ======= */
.nav-container {
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 3rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 6px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    animation: fadeSlideUp 1s ease 0.4s both;
}

.nav-btn {
    padding: 10px 24px;
    border-radius: 100px;
    border: none;
    cursor: pointer;
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 500;
    transition: all 0.25s ease;
    white-space: nowrap;
}

.nav-btn-active {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-muted));
    color: #000;
    box-shadow: 0 2px 16px rgba(200, 169, 107, 0.25);
}

.nav-btn-inactive {
    background: transparent;
    color: var(--text-muted);
}

.nav-btn-inactive:hover {
    background: var(--glass-hover);
    color: var(--text-secondary);
}

/* ======= SECTION HEADERS ======= */
.section-header {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-gold);
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: var(--font-display) !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}

.section-desc {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-weight: 300;
}

/* ======= PREMIUM CARDS ======= */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(200, 169, 107, 0.3), transparent);
}

.card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-gold);
    transform: translateY(-1px);
}

.card-sm {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.5rem;
}

/* ======= RECORDING CARD ======= */
.mic-icon-wrapper {
    width: 80px; height: 80px;
    background: linear-gradient(135deg, rgba(200, 169, 107, 0.15), rgba(200, 169, 107, 0.05));
    border: 1px solid rgba(200, 169, 107, 0.25);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem;
    font-size: 2rem;
    position: relative;
}

.mic-ring {
    position: absolute;
    top: -6px; left: -6px; right: -6px; bottom: -6px;
    border: 1px solid rgba(200, 169, 107, 0.1);
    border-radius: 50%;
    animation: ringPulse 2.5s ease-in-out infinite;
}

.mic-ring-2 {
    position: absolute;
    top: -14px; left: -14px; right: -14px; bottom: -14px;
    border: 1px solid rgba(200, 169, 107, 0.05);
    border-radius: 50%;
    animation: ringPulse 2.5s ease-in-out infinite 0.4s;
}

/* ======= RESULT CARD ======= */
.result-card {
    background: linear-gradient(135deg, rgba(200, 169, 107, 0.08), rgba(200, 169, 107, 0.02));
    border: 1px solid rgba(200, 169, 107, 0.25);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
}

.result-label-text {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-muted);
    margin-bottom: 0.75rem;
}

.result-value {
    font-family: var(--font-display) !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
    background: linear-gradient(135deg, #FFFFFF, var(--accent-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

.confidence-ring-wrapper {
    display: flex;
    justify-content: center;
    margin: 0 auto;
}

/* ======= METRIC CARDS ======= */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent-gold), transparent);
    border-radius: 0 0 0 var(--radius-sm);
}

.metric-number {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: var(--accent-gold) !important;
    letter-spacing: -0.03em !important;
    line-height: 1 !important;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-top: 0.4rem;
}

/* ======= VOICE CARDS ======= */
.voice-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.25rem 1.5rem;
    cursor: pointer;
    transition: all 0.25s ease;
    text-align: center;
}

.voice-card:hover {
    border-color: var(--border-hover);
    background: var(--glass-hover);
}

.voice-card-active {
    border-color: var(--accent-gold) !important;
    background: linear-gradient(135deg, rgba(200, 169, 107, 0.1), rgba(200, 169, 107, 0.03)) !important;
}

.voice-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.voice-name {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
}
.voice-desc { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }

/* ======= STREAMLIT WIDGET OVERRIDES ======= */
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    padding: 6px !important;
    width: fit-content !important;
    margin: 0 auto 2.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 100px !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
    transition: all 0.25s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-muted)) !important;
    color: #000 !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 16px rgba(200, 169, 107, 0.25) !important;
}

.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 0 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-gold), var(--accent-muted)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 100px !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(200, 169, 107, 0.2) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(200, 169, 107, 0.35) !important;
    filter: brightness(1.05) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* Secondary buttons */
.stButton > button[kind="secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}

/* Download button */
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--accent-gold) !important;
    border: 1px solid var(--border) !important;
    border-radius: 100px !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 1.75rem !important;
    transition: all 0.25s ease !important;
    width: auto !important;
    box-shadow: none !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent-gold) !important;
    background: rgba(200, 169, 107, 0.08) !important;
    transform: translateY(-1px) !important;
}

/* Text Area */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.25s ease !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: rgba(200, 169, 107, 0.4) !important;
    box-shadow: 0 0 0 3px rgba(200, 169, 107, 0.08) !important;
    outline: none !important;
}

.stTextArea textarea::placeholder { color: var(--text-muted) !important; }

.stTextArea label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* Audio input */
[data-testid="stAudioInput"] {
    background: var(--bg-card) !important;
    border: 1px dashed rgba(200, 169, 107, 0.25) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: all 0.25s ease !important;
}

[data-testid="stAudioInput"]:hover {
    border-color: rgba(200, 169, 107, 0.4) !important;
    background: rgba(200, 169, 107, 0.03) !important;
}

/* Audio player */
[data-testid="stAudio"] audio {
    width: 100% !important;
    border-radius: var(--radius-sm) !important;
    filter: invert(0.9) hue-rotate(180deg) !important;
    opacity: 0.9 !important;
}

/* Alerts */
.stSuccess {
    background: rgba(200, 169, 107, 0.08) !important;
    border: 1px solid rgba(200, 169, 107, 0.25) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--accent-light) !important;
}

.stError {
    background: rgba(255, 80, 80, 0.06) !important;
    border: 1px solid rgba(255, 80, 80, 0.2) !important;
    border-radius: var(--radius-sm) !important;
}

.stInfo {
    background: rgba(200, 169, 107, 0.05) !important;
    border: 1px solid rgba(200, 169, 107, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-muted) !important;
}

.stWarning {
    background: rgba(255, 160, 40, 0.06) !important;
    border: 1px solid rgba(255, 160, 40, 0.2) !important;
    border-radius: var(--radius-sm) !important;
}

/* Radio */
.stRadio label { color: var(--text-secondary) !important; font-family: var(--font-body) !important; }
.stRadio [data-testid="stMarkdownContainer"] { color: var(--text-secondary) !important; }

/* Select Slider */
.stSelectSlider > label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stSelectSlider [data-testid="stTickBarMin"],
.stSelectSlider [data-testid="stTickBarMax"] {
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
}

/* Spinner */
.stSpinner { color: var(--accent-gold) !important; }
[data-testid="stSpinner"] > div { border-top-color: var(--accent-gold) !important; }

/* Dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
}

/* Columns gap */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ======= ANALYTICS CARDS ======= */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.75rem 1.5rem;
    position: relative;
    overflow: hidden;
}

.stat-card-accent {
    border-color: rgba(200, 169, 107, 0.3);
    background: linear-gradient(135deg, rgba(200, 169, 107, 0.07), var(--bg-card));
}

.stat-icon { font-size: 1.4rem; margin-bottom: 1rem; opacity: 0.7; }

.stat-number {
    font-family: var(--font-display);
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 0.4rem;
}

.stat-number-accent { color: var(--accent-gold); }

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}

.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: #4ade80;
    font-weight: 500;
}

.status-dot::before {
    content: '';
    width: 6px; height: 6px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
}

/* ======= ABOUT PAGE ======= */
.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-weight: 500;
    margin: 4px;
}

.feature-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 1rem 0;
    border-bottom: 1px solid var(--border);
}

.feature-item:last-child { border-bottom: none; }

.feature-icon {
    width: 36px; height: 36px;
    background: rgba(200, 169, 107, 0.1);
    border: 1px solid rgba(200, 169, 107, 0.2);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
    margin-top: 2px;
}

.feature-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.9rem;
    margin-bottom: 2px;
}

.feature-desc { font-size: 0.82rem; color: var(--text-muted); line-height: 1.5; }

/* ======= INTEGRATION NOTICE ======= */
.integration-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(200, 169, 107, 0.06);
    border: 1px solid rgba(200, 169, 107, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.875rem 1.25rem;
    margin-bottom: 1.5rem;
}

.integration-badge-icon { font-size: 1.1rem; }

.integration-badge-text {
    font-size: 0.83rem;
    color: var(--text-secondary);
}

.integration-badge-text strong { color: var(--accent-gold); font-weight: 600; }

/* ======= AUDIO OUTPUT CARD ======= */
.audio-output-card {
    background: var(--bg-card);
    border: 1px solid rgba(200, 169, 107, 0.2);
    border-radius: var(--radius);
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.audio-output-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-gold), transparent);
}

/* ======= ANIMATIONS ======= */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.85); }
}

@keyframes ringPulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50% { opacity: 0; transform: scale(1.15); }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.animate-in { animation: fadeSlideUp 0.5s ease both; }

/* Label visibility */
[data-testid="stWidgetLabel"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Plotly chart background */
.js-plotly-plot { border-radius: var(--radius-sm) !important; }

/* Toast */
[data-testid="stToast"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    border-radius: var(--radius-sm) !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL & CONFIG
# ==========================================
N_MFCC = 40

@st.cache_resource
def load_ml_components():
    model = joblib.load('training/model_mobil.pkl')
    scaler = joblib.load('training/scaler.pkl')
    le = joblib.load('training/label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_ml_components()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)

def extract_all_audio_data(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfcc_matrix = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc_matrix.T, axis=0)
        mfcc_std = np.std(mfcc_matrix.T, axis=0)
        features_vector = np.hstack((mfcc_mean, mfcc_std))
        return features_vector, mfcc_matrix, sample_rate
    except Exception as e:
        st.error(f"Gagal memproses rekaman suara: {e}")
        return None, None, None

async def generate_edge_tts(text, voice_name, speed_percentage):
    speed_str = f"{speed_percentage:+}%"
    communicate = edge_tts.Communicate(text, voice_name, rate=speed_str)
    audio_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    audio_buffer.seek(0)
    return audio_buffer.getvalue()

# ==========================================
# SESSION STATE
# ==========================================
if "asr_result" not in st.session_state:
    st.session_state.asr_result = ""
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0
if "tts_count" not in st.session_state:
    st.session_state.tts_count = 0
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = None

# ==========================================
# HERO SECTION
# ==========================================
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-eyebrow">AI-Powered Audio Intelligence</div>
    <h1 class="hero-title">AI Voice Studio</h1>
    <p class="hero-subtitle">Transform speech into intelligence and text into natural voice with AI-powered audio technology.</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# MODEL STATUS
# ==========================================
if not model_loaded:
    st.error(f"⚠️ Model tidak dapat dimuat. Pastikan file `model_mobil.pkl`, `scaler.pkl`, dan `label_encoder.pkl` berada di folder `training/`.")

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎙️  Speech Recognition",
    "🔊  Text to Speech",
    "📊  Analytics",
    "✦  About"
])

# ==========================================
# TAB 1: SPEECH RECOGNITION
# ==========================================
with tab1:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Module 01</div>
        <div class="section-title">Live Audio Recognition</div>
        <div class="section-desc">Record audio and let the model identify patterns in real-time.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 2.5rem 2rem;">
            <div class="mic-icon-wrapper">
                🎙️
                <div class="mic-ring"></div>
                <div class="mic-ring-2"></div>
            </div>
            <div style="font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.4rem;">Recording Studio</div>
            <div style="font-size: 0.82rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 1.5rem;">Press the microphone button below to capture audio for analysis.</div>
        </div>
        """, unsafe_allow_html=True)

        recorded_audio = st.audio_input("Tap to record", label_visibility="collapsed")

        if recorded_audio is not None:
            st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
            analyze_btn = st.button("Analyze Recording  →", key="btn_predict")
        else:
            analyze_btn = False

    with col2:
        if not model_loaded:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 3rem 2rem; opacity: 0.5;">
                <div style="font-size: 2.5rem; margin-bottom: 1rem;">⚠️</div>
                <div style="font-family: var(--font-display); font-weight: 700; color: var(--text-muted); font-size: 1rem;">Model Offline</div>
                <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">Place model files in training/</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-muted); margin-bottom: 0.75rem;">Awaiting Analysis</div>
                <div style="font-family: var(--font-display); font-size: 3rem; font-weight: 800; color: rgba(255,255,255,0.12); letter-spacing: -0.04em; margin-bottom: 0.5rem;">— —</div>
                <div style="font-size: 0.8rem; color: var(--text-muted);">Results will appear here</div>
                <div style="margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid var(--border);">
                    <span class="status-dot">Model Ready</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Analysis logic
    if recorded_audio is not None and analyze_btn:
        if not model_loaded:
            st.error("Model tidak tersedia.")
        else:
            with st.spinner("Processing audio signal..."):
                temp_path = "temp_recorded_audio.wav"
                with open(temp_path, "wb") as f:
                    f.write(recorded_audio.getbuffer())

                features, mfcc_matrix, sr = extract_all_audio_data(temp_path)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if features is not None:
                    features_ready = features.reshape(1, -1)
                    features_scaled = scaler.transform(features_ready)

                    try:
                        predictions = model.predict_proba(features_scaled)
                        predicted_idx = np.argmax(predictions, axis=1)[0]
                        confidence = predictions[0][predicted_idx] * 100
                    except AttributeError:
                        predicted_idx = model.predict(features_scaled)[0]
                        confidence = None

                    predicted_label = le.inverse_transform([predicted_idx])[0]
                    st.session_state.prediction_count += 1
                    if confidence:
                        st.session_state.last_confidence = confidence

                    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

                    # Results row
                    r1, r2 = st.columns(2, gap="medium")
                    with r1:
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-label-text">Detected Class</div>
                            <div class="result-value">{predicted_label.upper()}</div>
                            <div style="font-size: 0.75rem; color: var(--text-muted);">{datetime.now().strftime('%H:%M:%S')}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with r2:
                        if confidence is not None:
                            conf_val = round(confidence, 1)
                            conf_color = "#4ade80" if conf_val >= 80 else "#f59e0b" if conf_val >= 60 else "#ef4444"
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-label-text">Confidence Score</div>
                                <div class="result-value" style="color: {conf_color}; -webkit-text-fill-color: {conf_color};">{conf_val}%</div>
                                <div style="font-size: 0.75rem; color: var(--text-muted);">Model certainty</div>
                            </div>
                            """, unsafe_allow_html=True)

                    # MFCC Plotly Visualization
                    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 0.75rem;">MFCC Spectrogram</div>
                    """, unsafe_allow_html=True)

                    times = np.linspace(0, mfcc_matrix.shape[1] / sr * 512, mfcc_matrix.shape[1])
                    fig = go.Figure(data=go.Heatmap(
                        z=mfcc_matrix,
                        x=times,
                        colorscale=[
                            [0, '#0B0B0D'],
                            [0.25, '#3D2B1A'],
                            [0.5, '#8B7355'],
                            [0.75, '#C8A96B'],
                            [1.0, '#F5E6C8']
                        ],
                        showscale=True,
                        colorbar=dict(
                            bgcolor='rgba(0,0,0,0)',
                            tickcolor='#9CA3AF',
                            tickfont=dict(color='#9CA3AF', size=10),
                            thickness=12,
                            len=0.9
                        )
                    ))
                    fig.update_layout(
                        paper_bgcolor='rgba(20,20,22,0)',
                        plot_bgcolor='rgba(20,20,22,0)',
                        margin=dict(l=0, r=0, t=8, b=0),
                        height=240,
                        xaxis=dict(
                            title='Time (s)', 
                            title_font=dict(color='#9CA3AF', size=11),
                            tickfont=dict(color='#9CA3AF', size=10),
                            gridcolor='rgba(255,255,255,0.04)',
                            zerolinecolor='rgba(255,255,255,0.08)'
                        ),
                        yaxis=dict(
                            title='MFCC Coefficient',
                            title_font=dict(color='#9CA3AF', size=11),
                            tickfont=dict(color='#9CA3AF', size=10),
                            gridcolor='rgba(255,255,255,0.04)',
                            zerolinecolor='rgba(255,255,255,0.08)'
                        ),
                        font=dict(family='DM Sans', color='#9CA3AF')
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.session_state.asr_result = f"Model berhasil mendeteksi perintah suara untuk mobil {predicted_label}"
                    st.toast("Analysis complete — results forwarded to TTS")


# ==========================================
# TAB 2: TEXT TO SPEECH
# ==========================================
with tab2:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Module 02</div>
        <div class="section-title">Text to Speech</div>
        <div class="section-desc">Convert text to natural Indonesian speech using Edge-TTS neural voices.</div>
    </div>
    """, unsafe_allow_html=True)

    # Integration notice
    if st.session_state.asr_result != "":
        st.markdown(f"""
        <div class="integration-badge">
            <span class="integration-badge-icon">⟳</span>
            <span class="integration-badge-text">
                <strong>ASR Integration Active</strong> — Text forwarded from Speech Recognition module.
            </span>
        </div>
        """, unsafe_allow_html=True)
        default_text = st.session_state.asr_result
    else:
        default_text = "Halo! Silakan ketik perintah teks atau rekam suara di tab sebelah."

    col_left, col_right = st.columns([1.3, 0.7], gap="large")

    with col_left:
        # Text input
        st.markdown("""<div style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;">Input Text</div>""", unsafe_allow_html=True)
        text_input = st.text_area(
            "Input Text",
            value=default_text,
            height=160,
            label_visibility="collapsed",
            placeholder="Start typing or paste your text here..."
        )

        # Character/word count
        word_count = len(text_input.split()) if text_input.strip() else 0
        char_count = len(text_input)
        st.markdown(f"""
        <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <span style="font-size: 0.75rem; color: var(--text-muted);"><span style="color: var(--accent-gold); font-weight: 600;">{word_count}</span> words</span>
            <span style="font-size: 0.75rem; color: var(--text-muted);"><span style="color: var(--accent-gold); font-weight: 600;">{char_count}</span> characters</span>
        </div>
        """, unsafe_allow_html=True)

        # Generate button
        gen_btn = st.button("Generate Speech  →", key="btn_tts")

    with col_right:
        # Voice selection
        st.markdown("""<div style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem;">Voice</div>""", unsafe_allow_html=True)

        gender_choice = st.radio(
            "Voice",
            ["Perempuan (Gadis)", "Laki-laki (Ardi)"],
            horizontal=False,
            label_visibility="collapsed"
        )
        selected_voice = "id-ID-ArdiNeural" if "Laki-laki" in gender_choice else "id-ID-GadisNeural"

        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

        # Speed control
        st.markdown("""<div style="font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.5rem;">Speed</div>""", unsafe_allow_html=True)
        speed_choice = st.select_slider(
            "Speed",
            options=["Slow", "Normal", "Fast"],
            value="Normal",
            label_visibility="collapsed"
        )
        speed_map = {"Slow": -20, "Normal": 0, "Fast": 25}
        target_speed = speed_map[speed_choice]

    # TTS Generation
    if gen_btn:
        if text_input.strip() == "":
            st.warning("Please enter some text before generating.")
        else:
            with st.spinner("Synthesizing speech..."):
                try:
                    audio_bytes = asyncio.run(generate_edge_tts(text_input, selected_voice, target_speed))
                    st.session_state.tts_count += 1

                    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                    st.markdown("""
                    <div class="audio-output-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem;">
                            <div>
                                <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 4px;">Generated Audio</div>
                                <div style="font-size: 0.82rem; color: var(--text-secondary); font-weight: 500;">Neural Voice Synthesis</div>
                            </div>
                            <span class="status-dot">Ready</span>
                        </div>
                    """, unsafe_allow_html=True)

                    st.audio(audio_bytes, format='audio/mp3')

                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)
                    st.download_button(
                        label="↓  Download MP3",
                        data=audio_bytes,
                        file_name=f"voice_{gender_choice.split()[0].lower()}_{int(time.time())}.mp3",
                        mime="audio/mp3"
                    )
                    st.success("Speech synthesis completed successfully.")

                except Exception as e:
                    st.error(f"TTS synthesis failed: {e}")

    # Clear ASR integration
    if st.session_state.asr_result != "":
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Clear ASR Integration", key="clear_asr"):
            st.session_state.asr_result = ""
            st.rerun()


# ==========================================
# TAB 3: ANALYTICS
# ==========================================
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Module 03</div>
        <div class="section-title">Session Analytics</div>
        <div class="section-desc">Real-time insights from your current session activity.</div>
    </div>
    """, unsafe_allow_html=True)

    avg_conf = f"{st.session_state.last_confidence:.1f}%" if st.session_state.last_confidence else "—"
    model_status = "Active" if model_loaded else "Offline"
    model_color = "#4ade80" if model_loaded else "#ef4444"

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card stat-card-accent">
            <div class="stat-icon">🎙️</div>
            <div class="stat-number stat-number-accent">{st.session_state.prediction_count}</div>
            <div class="stat-label">Total Predictions</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🔊</div>
            <div class="stat-number">{st.session_state.tts_count}</div>
            <div class="stat-label">Generated Audios</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-number stat-number-accent">{avg_conf}</div>
            <div class="stat-label">Last Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">⚙️</div>
            <div class="stat-number" style="font-size: 1.5rem; padding-top: 0.4rem; color: {model_color};">{model_status}</div>
            <div class="stat-label">Model Status</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Activity chart
    if st.session_state.prediction_count > 0 or st.session_state.tts_count > 0:
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        st.markdown("""<div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 0.75rem;">Session Activity</div>""", unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Speech Recognitions", "TTS Generations"],
            y=[st.session_state.prediction_count, st.session_state.tts_count],
            marker_color=[
                'rgba(200, 169, 107, 0.8)',
                'rgba(200, 169, 107, 0.35)'
            ],
            marker_line_color='rgba(200, 169, 107, 0.2)',
            marker_line_width=1,
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(20,20,22,0)',
            plot_bgcolor='rgba(20,20,22,0)',
            margin=dict(l=0, r=0, t=8, b=0),
            height=220,
            showlegend=False,
            xaxis=dict(tickfont=dict(color='#9CA3AF', size=11), gridcolor='rgba(0,0,0,0)', zerolinecolor='rgba(0,0,0,0)'),
            yaxis=dict(tickfont=dict(color='#9CA3AF', size=11), gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.04)'),
            font=dict(family='DM Sans', color='#9CA3AF'),
            bargap=0.4
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 3rem; opacity: 0.5;">
            <div style="font-size: 2rem; margin-bottom: 0.75rem;">∿</div>
            <div style="color: var(--text-muted); font-size: 0.85rem;">No activity recorded this session yet.</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 4: ABOUT
# ==========================================
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="section-label">Product</div>
        <div class="section-title">AI Voice Studio</div>
        <div class="section-desc">An AI-powered audio intelligence platform combining speech recognition and neural speech synthesis.</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("""
        <div class="card">
            <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 1.25rem;">Core Features</div>
            <div class="feature-item">
                <div class="feature-icon">🎙️</div>
                <div>
                    <div class="feature-title">Live Speech Recognition</div>
                    <div class="feature-desc">Real-time audio capture and ML-based classification using MFCC feature extraction.</div>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">🔊</div>
                <div>
                    <div class="feature-title">Neural Text to Speech</div>
                    <div class="feature-desc">Natural Indonesian voice synthesis with Microsoft Edge Neural TTS engine.</div>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">📊</div>
                <div>
                    <div class="feature-title">MFCC Visualization</div>
                    <div class="feature-desc">Interactive spectrogram visualization with Plotly for audio feature analysis.</div>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon">⟳</div>
                <div>
                    <div class="feature-title">ASR–TTS Integration</div>
                    <div class="feature-desc">Seamless pipeline — recognition results flow directly into the TTS module.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="card" style="margin-bottom: 1rem;">
            <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 1.25rem;">Technology Stack</div>
            <div style="line-height: 2.2;">
                <span class="tech-badge">🐍 Python 3.10+</span>
                <span class="tech-badge">⚡ Streamlit</span>
                <span class="tech-badge">🎵 Librosa</span>
                <span class="tech-badge">🤖 Scikit-learn</span>
                <span class="tech-badge">📈 Plotly</span>
                <span class="tech-badge">🔊 Edge-TTS</span>
                <span class="tech-badge">🔢 NumPy</span>
                <span class="tech-badge">📦 Joblib</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; color: var(--accent-gold); margin-bottom: 1.25rem;">Model Architecture</div>
            <div class="feature-item" style="padding: 0.75rem 0;">
                <div>
                    <div class="feature-title">Feature Extraction</div>
                    <div class="feature-desc">40-coefficient MFCC with mean + std statistics (80 total features).</div>
                </div>
            </div>
            <div class="feature-item" style="padding: 0.75rem 0;">
                <div>
                    <div class="feature-title">ML Classifier</div>
                    <div class="feature-desc">SVM / Random Forest with StandardScaler normalization and LabelEncoder.</div>
                </div>
            </div>
            <div class="feature-item" style="padding: 0.75rem 0; border-bottom: none;">
                <div>
                    <div class="feature-title">TTS Engine</div>
                    <div class="feature-desc">Microsoft Edge Neural voices — id-ID-GadisNeural & id-ID-ArdiNeural.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 3rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; border-top: 1px solid var(--border);">
        <div style="font-family: var(--font-display); font-size: 1.1rem; font-weight: 700; color: var(--accent-gold); letter-spacing: -0.01em;">AI Voice Studio</div>
        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem;">Powered by Streamlit · Edge-TTS · Librosa · Scikit-learn</div>
    </div>
    """, unsafe_allow_html=True)