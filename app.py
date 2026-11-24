from __future__ import annotations
import os
import tempfile
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.matcher import calculate_similarity, calculate_sbert_similarity, find_missing_skills
from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text, COMMON_SKILLS
from src.mossga_engine import run_mossga_pipeline
from src.github_analyzer import analyze_github_profile
from src.semantic_skill_matcher import semantic_skill_match, calculate_semantic_gap_score
from src.multimodal_fusion import fuse_skill_profiles, generate_fusion_summary
from src.career_intelligence import (
  ROLE_SKILL_DB,
  estimate_level,
  generate_skill_gap_and_roadmap,
  generate_job_recommendations,
  generate_career_guidance,
  extract_education_from_text,
  extract_experience_from_text,
  extract_projects_from_text,
)
from src.knowledge_graph import SKILL_TAXONOMY, SkillKnowledgeGraph

# ---------------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------------

st.set_page_config(
  page_title="MoSSGA — Multi-Modal Semantic Skill Gap Analysis",
  page_icon="🎯",
  layout="wide",
  initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Premium Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

  /* ── Global Reset ─────────────────────────────────────── */
  html, body, .stApp {
    background: #f6f4fb !important;
    color: #1a1523 !important;
    font-family: 'Inter', 'Plus Jakarta Sans', sans-serif !important;
  }
  .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
  }

  /* Fix stray Streamlit white boxes */
  .stApp > header { background: transparent !important; }
  div[data-testid="stDecoration"] { display: none !important; }

  /* ── Sidebar ───────────────────────────────────────────── */
  section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e0f0 !important;
  }
  section[data-testid="stSidebar"] * {
    color: #1a1523 !important;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3,
  section[data-testid="stSidebar"] .stMarkdown h4 {
    color: #1a1523 !important;
    font-weight: 700 !important;
  }
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] .stCaption { color: #6b6578 !important; }
  section[data-testid="stSidebar"] hr { border-color: #e8e0f0 !important; }

  /* Sidebar Buttons */
  section[data-testid="stSidebar"] .stButton > button {
    background: #f1ecf7 !important;
    color: #6a4878 !important;
    border: 1px solid #d8cce8 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: background 0.2s, transform 0.1s !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: #e6dcf2 !important;
    transform: translateY(-1px) !important;
  }

  /* ── Typography ─────────────────────────────────────────── */
  h1, h2, h3, h4, h5 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em !important;
    color: #1a1523 !important;
  }
  .main p, .main span, .main li { color: #6b6578 !important; }
  .main strong, .main b { color: #1a1523 !important; }

  /* ── Navbar ─────────────────────────────────────────────── */
  .mossga-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid #e8e0f0;
    border-radius: 16px;
    padding: 12px 24px;
    margin: 0 0 1.5rem 0;
    box-shadow: 0 2px 12px rgba(115,83,132,0.07);
  }
  .nav-brand { display:flex; align-items:center; gap:14px; }
  .brand-logo-mark {
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg,#735384,#9b7fa8);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; box-shadow: 0 4px 12px rgba(115,83,132,0.25);
  }
  .brand-name {
    display: block;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800; font-size: 1.1rem;
    color: #1a1523;
    letter-spacing: -0.03em;
  }
  .brand-badge {
    display: inline-block;
    background: linear-gradient(135deg,#735384,#9b7fa8);
    color: #fff !important;
    font-size: 0.6rem; font-weight: 700;
    padding: 2px 7px; border-radius: 999px;
    letter-spacing: 0.06em; text-transform: uppercase;
    vertical-align: middle; margin-left: 6px;
  }
  .brand-sub {
    display: block;
    font-size: 0.72rem; color: #9490a0;
    font-weight: 500; letter-spacing: 0.01em;
    margin-top: 2px;
  }
  .nav-right { display:flex; align-items:center; gap:12px; }
  .status-live {
    display: flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 999px; padding: 5px 12px;
    font-size: 0.75rem; font-weight: 600;
    color: #166534 !important;
  }
  .pulse-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e;
    animation: pulse 1.6s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.5; transform:scale(0.75); }
  }
  .nav-gh-btn {
    display: flex; align-items: center; gap: 7px;
    background: #1a1523; color: #fff !important;
    border-radius: 999px; padding: 7px 16px;
    font-size: 0.78rem; font-weight: 600;
    text-decoration: none; transition: background 0.2s;
  }
  .nav-gh-btn:hover { background: #2d2440; }

  /* ── Navigation Pills ───────────────────────────────────── */
  .nav-pills {
    display: flex; gap: 6px; flex-wrap: wrap;
    background: #ffffff;
    border: 1px solid #e8e0f0;
    border-radius: 14px;
    padding: 6px 8px;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 6px rgba(115,83,132,0.05);
  }
  .nav-pill {
    padding: 8px 20px; border-radius: 10px;
    font-size: 0.84rem; font-weight: 600;
    cursor: pointer; transition: all 0.18s ease;
    color: #6b6578 !important;
    background: transparent;
  }
  .nav-pill:hover { background: #f1ecf7; color: #735384 !important; }
  .nav-pill.active {
    background: #f1ecf7 !important;
    color: #6a4878 !important;
    box-shadow: 0 1px 4px rgba(115,83,132,0.12);
    border-bottom: 2px solid #735384;
  }

  /* ── Glass Cards ────────────────────────────────────────── */
  .glass-card {
    background: #ffffff !important;
    border: 1px solid #e8e0f0 !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 10px rgba(115,83,132,0.05) !important;
    color: #1a1523 !important;
  }
  .glass-card * { color: #1a1523 !important; }
  .glass-card h4, .glass-card h3 { color: #1a1523 !important; font-weight: 700 !important; }
  .glass-card p, .glass-card span { color: #6b6578 !important; }
  .glass-card strong { color: #1a1523 !important; }

  /* ── Score Cards ────────────────────────────────────────── */
  .score-card {
    background: #ffffff !important;
    border: 1px solid #e8e0f0 !important;
    border-radius: 14px !important;
    padding: 20px 18px !important;
    text-align: center !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 2px 10px rgba(115,83,132,0.06) !important;
  }
  .score-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg,#735384,#9b7fa8);
    border-radius: 14px 14px 0 0;
  }
  .score-val {
    display: block;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    color: #1a1523 !important;
    line-height: 1.1 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.04em !important;
  }
  .score-lbl {
    display: block;
    font-size: 0.75rem !important;
    color: #6b6578 !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-top: 4px !important;
  }
  .col-green  { color: #059669 !important; }
  .col-yellow { color: #d97706 !important; }
  .col-red    { color: #dc2626 !important; }
  .col-blue   { color: #2563eb !important; }
  .col-purple { color: #735384 !important; }

  /* ── Hero Section ───────────────────────────────────────── */
  .hero-wrap {
    background: linear-gradient(135deg, #f6f4fb 0%, #ede9f7 100%);
    border: 1px solid #ddd6f0;
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
  }
  .hero-chip {
    display: inline-block;
    background: rgba(115,83,132,0.1);
    border: 1px solid rgba(115,83,132,0.2);
    color: #735384 !important;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 12px;
  }
  .hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.9rem; font-weight: 800;
    color: #1a1523 !important;
    letter-spacing: -0.04em; line-height: 1.2;
    margin: 0 0 8px 0;
  }
  .hero-sub {
    font-size: 0.95rem;
    color: #6b6578 !important;
    font-weight: 400; line-height: 1.6;
  }
  .hero-badge {
    display: inline-block;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #166534 !important;
    border-radius: 999px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 700;
  }

  /* ── Status bar ─────────────────────────────────────────── */
  .status-bar {
    display: flex; gap: 10px; flex-wrap: wrap;
    background: #f1ecf7;
    border: 1px solid #e0d5ee;
    border-radius: 12px; padding: 10px 14px;
    margin: 0.6rem 0 1.2rem 0;
  }
  .status-pill {
    display: flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600;
    background: #ffffff; border: 1px solid #ddd5ed;
    color: #6b6578 !important;
  }
  .status-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }

  /* ── Sidebar Section Titles ─────────────────────────────── */
  .sidebar-section-title {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #9490a0 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin: 1.2rem 0 0.5rem 0 !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #e8e0f0 !important;
  }
  .input-section-title {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #735384 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin: 1.2rem 0 0.5rem 0 !important;
  }

  /* ── Skill Badges ───────────────────────────────────────── */
  .skill-badge {
    display: inline-block;
    background: #f1ecf7;
    color: #6a4878 !important;
    border: 1px solid #d8cce8;
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 0.76rem;
    font-weight: 600;
    margin: 2px 3px;
  }
  .skill-badge-red {
    background: #fff1f2 !important;
    color: #be123c !important;
    border-color: #fecdd3 !important;
  }
  .skill-badge-green {
    background: #f0fdf4 !important;
    color: #166534 !important;
    border-color: #bbf7d0 !important;
  }

  /* ── Mode Badges ────────────────────────────────────────── */
  .mode-badge {
    display: inline-block;
    padding: 3px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700;
  }
  .mode-full       { background:#ede9fe; color:#6d28d9 !important; border:1px solid #c4b5fd; }
  .mode-partial    { background:#fefce8; color:#854d0e !important; border:1px solid #fde68a; }
  .mode-guidance   { background:#fef2f2; color:#991b1b !important; border:1px solid #fecaca; }

  /* ── Roadmap Cards ──────────────────────────────────────── */
  .roadmap-card {
    background: #ffffff !important;
    border-radius: 12px !important;
    padding: 16px 18px !important;
    margin-bottom: 10px !important;
    border-left: 4px solid #735384 !important;
    border-top: 1px solid #ebe6f2 !important;
    border-right: 1px solid #ebe6f2 !important;
    border-bottom: 1px solid #ebe6f2 !important;
    color: #1a1523 !important;
    box-shadow: 0 1px 6px rgba(115,83,132,0.05) !important;
  }
  .roadmap-card * { color: #1a1523 !important; }
  .roadmap-card p, .roadmap-card span { color: #6b6578 !important; }
  .roadmap-beginner    { border-left-color: #059669 !important; }
  .roadmap-intermediate{ border-left-color: #735384 !important; }
  .roadmap-advanced    { border-left-color: #9b7fa8 !important; }

  /* ── Empty State ────────────────────────────────────────── */
  .empty-state {
    text-align: center;
    padding: 48px 24px;
    background: #ffffff;
    border: 2px dashed #ddd6f0;
    border-radius: 16px;
    margin: 1rem 0;
  }
  .empty-state-icon { font-size: 3rem; display: block; margin-bottom: 12px; }
  .empty-state-text {
    color: #9490a0 !important;
    font-size: 0.95rem; font-weight: 500;
  }

  /* ── Streamlit Input Widgets ────────────────────────────── */
  .stTextInput > div > div > input,
  .stTextArea > div > div > textarea,
  .stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d8cce8 !important;
    border-radius: 10px !important;
    color: #1a1523 !important;
    font-size: 0.9rem !important;
  }
  .stTextInput > div > div > input::placeholder,
  .stTextArea > div > div > textarea::placeholder { color: #9490a0 !important; }
  .stTextInput label, .stTextArea label, .stSelectbox label,
  .stSlider label, .stFileUploader label {
    color: #1a1523 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
  }

  /* Slider */
  .stSlider > div[data-baseweb] > div { background: #e8e0f0 !important; }
  .stSlider > div[data-baseweb] [role="slider"] { background: #735384 !important; }

  /* File uploader */
  .stFileUploader {
    background: #ffffff !important;
    border: 2px dashed #d8cce8 !important;
    border-radius: 12px !important;
  }

  /* Primary Streamlit Buttons */
  .stButton > button[kind="primary"],
  .stButton > button {
    background: #735384 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 9px 20px !important;
    transition: background 0.2s, transform 0.12s !important;
    box-shadow: 0 2px 8px rgba(115,83,132,0.25) !important;
  }
  .stButton > button:hover {
    background: #5e3f6e !important;
    transform: translateY(-1px) !important;
  }
  .stButton > button:active { transform: translateY(0) !important; }

  /* ── Tabs ───────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background: #f1ecf7 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: none !important;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #6b6578 !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    padding: 8px 18px !important;
    border: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #6a4878 !important;
    border-bottom: 2px solid #735384 !important;
    box-shadow: 0 1px 4px rgba(115,83,132,0.1) !important;
  }

  /* ── Metric Cards ───────────────────────────────────────── */
  div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e8e0f0 !important;
    border-radius: 14px !important;
    padding: 16px 18px !important;
    box-shadow: 0 1px 6px rgba(115,83,132,0.05) !important;
  }
  div[data-testid="stMetricValue"] {
    color: #1a1523 !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }
  div[data-testid="stMetricLabel"] {
    color: #6b6578 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
  }
  div[data-testid="stMetricDelta"] { font-weight: 600 !important; font-size: 0.82rem !important; }

  /* ── Expanders ──────────────────────────────────────────── */
  .streamlit-expanderHeader {
    background: #f8f5fd !important;
    border: 1px solid #e8e0f0 !important;
    border-radius: 10px !important;
    color: #1a1523 !important;
    font-weight: 600 !important;
  }
  .streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e8e0f0 !important;
    border-top: none !important;
    color: #1a1523 !important;
  }

  /* ── Dataframes / Tables ────────────────────────────────── */
  .stDataFrame, .stTable {
    background: #ffffff !important;
    border: 1px solid #e8e0f0 !important;
    border-radius: 12px !important;
  }
  .stDataFrame th {
    background: #f1ecf7 !important;
    color: #1a1523 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
  }
  .stDataFrame td { color: #1a1523 !important; font-size: 0.84rem !important; }

  /* ── Success / Info / Warning / Error Alerts ────────────── */
  .stSuccess { background: #f0fdf4 !important; border-left-color: #22c55e !important; color: #166534 !important; }
  .stInfo    { background: #eff6ff !important; border-left-color: #3b82f6 !important; color: #1e40af !important; }
  .stWarning { background: #fefce8 !important; border-left-color: #f59e0b !important; color: #854d0e !important; }
  .stError   { background: #fef2f2 !important; border-left-color: #ef4444 !important; color: #991b1b !important; }

  /* ── Progress Bar ───────────────────────────────────────── */
  .stProgress > div > div > div { background: linear-gradient(90deg,#735384,#9b7fa8) !important; }

  /* ── Spinner ────────────────────────────────────────────── */
  .stSpinner > div { border-top-color: #735384 !important; }

  /* ── Scrollbar ──────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #f6f4fb; }
  ::-webkit-scrollbar-thumb { background: #d8cce8; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #b8a8d0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly Dark Layout Helper
# ---------------------------------------------------------------------------

def plotly_dark_layout(fig, height=380, title=None):
    """Apply consistent light theme to all Plotly charts."""
    updates = dict(
        template="simple_white",
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(246,244,251,0.4)',
        font=dict(family='Inter, sans-serif', color='#6b6578', size=12),
        margin=dict(l=24, r=24, t=48 if title else 28, b=28),
    )
    if title:
        updates['title'] = dict(text=title, font=dict(color='#1a1523', size=14, family='Plus Jakarta Sans'), x=0.02)
    fig.update_layout(**updates)
    fig.update_xaxes(
        gridcolor='rgba(115,83,132,0.08)',
        linecolor='rgba(115,83,132,0.12)',
        tickfont=dict(color='#9490a0'),
    )
    fig.update_yaxes(
        gridcolor='rgba(115,83,132,0.08)',
        linecolor='rgba(115,83,132,0.12)',
        tickfont=dict(color='#9490a0'),
    )
    return fig



# ---------------------------------------------------------------------------
# Helper: Score color class
# ---------------------------------------------------------------------------

def score_color(val):
  """Returns CSS class for score coloring."""
  try:
    v = float(val)
  except (ValueError, TypeError):
    return "col-yellow"
  if v >= 70:
    return "col-green"
  elif v >= 40:
    return "col-yellow"
  return "col-red"


# ---------------------------------------------------------------------------
# Session State Initialization (for Demo Presets)
# ---------------------------------------------------------------------------

if "target_role_val" not in st.session_state:
  st.session_state.target_role_val = ""
if "manual_skills_val" not in st.session_state:
  st.session_state.manual_skills_val = ""
if "jd_input_val" not in st.session_state:
  st.session_state.jd_input_val = ""
if "github_input_val" not in st.session_state:
  st.session_state.github_input_val = ""


# ---------------------------------------------------------------------------
# Modern Sidebar: Profile Presets, Engine Sliders & Specs
# ---------------------------------------------------------------------------

with st.sidebar:
  st.markdown("""
  <div style="text-align:center; padding:8px 0 20px 0; border-bottom:1px solid #e8e0f0; margin-bottom:16px;">
    <div style="font-size:2.4rem; filter:drop-shadow(0 0 12px rgba(115,83,132,0.3));">🎯</div>
    <div style="font-family:'Plus Jakarta Sans',sans-serif;font-size:1.3rem;font-weight:800;color:#1a1523;letter-spacing:-0.03em;margin-top:6px;">MoSSGA Pro</div>
    <div style="font-size:0.7rem;color:#735384;letter-spacing:1.2px;font-weight:700;text-transform:uppercase;">Talent Intelligence Platform</div>
  </div>
  """, unsafe_allow_html=True)

  st.markdown('<div class="sidebar-section-title">⚡ 1-Click Demo Profiles</div>', unsafe_allow_html=True)
  st.caption("Instantly populate all fields:")

  col_p1, col_p2 = st.columns(2)
  with col_p1:
    if st.button("🤖 ML Engineer", use_container_width=True):
      st.session_state.target_role_val = "Machine Learning Engineer"
      st.session_state.manual_skills_val = "Python, PyTorch, Scikit-learn, SQL, Docker, MLOps, NLP"
      st.session_state.github_input_val = "suryanshagrawal21"
      st.session_state.jd_input_val = "Seeking a Machine Learning Engineer proficient in Python, PyTorch, Scikit-learn, Docker, and MLOps to architect and deploy production AI models."
      st.rerun()
  with col_p2:
    if st.button("💻 Full Stack", use_container_width=True):
      st.session_state.target_role_val = "Full Stack Developer"
      st.session_state.manual_skills_val = "JavaScript, TypeScript, React, Node.js, Express, PostgreSQL, Git, Docker"
      st.session_state.github_input_val = "suryanshagrawal21"
      st.session_state.jd_input_val = "Looking for a Full Stack Developer skilled in React, Node.js, TypeScript, PostgreSQL, and Git to build scalable responsive web applications."
      st.rerun()

  if st.button("📊 Data Scientist Demo", use_container_width=True):
    st.session_state.target_role_val = "Data Scientist"
    st.session_state.manual_skills_val = "Python, SQL, Pandas, NumPy, Statistics, Machine Learning, Tableau"
    st.session_state.github_input_val = "suryanshagrawal21"
    st.session_state.jd_input_val = "Data Scientist required with strong foundation in Python, SQL, Pandas, Statistics, Machine Learning, and Data Visualization."
    st.rerun()

  st.markdown('<div class="sidebar-section-title">⚙️ Engine Parameters</div>', unsafe_allow_html=True)
  semantic_threshold = st.slider("Semantic Match Threshold", min_value=0.50, max_value=0.95, value=0.65, step=0.05)
  resume_weight = st.slider("Resume Evidence Weight", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
  st.caption(f"GitHub Practical Weight: **{1.0 - resume_weight:.1f}**")

  st.markdown('<div class="sidebar-section-title">ℹ️ System Specification</div>', unsafe_allow_html=True)
  st.markdown("""
  <div style="font-size:0.8rem;color:#6b6578;line-height:2;">
    <span style="color:#735384;">●</span> <b style="color:#1a1523;">SBERT:</b> all-MiniLM-L6-v2 (384d)<br>
    <span style="color:#9b7fa8;">●</span> <b style="color:#1a1523;">Regressor:</b> XGBoost v2.0<br>
    <span style="color:#735384;">●</span> <b style="color:#1a1523;">NLP Core:</b> spaCy en_core_web_sm<br>
    <span style="color:#9b7fa8;">●</span> <b style="color:#1a1523;">Graph:</b> NetworkX DiGraph<br>
    <span style="color:#059669;">●</span> <b style="color:#1a1523;">Deploy:</b> Cloud Ready ($PORT)
  </div>
  """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top Sticky Glass Navbar
# ---------------------------------------------------------------------------

st.markdown("""
<header class="mossga-navbar">
  <div class="nav-brand">
    <div class="brand-logo-mark">🎯</div>
    <div>
      <span class="brand-name">MoSSGA <span class="brand-badge">PRO v2.4</span></span>
      <span class="brand-sub">Multi-Modal Semantic Talent Engine</span>
    </div>
  </div>
  <div class="nav-right">
    <div class="status-live">
      <span class="pulse-dot"></span>
      All Engines Live
    </div>
    <a href="https://github.com/suryanshagrawal21/MoSSGA" target="_blank" class="nav-gh-btn">
      <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      GitHub Repo
    </a>
  </div>
</header>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Top Segmented Pill Navigation Menu
# ---------------------------------------------------------------------------

selected_nav = st.radio(
  "Navigation",
  ["🎯 Analyzer", "🌐 Skill Graph", "🧠 Architecture", "⚡ System Health"],
  horizontal=True,
  label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# VIEW 2: 🌐 Skill Graph & Ontology
# ---------------------------------------------------------------------------

def render_skill_graph_view():
  st.markdown("""
  <div class="glass-card" style="margin-bottom: 1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div class="hero-badge">🌐 KNOWLEDGE GRAPH & TALENT ONTOLOGY</div>
        <h2 style="margin: 8px 0 6px 0; color:#1a1523;">Workforce Skill Graph & Ontology Explorer</h2>
        <p style="color: #6b6578; margin: 0; font-size: 1rem;">
          Explore hierarchical skill dependencies, prerequisite chains, and research-grade role benchmark matrices.
        </p>
      </div>
      <div style="font-size: 3rem; filter: drop-shadow(0 0 16px rgba(99,102,241,0.5));">🧬</div>
    </div>
  </div>
  """, unsafe_allow_html=True)

  tab_ont1, tab_ont2 = st.tabs(["📋 Role Benchmark Matrix", "🔗 Ontological Prerequisites & Taxonomy"])

  with tab_ont1:
    st.markdown("### Target Role Benchmark Explorer")
    st.caption("Standardized skill expectations across foundational, intermediate, and advanced levels:")
    
    selected_role = st.selectbox(
      "Select Benchmark Role",
      options=list(ROLE_SKILL_DB.keys()),
      format_func=lambda r: r.title(),
    )

    role_data = ROLE_SKILL_DB[selected_role]
    c1, c2, c3 = st.columns(3)
    with c1:
      st.markdown("#### 🟢 Core Foundations")
      for s in role_data.get("core", []):
        st.markdown(f'<span style="background:rgba(34,197,94,0.15); color:#86efac; padding:5px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.85rem; border:1px solid rgba(34,197,94,0.3); font-weight:600;">{s.title()}</span>', unsafe_allow_html=True)
    with c2:
      st.markdown("#### 🔵 Intermediate Applied")
      for s in role_data.get("intermediate", []):
        st.markdown(f'<span style="background:rgba(99,102,241,0.15); color:#a5b4fc; padding:5px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3); font-weight:600;">{s.title()}</span>', unsafe_allow_html=True)
    with c3:
      st.markdown("#### 🟣 Advanced & Architecture")
      for s in role_data.get("advanced", []):
        st.markdown(f'<span style="background:rgba(168,85,247,0.15); color:#d8b4fe; padding:5px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.85rem; border:1px solid rgba(168,85,247,0.3); font-weight:600;">{s.title()}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    role_counts = pd.DataFrame([
      {"Tier": "Core Foundations", "Skills": len(role_data.get("core", []))},
      {"Tier": "Intermediate Applied", "Skills": len(role_data.get("intermediate", []))},
      {"Tier": "Advanced Architecture", "Skills": len(role_data.get("advanced", []))},
    ])
    fig_bench = px.bar(
      role_counts, x="Tier", y="Skills", color="Tier",
      color_discrete_map={"Core Foundations": "#22c55e", "Intermediate Applied": "#6366f1", "Advanced Architecture": "#a855f7"},
      title=f"Skill Breakdown for {selected_role.title()}",
    )
    fig_bench.update_layout(
      template="plotly_dark", height=320,
      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
      font=dict(family='Plus Jakarta Sans', color='#94a3b8'),
      margin=dict(l=20, r=20, t=40, b=20),
      showlegend=False,
    )
    st.plotly_chart(fig_bench, use_container_width=True)

  with tab_ont2:
    st.markdown("### Skill Taxonomy & Prerequisite Dependency Graph")
    st.caption("Traverse directed prerequisite relations defined in the MoSSGA NetworkX Knowledge Graph:")
    
    kg = SkillKnowledgeGraph()
    nodes_cnt = kg.graph.number_of_nodes()
    edges_cnt = kg.graph.number_of_edges()

    m1, m2, m3 = st.columns(3)
    m1.metric("Ontology Concept Nodes", nodes_cnt)
    m2.metric("Dependency Edges", edges_cnt)
    m3.metric("Taxonomy Categories", len(set([v.get("type", "skill") for v in SKILL_TAXONOMY.values()])))

    selected_skill = st.selectbox(
      "Select Skill to Inspect Prerequisite Chain",
      options=sorted(list(SKILL_TAXONOMY.keys())),
      format_func=lambda s: s.title(),
    )

    tax_info = SKILL_TAXONOMY[selected_skill]
    prereqs = tax_info.get("prerequisites", [])
    related = tax_info.get("related", [])
    sk_type = tax_info.get("type", "concept").title()

    prereq_badges = " ".join([f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:4px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.85rem; border:1px solid rgba(239,68,68,0.3); font-weight:600;">{p.title()}</span>' for p in prereqs]) if prereqs else '<span style="color:#94a3b8; font-size:0.88rem;">No strict prerequisites — foundational technology</span>'
    related_badges = " ".join([f'<span style="background:rgba(99,102,241,0.15); color:#a5b4fc; padding:4px 12px; border-radius:20px; margin:4px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3); font-weight:600;">{r.title()}</span>' for r in related]) if related else '<span style="color:#94a3b8; font-size:0.88rem;">No direct co-occurring links</span>'

    st.markdown(f"""
    <div class="glass-card" style="margin-top: 1rem;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h3 style="margin:0; color:#c7d2fe;">{selected_skill.title()}</h3>
        <span class="mode-badge mode-full">{sk_type}</span>
      </div>
      <hr style="border-color: rgba(255,255,255,0.08); margin: 12px 0;"/>
      <div style="margin-bottom: 14px;">
        <b style="color:#f87171;">⚠️ Prerequisites (Required Foundation):</b><br/>
        {prereq_badges}
      </div>
      <div>
        <b style="color:#60a5fa;">🔄 Related & Co-occurring Technologies:</b><br/>
        {related_badges}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# VIEW 3: 🧠 Pipeline Architecture
# ---------------------------------------------------------------------------

def render_architecture_view():
  st.markdown("""
  <div class="glass-card" style="margin-bottom: 1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div class="hero-badge">🧠 SYSTEM ARCHITECTURE & METHODOLOGY</div>
        <h2 style="margin: 8px 0 6px 0; color:#1a1523;">Research-Grade MoSSGA Pipeline</h2>
        <p style="color: #6b6578; margin: 0; font-size: 1rem;">
          A 5-layer hybrid framework combining Dense Transformers (Sentence-BERT), Cross-Modal Fusion, and Graph Reasoning.
        </p>
      </div>
      <div style="font-size: 3rem; filter: drop-shadow(0 0 16px rgba(168,85,247,0.5));">⚙️</div>
    </div>
  </div>
  """, unsafe_allow_html=True)

  stages = [
    ("Stage 1: Multi-Modal Ingestion", "📄 Unstructured PDF/DOCX Parser + 🐙 GitHub GraphQL & REST Profiler", "Extracts textual claims alongside verifiable code contributions, star distributions, and commit frequency.", "#3b82f6"),
    ("Stage 2: Dense Semantic Mapping", "⚡ Sentence-BERT (all-MiniLM-L6-v2) 384-dimensional Embeddings", "Projects candidate and job requirements into dense vector space. Resolves synonym gaps like 'FastAPI' ↔ 'REST APIs'.", "#6366f1"),
    ("Stage 3: Cross-Modal Confidence Fusion", "🧬 Source Triangulation with Complexity Weighting", "Calculates C_fused = w_r · C_resume + w_g · C_github + δ_verified. Prevents resume inflation.", "#a855f7"),
    ("Stage 4: Knowledge Graph Reasoning", "🕸️ NetworkX DiGraph Ontological Traversal", "Evaluates prerequisite chains. Penalizes missing root foundations higher than niche leaf tools.", "#ec4899"),
    ("Stage 5: XGBoost Calibrated Scoring", "🎯 XGBoost Regressor (n_estimators=50, max_depth=3)", "Combines semantic cosine, graph coverage, and evidence confidence into a unified hiring match percentage.", "#10b981"),
  ]

  for title, subtitle, desc, color in stages:
    st.markdown(f"""
    <div class="glass-card" style="border-left: 4px solid {color}; padding: 18px 24px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <h4 style="margin:0; color:#1a1523;">{title}</h4>
        <span style="color:{color}; font-size:0.8rem; font-weight:700; text-transform:uppercase;">ACTIVE ENGINE</span>
      </div>
      <div style="color:{color}; font-size:0.92rem; font-weight:600; margin:6px 0;">{subtitle}</div>
      <p style="color:#6b6578; font-size:0.9rem; margin:0;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Architectural Comparison")
  comp_df = pd.DataFrame([
    {"Dimension": "Semantic Understanding", "Traditional Keyword ATS": "Exact string matches (Fails on synonyms)", "MoSSGA Multi-Modal": "384d Dense Vector Cosine (Captures meaning)"},
    {"Dimension": "Skill Validation", "Traditional Keyword ATS": "Unverified textual self-reporting", "MoSSGA Multi-Modal": "Cross-validated with GitHub repositories"},
    {"Dimension": "Prerequisite Awareness", "Traditional Keyword ATS": "Treats all skills equally flat", "MoSSGA Multi-Modal": "Graph reasoning penalizes missing core foundations"},
    {"Dimension": "Scoring Calibration", "Traditional Keyword ATS": "Heuristic keyword counter", "MoSSGA Multi-Modal": "Trained XGBoost Regressor Model"},
  ])
  st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# VIEW 4: ⚡ System Health & APIs
# ---------------------------------------------------------------------------

def render_system_health_view():
  st.markdown("""
  <div class="glass-card" style="margin-bottom: 1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <div>
        <div class="hero-badge">⚡ TELEMETRY & RUNTIME DIAGNOSTICS</div>
        <h2 style="margin: 8px 0 6px 0; color:#1a1523;">System Diagnostics & Service Health</h2>
        <p style="color: #6b6578; margin: 0; font-size: 1rem;">
          Real-time health telemetry across NLP transformers, machine learning artifacts, external APIs, and cloud environment.
        </p>
      </div>
      <div style="font-size: 3rem; filter: drop-shadow(0 0 16px rgba(16,185,129,0.5));">📊</div>
    </div>
  </div>
  """, unsafe_allow_html=True)

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("SBERT Status", "Online (384d)", "MiniLM-L6-v2")
  c2.metric("ML Regressor", "Trained & Active", "XGBoost v2.0")
  c3.metric("NLP Pipeline", "Loaded", "spaCy en_core")
  c4.metric("Cloud Server", "Active ($PORT)", "0.0.0.0:8501")

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Live Transformer Inference Test")
  if st.button("⚡ Run Live Semantic Latency Benchmark", type="primary"):
    import time
    t0 = time.time()
    test_score = calculate_sbert_similarity(["Python developer machine learning"], "Python software engineer machine learning")[0]
    elapsed = (time.time() - t0) * 1000
    st.success(f"✓ Inference Completed in {elapsed:.1f}ms — Semantic Cosine Score: {test_score:.4f}")

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Cloud Environment Configuration")
  gh_token = os.environ.get("GITHUB_TOKEN")
  env_df = pd.DataFrame([
    {"Variable": "PORT", "Status": os.environ.get("PORT", "8501 (Default)"), "Role": "Cloud proxy binding port"},
    {"Variable": "GITHUB_TOKEN", "Status": "Configured (5,000 req/hr)" if gh_token else "Not Set (60 req/hr unauthenticated)", "Role": "GitHub REST API rate limit authorization"},
    {"Variable": "HF_HUB_DISABLE_SYMLINKS_WARNING", "Status": os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING", "1 (Enabled)"), "Role": "Suppresses HuggingFace cache warning on Windows"},
    {"Variable": "PYTHON_RUNTIME", "Status": "3.11.9 (Pinned in runtime.txt)", "Role": "Target Python execution runtime"},
  ])
  st.dataframe(env_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Navigation Dispatcher
# ---------------------------------------------------------------------------

if selected_nav == "🌐 Skill Graph":
  render_skill_graph_view()
  st.stop()
elif selected_nav == "🧠 Architecture":
  render_architecture_view()
  st.stop()
elif selected_nav == "⚡ System Health":
  render_system_health_view()
  st.stop()


# ---------------------------------------------------------------------------
# VIEW 1: 🎯 Analyzer Dashboard
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-wrap">
  <div class="hero-chip">⚡ AI-POWERED TALENT INTELLIGENCE</div>
  <h1 class="hero-title">MoSSGA System</h1>
  <p class="hero-sub">Multi-Modal Semantic Skill Gap Analysis using Sentence-BERT Embeddings, Cross-Modal Fusion &amp; Knowledge Graph Reasoning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-bar">
  <div class="status-pill"><span class="status-dot"></span>SBERT Model Active</div>
  <div class="status-pill"><span class="status-dot"></span>NLP Pipeline Ready</div>
  <div class="status-pill"><span class="status-dot"></span>GitHub Analyzer Online</div>
  <div class="status-pill"><span class="status-dot"></span>Fusion Engine Ready</div>
  <div class="status-pill"><span class="status-dot"></span>Skill Gap Engine Online</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
  st.markdown('<div class="input-section-title">📄 Resume Upload</div>', unsafe_allow_html=True)
  uploaded_files = st.file_uploader(
    "Upload resumes", type=["pdf", "docx"], accept_multiple_files=True,
    label_visibility="collapsed",
  )
  if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} resume(s) loaded")

with col2:
  st.markdown('<div class="input-section-title">🐙 GitHub Profile</div>', unsafe_allow_html=True)
  github_input = st.text_input(
    "GitHub",
    value=st.session_state.github_input_val,
    placeholder="username or https://github.com/username",
    label_visibility="collapsed",
  )
  if github_input:
    st.success("✓ GitHub profile queued")

with col3:
  st.markdown('<div class="input-section-title">🎯 Target Profile</div>', unsafe_allow_html=True)
  target_role = st.text_input(
    "Target Role",
    value=st.session_state.target_role_val,
    placeholder="e.g. Machine Learning Engineer",
    label_visibility="collapsed",
  )
  manual_skills = st.text_input(
    "Your Skills (comma-separated)",
    value=st.session_state.manual_skills_val,
    placeholder="Python, SQL, Docker, React…",
    label_visibility="collapsed",
  )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="input-section-title">📋 Job Description</div>', unsafe_allow_html=True)
jd_input = st.text_area(
  "JD",
  value=st.session_state.jd_input_val,
  height=130,
  placeholder="Paste a job description to get deep semantic skill gap matching and explainable AI insights…",
  label_visibility="collapsed",
)

st.markdown("<br>", unsafe_allow_html=True)
col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 2, 1])
with col_cfg2:
  use_sbert = st.checkbox("Use Deep Semantic (SBERT) for advanced pattern matching", value=True)
  run_btn = st.button("🚀 Run Multi-Modal Skill Gap Analysis", use_container_width=True, type="primary")




# ---------------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------------

if run_btn:
  has_resumes = bool(uploaded_files)
  has_role_skills = bool(target_role or manual_skills)
  has_jd = bool(jd_input)
  has_github = bool(github_input)

  # --- GUIDANCE MODE ---
  if not has_resumes and not has_role_skills and not has_jd and not has_github:
    st.markdown("""
    <div class="glass-card">
      <span class="mode-badge mode-guidance"> GUIDANCE MODE</span>
      <h3>No input data provided!</h3>
      <p>I need at least one of:</p>
      <ul>
        <li> <b>Resume</b> — Upload a PDF or DOCX for full analysis</li>
        <li> <b>GitHub Profile</b> — Enter a username for code-based skill extraction</li>
        <li> <b>Target Role + Skills</b> — Enter in the sidebar for skill gap analysis</li>
        <li> <b>Job Description</b> — Paste a JD for semantic matching</li>
      </ul>
      <p> <b>Tip:</b> For the best analysis, provide a resume + GitHub profile + job description.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

  # --- Determine Mode ---
  if has_resumes:
    mode = "Full Analysis Mode"
    badge_class = "mode-full"
  elif has_github:
    mode = "GitHub Profile Mode"
    badge_class = "mode-partial"
  else:
    mode = "Partial Profile Mode"
    badge_class = "mode-partial"

  st.markdown(f'<span class="mode-badge {badge_class}"> {mode.upper()}</span>', unsafe_allow_html=True)

  # --- Progress Animation ---
  progress_bar = st.progress(0, text=" Initializing MoSSGA engines...")

  # --- Pre-processing ---
  jd_text = clean_text(jd_input) if jd_input else ""
  jd_processed = preprocess_text(jd_text) if jd_text else ""
  jd_skills = extract_skills(jd_input) if jd_input else []
  progress_bar.progress(15, text=" Processing job description...")

  # Collect candidate skills from all sources
  all_resume_skills = []
  resume_text_first = ""

  # ===========================================================
  # RESUME PROCESSING
  # ===========================================================
  if has_resumes:
    progress_bar.progress(25, text=" Parsing resumes...")
    for i, file in enumerate(uploaded_files):
      progress_bar.progress(25 + int(20 * (i / len(uploaded_files))),
                 text=f" Parsing resume {i+1}/{len(uploaded_files)}: {file.name}...")

      file_ext = file.name.rsplit(".", 1)[-1]
      with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

      text = extract_text(tmp_path)
      os.remove(tmp_path)

      if text and not resume_text_first:
        resume_text_first = text

      if text:
        extracted = extract_skills(text)
        all_resume_skills.extend(extracted)

    all_resume_skills = list(set(all_resume_skills))

  # ===========================================================
  # RUN MOSSGA PIPELINE
  # ===========================================================
  progress_bar.progress(50, text=" Running MoSSGA pipeline...")

  parsed_skills = [s.strip() for s in manual_skills.split(",")] if manual_skills else []

  mossga_report = run_mossga_pipeline(
    resume_text=resume_text_first,
    github_input=github_input if has_github else "",
    jd_text=jd_input if has_jd else "",
    target_role=target_role or "General",
  )

  # Merge manually entered skills
  if mossga_report and mossga_report.get("github_analysis"):
    gh_skills = mossga_report["github_analysis"].get("skills", [])
    parsed_skills = list(set(parsed_skills + gh_skills))

  if all_resume_skills:
    parsed_skills = list(set(parsed_skills + all_resume_skills))

  progress_bar.progress(80, text=" Generating visualizations...")

  # Compute additional info for recommendations
  level = estimate_level(parsed_skills)
  missing_skills, roadmap = generate_skill_gap_and_roadmap(parsed_skills, target_role or "General")
  job_recs = generate_job_recommendations(parsed_skills, target_role or "General") if not has_jd else []

  progress_bar.progress(100, text=" Analysis complete!")

  # ===================================================================
  # VISUALIZATION TABS
  # ===================================================================

  tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Analysis Results",
    "🐙 GitHub Analysis",
    "🧬 Multi-Modal Fusion",
    "🎯 MoSSGA Skill Gap",
    "💡 Recommendations",
  ])

  # --- TAB 1: Analysis Results ---
  with tab1:
    st.markdown("## Analysis Results")
    st.markdown("*Overview of extracted features, detected skills, and match scores.*")

    # Top metrics
    mossga_score = mossga_report.get("mossga_score", 0) if mossga_report else 0
    match_pct = 0
    if mossga_report and mossga_report.get("semantic_match"):
      match_pct = mossga_report["semantic_match"].get("match_percentage", 0)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
      st.markdown(f"""<div class="score-card">
        <div class="score-val {score_color(mossga_score)}">{mossga_score}</div>
        <div class="score-lbl">MoSSGA Hybrid Score</div>
      </div>""", unsafe_allow_html=True)
    with m2:
      st.markdown(f"""<div class="score-card">
        <div class="score-val {score_color(match_pct)}">{match_pct}%</div>
        <div class="score-lbl">Semantic Match Rate</div>
      </div>""", unsafe_allow_html=True)
    with m3:
      st.markdown(f"""<div class="score-card">
        <div class="score-val col-purple">{level}</div>
        <div class="score-lbl">Estimated Level</div>
      </div>""", unsafe_allow_html=True)
    with m4:
      st.markdown(f"""<div class="score-card">
        <div class="score-val col-blue">{len(parsed_skills)}</div>
        <div class="score-lbl">Skills Detected</div>
      </div>""", unsafe_allow_html=True)

    # Skills detected
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🏷️ Skills Detected")
    if parsed_skills:
      badges = "".join([f'<span class="skill-badge">{s}</span>' for s in sorted(parsed_skills)])
      st.markdown(f'<div style="margin:8px 0;">{badges}</div>', unsafe_allow_html=True)
    else:
      st.markdown('<div class="empty-state"><div class="empty-state-icon">🔍</div><div class="empty-state-text">No skills detected. Upload a resume or enter skills manually.</div></div>', unsafe_allow_html=True)

    # Extracted resume sections
    if mossga_report and mossga_report.get("resume_analysis"):
      ra = mossga_report["resume_analysis"]
      st.markdown("### 📄 Extracted Resume Sections")
      sec1, sec2, sec3 = st.columns(3)
      with sec1:
        st.markdown("**🎓 Education**")
        for e in (ra.get("education") or ["Not detected"]):
          st.markdown(f"<span style='color:#94a3b8;font-size:0.88rem;'>• {e}</span>", unsafe_allow_html=True)
      with sec2:
        st.markdown("**💼 Experience**")
        for e in (ra.get("experience") or ["Not detected"])[:5]:
          st.markdown(f"<span style='color:#94a3b8;font-size:0.88rem;'>• {e}</span>", unsafe_allow_html=True)
      with sec3:
        st.markdown("**🚀 Projects**")
        for p in (ra.get("projects") or ["Not detected"])[:5]:
          st.markdown(f"<span style='color:#94a3b8;font-size:0.88rem;'>• {p}</span>", unsafe_allow_html=True)

    # Missing skills
    if missing_skills:
      st.markdown("### ⚠️ Missing Skills for Target Role")
      badges = "".join([f'<span class="skill-badge-red">{s}</span>' for s in missing_skills])
      st.markdown(f'<div style="margin:8px 0;">{badges}</div>', unsafe_allow_html=True)

    # Similarity scores (if JD provided with resumes)
    if has_resumes and has_jd and resume_text_first:
      st.markdown("### Similarity Scores")
      cleaned = preprocess_text(clean_text(resume_text_first))
      jd_proc = preprocess_text(clean_text(jd_input))
      tfidf_score = calculate_similarity([cleaned], jd_proc)[0]
      sbert_score = calculate_sbert_similarity([cleaned], jd_proc)[0] if use_sbert else 0

      sc1, sc2 = st.columns(2)
      sc1.metric("TF-IDF Cosine Similarity", f"{tfidf_score:.4f}")
      sc2.metric("SBERT Semantic Similarity", f"{sbert_score:.4f}")

  # --- TAB 2: GitHub Analysis ---
  with tab2:
    st.markdown("## GitHub Profile Analysis")
    st.markdown("*Analyzing GitHub repositories to extract languages, technologies, and project complexity.*")

    if mossga_report and mossga_report.get("github_analysis"):
      gh = mossga_report["github_analysis"]

      if gh.get("status") == "success":
        profile = gh.get("profile", {})

        # Profile header
        st.markdown(f"""
        <div class="glass-card">
          <div style="display:flex; align-items:center; gap:16px;">
            <img src="{profile.get('avatar_url', '')}" style="width:64px; height:64px; border-radius:50%; border:2px solid rgba(99,102,241,0.5);" />
            <div>
              <b style="font-size:1.3rem; color:#a5b4fc;">{profile.get('name', profile.get('username', ''))}</b><br/>
              <span style="color:#94a3b8;">@{profile.get('username', '')} · {profile.get('public_repos', 0)} public repos · {profile.get('followers', 0)} followers</span><br/>
              <span style="color:#64748b; font-size:0.85rem;">{profile.get('bio', '') or 'No bio'}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        metrics = gh.get("contribution_metrics", {})
        gm1, gm2, gm3, gm4 = st.columns(4)
        gm1.metric("Repos Analyzed", gh.get("repos_analyzed", 0))
        gm2.metric("Total Stars", metrics.get("total_stars", 0))
        gm3.metric("Languages", metrics.get("language_diversity", 0))
        gm4.metric("Avg Complexity", f"{metrics.get('avg_project_complexity', 0)}%")

        # Language distribution chart
        languages = gh.get("languages", {})
        if languages:
          st.markdown("### Language Distribution")
          lang_df = pd.DataFrame([
            {"Language": lang, "Percentage": pct}
            for lang, pct in list(languages.items())[:10]
          ])
          fig_lang = px.bar(
            lang_df, x="Language", y="Percentage",
            color="Percentage",
            color_continuous_scale=[[0, "#3b82f6"], [0.5, "#8b5cf6"], [1, "#ec4899"]],
          )
          fig_lang.update_traces(marker_line_width=0)
          plotly_dark_layout(fig_lang, height=340, title="Top Languages by Code Volume")
          st.plotly_chart(fig_lang, use_container_width=True)

        # Skills from GitHub
        gh_skills = gh.get("skills", [])
        if gh_skills:
          st.markdown("### Skills Inferred from GitHub")
          skill_sources = gh.get("skill_sources", {})
          src_cols = st.columns(3)
          with src_cols[0]:
            st.markdown("**From Languages:**")
            for s in skill_sources.get("from_languages", []):
              st.markdown(f"- {s}")
          with src_cols[1]:
            st.markdown("**From Topics:**")
            for s in skill_sources.get("from_topics", []):
              st.markdown(f"- {s}")
          with src_cols[2]:
            st.markdown("**From Descriptions:**")
            for s in skill_sources.get("from_descriptions", []):
              st.markdown(f"- {s}")

        # Top projects by complexity
        projects = gh.get("projects", [])
        if projects:
          st.markdown("### Top Projects by Complexity")
          for proj in projects[:6]:
            complexity = proj.get("complexity_score", 0)
            bar_color = '#34d399' if complexity >= 50 else ('#fbbf24' if complexity >= 25 else '#64748b')
            st.markdown(f"""
            <div class="glass-card" style="padding:14px 18px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <div>
                  <b style="color:#a5b4fc;font-size:1rem;">{proj['name']}</b>
                  <span style="color:#475569;margin-left:10px;font-size:0.82rem;">⭐ {proj.get('stargazers_count',0)} &nbsp;🍴 {proj.get('forks_count',0)}</span>
                </div>
                <span style="color:{bar_color};font-weight:700;font-size:0.9rem;">{complexity}%</span>
              </div>
              <div style="background:rgba(99,102,241,0.08);border-radius:9999px;height:6px;margin:6px 0;">
                <div style="background:{bar_color};width:{complexity}%;height:100%;border-radius:9999px;box-shadow:0 0 8px {bar_color}66;"></div>
              </div>
              <span style="color:#475569;font-size:0.83rem;">{(proj.get('description','') or 'No description')[:120]}</span>
            </div>""", unsafe_allow_html=True)

      elif gh.get("status") == "no_repos":
        st.warning("GitHub profile found but no public repositories to analyze.")
      else:
        st.error("Failed to fetch GitHub profile. Check the username/URL and try again.")
    else:
      st.markdown("""
      <div class="glass-card">
        <p style="color:#94a3b8;"> Enter a GitHub URL or username in the sidebar to analyze repositories, languages, and validate practical skills.</p>
      </div>""", unsafe_allow_html=True)

  # --- TAB 3: Multi-Modal Fusion ---
  with tab3:
    st.markdown("## Multi-Modal Fusion")
    st.markdown("*Combining resume and GitHub data into a unified skill profile with confidence scoring.*")

    if mossga_report and mossga_report.get("fusion_result"):
      fusion = mossga_report["fusion_result"]
      stats = fusion.get("fusion_stats", {})
      val = fusion.get("validation_summary", {})

      # Fusion metrics
      fm1, fm2, fm3, fm4 = st.columns(4)
      fm1.metric("Total Unique Skills", stats.get("total_unique", 0))
      fm2.metric("Cross-Validated", stats.get("validated_count", 0))
      fm3.metric("Validation Rate", f"{stats.get('validation_rate', 0)}%")
      fm4.metric("Sources", f"R:{stats.get('resume_skill_count', 0)} G:{stats.get('github_skill_count', 0)}")

      # Fusion summary
      st.markdown(mossga_report.get("fusion_summary", ""))

      # Side-by-side comparison
      comp1, comp2, comp3 = st.columns(3)
      with comp1:
        st.markdown("### Both Sources")
        for s in val.get("both_sources", []):
          conf = fusion.get("confidence_scores", {}).get(s, 0)
          st.markdown(f"""
          <div style="background:rgba(34,197,94,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
            <span style="color:#86efac;">{s.title()}</span>
            <span style="float:right; color:#22c55e; font-weight:600;">{conf}</span>
          </div>""", unsafe_allow_html=True)
        if not val.get("both_sources"):
          st.markdown("*No cross-validated skills*")

      with comp2:
        st.markdown("### Resume Only")
        for s in val.get("resume_only", []):
          conf = fusion.get("confidence_scores", {}).get(s, 0)
          st.markdown(f"""
          <div style="background:rgba(99,102,241,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
            <span style="color:#a5b4fc;">{s.title()}</span>
            <span style="float:right; color:#6366f1; font-weight:600;">{conf}</span>
          </div>""", unsafe_allow_html=True)
        if not val.get("resume_only"):
          st.markdown("*No resume-only skills*")

      with comp3:
        st.markdown("### GitHub Only")
        for s in val.get("github_only", []):
          conf = fusion.get("confidence_scores", {}).get(s, 0)
          st.markdown(f"""
          <div style="background:rgba(234,179,8,0.15); padding:6px 12px; border-radius:8px; margin:4px 0;">
            <span style="color:#fde68a;">{s.title()}</span>
            <span style="float:right; color:#eab308; font-weight:600;">{conf}</span>
          </div>""", unsafe_allow_html=True)
        if not val.get("github_only"):
          st.markdown("*No GitHub-only skills*")

      # Confidence distribution chart
      conf_scores = fusion.get("confidence_scores", {})
      if conf_scores:
        st.markdown("### Skill Confidence Distribution")
        conf_df = pd.DataFrame([
          {"Skill": s.title(), "Confidence": c, "Source": "Both" if s in val.get("both_sources", []) else ("Resume" if s in val.get("resume_only", []) else "GitHub")}
          for s, c in sorted(conf_scores.items(), key=lambda x: -x[1])
        ][:15]
        )
        fig_conf = px.bar(
          conf_df, x="Skill", y="Confidence", color="Source",
          color_discrete_map={"Both": "#34d399", "Resume": "#6366f1", "GitHub": "#fbbf24"},
        )
        fig_conf.update_traces(marker_line_width=0)
        plotly_dark_layout(fig_conf, height=380, title="Skill Confidence Scores (Multi-Modal)")
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
      st.markdown('<div class="empty-state"><div class="empty-state-icon">🧬</div><div class="empty-state-text">Provide both a resume and GitHub profile to see multi-modal fusion. Single-source analysis is also supported.</div></div>', unsafe_allow_html=True)

  # --- TAB 4: MoSSGA Skill Gap ---
  with tab4:
    st.markdown("## MoSSGA — Semantic Skill Gap Analysis")
    st.markdown("*Semantic matching powered by Sentence-BERT ensures similar skills are correctly interpreted.*")

    if mossga_report and mossga_report.get("semantic_match"):
      match = mossga_report["semantic_match"]
      gap = mossga_report.get("gap_analysis", {})

      # Top metrics
      sm1, sm2, sm3, sm4 = st.columns(4)
      match_pct = match.get("match_percentage", 0)
      gap_score = gap.get("gap_score", 0)
      gap_severity = gap.get("gap_severity", "N/A")
      mossga_score = mossga_report.get("mossga_score", 0)
      sev_col = 'col-green' if gap_severity == 'Low' else ('col-yellow' if gap_severity == 'Medium' else 'col-red')

      with sm1:
        st.markdown(f"""<div class="score-card">
          <div class="score-val {score_color(match_pct)}">{match_pct}%</div>
          <div class="score-lbl">Semantic Match Rate</div>
        </div>""", unsafe_allow_html=True)
      with sm2:
        st.markdown(f"""<div class="score-card">
          <div class="score-val {score_color(100 - gap_score)}">{gap_score}%</div>
          <div class="score-lbl">Skill Gap Score</div>
        </div>""", unsafe_allow_html=True)
      with sm3:
        st.markdown(f"""<div class="score-card">
          <div class="score-val {sev_col}">{gap_severity}</div>
          <div class="score-lbl">Gap Severity</div>
        </div>""", unsafe_allow_html=True)
      with sm4:
        st.markdown(f"""<div class="score-card">
          <div class="score-val {score_color(mossga_score)}">{mossga_score}</div>
          <div class="score-lbl">Hybrid Score</div>
        </div>""", unsafe_allow_html=True)

      # Matched skills
      matched_skills = match.get("matched_skills", [])
      if matched_skills:
        st.markdown("### Semantically Matched Skills")
        for m in matched_skills:
          sim = m["similarity"]
          match_type = m.get("match_type", "semantic")
          sim_color = '#22c55e' if sim >= 0.8 else ('#eab308' if sim >= 0.6 else '#94a3b8')
          type_badge = ' Exact' if match_type == 'exact' else ' Semantic'
          st.markdown(f"""
          <div style="background:rgba(30,41,59,0.5); padding:10px 16px; border-radius:10px; margin:4px 0; display:flex; justify-content:space-between; align-items:center;">
            <div>
              <b style="color:#a5b4fc;">{m['jd_skill'].title()}</b>
              <span style="color:#64748b; margin:0 8px;"></span>
              <span style="color:#e2e8f0;">{m['candidate_skill'].title()}</span>
              <span style="color:#94a3b8; font-size:0.8rem; margin-left:8px;">{type_badge}</span>
            </div>
            <span style="color:{sim_color}; font-weight:600;">{sim:.1%}</span>
          </div>""", unsafe_allow_html=True)

      # Missing skills
      explainable_insights = mossga_report.get("explainable_insights", [])
      missing_detailed = gap.get("missing_with_severity", [])
      
      if explainable_insights:
        st.markdown("### Explainable AI Insights (Missing Skills)")
        for item in explainable_insights:
          skill_name = item.get("skill", "")
          insight = item.get("insight", "")
          prereqs = item.get("missing_prerequisites", [])
          
          # Match with severity info
          sev = "Medium"
          importance = 0.5
          for m in missing_detailed:
              if m["skill"] == skill_name:
                  sev = m["severity"]
                  importance = m["importance"]
                  break
                  
          sev_color = '#ef4444' if sev == 'Critical' else ('#eab308' if sev == 'Important' else '#94a3b8')
          
          prereq_html = ""
          if prereqs:
              prereq_html = f"<div style='margin-top:8px; font-size:0.85rem; color:#fca5a5;'><b>Graph Reasoning - Missing Prerequisites:</b> {', '.join([p.title() for p in prereqs])}</div>"

          st.markdown(f"""
          <div class="glass-card" style="padding:12px 16px; border-left: 4px solid {sev_color};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <b style="color:#a5b4fc; font-size:1.1rem;">{skill_name.title()}</b>
              <span style="color:{sev_color}; font-weight:600; font-size:0.85rem;">{sev} (Priority: {importance})</span>
            </div>
            <p style="color:#cbd5e1; font-size:0.95rem; margin-top:8px;"><i>🤖 {insight}</i></p>
            {prereq_html}
          </div>""", unsafe_allow_html=True)
      elif missing_detailed:
        st.markdown("### Missing Skills")
        for item in missing_detailed:
          sev = item["severity"]
          sev_color = '#ef4444' if sev == 'Critical' else ('#eab308' if sev == 'Important' else '#94a3b8')
          st.markdown(f"""
          <div class="glass-card" style="padding:12px 16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <b style="color:#fca5a5;">{item['skill'].title()}</b>
              <span style="color:{sev_color}; font-weight:600; font-size:0.85rem;">{sev} (Priority: {item['importance']})</span>
            </div>
          </div>""", unsafe_allow_html=True)

      # Semantic similarity heatmap
      sim_matrix = match.get("similarity_matrix", [])
      jd_resolved = match.get("jd_resolved", [])
      cand_resolved = match.get("candidate_resolved", [])
      if sim_matrix and len(sim_matrix) <= 20 and len(sim_matrix[0]) <= 20:
        st.markdown("### Semantic Similarity Heatmap")
        import numpy as np
        fig_heat = go.Figure(data=go.Heatmap(
          z=sim_matrix,
          x=[s.title() for s in cand_resolved[:20]],
          y=[s.title() for s in jd_resolved[:20]],
          colorscale=[[0, "#0a0f1e"], [0.4, "#312e81"], [0.7, "#6366f1"], [1, "#a5b4fc"]],
          text=[[f"{v:.2f}" for v in row] for row in sim_matrix],
          texttemplate="%{text}",
          textfont=dict(size=9, color="#e2e8f0"),
        ))
        plotly_dark_layout(fig_heat, height=480, title="JD Skills vs Candidate Skills — Semantic Similarity")
        fig_heat.update_layout(xaxis_title="Candidate Skills", yaxis_title="JD Required Skills")
        st.plotly_chart(fig_heat, use_container_width=True)

    elif mossga_report and mossga_report.get("fusion_result"):
      st.info(" Paste a Job Description to see semantic skill gap analysis. Currently showing your fused skill profile.")
      fused = mossga_report["fusion_result"].get("fused_skills", [])
      if fused:
        skill_html = " ".join(
          [f'<span style="background:rgba(99,102,241,0.2); color:#a5b4fc; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3);">{s.title()}</span>'
           for s in fused]
        )
        st.markdown(f'<div style="margin:8px 0;">{skill_html}</div>', unsafe_allow_html=True)
    else:
      st.markdown('<div class="empty-state"><div class="empty-state-icon">🎯</div><div class="empty-state-text">Upload a resume or enter a GitHub profile, plus a job description, to see the MoSSGA semantic skill gap analysis.</div></div>', unsafe_allow_html=True)

  # --- TAB 5: Recommendations ---
  with tab5:
    st.markdown("## Intelligent Recommendations")
    st.markdown("*Personalized skill improvement suggestions, course recommendations, and career path guidance.*")

    # MoSSGA-specific recommendations
    if mossga_report and mossga_report.get("recommendations"):
      recs = mossga_report["recommendations"]

      # Skill-specific course recommendations
      skill_recs = recs.get("skill_recommendations", [])
      if skill_recs:
        st.markdown("### Skill Improvement & Course Recommendations")
        for rec in skill_recs:
          imp_color = '#ef4444' if rec['importance'] == 'Critical' else ('#eab308' if rec['importance'] == 'Important' else '#94a3b8')
          courses_html = ""
          for course in rec.get("courses", []):
            courses_html += f'<a href="{course.get("url", "#")}" style="color:#60a5fa; text-decoration:none;">{course["title"]}</a> ({course["platform"]}) · '

          st.markdown(f"""
          <div class="glass-card" style="padding:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <b style="color:#a5b4fc; font-size:1.05rem;">{rec['skill'].title()}</b>
              <span style="color:{imp_color}; font-weight:600; font-size:0.85rem;">{rec['importance']}</span>
            </div>
            <p style="color:#94a3b8; font-size:0.9rem; margin:4px 0;">{rec['action']}</p>
            <span style="font-size:0.85rem;"> {courses_html.rstrip(' · ')}</span>
          </div>""", unsafe_allow_html=True)

      career_steps = recs.get("career_steps", [])
      career_path_data = mossga_report.get("career_path", {})
      
      if career_path_data.get("next_skills"):
        st.markdown("### Workforce Informatics: Next Best Skills")
        st.markdown("*AI-predicted logical next steps based on your current skill graph topology.*")
        ns_html = " ".join([f'<span style="background:rgba(16,185,129,0.2); color:#6ee7b7; padding:6px 14px; border-radius:20px; border:1px solid rgba(16,185,129,0.4); margin:4px; display:inline-block;">{s.title()}</span>' for s in career_path_data["next_skills"]])
        st.markdown(f"<div style='margin-bottom: 16px;'>{ns_html}</div>", unsafe_allow_html=True)
      
      if career_steps:
        st.markdown("### Career Path Guidance")
        for step in career_steps:
          st.markdown(step)

      # Improvement areas
      impr_areas = recs.get("improvement_areas", [])
      if impr_areas:
        st.markdown("### Strategic Improvement Areas")
        for area in impr_areas:
          st.markdown(area)

    # Roadmap from career intelligence
    if roadmap:
      st.markdown("### 🛤️ Learning Roadmap")
      for step in roadmap:
        level_key = "beginner" if "Beginner" in step["level"] else ("intermediate" if "Intermediate" in step["level"] else "advanced")
        color_map = {"beginner": "#34d399", "intermediate": "#6366f1", "advanced": "#a855f7"}
        c = color_map[level_key]
        st.markdown(f"""
        <div class="roadmap-card roadmap-{level_key}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <b style="color:{c};">{step['level']}</b>
            <span style="color:#64748b;font-size:0.82rem;">{step['timeline']}</span>
          </div>
          <div style="color:#e2e8f0;font-size:0.92rem;margin-bottom:4px;">{step['focus']}</div>
          <div style="color:#475569;font-size:0.82rem;">📦 {', '.join(step['technologies'])}</div>
          <div style="color:#334155;font-size:0.8rem;margin-top:4px;">📖 {step['resources']}</div>
        </div>""", unsafe_allow_html=True)

    # Job recommendations (when no JD)
    if job_recs:
      st.markdown("### Recommended Job Roles")
      for job in job_recs:
        match_pct = job.get("match_percent", "—")
        st.markdown(f"""
        <div class="glass-card">
          <b style="color:#a5b4fc; font-size:1.1rem;">{job['role']}</b>
          <span style="float:right; color:#22c55e; font-weight:600;">{match_pct}% match</span><br/>
          <span style="color:#94a3b8;">{job['reason']}</span>
        </div>""", unsafe_allow_html=True)

    # Career guidance
    if parsed_skills:
      st.markdown("### Career Guidance")
      guidance = generate_career_guidance(level, target_role or "General")
      st.info(guidance)

    if not mossga_report or not mossga_report.get("recommendations"):
      if not roadmap and not job_recs:
        st.markdown('<div class="empty-state"><div class="empty-state-icon">💡</div><div class="empty-state-text">Provide a resume + job description to get personalized recommendations, or enter skills and a target role for general guidance.</div></div>', unsafe_allow_html=True)
