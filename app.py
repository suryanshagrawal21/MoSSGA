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
  estimate_level,
  generate_skill_gap_and_roadmap,
  generate_job_recommendations,
  generate_career_guidance,
  extract_education_from_text,
  extract_experience_from_text,
  extract_projects_from_text,
)

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
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* Global background & text */
  .stApp {
    background-color: #080c14;
    background-image: 
      radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
      radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
      radial-gradient(at 50% 100%, rgba(14, 165, 233, 0.1) 0px, transparent 50%);
    background-attachment: fixed;
    color: #f1f5f9;
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  /* Sidebar styling */
  section[data-testid="stSidebar"] {
    background: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    font-weight: 700;
  }
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
  }

  /* Typography */
  h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em;
    color: #ffffff;
  }
  
  .main p, .main span, .main div {
    color: #cbd5e1;
  }

  /* Hero Banner */
  .hero-container {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    margin-bottom: 1rem;
  }
  .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 9999px;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 24px rgba(99, 102, 241, 0.2);
  }
  .hero-title {
    font-size: 3.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -1.5px !important;
    line-height: 1.15 !important;
    background: linear-gradient(135deg, #ffffff 0%, #c7d2fe 50%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.8rem !important;
  }
  .hero-subtitle {
    color: #94a3b8 !important;
    font-size: 1.15rem !important;
    max-width: 780px;
    margin: 0 auto !important;
    line-height: 1.6 !important;
    font-weight: 400;
  }

  /* Status Bar */
  .status-bar {
    display: flex;
    justify-content: center;
    gap: 20px;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    padding: 10px 24px;
    margin: 1.5rem auto 2.5rem auto;
    max-width: 950px;
    flex-wrap: wrap;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  }
  .status-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #cbd5e1;
    font-weight: 500;
    letter-spacing: 0.2px;
  }
  .status-dot-active {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-green 2s infinite ease-in-out;
  }
  @keyframes pulse-green {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px #10b981; }
    50% { opacity: 0.5; transform: scale(0.85); box-shadow: 0 0 3px #10b981; }
  }

  /* Glass Cards */
  .glass-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 24px;
    margin: 14px 0;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .glass-card:hover {
    border-color: rgba(99, 102, 241, 0.35);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.15);
    transform: translateY(-2px);
  }

  /* Mode badge */
  .mode-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.8px;
    margin: 8px 0;
    text-transform: uppercase;
  }
  .mode-full {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(168, 85, 247, 0.25));
    color: #c7d2fe;
    border: 1px solid rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.2);
  }
  .mode-partial {
    background: rgba(234, 179, 8, 0.15);
    color: #fde047;
    border: 1px solid rgba(234, 179, 8, 0.35);
  }
  .mode-guidance {
    background: rgba(148, 163, 184, 0.15);
    color: #cbd5e1;
    border: 1px solid rgba(148, 163, 184, 0.3);
  }

  /* Score Gauge Cards */
  .score-gauge {
    text-align: center;
    padding: 16px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    transition: all 0.25s ease;
  }
  .score-gauge:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
  }
  .score-value {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
  }
  .score-label {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-top: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* Dynamic Score Colors */
  .score-green {
    color: #34d399 !important;
    text-shadow: 0 0 16px rgba(52, 211, 153, 0.45);
  }
  .score-yellow {
    color: #fbbf24 !important;
    text-shadow: 0 0 16px rgba(251, 191, 36, 0.45);
  }
  .score-red {
    color: #f87171 !important;
    text-shadow: 0 0 16px rgba(248, 113, 113, 0.45);
  }

  /* Roadmap steps */
  .roadmap-step {
    background: rgba(17, 24, 39, 0.7);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 18px 22px;
    margin: 10px 0;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    transition: all 0.2s ease;
  }
  .roadmap-step:hover {
    background: rgba(17, 24, 39, 0.9);
    transform: translateX(4px);
  }
  .roadmap-beginner { border-left-color: #10b981; }
  .roadmap-intermediate { border-left-color: #6366f1; }
  .roadmap-advanced { border-left-color: #a855f7; }

  /* Metric cards override */
  [data-testid="stMetric"] {
    background: rgba(17, 24, 39, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
  }
  [data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700 !important;
  }
  [data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 10px 18px;
    border-radius: 8px;
    transition: all 0.2s ease;
  }
  .stTabs [data-baseweb="tab"]:hover {
    color: #ffffff;
    background: rgba(255, 255, 255, 0.04);
  }
  .stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15) !important;
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
  }

  /* Primary Button Styling */
  div.stButton > button[kind="primary"],
  div.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    box-shadow: 0 0 25px rgba(124, 58, 237, 0.4), 0 4px 10px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  }
  div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 0 35px rgba(124, 58, 237, 0.65), 0 8px 20px rgba(0, 0, 0, 0.4) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
  }

  /* Input fields */
  .stTextInput input, .stTextArea textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.25) !important;
    background: rgba(15, 23, 42, 0.8) !important;
  }
  [data-testid="stFileUploaderDropzone"] {
    background: rgba(15, 23, 42, 0.45) !important;
    border: 2px dashed rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
  }
  [data-testid="stFileUploaderDropzone"]:hover {
    border-color: #818cf8 !important;
    background: rgba(99, 102, 241, 0.08) !important;
  }
  
  .stTextInput label, .stTextArea label, [data-testid="stFileUploaderDropzone"] label {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header + System Status Bar
# ---------------------------------------------------------------------------

st.markdown("""
<div class="hero-container">
  <div class="hero-badge">⚡ AI-POWERED TALENT INTELLIGENCE</div>
  <h1 class="hero-title">MoSSGA System</h1>
  <p class="hero-subtitle">Multi-Modal Semantic Skill Gap Analysis · Sentence-BERT Embeddings & Cross-Modal Fusion</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="status-bar">
  <div class="status-item"><div class="status-dot-active"></div> SBERT Model Active</div>
  <div class="status-item"><div class="status-dot-active"></div> NLP Pipeline Ready</div>
  <div class="status-item"><div class="status-dot-active"></div> GitHub Analyzer Active</div>
  <div class="status-item"><div class="status-dot-active"></div> Fusion Engine Ready</div>
  <div class="status-item"><div class="status-dot-active"></div> Skill Gap Engine Online</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MoSSGA Hero Input Engine
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
  st.markdown("### 📄 Resume Upload")
  uploaded_files = st.file_uploader(
    "Upload PDF / DOCX resumes", type=["pdf", "docx"], accept_multiple_files=True,
    label_visibility="collapsed",
  )
  if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} resume(s) loaded")

with col2:
  st.markdown("### 🐙 GitHub Profile")
  github_input = st.text_input(
    "GitHub URL or Username",
    placeholder="e.g. https://github.com/username or username"
  )
  if github_input:
    st.success("✓ GitHub profile queued for analysis")

with col3:
  st.markdown("### 🎯 Target Profile")
  target_role = st.text_input("Target Role", placeholder="e.g. Machine Learning Engineer")
  manual_skills = st.text_input(
    "Your Skills (comma separated)",
    placeholder="e.g. Python, SQL, Docker"
  )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📋 Target Job Description")
jd_input = st.text_area(
  "Paste Target Job Description here", 
  height=140, 
  placeholder="Paste a job description to get deep semantic skill gap matching and explainable AI insights...",
  label_visibility="collapsed"
)

st.markdown("<br>", unsafe_allow_html=True)
col_cfg1, col_cfg2, col_cfg3 = st.columns([1, 2, 1])
with col_cfg2:
  use_sbert = st.checkbox("Use Deep Semantic (SBERT) for advanced pattern matching", value=True)
  run_btn = st.button("🚀 Run Multi-Modal Skill Gap Analysis", use_container_width=True, type="primary")



# ---------------------------------------------------------------------------
# Helper: Score color class
# ---------------------------------------------------------------------------

def score_color(val):
  """Returns CSS class for score coloring."""
  try:
    v = float(val)
  except (ValueError, TypeError):
    return "score-yellow"
  if v >= 70:
    return "score-green"
  elif v >= 40:
    return "score-yellow"
  return "score-red"


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
    m1, m2, m3, m4 = st.columns(4)
    with m1:
      mossga_score = mossga_report.get("mossga_score", 0) if mossga_report else 0
      st.markdown(f"""
      <div class="score-gauge">
        <div class="score-value {score_color(mossga_score)}">{mossga_score}</div>
        <div class="score-label">MoSSGA Hybrid Score</div>
      </div>""", unsafe_allow_html=True)
    with m2:
      match_pct = 0
      if mossga_report and mossga_report.get("semantic_match"):
        match_pct = mossga_report["semantic_match"].get("match_percentage", 0)
      st.markdown(f"""
      <div class="score-gauge">
        <div class="score-value {score_color(match_pct)}">{match_pct}%</div>
        <div class="score-label">Semantic Match Rate</div>
      </div>""", unsafe_allow_html=True)
    with m3:
      st.markdown(f"""
      <div class="score-gauge">
        <div class="score-value" style="color: #a78bfa;">{level}</div>
        <div class="score-label">Estimated Level</div>
      </div>""", unsafe_allow_html=True)
    with m4:
      st.markdown(f"""
      <div class="score-gauge">
        <div class="score-value" style="color: #60a5fa;">{len(parsed_skills)}</div>
        <div class="score-label">Skills Detected</div>
      </div>""", unsafe_allow_html=True)

    # Skills detected
    st.markdown("### Skills Detected")
    if parsed_skills:
      skill_html = " ".join(
        [f'<span style="background:rgba(99,102,241,0.2); color:#a5b4fc; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(99,102,241,0.3);">{s}</span>'
         for s in parsed_skills]
      )
      st.markdown(f'<div style="margin:8px 0;">{skill_html}</div>', unsafe_allow_html=True)
    else:
      st.info("No skills detected. Please upload a resume or enter skills manually.")

    # Extracted resume sections
    if mossga_report and mossga_report.get("resume_analysis"):
      ra = mossga_report["resume_analysis"]
      sec1, sec2, sec3 = st.columns(3)
      with sec1:
        st.markdown("### Education")
        for e in (ra.get("education") or ["Not detected"]):
          st.markdown(f"- {e}")
      with sec2:
        st.markdown("### Experience")
        for e in (ra.get("experience") or ["Not detected"])[:5]:
          st.markdown(f"- {e}")
      with sec3:
        st.markdown("### Projects")
        for p in (ra.get("projects") or ["Not detected"])[:5]:
          st.markdown(f"- {p}")

    # Missing skills
    if missing_skills:
      st.markdown("### Missing Skills for Target Role")
      miss_html = " ".join(
        [f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:4px 12px; border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem; border:1px solid rgba(239,68,68,0.3);">{s}</span>'
         for s in missing_skills]
      )
      st.markdown(f'<div style="margin:8px 0;">{miss_html}</div>', unsafe_allow_html=True)

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
            color="Percentage", color_continuous_scale="Viridis",
            title="Top Languages by Code Volume",
          )
          fig_lang.update_layout(
            template="plotly_dark",
            height=360,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Plus Jakarta Sans', color='#94a3b8'),
            margin=dict(l=20, r=20, t=40, b=20),
          )
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
            bar_color = '#22c55e' if complexity >= 50 else ('#eab308' if complexity >= 25 else '#94a3b8')
            st.markdown(f"""
            <div class="glass-card" style="padding:14px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <b style="color:#a5b4fc;">{proj['name']}</b>
                  <span style="color:#94a3b8; margin-left:8px; font-size:0.85rem;">Stars: {proj.get('stargazers_count', 0)} · Forks: {proj.get('forks_count', 0)}</span>
                </div>
                <span style="color:{bar_color}; font-weight:600;">Complexity: {complexity}%</span>
              </div>
              <div style="background:rgba(0,0,0,0.3); border-radius:8px; height:8px; margin:8px 0;">
                <div style="background:{bar_color}; width:{complexity}%; height:100%; border-radius:8px;"></div>
              </div>
              <span style="color:#64748b; font-size:0.85rem;">{proj.get('description', 'No description')[:120]}</span>
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
          color_discrete_map={"Both": "#22c55e", "Resume": "#6366f1", "GitHub": "#eab308"},
          title="Skill Confidence Scores (Multi-Modal)",
        )
        fig_conf.update_layout(
          template="plotly_dark",
          height=400,
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          font=dict(family='Plus Jakarta Sans', color='#94a3b8'),
          margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
      st.markdown("""
      <div class="glass-card">
        <p style="color:#94a3b8;"> Provide both a resume and GitHub profile to see multi-modal fusion analysis. Single-source analysis is also available.</p>
      </div>""", unsafe_allow_html=True)

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

      sm1.markdown(f"""
      <div class="score-gauge">
        <div class="score-value {score_color(match_pct)}">{match_pct}%</div>
        <div class="score-label">Semantic Match Rate</div>
      </div>""", unsafe_allow_html=True)
      sm2.markdown(f"""
      <div class="score-gauge">
        <div class="score-value {score_color(100 - gap_score)}">{gap_score}%</div>
        <div class="score-label">Skill Gap Score</div>
      </div>""", unsafe_allow_html=True)
      sm3.markdown(f"""
      <div class="score-gauge">
        <div class="score-value" style="color:{'#22c55e' if gap_severity in ('Low',) else '#eab308' if gap_severity in ('Medium',) else '#ef4444'};">{gap_severity}</div>
        <div class="score-label">Gap Severity</div>
      </div>""", unsafe_allow_html=True)
      sm4.markdown(f"""
      <div class="score-gauge">
        <div class="score-value {score_color(mossga_score)}">{mossga_score}</div>
        <div class="score-label">Hybrid Score</div>
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
          colorscale='Viridis',
          text=[[f"{v:.2f}" for v in row] for row in sim_matrix],
          texttemplate="%{text}",
          textfont=dict(size=9),
        ))
        fig_heat.update_layout(
          template='plotly_dark',
          height=500,
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          font=dict(family='Plus Jakarta Sans', color='#94a3b8'),
          margin=dict(l=20, r=20, t=50, b=20),
          title='JD Skills vs Candidate Skills — Semantic Similarity',
          xaxis_title='Candidate Skills',
          yaxis_title='JD Required Skills',
        )
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
      st.markdown("""
      <div class="glass-card">
        <p style="color:#94a3b8;"> Upload a resume or enter a GitHub profile, plus a job description, to see the MoSSGA semantic skill gap analysis.</p>
      </div>""", unsafe_allow_html=True)

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
      st.markdown("### Learning Roadmap")
      for step in roadmap:
        level_key = "beginner" if "Beginner" in step["level"] else ("intermediate" if "Intermediate" in step["level"] else "advanced")
        st.markdown(f"""
        <div class="roadmap-step roadmap-{level_key}">
          <b>{step['level']}</b> — {step['focus']}<br/>
          <span style="color:#94a3b8;"> {step['timeline']} | {', '.join(step['technologies'])}</span><br/>
          <span style="color:#64748b; font-size:0.8rem;"> {step['resources']}</span>
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
        st.markdown("""
        <div class="glass-card">
          <p style="color:#94a3b8;"> Provide a resume + job description to get personalized recommendations, or enter skills and a target role for general guidance.</p>
        </div>""", unsafe_allow_html=True)
