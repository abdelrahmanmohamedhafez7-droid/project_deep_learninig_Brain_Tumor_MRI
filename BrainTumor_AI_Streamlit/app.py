import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from src.inference import load_brain_tumor_model, predict_image, MODEL_PATH

st.set_page_config(
    page_title="BrainTumor AI | MRI Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The selected appearance is read before the sidebar control renders, so the
# complete app updates on the same Streamlit rerun.
theme_mode = "Dark"

# ---------- Theme / CSS ----------
st.markdown("""
<style>
:root {
    --navy: #07111f;
    --navy-2: #0c1a2b;
    --blue: #2f6df6;
    --blue-2: #6c8cff;
    --ink: #102033;
    --muted: #617087;
    --line: #dfe6ef;
    --surface: #ffffff;
    --bg: #f5f8fc;
    --danger: #d83a4b;
    --success: #16855b;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111f 0%, #0b1728 55%, #081421 100%);
    border-right: 1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
    color: #edf4ff !important;
}

.brand {
    padding: 8px 8px 22px 8px;
    border-bottom: 1px solid rgba(255,255,255,.10);
    margin-bottom: 16px;
}

.brand-mark {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #1c4fd7, #7c5cff);
    font-size: 29px;
    box-shadow: 0 10px 30px rgba(47,109,246,.22);
}

.brand-title {
    font-size: 21px;
    font-weight: 800;
    letter-spacing: -.4px;
    margin-top: 10px;
}

.brand-title span {
    color: #6c8cff;
}

.brand-sub {
    color: #9eb0c8 !important;
    font-size: 12px;
    margin-top: -2px;
}

.sidebar-quote {
    color: #9eb0c8 !important;
    font-size: 12px;
    line-height: 1.6;
    margin-top: 30px;
    padding: 14px 2px;
    text-align: center;
}

.hero {
    min-height: 235px;
    padding: 34px 42px 30px;
    border-radius: 24px;
    background:
        radial-gradient(circle at 84% 45%, rgba(84,124,255,.28), transparent 25%),
        radial-gradient(circle at 62% 10%, rgba(112,80,255,.16), transparent 28%),
        linear-gradient(125deg, #07111f 0%, #0d1c31 62%, #101c34 100%);
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 45px rgba(7,17,31,.18);
}

.hero:after {
    content: "🧠";
    position: absolute;
    right: 6%;
    top: 18px;
    font-size: 150px;
    opacity: .13;
    filter: grayscale(1);
}

.eyebrow {
    color: #9fb9ff;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: 48px;
    line-height: 1;
    margin: 8px 0 10px;
    letter-spacing: -2px;
}

.hero h1 span {
    color: #6c8cff;
}

.hero p {
    color: #d2deef;
    max-width: 660px;
    font-size: 16px;
    line-height: 1.65;
}

.hero-pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.hero-pill {
    padding: 9px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.11);
    color: #e8effb;
    font-size: 12px;
}

.card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 8px 28px rgba(16,32,51,.055);
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 4px;
}

.section-sub {
    color: var(--muted);
    font-size: 13px;
    margin-bottom: 15px;
}

.result-card {
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #f3c7cd;
    background: linear-gradient(135deg, #fff9fa, #fff1f3);
}

.result-ok {
    border-color: #bfe4d4;
    background: linear-gradient(135deg, #f8fffc, #effbf6);
}

.result-label {
    color: #68758a;
    font-size: 13px;
    font-weight: 700;
}

.result-main {
    font-size: 30px;
    font-weight: 900;
    margin: 3px 0 8px;
    letter-spacing: -.8px;
}

.result-danger { color: #c93245; }
.result-success { color: #137952; }

.metric-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 17px;
    min-height: 112px;
}

.metric-kicker {
    color: #738197;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .4px;
}

.metric-value {
    color: #132238;
    font-size: 21px;
    font-weight: 850;
    margin-top: 7px;
}

.metric-note {
    color: #718096;
    font-size: 11px;
    margin-top: 2px;
}

.disclaimer {
    padding: 16px 18px;
    border-radius: 14px;
    background: #fff9e8;
    border: 1px solid #f0dfac;
    color: #5e4b17;
    font-size: 13px;
    line-height: 1.6;
}

.info-box {
    padding: 17px;
    border-radius: 15px;
    background: #f3f7ff;
    border: 1px solid #d9e4ff;
    color: #2c4163;
    line-height: 1.65;
    font-size: 13px;
}

.footer {
    color: #718096;
    text-align: center;
    padding: 35px 0 12px;
    font-size: 12px;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #9bb2d7;
    border-radius: 16px;
    padding: 7px;
    background: #fbfdff;
}

div.stButton > button {
    border-radius: 11px;
    font-weight: 750;
    border: 1px solid #d5deeb;
    min-height: 42px;
}

div.stDownloadButton > button {
    border-radius: 11px;
    font-weight: 750;
}

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 12px;
}

.small-note {
    color: #718096;
    font-size: 11px;
    line-height: 1.5;
}

/* ---------- Dark Navy / Blue-Purple visual refresh ---------- */
html, body, [class*="css"] {
    font-family: Inter, "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 80% -10%, rgba(91,124,255,.16), transparent 28rem),
        radial-gradient(circle at 18% 20%, rgba(155,108,255,.09), transparent 25rem),
        #070b18;
    color: #f5f7ff;
}

[data-testid="stHeader"] { background: transparent; }

.block-container {
    max-width: 1440px;
    padding: 2.2rem 3rem 2.5rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070b18 0%, #0b1124 55%, #090d1c 100%);
    border-right: 1px solid rgba(160,181,255,.12);
}

[data-testid="stSidebar"] > div:first-child { padding: 1.25rem .95rem; }
[data-testid="stSidebar"] * { color: #edf2ff !important; }

.brand {
    padding: .45rem .5rem 1.5rem;
    border-bottom: 1px solid rgba(160,181,255,.14);
    margin-bottom: 1.2rem;
}

.brand-mark {
    width: 58px;
    height: 58px;
    border-radius: 18px;
    background: linear-gradient(135deg, #5375ff, #9a63ff);
    font-size: 30px;
    box-shadow: 0 12px 30px rgba(83,117,255,.32);
}

.brand-title { font-size: 1.42rem; font-weight: 850; letter-spacing: -.7px; margin-top: .8rem; }
.brand-title span { color: #a88aff; }
.brand-sub { color: #aab6d1 !important; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; margin-top: .15rem; }

[data-testid="stSidebar"] .stRadio label {
    padding: .52rem .65rem;
    border-radius: 10px;
    font-size: .93rem;
    transition: background .2s ease;
}

[data-testid="stSidebar"] .stRadio label:hover { background: rgba(117,141,255,.13); }

.hero {
    min-height: 270px;
    padding: 3rem 3.25rem;
    border: 1px solid rgba(150,170,255,.18);
    border-radius: 26px;
    background:
        radial-gradient(circle at 88% 40%, rgba(91,124,255,.34), transparent 23%),
        radial-gradient(circle at 68% 5%, rgba(155,108,255,.22), transparent 28%),
        linear-gradient(125deg, #0a1022, #121d3a 58%, #151d40);
    box-shadow: 0 24px 60px rgba(0,0,0,.28);
}

.hero:after { font-size: 10rem; right: 5%; top: 18px; }
.eyebrow { color: #92aeff; font-size: .76rem; font-weight: 800; letter-spacing: .13em; }
.hero h1 { color: #fff; font-size: clamp(2.35rem, 4vw, 4rem); margin: .65rem 0 .8rem; letter-spacing: -2.5px; }
.hero h1 span { color: #a88aff; }
.hero h2 { font-size: clamp(1rem, 1.5vw, 1.35rem) !important; font-weight: 600; }
.hero p { color: #d6def4; max-width: 680px; font-size: 1rem; line-height: 1.75; }
.hero-pill-row { gap: .6rem; margin-top: 1.45rem; }
.hero-pill { padding: .55rem .82rem; background: rgba(255,255,255,.075); border-color: rgba(255,255,255,.13); color: #edf2ff; font-size: .78rem; }

.card, .metric-card {
    background: linear-gradient(145deg, rgba(21,31,56,.98), rgba(14,22,42,.98));
    border: 1px solid rgba(160,181,255,.16);
    border-radius: 19px;
    box-shadow: 0 12px 32px rgba(0,0,0,.16);
}

.card { padding: 1.45rem; }
.section-title { color: #f5f7ff; font-size: 1.12rem; font-weight: 800; letter-spacing: -.3px; margin-bottom: .3rem; }
.section-sub, .small-note { color: #aab6d1; }
.section-sub { font-size: .82rem; margin-bottom: 1.05rem; }

.metric-card { padding: 1.15rem; min-height: 122px; }
.metric-kicker { color: #aebbe0; font-size: .7rem; font-weight: 800; letter-spacing: .08em; }
.metric-value { color: #fff; font-size: 1.25rem; font-weight: 850; margin-top: .55rem; }
.metric-note { color: #91a0c1; font-size: .72rem; margin-top: .22rem; }

.result-card { border-color: rgba(255,113,141,.34); background: linear-gradient(135deg, rgba(115,27,52,.32), rgba(37,17,34,.56)); }
.result-ok { border-color: rgba(54,211,153,.32); background: linear-gradient(135deg, rgba(18,98,76,.30), rgba(12,42,42,.56)); }
.result-label { color: #b7c3df; }
.result-main { font-size: 1.65rem; }
.result-danger { color: #ff8ba1; }
.result-success { color: #55e4ab; }

.info-box { background: rgba(76,111,225,.13); border: 1px solid rgba(112,145,255,.25); color: #d7e1ff; }
.disclaimer { background: rgba(155,108,255,.11); border: 1px solid rgba(184,139,255,.26); color: #e9ddff; }
.info-box, .disclaimer { padding: 1rem 1.1rem; border-radius: 14px; font-size: .88rem; line-height: 1.7; }

div[data-testid="stFileUploader"] { background: rgba(10,16,33,.5); border-color: rgba(144,169,255,.55); border-radius: 15px; padding: .65rem; }
div[data-testid="stFileUploader"]:hover { border-color: #819dff; background: rgba(73,100,210,.12); }

div.stButton > button, div.stDownloadButton > button {
    min-height: 44px;
    border-radius: 12px;
    border: 1px solid rgba(131,155,255,.48);
    background: linear-gradient(135deg, #5276ff, #805ff1);
    color: white;
    font-weight: 750;
    transition: transform .18s ease, box-shadow .18s ease;
}

div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(88,112,255,.32);
    border-color: transparent;
}

[data-testid="stMetric"] { background: #10182d; border: 1px solid rgba(160,181,255,.16); border-radius: 14px; padding: .85rem; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid rgba(160,181,255,.16); border-radius: 12px; overflow: hidden; }

.social-links { display: flex; gap: .7rem; margin: 1rem 0 1.45rem; }
.social-icon {
    width: 42px;
    height: 42px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 12px;
    background: rgba(130,151,255,.12);
    border: 1px solid rgba(150,170,255,.22);
    color: #eef2ff !important;
    transition: .2s ease;
}
.social-icon:hover { background: linear-gradient(135deg, #5276ff, #8c63ff); border-color: transparent; transform: translateY(-3px); }
.social-icon svg { width: 21px; height: 21px; fill: currentColor; }

.footer { color: #8f9cbd; padding: 2.7rem 0 .7rem; font-size: .76rem; }

@media (max-width: 900px) {
    .block-container { padding: 1.15rem 1rem 2rem; }
    .hero { min-height: auto; padding: 2rem 1.4rem; border-radius: 20px; }
    .hero:after { font-size: 7rem; right: -4%; }
}

@media (max-width: 600px) {
    .hero h1 { letter-spacing: -1.5px; }
    .hero p { font-size: .9rem; }
    .card { padding: 1.1rem; }
    .hero-pill { font-size: .7rem; }
}
/* ---------- Final polish: native Streamlit controls and readable contrast ---------- */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 90% -5%, var(--glow-one), transparent 30rem),
        radial-gradient(circle at 5% 30%, var(--glow-two), transparent 26rem),
        var(--app-bg) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sidebar-top), var(--sidebar-bottom)) !important;
    border-right-color: var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: var(--field-bg) !important; border-color: var(--border) !important; }

.theme-label {
    color: var(--sidebar-muted) !important;
    font-size: .7rem;
    font-weight: 800;
    letter-spacing: .1em;
    margin: 1.25rem 0 .4rem;
    text-transform: uppercase;
}

.brand-sub, .sidebar-quote { color: var(--sidebar-muted) !important; }
.sidebar-quote {
    font-size: .86rem;
    font-weight: 650;
    line-height: 1.75;
    letter-spacing: .01em;
    margin: 2.5rem .15rem .5rem;
    padding: 1rem .7rem;
    border-radius: 14px;
    background: var(--quote-bg);
    border: 1px solid var(--border);
}

.sidebar-meta {
    color: var(--sidebar-muted) !important;
    font-size: .76rem;
    line-height: 1.8;
    text-align: center;
}
.heart { color: #ff4d6d !important; }

.hero, .card, .metric-card, [data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card-bg) !important;
    border-color: var(--border) !important;
    box-shadow: 0 16px 36px var(--shadow) !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 20px !important;
    padding: .25rem !important;
}

.hero {
    background:
        radial-gradient(circle at 88% 40%, var(--hero-glow), transparent 23%),
        radial-gradient(circle at 68% 5%, var(--hero-glow-two), transparent 28%),
        var(--hero-bg) !important;
}

.section-title, .metric-value, .result-main, .card b, .card strong { color: var(--text) !important; }
.section-sub, .small-note, .metric-kicker, .metric-note, .card p[style], .card li, .card td { color: var(--muted-text) !important; }
.card table { color: var(--muted-text) !important; }
.card table td { padding: .45rem 0; border-bottom: 1px solid var(--border); }
.card table tr:last-child td { border-bottom: 0; }
.model-table { width: 100%; border-collapse: separate; border-spacing: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 12px; font-size: .8rem; }
.model-table th { background: var(--button-soft); color: var(--text); font-weight: 800; }
.model-table th, .model-table td { padding: .68rem .55rem; text-align: left; border-right: 1px solid var(--border); border-bottom: 1px solid var(--border); color: var(--muted-text); }
.model-table th:last-child, .model-table td:last-child { border-right: 0; }
.model-table tr:last-child td { border-bottom: 0; }

div[data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"] {
    background: var(--field-bg) !important;
    border-color: var(--accent) !important;
    border-radius: 15px !important;
}

[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] p { background: transparent !important; color: var(--muted-text) !important; }

[data-testid="stFileUploaderDropzone"] button {
    min-height: 38px !important;
    padding: .35rem .85rem !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    background: var(--button-soft) !important;
    color: var(--text) !important;
}

div[style*="height:280px"] {
    min-height: 280px;
    background: var(--preview-bg) !important;
    border: 1px dashed var(--accent) !important;
    color: var(--muted-text) !important;
    font-size: .95rem;
}

[data-testid="stAlert"] {
    border-radius: 14px !important;
    background: var(--info-bg) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent-two)) !important;
    color: #fff !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] * { color: var(--table-text) !important; }
[data-testid="stDataFrame"] [role="grid"] { background: var(--table-bg) !important; }

.footer { color: var(--muted-text) !important; }

@media (max-width: 900px) {
    .block-container { padding: 1rem .85rem 2rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { margin-bottom: .3rem; }
}
""" + ("""
:root {
    --app-bg: #070b18; --card-bg: linear-gradient(145deg, #151f38, #0e162a);
    --text: #f5f7ff; --muted-text: #aebbe0; --border: rgba(160,181,255,.20);
    --field-bg: #101a31; --preview-bg: #090f20; --table-bg: #10182d; --table-text: #eef2ff;
    --accent: #7894ff; --accent-two: #9b6cff; --button-soft: #1b294b;
    --glow-one: rgba(91,124,255,.16); --glow-two: rgba(155,108,255,.09);
    --hero-bg: linear-gradient(125deg, #0a1022, #121d3a 58%, #151d40);
    --hero-glow: rgba(91,124,255,.34); --hero-glow-two: rgba(155,108,255,.22);
    --sidebar-top: #070b18; --sidebar-bottom: #090d1c; --sidebar-text: #edf2ff;
    --sidebar-muted: #aab6d1; --quote-bg: rgba(116,145,255,.10); --info-bg: #0d2347;
    --shadow: rgba(0,0,0,.22);
}
""" if theme_mode == "Dark" else """
:root {
    --app-bg: #eef3ff; --card-bg: linear-gradient(145deg, #ffffff, #f5f8ff);
    --text: #14213d; --muted-text: #53647f; --border: #d8e1f2;
    --field-bg: #f9fbff; --preview-bg: #f5f8fe; --table-bg: #ffffff; --table-text: #17233d;
    --accent: #516fe8; --accent-two: #855ae8; --button-soft: #eef2ff;
    --glow-one: rgba(91,124,255,.14); --glow-two: rgba(155,108,255,.10);
    --hero-bg: linear-gradient(125deg, #10245c, #304c9f 58%, #5c43a8);
    --hero-glow: rgba(175,205,255,.38); --hero-glow-two: rgba(225,194,255,.20);
    --sidebar-top: #ffffff; --sidebar-bottom: #edf2ff; --sidebar-text: #17233d;
    --sidebar-muted: #5d6d88; --quote-bg: #f3f6ff; --info-bg: #edf3ff;
    --shadow: rgba(39,58,111,.10);
}
""") + """
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-mark">🧠</div>
        <div class="brand-title">BrainTumor <span>AI</span></div>
        <div class="brand-sub">MRI Classification</div>
    </div>
    """, unsafe_allow_html=True)

    

    page = st.radio(
        "Navigation",
        ["Home", "About", "Model Info", "How It Works", "Limitations & Disclaimer", "Contact"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**EXTERNAL LINKS**")
    st.markdown("""
    <div class="social-links">
        <a class="social-icon" href="https://github.com/abdelrahmanmohamedhafez7-droid/project_deep_learninig_Brain_Tumor_MRI" target="_blank" title="GitHub" aria-label="GitHub">
            <svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.7 7.7 0 0 1 8 4.8c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
        </a>
        <a class="social-icon" href="https://eg.linkedin.com/in/abdelrahman-mohamed-hafez-41bbb7397" target="_blank" title="LinkedIn" aria-label="LinkedIn">
            <svg viewBox="0 0 16 16"><path d="M0 1.15C0 .51.53 0 1.19 0h13.62C15.47 0 16 .51 16 1.15v13.7c0 .64-.53 1.15-1.19 1.15H1.19C.53 16 0 15.49 0 14.85V1.15Zm4.94 12.54V6.17H2.44v7.52h2.5ZM3.69 5.14c.87 0 1.57-.7 1.57-1.57A1.57 1.57 0 0 0 3.69 2a1.57 1.57 0 0 0 0 3.14Zm10.12 8.55V9.57c0-2.21-.47-3.91-3.05-3.91-1.24 0-2.07.68-2.41 1.32h-.04V6.17H5.92v7.52h2.49V9.97c0-.98.19-1.93 1.4-1.93 1.2 0 1.22 1.12 1.22 1.99v3.66h2.78Z"/></svg>
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="sidebar-quote">“AI today<br>for a healthier<br>tomorrow”</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-meta">Version 1.0.0<br>Made with <span class="heart">♥</span> by Abdelrahman Mohamed</div>', unsafe_allow_html=True)

# ---------- Helpers ----------
def render_metric_cards():
    cols = st.columns(4)
    metrics = [
        ("Model Used", "VGG16", "Transfer Learning"),
        ("Input Size", "128 × 128 × 3", "RGB"),
        ("Prediction Time", "< 1 sec*", "depends on runtime"),
        ("Test Accuracy", "91.56%", "recorded test split"),
    ]
    for col, (k, v, n) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-kicker">{k}</div>'
                f'<div class="metric-value">{v}</div><div class="metric-note">{n}</div></div>',
                unsafe_allow_html=True,
            )

def apply_plotly_theme(fig):
    """Keep Plotly charts visually integrated with the selected UI theme."""
    dark = theme_mode == "Dark"
    paper = "#10182d" if dark else "#ffffff"
    grid = "rgba(173,190,235,.16)" if dark else "#dce4f3"
    text = "#edf2ff" if dark else "#17233d"
    muted = "#aebbe0" if dark else "#5b6b86"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper,
        plot_bgcolor=paper,
        font=dict(color=text, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=28, r=24, t=28, b=42),
        xaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(color=muted), title_font=dict(color=muted)),
        yaxis=dict(gridcolor=grid, linecolor=grid, tickfont=dict(color=muted), title_font=dict(color=muted)),
    )
    return fig

def download_report(pred):
    report = f"""BrainTumor AI — Prediction Report
================================

Prediction: {pred["label"]}
Tumor probability: {pred["tumor_probability"]:.2%}
No-tumor probability: {pred["no_tumor_probability"]:.2%}

Model: {pred["model_name"]}
Input: 128 x 128 RGB
Recorded model test accuracy: 91.56%

Important:
This result is generated by an AI image-classification model and is not a medical diagnosis.
Clinical interpretation should be performed by a qualified healthcare professional.
"""
    return report.encode("utf-8")

# ---------- HOME ----------
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">Artificial Intelligence for Medical Imaging</div>
        <h1>BrainTumor <span>AI</span></h1>
        <h2 style="margin:0;color:#f3f7ff;font-size:23px;">MRI Brain Tumor Classification</h2>
        <p>Upload a brain MRI image to receive an AI-based classification result indicating whether the image is predicted as tumor-present or tumor-absent.</p>
        <div class="hero-pill-row">
            <div class="hero-pill">⚡ Fast Inference</div>
            <div class="hero-pill">🛡 Deep Learning Powered</div>
            <div class="hero-pill">▥ VGG16 Transfer Learning</div>
            <div class="hero-pill">◈ Simple Web Interface</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    up_col, prev_col, result_col = st.columns([1.12, 1.0, 1.18], gap="medium")

    with up_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">1. Upload MRI Image</div><div class="section-sub">Supported formats: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload MRI",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
            st.markdown('<div class="small-note">Images are resized to 128 × 128 and converted to RGB before inference.</div>', unsafe_allow_html=True)

    with prev_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">2. Preview</div><div class="section-sub">Uploaded image</div>', unsafe_allow_html=True)
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, use_container_width=True)
            else:
                st.markdown(
                    '<div style="height:280px;border:1px dashed #c8d3e3;border-radius:14px;display:flex;align-items:center;justify-content:center;color:#7a8799;">MRI preview will appear here</div>',
                    unsafe_allow_html=True,
                )

    with result_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">3. Prediction Result</div><div class="section-sub">AI classification output</div>', unsafe_allow_html=True)

            if uploaded:
                if not os.path.exists(MODEL_PATH):
                    st.error(
                        "The trained model file is missing. Put `vgg16_model_final.keras` inside the `models/` folder."
                    )
                    st.markdown(
                        '<div class="info-box">The application structure is ready, but the trained `.keras` artifact is not part of the supplied GitHub repository. See <b>models/MODEL_SETUP.txt</b>.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    try:
                        model = load_brain_tumor_model()
                        with st.spinner("Analyzing MRI..."):
                            start = time.perf_counter()
                            pred = predict_image(model, image)
                            elapsed = time.perf_counter() - start
                        st.session_state["last_prediction"] = pred

                        is_tumor = pred["label"] == "Tumor Detected"
                        cls = "result-danger" if is_tumor else "result-success"
                        st.markdown(
                            f'<div class="result-card {" " if is_tumor else "result-ok"}>'
                            f'<div class="result-label">MODEL PREDICTION</div>'
                            f'<div class="result-main {cls}">{"Tumor Detected" if is_tumor else "No Tumor Detected"}</div>'
                            f'<div style="color:#667085;font-size:13px;line-height:1.5;">'
                            f'Classification result generated from the uploaded MRI image.</div><hr>'
                            f'<b>Tumor probability</b><div style="margin-top:6px;">{pred["tumor_probability"]:.1%}</div>'
                            f'<b style="display:block;margin-top:10px;">No-tumor probability</b>'
                            f'<div style="margin-top:6px;">{pred["no_tumor_probability"]:.1%}</div>'
                            f'<div class="small-note" style="margin-top:12px;">Inference time: {elapsed:.2f}s</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
            else:
                st.info("Upload an MRI image to run the model.")

    st.write("")
    render_metric_cards()

    st.write("")
    left, mid, right = st.columns([1.0, 1.2, 1.0], gap="medium")

    with left:
        st.markdown('<div class="card"><div class="section-title">Example Images</div><div class="section-sub">Samples preserved from the project notebook</div>', unsafe_allow_html=True)
        ex = Path("assets/example_samples.png")
        if ex.exists():
            st.image(str(ex), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with mid:
        st.markdown("""
        <div class="card">
        <div class="section-title">About This Project</div>
        <div class="section-sub">Deep learning image classification</div>
        <p style="color:#536276;line-height:1.65;font-size:13px;">
        BrainTumor AI is an educational/research application built around the Brain MRI classification experiment.
        The project compares a custom CNN with VGG16 transfer learning for binary classification.
        </p>
        <div class="info-box"><b>Important:</b> the model classifies images; it does not localize, segment, or subtype tumors.</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card">
        <div class="section-title">Quick Info</div>
        <div class="section-sub">Current recorded experiment</div>
        <table style="width:100%;font-size:12px;color:#536276;">
        <tr><td>Classes</td><td><b>Tumor / No Tumor</b></td></tr>
        <tr><td>Architecture</td><td><b>VGG16</b></td></tr>
        <tr><td>Framework</td><td><b>TensorFlow / Keras</b></td></tr>
        <tr><td>Input</td><td><b>128 × 128 × 3</b></td></tr>
        <tr><td>Test accuracy</td><td><b>91.56%</b></td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("last_prediction"):
        pred = st.session_state["last_prediction"]
        st.write("")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.download_button(
                "⬇ Download Result",
                data=download_report(pred),
                file_name="brain_tumor_ai_report.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with c2:
            if st.button("↻ Reload Detection", use_container_width=True):
                st.session_state.pop("last_prediction", None)
                st.rerun()

    st.write("")
    st.markdown("""
    <div class="disclaimer">
    <b>⚠ Important Disclaimer</b><br>
    BrainTumor AI provides an AI-based image-classification result and is not a medical diagnosis.
    Do not use the result alone to make medical decisions. Please consult a qualified healthcare professional
    for clinical interpretation and advice.
    </div>
    """, unsafe_allow_html=True)

# ---------- ABOUT ----------
elif page == "About":
    st.markdown('<div class="hero"><div class="eyebrow">Project Overview</div><h1>About <span>BrainTumor AI</span></h1><p>A polished interface around a deep-learning MRI classification experiment using TensorFlow/Keras.</p></div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.2, .8], gap="large")
    with a:
        st.markdown("""
        <div class="card">
        <div class="section-title">What does this application do?</div>
        <p style="color:#536276;line-height:1.75;">
        The application receives a brain MRI image, applies the same 128×128 RGB normalization used by the trained experiment,
        and passes the image to the selected trained model for binary classification.
        </p>
        <p style="color:#536276;line-height:1.75;">
        The original project compared a custom CNN against a frozen-backbone VGG16 transfer-learning model.
        The recorded test accuracy was 87.56% for the custom CNN and 91.56% for VGG16.
        </p>
        </div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="card">
        <div class="section-title">Project Scope</div>
        <ul style="color:#536276;line-height:1.9;">
        <li>Binary MRI image classification</li>
        <li>128 × 128 RGB input</li>
        <li>TensorFlow / Keras</li>
        <li>VGG16 transfer learning</li>
        <li>Streamlit interface</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------- MODEL INFO ----------
elif page == "Model Info":
    st.markdown('<div class="hero"><div class="eyebrow">Performance & Architecture</div><h1>Model <span>Info</span></h1><p>Compare the two architectures recorded in the original experiment and inspect the evaluation evidence.</p></div>', unsafe_allow_html=True)
    st.write("")

    df = pd.DataFrame({
        "Model": ["Custom CNN", "VGG16 Transfer Learning"],
        "Test Accuracy": [87.56, 91.56],
        "Trainable Strategy": ["All CNN layers", "Classifier head only"],
        "Input": ["128 × 128 × 3", "128 × 128 × 3"],
    })

    c1, c2 = st.columns([1.15, .85], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown('<div class="section-title">Recorded Test Accuracy</div><div class="section-sub">Single recorded test split from the notebook</div>', unsafe_allow_html=True)
            fig = px.bar(
                df, x="Model", y="Test Accuracy", text="Test Accuracy",
                range_y=[0, 100], labels={"Test Accuracy": "Accuracy (%)"},
                color="Model", color_discrete_sequence=["#637ff4", "#a06bf4"],
            )
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", textfont=dict(color="#9fb4ff"))
            fig.update_layout(height=330, showlegend=False)
            st.plotly_chart(apply_plotly_theme(fig), use_container_width=True, config={"displayModeBar": False})

    with c2:
        with st.container(border=True):
            st.markdown('<div class="section-title">Architecture Comparison</div><div class="section-sub">From the original training configuration</div>', unsafe_allow_html=True)
            st.markdown("""
            <table class="model-table">
                <thead><tr><th>Setting</th><th>Custom CNN</th><th>VGG16</th></tr></thead>
                <tbody>
                    <tr><td>Feature extractor</td><td>2 Conv blocks</td><td>Frozen ImageNet VGG16</td></tr>
                    <tr><td>Pooling</td><td>MaxPool</td><td>Global Avg Pool</td></tr>
                    <tr><td>Dense</td><td>128 ReLU</td><td>128 ReLU</td></tr>
                    <tr><td>Dropout</td><td>0.5</td><td>0.3</td></tr>
                    <tr><td>Output</td><td>2-class softmax</td><td>2-class softmax</td></tr>
                    <tr><td>Optimizer</td><td>Adam</td><td>Adam</td></tr>
                    <tr><td>Epochs</td><td>20</td><td>10</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

    st.write("")
    with st.container(border=True):
        st.markdown('<div class="section-title">CNN Confusion Matrix</div><div class="section-sub">Recorded output from the notebook — VGG16 confusion matrix was not generated in the original experiment.</div>', unsafe_allow_html=True)
        cm = np.array([[194, 31], [25, 200]])
        cm_df = pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
        fig2 = px.imshow(
            cm_df, text_auto=True, aspect="auto",
            color_continuous_scale=["#293b72", "#526fd0", "#a26cf4"],
        )
        fig2.update_traces(textfont=dict(color="#ffffff"))
        fig2.update_layout(height=330, coloraxis_colorbar=dict(title="Cases", tickfont=dict(color="#aebbe0")))
        st.plotly_chart(apply_plotly_theme(fig2), use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="info-box" style="margin-top:16px;">
    <b>Interpretation:</b> VGG16 has the higher recorded test accuracy in this experiment, but the result comes from a single image-level split.
    Patient-level separation, duplicate detection, external validation, and statistical uncertainty were not established by the original notebook.
    </div>
    """, unsafe_allow_html=True)

# ---------- HOW IT WORKS ----------
elif page == "How It Works":
    st.markdown('<div class="hero"><div class="eyebrow">Pipeline</div><h1>How It <span>Works</span></h1><p>From an uploaded MRI image to an AI classification result.</p></div>', unsafe_allow_html=True)
    st.write("")
    steps = [
        ("01", "Upload MRI", "The user uploads a JPG, JPEG, or PNG brain MRI image."),
        ("02", "Preprocess", "The image is converted to RGB, resized to 128×128, and scaled to [0, 1], matching the trained experiment."),
        ("03", "Inference", "The trained VGG16-based classifier receives the processed image."),
        ("04", "Prediction", "The model returns class probabilities for tumor-present and tumor-absent."),
        ("05", "Explain the Result", "The interface displays the predicted class, probabilities, model details, and limitations."),
    ]
    for n, title, body in steps:
        st.markdown(f"""
        <div class="card" style="margin-bottom:12px;display:flex;gap:18px;align-items:flex-start;">
            <div style="min-width:48px;height:48px;border-radius:14px;background:#eaf0ff;color:#2f6df6;display:flex;align-items:center;justify-content:center;font-weight:900;">{n}</div>
            <div><div class="section-title">{title}</div><div style="color:#627086;font-size:13px;line-height:1.6;">{body}</div></div>
        </div>
        """, unsafe_allow_html=True)

# ---------- LIMITATIONS ----------
elif page == "Limitations & Disclaimer":
    st.markdown('<div class="hero"><div class="eyebrow">Responsible AI</div><h1>Limitations <span>& Disclaimer</span></h1><p>Important context for interpreting the current project and its recorded results.</p></div>', unsafe_allow_html=True)
    st.write("")
    limitations = [
        ("Educational / research scope", "This application is a project demonstration and does not establish clinical diagnostic performance."),
        ("Image-level split", "The original experiment splits images rather than patient identifiers; patient-level leakage cannot be ruled out from the supplied workflow."),
        ("No external validation", "The recorded evaluation does not include a separate external dataset."),
        ("Single recorded run", "The reported 87.56% and 91.56% values are from the saved experiment output and are not a confidence interval or clinical benchmark."),
        ("No tumor localization", "The classifier predicts a class; it does not provide segmentation or a medically validated tumor boundary."),
        ("Preprocessing dependency", "Inference must preserve the training preprocessing: RGB conversion, 128×128 resize, and division by 255."),
    ]
    for title, body in limitations:
        st.markdown(f"""
        <div class="card" style="margin-bottom:12px;">
            <div class="section-title">⚠ {title}</div>
            <div style="color:#627086;font-size:13px;line-height:1.7;margin-top:5px;">{body}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
    <b>Medical disclaimer</b><br>
    This tool is not a medical device and must not be used as a substitute for professional medical advice,
    diagnosis, or treatment. Any clinical decision must be made by a qualified healthcare professional using
    appropriate medical evidence and patient context.
    </div>
    """, unsafe_allow_html=True)

# ---------- CONTACT ----------
elif page == "Contact":
    st.markdown('<div class="hero"><div class="eyebrow">Connect</div><h1>Contact <span>Abdelrahman</span></h1><p>For project feedback, collaboration, or technical discussion.</p></div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="card">
        <div class="section-title">GitHub</div>
        <div class="section-sub">Source code and project files</div>
        <a class="social-icon" href="https://github.com/abdelrahmanmohamedhafez7-droid/project_deep_learninig_Brain_Tumor_MRI" target="_blank"
           title="Open GitHub repository" aria-label="Open GitHub repository">
           <svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.7 7.7 0 0 1 8 4.8c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>
        </a>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="card">
        <div class="section-title">LinkedIn</div>
        <div class="section-sub">Professional profile</div>
        <a class="social-icon" href="https://eg.linkedin.com/in/abdelrahman-mohamed-hafez-41bbb7397" target="_blank"
           title="Open LinkedIn profile" aria-label="Open LinkedIn profile">
           <svg viewBox="0 0 16 16"><path d="M0 1.15C0 .51.53 0 1.19 0h13.62C15.47 0 16 .51 16 1.15v13.7c0 .64-.53 1.15-1.19 1.15H1.19C.53 16 0 15.49 0 14.85V1.15Zm4.94 12.54V6.17H2.44v7.52h2.5ZM3.69 5.14c.87 0 1.57-.7 1.57-1.57A1.57 1.57 0 0 0 3.69 2a1.57 1.57 0 0 0 0 3.14Zm10.12 8.55V9.57c0-2.21-.47-3.91-3.05-3.91-1.24 0-2.07.68-2.41 1.32h-.04V6.17H5.92v7.52h2.49V9.97c0-.98.19-1.93 1.4-1.93 1.2 0 1.22 1.12 1.22 1.99v3.66h2.78Z"/></svg>
        </a>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="footer">BrainTumor AI · Educational / Research Project · Built with TensorFlow, Keras & Streamlit</div>', unsafe_allow_html=True)
