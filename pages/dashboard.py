"""
pages/dashboard.py
──────────────────
Analytics Dashboard — Smart Waste Management System
Government-grade operational intelligence dashboard.

Restyled to match the light "Government of India" portal theme used in
app.py — Ashoka Blue / Saffron / India Green on a paper-white background.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import (
    fetch_summary,
    fetch_distribution,
    fetch_nature_split,
    fetch_recent,
    fetch_all_detections,
    clear_all_detections,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SWMS Analytics Dashboard | Government of India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS — matches app.py's official portal theme
#   Ink:        #14213D   (headings / primary text)
#   Ashoka Blue #0F4C81   (primary / links / header)
#   Saffron     #E17A1D   (accent / CTAs)
#   India Green #15803D   (biodegradable / success)
#   Paper       #F5F6F8   (page background)
#   Card        #FFFFFF
#   Border      #DCE1E8
#   Muted text  #5B6B7C
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Source+Sans+3:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #F5F6F8 !important;
    color: #263241;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1280px; }

/* ── Tricolour strip ── */
.tricolour-strip { display: flex; width: 100%; height: 5px; margin-bottom: 0; }
.tricolour-strip div { flex: 1; }
.strip-saffron { background: #FF9933; }
.strip-white   { background: #FFFFFF; border-top: 1px solid #E5E7EB; border-bottom: 1px solid #E5E7EB; }
.strip-green   { background: #138808; }

/* ── Official header bar ── */
.gov-header {
    background: linear-gradient(180deg, #123A66 0%, #0F4C81 100%);
    padding: 0.85rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.gov-header-left { display: flex; align-items: center; gap: 0.9rem; }
.emblem-badge {
    width: 46px; height: 46px;
    border-radius: 50%;
    background: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    border: 2px solid #E17A1D;
    flex-shrink: 0;
}
.gov-header-title { color: #FFFFFF; }
.gov-header-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #B9D3EC;
    margin-bottom: 2px;
}
.gov-header-name {
    font-family: 'Fraunces', serif;
    font-size: 1.18rem;
    font-weight: 600;
    line-height: 1.25;
}
.gov-header-right { display: flex; align-items: center; gap: 0.6rem; }
.gov-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    color: #DCE9F7;
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 3px;
    padding: 4px 10px;
    text-transform: uppercase;
}

/* ── Breadcrumb ── */
.breadcrumb {
    background: #FFFFFF;
    border-bottom: 1px solid #DCE1E8;
    padding: 0.55rem 2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #5B6B7C;
    letter-spacing: 0.02em;
}
.breadcrumb span.current { color: #0F4C81; font-weight: 600; }

/* ── Hero / circular notice ── */
.hero-block {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-top: 4px solid #E17A1D;
    border-radius: 4px;
    padding: 1.7rem 2.4rem;
    margin: 1.6rem 0 1.4rem;
}
.hero-ref {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #E17A1D;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.hero-ref::before { content: '§'; font-size: 0.9rem; }
.hero-title {
    font-family: 'Fraunces', serif;
    font-size: 1.85rem;
    font-weight: 600;
    color: #14213D;
    line-height: 1.2;
    margin-bottom: 0.4rem;
}
.hero-title span { color: #0F4C81; }
.hero-sub {
    font-size: 0.92rem;
    color: #5B6B7C;
    max-width: 680px;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #0F4C81;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.7rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #DCE1E8;
}

/* ── KPI cards ── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #0F4C81; }
.kpi-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:4px;
}
.kpi-teal::before  { background: #0F4C81; }
.kpi-amber::before { background: #E17A1D; }
.kpi-green::before { background: #15803D; }
.kpi-red::before   { background: #B91C1C; }
.kpi-blue::before  { background: #0E7490; }

.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: #8593A3;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem; font-weight: 600; color: #14213D;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.7rem; color: #A0ACB9;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.35rem; letter-spacing: 0.05em;
}
.kpi-icon {
    position: absolute; right:1.1rem; top:50%;
    transform:translateY(-50%);
    font-size: 1.7rem; opacity:0.14;
}

/* ── Chart wrapper ── */
.chart-card {
    background: #FFFFFF;
    border: 1px solid #DCE1E8;
    border-radius: 4px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Fraunces', serif;
    font-size: 0.92rem; font-weight: 600;
    color: #14213D;
    letter-spacing: 0.01em; margin-bottom: 0.8rem;
    border-bottom: 1px solid #DCE1E8; padding-bottom: 0.5rem;
}

/* ── Plotly chart bg ── */
.stPlotlyChart { border-radius: 4px; overflow: hidden; }

/* ── Table ── */
.stDataFrame { border-radius: 4px !important; border: 1px solid #DCE1E8 !important; }
.stDataFrame th {
    background: #F5F6F8 !important;
    color: #0F4C81 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stDataFrame td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #445468 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF !important;
    border-radius: 4px !important;
    border: 1px solid #DCE1E8 !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 3px !important;
    color: #5B6B7C !important;
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #EAF2FA !important;
    color: #0F4C81 !important;
    border: 1px solid #0F4C81 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    border-radius: 4px !important; letter-spacing: 0.01em !important;
}
.stButton > button[kind="primary"] {
    background: #E17A1D !important;
    color: #FFFFFF !important; border: none !important;
    box-shadow: 0 2px 8px rgba(225,122,29,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #C2670F !important;
    box-shadow: 0 3px 10px rgba(225,122,29,0.4) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important; color: #0F4C81 !important;
    border: 1px solid #C6CEDA !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #0F4C81 !important; background: #EAF2FA !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DCE1E8 !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Fraunces', serif !important;
    color: #14213D !important; font-size: 0.85rem !important;
    letter-spacing: 0.02em !important; text-transform: none !important;
    font-weight: 600 !important;
}

/* ── Nature pill ── */
.nature-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem; letter-spacing: 0.07em;
    padding: 2px 9px; border-radius: 3px; border: 1px solid;
    text-transform: uppercase;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #FFFFFF !important;
    border: 1px solid #0F4C81 !important;
    color: #0F4C81 !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important; border-radius: 4px !important;
}
.stDownloadButton > button:hover {
    background: #EAF2FA !important;
}

.stAlert { border-radius: 4px !important; }

/* ── Footer ── */
.gov-footer-wrap { background: #14213D; margin-top: 2.4rem; }
.gov-footer {
    max-width: 1280px; margin: 0 auto; padding: 1rem 2rem;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.66rem; color: #6C7D93;
    letter-spacing: 0.06em; text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Tricolour strip + official header
# ─────────────────────────────────────────────
st.markdown("""
<div class="tricolour-strip"><div class="strip-saffron"></div><div class="strip-white"></div><div class="strip-green"></div></div>
<div class="gov-header">
    <div class="gov-header-left">
        <div class="emblem-badge">🇮🇳</div>
        <div class="gov-header-title">
            <div class="gov-header-eyebrow">Government of India · Ministry of Environment, Forest &amp; Climate Change</div>
            <div class="gov-header-name">Smart Waste Management System</div>
        </div>
    </div>
    <div class="gov-header-right">
        <span class="gov-pill">Digital India</span>
        <span class="gov-pill">EN / हिं</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
    <div class="hero-ref">Circular No. SWMS/2026/AI-RPT &nbsp;·&nbsp; Analytics &amp; Reporting Module</div>
    <div class="hero-title">Waste Stream <span>Intelligence</span></div>
    <div class="hero-sub">
        Longitudinal detection analytics, source-wise distribution, and biodegradability classification
        drawn from the AI Detection Portal's recorded observations.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    st.page_link("app.py", label="🏠 Back to Detection Console", icon="🏠")
    st.markdown("---")
    st.markdown("### Data Management")
    if st.button("🔄 Refresh Data", width='stretch'):
        st.rerun()
    st.markdown("")
    if st.button("🗑️ Clear All History", width='stretch'):
        clear_all_detections()
        st.success("History cleared!")
        st.rerun()

# ─────────────────────────────────────────────
# Fetch data
# ─────────────────────────────────────────────
summary = fetch_summary()
dist    = fetch_distribution()
nature  = fetch_nature_split()
recent  = fetch_recent(limit=50)
all_det = fetch_all_detections()

df_dist   = pd.DataFrame(dist)   if dist   else pd.DataFrame(columns=["class_name","waste_nature","count"])
df_recent = pd.DataFrame(recent) if recent else pd.DataFrame()
df_all    = pd.DataFrame(all_det) if all_det else pd.DataFrame()

# ─────────────────────────────────────────────
# Plotly theme — light, matches paper-white portal
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Source Sans 3, sans-serif", color="#5B6B7C"),
    margin=dict(t=20, b=20, l=10, r=10),
)

GRID_COLOR   = "#EDEFF3"
LINE_COLOR   = "#DCE1E8"

# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Operational Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi_data = [
    (c1, summary["total"],                    "Total Detections",      "kpi-teal",  "🔍", "CUMULATIVE SCANS"),
    (c2, summary["most_common"],              "Most Common Class",     "kpi-amber", "📦", "HIGHEST FREQUENCY"),
    (c3, f"{summary['avg_confidence']:.2f}",  "Avg. Confidence",       "kpi-blue",  "🎯", "MODEL ACCURACY"),
    (c4, summary["biodegradable"],            "Biodegradable Items",   "kpi-green", "🌿", "ORGANIC / PAPER"),
    (c5, summary["non_biodegradable"],        "Non-Biodegradable",     "kpi-red",   "⚠️",  "PLASTIC / METAL / E-WASTE"),
]
for col, val, label, cls, icon, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi-card {cls}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
            <div class="kpi-icon">{icon}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────
if not df_dist.empty:

    chart_col1, chart_col2 = st.columns([1, 1], gap="medium")

    # ── Donut — Bio vs Non-Bio ──────────────────────────────────────
    with chart_col1:
        st.markdown('<div class="section-label">Biodegradability Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Waste Nature Distribution</div>', unsafe_allow_html=True)
        nature_labels = list(nature.keys())
        nature_values = list(nature.values())
        color_map_donut = {
            "Biodegradable":     "#15803D",
            "Non-Biodegradable": "#E17A1D",
            "Hazardous":         "#B91C1C",
            "Unknown":           "#8593A3",
        }
        donut_colors = [color_map_donut.get(l, "#8593A3") for l in nature_labels]

        fig_donut = go.Figure(data=[go.Pie(
            labels=nature_labels,
            values=nature_values,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#FFFFFF", width=3)),
            textfont=dict(size=11, color="#263241"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig_donut.add_annotation(
            text=f"<b>{sum(nature_values)}</b><br><span style='font-size:10px;'>TOTAL</span>",
            x=0.5, y=0.5, font=dict(size=16, color="#14213D"),
            showarrow=False,
        )
        fig_donut.update_layout(
            **PLOTLY_BASE,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        font=dict(color="#5B6B7C", size=11)),
            height=280,
        )
        st.plotly_chart(fig_donut, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Bar — class distribution ────────────────────────────────────
    with chart_col2:
        st.markdown('<div class="section-label">Detection Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Count by Waste Class</div>', unsafe_allow_html=True)
        color_map_bar = {
            "Biodegradable":     "#15803D",
            "Non-Biodegradable": "#E17A1D",
            "Hazardous":         "#B91C1C",
            "Unknown":           "#8593A3",
        }
        fig_bar = px.bar(
            df_dist.sort_values("count", ascending=False),
            x="class_name", y="count",
            color="waste_nature",
            color_discrete_map=color_map_bar,
            labels={"class_name": "", "count": "Count", "waste_nature": "Nature"},
            text="count",
        )
        fig_bar.update_traces(textposition="outside", textfont=dict(size=10, color="#5B6B7C"),
                              marker_line_width=0)
        fig_bar.update_layout(
            **PLOTLY_BASE,
            xaxis=dict(tickangle=-35, gridcolor=GRID_COLOR, tickfont=dict(size=10), linecolor=LINE_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, font=dict(color="#5B6B7C", size=11)),
            legend_title_text="",
            height=280,
            bargap=0.25,
        )
        st.plotly_chart(fig_bar, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Confidence scatter ──────────────────────────────────────────
    if not df_all.empty and "timestamp" in df_all.columns:
        st.markdown('<div class="section-label">Confidence Timeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Model Confidence over Time</div>', unsafe_allow_html=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        scatter_color_map = {
            "Biodegradable":     "#15803D",
            "Non-Biodegradable": "#E17A1D",
            "Hazardous":         "#B91C1C",
            "Unknown":           "#8593A3",
        }
        fig_line = px.scatter(
            df_all.sort_values("timestamp"),
            x="timestamp", y="confidence",
            color="waste_nature",
            color_discrete_map=scatter_color_map,
            labels={"timestamp": "", "confidence": "Confidence", "waste_nature": "Nature"},
            opacity=0.75,
            hover_data=["class_name"],
        )
        fig_line.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig_line.update_layout(
            **PLOTLY_BASE,
            xaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, tickfont=dict(size=10)),
            yaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, range=[0, 1],
                       tickformat=".0%", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color="#5B6B7C", size=11)),
            legend_title_text="",
            height=240,
        )
        st.plotly_chart(fig_line, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Source breakdown ────────────────────────────────────────────
    if not df_all.empty and "source" in df_all.columns:
        st.markdown('<div class="section-label">Detection Source Analysis</div>', unsafe_allow_html=True)
        src_col1, src_col2 = st.columns([1, 2], gap="medium")

        with src_col1:
            st.markdown('<div class="chart-card"><div class="chart-title">Source Split</div>', unsafe_allow_html=True)
            src_counts = df_all["source"].value_counts().reset_index()
            src_counts.columns = ["Source", "Count"]
            fig_src = px.pie(
                src_counts, names="Source", values="Count", hole=0.5,
                color_discrete_sequence=["#0F4C81", "#E17A1D", "#0E7490"],
            )
            fig_src.update_layout(**PLOTLY_BASE, height=220,
                                  legend=dict(orientation="h", y=-0.2, font=dict(size=11, color="#5B6B7C")))
            st.plotly_chart(fig_src, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with src_col2:
            st.markdown('<div class="chart-card"><div class="chart-title">Class × Source Heatmap</div>', unsafe_allow_html=True)
            if "class_name" in df_all.columns:
                pivot = df_all.pivot_table(index="class_name", columns="source",
                                           values="confidence", aggfunc="count", fill_value=0)
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale=[[0, "#F5F6F8"], [0.5, "#7BA7CC"], [1, "#0F4C81"]],
                    aspect="auto",
                    text_auto=True,
                )
                fig_heat.update_layout(
                    **PLOTLY_BASE, height=220,
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#FFFFFF; border:2px dashed #C6CEDA; border-radius:6px;
                padding:4rem 2rem; text-align:center; color:#8593A3; margin:1rem 0;">
        <div style="font-size:2.4rem; margin-bottom:1rem;">📭</div>
        <div style="font-family:'Fraunces',serif; font-size:1.05rem; color:#14213D; font-weight:600;">
            No detection data available
        </div>
        <div style="font-size:0.78rem; font-family:'IBM Plex Mono',monospace; margin-top:0.4rem;
                    letter-spacing:0.06em; color:#A0ACB9;">
            RUN THE CLASSIFIER ON IMAGES OR WEBCAM FEED TO POPULATE THIS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Activity log
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Detection Activity Log</div>', unsafe_allow_html=True)

if not df_recent.empty:
    NATURE_COLORS = {
        'Biodegradable':     {'bg': '#ECFDF3', 'text': '#15803D'},
        'Non-Biodegradable': {'bg': '#FFF7ED', 'text': '#C2410C'},
        'Hazardous':         {'bg': '#FEF2F2', 'text': '#B91C1C'},
        'Unknown':           {'bg': '#F3F4F6', 'text': '#4B5563'},
    }

    def style_nature(val):
        nc = NATURE_COLORS.get(val, NATURE_COLORS['Unknown'])
        return f"background-color:{nc['bg']}; color:{nc['text']}; font-weight:600"

    display_cols = ["id", "timestamp", "source", "class_name", "confidence", "bin_label", "waste_nature"]
    df_show = df_recent[display_cols].copy()
    df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x*100:.1f}%")
    df_show.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]

    styled = df_show.style.map(style_nature, subset=["Nature"])
    st.dataframe(styled, width='stretch', height=340)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_bio, tab_nonbio, tab_haz = st.tabs([
        "🌿  Biodegradable", "⚠️  Non-Biodegradable", "☣️  Hazardous"
    ])

    for tab, nature_val, label_color, accent in [
        (tab_bio,    "Biodegradable",     "#15803D", "success"),
        (tab_nonbio, "Non-Biodegradable", "#E17A1D", "warning"),
        (tab_haz,    "Hazardous",         "#B91C1C", "error"),
    ]:
        with tab:
            df_sub = df_recent[df_recent["waste_nature"] == nature_val].copy()
            if not df_sub.empty:
                df_sub = df_sub[display_cols].copy()
                df_sub["confidence"] = df_sub["confidence"].apply(lambda x: f"{x*100:.1f}%")
                df_sub.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]

                st.markdown(f"""
                <div style="display:flex; gap:1rem; margin-bottom:0.8rem; align-items:center;">
                    <span style="font-family:'Fraunces',serif; font-size:1.4rem;
                                 font-weight:700; color:{label_color};">{len(df_sub)}</span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#5B6B7C;
                                 letter-spacing:0.1em; text-transform:uppercase;">
                        {nature_val.upper()} ITEMS DETECTED
                    </span>
                </div>""", unsafe_allow_html=True)

                st.dataframe(df_sub, width='stretch', hide_index=True)

                st.markdown("**Frequency by waste class**")
                summary_df = df_sub.groupby("Waste Type").size().reset_index(name="Count").sort_values("Count", ascending=False)
                fig_mini = px.bar(
                    summary_df, x="Waste Type", y="Count",
                    color_discrete_sequence=[label_color],
                    text="Count",
                )
                fig_mini.update_traces(textposition="outside", marker_line_width=0)
                fig_mini.update_layout(
                    **PLOTLY_BASE, height=180,
                    xaxis=dict(tickfont=dict(size=10), gridcolor=GRID_COLOR),
                    yaxis=dict(gridcolor=GRID_COLOR),
                    showlegend=False,
                )
                st.plotly_chart(fig_mini, width='stretch')
            else:
                st.markdown(f"""
                <div style="padding:2rem; text-align:center; color:#A0ACB9;
                            font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;">
                    NO {nature_val.upper()} ITEMS IN RECENT HISTORY
                </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="padding:2rem; text-align:center; color:#A0ACB9;
                font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;">
        ACTIVITY LOG EMPTY — NO DETECTIONS RECORDED YET
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
if not df_all.empty:
    st.markdown('<div class="section-label">Data Export</div>', unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([2, 3])
    with exp_col1:
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥  Download Full Detection History (CSV)",
            data=csv,
            file_name="swms_waste_detections.csv",
            mime="text/csv",
            width='stretch',
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="gov-footer-wrap">
    <div class="gov-footer">
        Smart Waste Management System &nbsp;·&nbsp; Analytics Module &nbsp;·&nbsp;
        Ministry of Environment, Forest &amp; Climate Change &nbsp;·&nbsp; Government of India
    </div>
</div>
""", unsafe_allow_html=True)
