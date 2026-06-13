"""
pages/dashboard.py
──────────────────
Analytics Dashboard — Smart Waste Management System
Government-grade operational intelligence dashboard.
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
    page_title="SWMS Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #070D1A !important;
    color: #D0D8E8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* ── Government banner ── */
.gov-banner {
    background: linear-gradient(90deg, #040912 0%, #0A1628 40%, #0D2040 70%, #040912 100%);
    border-bottom: 2px solid #00B4A6;
    padding: 0.5rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    font-family: 'DM Sans', sans-serif; font-size: 0.72rem;
    color: #6E7E9A; letter-spacing: 0.06em; text-transform: uppercase;
}
.gov-banner span { color: #00B4A6; font-weight: 600; }

/* ── Page header ── */
.dash-hero {
    background: linear-gradient(135deg, #080F1E 0%, #0A1C30 50%, #060E1C 100%);
    border: 1px solid #0F2545;
    border-radius: 16px;
    padding: 1.8rem 2.4rem;
    margin-bottom: 1.4rem;
    position: relative; overflow: hidden;
}
.dash-hero::before {
    content: '';
    position: absolute; top:-60px; right:-60px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(0,180,166,0.1) 0%, transparent 70%);
    pointer-events: none;
}
.dash-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: #00B4A6;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.dash-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #EDF2FA;
    margin-bottom: 0.3rem; line-height:1.2;
}
.dash-title span { color: #00B4A6; }
.dash-sub { font-size: 0.88rem; color: #4A6080; }

/* ── KPI cards ── */
.kpi-card {
    background: #080F1E;
    border: 1px solid #0F2545;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #1A3C5E; }
.kpi-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:3px;
    border-radius: 12px 12px 0 0;
}
.kpi-teal::before  { background: linear-gradient(90deg, #00B4A6, #007A72); }
.kpi-amber::before { background: linear-gradient(90deg, #F5A623, #C07A10); }
.kpi-green::before { background: linear-gradient(90deg, #50C878, #308050); }
.kpi-red::before   { background: linear-gradient(90deg, #DC3C50, #A02030); }
.kpi-blue::before  { background: linear-gradient(90deg, #50A0DC, #3070A8); }

.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; color: #3A5070;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700; color: #EDF2FA;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem; color: #2A4060;
    font-family: 'DM Mono', monospace;
    margin-top: 0.3rem; letter-spacing: 0.05em;
}
.kpi-icon {
    position: absolute; right:1.2rem; top:50%;
    transform:translateY(-50%);
    font-size: 1.8rem; opacity:0.12;
}

/* ── Section header ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; color: #00B4A6;
    letter-spacing: 0.16em; text-transform: uppercase;
    margin: 1.4rem 0 0.8rem; padding-left: 2px;
}

/* ── Chart wrapper ── */
.chart-card {
    background: #080F1E;
    border: 1px solid #0F2545;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem; font-weight: 600;
    color: #8090A8; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 0.8rem;
    border-bottom: 1px solid #0A1628; padding-bottom: 0.5rem;
}

/* ── Plotly chart bg ── */
.stPlotlyChart { border-radius: 8px; overflow: hidden; }

/* ── Table ── */
.stDataFrame { border-radius: 8px !important; }
.stDataFrame th {
    background: #040912 !important;
    color: #00B4A6 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stDataFrame td {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #8090A8 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #040912 !important;
    border-radius: 8px !important;
    border: 1px solid #0F2545 !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    color: #4A6080 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #0A1C30 !important;
    color: #00B4A6 !important;
    border: 1px solid #00B4A6 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 0.82rem !important;
    border-radius: 8px !important; letter-spacing: 0.04em !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00B4A6, #008A80) !important;
    color: #FFF !important; border: none !important;
    box-shadow: 0 4px 16px rgba(0,180,166,0.3) !important;
}
.stButton > button:not([kind="primary"]) {
    background: #080F1E !important; color: #80A0C0 !important;
    border: 1px solid #1A3050 !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #00B4A6 !important; color: #00B4A6 !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #040912 !important;
    border-right: 1px solid #0A1C2E !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #6E7E9A !important; font-size: 0.72rem !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    font-weight: 600 !important;
}

/* ── Nature pill ── */
.nature-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; letter-spacing: 0.08em;
    padding: 2px 8px; border-radius: 4px; border: 1px solid;
    text-transform: uppercase;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #080F1E !important;
    border: 1px solid #00B4A6 !important;
    color: #00B4A6 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; border-radius: 8px !important;
}

/* ── Footer ── */
.gov-footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem; color: #1E2E3E;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #0A1628;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="dash-hero">
    <div class="dash-eyebrow">⬡ SWMS / Analytics & Reporting Module</div>
    <div class="dash-title">Waste Stream <span>Intelligence</span></div>
    <div class="dash-sub">Longitudinal detection analytics · Source-wise distribution · Biodegradability classification</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔗 Navigation")
    st.page_link("app.py", label="🏠 Back to Detection App", icon="🏠")
    st.markdown("---")
    st.markdown("### 🗄️ Data Management")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    st.markdown("")
    if st.button("🗑️ Clear All History", use_container_width=True):
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
# Plotly theme
# ─────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#6E7E9A"),
    margin=dict(t=20, b=20, l=10, r=10),
)

# ─────────────────────────────────────────────
# KPI Cards
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Operational Overview</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="section-label">▸ Biodegradability Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Waste Nature Distribution</div>', unsafe_allow_html=True)
        nature_labels = list(nature.keys())
        nature_values = list(nature.values())
        color_map_donut = {
            "Biodegradable":     "#50C878",
            "Non-Biodegradable": "#F5A623",
            "Hazardous":         "#DC3C50",
            "Unknown":           "#3A5070",
        }
        donut_colors = [color_map_donut.get(l, "#3A5070") for l in nature_labels]

        fig_donut = go.Figure(data=[go.Pie(
            labels=nature_labels,
            values=nature_values,
            hole=0.62,
            marker=dict(colors=donut_colors, line=dict(color="#070D1A", width=3)),
            textfont=dict(size=11, color="#D0D8E8"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        )])
        fig_donut.add_annotation(
            text=f"<b>{sum(nature_values)}</b><br><span style='font-size:10px;'>TOTAL</span>",
            x=0.5, y=0.5, font=dict(size=16, color="#EDF2FA"),
            showarrow=False,
        )
        fig_donut.update_layout(
            **PLOTLY_BASE,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        font=dict(color="#6E7E9A", size=11)),
            height=280,
        )
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Bar — class distribution ────────────────────────────────────
    with chart_col2:
        st.markdown('<div class="section-label">▸ Detection Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Count by Waste Class</div>', unsafe_allow_html=True)
        color_map_bar = {
            "Biodegradable":     "#50C878",
            "Non-Biodegradable": "#F5A623",
            "Hazardous":         "#DC3C50",
            "Unknown":           "#3A5070",
        }
        fig_bar = px.bar(
            df_dist.sort_values("count", ascending=False),
            x="class_name", y="count",
            color="waste_nature",
            color_discrete_map=color_map_bar,
            labels={"class_name": "", "count": "Count", "waste_nature": "Nature"},
            text="count",
        )
        fig_bar.update_traces(textposition="outside", textfont=dict(size=10, color="#6E7E9A"),
                              marker_line_width=0)
        fig_bar.update_layout(
            **PLOTLY_BASE,
            xaxis=dict(tickangle=-35, gridcolor="#0A1628", tickfont=dict(size=10), linecolor="#0F2545"),
            yaxis=dict(gridcolor="#0A1628", linecolor="#0F2545"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, font=dict(color="#6E7E9A", size=11)),
            legend_title_text="",
            height=280,
            bargap=0.25,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Confidence scatter ──────────────────────────────────────────
    if not df_all.empty and "timestamp" in df_all.columns:
        st.markdown('<div class="section-label">▸ Confidence Timeline</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card"><div class="chart-title">Model Confidence over Time</div>', unsafe_allow_html=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        scatter_color_map = {
            "Biodegradable":     "#50C878",
            "Non-Biodegradable": "#F5A623",
            "Hazardous":         "#DC3C50",
            "Unknown":           "#3A5070",
        }
        fig_line = px.scatter(
            df_all.sort_values("timestamp"),
            x="timestamp", y="confidence",
            color="waste_nature",
            color_discrete_map=scatter_color_map,
            labels={"timestamp": "", "confidence": "Confidence", "waste_nature": "Nature"},
            opacity=0.7,
            hover_data=["class_name"],
        )
        # Add trend line
        fig_line.update_traces(marker=dict(size=6, line=dict(width=0)))
        fig_line.update_layout(
            **PLOTLY_BASE,
            xaxis=dict(gridcolor="#0A1628", linecolor="#0F2545", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#0A1628", linecolor="#0F2545", range=[0, 1],
                       tickformat=".0%", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color="#6E7E9A", size=11)),
            legend_title_text="",
            height=240,
        )
        st.plotly_chart(fig_line, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Source breakdown ────────────────────────────────────────────
    if not df_all.empty and "source" in df_all.columns:
        st.markdown('<div class="section-label">▸ Detection Source Analysis</div>', unsafe_allow_html=True)
        src_col1, src_col2 = st.columns([1, 2], gap="medium")

        with src_col1:
            st.markdown('<div class="chart-card"><div class="chart-title">Source Split</div>', unsafe_allow_html=True)
            src_counts = df_all["source"].value_counts().reset_index()
            src_counts.columns = ["Source", "Count"]
            fig_src = px.pie(
                src_counts, names="Source", values="Count", hole=0.5,
                color_discrete_sequence=["#00B4A6", "#F5A623", "#50A0DC"],
            )
            fig_src.update_layout(**PLOTLY_BASE, height=220,
                                  legend=dict(orientation="h", y=-0.2, font=dict(size=11, color="#6E7E9A")))
            st.plotly_chart(fig_src, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with src_col2:
            st.markdown('<div class="chart-card"><div class="chart-title">Class × Source Heatmap</div>', unsafe_allow_html=True)
            if "class_name" in df_all.columns:
                pivot = df_all.pivot_table(index="class_name", columns="source",
                                           values="confidence", aggfunc="count", fill_value=0)
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale=[[0, "#040912"], [0.5, "#004A44"], [1, "#00B4A6"]],
                    aspect="auto",
                    text_auto=True,
                )
                fig_heat.update_layout(
                    **PLOTLY_BASE, height=220,
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(size=10)), yaxis=dict(tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#040912; border:2px dashed #0F2545; border-radius:12px;
                padding:4rem 2rem; text-align:center; color:#2A3A4A; margin:1rem 0;">
        <div style="font-size:2.5rem; margin-bottom:0.8rem;">📭</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; color:#3A5070;">
            No detection data available
        </div>
        <div style="font-size:0.75rem; font-family:'DM Mono',monospace; margin-top:0.4rem;
                    letter-spacing:0.1em; color:#1A2A3A;">
            RUN THE CLASSIFIER ON IMAGES OR WEBCAM FEED TO POPULATE THIS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Activity log
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">▸ Detection Activity Log</div>', unsafe_allow_html=True)

if not df_recent.empty:
    NATURE_COLORS = {
        'Biodegradable':     {'bg': '#0D3320', 'text': '#50C878'},
        'Non-Biodegradable': {'bg': '#2A1A08', 'text': '#F5A623'},
        'Hazardous':         {'bg': '#2A0A0A', 'text': '#DC3C50'},
        'Unknown':           {'bg': '#1A1E24', 'text': '#78828C'},
    }

    def style_nature(val):
        nc = NATURE_COLORS.get(val, NATURE_COLORS['Unknown'])
        return f"background-color:{nc['bg']}; color:{nc['text']}; font-weight:600"

    display_cols = ["id", "timestamp", "source", "class_name", "confidence", "bin_label", "waste_nature"]
    df_show = df_recent[display_cols].copy()
    df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x*100:.1f}%")
    df_show.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]

    styled = df_show.style.applymap(style_nature, subset=["Nature"])
    st.dataframe(styled, use_container_width=True, height=340)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_bio, tab_nonbio, tab_haz = st.tabs([
        "🌿  Biodegradable", "⚠️  Non-Biodegradable", "☣️  Hazardous"
    ])

    for tab, nature_val, label_color, accent in [
        (tab_bio,    "Biodegradable",     "#50C878", "success"),
        (tab_nonbio, "Non-Biodegradable", "#F5A623", "warning"),
        (tab_haz,    "Hazardous",         "#DC3C50", "error"),
    ]:
        with tab:
            df_sub = df_recent[df_recent["waste_nature"] == nature_val].copy()
            if not df_sub.empty:
                df_sub = df_sub[display_cols].copy()
                df_sub["confidence"] = df_sub["confidence"].apply(lambda x: f"{x*100:.1f}%")
                df_sub.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]

                st.markdown(f"""
                <div style="display:flex; gap:1rem; margin-bottom:0.8rem; align-items:center;">
                    <span style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem;
                                 font-weight:700; color:{label_color};">{len(df_sub)}</span>
                    <span style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#3A5070;
                                 letter-spacing:0.1em; text-transform:uppercase;">
                        {nature_val.upper()} ITEMS DETECTED
                    </span>
                </div>""", unsafe_allow_html=True)

                st.dataframe(df_sub, use_container_width=True, hide_index=True)

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
                    xaxis=dict(tickfont=dict(size=10), gridcolor="#0A1628"),
                    yaxis=dict(gridcolor="#0A1628"),
                    showlegend=False,
                )
                st.plotly_chart(fig_mini, use_container_width=True)
            else:
                st.markdown(f"""
                <div style="padding:2rem; text-align:center; color:#1A2A3A;
                            font-family:'DM Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;">
                    NO {nature_val.upper()} ITEMS IN RECENT HISTORY
                </div>""", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="padding:2rem; text-align:center; color:#1A2A3A;
                font-family:'DM Mono',monospace; font-size:0.75rem; letter-spacing:0.1em;">
        ACTIVITY LOG EMPTY — NO DETECTIONS RECORDED YET
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────
if not df_all.empty:
    st.markdown('<div class="section-label">▸ Data Export</div>', unsafe_allow_html=True)
    exp_col1, exp_col2 = st.columns([2, 3])
    with exp_col1:
        csv = df_all.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥  Download Full Detection History (CSV)",
            data=csv,
            file_name="swms_waste_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="gov-footer">
    Smart Waste Management System &nbsp;·&nbsp; Analytics Module &nbsp;·&nbsp;
    Ministry of Environment, Forest &amp; Climate Change &nbsp;·&nbsp; Government of India
</div>
""", unsafe_allow_html=True)
