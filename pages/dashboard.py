"""
pages/dashboard.py
──────────────────
Analytics Dashboard for Waste Classification System.
Place this file inside a `pages/` folder next to app.py.
Streamlit will auto-discover it as a multi-page app.
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
    page_title="Waste Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #2E7D32, #66BB6A);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.dash-sub { color:#888; font-size:0.95rem; margin-bottom:1.6rem; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1b2a1b, #243324);
    border: 1px solid #2e5c2e;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.metric-card .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #66BB6A;
}
.metric-card .metric-label {
    font-size: 0.8rem; color: #aaa; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.08em;
}

/* Section headers */
.section-head {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem; font-weight: 700;
    color: #66BB6A; margin: 1.2rem 0 0.6rem;
    border-bottom: 1px solid #2e5c2e; padding-bottom: 4px;
}

/* Nature badge */
.badge-bio    { background:#2e7d32; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-nonbio { background:#b71c1c; color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
.badge-unk    { background:#555;    color:#fff; padding:2px 10px; border-radius:99px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="dash-title">📊 Waste Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-sub">Long-term analytics & detection history</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔗 Navigation")
    st.page_link("app.py", label="🏠 Back to Main App", icon="🏠")
    st.markdown("---")
    st.header("🗄️ Database")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    st.markdown("---")
    if st.button("🗑️ Clear All History", type="secondary"):
        clear_all_detections()
        st.success("History cleared!")
        st.rerun()

# ─────────────────────────────────────────────
# Fetch data
# ─────────────────────────────────────────────
summary = fetch_summary()
dist    = fetch_distribution()     # list of {class_name, waste_nature, count}
nature  = fetch_nature_split()     # {Biodegradable: N, Non-Biodegradable: M}
recent  = fetch_recent(limit=50)
all_det = fetch_all_detections()

df_dist   = pd.DataFrame(dist)   if dist   else pd.DataFrame(columns=["class_name","waste_nature","count"])
df_recent = pd.DataFrame(recent) if recent else pd.DataFrame()
df_all    = pd.DataFrame(all_det) if all_det else pd.DataFrame()

# ─────────────────────────────────────────────
# KPI cards
# ─────────────────────────────────────────────
st.markdown('<div class="section-head">📈 Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
cards = [
    (c1, summary["total"],              "Total Scans"),
    (c2, summary["most_common"],        "Most Common"),
    (c3, f"{summary['avg_confidence']:.2f}", "Avg Confidence"),
    (c4, summary["biodegradable"],      "Biodegradable"),
    (c5, summary["non_biodegradable"],  "Non-Biodegradable"),
]
for col, val, label in cards:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Charts row
# ─────────────────────────────────────────────
if not df_dist.empty:
    chart_col1, chart_col2 = st.columns(2)

    # ── Donut: Biodegradable vs Non-Biodegradable ──────────────────
    with chart_col1:
        st.markdown('<div class="section-head">🌿 Biodegradable vs Non-Biodegradable</div>', unsafe_allow_html=True)
        nature_labels = list(nature.keys())
        nature_values = list(nature.values())
        nature_colors = ["#2E7D32", "#B71C1C", "#555555"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=nature_labels,
            values=nature_values,
            hole=0.55,
            marker=dict(colors=nature_colors[:len(nature_labels)]),
            textfont=dict(size=13),
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=300,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Bar: count per waste class ─────────────────────────────────
    with chart_col2:
        st.markdown('<div class="section-head">📦 Detection Distribution by Class</div>', unsafe_allow_html=True)
        color_map = {"Biodegradable": "#2E7D32", "Non-Biodegradable": "#B71C1C", "Unknown": "#555"}
        fig_bar = px.bar(
            df_dist,
            x="class_name", y="count",
            color="waste_nature",
            color_discrete_map=color_map,
            labels={"class_name": "Waste Type", "count": "Count", "waste_nature": "Nature"},
            text="count",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            xaxis=dict(tickangle=-30, gridcolor="#333"),
            yaxis=dict(gridcolor="#333"),
            legend_title_text="",
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Confidence over time (if enough data) ─────────────────────
    if not df_all.empty and "timestamp" in df_all.columns:
        st.markdown('<div class="section-head">📉 Confidence Over Time</div>', unsafe_allow_html=True)
        df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
        fig_line = px.scatter(
            df_all.sort_values("timestamp"),
            x="timestamp", y="confidence",
            color="waste_nature",
            color_discrete_map={"Biodegradable": "#66BB6A", "Non-Biodegradable": "#EF5350", "Unknown": "#888"},
            labels={"timestamp": "Time", "confidence": "Confidence", "waste_nature": "Nature"},
            opacity=0.75,
        )
        fig_line.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333", range=[0,1]),
            legend_title_text="",
            margin=dict(t=10, b=10),
            height=260,
        )
        st.plotly_chart(fig_line, use_container_width=True)

else:
    st.info("📭 No detections yet. Run the classifier on some images first!")

# ─────────────────────────────────────────────
# Activity Log Tables — split by nature
# ─────────────────────────────────────────────
st.markdown('<div class="section-head">📋 Recent Activity Log</div>', unsafe_allow_html=True)

if not df_recent.empty:

    # ── helper to style rows ──────────────────────────────────────
    def style_nature(val):
        if val == "Biodegradable":
            return "background-color:#1b3a1b; color:#66BB6A; font-weight:600"
        elif val == "Non-Biodegradable":
            return "background-color:#3a1b1b; color:#EF5350; font-weight:600"
        return ""

    # ── Full table with colour-coded nature column ────────────────
    display_cols = ["id", "timestamp", "source", "class_name", "confidence", "bin_label", "waste_nature"]
    df_show = df_recent[display_cols].copy()
    df_show["confidence"] = df_show["confidence"].apply(lambda x: f"{x*100:.1f}%")
    df_show.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]

    styled = df_show.style.applymap(style_nature, subset=["Nature"])
    st.dataframe(styled, use_container_width=True, height=380)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Split tables: Biodegradable ───────────────────────────────
    tab_bio, tab_nonbio = st.tabs(["🌿 Biodegradable Items", "🚫 Non-Biodegradable Items"])

    with tab_bio:
        df_bio = df_recent[df_recent["waste_nature"] == "Biodegradable"].copy()
        if not df_bio.empty:
            df_bio = df_bio[display_cols].copy()
            df_bio["confidence"] = df_bio["confidence"].apply(lambda x: f"{x*100:.1f}%")
            df_bio.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]
            st.success(f"✅ {len(df_bio)} biodegradable item(s) detected")
            st.dataframe(df_bio, use_container_width=True)

            # Summarise by class
            st.markdown("**Summary by type**")
            summary_bio = df_bio.groupby("Waste Type").size().reset_index(name="Count").sort_values("Count", ascending=False)
            st.dataframe(summary_bio, use_container_width=True, hide_index=True)
        else:
            st.info("No biodegradable items in recent history.")

    with tab_nonbio:
        df_nonbio = df_recent[df_recent["waste_nature"] == "Non-Biodegradable"].copy()
        if not df_nonbio.empty:
            df_nonbio = df_nonbio[display_cols].copy()
            df_nonbio["confidence"] = df_nonbio["confidence"].apply(lambda x: f"{x*100:.1f}%")
            df_nonbio.columns = ["ID", "Timestamp", "Source", "Waste Type", "Confidence", "Bin", "Nature"]
            st.error(f"⚠️ {len(df_nonbio)} non-biodegradable item(s) detected")
            st.dataframe(df_nonbio, use_container_width=True)

            # Summarise by class
            st.markdown("**Summary by type**")
            summary_nonbio = df_nonbio.groupby("Waste Type").size().reset_index(name="Count").sort_values("Count", ascending=False)
            st.dataframe(summary_nonbio, use_container_width=True, hide_index=True)
        else:
            st.info("No non-biodegradable items in recent history.")

else:
    st.info("📭 Activity log is empty.")

# ─────────────────────────────────────────────
# Download full CSV
# ─────────────────────────────────────────────
if not df_all.empty:
    st.markdown('<div class="section-head">⬇️ Export Data</div>', unsafe_allow_html=True)
    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full History as CSV",
        data=csv,
        file_name="waste_detections.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown('<div style="text-align:center;color:#555;font-size:0.85rem;">♻️ Waste Classification Dashboard</div>', unsafe_allow_html=True)
