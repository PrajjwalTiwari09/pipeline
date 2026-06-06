"""
app.py
AI Data Quality Pipeline — Redesigned Streamlit frontend.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import io
import tempfile
import os

from azure_openai_service import generate_ai_summary
from Cleaner import clean
from Detector import detect

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="DataPulse · AI Quality Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# DESIGN SYSTEM — FULL CSS OVERHAUL
# ──────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">

<style>

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:        #05070f;
    --bg2:       #0c0f1d;
    --bg3:       #111525;
    --border:    #1e2540;
    --accent:    #00e5ff;
    --accent2:   #7c3aed;
    --accent3:   #f59e0b;
    --red:       #ff4d6d;
    --orange:    #fb923c;
    --green:     #22d3a0;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

/* ── App background ── */
.stApp {
    background: var(--bg) !important;
    font-family: var(--font-mono) !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0c0f1d 0%, #111525 50%, #0d1230 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 200px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(124,58,237,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(0,229,255,0.1);
    border: 1px solid rgba(0,229,255,0.25);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: var(--font-head);
    font-size: 3rem;
    font-weight: 800;
    color: #fff;
    line-height: 1.1;
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.02em;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: var(--font-mono);
    font-size: 0.9rem;
    color: var(--muted);
    max-width: 520px;
    line-height: 1.7;
    margin: 0;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 2.5rem 0 1.2rem 0;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
    box-shadow: 0 0 8px var(--accent);
}
.section-title {
    font-family: var(--font-head);
    font-size: 1.15rem;
    font-weight: 700;
    color: #fff;
    margin: 0;
    letter-spacing: -0.01em;
}
.section-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Metric cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0,229,255,0.3); }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.4;
}
.metric-card.warn::before { background: linear-gradient(90deg, transparent, var(--red), transparent); }
.metric-card.ok::before   { background: linear-gradient(90deg, transparent, var(--green), transparent); }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: var(--font-head);
    font-size: 2.1rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
}
.metric-value.accent { color: var(--accent); }
.metric-value.red    { color: var(--red); }
.metric-value.green  { color: var(--green); }
.metric-value.orange { color: var(--orange); }

/* ── Health score ring ── */
.health-ring-wrap {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.health-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.5rem;
}

/* ── Issue badges ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-high   { background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.3); }
.badge-medium { background: rgba(251,146,60,0.15);  color: #fb923c; border: 1px solid rgba(251,146,60,0.3); }
.badge-low    { background: rgba(34,211,160,0.12);  color: #22d3a0; border: 1px solid rgba(34,211,160,0.3); }

/* ── Decision cards ── */
.decision-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.decision-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
}
.decision-card.critical { border-color: rgba(255,77,109,0.35); background: rgba(255,77,109,0.04); }
.decision-card.warning  { border-color: rgba(251,146,60,0.35);  background: rgba(251,146,60,0.04); }
.decision-card.good     { border-color: rgba(34,211,160,0.35);  background: rgba(34,211,160,0.04); }
.decision-icon { font-size: 1.6rem; margin-bottom: 0.6rem; }
.decision-title {
    font-family: var(--font-head);
    font-size: 0.95rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
}
.decision-body {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── AI insight box ── */
.ai-box {
    background: linear-gradient(135deg, #0c0f1d, #111a2e);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    position: relative;
    overflow: hidden;
}
.ai-box::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(124,58,237,0.12), transparent 70%);
    pointer-events: none;
}
.ai-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}
.ai-header-title {
    font-family: var(--font-head);
    font-size: 1rem;
    font-weight: 700;
    color: #fff;
}
.ai-tag {
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    color: #a78bfa;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
}
.ai-content {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.9;
}
.ai-content strong, .ai-content b { color: #e2e8f0; }

/* ── Upload zone ── */
.upload-hint {
    background: var(--bg2);
    border: 1px dashed var(--border);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.upload-hint-title {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 0.4rem;
}
.upload-hint-sub {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--muted);
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    background: var(--bg2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 14px !important;
}
.stFileUploader label { color: var(--text) !important; font-family: var(--font-mono) !important; }
.stDataFrame { border-radius: 12px !important; overflow: hidden; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 12px !important; }
.stExpander { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; }
.stAlert { border-radius: 10px !important; font-family: var(--font-mono) !important; font-size: 0.85rem !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #00e5ff22, #7c3aed22) !important;
    border: 1px solid var(--accent) !important;
    color: var(--accent) !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #00e5ff33, #7c3aed33) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.2) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }
p, li { font-family: var(--font-mono) !important; color: var(--text) !important; font-size: 0.85rem !important; }
h1, h2, h3 { font-family: var(--font-head) !important; }
.stMarkdown h3 { color: #fff !important; font-size: 1rem !important; }

/* ── Pipeline steps ── */
.pipeline-steps {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}
.step {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
}
.step.active { border-color: var(--accent); color: var(--accent); background: rgba(0,229,255,0.05); }
.step.done   { border-color: var(--green);  color: var(--green);  background: rgba(34,211,160,0.05); }
.step-num {
    width: 20px; height: 20px;
    border-radius: 50%;
    background: var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.65rem; font-weight: 600;
    flex-shrink: 0;
}
.step.active .step-num { background: var(--accent); color: #000; }
.step.done   .step-num { background: var(--green);  color: #000; }
.step-arrow { color: var(--border); padding: 0 0.4rem; font-size: 0.9rem; }

/* ── Completeness bar ── */
.completeness-bar-wrap { margin: 0.5rem 0 1.5rem; }
.completeness-bar-bg {
    background: var(--border);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.completeness-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 1s ease;
}
.completeness-label {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
}

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def read_csv_with_fallback(file) -> pd.DataFrame:
    for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("Unable to read CSV with any supported encoding.")


def read_json(file) -> pd.DataFrame:
    """Handles JSON array, newline-delimited JSON (NDJSON), and nested objects."""
    file.seek(0)
    raw = file.read().decode("utf-8", errors="replace").strip()

    # Try standard JSON (array or single object)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return pd.json_normalize(parsed)
        elif isinstance(parsed, dict):
            # Could be {data: [...]} or a single record
            for v in parsed.values():
                if isinstance(v, list):
                    return pd.json_normalize(v)
            return pd.json_normalize([parsed])
    except json.JSONDecodeError:
        pass

    # Try NDJSON (one JSON object per line)
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return pd.json_normalize(records)
    except Exception:
        pass

    raise ValueError("Could not parse JSON file. Ensure it is a JSON array, object, or newline-delimited JSON.")


def read_parquet(file) -> pd.DataFrame:
    """Read a Parquet file uploaded via Streamlit."""
    file.seek(0)
    return pd.read_parquet(io.BytesIO(file.read()))


def read_delta(file) -> pd.DataFrame:
    """
    Read a Delta Lake table.
    Streamlit only supports single-file uploads, so we expect a .parquet
    part-file extracted from a Delta table, or a zipped Delta folder.
    For full Delta support (transaction log) users should mount a path.
    """
    name = file.name.lower()

    # Single parquet file from a Delta table
    if name.endswith(".parquet"):
        return read_parquet(file)

    # Zipped Delta table folder
    if name.endswith(".zip"):
        import zipfile
        file.seek(0)
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(io.BytesIO(file.read())) as zf:
                zf.extractall(tmpdir)
            # Find all parquet files and combine
            frames = []
            for root, _, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".parquet") and not f.startswith("."):
                        frames.append(pd.read_parquet(os.path.join(root, f)))
            if frames:
                return pd.concat(frames, ignore_index=True)
            raise ValueError("No parquet files found inside the Delta zip archive.")

    raise ValueError("For Delta files, upload a .parquet part-file or a zipped Delta folder (.zip).")


def read_file(file) -> pd.DataFrame:
    """Universal file reader — routes to the right handler by extension."""
    name = file.name.lower()
    if name.endswith(".csv"):
        return read_csv_with_fallback(file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    elif name.endswith(".json") or name.endswith(".ndjson"):
        return read_json(file)
    elif name.endswith(".parquet"):
        return read_parquet(file)
    elif name.endswith((".delta", ".zip")):
        return read_delta(file)
    else:
        raise ValueError(f"Unsupported file type: {file.name}")


def health_colour(score: float) -> str:
    if score >= 80: return "#22d3a0"
    if score >= 55: return "#f59e0b"
    return "#ff4d6d"


def section(title: str, icon: str = ""):
    label = f"{icon} {title}" if icon else title
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot"></div>
        <p class="section-title">{label}</p>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)


def metric_card(label: str, value: str, style: str = ""):
    cls = f"metric-card {style}"
    val_cls = {"warn": "red", "ok": "green", "accent": "accent"}.get(style, "accent")
    return f"""
    <div class="{cls}">
        <div class="metric-label">{label}</div>
        <div class="metric-value {val_cls}">{value}</div>
    </div>"""


# ──────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-tag">⚡ Powered by Azure OpenAI GPT-4o</div>
    <h1 class="hero-title">Data<span>Pulse</span></h1>
    <p class="hero-sub">
        Upload CSV, Excel, JSON, Parquet, or Delta files. The pipeline automatically detects issues,
        cleans your data, and delivers AI-powered insights so your team can act fast.
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PIPELINE TRACKER — live placeholder
# ──────────────────────────────────────────────

STEPS = ["Upload File", "Detect Issues", "Clean Data", "AI Insights", "Download"]

def render_steps(active: int, done: list):
    """Render the pipeline tracker. active=1-5, done=[1,2,...] completed steps."""
    html = '<div class="pipeline-steps">'
    for i, label in enumerate(STEPS, 1):
        if i in done:
            cls = "step done"
            num = "✓"
        elif i == active:
            cls = "step active"
            num = str(i)
        else:
            cls = "step"
            num = str(i)
        html += f'<div class="{cls}"><div class="step-num">{num}</div> {label}</div>'
        if i < len(STEPS):
            html += '<div class="step-arrow">›</div>'
    html += "</div>"
    return html

tracker = st.empty()
tracker.markdown(render_steps(1, []), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FILE UPLOADER
# ──────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Drop your file here",
    type=["csv", "xlsx", "xls", "json", "ndjson", "parquet", "zip"],
    help="Supports CSV · Excel · JSON · NDJSON · Parquet · Delta (zip) — up to 200 MB",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
    <div class="upload-hint">
        <div class="upload-hint-title">📂 Drop your file above to begin</div>
        <div class="upload-hint-sub">
            CSV &nbsp;·&nbsp; XLSX / XLS &nbsp;·&nbsp; JSON / NDJSON &nbsp;·&nbsp; Parquet &nbsp;·&nbsp; Delta (zip)
            &nbsp;|&nbsp; Auto-detects encoding & format
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

if uploaded_file is not None:
    try:

        # ── Read ────────────────────────────────────────
        df = read_file(uploaded_file)
        fmt = uploaded_file.name.rsplit(".", 1)[-1].upper()

        tracker.markdown(render_steps(2, [1]), unsafe_allow_html=True)
        st.success(f"✅  **{uploaded_file.name}** [{fmt}] · {len(df):,} rows · {len(df.columns)} columns")

        with st.expander("📋  Preview Original Data", expanded=False):
            st.dataframe(df.head(30), use_container_width=True)

        # ── Detect ──────────────────────────────────────
        with st.spinner("Scanning for data quality issues…"):
            report = detect(df)

        tracker.markdown(render_steps(3, [1, 2]), unsafe_allow_html=True)

        # ── Health Score (derived) ───────────────────────
        high_count   = sum(1 for i in report.issues if i.severity == "High")
        medium_count = sum(1 for i in report.issues if i.severity == "Medium")
        penalty      = (high_count * 15) + (medium_count * 5)
        health_score = max(0, min(100, report.completeness_pct - penalty))
        hc           = health_colour(health_score)

        # ── Top metrics row ──────────────────────────────
        section("Dataset Overview", "📊")

        cards_html = '<div class="metrics-grid">'
        cards_html += metric_card("Total Rows",     f"{report.total_rows:,}")
        cards_html += metric_card("Columns",        f"{report.total_columns}")
        cards_html += metric_card("Missing Cells",  f"{report.missing_cells:,}", "warn" if report.missing_cells > 0 else "ok")
        cards_html += metric_card("Duplicates",     f"{report.duplicate_rows:,}", "warn" if report.duplicate_rows > 0 else "ok")
        cards_html += metric_card("Issues Found",   f"{len(report.issues)}", "warn" if report.issues else "ok")
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Completeness bar ─────────────────────────────
        bar_color = health_colour(report.completeness_pct)
        st.markdown(f"""
        <div class="completeness-bar-wrap">
            <div class="completeness-label">
                <span>Data Completeness</span>
                <span style="color:{bar_color};font-weight:600">{report.completeness_pct:.1f}%</span>
            </div>
            <div class="completeness-bar-bg">
                <div class="completeness-bar-fill" style="width:{report.completeness_pct}%;background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column layout: chart + health ring ───────
        col_left, col_right = st.columns([3, 1])

        with col_left:
            section("Detected Issues", "🚨")
            if report.issues:
                issue_df = pd.DataFrame([{
                    "Severity":      i.severity,
                    "Category":      i.category,
                    "Field":         i.field,
                    "Description":   i.description,
                    "Affected Rows": i.affected_rows,
                } for i in report.issues])

                # Styled dataframe
                def sev_style(val):
                    c = {"High": "#ff4d6d", "Medium": "#fb923c", "Low": "#22d3a0"}
                    return f"color:{c.get(val,'#fff')};font-weight:600"

                st.dataframe(
                    issue_df.style.map(sev_style, subset=["Severity"]),
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, 45 + len(issue_df) * 35),
                )

                # Horizontal bar chart
                chart_df = (
                    issue_df.groupby(["Category", "Severity"])["Affected Rows"]
                    .sum().reset_index()
                    .sort_values("Affected Rows", ascending=True)
                )
                colour_map = {"High": "#ff4d6d", "Medium": "#fb923c", "Low": "#22d3a0"}
                fig = px.bar(
                    chart_df, x="Affected Rows", y="Category",
                    color="Severity", orientation="h",
                    color_discrete_map=colour_map,
                    title="Affected Rows by Issue Category",
                    barmode="stack",
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Mono", color="#94a3b8", size=11),
                    title_font=dict(family="Syne", color="#e2e8f0", size=13),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(0,0,0,0)"
                    ),
                    margin=dict(l=0, r=0, t=50, b=0),
                    xaxis=dict(gridcolor="#1e2540", zerolinecolor="#1e2540"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No issues detected — dataset is clean!")

        with col_right:
            section("Health Score", "❤️")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=health_score,
                number={"suffix": "%", "font": {"size": 32, "color": hc, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#1e2540",
                             "tickfont": {"color": "#64748b", "size": 9}},
                    "bar":  {"color": hc, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "rgba(255,77,109,0.08)"},
                        {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                        {"range": [70,100], "color": "rgba(34,211,160,0.08)"},
                    ],
                    "threshold": {"line": {"color": hc, "width": 2}, "value": health_score},
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=200,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            status = "Critical" if health_score < 40 else "Needs Work" if health_score < 70 else "Good"
            status_col = "#ff4d6d" if health_score < 40 else "#f59e0b" if health_score < 70 else "#22d3a0"
            st.markdown(f"""
            <div style="text-align:center;font-family:'DM Mono';font-size:0.8rem;">
                <span style="color:{status_col};font-weight:600">{status}</span>
                <span style="color:#64748b"> · {high_count} High · {medium_count} Medium</span>
            </div>""", unsafe_allow_html=True)

        # ── Decision Guidance ────────────────────────────
        section("Decision Guide", "🎯")

        if health_score >= 75:
            d1_cls, d1_icon, d1_title, d1_body = "good",     "✅", "Ready to Use",       "Dataset quality is good. Proceed with analysis, modelling, or reporting with confidence."
        elif health_score >= 45:
            d1_cls, d1_icon, d1_title, d1_body = "warning",  "⚠️", "Use with Caution",   "Moderate issues present. Review flagged columns before using in critical decisions."
        else:
            d1_cls, d1_icon, d1_title, d1_body = "critical", "🚫", "Do Not Use Raw",     "Significant data quality problems. Use the cleaned version only, or investigate source."

        if report.duplicate_rows > 0:
            d2_cls, d2_icon, d2_title, d2_body = "critical", "🔁", "Duplicates Detected", f"{report.duplicate_rows} duplicate rows found. Metrics like totals and averages will be skewed."
        else:
            d2_cls, d2_icon, d2_title, d2_body = "good",     "✅", "No Duplicates",        "No duplicate rows detected. Row-level analysis is safe."

        if report.missing_cells > 0:
            miss_pct = round((report.missing_cells / (report.total_rows * report.total_columns)) * 100, 1)
            d3_cls, d3_icon, d3_title, d3_body = "warning",  "🕳️", f"{miss_pct}% Missing Data", f"{report.missing_cells:,} cells are null. Cleaned version fills these with median/mode values."
        else:
            d3_cls, d3_icon, d3_title, d3_body = "good",     "✅", "No Missing Values",    "All cells are populated. No imputation needed."

        st.markdown(f"""
        <div class="decision-grid">
            <div class="decision-card {d1_cls}">
                <div class="decision-icon">{d1_icon}</div>
                <div class="decision-title">{d1_title}</div>
                <div class="decision-body">{d1_body}</div>
            </div>
            <div class="decision-card {d2_cls}">
                <div class="decision-icon">{d2_icon}</div>
                <div class="decision-title">{d2_title}</div>
                <div class="decision-body">{d2_body}</div>
            </div>
            <div class="decision-card {d3_cls}">
                <div class="decision-icon">{d3_icon}</div>
                <div class="decision-title">{d3_title}</div>
                <div class="decision-body">{d3_body}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Clean ────────────────────────────────────────
        with st.spinner("Cleaning data…"):
            cleaned_df, cr = clean(df)

        tracker.markdown(render_steps(4, [1, 2, 3]), unsafe_allow_html=True)

        section("Cleaned Dataset", "✨")

        c_cards = '<div class="metrics-grid">'
        c_cards += metric_card("Dupes Removed",     f"{cr['duplicates_removed']:,}", "ok" if cr['duplicates_removed'] > 0 else "")
        c_cards += metric_card("Nulls Filled",      f"{cr['nulls_filled']:,}",       "ok" if cr['nulls_filled'] > 0 else "")
        c_cards += metric_card("Casing Fixed",      f"{cr['casing_normalised']:,}",  "ok" if cr['casing_normalised'] > 0 else "")
        c_cards += metric_card("Dates Standardised",f"{cr['dates_standardised']:,}", "ok" if cr['dates_standardised'] > 0 else "")
        c_cards += metric_card("Final Row Count",   f"{cr['cleaned_rows']:,}")
        c_cards += '</div>'
        st.markdown(c_cards, unsafe_allow_html=True)

        with st.expander("🔍  View Cleaned Data", expanded=False):
            st.dataframe(cleaned_df, use_container_width=True)

        if cr["negative_values"]:
            st.warning(f"⚠️  Negative values found in {len(cr['negative_values'])} column(s) — review before aggregating.")
            st.dataframe(pd.DataFrame(cr["negative_values"]), use_container_width=True, hide_index=True)

        if cr["outliers_flagged"]:
            st.info(f"ℹ️  Outliers flagged in {len(cr['outliers_flagged'])} column(s) — rows retained, not removed.")
            st.dataframe(pd.DataFrame(cr["outliers_flagged"]), use_container_width=True, hide_index=True)

        # ── AI Insights ──────────────────────────────────
        section("AI-Generated Insights", "🤖")

        issue_str = pd.DataFrame([{
            "Severity": i.severity, "Category": i.category,
            "Field": i.field, "Description": i.description,
            "Affected Rows": i.affected_rows,
        } for i in report.issues]).to_string() if report.issues else "No issues detected."

        with st.spinner("💡  Generating insights via Azure OpenAI GPT-4o…"):
            try:
                summary = generate_ai_summary(
                    cleaned_df.head(20).to_string(),
                    issue_str,
                )
                st.markdown(f"""
                <div class="ai-box">
                    <div class="ai-header">
                        <span style="font-size:1.2rem">🤖</span>
                        <span class="ai-header-title">GPT-4o Analysis</span>
                        <span class="ai-tag">Azure OpenAI</span>
                    </div>
                    <div class="ai-content">{summary.replace(chr(10), '<br>')}</div>
                </div>
                """, unsafe_allow_html=True)
            except EnvironmentError as env_err:
                st.error(f"Azure OpenAI config error: {env_err}")
            except Exception as ai_err:
                st.error(f"AI summary failed: {ai_err}")

        tracker.markdown(render_steps(5, [1, 2, 3, 4]), unsafe_allow_html=True)

        # ── Natural Language Query ────────────────────────
        section("Ask Your Data", "💬")

        st.markdown("""
        <div class="ai-box" style="margin-bottom:1.5rem;">
            <div class="ai-header">
                <span style="font-size:1.2rem">💬</span>
                <span class="ai-header-title">Natural Language Query</span>
                <span class="ai-tag">GPT-4o · Live</span>
            </div>
            <div class="ai-content" style="margin-bottom:0;">
                Ask any question about your cleaned data in plain English.
                GPT-4o will write and run the pandas code instantly.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Suggested queries based on actual columns
        numeric_cols = cleaned_df.select_dtypes(include="number").columns.tolist()
        text_cols    = cleaned_df.select_dtypes(include="object").columns.tolist()

        suggestions = []
        if numeric_cols:
            suggestions.append(f"Show rows where {numeric_cols[0]} is above average")
            suggestions.append(f"What is the total {numeric_cols[0]}?")
            suggestions.append(f"Which {text_cols[0] if text_cols else 'row'} has the highest {numeric_cols[0]}?")
        if text_cols:
            suggestions.append(f"How many unique values are in {text_cols[0]}?")
        suggestions.append("Show me rows with any remaining issues")
        suggestions.append("What are the top 5 rows by row number?")

        st.markdown("""
        <p style="font-family:'DM Mono';font-size:0.75rem;color:#64748b;margin-bottom:0.4rem;">
        ✨ Try a suggestion or type your own:
        </p>""", unsafe_allow_html=True)

        # ── Session state ─────────────────────────────────
        if "nl_query_result" not in st.session_state:
            st.session_state.nl_query_result = None
        if "nl_last_query" not in st.session_state:
            st.session_state.nl_last_query = ""
        if "nl_prefill" not in st.session_state:
            st.session_state.nl_prefill = ""

        # Suggestion chips
        chip_cols = st.columns(len(suggestions[:4]))
        for i, (c, sug) in enumerate(zip(chip_cols, suggestions[:4])):
            with c:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state.nl_prefill = sug
                    st.session_state.nl_query_result = None
                    st.rerun()

        # st.form batches input + button into ONE rerun — no race condition
        with st.form(key="nl_query_form", clear_on_submit=False):
            nl_query = st.text_input(
                "Your question",
                value=st.session_state.nl_prefill,
                placeholder="e.g. show me 10 rows",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button(
                "🔍  Run Query",
                type="primary",
                use_container_width=False,
            )

        if submitted and nl_query and nl_query.strip():
            st.session_state.nl_prefill = nl_query
            with st.spinner("🧠  GPT-4o is thinking…"):
                try:
                    from NLQueryEngine import NLQueryEngine
                    engine = NLQueryEngine()
                    qr = engine.query(nl_query.strip(), cleaned_df)
                    st.session_state.nl_query_result = qr
                    st.session_state.nl_last_query   = nl_query.strip()
                except Exception as eng_err:
                    st.error(f"Query engine error: {eng_err}")
                    st.session_state.nl_query_result = None
        
        qr = st.session_state.nl_query_result

        # ── DEBUG (remove after confirming it works) ──────
        if qr is not None:
            st.info(
                f"**Debug** | success={qr.success} | "
                f"type={qr.result_type} | "
                f"code=`{qr.pandas_code[:80]}` | "
                f"error={qr.error!r}"
            )

        if qr:
            if qr.success:
                # ── Explanation ──────────────────────────
                st.markdown(f"""
                <div style="background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.2);
                            border-radius:10px;padding:1rem 1.2rem;margin:0.8rem 0;">
                    <span style="font-family:'DM Mono';font-size:0.8rem;color:#94a3b8;">
                    🔍 <strong style="color:#e2e8f0;">{qr.explanation}</strong>
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # ── Generated code ────────────────────────
                with st.expander("🧾  View generated pandas code", expanded=False):
                    st.code(qr.pandas_code, language="python")

                # ── Result ────────────────────────────────
                if qr.result_type == "dataframe":
                    result_df = qr.result
                    st.markdown(f"""
                    <p style="font-family:'DM Mono';font-size:0.78rem;color:#64748b;margin:0.5rem 0;">
                    Returned <strong style="color:#00e5ff;">{len(result_df):,} rows</strong>
                    × {len(result_df.columns)} columns
                    </p>""", unsafe_allow_html=True)

                    if len(result_df) == 0:
                        st.info("No rows match your query.")
                    else:
                        st.dataframe(result_df, use_container_width=True, hide_index=True)

                        # Auto chart if label + number cols present
                        num_cols_res = result_df.select_dtypes(include="number").columns.tolist()
                        obj_cols_res = result_df.select_dtypes(include="object").columns.tolist()
                        if num_cols_res and obj_cols_res and len(result_df) <= 50:
                            try:
                                fig_q = px.bar(
                                    result_df.head(20),
                                    x=obj_cols_res[0],
                                    y=num_cols_res[0],
                                    title=f"{num_cols_res[0]} by {obj_cols_res[0]}",
                                    color=num_cols_res[0],
                                    color_continuous_scale="Teal",
                                )
                                fig_q.update_layout(
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="DM Mono", color="#94a3b8", size=11),
                                    title_font=dict(family="Syne", color="#e2e8f0", size=13),
                                    coloraxis_showscale=False,
                                    margin=dict(l=0, r=0, t=50, b=0),
                                    xaxis=dict(gridcolor="#1e2540"),
                                    yaxis=dict(gridcolor="#1e2540"),
                                )
                                st.plotly_chart(fig_q, use_container_width=True)
                            except Exception:
                                pass

                        q_csv = result_df.to_csv(index=False).encode("utf-8-sig")  # BOM ensures Hindi/non-Latin text opens correctly in Excel
                        st.download_button(
                            "⬇️  Download query result",
                            data=q_csv,
                            file_name="query_result.csv",
                            mime="text/csv",
                            key="dl_query",
                        )

                elif qr.result_type == "scalar":
                    st.markdown(f"""
                    <div style="background:var(--bg2);border:1px solid var(--border);
                                border-radius:12px;padding:1.5rem 2rem;text-align:center;margin:0.5rem 0;">
                        <div style="font-family:'DM Mono';font-size:0.72rem;color:#64748b;
                                    text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">
                            Result
                        </div>
                        <div style="font-family:'Syne';font-size:2.4rem;font-weight:800;color:#00e5ff;">
                            {qr.result}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(f"❌  Query failed: {qr.error}")
                if qr.pandas_code:
                    with st.expander("View attempted code"):
                        st.code(qr.pandas_code, language="python")

        # ── Query history (session state) ─────────────
        if "query_history" not in st.session_state:
            st.session_state.query_history = []

        if nl_query and nl_query.strip() and "qr" in dir() and qr and qr.success:
            entry = {"question": nl_query, "code": qr.pandas_code, "explanation": qr.explanation}
            if entry not in st.session_state.query_history:
                st.session_state.query_history.insert(0, entry)
                st.session_state.query_history = st.session_state.query_history[:10]

        if st.session_state.get("query_history"):
            with st.expander(f"🕓  Query history ({len(st.session_state.query_history)} recent)", expanded=False):
                for i, h in enumerate(st.session_state.query_history):
                    st.markdown(f"""
                    <div style="border-left:2px solid #1e2540;padding:0.5rem 0.8rem;margin-bottom:0.6rem;">
                        <div style="font-family:'DM Mono';font-size:0.78rem;color:#e2e8f0;">
                            <strong>Q:</strong> {h["question"]}
                        </div>
                        <div style="font-family:'DM Mono';font-size:0.72rem;color:#64748b;margin-top:2px;">
                            {h["explanation"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Download ─────────────────────────────────────
        section("Export", "⬇️")
        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8-sig")  # BOM ensures Hindi/non-Latin text opens correctly in Excel
        st.download_button(
            label="⬇️  Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"❌  Pipeline Error: {e}")
        st.exception(e)

# ──────────────────────────────────────────────────────────────────────
# NATURAL LANGUAGE QUERY ENGINE  (injected after pipeline completes)
# ──────────────────────────────────────────────────────────────────────