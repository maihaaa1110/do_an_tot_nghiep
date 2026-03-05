"""Shared style and utility functions for BDS Dashboard."""

DARK_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

    /* ══════════════════════════════════════════════
       CORE LAYOUT
    ══════════════════════════════════════════════ */
    .stApp {
        background-color: #0b132b;
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(91,192,190,0.06) 0%, transparent 60%),
            radial-gradient(ellipse at 80% 20%, rgba(58,80,107,0.10) 0%, transparent 50%),
            linear-gradient(180deg, #0b132b 0%, #111827 100%);
        color: #e0e6ed;
        font-family: 'Arial', sans-serif;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1d38 0%, #1c2541 100%) !important;
        border-right: 1px solid rgba(91,192,190,0.2);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] p {
        color: #a8b8cc !important;
        font-size: 0.85rem;
    }

    /* HEADER */
    [data-testid="stHeader"] {
        background: rgba(11,19,43,0.95) !important;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(91,192,190,0.15);
    }

    /* ══════════════════════════════════════════════
       TYPOGRAPHY
    ══════════════════════════════════════════════ */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Arial', sans-serif !important;
        color: #5bc0be !important;
        letter-spacing: -0.02em;
    }
    h1 { font-size: 2.2rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.2rem !important; font-weight: 600 !important; }

    .mono { font-family: 'Space Mono', monospace; }

    /* ══════════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════════ */
    div[data-testid="stMetricValue"] {
        color: #5bc0be !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8899aa !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetricDelta"] { color: #5bc0be !important; }

    /* Metric container card */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2540 0%, #1c2e4a 100%);
        border: 1px solid rgba(91,192,190,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(91,192,190,0.5);
        transform: translateY(-2px);
    }

    /* ══════════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #5bc0be 0%, #3a7bd5 100%);
        color: #0b132b !important;
        font-weight: 700;
        font-family: 'Arial', sans-serif;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        letter-spacing: 0.03em;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(91,192,190,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(91,192,190,0.4);
    }
    .stButton > button:active { transform: translateY(0); }

    /* ══════════════════════════════════════════════
       INPUTS
    ══════════════════════════════════════════════ */
    input, select, textarea,
    div[data-baseweb="select"] div {
        background-color: #1a2540 !important;
        color: #e0e6ed !important;
        border-color: rgba(91,192,190,0.3) !important;
        border-radius: 8px !important;
    }
    input:focus, select:focus {
        border-color: #5bc0be !important;
        box-shadow: 0 0 0 2px rgba(91,192,190,0.2) !important;
    }

    /* ══════════════════════════════════════════════
       TABLES / DATAFRAMES
    ══════════════════════════════════════════════ */
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(91,192,190,0.2);
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] table {
        background-color: #131f38;
        color: #c8d8e8;
    }
    div[data-testid="stDataFrame"] thead tr th {
        background-color: #1c2e4a !important;
        color: #5bc0be !important;
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
    }
    div[data-testid="stDataFrame"] tbody tr:hover td {
        background-color: rgba(91,192,190,0.08) !important;
    }

    /* ══════════════════════════════════════════════
       TABS
    ══════════════════════════════════════════════ */
    div[data-baseweb="tab-list"] {
        gap: 4px;
        background: transparent;
        border-bottom: 1px solid rgba(91,192,190,0.2);
    }
    button[data-baseweb="tab"] {
        color: #6b7f99 !important;
        font-family: 'Arial', sans-serif !important;
        font-weight: 500;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.2s;
    }
    button[data-baseweb="tab"]:hover { color: #5bc0be !important; }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #5bc0be !important;
        background: rgba(91,192,190,0.12) !important;
        border-bottom: 2px solid #5bc0be !important;
        font-weight: 700;
    }

    /* ══════════════════════════════════════════════
       ALERTS / INFO BOXES
    ══════════════════════════════════════════════ */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border-left-width: 4px;
    }

    /* ══════════════════════════════════════════════
       DIVIDER
    ══════════════════════════════════════════════ */
    hr {
        border-color: rgba(91,192,190,0.2) !important;
        margin: 1.5rem 0 !important;
    }

    /* ══════════════════════════════════════════════
       SCROLLBAR
    ══════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #3a506b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #5bc0be; }
    ::-webkit-scrollbar-track { background: #0b132b; }

    /* ══════════════════════════════════════════════
       PROGRESS BAR
    ══════════════════════════════════════════════ */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #5bc0be, #3a7bd5) !important;
        border-radius: 10px;
    }

    /* ══════════════════════════════════════════════
       EXPANDER
    ══════════════════════════════════════════════ */
    div[data-testid="stExpander"] {
        background: #131f38;
        border: 1px solid rgba(91,192,190,0.15);
        border-radius: 10px;
    }
    div[data-testid="stExpander"] summary {
        color: #5bc0be !important;
        font-weight: 600;
    }

    /* ══════════════════════════════════════════════
       CUSTOM COMPONENTS
    ══════════════════════════════════════════════ */
    .section-title {
        font-family: 'Arial', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #5bc0be;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(91,192,190,0.3);
        margin-bottom: 1rem;
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        font-family: 'Space Mono', monospace;
    }
    .badge-teal { background: rgba(91,192,190,0.15); color: #5bc0be; border: 1px solid rgba(91,192,190,0.4); }
    .badge-green { background: rgba(46,204,113,0.15); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }
    .badge-red { background: rgba(231,76,60,0.15); color: #e74c3c; border: 1px solid rgba(231,76,60,0.4); }
    .badge-yellow { background: rgba(241,196,15,0.15); color: #f1c40f; border: 1px solid rgba(241,196,15,0.4); }

    .card {
        background: linear-gradient(135deg, #131f38 0%, #1a2a45 100%);
        border: 1px solid rgba(91,192,190,0.15);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
    }
    .card:hover { border-color: rgba(91,192,190,0.35); }

    .highlight-box {
        background: linear-gradient(135deg, rgba(91,192,190,0.08), rgba(58,80,107,0.12));
        border: 1px solid rgba(91,192,190,0.25);
        border-left: 4px solid #5bc0be;
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }

    .stat-pill {
        background: rgba(91,192,190,0.1);
        border: 1px solid rgba(91,192,190,0.3);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #5bc0be;
        margin: 2px;
    }

    /* Radio buttons */
    div[data-testid="stRadio"] label {
        color: #a8b8cc !important;
    }
    div[data-testid="stRadio"] input:checked + div {
        background: rgba(91,192,190,0.15) !important;
    }

    /* Number input */
    div[data-testid="stNumberInput"] input {
        font-family: 'Space Mono', monospace;
    }

    /* Selectbox dropdown */
    li[role="option"] {
        color: #e0e6ed !important;
        background: #1a2540 !important;
    }
    li[role="option"]:hover {
        background: rgba(91,192,190,0.2) !important;
    }

    /* Links */
    a { color: #5bc0be; }
    a:hover { color: #7dd4d2; }
</style>
"""

PLOTLY_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"family": "Arial, sans-serif", "color": "#c8d8e8", "size": 12},
    "colorway": ["#5bc0be", "#3a7bd5", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c", "#f1c40f"],
    "xaxis": {"gridcolor": "rgba(91,192,190,0.1)", "linecolor": "rgba(91,192,190,0.2)"},
    "yaxis": {"gridcolor": "rgba(91,192,190,0.1)", "linecolor": "rgba(91,192,190,0.2)"},
}


def apply_style(st):
    """Apply global dark style."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)


def page_header(st, title, subtitle=None, icon=None):
    """Render a styled page header."""
    icon_html = f"<span style='font-size:2rem;margin-right:0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(f"""
    <div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
        {icon_html}
        <h1 style='margin:0; font-size:2rem; font-weight:800;
                   background: linear-gradient(135deg, #5bc0be, #7dd4d2);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            {title}
        </h1>
        {"<p style='color:#8899aa; margin-top:0.4rem; font-size:1rem;'>"+subtitle+"</p>" if subtitle else ""}
    </div>
    <hr>
    """, unsafe_allow_html=True)


def section_header(st, title, icon=""):
    st.markdown(f"<div class='section-title'>{icon} {title}</div>", unsafe_allow_html=True)
