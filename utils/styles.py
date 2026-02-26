"""CSS styles for the app."""

GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp, .main, section[data-testid="stMain"] {
        background-color: #fafafa !important;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 1rem;
        max-width: 1280px;
    }
    .block-container { padding-top: 0.5rem !important; }
    div[data-testid="stAppViewBlockContainer"] { padding-top: 0.5rem !important; }
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #171717;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    section[data-testid="stSidebar"] * {
        color: #525252 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] strong {
        color: #171717 !important;
    }

    .sidebar-section-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: #a3a3a3 !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 20px 0 8px 0;
    }
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0 4px 0;
    }
    .sidebar-logo img {
        width: 28px;
        height: 28px;
        border-radius: 6px;
    }
    .sidebar-logo span {
        font-size: 0.95rem;
        font-weight: 700;
        color: #171717 !important;
    }
    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 7px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #525252 !important;
        margin: 1px 0;
    }
    .sidebar-nav-active {
        background-color: #f5f5f5;
        color: #171717 !important;
        font-weight: 500;
    }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.82rem;
        color: #525252 !important;
        padding: 3px 10px;
    }
    .sidebar-stat span:last-child {
        color: #a3a3a3 !important;
        font-size: 0.78rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 3px;
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #737373;
        background: transparent;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #171717 !important;
        font-weight: 600;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        border: 1px solid #e5e7eb;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .kpi-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: #a3a3a3;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        margin-bottom: 2px;
    }
    .kpi-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #171717;
        line-height: 1.2;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 24px;
        margin: 8px 0;
    }
    .card h3 {
        color: #171717;
        font-size: 0.95rem;
        margin: 0 0 8px 0;
        font-weight: 600;
    }
    .card p {
        color: #525252;
        line-height: 1.65;
        font-size: 0.88rem;
        margin: 0;
    }
    .accent-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        margin: 6px 0;
    }
    .accent-card h4 {
        color: #171717;
        font-size: 0.88rem;
        margin: 0 0 4px 0;
        font-weight: 600;
    }
    .accent-card p {
        color: #525252;
        font-size: 0.84rem;
        line-height: 1.6;
        margin: 0;
    }

    /* ── Section headers ── */
    .section-title {
        color: #171717;
        font-size: 1rem;
        font-weight: 600;
        margin: 18px 0 10px 0;
    }

    /* ── Page header ── */
    .page-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 0 16px 0;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 20px;
    }
    .page-header h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #171717;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .page-header-sub {
        font-size: 0.82rem;
        color: #a3a3a3;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 500;
        background: #f5f5f5;
        color: #525252;
        border: 1px solid #e5e7eb;
    }

    /* ── Prediction ── */
    .prediction-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }
    .prediction-box h2 {
        color: #171717;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .pred-label {
        font-size: 0.7rem;
        color: #a3a3a3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 16px;
        margin-bottom: 2px;
    }
    .pred-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 2px 0;
    }
    .pred-green { color: #16a34a; }
    .pred-red { color: #dc2626; }
    .pred-blue { color: #171717; }

    /* ── Team ── */
    .team-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .team-card h4 {
        color: #171717;
        margin: 0 0 2px 0;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .team-card p {
        color: #a3a3a3;
        font-size: 0.78rem;
        margin: 0;
    }
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #171717;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 8px;
    }

    /* ── Misc ── */
    .stPlotlyChart {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
    }
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e5e7eb;
    }
    stButton {
        padding-left: 20px;
    }
    .stButton > button[kind="primary"] {
        background-color: #171717 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 500 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #404040 !important;
    }
    hr { border-color: #e5e7eb !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Transparent header */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
        border: none !important;
    }

    /* Hide toolbar */
    [data-testid="stToolbar"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stToolbar"] * {
        visibility: hidden !important;
    }

    /* Hide sidebar toggle buttons */
    button[data-testid="stExpandSidebarButton"],
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Hide sidebar entirely */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
