"""TATA Retail Analytics — Main Entry Point."""

import streamlit as st
import base64
import warnings

warnings.filterwarnings("ignore")

from utils.styles import GLOBAL_CSS
from utils.data import load_and_clean_data, compute_rfm, train_models
from components import dashboard, segmentation, predictor, insights, about

# ───────────────── Logo Helper ─────────────────
@st.cache_data
def get_logo_b64():
    try:
        with open("assets/tata-logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# ───────────────── Page Config ─────────────────
st.set_page_config(
    page_title="TATA Retail Analytics",
    page_icon="assets/tata-logo.png",
    layout="wide",
    initial_sidebar_state="collapsed", 
)

# ───────────────── BRUTE FORCE NAVBAR CSS ─────────────────
st.markdown("""
<style>
    /* 1. Hide default Streamlit headers */
    header, [data-testid="stHeader"], [data-testid="stSidebar"] { display: none !important; }
    
    /* 2. Main Content Gutter */
    .stMainBlockContainer {
        padding-top: 120px !important; 
        padding-left: 2rem !important; 
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* 3. The Horizontal Navbar Container */
    div[data-testid="stHorizontalBlock"]:has(button[key^="nav_"]) {
        position: fixed !important;
        top: 30px !important;
        left: 35px !important; 
        z-index: 999998 !important;
        display: flex !important;
        gap: 20px !important;  /* Creates the 20px gap */
        width: auto !important;
    }

    /* 4. Target Individual Columns */
    div[data-testid="stHorizontalBlock"]:has(button[key^="nav_"]) > div[data-testid="column"] {
        flex: none !important;
        width: auto !important;
        min-width: auto !important;
        padding: 0 !important;
    }

    /* 5. Pill Button Styling */
    .stButton > button {
        border-radius: 30px !important;
        padding: 10px 24px !important;
        height: auto !important;
        width: auto !important;
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #e5e7eb !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        transition: 0.2s ease !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }
    
    .stButton > button:hover {
        border-color: #d1d5db !important;
        background: #f8fafc !important;
    }
    
    /* Active State (Black Pill) */
    .stButton > button[kind="primary"] {
        background: #0f172a !important;
        color: white !important;
        border: 1px solid #0f172a !important;
        border-radius: 30px !important; /* EXPLICIT FIX: Keeps the pill shape */
    }
    
    .big-header-container { margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# ───────────────── Navigation Logic ─────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

# Lag Fix: Update the session state via callback, avoiding a double-rerun
def change_page(new_page):
    st.session_state["page"] = new_page

PAGE_CONFIG = {
    "Dashboard": {"title": "Dashboard Overview", "subtitle": "Monitoring TATA retail revenue and performance."},
    "Segmentation": {"title": "Customer Groups", "subtitle": "AI-driven RFM customer clustering."},
    "Predictor": {"title": "AI Predictor Agent", "subtitle": "Forecast sales and identify buyer trends."},
    "Insights": {"title": "Business Insights", "subtitle": "Strategic recommendations for inventory."},
    "About": {"title": "Project Config", "subtitle": "Group 11 · Business Studies · 2026"}
}

# ───────────────── Render Unified Navbar ─────────────────
with st.container():
    cols = st.columns(11)
    
    # Col 0 is the Logo
    with cols[0]:
        logo_b64 = get_logo_b64()
        st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:40px;height:40px;object-fit:contain;margin-top:2px;">', unsafe_allow_html=True)
    
    # Cols 1-5 are the Buttons (Now using on_click)
    for i, (page_name, config) in enumerate(PAGE_CONFIG.items()):
        btn_type = "primary" if st.session_state["page"] == page_name else "secondary"
        with cols[i + 1]: 
            st.button(
                page_name, 
                key=f"nav_{page_name}", 
                type=btn_type,
                on_click=change_page,  # Hooks up the callback
                args=(page_name,)      # Passes the page name to the callback
            )

# ───────────────── Main Content ─────────────────
with st.spinner("Synchronizing Data..."):
    df = load_and_clean_data()
    rfm_base = compute_rfm(df)
    rfm, km_model, lr_model, scaler_obj, cluster_map, model_acc, med_m = train_models(rfm_base)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

curr_page = st.session_state["page"]
curr_config = PAGE_CONFIG[curr_page]

# Page Header
st.markdown(f"""
<div class="big-header-container">
    <h1 style="font-size:3.2rem; font-weight:600; color:#111827; letter-spacing:-1.2px; margin:0;">{curr_config['title']}</h1>
    <p style="color:#64748b; font-size:1.1rem; margin-top:5px;">{curr_config['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# Dynamic Content Rendering
if curr_page == "Dashboard":
    dashboard.render(df)
elif curr_page == "Segmentation":
    segmentation.render(rfm)
elif curr_page == "Predictor":
    predictor.render(model_acc, km_model, lr_model, scaler_obj, cluster_map, rfm, df)
elif curr_page == "Insights":
    insights.render(df)
elif curr_page == "About":
    about.render()