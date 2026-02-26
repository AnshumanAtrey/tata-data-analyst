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
    with open("assets/tata-logo.png", "rb") as f:
        return base64.b64encode(f.read()).decode()


# ───────────────── Page Config ─────────────────
st.set_page_config(
    page_title="TATA Retail Analytics",
    page_icon="assets/tata-logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────── Styles ─────────────────
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ───────────────── Load Data & Models ─────────────────
with st.spinner("Loading data & training models..."):
    df = load_and_clean_data()
    rfm_base = compute_rfm(df)
    rfm, km_model, lr_model, scaler_obj, cluster_map, model_acc, med_m = train_models(rfm_base)

logo_b64 = get_logo_b64()
logo_img = f'<img src="data:image/png;base64,{logo_b64}" style="width:24px;height:24px;object-fit:contain;">'
logo_img_lg = f'<img src="data:image/png;base64,{logo_b64}" style="width:28px;height:28px;object-fit:contain;">'

# ───────────────── Sidebar ─────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-logo">
        {logo_img}
        <span>TATA Retail</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Navigation</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-nav-item sidebar-nav-active">📊 &nbsp;Dashboard</div>
    <div class="sidebar-nav-item">👥 &nbsp;Segmentation</div>
    <div class="sidebar-nav-item">🎯 &nbsp;Predictor</div>
    <div class="sidebar-nav-item">💡 &nbsp;Insights</div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Team</p>', unsafe_allow_html=True)
    for m in ["Payal Kunwar", "Gaurav Kulkarni", "Anshuman Atrey", "Abdullah Haque", "Shlok Vijay Kadam"]:
        st.markdown(f'<div class="sidebar-nav-item">{m}</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-label">Dataset</p>', unsafe_allow_html=True)
    d_min = df["InvoiceDate"].min().strftime("%b %Y")
    d_max = df["InvoiceDate"].max().strftime("%b %Y")
    st.markdown(f"""
    <div class="sidebar-stat"><span>Transactions</span><span>{len(df):,}</span></div>
    <div class="sidebar-stat"><span>Customers</span><span>{df['CustomerID'].nunique():,}</span></div>
    <div class="sidebar-stat"><span>Period</span><span>{d_min} – {d_max}</span></div>
    <div class="sidebar-stat"><span>Model Accuracy</span><span>{model_acc:.1%}</span></div>
    """, unsafe_allow_html=True)

# ───────────────── Page Header ─────────────────
st.markdown(f"""
<div class="page-header">
    <div>
        <h1>
            {logo_img_lg}
            TATA Online Retail Store
        </h1>
        <span class="page-header-sub">Revenue Drivers Analysis &middot; Business Studies &middot; Group 11</span>
    </div>
    <span class="badge">Feb 2026</span>
</div>
""", unsafe_allow_html=True)

# ───────────────── Tabs ─────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Segmentation", "Predictor", "Insights", "About"])

with tab1:
    dashboard.render(df)

with tab2:
    segmentation.render(rfm)

with tab3:
    predictor.render(model_acc, km_model, lr_model, scaler_obj, cluster_map)

with tab4:
    insights.render(df)

with tab5:
    about.render()
