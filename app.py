import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="TATA Retail Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 25, 40, 0.6);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066ff 0%, #5c3bff 100%);
        color: white !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(17,25,40,0.85) 0%, rgba(30,41,59,0.85) 100%);
        border: 1px solid rgba(100, 150, 255, 0.15);
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 500;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-weight: 700;
        font-size: 1.75rem !important;
    }

    /* Custom cards */
    .insight-card {
        background: linear-gradient(135deg, rgba(17,25,40,0.9) 0%, rgba(30,41,59,0.9) 100%);
        border: 1px solid rgba(100, 150, 255, 0.12);
        border-radius: 14px;
        padding: 28px;
        margin: 10px 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.12);
    }
    .insight-card h3 {
        color: #60a5fa;
        margin-bottom: 12px;
        font-weight: 700;
    }
    .insight-card p {
        color: #cbd5e1;
        line-height: 1.7;
        font-size: 0.95rem;
    }

    /* Header hero */
    .hero-header {
        background: linear-gradient(135deg, #0a1628 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(100, 150, 255, 0.1);
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-header h1 {
        background: linear-gradient(135deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* Prediction result */
    .prediction-box {
        background: linear-gradient(135deg, rgba(17,25,40,0.95) 0%, rgba(30,41,59,0.95) 100%);
        border: 1px solid rgba(100, 150, 255, 0.2);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .prediction-box h2 {
        color: #60a5fa;
        font-size: 1.5rem;
        margin-bottom: 8px;
    }
    .prediction-box .big-number {
        font-size: 3rem;
        font-weight: 800;
        margin: 12px 0;
    }
    .prediction-box p {
        color: #94a3b8;
    }

    .high-value { color: #34d399; }
    .low-value { color: #f87171; }

    /* Team card */
    .team-card {
        background: linear-gradient(135deg, rgba(17,25,40,0.9) 0%, rgba(30,41,59,0.9) 100%);
        border: 1px solid rgba(100, 150, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .team-card h4 {
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .team-card p {
        color: #64748b;
        font-size: 0.85rem;
    }

    /* Section title */
    .section-title {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(96, 165, 250, 0.3);
    }

    /* Recommendation cards */
    .rec-card {
        background: rgba(17, 25, 40, 0.7);
        border-left: 4px solid #60a5fa;
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        margin: 10px 0;
    }
    .rec-card h4 { color: #60a5fa; margin-bottom: 6px; }
    .rec-card p { color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 14px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Plotly Template ───────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(10,22,40,0.6)",
        plot_bgcolor="rgba(10,22,40,0.4)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        title_font=dict(size=18, color="#e2e8f0"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        colorway=["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#38bdf8", "#e879f9", "#fb923c"],
        margin=dict(l=40, r=40, t=60, b=40),
    )
)

COLOR_PALETTE = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#38bdf8", "#e879f9", "#fb923c"]

CLUSTER_NAMES = {
    0: "Champions",
    1: "Potential Loyalists",
    2: "At-Risk Customers",
    3: "Hibernating",
}
CLUSTER_COLORS = {
    0: "#34d399",
    1: "#60a5fa",
    2: "#fbbf24",
    3: "#f87171",
}
CLUSTER_DESCRIPTIONS = {
    0: "High-frequency, high-spend customers who purchased recently. These are your most valuable customers and brand advocates.",
    1: "Moderate frequency and spend with recent activity. They have strong potential to become Champions with the right engagement.",
    2: "Previously valuable customers whose activity is declining. Immediate re-engagement strategies are critical to prevent churn.",
    3: "Low activity across all dimensions. These customers have become dormant and need significant incentives to reactivate.",
}
CLUSTER_RECOMMENDATIONS = {
    0: [
        "Offer exclusive VIP loyalty rewards",
        "Early access to new product launches",
        "Personalized thank-you communications",
        "Referral program incentives",
    ],
    1: [
        "Upsell and cross-sell related products",
        "Tiered loyalty program enrollment",
        "Personalized product recommendations",
        "Free shipping on next 3 orders",
    ],
    2: [
        "Win-back email campaign with special discounts",
        "Survey to understand disengagement reasons",
        "Limited-time exclusive comeback offers",
        "Highlight new products since last purchase",
    ],
    3: [
        "Aggressive reactivation discount (20-30% off)",
        "Brand story re-engagement campaign",
        "Low-cost sampling or trial offers",
        "Consider reducing marketing spend if unresponsive",
    ],
}


# ─────────────────────── Data Loading & Models ─────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean_data():
    """Load and clean the Online Retail dataset."""
    df = pd.read_csv(
        "dataset/Online Retail Data Set.csv",
        encoding="ISO-8859-1",
        dtype={"CustomerID": str, "InvoiceNo": str},
    )
    # Remove cancellations (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    # Remove negative / zero quantity and price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # Drop rows without CustomerID
    df = df.dropna(subset=["CustomerID"])
    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True)
    # Total revenue column
    df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data(show_spinner=False)
def compute_rfm(df):
    """Compute RFM table from cleaned dataframe."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalRevenue", "sum"),
    ).reset_index()
    return rfm


@st.cache_data(show_spinner=False)
def train_models(_rfm):
    """Train K-Means and Logistic Regression on RFM data."""
    rfm = _rfm.copy()
    features = rfm[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-Means with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=300)
    rfm["Cluster"] = kmeans.fit_predict(features_scaled)

    # Sort clusters by mean Monetary descending so cluster 0 = best
    cluster_order = (
        rfm.groupby("Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    mapping = {old: new for new, old in enumerate(cluster_order)}
    rfm["Cluster"] = rfm["Cluster"].map(mapping)

    # Re-fit kmeans with relabelled data is not necessary for prediction;
    # instead we store the mapping and scaler for prediction.

    # Logistic Regression: high-value = Monetary > median
    median_monetary = rfm["Monetary"].median()
    rfm["HighValue"] = (rfm["Monetary"] > median_monetary).astype(int)

    X = features_scaled
    y = rfm["HighValue"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logreg = LogisticRegression(random_state=42, max_iter=1000)
    logreg.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, logreg.predict(X_test))

    return rfm, kmeans, logreg, scaler, mapping, accuracy, median_monetary


# ─────────────────────── Helpers ───────────────────────────────────
def fmt_currency(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:,.1f}M"
    if val >= 1_000:
        return f"${val/1_000:,.1f}K"
    return f"${val:,.2f}"


def fmt_number(val):
    if val >= 1_000_000:
        return f"{val/1_000_000:,.1f}M"
    if val >= 1_000:
        return f"{val/1_000:,.0f}K"
    return f"{val:,.0f}"


def predict_cluster(recency, frequency, monetary, scaler, kmeans, mapping):
    """Predict cluster for given RFM values."""
    point = scaler.transform([[recency, frequency, monetary]])
    raw_cluster = kmeans.predict(point)[0]
    return mapping.get(raw_cluster, raw_cluster)


def predict_high_value(recency, frequency, monetary, scaler, logreg):
    """Predict high/low value for given RFM values."""
    point = scaler.transform([[recency, frequency, monetary]])
    pred = logreg.predict(point)[0]
    prob = logreg.predict_proba(point)[0]
    return pred, prob


# Country code mapping for choropleth
COUNTRY_CODES = {
    "United Kingdom": "GBR", "France": "FRA", "Germany": "DEU", "Spain": "ESP",
    "Belgium": "BEL", "Switzerland": "CHE", "Portugal": "PRT", "Italy": "ITA",
    "Finland": "FIN", "Austria": "AUT", "Norway": "NOR", "Netherlands": "NLD",
    "Australia": "AUS", "Sweden": "SWE", "Channel Islands": "GBR", "Denmark": "DNK",
    "Japan": "JPN", "Poland": "POL", "Singapore": "SGP", "Iceland": "ISL",
    "Israel": "ISR", "Canada": "CAN", "Greece": "GRC", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Lithuania": "LTU", "United Arab Emirates": "ARE",
    "USA": "USA", "Lebanon": "LBN", "Malta": "MLT", "Bahrain": "BHR",
    "RSA": "ZAF", "Saudi Arabia": "SAU", "Brazil": "BRA", "European Community": "EUR",
    "EIRE": "IRL", "Unspecified": None,
}


# ─────────────────────── Load Everything ───────────────────────────
with st.spinner("Loading data and training models..."):
    df = load_and_clean_data()
    rfm_base = compute_rfm(df)
    rfm, kmeans_model, logreg_model, scaler_obj, cluster_mapping, model_accuracy, median_monetary = train_models(rfm_base)

# ─────────────────────── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 2.8rem;">📊</div>
        <h2 style="background: linear-gradient(135deg, #60a5fa, #a78bfa);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-size: 1.4rem; font-weight: 800; margin: 4px 0;">
            TATA Retail Analytics
        </h2>
        <p style="color: #64748b; font-size: 0.8rem; margin-top: -4px;">Revenue Drivers Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("##### 👥 Group 11")
    members = ["Payal Kunwar", "Gaurav Kulkarni", "Anshuman Atrey", "Abdullah Haque", "Shlok Vijay Kadam"]
    for m in members:
        st.markdown(f"<span style='color:#94a3b8; font-size:0.88rem;'>• {m}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 📁 Dataset Overview")
    date_min = df["InvoiceDate"].min().strftime("%b %d, %Y")
    date_max = df["InvoiceDate"].max().strftime("%b %d, %Y")
    st.markdown(f"""
    <div style="background: rgba(30,41,59,0.5); border-radius: 10px; padding: 14px; font-size: 0.85rem; color: #94a3b8;">
        <b style="color:#e2e8f0;">Rows:</b> {len(df):,}<br>
        <b style="color:#e2e8f0;">Columns:</b> {len(df.columns)}<br>
        <b style="color:#e2e8f0;">Customers:</b> {df['CustomerID'].nunique():,}<br>
        <b style="color:#e2e8f0;">Date Range:</b><br>{date_min} &ndash; {date_max}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; font-size: 0.75rem; color: #475569;">
        Model Accuracy: <b style="color:#34d399;">{model_accuracy:.1%}</b><br>
        Built with Streamlit + Plotly
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────── Hero Header ──────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>TATA Online Retail Store</h1>
    <p>Revenue Drivers Analysis &nbsp;|&nbsp; Business Studies Project &nbsp;|&nbsp; Group 11</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────── Tabs ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Dashboard",
    "👥  Customer Segmentation",
    "🔮  Prediction Engine",
    "💡  Business Insights",
    "ℹ️  About",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 - DASHBOARD OVERVIEW
# ═══════════════════════════════════════════════════════════════════
with tab1:
    # KPIs
    total_revenue = df["TotalRevenue"].sum()
    total_customers = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()
    total_products = df["StockCode"].nunique()
    avg_order = df.groupby("InvoiceNo")["TotalRevenue"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", fmt_currency(total_revenue))
    c2.metric("Unique Customers", fmt_number(total_customers))
    c3.metric("Total Orders", fmt_number(total_orders))
    c4.metric("Unique Products", fmt_number(total_products))
    c5.metric("Avg Order Value", fmt_currency(avg_order))

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly Revenue Trend
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-title">Monthly Revenue Trend</p>', unsafe_allow_html=True)
        monthly = df.set_index("InvoiceDate").resample("M")["TotalRevenue"].sum().reset_index()
        monthly.columns = ["Month", "Revenue"]
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color="#60a5fa", width=3),
            marker=dict(size=7, color="#a78bfa"),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Revenue: $%{y:,.0f}<extra></extra>",
        ))
        fig_monthly.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=380,
            showlegend=False,
            yaxis_title="Revenue ($)",
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_right:
        st.markdown('<p class="section-title">Top 10 Products by Revenue</p>', unsafe_allow_html=True)
        top_products = (
            df.groupby("Description")["TotalRevenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )
        fig_products = go.Figure(go.Bar(
            x=top_products.values,
            y=top_products.index,
            orientation="h",
            marker=dict(
                color=top_products.values,
                colorscale=[[0, "#1e3a5f"], [0.5, "#60a5fa"], [1, "#a78bfa"]],
            ),
            hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.0f}<extra></extra>",
        ))
        fig_products.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=380,
            showlegend=False,
            xaxis_title="Revenue ($)",
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_products, use_container_width=True)

    # Row 2: Choropleth + Hourly/Daily
    col_map, col_patterns = st.columns([3, 2])

    with col_map:
        st.markdown('<p class="section-title">Revenue by Country</p>', unsafe_allow_html=True)
        country_rev = df.groupby("Country")["TotalRevenue"].sum().reset_index()
        country_rev["ISO"] = country_rev["Country"].map(COUNTRY_CODES)
        country_rev = country_rev.dropna(subset=["ISO"])
        fig_map = px.choropleth(
            country_rev,
            locations="ISO",
            color="TotalRevenue",
            hover_name="Country",
            color_continuous_scale=["#0f172a", "#1e3a5f", "#60a5fa", "#a78bfa"],
            labels={"TotalRevenue": "Revenue ($)"},
        )
        fig_map.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=420,
            geo=dict(
                bgcolor="rgba(10,22,40,0.4)",
                lakecolor="rgba(10,22,40,0.4)",
                landcolor="rgba(30,41,59,0.6)",
                showframe=False,
            ),
            coloraxis_colorbar=dict(title="Revenue", tickformat="$,.0f"),
        )
        fig_map.update_geos(projection_type="natural earth", showcoastlines=True, coastlinecolor="rgba(148,163,184,0.2)")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_patterns:
        st.markdown('<p class="section-title">Sales Patterns</p>', unsafe_allow_html=True)

        # Hourly
        df_hour = df.copy()
        df_hour["Hour"] = df_hour["InvoiceDate"].dt.hour
        hourly = df_hour.groupby("Hour")["TotalRevenue"].sum().reset_index()

        fig_patterns = make_subplots(rows=2, cols=1, subplot_titles=("Hourly Revenue", "Daily Revenue"), vertical_spacing=0.18)
        fig_patterns.add_trace(go.Bar(
            x=hourly["Hour"], y=hourly["TotalRevenue"],
            marker_color="#60a5fa", name="Hourly",
            hovertemplate="Hour %{x}:00<br>$%{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # Daily
        df_day = df.copy()
        df_day["DayOfWeek"] = df_day["InvoiceDate"].dt.dayofweek
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = df_day.groupby("DayOfWeek")["TotalRevenue"].sum().reset_index()
        daily["DayName"] = daily["DayOfWeek"].map(lambda x: day_names[x])

        fig_patterns.add_trace(go.Bar(
            x=daily["DayName"], y=daily["TotalRevenue"],
            marker_color="#a78bfa", name="Daily",
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ), row=2, col=1)

        fig_patterns.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=420,
            showlegend=False,
        )
        fig_patterns.update_annotations(font_size=13, font_color="#94a3b8")
        st.plotly_chart(fig_patterns, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 - CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">RFM Analysis & Customer Segmentation</p>', unsafe_allow_html=True)

    # RFM Explanation
    st.markdown("""
    <div class="insight-card">
        <h3>What is RFM Analysis?</h3>
        <p>
            <b>Recency</b> &mdash; How recently a customer made a purchase (fewer days = better).<br>
            <b>Frequency</b> &mdash; How often a customer makes purchases (more = better).<br>
            <b>Monetary</b> &mdash; How much a customer spends in total (more = better).<br><br>
            RFM analysis segments customers based on these three behavioural dimensions, enabling targeted
            marketing strategies that maximize ROI by treating different customer groups differently.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # RFM Distributions
    st.markdown('<p class="section-title">RFM Feature Distributions</p>', unsafe_allow_html=True)
    rfm_cols = st.columns(3)
    for i, (col_name, color) in enumerate([("Recency", "#60a5fa"), ("Frequency", "#a78bfa"), ("Monetary", "#34d399")]):
        with rfm_cols[i]:
            fig_dist = go.Figure(go.Histogram(
                x=rfm[col_name],
                nbinsx=50,
                marker_color=color,
                opacity=0.85,
                hovertemplate=f"{col_name}: " + "%{x}<br>Count: %{y}<extra></extra>",
            ))
            fig_dist.update_layout(
                **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                height=280,
                title=dict(text=col_name, font=dict(size=15)),
                showlegend=False,
                xaxis_title=col_name,
                yaxis_title="Customers",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3D Scatter + Cluster Profiles
    col_3d, col_profiles = st.columns([3, 2])

    with col_3d:
        st.markdown('<p class="section-title">Customer Clusters (3D View)</p>', unsafe_allow_html=True)
        # Sample for performance
        rfm_sample = rfm.sample(n=min(3000, len(rfm)), random_state=42)
        rfm_sample["ClusterName"] = rfm_sample["Cluster"].map(CLUSTER_NAMES)
        rfm_sample["ClusterColor"] = rfm_sample["Cluster"].map(CLUSTER_COLORS)

        fig_3d = px.scatter_3d(
            rfm_sample,
            x="Recency", y="Frequency", z="Monetary",
            color="ClusterName",
            color_discrete_map={v: CLUSTER_COLORS[k] for k, v in CLUSTER_NAMES.items()},
            opacity=0.7,
            hover_data={"CustomerID": True, "Recency": True, "Frequency": True, "Monetary": ":.2f"},
        )
        fig_3d.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=500,
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(10,22,40,0.4)", gridcolor="rgba(148,163,184,0.08)"),
                yaxis=dict(backgroundcolor="rgba(10,22,40,0.4)", gridcolor="rgba(148,163,184,0.08)"),
                zaxis=dict(backgroundcolor="rgba(10,22,40,0.4)", gridcolor="rgba(148,163,184,0.08)"),
            ),
            legend=dict(title="Segment", font=dict(size=11)),
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_profiles:
        st.markdown('<p class="section-title">Cluster Profiles</p>', unsafe_allow_html=True)

        cluster_summary = rfm.groupby("Cluster").agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
        ).reset_index()
        cluster_summary["Segment"] = cluster_summary["Cluster"].map(CLUSTER_NAMES)
        cluster_summary = cluster_summary[["Segment", "Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary"]]
        cluster_summary.columns = ["Segment", "Customers", "Avg Recency (days)", "Avg Frequency", "Avg Monetary ($)"]
        cluster_summary["Avg Monetary ($)"] = cluster_summary["Avg Monetary ($)"].apply(lambda x: f"${x:,.0f}")
        cluster_summary["Avg Recency (days)"] = cluster_summary["Avg Recency (days)"].apply(lambda x: f"{x:.0f}")
        cluster_summary["Avg Frequency"] = cluster_summary["Avg Frequency"].apply(lambda x: f"{x:.1f}")
        st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

        # Pie chart
        st.markdown("<br>", unsafe_allow_html=True)
        dist_data = rfm["Cluster"].value_counts().sort_index()
        fig_pie = go.Figure(go.Pie(
            labels=[CLUSTER_NAMES[i] for i in dist_data.index],
            values=dist_data.values,
            marker=dict(colors=[CLUSTER_COLORS[i] for i in dist_data.index]),
            hole=0.45,
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
            height=320,
            title=dict(text="Segment Distribution", font=dict(size=15)),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Segment Descriptions
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Segment Business Interpretation</p>', unsafe_allow_html=True)
    seg_cols = st.columns(4)
    for idx in range(4):
        with seg_cols[idx]:
            st.markdown(f"""
            <div class="insight-card" style="border-top: 3px solid {CLUSTER_COLORS[idx]};">
                <h3 style="color: {CLUSTER_COLORS[idx]};">{CLUSTER_NAMES[idx]}</h3>
                <p>{CLUSTER_DESCRIPTIONS[idx]}</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 - PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Customer Value Predictor</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <h3>How It Works</h3>
        <p>
            Enter a customer's RFM profile below. The engine uses two machine learning models trained on the
            dataset to predict the customer segment (K-Means Clustering) and whether the customer is
            <b>High-Value</b> or <b>Low-Value</b> (Logistic Regression with <b>{acc}</b> accuracy).
        </p>
    </div>
    """.format(acc=f"{model_accuracy:.1%}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    input_col, result_col = st.columns([2, 3])

    with input_col:
        st.markdown("#### Input Customer RFM Profile")

        recency_input = st.slider(
            "Recency (days since last purchase)",
            min_value=0, max_value=365, value=30, step=1,
            help="How many days ago the customer last purchased",
        )
        frequency_input = st.slider(
            "Frequency (number of purchases)",
            min_value=1, max_value=500, value=10, step=1,
            help="Total number of distinct transactions",
        )
        monetary_input = st.slider(
            "Monetary (total spend in $)",
            min_value=0, max_value=50000, value=1500, step=50,
            help="Total amount spent by the customer",
        )

        predict_btn = st.button("🔮  Predict Customer Value", use_container_width=True, type="primary")

    with result_col:
        if predict_btn:
            cluster_pred = predict_cluster(recency_input, frequency_input, monetary_input, scaler_obj, kmeans_model, cluster_mapping)
            hv_pred, hv_prob = predict_high_value(recency_input, frequency_input, monetary_input, scaler_obj, logreg_model)

            segment_name = CLUSTER_NAMES.get(cluster_pred, "Unknown")
            segment_color = CLUSTER_COLORS.get(cluster_pred, "#60a5fa")
            value_label = "High Value" if hv_pred == 1 else "Low Value"
            value_class = "high-value" if hv_pred == 1 else "low-value"
            confidence = max(hv_prob) * 100

            st.markdown(f"""
            <div class="prediction-box">
                <h2>Prediction Results</h2>
                <hr style="border-color: rgba(100,150,255,0.15); margin: 12px 0;">
                <p style="color:#94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;">Customer Segment</p>
                <div class="big-number" style="color: {segment_color};">{segment_name}</div>
                <p style="color:#94a3b8; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 16px;">Value Classification</p>
                <div class="big-number {value_class}">{value_label}</div>
                <p style="margin-top: 8px;">Confidence: <b style="color:#e2e8f0;">{confidence:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            st.markdown("<br>", unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number=dict(suffix="%", font=dict(size=36, color="#e2e8f0")),
                title=dict(text="Prediction Confidence", font=dict(size=14, color="#94a3b8")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#475569"),
                    bar=dict(color=segment_color),
                    bgcolor="rgba(30,41,59,0.5)",
                    bordercolor="rgba(100,150,255,0.15)",
                    steps=[
                        dict(range=[0, 50], color="rgba(248,113,113,0.15)"),
                        dict(range=[50, 75], color="rgba(251,191,36,0.15)"),
                        dict(range=[75, 100], color="rgba(52,211,153,0.15)"),
                    ],
                ),
            ))
            fig_gauge.update_layout(
                **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                height=250,
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommendations
            st.markdown(f'<p class="section-title">Recommendations for {segment_name}</p>', unsafe_allow_html=True)
            recs = CLUSTER_RECOMMENDATIONS.get(cluster_pred, [])
            icons = ["🎯", "📦", "💬", "🚀"]
            rec_cols = st.columns(2)
            for ri, rec in enumerate(recs):
                with rec_cols[ri % 2]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <h4>{icons[ri % len(icons)]} Strategy {ri + 1}</h4>
                        <p>{rec}</p>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="prediction-box" style="min-height: 380px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 4rem; margin-bottom: 16px;">🔮</div>
                <h2>Awaiting Input</h2>
                <p style="color: #64748b; max-width: 400px; margin: 8px auto 0;">
                    Adjust the sliders on the left and click <b>"Predict Customer Value"</b> to see the
                    machine learning model's prediction for this customer profile.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 4 - BUSINESS INSIGHTS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Pareto Analysis (80/20 Rule)</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <h3>The Pareto Principle in Retail</h3>
        <p>
            The Pareto Principle states that roughly 80% of outcomes come from 20% of causes.
            In retail, this often means a small fraction of customers generates the majority of revenue.
            Understanding this distribution is critical for efficient resource allocation and targeted marketing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pareto Chart
    customer_rev = df.groupby("CustomerID")["TotalRevenue"].sum().sort_values(ascending=False).reset_index()
    customer_rev["CumulativePct"] = customer_rev["TotalRevenue"].cumsum() / customer_rev["TotalRevenue"].sum() * 100
    customer_rev["CustomerPct"] = (np.arange(1, len(customer_rev) + 1) / len(customer_rev)) * 100

    # Find exact 80% revenue threshold
    threshold_idx = (customer_rev["CumulativePct"] >= 80).idxmax()
    pct_customers_80 = customer_rev.loc[threshold_idx, "CustomerPct"]

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(
            x=customer_rev["CustomerPct"],
            y=customer_rev["TotalRevenue"],
            marker_color="rgba(96,165,250,0.4)",
            name="Customer Revenue",
            hovertemplate="Top %{x:.1f}% of customers<br>Revenue: $%{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_pareto.add_trace(
        go.Scatter(
            x=customer_rev["CustomerPct"],
            y=customer_rev["CumulativePct"],
            mode="lines",
            line=dict(color="#a78bfa", width=3),
            name="Cumulative %",
            hovertemplate="Top %{x:.1f}% of customers<br>Cumulative: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    # 80% line
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="#f87171", line_width=1.5,
                         annotation_text="80% Revenue", annotation_font_color="#f87171",
                         secondary_y=True)
    fig_pareto.add_vline(x=pct_customers_80, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                         annotation_text=f"{pct_customers_80:.1f}% Customers", annotation_font_color="#fbbf24")
    fig_pareto.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=420,
        xaxis_title="Customers (cumulative %)",
        yaxis_title="Revenue per Customer ($)",
        showlegend=True,
        legend=dict(x=0.6, y=0.3),
    )
    fig_pareto.update_yaxes(title_text="Cumulative Revenue %", secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown(f"""
    <div class="insight-card" style="border-left: 4px solid #fbbf24;">
        <h3 style="color:#fbbf24;">Key Finding</h3>
        <p>
            <b style="color:#e2e8f0;">{pct_customers_80:.1f}%</b> of customers generate
            <b style="color:#e2e8f0;">80%</b> of total revenue. This confirms the Pareto Principle
            and highlights the importance of retaining and nurturing top-tier customers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue Drivers Summary
    st.markdown('<p class="section-title">Key Revenue Drivers</p>', unsafe_allow_html=True)

    drivers = [
        {
            "icon": "🇬🇧",
            "title": "Geographic Concentration",
            "text": f"The United Kingdom accounts for {df[df['Country']=='United Kingdom']['TotalRevenue'].sum()/total_revenue*100:.1f}% of total revenue, making domestic market performance the single largest revenue driver.",
        },
        {
            "icon": "👑",
            "title": "High-Value Customer Retention",
            "text": f"The top {pct_customers_80:.0f}% of customers generate 80% of revenue. Customer retention strategies for this group directly impact the bottom line.",
        },
        {
            "icon": "📅",
            "title": "Seasonal & Temporal Patterns",
            "text": "Revenue peaks in Q4 (October-December) driven by holiday shopping. Weekday sales significantly outperform weekends, with peak hours between 10 AM and 3 PM.",
        },
        {
            "icon": "📦",
            "title": "Product Portfolio Optimization",
            "text": f"The top 10 products contribute a disproportionate share of revenue. Strategic inventory and marketing focus on these high-performers can maximize returns.",
        },
    ]

    dcols = st.columns(2)
    for di, driver in enumerate(drivers):
        with dcols[di % 2]:
            st.markdown(f"""
            <div class="rec-card">
                <h4>{driver['icon']} {driver['title']}</h4>
                <p>{driver['text']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Economic Concepts
    st.markdown('<p class="section-title">Economic Concepts Applied</p>', unsafe_allow_html=True)
    concepts = [
        {
            "title": "Market Segmentation",
            "desc": "Dividing the customer base into distinct groups (Champions, Potential Loyalists, At-Risk, Hibernating) allows for differentiated marketing strategies that optimize resource allocation and maximize customer lifetime value.",
        },
        {
            "title": "Price Elasticity of Demand",
            "desc": "The wide distribution of unit prices and varying purchase quantities across segments suggests different price sensitivities. High-value customers demonstrate lower price elasticity, while at-risk customers may be more responsive to price-based incentives.",
        },
        {
            "title": "Economies of Scale in Marketing",
            "desc": "By identifying the 80/20 customer split, TATA can achieve economies of scale by concentrating marketing expenditure on the highest-ROI segments rather than uniform mass marketing.",
        },
        {
            "title": "Consumer Surplus & Perceived Value",
            "desc": "Champions consistently spend above average, indicating significant consumer surplus and high perceived value. This surplus represents untapped pricing power or upselling potential for premium offerings.",
        },
    ]

    for concept in concepts:
        st.markdown(f"""
        <div class="insight-card">
            <h3>{concept['title']}</h3>
            <p>{concept['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Recommendations
    st.markdown('<p class="section-title">Strategic Recommendations</p>', unsafe_allow_html=True)
    recs_biz = [
        ("🎯", "Implement Tiered Loyalty Program", "Design a 4-tier loyalty program mirroring the customer segments. Champions receive premium benefits (free shipping, early access), while lower tiers are incentivized to move up."),
        ("📊", "Data-Driven Inventory Management", "Use purchase frequency and product revenue data to optimize stock levels. Prioritize high-revenue products and reduce carrying costs on slow movers."),
        ("🌍", "International Expansion Strategy", "While the UK dominates revenue, countries like the Netherlands, EIRE, and Germany show growth potential. Targeted localized campaigns can unlock new revenue streams."),
        ("🔄", "Automated Re-Engagement Campaigns", "Deploy automated email workflows triggered by RFM score changes. When a customer's recency score drops below a threshold, initiate win-back campaigns immediately."),
        ("📱", "Omnichannel Experience", "Integrate online retail data with potential offline touchpoints. Unified customer profiles enable consistent experiences and better segmentation accuracy."),
        ("💰", "Dynamic Pricing Strategy", "Leverage segment-specific price elasticity insights to implement personalized pricing or targeted discounts that maximize revenue without eroding margins."),
    ]
    rec_biz_cols = st.columns(3)
    for ri, (icon, title, text) in enumerate(recs_biz):
        with rec_biz_cols[ri % 3]:
            st.markdown(f"""
            <div class="rec-card">
                <h4>{icon} {title}</h4>
                <p>{text}</p>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5 - ABOUT
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-card">
        <h3>Project Overview</h3>
        <p>
            This project was developed as part of our <b>Business Studies</b> curriculum to analyze
            revenue drivers for the <b>TATA Online Retail Store</b>. Using real-world transactional data,
            we applied data science and machine learning techniques to uncover actionable business insights.<br><br>
            The analysis covers customer segmentation via RFM analysis and K-Means clustering,
            customer value prediction using Logistic Regression, Pareto analysis of revenue concentration,
            and strategic business recommendations grounded in economic theory.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Team
    st.markdown('<p class="section-title">Team Members &mdash; Group 11</p>', unsafe_allow_html=True)
    team = [
        ("Payal Kunwar", "Research & Analysis"),
        ("Gaurav Kulkarni", "Data Processing"),
        ("Anshuman Atrey", "ML & Development"),
        ("Abdullah Haque", "Business Strategy"),
        ("Shlok Vijay Kadam", "Visualization & Report"),
    ]
    team_cols = st.columns(5)
    for ti, (name, role) in enumerate(team):
        with team_cols[ti]:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size: 2.2rem; margin-bottom: 8px;">{'👩‍💼' if ti == 0 else '👨‍💼'}</div>
                <h4>{name}</h4>
                <p>{role}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Technologies
    st.markdown('<p class="section-title">Technologies Used</p>', unsafe_allow_html=True)
    tech_cols = st.columns(4)
    techs = [
        ("🐍", "Python", "Core programming language"),
        ("📊", "Streamlit", "Interactive web application framework"),
        ("📈", "Plotly", "Advanced interactive visualizations"),
        ("🤖", "Scikit-learn", "Machine learning (K-Means, Logistic Regression)"),
    ]
    for ti, (icon, name, desc) in enumerate(techs):
        with tech_cols[ti]:
            st.markdown(f"""
            <div class="team-card">
                <div style="font-size: 2rem;">{icon}</div>
                <h4>{name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset Source
    st.markdown('<p class="section-title">Dataset</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-card">
        <h3>Online Retail Data Set</h3>
        <p>
            <b>Source:</b> UCI Machine Learning Repository<br>
            <b>Link:</b> <a href="https://archive.ics.uci.edu/ml/datasets/Online+Retail" target="_blank"
                style="color: #60a5fa;">https://archive.ics.uci.edu/ml/datasets/Online+Retail</a><br><br>
            This is a transnational dataset containing all transactions occurring between 01/12/2010 and
            09/12/2011 for a UK-based non-store online retail company. The company mainly sells unique
            all-occasion gifts, with many of its customers being wholesalers.
        </p>
    </div>
    """, unsafe_allow_html=True)
