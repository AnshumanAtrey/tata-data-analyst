"""Tab 2: Customer Segmentation."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.config import CHART_LAYOUT, C_PRIMARY, CLUSTER_NAMES, CLUSTER_ACCENT, CLUSTER_DESCRIPTIONS


def render(rfm):
    st.markdown('<p class="section-title">RFM Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>What is RFM?</h3>
        <p>
            <strong>Recency</strong> — Days since last purchase (lower = better)<br>
            <strong>Frequency</strong> — Number of purchases (higher = better)<br>
            <strong>Monetary</strong> — Total spend (higher = better)<br><br>
            RFM segments customers into actionable groups for targeted marketing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Distributions</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, name in enumerate(["Recency", "Frequency", "Monetary"]):
        with cols[i]:
            fig = go.Figure(go.Histogram(x=rfm[name], nbinsx=50, marker_color=C_PRIMARY, opacity=0.75,
                hovertemplate=f"{name}: " + "%{x}<br>Count: %{y}<extra></extra>"))
            fig.update_layout(**CHART_LAYOUT, height=240, title=dict(text=name), showlegend=False,
                xaxis_title=name, yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    col3d, colprof = st.columns([3, 2])

    with col3d:
        st.markdown('<p class="section-title">Customer Clusters</p>', unsafe_allow_html=True)
        sample = rfm.sample(n=min(3000, len(rfm)), random_state=42)
        sample["Segment"] = sample["Cluster"].map(CLUSTER_NAMES)
        fig = px.scatter_3d(sample, x="Recency", y="Frequency", z="Monetary", color="Segment",
            color_discrete_map={v: CLUSTER_ACCENT[k] for k, v in CLUSTER_NAMES.items()},
            opacity=0.65, hover_data={"CustomerID": True, "Recency": True, "Frequency": True, "Monetary": ":.2f"})
        fig.update_layout(**CHART_LAYOUT, height=460,
            scene=dict(
                xaxis=dict(backgroundcolor="#fafafa", gridcolor="#e5e7eb"),
                yaxis=dict(backgroundcolor="#fafafa", gridcolor="#e5e7eb"),
                zaxis=dict(backgroundcolor="#fafafa", gridcolor="#e5e7eb")),
            legend=dict(title="", font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    with colprof:
        st.markdown('<p class="section-title">Cluster Profiles</p>', unsafe_allow_html=True)
        cs = rfm.groupby("Cluster").agg(
            Customers=("CustomerID", "count"), Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean")).reset_index()
        cs["Segment"] = cs["Cluster"].map(CLUSTER_NAMES)
        cs = cs[["Segment", "Customers", "Recency", "Frequency", "Monetary"]]
        cs.columns = ["Segment", "Count", "Avg Recency", "Avg Frequency", "Avg Monetary"]
        cs["Avg Monetary"] = cs["Avg Monetary"].apply(lambda x: f"£{x:,.0f}")
        cs["Avg Recency"] = cs["Avg Recency"].apply(lambda x: f"{x:.0f}d")
        cs["Avg Frequency"] = cs["Avg Frequency"].apply(lambda x: f"{x:.1f}")
        st.dataframe(cs, use_container_width=True, hide_index=True)

        st.markdown("")
        dist = rfm["Cluster"].value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=[CLUSTER_NAMES[i] for i in dist.index], values=dist.values,
            marker=dict(colors=[CLUSTER_ACCENT[i] for i in dist.index]),
            hole=0.55, textinfo="label+percent", textfont=dict(size=10, color="#171717"),
            hovertemplate="<b>%{label}</b><br>%{value:,}<br>%{percent}<extra></extra>"))
        fig.update_layout(**CHART_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("")
    st.markdown('<p class="section-title">Segment Profiles</p>', unsafe_allow_html=True)
    scols = st.columns(4)
    for idx in range(4):
        with scols[idx]:
            st.markdown(f"""
            <div class="card" style="border-top: 3px solid {CLUSTER_ACCENT[idx]};">
                <h3>{CLUSTER_NAMES[idx]}</h3>
                <p>{CLUSTER_DESCRIPTIONS[idx]}</p>
            </div>
            """, unsafe_allow_html=True)
