"""Tab 4: Business Insights."""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.config import CHART_LAYOUT, C_PRIMARY


def render(df):
    total_revenue = df["TotalRevenue"].sum()

    st.markdown('<p class="section-title">Pareto Analysis</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>The 80/20 Rule</h3>
        <p>The Pareto Principle states roughly 80% of outcomes come from 20% of causes.
        In retail, a small fraction of customers typically generates the majority of revenue.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    crev = df.groupby("CustomerID")["TotalRevenue"].sum().sort_values(ascending=False).reset_index()
    crev["CumPct"] = crev["TotalRevenue"].cumsum() / crev["TotalRevenue"].sum() * 100
    crev["CustPct"] = (np.arange(1, len(crev)+1) / len(crev)) * 100
    t_idx = (crev["CumPct"] >= 80).idxmax()
    pct80 = crev.loc[t_idx, "CustPct"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=crev["CustPct"], y=crev["TotalRevenue"],
        marker_color="rgba(23,23,23,0.15)", name="Revenue",
        hovertemplate="Top %{x:.1f}%<br>£%{y:,.0f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=crev["CustPct"], y=crev["CumPct"],
        mode="lines", line=dict(color=C_PRIMARY, width=2), name="Cumulative %",
        hovertemplate="Top %{x:.1f}%<br>%{y:.1f}%<extra></extra>"), secondary_y=True)
    fig.add_hline(y=80, line_dash="dot", line_color="#dc2626", line_width=1,
        annotation_text="80%", annotation_font_color="#dc2626", secondary_y=True)
    fig.add_vline(x=pct80, line_dash="dot", line_color="#d97706", line_width=1,
        annotation_text=f"{pct80:.1f}%", annotation_font_color="#d97706")
    fig.update_layout(**CHART_LAYOUT, height=380, xaxis_title="Customers (%)", yaxis_title="Revenue (£)",
        legend=dict(x=0.55, y=0.3))
    fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="accent-card">
        <h4>Key Finding</h4>
        <p><strong>{pct80:.1f}%</strong> of customers generate <strong>80%</strong> of total revenue — confirming the Pareto Principle.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Revenue Drivers</p>', unsafe_allow_html=True)
    uk_pct = df[df["Country"]=="United Kingdom"]["TotalRevenue"].sum()/total_revenue*100
    drivers = [
        ("Geographic Concentration", f"UK accounts for {uk_pct:.1f}% of revenue — domestic performance is the largest driver."),
        ("Customer Retention", f"Top {pct80:.0f}% of customers generate 80% of revenue. Retention directly impacts the bottom line."),
        ("Seasonal Patterns", "Revenue peaks in Q4 (Oct-Dec). Weekday sales outperform weekends, peak hours 10AM-3PM."),
        ("Product Focus", "Top 10 products drive disproportionate revenue. Strategic inventory focus maximizes returns."),
    ]
    dc = st.columns(2)
    for i, (t, d) in enumerate(drivers):
        with dc[i%2]:
            st.markdown(f'<div class="accent-card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Economic Concepts</p>', unsafe_allow_html=True)
    concepts = [
        ("Market Segmentation", "Dividing customers into Champions, Loyalists, At-Risk, and Hibernating enables differentiated strategies."),
        ("Price Elasticity", "Varying quantities across segments suggest different price sensitivities. High-value customers show lower elasticity."),
        ("Economies of Scale", "The 80/20 split enables concentrated marketing spend on highest-ROI segments."),
        ("Consumer Surplus", "Champions spend above average, indicating untapped pricing power for premium offerings."),
    ]
    for t, d in concepts:
        st.markdown(f'<div class="card"><h3>{t}</h3><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Recommendations</p>', unsafe_allow_html=True)
    recs = [
        ("Tiered Loyalty Program", "4-tier program mirroring segments. Champions get premium benefits."),
        ("Data-Driven Inventory", "Optimize stock using purchase frequency and revenue data."),
        ("International Expansion", "Netherlands, Ireland, Germany show growth potential."),
        ("Automated Re-Engagement", "Email workflows triggered by declining RFM scores."),
        ("Omnichannel Experience", "Unified customer profiles across all touchpoints."),
        ("Dynamic Pricing", "Segment-specific pricing to maximize revenue without margin erosion."),
    ]
    rc = st.columns(3)
    for i, (t, d) in enumerate(recs):
        with rc[i%3]:
            st.markdown(f'<div class="accent-card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
