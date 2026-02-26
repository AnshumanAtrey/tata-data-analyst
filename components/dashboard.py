"""Tab 1: Dashboard Overview."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from utils.config import CHART_LAYOUT, C_PRIMARY, C_SECONDARY, COUNTRY_CODES
from utils.data import fmt_currency, fmt_num


def render(df):
    total_revenue = df["TotalRevenue"].sum()
    total_customers = df["CustomerID"].nunique()
    total_orders = df["InvoiceNo"].nunique()
    total_products = df["StockCode"].nunique()
    avg_order = df.groupby("InvoiceNo")["TotalRevenue"].sum().mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Revenue", fmt_currency(total_revenue))
    c2.metric("Customers", fmt_num(total_customers))
    c3.metric("Orders", fmt_num(total_orders))
    c4.metric("Products", fmt_num(total_products))
    c5.metric("Avg Order", fmt_currency(avg_order))

    st.markdown("")

    # Row 1
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<p class="section-title">Monthly Revenue Trend</p>', unsafe_allow_html=True)
        monthly = df.set_index("InvoiceDate").resample("M")["TotalRevenue"].sum().reset_index()
        monthly.columns = ["Month", "Revenue"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Revenue"],
            mode="lines+markers",
            line=dict(color=C_PRIMARY, width=2, shape="spline"),
            marker=dict(size=5, color=C_PRIMARY),
            fill="tozeroy", fillcolor="rgba(23,23,23,0.04)",
            hovertemplate="<b>%{x|%b %Y}</b><br>£%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(**CHART_LAYOUT, height=340, showlegend=False, yaxis_title="Revenue (£)")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section-title">Top 10 Products</p>', unsafe_allow_html=True)
        top = df.groupby("Description")["TotalRevenue"].sum().nlargest(10).sort_values()
        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker=dict(color=C_PRIMARY, cornerradius=4),
            hovertemplate="<b>%{y}</b><br>£%{x:,.0f}<extra></extra>",
        ))
        layout = {**CHART_LAYOUT, "height": 340, "showlegend": False, "xaxis_title": "Revenue (£)"}
        layout["yaxis"] = {**CHART_LAYOUT.get("yaxis", {}), "tickfont": dict(size=10)}
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    left2, right2 = st.columns([3, 2])

    with left2:
        st.markdown('<p class="section-title">Revenue by Country</p>', unsafe_allow_html=True)
        cr = df.groupby("Country")["TotalRevenue"].sum().reset_index()
        cr["ISO"] = cr["Country"].map(COUNTRY_CODES)
        cr = cr.dropna(subset=["ISO"])
        fig = px.choropleth(cr, locations="ISO", color="TotalRevenue", hover_name="Country",
            color_continuous_scale=["#fafafa", "#d4d4d4", "#737373", "#171717"],
            labels={"TotalRevenue": "Revenue (£)"})
        fig.update_layout(**CHART_LAYOUT, height=380,
            geo=dict(bgcolor="#ffffff", lakecolor="#fafafa", landcolor="#fafafa", showframe=False),
            coloraxis_colorbar=dict(title="Revenue", tickprefix="£"))
        fig.update_geos(projection_type="natural earth", showcoastlines=True, coastlinecolor="#d4d4d4")
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.markdown('<p class="section-title">Sales Patterns</p>', unsafe_allow_html=True)
        hr = df.copy()
        hr["Hour"] = hr["InvoiceDate"].dt.hour
        hourly = hr.groupby("Hour")["TotalRevenue"].sum().reset_index()

        fig = make_subplots(rows=2, cols=1, subplot_titles=("By Hour", "By Day"), vertical_spacing=0.22)
        fig.add_trace(go.Bar(x=hourly["Hour"], y=hourly["TotalRevenue"],
            marker=dict(color=C_PRIMARY, cornerradius=2),
            hovertemplate="%{x}:00<br>£%{y:,.0f}<extra></extra>"), row=1, col=1)

        dy = df.copy()
        dy["DOW"] = dy["InvoiceDate"].dt.dayofweek
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        daily = dy.groupby("DOW")["TotalRevenue"].sum().reset_index()
        daily["Day"] = daily["DOW"].map(lambda x: days[x])
        fig.add_trace(go.Bar(x=daily["Day"], y=daily["TotalRevenue"],
            marker=dict(color=C_SECONDARY, cornerradius=2),
            hovertemplate="%{x}<br>£%{y:,.0f}<extra></extra>"), row=2, col=1)

        fig.update_layout(**CHART_LAYOUT, height=380, showlegend=False)
        fig.update_annotations(font_size=11, font_color="#737373")
        st.plotly_chart(fig, use_container_width=True)
