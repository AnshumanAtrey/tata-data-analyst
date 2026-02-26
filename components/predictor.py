"""Tab 3: Customer Value Predictor."""

import streamlit as st
import plotly.graph_objects as go
from utils.config import CHART_LAYOUT, CLUSTER_NAMES, CLUSTER_ACCENT, CLUSTER_RECOMMENDATIONS


def render(model_acc, km_model, lr_model, scaler_obj, cluster_map):
    st.markdown('<p class="section-title">Customer Value Predictor</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <h3>How It Works</h3>
        <p>Enter a customer's RFM profile. Uses <strong>K-Means</strong> for segment prediction
        and <strong>Logistic Regression</strong> ({model_acc:.1%} accuracy) for value classification.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    inp, res = st.columns([2, 3])

    with inp:
        st.markdown('<p class="section-title">Input Profile</p>', unsafe_allow_html=True)
        r_in = st.slider("Recency (days)", 0, 365, 30)
        f_in = st.slider("Frequency (purchases)", 1, 500, 10)
        m_in = st.slider("Monetary (total $)", 0, 50000, 1500, step=50)
        btn = st.button("Predict", use_container_width=True, type="primary")

    with res:
        if btn:
            pt = scaler_obj.transform([[r_in, f_in, m_in]])
            raw_cl = km_model.predict(pt)[0]
            cl = cluster_map.get(raw_cl, raw_cl)
            hv = lr_model.predict(pt)[0]
            prob = lr_model.predict_proba(pt)[0]
            conf = max(prob) * 100
            seg = CLUSTER_NAMES.get(cl, "Unknown")
            vl = "High Value" if hv == 1 else "Low Value"
            vc = "pred-green" if hv == 1 else "pred-red"

            st.markdown(f"""
            <div class="prediction-box">
                <h2>Prediction Results</h2>
                <hr style="border-color: #e5e7eb; margin: 12px 0;">
                <p class="pred-label">Customer Segment</p>
                <div class="pred-value pred-blue">{seg}</div>
                <p class="pred-label">Value Classification</p>
                <div class="pred-value {vc}">{vl}</div>
                <p style="margin-top: 12px; color: #a3a3a3; font-size: 0.85rem;">
                    Confidence: <strong style="color:#171717;">{conf:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=conf,
                number=dict(suffix="%", font=dict(size=28, color="#171717")),
                title=dict(text="Confidence", font=dict(size=12, color="#a3a3a3")),
                gauge=dict(axis=dict(range=[0, 100], tickcolor="#d4d4d4"),
                    bar=dict(color="#171717"), bgcolor="#fafafa", bordercolor="#e5e7eb",
                    steps=[dict(range=[0, 50], color="#fef2f2"),
                           dict(range=[50, 75], color="#fefce8"),
                           dict(range=[75, 100], color="#f0fdf4")])))
            fig.update_layout(**CHART_LAYOUT, height=210)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f'<p class="section-title">Recommendations for {seg}</p>', unsafe_allow_html=True)
            recs = CLUSTER_RECOMMENDATIONS.get(cl, [])
            rc = st.columns(2)
            for ri, rec in enumerate(recs):
                with rc[ri % 2]:
                    st.markdown(f'<div class="accent-card"><h4>Strategy {ri+1}</h4><p>{rec}</p></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box" style="min-height: 340px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                <div style="width:48px; height:48px; border-radius:12px; background:#f5f5f5; display:flex; align-items:center; justify-content:center; font-size:1.4rem; margin-bottom:12px;">📊</div>
                <h2 style="color:#a3a3a3; font-weight:500;">Awaiting Input</h2>
                <p style="max-width:340px; text-align:center; color:#a3a3a3; font-size:0.85rem;">
                    Adjust the sliders and click <strong>Predict</strong> to see results.</p>
            </div>
            """, unsafe_allow_html=True)
