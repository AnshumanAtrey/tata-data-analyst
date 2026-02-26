"""Tab 5: About."""

import streamlit as st


def render():
    st.markdown('<p class="section-title">About</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>Project Overview</h3>
        <p>Developed for <strong>Business Studies</strong> to analyze revenue drivers for the
        <strong>TATA Online Retail Store</strong>. Uses real-world transactional data with
        machine learning techniques to uncover actionable business insights — including
        RFM analysis, K-Means clustering, Logistic Regression, and Pareto analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Team — Group 11</p>', unsafe_allow_html=True)
    team = [
        ("Payal Kunwar", "Research"),
        ("Gaurav Kulkarni", "Data"),
        ("Anshuman Atrey", "ML & Dev"),
        ("Abdullah Haque", "Strategy"),
        ("Shlok Vijay Kadam", "Visualization"),
    ]
    tc = st.columns(5)
    for i, (n, r) in enumerate(team):
        with tc[i]:
            st.markdown(f"""
            <div class="team-card">
                <div class="avatar">{n[0]}</div>
                <h4>{n}</h4>
                <p>{r}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Stack</p>', unsafe_allow_html=True)
    techs = [("Python", "Core language"), ("Streamlit", "Web framework"),
             ("Plotly", "Visualizations"), ("Scikit-learn", "ML models")]
    tec = st.columns(4)
    for i, (n, d) in enumerate(techs):
        with tec[i]:
            st.markdown(f'<div class="team-card"><h4>{n}</h4><p>{d}</p></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-title">Dataset</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <h3>Online Retail Data Set</h3>
        <p><strong>Source:</strong> Kaggle / UCI ML Repository<br>
        <strong>Link:</strong> <a href="https://www.kaggle.com/datasets/ishanshrivastava28/tata-online-retail-dataset"
            target="_blank" style="color:#171717; text-decoration:underline;">
            kaggle.com/datasets/ishanshrivastava28/tata-online-retail-dataset</a><br>
        <strong>Provenance:</strong> <a href="https://www.theforage.com/modules/MyXvBcppsW2FkNYCX/ifGZCL6vAeE9mzxt3"
            target="_blank" style="color:#171717; text-decoration:underline;">
            The Forage — TATA Data Visualisation Program</a><br><br>
        Part of TATA's virtual experience program: <em>Empowering Business with Effective Insights</em>.
        Transnational dataset containing all transactions between Dec 2010 and Dec 2011 for a UK-based
        online retail company. The goal is to ensure data quality and create visualisations that help
        executives with effective decision making and expansion strategy.<br><br>
        <strong>541,909</strong> records &middot; <strong>8</strong> columns &middot;
        <strong>4,372</strong> customers &middot; <strong>38</strong> countries &middot; Format: CSV</p>
    </div>
    """, unsafe_allow_html=True)
