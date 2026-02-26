"""Tab 3: Customer Value Predictor (Agentic AI Chatbot)."""

import streamlit as st
import pandas as pd
import google.generativeai as genai
from utils.config import CLUSTER_NAMES

def render(model_acc, km_model, lr_model, scaler_obj, cluster_map, rfm_df, raw_df):
    
    # ==========================================
    # CUSTOM CSS FOR FLOATING CHAT INPUT
    # ==========================================
    st.markdown("""
    <style>
        /* Force the chat input to float above the page like a pill */
        div[data-testid="stChatInput"] {
            position: fixed !important;
            bottom: 40px !important; 
            z-index: 1000 !important; 
            width: 70% !important; 
            left: 15% !important; /* Keeps it perfectly centered */
            border-radius: 20px !important;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.15) !important; 
        }
        
        /* Add padding to the bottom of the main page 
           so the floating box doesn't cover up the last message */
        .block-container {
            padding-bottom: 120px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">AI Retail Strategist</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <h3>Agentic AI Chat</h3>
        <p>Chat naturally with our AI Consultant. You can now ask it to:</p>
        <ul>
            <li>Analyze specific customers (e.g., <strong>"Strategy for Customer 13047?"</strong>)</li>
            <li>Find top customers (e.g., <strong>"Top 7 customers in France?"</strong>)</li>
            <li>Forecast products (e.g., <strong>"When do vintage jigsaw blocks peak?"</strong>)</li>
            <li>Find market trends (e.g., <strong>"What sells best in October?"</strong>)</li>
            <li>Search the catalog (e.g., <strong>"What kind of bags do we sell?"</strong>)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ==========================================
    # TOOL 1: CUSTOMER ANALYZER
    # ==========================================
    def analyze_customer(customer_id: str) -> str:
        """Searches the database for a customer ID, runs ML predictions, and returns their RFM and segment data."""
        try:
            cid_search = float(customer_id)
        except ValueError:
            return f"Tell the user '{customer_id}' is not a valid number."

        customer_data = rfm_df[rfm_df['CustomerID'].astype(float) == cid_search]
        
        if customer_data.empty:
            return f"Tell the user that Customer ID {int(cid_search)} was not found."
        
        r_in = customer_data['Recency'].values[0]
        f_in = customer_data['Frequency'].values[0]
        m_in = customer_data['Monetary'].values[0]
        
        pt = scaler_obj.transform([[r_in, f_in, m_in]])
        raw_cl = km_model.predict(pt)[0]
        cl = cluster_map.get(raw_cl, raw_cl)
        hv = lr_model.predict(pt)[0]
        
        seg = CLUSTER_NAMES.get(cl, "Unknown")
        vl = "High Value / Profitable" if hv == 1 else "Low Value / At-Risk"
        
        return f"CUSTOMER DATA: Customer {int(cid_search)}. Recency={r_in} days, Frequency={f_in} orders, Spend=${m_in:.2f}. ML Segment: {seg}. Profitability: {vl}."

    # ==========================================
    # TOOL 2: PRODUCT FORECASTER
    # ==========================================
    def analyze_product(search_term: str) -> str:
        """Searches for a specific product to return its sales, geographic demand, and monthly timeline."""
        term = str(search_term).strip().lower()
        prod_data = raw_df[raw_df['StockCode'].astype(str).str.lower() == term].copy()
        
        if prod_data.empty:
            prod_data = raw_df[raw_df['Description'].astype(str).str.lower().str.contains(term, na=False)].copy()
        
        if prod_data.empty:
            return f"Tell the user no product matching '{search_term}' was found."
        
        total_qty_sold = int(prod_data['Quantity'].sum())
        prod_data['Rev_Calc'] = prod_data['Quantity'] * prod_data['UnitPrice']
        total_revenue = float(prod_data['Rev_Calc'].sum())
        product_name = str(prod_data['Description'].iloc[0]) 
        
        top_countries = prod_data.groupby('Country')['Rev_Calc'].sum().nlargest(3)
        country_trend = ", ".join([f"{c} (${v:.0f})" for c, v in top_countries.items()])
        
        if not pd.api.types.is_datetime64_any_dtype(prod_data['InvoiceDate']):
            prod_data['InvoiceDate'] = pd.to_datetime(prod_data['InvoiceDate'], errors='coerce')
        monthly_qty = prod_data.groupby(prod_data['InvoiceDate'].dt.to_period('M'))['Quantity'].sum()
        time_trend = ", ".join([f"{str(month)}: {qty} units" for month, qty in monthly_qty.items()])
        
        return (f"PRODUCT: '{product_name}'. Units Sold: {total_qty_sold}. Revenue: ${total_revenue:.2f}. "
                f"Top Regions: {country_trend}. Monthly Timeline: {time_trend}.")

    # ==========================================
    # TOOL 3: MARKET TRENDS & SEASONALITY
    # ==========================================
    def get_trending_products(month_number: int = 0, country: str = "", limit: int = 5) -> str:
        """Finds top selling products. Pass month_number=0 or country='' if not asked. Default limit is 5."""
        df_filtered = raw_df.copy()
        
        search_country = str(country).strip().lower()
        if search_country in ["uk", "u.k.", "britain", "great britain"]:
            search_country = "united kingdom"
        elif search_country in ["usa", "us", "u.s.a.", "united states of america"]:
            search_country = "united states"
            
        try:
            m_num = int(month_number)
        except ValueError:
            m_num = 0
            
        try:
            lim = int(limit)
            if lim <= 0 or lim > 20: 
                lim = 5
        except ValueError:
            lim = 5
            
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['InvoiceDate']):
            df_filtered['InvoiceDate'] = pd.to_datetime(df_filtered['InvoiceDate'], errors='coerce')
            
        if m_num > 0 and m_num <= 12:
            df_filtered = df_filtered[df_filtered['InvoiceDate'].dt.month == m_num]
            
        if search_country != "":
            df_filtered = df_filtered[df_filtered['Country'].astype(str).str.lower().str.contains(search_country, na=False)]
            
        if df_filtered.empty:
            return f"No sales data found for month {m_num} or location '{country}'."
            
        df_filtered['Rev_Calc'] = df_filtered['Quantity'] * df_filtered['UnitPrice']
        top_items = df_filtered.groupby('Description')['Rev_Calc'].sum().nlargest(lim)
        
        res_list = []
        for item, rev in top_items.items():
            item_df = df_filtered[df_filtered['Description'] == item]
            top_region = item_df.groupby('Country')['Rev_Calc'].sum().idxmax()
            res_list.append(f"'{item}' (${rev:.0f} rev, Top Region: {top_region})")
            
        res_str = ", ".join(res_list)
        return f"TRENDS (Month: {m_num}, Country: '{search_country}', Limit: {lim}): {res_str}."

    # ==========================================
    # TOOL 4: CATALOG SEARCH
    # ==========================================
    def search_catalog(keyword: str) -> str:
        """Finds available product names in the store if the user asks about a general category (like 'bags' or 'lights')."""
        term = str(keyword).strip().lower()
        matches = raw_df[raw_df['Description'].astype(str).str.lower().str.contains(term, na=False)]['Description'].unique()
        
        if len(matches) == 0:
            return f"Tell the user we don't sell anything related to '{keyword}'."
        
        top_matches = list(matches)[:10]
        return f"Found {len(matches)} products matching '{keyword}'. Here are up to 10 examples: {', '.join(top_matches)}."

    # ==========================================
    # TOOL 5: TOP CUSTOMERS FINDER
    # ==========================================
    def get_top_customers(country: str = "", limit: int = 5) -> str:
        """Finds the highest spending customers overall or in a specific country. Default limit is 5."""
        df_filtered = raw_df.copy()
        
        search_country = str(country).strip().lower()
        if search_country in ["uk", "u.k.", "britain", "great britain"]:
            search_country = "united kingdom"
        elif search_country in ["usa", "us", "u.s.a.", "united states of america"]:
            search_country = "united states"
            
        try:
            lim = int(limit)
            if lim <= 0 or lim > 20: 
                lim = 5
        except ValueError:
            lim = 5
            
        if search_country != "":
            df_filtered = df_filtered[df_filtered['Country'].astype(str).str.lower().str.contains(search_country, na=False)]
            
        if df_filtered.empty:
            return f"No customer data found for location '{country}'."
            
        df_filtered['Rev_Calc'] = df_filtered['Quantity'] * df_filtered['UnitPrice']
        
        df_filtered = df_filtered.dropna(subset=['CustomerID'])
        top_custs = df_filtered.groupby('CustomerID')['Rev_Calc'].sum().nlargest(lim)
        
        if top_custs.empty:
            return f"No valid customer IDs found for location '{country}'."
            
        res_list = []
        for cid, rev in top_custs.items():
            res_list.append(f"Customer {int(cid)} (${rev:.0f} revenue)")
            
        res_str = ", ".join(res_list)
        return f"TOP {lim} CUSTOMERS for Country '{search_country}': {res_str}. You can suggest the user ask to 'analyze' these specific IDs for more ML insights."

    # ==========================================
    # INITIALIZE GEMINI
    # ==========================================
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("⚠️ API Key missing. Please check your `.streamlit/secrets.toml` file.")
        return

    system_instruction = """
    You are an expert AI Business Forecaster for the TATA Online Retail Store.
    You have FIVE tools:
    1. `analyze_customer`: Use for specific customer IDs (segmentation, ML predictions).
    2. `analyze_product`: Use for specific product sales, demand, and timelines.
    3. `get_trending_products`: Use for top items or seasonal trends. 
    4. `search_catalog`: Use when the user asks what kinds of general items we sell.
    5. `get_top_customers`: Use when the user asks for the highest spending customers (e.g., "Top 7 customers in France").
    
    CRITICAL RULES:
    - ONLY ANSWER WHAT THE USER ASKS.
    - Format beautifully using Markdown (bullet points, bold text).
    - If you provide a list of top customers, gently remind the user they can ask you to "analyze" any of those specific IDs for a full AI strategy.
    """

    if "chat_session" not in st.session_state:
        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            tools=[analyze_customer, analyze_product, get_trending_products, search_catalog, get_top_customers],
            system_instruction=system_instruction
        )
        st.session_state.chat_session = model.start_chat(enable_automatic_function_calling=True)
        
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "Hello! I am your AI Strategist. Ask me to analyze a customer, forecast a product, search the catalog, or find top customers and seasonal trends."}
        ]

    # ==========================================
    # BUILD THE CHAT UI
    # ==========================================
    
    # Render existing messages
    for msg in st.session_state.ui_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box (Now floating thanks to the CSS!)
    if prompt := st.chat_input("E.g., What are France's top 7 customers?"):
        
        st.session_state.ui_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing database..."):
                try:
                    response = st.session_state.chat_session.send_message(prompt)
                    final_text = response.text
                    
                    st.markdown(final_text)
                    st.session_state.ui_messages.append({"role": "assistant", "content": final_text})
                
                except Exception as e:
                    fallback_text = f"⚠️ **AI Connection Error:** {str(e)}"
                    st.error(fallback_text)
                    st.session_state.ui_messages.append({"role": "assistant", "content": fallback_text})