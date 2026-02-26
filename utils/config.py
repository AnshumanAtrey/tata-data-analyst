"""Design system constants and configuration."""

# ── KPI Icons (Lucide SVGs) ──
KPI_ICONS = {
    "revenue": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 7c0-5.333-8-5.333-8 0"/><path d="M10 7v14"/><path d="M6 21h12"/><path d="M6 13h10"/></svg>',
    "orders": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="8" cy="21" r="1"/><circle cx="19" cy="21" r="1"/><path d="M2.05 2.05h2l2.66 12.42a2 2 0 0 0 2 1.58h9.78a2 2 0 0 0 1.95-1.57l1.65-7.43H5.12"/></svg>',
    "customers": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "products": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m7.5 4.27 9 5.15"/><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/><path d="m3.3 7 8.7 5 8.7-5"/><path d="M12 22V12"/></svg>',
    "avg_order": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
}

KPI_COLORS = {
    "revenue": ("#3b82f6", "#eff6ff"),      # blue
    "orders": ("#16a34a", "#f0fdf4"),        # green
    "customers": ("#8b5cf6", "#f5f3ff"),     # purple
    "products": ("#f59e0b", "#fffbeb"),      # amber
    "avg_order": ("#0d9488", "#f0fdfa"),     # teal
}

# ── Chart Layout ──
CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter, -apple-system, sans-serif", color="#525252", size=11),
    title_text="",
    title_font=dict(size=13, color="#171717", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#f5f5f5", zerolinecolor="#e5e7eb", linecolor="#e5e7eb"),
    yaxis=dict(gridcolor="#f5f5f5", zerolinecolor="#e5e7eb", linecolor="#e5e7eb"),
    colorway=["#171717", "#525252", "#737373", "#a3a3a3", "#d4d4d4"],
    margin=dict(l=40, r=24, t=44, b=40),
    hoverlabel=dict(bgcolor="#ffffff", font_size=12, bordercolor="#e5e7eb"),
)

# ── Colors ──
C_PRIMARY = "#171717"
C_SECONDARY = "#525252"
C_MUTED = "#a3a3a3"
C_BORDER = "#e5e7eb"

# ── Cluster Config ──
CLUSTER_NAMES = {0: "Champions", 1: "Potential Loyalists", 2: "At-Risk", 3: "Hibernating"}
CLUSTER_ACCENT = {0: "#16a34a", 1: "#2563eb", 2: "#d97706", 3: "#dc2626"}

CLUSTER_DESCRIPTIONS = {
    0: "High-frequency, high-spend customers who purchased recently. Your most valuable customers and brand advocates.",
    1: "Moderate frequency and spend with recent activity. Strong potential to become Champions with right engagement.",
    2: "Previously valuable customers whose activity is declining. Immediate re-engagement strategies are critical.",
    3: "Low activity across all dimensions. Dormant customers needing significant incentives to reactivate.",
}

CLUSTER_RECOMMENDATIONS = {
    0: ["Exclusive VIP loyalty rewards", "Early access to new launches", "Personalized thank-you comms", "Referral program incentives"],
    1: ["Upsell & cross-sell products", "Tiered loyalty enrollment", "Personalized recommendations", "Free shipping on next orders"],
    2: ["Win-back email campaign", "Disengagement survey", "Limited-time comeback offers", "Highlight new products"],
    3: ["Aggressive reactivation discount", "Brand re-engagement campaign", "Low-cost trial offers", "Reduce marketing spend if unresponsive"],
}

# ── Country Codes ──
COUNTRY_CODES = {
    "United Kingdom": "GBR", "France": "FRA", "Germany": "DEU", "Spain": "ESP",
    "Belgium": "BEL", "Switzerland": "CHE", "Portugal": "PRT", "Italy": "ITA",
    "Finland": "FIN", "Austria": "AUT", "Norway": "NOR", "Netherlands": "NLD",
    "Australia": "AUS", "Sweden": "SWE", "Channel Islands": "GBR", "Denmark": "DNK",
    "Japan": "JPN", "Poland": "POL", "Singapore": "SGP", "Iceland": "ISL",
    "Israel": "ISR", "Canada": "CAN", "Greece": "GRC", "Cyprus": "CYP",
    "Czech Republic": "CZE", "Lithuania": "LTU", "United Arab Emirates": "ARE",
    "USA": "USA", "Lebanon": "LBN", "Malta": "MLT", "Bahrain": "BHR",
    "RSA": "ZAF", "Saudi Arabia": "SAU", "Brazil": "BRA", "EIRE": "IRL",
}
