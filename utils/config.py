"""Design system constants and configuration."""

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
