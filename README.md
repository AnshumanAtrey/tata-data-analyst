# TATA Online Retail Store — Revenue Drivers Analysis

> Business Studies Mini Project | Group 11 | ISU

A data-driven analysis of TATA's online retail store to identify what drives revenue, segment customers into actionable groups, and predict customer value using machine learning.

**Live App:** [Streamlit Dashboard](https://tata-analyst.streamlit.app)

---

## What This Project Does

1. **Cleans** 541K+ raw transaction records into usable data
2. **Explores** revenue trends, top products, country-wise sales, and time patterns
3. **Segments** 4,300+ customers into 4 groups using RFM analysis + K-Means clustering
4. **Predicts** whether a customer is high-value or low-value using Logistic Regression
5. **Deploys** an interactive dashboard where you can explore data and run predictions live

---

## Project Structure

```
tata-retail-analysis/
│
├── app.py                        # Main entry point — loads data, renders sidebar/header, connects tabs
│
├── components/                   # Each tab is a separate module
│   ├── dashboard.py              # Tab 1: KPI cards, monthly trend, top products, country map, sales patterns
│   ├── segmentation.py           # Tab 2: RFM distributions, 3D cluster plot, segment profiles
│   ├── predictor.py              # Tab 3: Sliders for input, runs ML models, shows prediction + recommendations
│   ├── insights.py               # Tab 4: Pareto chart, revenue drivers, economic concepts, recommendations
│   └── about.py                  # Tab 5: Team info, tech stack, dataset source
│
├── utils/                        # Shared logic used across components
│   ├── config.py                 # All constants — chart theme, colors, cluster names, country codes
│   ├── data.py                   # Data loading, cleaning, RFM computation, model training, formatters
│   └── styles.py                 # Full CSS for the app — cards, sidebar, tabs, metrics, typography
│
├── assets/
│   └── tata-logo.png             # TATA logo used in sidebar and page header
│
├── dataset/
│   ├── Online Retail Data Set.csv    # Main dataset (541K rows, 8 columns)
│   └── Online Retail Data Set.xlsx   # Same data in Excel format
│
├── TATA_Retail_Analysis.ipynb    # Google Colab notebook with full analysis
├── requirements.txt              # Python dependencies
│
├── .streamlit/
│   └── config.toml               # Forces light theme + hides usage stats
│
└── resources/                    # Assignment reference files
    ├── Business Studies EM Groups.xlsx
    ├── Business Studies Mini Projects EM.xlsx
    └── mail.txt
```

---

## Dataset

| Field | Description |
|-------|-------------|
| **InvoiceNo** | Unique ID per transaction. Starts with "C" if cancelled |
| **StockCode** | Product code |
| **Description** | Product name |
| **Quantity** | Units purchased (negative = return) |
| **InvoiceDate** | Date and time of purchase |
| **UnitPrice** | Price per unit in GBP (£) |
| **CustomerID** | Unique customer identifier |
| **Country** | Customer's country |

**Source:** [Kaggle — TATA Online Retail Dataset](https://www.kaggle.com/datasets/ishanshrivastava28/tata-online-retail-dataset)

**Original Source:** [The Forage — TATA Data Visualisation: Empowering Business with Effective Insights](https://www.theforage.com/modules/MyXvBcppsW2FkNYCX/ifGZCL6vAeE9mzxt3)

**Context:** This dataset is part of TATA's virtual experience program focused on helping executives make data-driven decisions for expansion strategy. The task requires ensuring data quality, cleaning bad data, and creating visualisations that support effective decision making.

**Format:** CSV | **Period:** December 2010 — December 2011 | **Records:** 541,909 | **Customers:** 4,372 | **Countries:** 38

---

## Data Cleaning

Raw retail data is messy. We clean it using **Pandas** — a Python library for data manipulation. Here's every step, why we do it, and the exact code:

### Step 1: Load the data

```python
df = pd.read_csv("dataset/Online Retail Data Set.csv", encoding="ISO-8859-1")
```

`encoding="ISO-8859-1"` is needed because product descriptions contain special characters (accents, symbols) that the default UTF-8 encoding can't read.

### Step 2: Remove cancelled orders

```python
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
```

**Why:** InvoiceNo starting with "C" means the order was cancelled. These aren't real sales — including them would inflate order counts and then deflate revenue (since cancelled orders have negative quantities). The `~` means "not" — so we keep only rows that don't start with C.

### Step 3: Remove negative/zero Quantity and UnitPrice

```python
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
```

**Why:** Negative quantities are product returns. Zero-price items are free samples or internal transfers. Neither represents actual revenue-generating transactions. Including them would distort revenue calculations and customer spending patterns.

### Step 4: Drop rows with missing CustomerID

```python
df = df.dropna(subset=["CustomerID"])
```

**Why:** 135,080 rows (25% of data) have no CustomerID. Without a CustomerID, we can't track which customer made the purchase — which means we can't compute RFM scores or segment them. These rows are useless for our analysis.

### Step 5: Parse dates

```python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True)
```

**Why:** Dates are stored as text strings like "01-12-2010 08:26". We convert them to proper datetime objects so we can extract month, hour, day-of-week, and calculate recency (days since last purchase). `dayfirst=True` because the format is DD-MM-YYYY (British format).

### Step 6: Create TotalRevenue column

```python
df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
```

**Why:** The dataset only has quantity and unit price separately. To analyze revenue, we need the actual money earned per line item. This is our primary metric for everything — monthly trends, top products, customer value.

**Result:** 541,909 rows → **~397,000 clean rows** ready for analysis.

---

## Algorithms Used

### 1. RFM Analysis

RFM is a marketing technique (not an ML algorithm) that scores every customer on three behaviour metrics. It's widely used in industry to understand customer value.

| Metric | What it measures | How we calculate it |
|--------|-----------------|-------------------|
| **Recency** | How recently they bought | Days between their last purchase and the dataset's latest date |
| **Frequency** | How often they buy | Count of unique invoices per customer |
| **Monetary** | How much they spend | Sum of TotalRevenue per customer |

**Code:**

```python
snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)  # day after last transaction

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalRevenue", "sum"),
).reset_index()
```

**Why this works:** `groupby("CustomerID")` groups all transactions by customer, then `.agg()` calculates each metric in one pass. The result is one row per customer with 3 numbers — this becomes the input for our ML models.

**Why we use `snapshot` date:** Recency needs a reference point. We use "one day after the latest transaction in the dataset" so that the most recent customer gets Recency = 1 (not 0).

### 2. K-Means Clustering

**What it does:** Groups customers into K clusters where customers within a cluster behave similarly.

**How it works:**
1. Pick K random center points (centroids)
2. Assign every customer to the nearest centroid (by Euclidean distance)
3. Move each centroid to the average position of its assigned customers
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Why we standardize first:**

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
```

Monetary values range from £3 to £280,000. Recency ranges from 1 to 373 days. Without standardizing, K-Means would only care about Monetary (because it has the largest numbers) and ignore Recency and Frequency. `StandardScaler` converts each feature to mean=0, std=1 — giving all three equal weight.

**Training:**

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["Cluster"] = km.fit_predict(X)
```

- `n_clusters=4` — we chose 4 groups (determined by the elbow method in our Colab notebook)
- `random_state=42` — ensures same results every time we run it
- `n_init=10` — runs the algorithm 10 times with different starting points and picks the best result

**We then sort clusters by value:**

```python
order = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index.tolist()
mapping = {old: new for new, old in enumerate(order)}
rfm["Cluster"] = rfm["Cluster"].map(mapping)
```

**Why:** K-Means assigns random cluster numbers (0-3). We remap them so Cluster 0 always = highest spenders. This makes the output meaningful and consistent.

**Output segments:**

| Cluster | Name | Profile |
|---------|------|---------|
| 0 | Champions | Recent, frequent, high-spend |
| 1 | Potential Loyalists | Moderate activity, room to grow |
| 2 | At-Risk | Declining activity, need re-engagement |
| 3 | Hibernating | Dormant, low across all metrics |

### 3. Logistic Regression

**What it does:** Predicts a yes/no outcome — is this customer high-value or not?

**How it works:**
1. Takes input features (Recency, Frequency, Monetary)
2. Multiplies each by a learned weight and sums them up
3. Passes the sum through a sigmoid function → output between 0 and 1
4. If output > 0.5 → High Value, otherwise → Low Value

**Creating the target:**

```python
median_m = rfm["Monetary"].median()
rfm["HighValue"] = (rfm["Monetary"] > median_m).astype(int)
```

**Why median as threshold:** The median splits customers into two equal halves. Customers above median spend = high-value (1), below = low-value (0). This gives a balanced dataset (50/50 split), which helps the model learn fairly from both classes.

**Training:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
```

- `test_size=0.2` — 80% of data for training, 20% held back for testing. This tells us how well the model works on data it hasn't seen before.
- `max_iter=1000` — gives the algorithm enough iterations to converge (find the best weights)
- Achieves **~90% accuracy** on test data

**Why Logistic Regression:** It's interpretable (you can see which features matter most), fast to train, and works well for binary classification. For a business audience, being able to explain *why* a prediction was made is more valuable than a black-box model with slightly higher accuracy.

### 4. Pareto Analysis (80/20 Rule)

**Code:**

```python
customer_rev = df.groupby("CustomerID")["TotalRevenue"].sum().sort_values(ascending=False)
customer_rev["CumulativePct"] = customer_rev.cumsum() / customer_rev.sum() * 100
```

**What it does:** Sorts all customers by revenue (highest first), then calculates running total as a percentage. We find the point where cumulative revenue hits 80%.

**Why it matters:** If ~20% of customers generate ~80% of revenue (which they do), then losing even a few top customers has massive impact. This tells the business exactly where to focus retention budgets — on the small group that drives most of the money.

---

## Libraries Used

| Library | Version | What we use it for |
|---------|---------|-------------------|
| **Streamlit** | 1.54+ | Web app framework — renders the dashboard, handles tabs, sliders, metrics |
| **Pandas** | 2.0+ | Data loading (read_csv), cleaning (filtering, dropna), groupby aggregations for RFM |
| **NumPy** | 1.24+ | Numerical operations — cumulative sums for Pareto, array indexing |
| **Plotly** | 5.0+ | All charts — line, bar, choropleth map, 3D scatter, pie, gauge. Interactive with hover |
| **Scikit-learn** | 1.3+ | `StandardScaler` for feature normalization, `KMeans` for clustering, `LogisticRegression` for prediction, `train_test_split` for evaluation |

---

## How Caching Works

The app uses Streamlit's `@st.cache_data` decorator on all heavy functions — data loading, RFM computation, and model training.

**What this means:**
- **First visit:** The app reads the 46MB CSV, cleans it, computes RFM for 4,300+ customers, and trains both ML models. This takes ~10-15 seconds.
- **Every visit after that:** Streamlit serves the cached result instantly. No re-computation.
- **Cache resets** when the server restarts (e.g., after deploying new code or after the app has been idle for a while on Streamlit Cloud).

In practice: open the app once before presenting. After that, anyone who visits gets instant load times.

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/AnshumanAtrey/tata-data-analyst.git
cd tata-data-analyst

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Team — Group 11

| Name | Role |
|------|------|
| Payal Kunwar | Research & Analysis |
| Gaurav Kulkarni | Data Processing |
| Anshuman Atrey | ML & Development |
| Abdullah Haque | Business Strategy |
| Shlok Vijay Kadam | Visualization & Report |

---

## Key Findings

- **UK dominates:** ~82% of total revenue comes from the United Kingdom
- **Pareto confirmed:** ~20% of customers generate ~80% of revenue
- **Seasonal peak:** Revenue spikes in Oct-Nov (holiday season)
- **Weekday bias:** Most sales happen Monday-Friday, 10 AM - 3 PM
- **Champions matter:** A small cluster of high-frequency, high-spend customers drives the business
