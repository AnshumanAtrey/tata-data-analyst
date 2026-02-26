"""Data loading, cleaning, and model training."""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@st.cache_data(show_spinner=False)
def load_and_clean_data():
    df = pd.read_csv(
        "dataset/Online Retail Data Set.csv",
        encoding="ISO-8859-1",
        dtype={"CustomerID": str, "InvoiceNo": str},
    )
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df = df.dropna(subset=["CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed", dayfirst=True)
    df["TotalRevenue"] = df["Quantity"] * df["UnitPrice"]
    return df


@st.cache_data(show_spinner=False)
def compute_rfm(df):
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    return df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalRevenue", "sum"),
    ).reset_index()


@st.cache_data(show_spinner=False)
def train_models(_rfm):
    rfm = _rfm.copy()
    feats = rfm[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(X)

    order = rfm.groupby("Cluster")["Monetary"].mean().sort_values(ascending=False).index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    rfm["Cluster"] = rfm["Cluster"].map(mapping)

    median_m = rfm["Monetary"].median()
    rfm["HighValue"] = (rfm["Monetary"] > median_m).astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, rfm["HighValue"].values, test_size=0.2, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(Xtr, ytr)
    acc = accuracy_score(yte, lr.predict(Xte))

    return rfm, km, lr, scaler, mapping, acc, median_m


def fmt_currency(v):
    return f"£{v:,.0f}"


def fmt_num(v):
    return f"{v:,}"
