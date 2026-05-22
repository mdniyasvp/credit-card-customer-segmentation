# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.pipeline import full_pipeline
from src.clustering import apply_dbscan
from src.feature_engineering import apply_pca


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📌 Navigation")

    st.markdown("""
### Credit Card Customer Segmentation

### Algorithms
- DBSCAN
- PCA
- RobustScaler

### Built Using
- Streamlit
- Scikit-learn
- Python
""")

    st.divider()

    st.markdown("""
### Workflow

CSV Upload

↓

Preprocessing

↓

Feature Engineering

↓

PCA

↓

DBSCAN

↓

Business Insights
""")

# ==========================================
# TITLE
# ==========================================

st.title(
    "💳 Credit Card Customer Segmentation"
)

st.markdown("""
This application performs **behavior-based customer segmentation**
using **unsupervised machine learning**.

Goal:
identify meaningful financial behavior patterns.
""")

# ==========================================
# MODEL SUMMARY
# ==========================================

with st.expander(
    "Model Selection"
):

    st.markdown("""
### Evaluation Summary

| Algorithm | Silhouette | Davies-Bouldin |
|---|---:|---:|
| KMeans | 0.252 | 1.343 |
| Hierarchical | 0.230 | 1.494 |
| DBSCAN | 0.115 | **0.900** |

### Selected Model:
DBSCAN

Reason:
- better anomaly detection
- lower cluster overlap
- more realistic segmentation
""")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ==========================================
# PROCESS
# ==========================================

if uploaded_file:

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:

        st.error(e)

        st.stop()

    st.divider()

    st.subheader(
        "Raw Dataset"
    )

    st.dataframe(
        df.head()
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Features",
        df.shape[1]
    )

    # ======================================
    # PIPELINE
    # ======================================

    engineered_scaled, _, _ = (
        full_pipeline(df)
    )

    st.success(
        "Preprocessing completed."
    )

    # ======================================
    # DBSCAN
    # ======================================

    model, labels = (
        apply_dbscan(
            engineered_scaled,
            eps=2.0,
            min_samples=10
        )
    )

    result_df = df.copy()

    result_df[
        "CLUSTER"
    ] = labels

    st.divider()

    st.subheader(
        "Clustered Customers"
    )

    st.dataframe(
        result_df.head()
    )

    # ======================================
    # DISTRIBUTION
    # ======================================

    st.subheader(
        "Cluster Distribution"
    )

    counts = (
        result_df[
            "CLUSTER"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        x=counts.index,
        y=counts.values,
        ax=ax
    )

    ax.set_xlabel(
        "Cluster"
    )

    ax.set_ylabel(
        "Customers"
    )

    st.pyplot(
        fig
    )

    # ======================================
    # PERCENTAGES
    # ======================================

    st.subheader(
        "Cluster Percentages"
    )

    percentages = (
        counts
        /
        counts.sum()
        *
        100
    ).round(2)

    st.dataframe(
        pd.DataFrame({

            "Cluster":
            percentages.index,

            "Percentage":
            percentages.values
        })
    )

    # ======================================
    # PCA
    # ======================================

    st.subheader(
        "Customer Segments (PCA Projection)"
    )

    pca_df, pca = apply_pca(
        engineered_scaled
    )

    variance = (
        pca.explained_variance_ratio_
        .sum()
        *
        100
    )

    pca_df[
        "Cluster"
    ] = labels

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="Set2",
        s=35,
        alpha=0.7
    )

    ax.set_title(

        f"PCA Projection "

        f"(Variance "

        f"{variance:.2f}%)"

    )

    st.pyplot(
        fig
    )

    # ======================================
    # INTERPRETATION
    # ======================================

    st.subheader(
        "Business Interpretation"
    )

    st.info("""

Cluster 0:
Regular customers with balanced behavior.

Cluster -1:
Anomalous customers detected by DBSCAN.

Small clusters:
Specialized customer groups with unique patterns.

DBSCAN naturally detects
behavioral irregularities.
""")

    # ======================================
    # DOWNLOAD
    # ======================================

    st.download_button(

        "Download Results",

        result_df.to_csv(
            index=False
        ),

        file_name=
        "clustered_customers.csv"
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Built using Streamlit • Scikit-learn • Python"
)