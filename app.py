# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
import pandas as pd
from src.pipeline import (
    full_pipeline
)

from src.clustering import (
    apply_dbscan
)

from src.feature_engineering import (
    apply_pca
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title(
    "Navigation"
)

st.sidebar.info("""

Credit Card Customer Segmentation App

Algorithms Used:
- DBSCAN
- PCA
- Robust Scaling

Developed using:
- Streamlit
- Scikit-learn
- Python
""")
# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title(
    "Credit Card Customer Segmentation"
)
st.markdown("""
This application performs customer segmentation using:

- preprocessing
- feature engineering
- PCA
- DBSCAN clustering

The goal is to identify meaningful customer behavior groups from financial transaction patterns.
""")
st.write(
    """
    Upload customer financial data to
    perform behavioral segmentation
    using DBSCAN clustering.
    """
)

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload CSV File", type=['csv']
)

# ==========================================
# PROCESS FILE
# ==========================================

if uploaded_file is not None:
    # Load data
    try:
        df = pd.read_csv(
            uploaded_file
        )

    except Exception as e:
        st.error(
            f"Error loading file: {e}"
        )

        st.stop()
    st.subheader("Raw Dataset")
    st.dataframe(df.head())
    st.write(
        f"Dataset Shape: {df.shape}"
    )
    
    # ==========================================
    # PREPROCESSING PIPELINE
    # ==========================================

    engineered_scaled, _, _ = (
        full_pipeline(df)
    )

    st.success(
        "Preprocessing completed successfully!"
    )

    # ==========================================
    # APPLY DBSCAN
    # ==========================================

    dbscan_model, labels = (
        apply_dbscan(
            engineered_scaled,
            eps=2.0,
            min_samples=10
        )
    )

    # Add clusters
    result_df = df.copy()
    result_df['CLUSTER'] = labels

    # ==========================================
    # SHOW RESULTS
    # ==========================================

    st.subheader(
        "Clustered Customers"
    )

    st.dataframe(
        result_df.head()
    )

    # ==========================================
    # CLUSTER DISTRIBUTION
    # ==========================================

    st.subheader(
        "Cluster Distribution"
    )

    cluster_counts = (
        result_df['CLUSTER'].value_counts().sort_index()
    )

    # ==========================================
    # CLUSTER DISTRIBUTION VISUALIZATION
    # ==========================================

    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(
        figsize=(10, 5)
    )
    sns.barplot(
        x=cluster_counts.index,
        y=cluster_counts.values,
        ax=ax
    )
    ax.set_title(
        "Customer Cluster Distribution"
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

    # ==========================================
    # DOWNLOAD RESULTS
    # ==========================================

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Clustered Data",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

    # ==========================================
    # CLUSTER PERCENTAGES
    # ==========================================

    cluster_percentage = round(
        cluster_counts / cluster_counts.sum() * 100,        2
    )

    percentage_df = pd.DataFrame({
        'Cluster': cluster_percentage.index,
        'Percentage': cluster_percentage.values
    })

    st.subheader(
        "Cluster Percentages"
    )

    st.dataframe(
        percentage_df
    )

    # ==========================================
    # CLUSTER INTERPRETATION
    # ==========================================

    st.subheader(
        "Business Interpretation"
    )

    st.markdown("""

    - **Cluster 0:** Mainstream customers with stable financial behavior.

    - **Cluster -1:** Anomalous or unusual customer behavior detected by DBSCAN.

    - Smaller clusters represent niche customer segments with unique spending or payment patterns.

    DBSCAN naturally identifies:
    - dense customer populations
    - anomalies
    - rare customer behaviors

    rather than forcing equal-sized clusters.
    """)

    # ==========================================
    # PCA VISUALIZATION
    # ==========================================

    pca_df, _ = apply_pca(
        engineered_scaled,
        n_components=2
    )

    pca_df['Cluster'] = labels
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='Set2',
        s=40,
        alpha=0.7,
        ax=ax
    )

    ax.set_title(
        "Customer Segments (PCA Projection)"
    )

    st.pyplot(fig)