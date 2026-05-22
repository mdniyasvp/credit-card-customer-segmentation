# ==========================================
# IMPORT LIBRARIES
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

def plot_cluster_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x='CLUSTER',
        data=df
    )

    plt.title(
        "Customer Cluster Distribution"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Customer Count")
    plt.savefig(
        "../outputs/clusters/cluster_distribution.png"
    )

    plt.show()

# ==========================================
# PCA CLUSTER VISUALIZATION
# ==========================================

def plot_pca_clusters(
    pca_df,
    labels
):

    plot_df = pca_df.copy()
    plot_df['Cluster'] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='Cluster',
        palette='Set2'
    )
    plt.title(
        "Customer Segments (PCA Projection)"
    )
    plt.savefig(
        "../outputs/clusters/pca_clusters.png"
    )
    plt.show()

# ==========================================
# PURCHASES BY CLUSTER
# ==========================================

def plot_purchases_by_cluster(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x='CLUSTER',
        y='PURCHASES'
    )

    plt.title(
        "Purchases by Cluster"
    )

    plt.savefig(
        "../outputs/clusters/purchases_by_cluster.png"
    )

    plt.show()

# ==========================================
# CASH ADVANCE BY CLUSTER
# ==========================================

def plot_cash_advance_by_cluster(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x='CLUSTER',
        y='CASH_ADVANCE'
    )
    plt.title(
        "Cash Advance by Cluster"
    )

    plt.savefig(
        "../outputs/clusters/cash_advance_by_cluster.png"
    )

    plt.show()

# ==========================================
# PAYMENTS BY CLUSTER
# ==========================================

def plot_payments_by_cluster(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x='CLUSTER',
        y='PAYMENTS'
    )

    plt.title(
        "Payments by Cluster"
    )
    plt.savefig(
        "../outputs/clusters/payments_by_cluster.png"
    )
    plt.show()