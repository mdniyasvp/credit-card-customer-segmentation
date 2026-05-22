# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN
)

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram
)

import matplotlib.pyplot as plt

# ==========================================
# ELBOW METHOD
# ==========================================

def elbow_method(data,max_clusters=10):
    inertia_values = []
    cluster_range = range(1,max_clusters + 1
    )

    for k in cluster_range:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(data)

        inertia_values.append(model.inertia_
        )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.plot(
        cluster_range,
        inertia_values,
        marker='o'
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    plt.savefig(
        "../outputs/clusters/elbow_method.png"
    )

    plt.show()

# ==========================================
# KMEANS CLUSTERING
# ==========================================

def apply_kmeans(data,n_clusters=4):

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(data)
    return kmeans, labels

# ==========================================
# HIERARCHICAL CLUSTERING
# ==========================================

def apply_hierarchical(data,n_clusters=4):

    hierarchical = (
        AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
    )

    labels = hierarchical.fit_predict(
        data
    )

    return hierarchical, labels

# ==========================================
# DENDROGRAM
# ==========================================

def plot_dendrogram(data):

    # Sample data for performance
    if len(data) > 1000:
        sample_data = (
            data.sample(
                n=1000,
                random_state=42
            )
        )

    else:

        sample_data = data

    linked = linkage(
        sample_data,
        method='ward'
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        linked,
        truncate_mode='level',
        p=5
    )

    plt.title("Hierarchical Dendrogram")
    plt.xlabel("Customers")
    plt.ylabel("Distance")

    plt.savefig(
        "../outputs/clusters/dendrogram.png"
    )

    plt.show()

# ==========================================
# DBSCAN
# ==========================================

def apply_dbscan(data,
                 eps=1.5,
                 min_samples=10):

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean'
    )

    labels = dbscan.fit_predict(data)
    return dbscan, labels