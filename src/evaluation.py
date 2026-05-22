# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# ==========================================
# SILHOUETTE SCORE
# ==========================================

def calculate_silhouette(
    data,
    labels,
    remove_noise=False
):

    labels = np.array(labels)

    # Remove DBSCAN noise if needed
    if remove_noise:
        mask = labels != -1

        data = data[mask]
        labels = labels[mask]

    # Check cluster count
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        print(
            "Silhouette Score requires "
            "at least 2 clusters."
        )
        return None

    score = silhouette_score(
        data,
        labels
    )

    return round(score, 4)


# ==========================================
# DAVIES-BOULDIN SCORE
# ==========================================

def calculate_davies_bouldin(
    data,
    labels,
    remove_noise=False
):

    labels = np.array(labels)

    # Remove DBSCAN noise if needed
    if remove_noise:

        mask = labels != -1

        data = data[mask]
        labels = labels[mask]

    # Check cluster count
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(
            "Davies-Bouldin Score requires "
            "at least 2 clusters."
        )
        return None

    score = davies_bouldin_score(
        data,
        labels
    )

    return round(score, 4)

# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

def cluster_distribution(labels):

    labels = pd.Series(labels)
    counts = (
        labels.value_counts().sort_index()
    )

    percentages = round(
        (counts / len(labels)) * 100,2
    )

    distribution_df = pd.DataFrame({
        'Count': counts,
        'Percentage': percentages
    })

    return distribution_df