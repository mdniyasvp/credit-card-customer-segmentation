# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# ==========================================
# IMPORT CUSTOM MODULES
# ==========================================

from src.preprocessing import (
    preprocess_data
)

from src.feature_engineering import (
    engineer_features,
    apply_pca,
    explained_variance
)

from src.clustering import (
    elbow_method,
    apply_kmeans,
    apply_hierarchical,
    apply_dbscan
)

from src.evaluation import (
    calculate_silhouette,
    calculate_davies_bouldin,
    cluster_distribution
)

from src.visualization import (
    plot_cluster_distribution,
    plot_pca_clusters,
    plot_purchases_by_cluster,
    plot_cash_advance_by_cluster,
    plot_payments_by_cluster
)

from src.utils import (
    print_section
)

# ==========================================
# LOAD DATA
# ==========================================

print_section("LOADING DATA")
data_path = "data/raw/CC GENERAL.csv"
df = pd.read_csv(data_path)
print("Dataset loaded successfully!")
print(df.shape)

# ==========================================
# PREPROCESSING
# ==========================================

print_section("PREPROCESSING")
processed_df, preprocessing_scaler = (
    preprocess_data(df)
)

print("Preprocessing completed successfully!")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print_section("FEATURE ENGINEERING")
engineered_df = engineer_features(
    processed_df.copy()
)

print("Feature engineering completed!")

# ==========================================
# FINAL SCALING
# ==========================================

print_section("FINAL SCALING")
final_scaler = RobustScaler()
engineered_scaled = pd.DataFrame(

    final_scaler.fit_transform(
        engineered_df
    ),
    columns=engineered_df.columns,
    index=engineered_df.index
)

print("Final scaling completed!")

# ==========================================
# PCA
# ==========================================

print_section("PCA")
pca_df, pca_model = apply_pca(
    engineered_scaled,
    n_components=2
)

variance = explained_variance(
    pca_model
)

print("Explained Variance Ratio:")
print(variance)

# ==========================================
# ELBOW METHOD
# ==========================================

print_section("ELBOW METHOD")
elbow_method(
    engineered_scaled,
    max_clusters=10
)

print("Elbow method completed!")

# ==========================================
# KMEANS
# ==========================================

print_section("KMEANS")
kmeans_model, kmeans_labels = (
    apply_kmeans(
        engineered_scaled,
        n_clusters=4
    )
)

print("KMeans clustering completed!")

# ==========================================
# HIERARCHICAL
# ==========================================

print_section("HIERARCHICAL")
hierarchical_model, hierarchical_labels = (
    apply_hierarchical(
        engineered_scaled,
        n_clusters=4
    )
)

print("Hierarchical clustering completed!")

# ==========================================
# DBSCAN
# ==========================================

print_section("DBSCAN")
final_dbscan_model, final_dbscan_labels = (

    apply_dbscan(
        engineered_scaled,
        eps=2.0,
        min_samples=10
    )
)

print("DBSCAN clustering completed!")

# ==========================================
# EVALUATION
# ==========================================

print_section("CLUSTER EVALUATION")
dbscan_silhouette = (
    calculate_silhouette(
        engineered_scaled,
        final_dbscan_labels,
        remove_noise=True
    )
)

dbscan_db = (
    calculate_davies_bouldin(
        engineered_scaled,
        final_dbscan_labels,
        remove_noise=True
    )
)

print(
    f"Silhouette Score: "
    f"{dbscan_silhouette}"
)

print(
    f"Davies-Bouldin Score: "
    f"{dbscan_db}"
)

# ==========================================
# CLUSTER DISTRIBUTION
# ==========================================

distribution = cluster_distribution(
    final_dbscan_labels
)

print(distribution)

# ==========================================
# FINAL CLUSTERED DATAFRAME
# ==========================================

final_clustered_df = (
    engineered_df.copy()
)

final_clustered_df['CLUSTER'] = (
    final_dbscan_labels
)

# ==========================================
# VISUALIZATIONS
# ==========================================

print_section("VISUALIZATIONS")
plot_cluster_distribution(
    final_clustered_df
)

plot_pca_clusters(
    pca_df,
    final_dbscan_labels
)

plot_purchases_by_cluster(
    final_clustered_df
)

plot_cash_advance_by_cluster(
    final_clustered_df
)

plot_payments_by_cluster(
    final_clustered_df
)

print("Visualizations completed!")

# ==========================================
# SAVE CLUSTERED DATA
# ==========================================

print_section("SAVING OUTPUTS")
final_clustered_df.to_csv("data/processed/clustered_customers.csv",index=False)

print("Clustered dataset saved!")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    final_dbscan_model,
    "models/dbscan_model.pkl"
)

print("DBSCAN model saved!")

# ==========================================
# FINAL MESSAGE
# ==========================================

print_section("PROJECT COMPLETED")

print(
    "Customer segmentation pipeline "
    "executed successfully!"
)