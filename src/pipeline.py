# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
from sklearn.preprocessing import RobustScaler
from src.preprocessing import (
    preprocess_data
)

from src.feature_engineering import (
    engineer_features
)

# ==========================================
# FULL PIPELINE
# ==========================================

def full_pipeline(df):

    # ==========================================
    # PREPROCESSING
    # ==========================================

    processed_df, preprocessing_scaler = (
        preprocess_data(df)
    )

    # ==========================================
    # FEATURE ENGINEERING
    # ==========================================

    engineered_df = engineer_features(
        processed_df.copy()
    )

    # ==========================================
    # FINAL SCALING
    # ==========================================

    final_scaler = RobustScaler()
    engineered_scaled = pd.DataFrame(
        final_scaler.fit_transform(
            engineered_df
        ),

        columns=engineered_df.columns,
        index=engineered_df.index
    )
    return (
        engineered_scaled,
        preprocessing_scaler,
        final_scaler
    )