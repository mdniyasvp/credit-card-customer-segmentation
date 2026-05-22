# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# ==========================================
# REMOVE CUSTOMER ID
# ==========================================

def remove_id_column(df):
    df = df.drop(
        'CUST_ID',
        axis=1,
        errors='ignore'
    )

    return df

# ==========================================
# HANDLE MISSING VALUES
# ==========================================

def handle_missing_values(df):
    numeric_cols = (
        df.select_dtypes(include=np.number).columns
    )

    for col in numeric_cols:

        df[col] = (
            df[col].fillna(df[col].median())
        )

    return df

# ==========================================
# REMOVE DUPLICATES
# ==========================================

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df

# ==========================================
# LOG TRANSFORMATION
# ==========================================

def log_transform(df, columns):
    for col in columns:
        # Ensure no negative values
        clipped_values = np.clip(
            df[col],
            a_min=0,
            a_max=None
        )

        df[col] = np.log1p(clipped_values
        )
    return df

# ==========================================
# CLIP EXTREME OUTLIERS
# ==========================================

def clip_outliers(df):

    # Only clip monetary/transaction features
    clip_cols = [

        'BALANCE',
        'PURCHASES',
        'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE',
        'PAYMENTS',
        'MINIMUM_PAYMENTS',
        'PURCHASES_TRX',
        'CASH_ADVANCE_TRX',
        'CREDIT_LIMIT'
    ]

    # Keep only available columns
    available_clip_cols = [
        col for col in clip_cols
        if col in df.columns
    ]

    for col in available_clip_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)

        df[col] = df[col].clip(
            lower=lower,
            upper=upper
        )

    return df


# ==========================================
# SCALE FEATURES
# ==========================================

def scale_features(df):

    # Features requiring scaling
    scale_cols = [
        'BALANCE',
        'PURCHASES',
        'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE',
        'CASH_ADVANCE_TRX',
        'PURCHASES_TRX',
        'CREDIT_LIMIT',
        'PAYMENTS',
        'MINIMUM_PAYMENTS'
    ]

    # Behavioral features (not scaled)
    non_scale_cols = [
        'BALANCE_FREQUENCY',
        'PURCHASES_FREQUENCY',
        'ONEOFF_PURCHASES_FREQUENCY',
        'PURCHASES_INSTALLMENTS_FREQUENCY',
        'CASH_ADVANCE_FREQUENCY',
        'PRC_FULL_PAYMENT',
        'TENURE'
    ]

    # Keep only available columns
    available_scale_cols = [
        col for col in scale_cols
        if col in df.columns
    ]

    available_non_scale_cols = [
        col for col in non_scale_cols
        if col in df.columns
    ]
    scaler = RobustScaler()
    # Scale selected columns
    scaled_values = scaler.fit_transform(
        df[available_scale_cols]
    )

    scaled_df = pd.DataFrame(
        scaled_values,
        columns=available_scale_cols,
        index=df.index
    )

    # Add non-scaled columns back
    for col in available_non_scale_cols:
        scaled_df[col] = df[col]

    return scaled_df, scaler

# ==========================================
# COMPLETE PREPROCESSING
# ==========================================

def preprocess_data(df):

    # Remove customer ID
    df = remove_id_column(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Remove duplicates
    df = remove_duplicates(df)

    # Features requiring log transform
    skewed_cols = [

        'BALANCE',
        'PURCHASES',
        'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE',
        'PAYMENTS',
        'MINIMUM_PAYMENTS',
        'PURCHASES_TRX',
        'CASH_ADVANCE_TRX'
    ]

    # Keep only existing columns
    available_skewed_cols = [

        col for col in skewed_cols
        if col in df.columns
    ]

    # Apply log transformation
    df = log_transform(
        df,
        available_skewed_cols
    )

    # Clip remaining outliers
    df = clip_outliers(df)

    # Scale features
    scaled_df, scaler = scale_features(df)

    return scaled_df, scaler