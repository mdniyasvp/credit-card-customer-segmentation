# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# ==========================================
# PAYMENT RATIO
# ==========================================

def add_payment_ratio(df):
    df['PAYMENT_RATIO'] = (
        df['PAYMENTS'] /(np.abs(df['BALANCE']) + 1)
    )

    return df

# ==========================================
# CREDIT UTILIZATION
# ==========================================

def add_credit_utilization(df):
    df['CREDIT_UTILIZATION'] = (
        df['PURCHASES'] /(np.abs(df['CREDIT_LIMIT']) + 1)
    )

    return df

# ==========================================
# CASH ADVANCE RATIO
# ==========================================

def add_cash_advance_ratio(df):
    df['CASH_ADVANCE_RATIO'] = (
        df['CASH_ADVANCE'] /(np.abs(df['PURCHASES']) + 1)
    )

    return df

# ==========================================
# TRANSACTION INTENSITY
# ==========================================

def add_transaction_intensity(df):
    df['TRANSACTION_INTENSITY'] = (
        df['PURCHASES_TRX'] + df['CASH_ADVANCE_TRX']
    )

    return df

# ==========================================
# COMPLETE FEATURE ENGINEERING
# ==========================================

def engineer_features(df):
    df = add_payment_ratio(df)
    df = add_credit_utilization(df)
    df = add_cash_advance_ratio(df)
    df = add_transaction_intensity(df)

    return df

# ==========================================
# APPLY PCA
# ==========================================

def apply_pca(df, n_components=2):

    # Keep only numeric columns
    numeric_df = (
        df.select_dtypes(include=np.number)
    )

    pca = PCA(
        n_components=n_components
    )

    pca_components = pca.fit_transform(
        numeric_df
    )

    pca_df = pd.DataFrame(
        pca_components,
        columns=[
            f'PC{i+1}'
            for i in range(n_components)
        ],
        index=df.index
    )
    return pca_df, pca

# ==========================================
# PCA EXPLAINED VARIANCE
# ==========================================

def explained_variance(pca):
    variance = (
        pca.explained_variance_ratio_
    )
    variance_df = pd.DataFrame({
        'Principal_Component': [
            f'PC{i+1}'
            for i in range(len(variance))
        ],
        'Explained_Variance': variance
    })

    return variance_df