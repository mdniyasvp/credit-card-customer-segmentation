# Credit Card Customer Segmentation using Unsupervised Machine Learning

## Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques on credit card customer behavior data.

The goal was to identify meaningful customer groups based on:

* purchasing behavior
* repayment patterns
* cash advance usage
* transaction activity
* credit utilization

The project was developed using a complete end-to-end machine learning workflow including:

* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* PCA Dimensionality Reduction
* Clustering Algorithms
* Cluster Evaluation
* Streamlit Deployment

---

# Problem Statement

Financial institutions handle large volumes of customer transaction data. However, understanding customer behavior manually is difficult.

This project aims to:

* segment customers into meaningful groups
* identify spending and repayment patterns
* detect anomalous customer behavior
* support personalized marketing and risk analysis

---

# Dataset Information

Dataset: Credit Card Customer Dataset

Features include:

* balance
* purchases
* cash advance
* payments
* credit limit
* transaction frequency
* installment purchases
* repayment behavior
* tenure

Target:

* No target variable (unsupervised learning problem)

---

# Project Workflow

## 1. Exploratory Data Analysis (EDA)

Performed:

* missing value analysis
* duplicate analysis
* distribution analysis
* skewness analysis
* outlier detection
* correlation heatmaps
* PCA visualization

Key Findings:

* financial features were highly skewed
* extreme outliers were present
* customer behavior varied significantly

---

# 2. Data Preprocessing

Implemented inside:

```text
src/preprocessing.py
```

Steps:

* removed customer ID column
* handled missing values using median imputation
* removed duplicate rows
* applied log transformation
* clipped extreme outliers
* applied RobustScaler

Why RobustScaler?

* financial datasets naturally contain extreme outliers

---

# 3. Feature Engineering

Implemented inside:

```text
src/feature_engineering.py
```

Created behavioral features such as:

* PAYMENT_RATIO
* CREDIT_UTILIZATION
* CASH_ADVANCE_RATIO
* TRANSACTION_INTENSITY

Purpose:

* improve customer behavior representation
* enhance clustering quality

---

# 4. PCA (Principal Component Analysis)

Used PCA for:

* dimensionality reduction
* visualization
* variance compression

Benefits:

* reduced feature redundancy
* improved clustering visualization

---

# 5. Clustering Algorithms Used

## KMeans

* centroid-based clustering
* fast and scalable
* struggled with cluster imbalance

## Hierarchical Clustering

* agglomerative clustering approach
* useful for dendrogram analysis
* produced similar imbalance issues

## DBSCAN (Final Model)

Selected as the final clustering model because it:

* handled outliers effectively
* detected anomalies
* supported irregular cluster shapes
* produced more realistic customer behavior groups

---

# Cluster Evaluation Metrics

Used:

* Silhouette Score
* Davies-Bouldin Score
* Cluster Distribution Analysis

Important Insight:
High silhouette scores alone were misleading because KMeans collapsed most customers into a single cluster.

This project demonstrated the importance of:

* visual validation
* cluster balance
* business interpretability

---

# Final Clustering Insights

The project identified:

* mainstream customer groups
* high repayment customers
* low activity customers
* cash-advance dependent customers
* anomalous customer behavior

DBSCAN successfully detected:

* dense customer populations
* unusual financial behaviors
* rare customer segments

---

# Streamlit Application

A Streamlit application was developed to:

* upload customer CSV data
* run preprocessing pipeline
* apply DBSCAN clustering
* visualize clusters
* download clustered results

Features:

* cluster distribution visualization
* PCA projection visualization
* business interpretation section
* downloadable clustered dataset

---

# Project Structure

```text
credit-card-customer-segmentation/
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   └── project_analysis.ipynb
│
├── outputs/
│   └── clusters/
│
└── src/
    ├── preprocessing.py
    ├── feature_engineering.py
    ├── clustering.py
    ├── evaluation.py
    ├── visualization.py
    ├── pipeline.py
    └── utils.py
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Streamlit
* Joblib

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <your-repository-link>
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/project_analysis.ipynb
```

---

## 4. Run Streamlit App

```bash
streamlit run app.py
```

---

# Results

## Final Model

DBSCAN Clustering

## Key Achievements

* realistic customer segmentation
* anomaly detection
* modular ML pipeline
* business-focused insights
* interactive Streamlit deployment

---

# Future Improvements

Possible enhancements:

* automated hyperparameter tuning
* advanced anomaly detection
* cloud deployment
* interactive dashboards
* Gaussian Mixture Models (GMM)
* deep clustering approaches

---

# Author

Niyas

Aspiring Data Scientist passionate about:

* Machine Learning
* Data Analytics
* Business Intelligence
* AI Applications

---

# Final Conclusion

This project successfully developed an end-to-end unsupervised machine learning solution for customer segmentation using credit card transaction behavior.

The workflow combined:

* preprocessing
* feature engineering
* PCA
* clustering analysis
* business interpretation

to generate meaningful customer insights that can support:

* marketing personalization
* customer retention
* financial risk monitoring
* anomaly detection

DBSCAN ultimately proved to be the most effective clustering method for this financial dataset due to its ability to handle outliers and irregular customer behavior patterns.
