# ==========================================
# IMPORT LIBRARIES
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# SECTION TITLE
# ==========================================

def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

# ==========================================
# COUNT PLOT
# ==========================================

def plot_countplot(
    df,
    column,
    title=None,
    rotation=0,
    save_path=None
):

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )

    plt.title(
        title if title
        else f"{column} Distribution"
    )

    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=rotation)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ==========================================
# HISTOGRAM
# ==========================================

def plot_histogram(
    df,
    column,
    bins=30,
    save_path=None
):

    plt.figure(figsize=(8, 5))
    sns.histplot(
        df[column],
        kde=True,
        bins=bins
    )

    plt.title(
        f"{column} Distribution"
    )

    plt.xlabel(column)
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ==========================================
# BOXPLOT
# ==========================================

def plot_boxplot(
    df,
    column,
    save_path=None
):

    plt.figure(figsize=(8, 4))
    sns.boxplot(
        x=df[column]
    )

    plt.title(
        f"{column} Boxplot"
    )

    plt.xlabel(column)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

def plot_correlation(
    df,
    save_path=None
):

    plt.figure(figsize=(10, 6))
    correlation = df.corr(
        numeric_only=True
    )
    sns.heatmap(
        correlation,
        annot=False,
        cmap='coolwarm'
    )

    plt.title(
        "Correlation Heatmap"
    )

    if save_path:
        plt.savefig(save_path)
    plt.show()