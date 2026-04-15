"""
CMPE475 Assignment 1 - Machine Learning Algorithm Comparison
Dataset: tv-shows.csv (Netflix & streaming platform data)
Task: Binary Classification (Movie vs TV Show)
Author: Kuandyk Kyrykbayev
"""

# ============================================================
# SECTION 0: CONFIGURATION - EDIT THESE TO CUSTOMIZE
# ============================================================

# --- Plot Settings (change these freely) ---
PLOT_CONFIG = {
    # General
    "figsize_confusion": (22, 10),       # (width, height) for confusion matrix grid
    "figsize_roc": (12, 9),              # (width, height) for ROC curve plot
    "figsize_bar": (16, 7),              # (width, height) for bar chart
    "dpi": 150,                          # Resolution for saved images
    "save_plots": True,                  # True = save to disk, False = just display
    "show_plots": True,                  # True = show plt.show(), False = skip

    # Colors
    "roc_colors": [
        "#e74c3c",  # Bayesian - Red
        "#2ecc71",  # Decision Trees - Green
        "#3498db",  # Random Forest - Blue
        "#f39c12",  # Simple Linear Reg - Orange
        "#9b59b6",  # Multiple Linear Reg - Purple
        "#1abc9c",  # Logistic Regression - Teal
        "#e67e22",  # KNN - Dark Orange
        "#34495e",  # SVM - Dark Gray
    ],
    "bar_colormap": "viridis",           # Colormap for bar chart: viridis, plasma, coolwarm, Set2, etc.
    "confusion_cmap": "Blues",           # Colormap for confusion matrix: Blues, Reds, Greens, YlOrRd, etc.

    # Font Sizes
    "title_fontsize": 16,
    "label_fontsize": 13,
    "tick_fontsize": 11,
    "legend_fontsize": 11,
    "cm_title_fontsize": 11,

    # ROC Curve
    "roc_linewidth": 2.2,
    "roc_diagonal_style": "k--",         # Style for random baseline line
    "roc_legend_loc": "lower right",

    # Bar Chart
    "bar_edgecolor": "black",
    "bar_linewidth": 0.5,
    "bar_rotation": 30,                  # X-axis label rotation in degrees

    # Confusion Matrix
    "cm_values_fontsize": 14,
}

# --- Data / Model Settings ---
DATA_CONFIG = {
    "csv_path": "tv-shows.csv",
    "test_size": 0.2,                    # 80/20 train/test split
    "random_state": 42,
    "knn_neighbors": 5,                  # K for KNN
    "dt_max_depth": 5,                   # Max depth for Decision Tree
    "rf_n_estimators": 100,              # Number of trees in Random Forest
    "svm_kernel": "rbf",                 # SVM kernel: 'rbf', 'linear', 'poly'
    "logistic_max_iter": 1000,
}

# --- Output Settings ---
OUTPUT_CONFIG = {
    "results_csv": "results_table.csv",  # Save result table to CSV
    "confusion_file": "confusion_matrices.png",
    "roc_file": "roc_curves.png",
    "bar_file": "metrics_comparison.png",
    "print_table": True,                 # Print result table to console
}


# ============================================================
# SECTION 1: IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, ConfusionMatrixDisplay
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# SECTION 2: LOAD & EXPLORE DATA
# ============================================================

print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_CONFIG["csv_path"])

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nTarget distribution (type):")
print(df['type'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum())
print()


# ============================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================

print("=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

# 1. Extract numeric duration (minutes for Movies, seasons for TV Shows)
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)

# 2. Encode content rating (TV-MA, PG-13, R, etc.)
le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'].fillna('Unknown'))

# 3. Country flags (binary: is this country present?)
df['country_US'] = df['country'].fillna('').str.contains('United States').astype(int)
df['country_India'] = df['country'].fillna('').str.contains('India').astype(int)

# 4. Cast size (number of actors listed)
df['num_cast'] = df['cast'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)

# 5. Has director (binary: is director listed?)
df['has_director'] = df['director'].notna().astype(int)

# 6. Genre features
df['num_genres'] = df['listed_in'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
df['genre_drama'] = df['listed_in'].fillna('').str.contains('Drama', case=False).astype(int)
df['genre_comedy'] = df['listed_in'].fillna('').str.contains('Comed', case=False).astype(int)
df['genre_international'] = df['listed_in'].fillna('').str.contains('International', case=False).astype(int)

# 7. Month added (seasonality)
df['month_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.month.fillna(0).astype(int)

# 8. Description length
df['description_len'] = df['description'].fillna('').str.len()

# 9. Target variable
df['target'] = (df['type'] == 'TV Show').astype(int)

# Define feature list
features = [
    'release_year', 'duration_num', 'rating_encoded', 'country_US',
    'country_India', 'num_cast', 'has_director', 'num_genres',
    'genre_drama', 'genre_comedy', 'genre_international',
    'month_added', 'description_len'
]

print(f"Engineered {len(features)} features:")
for i, f in enumerate(features, 1):
    print(f"  {i:2d}. {f}")
print()


# ============================================================
# SECTION 4: PREPARE DATA & SPLIT
# ============================================================

print("=" * 60)
print("STEP 3: Train/Test Split & Scaling")
print("=" * 60)

X = df[features].fillna(0)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=DATA_CONFIG["test_size"],
    random_state=DATA_CONFIG["random_state"],
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print(f"Class balance (test): Movie={sum(y_test == 0)}, TV Show={sum(y_test == 1)}")
print()


# ============================================================
# SECTION 5: DEFINE & TRAIN ALL 8 MODELS
# ============================================================

print("=" * 60)
print("STEP 4: Training 8 ML Models")
print("=" * 60)

models = {
    'Bayesian (Naive Bayes)': GaussianNB(),
    'Decision Trees': DecisionTreeClassifier(
        max_depth=DATA_CONFIG["dt_max_depth"],
        random_state=DATA_CONFIG["random_state"]
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=DATA_CONFIG["rf_n_estimators"],
        random_state=DATA_CONFIG["random_state"]
    ),
    'Simple Linear Regression': LinearRegression(),
    'Multiple Linear Regression': LinearRegression(),
    'Logistic Regression': LogisticRegression(
        max_iter=DATA_CONFIG["logistic_max_iter"],
        random_state=DATA_CONFIG["random_state"]
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=DATA_CONFIG["knn_neighbors"]
    ),
    'SVM': SVC(
        kernel=DATA_CONFIG["svm_kernel"],
        probability=True,
        random_state=DATA_CONFIG["random_state"]
    ),
}


# ============================================================
# SECTION 6: EVALUATE & BUILD RESULT TABLE
# ============================================================

results = []
predictions = {}  # Store predictions for plots

for name, model in models.items():
    # Simple Linear Regression: use only 1 feature (release_year)
    if name == 'Simple Linear Regression':
        feature_idx = 0  # release_year index
        Xtr = X_train_scaled[:, feature_idx].reshape(-1, 1)
        Xte = X_test_scaled[:, feature_idx].reshape(-1, 1)
    else:
        Xtr = X_train_scaled
        Xte = X_test_scaled

    # Train
    model.fit(Xtr, y_train)

    # Predict
    if 'Linear Regression' in name and 'Logistic' not in name:
        # Linear regression outputs continuous values -> threshold at 0.5
        y_pred_raw = model.predict(Xte)
        y_pred = (y_pred_raw >= 0.5).astype(int)
        y_prob = np.clip(y_pred_raw, 0, 1)
    else:
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

    # Calculate 5 metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })

    predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    print(f"  [OK] {name:35s} - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

print()


# ============================================================
# SECTION 7: DISPLAY RESULT TABLE
# ============================================================

print("=" * 60)
print("STEP 5: Result Table")
print("=" * 60)

results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)

if OUTPUT_CONFIG["print_table"]:
    print()
    print(results_df.to_string())
    print()

# Save to CSV
results_df.to_csv(OUTPUT_CONFIG["results_csv"])
print(f"Result table saved to: {OUTPUT_CONFIG['results_csv']}")
print()


# ============================================================
# SECTION 8: PLOTS
# ============================================================

print("=" * 60)
print("STEP 6: Generating Plots")
print("=" * 60)

pc = PLOT_CONFIG  # shorthand


# --- PLOT 1: Confusion Matrices ---
def plot_confusion_matrices():
    """Generates a grid of confusion matrices for all 8 models."""
    fig, axes = plt.subplots(2, 4, figsize=pc["figsize_confusion"])
    axes = axes.ravel()

    model_names = list(models.keys())

    for i, name in enumerate(model_names):
        y_pred = predictions[name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        disp = ConfusionMatrixDisplay(cm, display_labels=['Movie', 'TV Show'])
        disp.plot(ax=axes[i], cmap=pc["confusion_cmap"], colorbar=False)
        axes[i].set_title(name, fontsize=pc["cm_title_fontsize"], fontweight='bold')
        axes[i].set_xlabel('Predicted', fontsize=pc["tick_fontsize"])
        axes[i].set_ylabel('Actual', fontsize=pc["tick_fontsize"])

        # Set values font size
        for text in disp.text_.ravel():
            text.set_fontsize(pc["cm_values_fontsize"])

    fig.suptitle('Confusion Matrices - All Models', fontsize=pc["title_fontsize"], fontweight='bold', y=1.02)
    plt.tight_layout()

    if pc["save_plots"]:
        fig.savefig(OUTPUT_CONFIG["confusion_file"], dpi=pc["dpi"], bbox_inches='tight')
        print(f"  [OK] Saved: {OUTPUT_CONFIG['confusion_file']}")

    if pc["show_plots"]:
        plt.show()
    else:
        plt.close()


# --- PLOT 2: ROC Curves ---
def plot_roc_curves():
    """Generates an overlaid ROC curve plot for all 8 models."""
    fig, ax = plt.subplots(figsize=pc["figsize_roc"])

    model_names = list(models.keys())
    colors = pc["roc_colors"]

    for i, name in enumerate(model_names):
        y_prob = predictions[name]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)

        color = colors[i % len(colors)]
        ax.plot(fpr, tpr,
                color=color,
                linewidth=pc["roc_linewidth"],
                label=f'{name} (AUC = {auc_val:.3f})')

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], pc["roc_diagonal_style"],
            linewidth=1.5, label='Random Baseline (AUC = 0.500)', alpha=0.7)

    ax.set_xlabel('False Positive Rate', fontsize=pc["label_fontsize"])
    ax.set_ylabel('True Positive Rate', fontsize=pc["label_fontsize"])
    ax.set_title('ROC Curve Comparison - All Models', fontsize=pc["title_fontsize"], fontweight='bold')
    ax.legend(loc=pc["roc_legend_loc"], fontsize=pc["legend_fontsize"])
    ax.tick_params(labelsize=pc["tick_fontsize"])
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    if pc["save_plots"]:
        fig.savefig(OUTPUT_CONFIG["roc_file"], dpi=pc["dpi"], bbox_inches='tight')
        print(f"  [OK] Saved: {OUTPUT_CONFIG['roc_file']}")

    if pc["show_plots"]:
        plt.show()
    else:
        plt.close()


# --- PLOT 3: Metrics Bar Chart ---
def plot_metrics_bar():
    """Generates a grouped bar chart comparing all metrics across models."""
    fig, ax = plt.subplots(figsize=pc["figsize_bar"])

    n_models = len(results_df)
    n_metrics = len(results_df.columns)
    x = np.arange(n_models)
    bar_width = 0.15

    cmap = plt.get_cmap(pc["bar_colormap"])
    colors = [cmap(i / n_metrics) for i in range(n_metrics)]

    for j, metric in enumerate(results_df.columns):
        offset = (j - n_metrics / 2) * bar_width + bar_width / 2
        bars = ax.bar(x + offset, results_df[metric], bar_width,
                      label=metric, color=colors[j],
                      edgecolor=pc["bar_edgecolor"],
                      linewidth=pc["bar_linewidth"])

    ax.set_xlabel('Model', fontsize=pc["label_fontsize"])
    ax.set_ylabel('Score', fontsize=pc["label_fontsize"])
    ax.set_title('Model Comparison - All Metrics', fontsize=pc["title_fontsize"], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df.index, rotation=pc["bar_rotation"],
                       ha='right', fontsize=pc["tick_fontsize"])
    ax.legend(fontsize=pc["legend_fontsize"], loc='lower left')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=pc["tick_fontsize"])

    plt.tight_layout()

    if pc["save_plots"]:
        fig.savefig(OUTPUT_CONFIG["bar_file"], dpi=pc["dpi"], bbox_inches='tight')
        print(f"  [OK] Saved: {OUTPUT_CONFIG['bar_file']}")

    if pc["show_plots"]:
        plt.show()
    else:
        plt.close()


# Generate all plots
plot_confusion_matrices()
plot_roc_curves()
plot_metrics_bar()

print()


# ============================================================
# SECTION 9: SUMMARY
# ============================================================

print("=" * 60)
print("DONE - Summary")
print("=" * 60)

best_model = results_df['Accuracy'].idxmax()
best_acc = results_df.loc[best_model, 'Accuracy']
best_auc_model = results_df['ROC-AUC'].idxmax()
best_auc = results_df.loc[best_auc_model, 'ROC-AUC']

print(f"  Dataset:          tv-shows.csv ({df.shape[0]} rows)")
print(f"  Task:             Binary Classification (Movie vs TV Show)")
print(f"  Features used:    {len(features)}")
print(f"  Train/Test split: {int((1-DATA_CONFIG['test_size'])*100)}/{int(DATA_CONFIG['test_size']*100)}")
print(f"  Best by Accuracy: {best_model} ({best_acc})")
print(f"  Best by ROC-AUC:  {best_auc_model} ({best_auc})")
print(f"\n  Output files:")
print(f"    - {OUTPUT_CONFIG['results_csv']}")
if PLOT_CONFIG["save_plots"]:
    print(f"    - {OUTPUT_CONFIG['confusion_file']}")
    print(f"    - {OUTPUT_CONFIG['roc_file']}")
    print(f"    - {OUTPUT_CONFIG['bar_file']}")
print()
