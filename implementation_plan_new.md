# Homework 2 — Add LSTM & GRU Models to ML Assignment

## Goal

Extend the existing CMPE475 Homework 1 (which compares 8 ML algorithms on `tv-shows.csv`) by adding two ANN/RNN models — **LSTM** and **GRU** — to the bottom of the current script, then updating the result table, plots, report, and explanation accordingly.

## Background

The current `ml_assignment.py` trains 8 models (Naive Bayes, Decision Trees, Random Forest, Simple/Multiple Linear Regression, Logistic Regression, KNN, SVM) on 13 engineered features from the `tv-shows.csv` dataset for binary classification (Movie vs TV Show). 

Homework 2 asks us to add LSTM and GRU neural networks. Since the data is **tabular** (not sequential/time-series), we will reshape the 13 features into a pseudo-sequence of shape `(samples, timesteps=13, features_per_step=1)` — treating each feature as one "timestep". This is a standard approach when applying RNNs to tabular data for academic comparison purposes.

> [!IMPORTANT]
> **TensorFlow/Keras is NOT installed.** We will need to install it via `pip install tensorflow` before running.

## Proposed Changes

### Dependency Installation

Install TensorFlow (includes Keras):
```
pip install tensorflow
```

---

### Code — `ml_assignment.py`

#### [MODIFY] [ml_assignment.py](file:///c:/Users/kuand/Documents/GitHub/CMPE475-ASSIGNMENT-1-KUANDYK-KYRYKBAYEV/ml_assignment.py)

**Changes (appended to the bottom, before SECTION 9: SUMMARY):**

1. **New configuration** in `DATA_CONFIG` at the top:
   - `lstm_epochs: 50`
   - `lstm_batch_size: 32`
   - `gru_epochs: 50`
   - `gru_batch_size: 32`
   - `ann_learning_rate: 0.001`

2. **New output file** in `OUTPUT_CONFIG`:
   - `ann_training_file: "ann_training_history.png"` (training loss/accuracy curves)
   - `confusion_file_all: "confusion_matrices_all.png"` (updated 10-model grid)
   - `roc_file_all: "roc_curves_all.png"` (updated ROC with 10 models)
   - `bar_file_all: "metrics_comparison_all.png"` (updated bar chart with 10 models)

3. **New colors** added to `PLOT_CONFIG["roc_colors"]`:
   - Two extra colors for LSTM and GRU

4. **New SECTION 10: ANN MODELS (LSTM & GRU)** inserted before the Summary section:
   - Import `tensorflow.keras` modules
   - Reshape `X_train_scaled` and `X_test_scaled` to 3D: `(samples, 13, 1)`
   - Build **LSTM model**:
     - `LSTM(64, return_sequences=True)` → `LSTM(32)` → `Dense(16, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
     - Compile with `adam` optimizer, `binary_crossentropy` loss
     - Train with `EarlyStopping(patience=5)`
   - Build **GRU model**:
     - `GRU(64, return_sequences=True)` → `GRU(32)` → `Dense(16, relu)` → `Dropout(0.3)` → `Dense(1, sigmoid)`
     - Same compilation and training approach
   - Evaluate both models using the same 5 metrics
   - Append results to the existing `results` list and `predictions` dict
   - Plot training history (loss + accuracy curves for both models)

5. **Updated SECTION 7 (Result Table)** — moved after ANN section so the table includes all 10 models

6. **Updated SECTION 8 (Plots)** — confusion matrix grid becomes 2×5 (10 models), ROC & bar chart include all 10 models

7. **Updated SECTION 9 (Summary)** — reflects 10 models total

---

### Report — `REPORT.txt`

#### [MODIFY] [REPORT.txt](file:///c:/Users/kuand/Documents/GitHub/CMPE475-ASSIGNMENT-1-KUANDYK-KYRYKBAYEV/REPORT.txt)

- Add **Section 7.9**: LSTM description, architecture, hyperparameters
- Add **Section 7.10**: GRU description, architecture, hyperparameters  
- Update **Section 8**: Result table with 10 models, updated analysis including ANN findings
- Update **Section 9**: New plot for training history curves
- Update **Section 10**: Conclusions expanded with ANN comparison findings
- Update **Section 11**: New files list
- Update **Section 12**: Add TensorFlow/Keras to libraries

---

### Explanation — `EXPLANATION.md`

#### [MODIFY] [EXPLANATION.md](file:///c:/Users/kuand/Documents/GitHub/CMPE475-ASSIGNMENT-1-KUANDYK-KYRYKBAYEV/EXPLANATION.md)

- Add **Algorithm 9 (LSTM)** and **Algorithm 10 (GRU)** sections explaining:
  - How RNNs work conceptually
  - LSTM gates (forget, input, output)
  - GRU gates (reset, update) — simplified version of LSTM
  - Why we reshape tabular data to 3D for RNN input
  - Architecture choices and hyperparameter rationale
- Update **Results Analysis** section with the two new models
- Update **Libraries Used** table with TensorFlow/Keras

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | TensorFlow/Keras | Simple API, widely used in academic settings |
| Reshape strategy | `(samples, 13, 1)` | Each feature becomes one timestep; standard for tabular-to-RNN |
| Hidden layers | 2 RNN layers (64→32) + 1 Dense(16) | Enough capacity without overfitting on 9K samples |
| Dropout | 0.3 | Regularization to prevent overfitting |
| Early stopping | patience=5 | Stops training when validation loss stops improving |
| Epochs | 50 max | Sufficient with early stopping; avoids unnecessary compute |
| Activation | sigmoid (output) | Binary classification |

## Verification Plan

### Automated Tests

1. `pip install tensorflow` — confirm installation succeeds
2. `python ml_assignment.py` — confirm full script runs without errors and produces:
   - Updated `results_table.csv` with 10 rows
   - Updated plot files (confusion matrices, ROC curves, bar chart)
   - New `ann_training_history.png`
3. Verify all 10 models appear in the result table with valid metric values

### Manual Verification

- Review the generated plots to confirm LSTM & GRU appear correctly
- Review the updated REPORT.txt and EXPLANATION.md for completeness
