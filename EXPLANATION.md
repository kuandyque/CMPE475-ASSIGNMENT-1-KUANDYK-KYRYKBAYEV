# CMPE475 Assignment 1 & 2 — Detailed Explanation

## Machine Learning Algorithm Comparison on TV Shows Dataset (Including LSTM & GRU)

---

## Table of Contents

1. [Database Selection](#1-database-selection)
2. [Data Understanding](#2-data-understanding)
3. [Target Selection — Why Classification?](#3-target-selection--why-classification)
4. [Feature Engineering — Why and How](#4-feature-engineering--why-and-how)
5. [Data Preprocessing](#5-data-preprocessing)
6. [The 5 Metrics — What They Mean](#6-the-5-metrics--what-they-mean)
7. [The 8 Traditional ML Algorithms](#7-the-8-traditional-ml-algorithms--how-each-works)
8. [The 2 ANN Models — LSTM & GRU (Homework 2)](#8-the-2-ann-models--lstm--gru-homework-2)
9. [Results Analysis](#9-results-analysis)
10. [Plots Explanation](#10-plots-explanation)
11. [Libraries Used](#11-libraries-used)

---

## 1. Database Selection

### What is the dataset?

The dataset is `tv-shows.csv`, a collection of **9,338 titles** (movies and TV shows) from streaming platforms including Netflix. Each row represents one title with metadata like director, cast, country, release year, content rating, duration, genre, and description.

### Why this dataset?

- **Tabular format (CSV)**: All 8 ML algorithms in the assignment work natively with tabular data. No image processing, audio feature extraction, or NLP pipelines are needed.
- **Large enough**: 9,338 rows provides sufficient data for training and testing without overfitting concerns.
- **Natural binary target**: The `type` column contains exactly two classes — "Movie" and "TV Show" — making it a natural binary classification problem.
- **Rich features**: Multiple columns (country, rating, duration, genres, cast) provide diverse signals for the algorithms to learn from.
- **Real-world data**: Contains missing values and mixed data types, which teaches practical data handling.

### Source

The dataset comes from publicly available streaming platform data aggregators. It follows the CSV format requirement and falls under the **Tabular** modality.

---

## 2. Data Understanding

### Column-by-Column Breakdown

| Column | Type | Example | Missing Values | Role |
|---|---|---|---|---|
| `id` | Integer | 1, 2, 3 | 0 | Identifier (dropped) |
| `type` | Categorical | "Movie", "TV Show" | 0 | **TARGET** |
| `title` | Text | "Star Trek" | 0 | Dropped (raw text) |
| `director` | Text | "J.J. Abrams" | 2,855 (30.6%) | Feature extracted |
| `cast` | Text | "Chris Pine, ..." | 938 (10.0%) | Feature extracted |
| `country` | Categorical | "United States" | 962 (10.3%) | Feature extracted |
| `date_added` | Date string | "July 1, 2021" | 13 (0.1%) | Feature extracted |
| `release_year` | Integer | 2009 | 0 | Direct feature |
| `rating` | Categorical | "PG-13", "TV-MA" | 7 (0.1%) | Encoded feature |
| `duration` | Mixed | "128 min" / "1 Season" | 3 (0.0%) | Feature extracted |
| `listed_in` | Multi-label | "Action, Sci-Fi" | 0 | Feature extracted |
| `description` | Text | "On their first voyage..." | 0 | Feature extracted |
| `platform` | Categorical | "Netflix" | 0 | Feature extracted |

### Class Distribution

- **Movie**: 6,528 (69.9%)
- **TV Show**: 2,810 (30.1%)

This is a **moderately imbalanced** dataset. The majority class (Movie) is about 2.3x more frequent than the minority class (TV Show). This imbalance is why we use metrics beyond just Accuracy (like F1-Score and ROC-AUC).

---

## 3. Target Selection — Why Classification?

### Why Classification over Regression?

The assignment requires comparing algorithms including Logistic Regression, Naive Bayes, KNN, and Decision Trees. These are primarily **classification** algorithms. While they can be adapted for regression, they perform optimally and most naturally for classification tasks.

### Why `type` as the target?

The `type` column is the best target because:

1. **Naturally binary**: Only two values ("Movie" vs "TV Show") — perfect for binary classification
2. **Meaningful distinction**: Movies and TV Shows have genuinely different characteristics (duration, cast size, genre patterns) — the algorithms have real patterns to learn
3. **Clean**: Zero missing values in the target column
4. **Balanced enough**: 70/30 split is workable without requiring advanced resampling techniques like SMOTE

### What does the model predict?

Given a title's metadata (release year, duration, rating, country, genres, etc.), the model predicts whether it is a **Movie** (class 0) or a **TV Show** (class 1).

---

## 4. Feature Engineering — Why and How

Machine learning algorithms require **numeric input**. Our raw data contains text, dates, and categorical values. Feature engineering transforms these into meaningful numbers.

### Feature 1: `release_year` (Direct Use)

```python
# Already numeric — used directly
# Example: 2009, 2020, 1986
```

**Why**: Release year captures temporal trends. Older titles tend to be movies; newer entries have more TV shows as streaming platforms invested in original series.

### Feature 2: `duration_num` (Extracted from `duration`)

```python
df['duration_num'] = df['duration'].str.extract(r'(\d+)').astype(float)
# "128 min" → 128, "3 Seasons" → 3
```

**Why**: This is the **most powerful feature**. Movies have durations in minutes (60-200), while TV Shows have durations in seasons (1-10). The numeric range alone almost perfectly separates the two classes. The regex `(\d+)` extracts just the number from the string.

### Feature 3: `rating_encoded` (Label-Encoded from `rating`)

```python
le_rating = LabelEncoder()
df['rating_encoded'] = le_rating.fit_transform(df['rating'].fillna('Unknown'))
```

**Why**: Content ratings (TV-MA, PG-13, R, TV-Y, etc.) correlate with content type. Movies tend to use MPAA ratings (PG, PG-13, R) while TV shows use TV ratings (TV-MA, TV-14, TV-Y). `LabelEncoder` converts each unique string to a unique integer.

### Features 4-5: `country_US`, `country_India` (Binary Flags)

```python
df['country_US'] = df['country'].fillna('').str.contains('United States').astype(int)
df['country_India'] = df['country'].fillna('').str.contains('India').astype(int)
```

**Why**: The US and India are the two most common production countries. India produces a disproportionate number of movies (Bollywood), while US production is more balanced. Binary flags (0 or 1) are simple and effective.

### Feature 6: `num_cast` (Cast Size)

```python
df['num_cast'] = df['cast'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
```

**Why**: TV shows often have larger ensemble casts listed, while movies typically list fewer top-billed actors. Counting commas in the cast string approximates the number of actors.

### Feature 7: `has_director` (Binary: Director Listed?)

```python
df['has_director'] = df['director'].notna().astype(int)
```

**Why**: 30.6% of entries have no director listed. TV shows are far more likely to omit a director (since they have multiple directors per season), making this a strong signal.

### Features 8-11: Genre Features

```python
df['num_genres'] = df['listed_in'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
df['genre_drama'] = df['listed_in'].fillna('').str.contains('Drama', case=False).astype(int)
df['genre_comedy'] = df['listed_in'].fillna('').str.contains('Comed', case=False).astype(int)
df['genre_international'] = df['listed_in'].fillna('').str.contains('International', case=False).astype(int)
```

**Why**: 
- `num_genres`: Number of genre tags. TV shows are often tagged with more categories (e.g., "International TV Shows, TV Dramas, Crime TV Shows").
- `genre_drama`, `genre_comedy`, `genre_international`: Binary flags for the most common genres. The word "TV" appears in TV show genre tags, so genre text patterns differ systematically between movies and shows.

### Feature 12: `month_added` (Seasonality)

```python
df['month_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.month.fillna(0).astype(int)
```

**Why**: Streaming platforms follow release patterns — movies and shows may be added in different seasons. `errors='coerce'` handles malformed dates gracefully by converting them to NaN instead of raising an error.

### Feature 13: `description_len` (Description Length)

```python
df['description_len'] = df['description'].fillna('').str.len()
```

**Why**: Description length in characters varies between movies and TV shows. TV shows may have shorter or longer descriptions depending on the platform's conventions.

### Summary: 13 Features

All 13 features are numeric and represent different aspects of each title. No raw text or categorical strings are fed directly to the algorithms.

---

## 5. Data Preprocessing

### Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

- **80/20 split**: 80% for training (7,470 samples), 20% for testing (1,868 samples). This is the standard practice for datasets of this size.
- **`random_state=42`**: Ensures reproducibility — running the code again produces the same split.
- **`stratify=y`**: Maintains the same class proportion (70/30 Movie/TV Show) in both train and test sets. Without stratification, the random split could accidentally put 80% movies in training and 50% in testing, skewing results.

### Feature Scaling (StandardScaler)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why scale?**

Different features have vastly different ranges:
- `release_year`: 1925–2021 (range ~100)
- `duration_num`: 1–312 (range ~300)
- `description_len`: 10–500 (range ~500)
- `country_US`: 0 or 1 (range 1)

**KNN** measures distance between points — a feature with range 500 would dominate one with range 1. **SVM** also relies on feature scales for its kernel computations. StandardScaler transforms each feature to have **mean=0** and **standard deviation=1**, making all features equally important.

**Important**: We `fit` the scaler on training data only, then `transform` both train and test data using the same parameters. This prevents **data leakage** — the test set's statistics should not influence preprocessing.

---

## 6. The 5 Metrics — What They Mean

### Why 5 metrics instead of just accuracy?

Accuracy alone can be misleading. In our dataset, if a model simply predicted "Movie" for every single sample, it would achieve ~70% accuracy (because 70% of the data is movies). But it would completely fail at identifying TV shows. Multiple metrics give a fuller picture.

### Metric 1: Accuracy

```
Accuracy = (TP + TN) / Total
```

- **What it measures**: Percentage of all predictions that are correct.
- **Strength**: Intuitive and easy to understand.
- **Weakness**: Misleading on imbalanced datasets (our 70/30 split).
- **Example**: If we correctly predict 1300/1306 Movies and 553/562 TV Shows → Accuracy = 1853/1868 = 99.2%

### Metric 2: Precision

```
Precision = TP / (TP + FP)
```

- **What it measures**: Of all samples predicted as "TV Show", how many actually were TV Shows?
- **Analogy**: If I flag 100 emails as spam, precision tells me how many were actually spam.
- **High precision** = Few false alarms.

### Metric 3: Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

- **What it measures**: Of all actual TV Shows, how many did the model correctly identify?
- **Analogy**: Out of 100 actual spam emails, recall tells me how many I caught.
- **High recall** = Few missed cases.

### Metric 4: F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **What it measures**: Harmonic mean of Precision and Recall. Balances both concerns.
- **Why harmonic mean?**: Unlike a simple average, F1 penalizes extreme imbalances. If Precision=1.0 and Recall=0.0, the arithmetic mean is 0.5, but F1 = 0.0, correctly indicating poor performance.
- **Best single metric** for imbalanced classification.

### Metric 5: ROC-AUC (Area Under the ROC Curve)

```
ROC-AUC = Area under the Receiver Operating Characteristic curve
```

- **What it measures**: The model's ability to distinguish between classes across all possible thresholds.
- **AUC = 1.0**: Perfect separation of classes.
- **AUC = 0.5**: No better than random guessing (coin flip).
- **Why it's important**: Unlike the other metrics which depend on a fixed 0.5 decision threshold, ROC-AUC evaluates performance across all possible thresholds, giving a threshold-independent measure of discriminative power.

---

## 7. The 8 Traditional ML Algorithms — How Each Works

### 1. Bayesian (Naive Bayes) — `GaussianNB()`

**How it works**: Based on **Bayes' Theorem** — calculates the probability of each class given the features. "Naive" because it assumes all features are independent of each other (which is rarely true, but works surprisingly well in practice).

```
P(class | features) = P(features | class) × P(class) / P(features)
```

**Gaussian** variant assumes features follow a normal (bell curve) distribution.

**Strengths**: Very fast, works well with small datasets, handles many features efficiently.
**Weaknesses**: The independence assumption is often violated. Performance decreases when features are correlated.

**Our result**: Accuracy = 0.9920, F1 = 0.9866

### 2. Decision Trees — `DecisionTreeClassifier(max_depth=5)`

**How it works**: Builds a tree-like flowchart of yes/no questions. At each node, it picks the feature and threshold that best splits the data into pure groups. Uses metrics like **Gini impurity** or **Information Gain** to decide the best split.

**Example decision path**: "Is duration_num > 15? → Yes → Is has_director = 1? → Yes → Predict Movie"

**`max_depth=5`**: Limits the tree to 5 levels deep. Without this limit, the tree would grow until every training sample is perfectly classified (overfitting — memorizing instead of learning).

**Strengths**: Highly interpretable, handles non-linear relationships, no scaling needed.
**Weaknesses**: Prone to overfitting without depth limits, unstable (small data changes can change the tree structure).

**Our result**: Accuracy = 0.9968, F1 = 0.9947

### 3. Random Forest — `RandomForestClassifier(n_estimators=100)`

**How it works**: Builds **100 different decision trees**, each trained on a random subset of the data and features. For prediction, each tree "votes" and the majority vote wins. This is called **ensemble learning** — combining many weak learners into one strong learner.

**`n_estimators=100`**: Number of trees in the forest. More trees = more stable predictions, but slower training.

**Why it outperforms a single Decision Tree**: Individual trees may overfit, but averaging 100 random trees smooths out the noise and reduces variance.

**Strengths**: Very high accuracy, resistant to overfitting, handles missing values well, provides feature importance rankings.
**Weaknesses**: Slower to train, less interpretable than a single tree (it's a "black box" of 100 trees).

**Our result**: Accuracy = 0.9968, F1 = 0.9947, ROC-AUC = 0.9999 (best)

### 4. Simple Linear Regression — `LinearRegression()` (1 feature)

**How it works**: Fits a straight line `y = mx + b` using a single feature (`release_year`). Since this is a regression algorithm (designed to predict continuous numbers, not classes), we threshold the output at 0.5: if the predicted value ≥ 0.5, classify as "TV Show"; otherwise, "Movie".

**Why only 1 feature?**: "Simple" in Simple Linear Regression means using exactly one independent variable. This is the definition that distinguishes it from Multiple Linear Regression.

**Strengths**: Extremely simple, fast, easy to interpret.
**Weaknesses**: Cannot capture complex relationships. Using only `release_year` means it can only learn "newer = more likely TV Show", which is insufficient. This is reflected in its poor performance.

**Our result**: Accuracy = 0.6991, F1 = 0.0000 — the worst performer. With only release_year, it essentially predicts "Movie" for everything because there's no clear linear boundary in year alone.

### 5. Multiple Linear Regression — `LinearRegression()` (all features)

**How it works**: Same as Simple Linear Regression, but uses **all 13 features**: `y = w1×x1 + w2×x2 + ... + w13×x13 + b`. Each feature gets its own weight. Output is again thresholded at 0.5.

**Strengths**: Can capture combined effects of all features, still relatively fast and interpretable.
**Weaknesses**: Still assumes a linear relationship between features and target. Cannot capture interactions or non-linearities. Not designed for classification (that's what Logistic Regression is for).

**Our result**: Accuracy = 0.9625, F1 = 0.9388 — huge improvement over Simple LR because all 13 features provide much more information.

### 6. Logistic Regression — `LogisticRegression(max_iter=1000)`

**How it works**: Despite the name, Logistic Regression is a **classification** algorithm. It applies a **sigmoid function** to the linear combination of features, squishing the output to a (0, 1) probability range:

```
P(TV Show) = 1 / (1 + e^(-(w1×x1 + w2×x2 + ... + b)))
```

If probability > 0.5 → predict "TV Show"; else → predict "Movie".

**`max_iter=1000`**: Maximum iterations for the optimization algorithm to converge. Default (100) sometimes isn't enough for larger datasets.

**Strengths**: True probabilistic classifier, outputs interpretable probabilities, works well when classes are linearly separable, fast and scalable.
**Weaknesses**: Assumes linear decision boundary. Struggles with highly non-linear problems.

**Our result**: Accuracy = 0.9963, F1 = 0.9938 — excellent performance, proving this problem has a nearly linear decision boundary.

### 7. KNN (K-Nearest Neighbors) — `KNeighborsClassifier(n_neighbors=5)`

**How it works**: To classify a new sample, KNN finds the **5 nearest training samples** (measured by Euclidean distance in feature space) and assigns the majority class among those 5 neighbors.

**`n_neighbors=5`**: The "K" in KNN. If K=1, it copies the label of the single closest point (very sensitive to noise). If K=100, it's very smooth but may lose local patterns. K=5 is a common default.

**Why scaling matters for KNN**: Distance is calculated as:
```
distance = sqrt((x1a - x1b)^2 + (x2a - x2b)^2 + ...)
```
If `release_year` ranges 0-2000 and `country_US` ranges 0-1, then release_year would dominate the distance. StandardScaler fixes this.

**Strengths**: Simple, intuitive, non-parametric (makes no assumptions about data distribution), naturally handles multi-class problems.
**Weaknesses**: Slow on large datasets (must compare every test point to every training point), sensitive to irrelevant features and scale.

**Our result**: Accuracy = 0.9764, F1 = 0.9614

### 8. SVM (Support Vector Machine) — `SVC(kernel='rbf', probability=True)`

**How it works**: SVM finds the **hyperplane** (decision boundary) that maximally separates the two classes. It focuses on the "support vectors" — the training points closest to the boundary.

**`kernel='rbf'`**: The Radial Basis Function kernel allows SVM to find **non-linear** decision boundaries by mapping data into a higher-dimensional space. Think of it as "bending" the feature space so that a curved boundary in the original space becomes a straight line in the transformed space.

**`probability=True`**: Enables probability estimates (needed for ROC-AUC calculation). SVM normally only outputs class labels, not probabilities.

**Strengths**: Effective in high-dimensional spaces, memory-efficient (only stores support vectors), excellent for clear-margin problems.
**Weaknesses**: Slow on very large datasets, sensitive to feature scaling, less interpretable.

**Our result**: Accuracy = 0.9930, F1 = 0.9885, ROC-AUC = 0.9999

---

## 8. The 2 ANN Models — LSTM & GRU (Homework 2)

### Why Add Neural Networks?

Homework 2 extends the comparison by adding two **Artificial Neural Network (ANN)** models from the Recurrent Neural Network (RNN) family. RNNs are designed to process **sequential data** where the order of inputs matters (e.g., time series, text, speech). By applying them to our tabular dataset, we can compare their performance against traditional ML algorithms and understand when deep learning approaches are beneficial.

### Data Reshaping for RNNs

RNNs require 3D input: `(samples, timesteps, features_per_step)`. Our tabular data is 2D: `(samples, 13_features)`. We reshape it to `(samples, 13, 1)` — treating each of the 13 features as one "timestep" with 1 value. This is a common approach for applying RNNs to tabular data in academic comparisons.

```python
X_train_rnn = X_train_scaled.reshape(-1, 13, 1)  # (7470, 13, 1)
X_test_rnn = X_test_scaled.reshape(-1, 13, 1)    # (1868, 13, 1)
```

### 9. LSTM (Long Short-Term Memory)

**How it works**: LSTM is a specialized RNN that solves the **vanishing gradient problem** — the inability of standard RNNs to learn dependencies over long sequences. It introduces a **cell state** (a conveyor belt of information) controlled by three gates:

1. **Forget Gate**: Decides what information to discard from the cell state
   ```
   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
   ```

2. **Input Gate**: Decides what new information to store in the cell state
   ```
   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
   C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
   ```

3. **Output Gate**: Decides what to output based on the cell state
   ```
   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
   h_t = o_t * tanh(C_t)
   ```

Each gate is a neural network layer with sigmoid activation (σ) that outputs values between 0 and 1, acting as a "valve" for information flow.

**Our architecture**:

```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(13, 1)),  # Layer 1: 64 units
    LSTM(32, return_sequences=False),                       # Layer 2: 32 units
    Dense(16, activation='relu'),                           # Fully connected layer
    Dropout(0.3),                                           # 30% dropout regularization
    Dense(1, activation='sigmoid')                          # Output: probability
])
```

**Hyperparameter choices**:
- **2 LSTM layers (64 → 32)**: Stacked LSTM layers allow learning hierarchical representations. The first layer processes raw feature sequences; the second learns higher-level patterns. `return_sequences=True` on the first layer passes the full sequence to the second layer.
- **Dense(16, relu)**: A fully connected layer after the LSTM layers adds non-linear transformation capacity before the final output.
- **Dropout(0.3)**: Randomly drops 30% of neurons during training to prevent overfitting — especially important with 9K samples and thousands of neural network parameters.
- **Adam optimizer (lr=0.001)**: Adaptive learning rate optimizer that's the standard choice for neural networks.
- **EarlyStopping (patience=5)**: Stops training when validation loss hasn't improved for 5 consecutive epochs, preventing overfitting.

**Our result**: Accuracy = 0.9845, F1 = 0.9742, AUC = 0.9977. Stopped at epoch 22/50.

### 10. GRU (Gated Recurrent Unit)

**How it works**: GRU is a **simplified variant of LSTM** proposed by Cho et al. (2014). It reduces the three gates to two and merges the cell state with the hidden state:

1. **Reset Gate**: Controls how much past information to forget
   ```
   r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
   ```

2. **Update Gate**: Controls how much new vs old information to keep (combines LSTM's forget + input gates)
   ```
   z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
   h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
   ```

**Key difference from LSTM**: GRU has **no separate cell state** — it uses only the hidden state. This means fewer parameters (faster training, less memory) but potentially less expressive power for complex patterns.

**Our architecture**:

```python
model = Sequential([
    GRU(64, return_sequences=True, input_shape=(13, 1)),   # Layer 1: 64 units
    GRU(32, return_sequences=False),                        # Layer 2: 32 units
    Dense(16, activation='relu'),                           # Fully connected layer
    Dropout(0.3),                                           # 30% dropout regularization
    Dense(1, activation='sigmoid')                          # Output: probability
])
```

**LSTM vs GRU comparison**:

| Aspect | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell state | Separate cell state + hidden state | Only hidden state |
| Parameters | More (slower training) | Fewer (faster training) |
| Expressiveness | Higher | Lower |
| Best for | Complex, long sequences | Simpler patterns, smaller datasets |

**Our result**: Accuracy = 0.9234, F1 = 0.8817, AUC = 0.9725. Stopped at epoch 5/50.

---

## 9. Results Analysis

### The Result Table (10 Models: 8 ML + 2 ANN)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Bayesian (Naive Bayes) | 0.9920 | 0.9893 | 0.9840 | 0.9866 | 0.9989 |
| Decision Trees | **0.9968** | **0.9947** | 0.9947 | **0.9947** | 0.9968 |
| Random Forest | **0.9968** | 0.9894 | **1.0000** | **0.9947** | **0.9999** |
| Simple Linear Regression | 0.6991 | 0.0000 | 0.0000 | 0.0000 | 0.6570 |
| Multiple Linear Regression | 0.9625 | 0.9227 | 0.9555 | 0.9388 | 0.9972 |
| Logistic Regression | 0.9963 | 0.9894 | 0.9982 | 0.9938 | 0.9998 |
| KNN | 0.9764 | 0.9481 | 0.9751 | 0.9614 | 0.9905 |
| SVM | 0.9930 | 0.9791 | 0.9982 | 0.9885 | 0.9999 |
| **LSTM** | 0.9845 | 0.9734 | 0.9751 | 0.9742 | 0.9977 |
| **GRU** | 0.9234 | 0.8238 | 0.9484 | 0.8817 | 0.9725 |

### Key Observations — Traditional ML

1. **Best overall performers**: Decision Trees and Random Forest tie at 99.68% accuracy. Random Forest has perfect recall (1.0000) — it found every single TV Show.

2. **Simple Linear Regression fails**: With only `release_year` as input, it cannot distinguish movies from TV shows. It predicts "Movie" for everything, resulting in 0% precision, recall, and F1 for the TV Show class.

3. **Multiple Linear Regression improves dramatically**: Using all 13 features instead of 1 jumps accuracy from 69.9% to 96.3%. This demonstrates the value of multiple features.

4. **Logistic Regression nearly matches Random Forest**: At 99.63% accuracy, it proves this problem has a nearly linear decision boundary when all features are available.

5. **Why are scores so high?** The `duration_num` feature is almost perfectly predictive — movies are measured in minutes (60-300) while TV shows are measured in seasons (1-15). This creates a very clear separation that most algorithms exploit easily.

6. **ROC-AUC scores**: Random Forest, Logistic Regression, and SVM all achieve ROC-AUC ≥ 0.9998, meaning they almost perfectly rank TV Shows higher than Movies in their probability outputs.

### Key Observations — ANN Models (Homework 2)

7. **LSTM achieves strong results** (Accuracy = 0.9845), placing between KNN and SVM in the overall ranking. Its 3-gate mechanism allows fine-grained control over which features are most informative, converging over 22 epochs.

8. **GRU underperforms** (Accuracy = 0.9234), stopping very early at epoch 5. Its simpler 2-gate architecture converged too quickly to a suboptimal solution. The validation loss plateaued before the model could learn the full complexity of feature interactions.

9. **Traditional ML beats ANN on tabular data**: All traditional ML models (except Simple Linear Regression) outperform GRU, and most outperform LSTM. This is expected — RNNs are designed for sequential data where temporal order matters. Our pseudo-sequence reshaping `(13, 1)` doesn't provide meaningful sequential relationships between features.

10. **LSTM > GRU**: LSTM's additional forget gate gives it more capacity to selectively retain or discard feature information, resulting in 6.1 percentage points higher accuracy than GRU.

---

## 10. Plots Explanation

### Plot 1: Confusion Matrices (2×5 grid for 10 models)

A confusion matrix is a 2×2 grid showing:

```
                Predicted Movie    Predicted TV Show
Actual Movie    True Negative (TN)  False Positive (FP)
Actual TV Show  False Negative (FN) True Positive (TP)
```

- **Top-left (TN)**: Movie correctly predicted as Movie
- **Top-right (FP)**: Movie incorrectly predicted as TV Show (false alarm)
- **Bottom-left (FN)**: TV Show incorrectly predicted as Movie (missed)
- **Bottom-right (TP)**: TV Show correctly predicted as TV Show

**How to read**: The diagonal (top-left to bottom-right) shows correct predictions. Higher numbers on the diagonal = better model. Numbers off the diagonal are errors. The LSTM and GRU matrices show slightly more off-diagonal errors compared to the top-performing traditional ML models.

### Plot 2: ROC Curves (10 models)

The ROC (Receiver Operating Characteristic) curve plots:
- **X-axis**: False Positive Rate (FPR) — how many movies are wrongly called TV Shows
- **Y-axis**: True Positive Rate (TPR/Recall) — how many TV Shows are correctly identified

**How to read**: 
- A curve hugging the **top-left corner** = excellent model (high TPR, low FPR)
- A curve along the **diagonal** = random guessing
- The **area under the curve** (AUC) quantifies this — closer to 1.0 is better

In our ROC plot, Simple Linear Regression's curve is far from the top-left (AUC=0.657), LSTM and GRU are close to the top-left (AUC > 0.97), and the best traditional ML models hug the corner (AUC > 0.99).

### Plot 3: Metrics Bar Chart (10 models)

A grouped bar chart comparing all 5 metrics across all 10 models side by side.

**How to read**: Taller bars = better performance. Look for models where all 5 bars are consistently tall (Random Forest, Decision Trees, Logistic Regression) versus models with short bars (Simple Linear Regression). LSTM shows bars comparable to KNN/SVM, while GRU bars are shorter, reflecting its lower performance.

### Plot 4: ANN Training History (Homework 2)

A 2×2 grid showing training curves for LSTM (left column) and GRU (right column):

- **Top row (Loss)**: Training loss (solid line) and validation loss (dashed line) over epochs. Both should decrease over time. If validation loss starts increasing while training loss continues to decrease, the model is **overfitting**.
- **Bottom row (Accuracy)**: Training accuracy (solid) and validation accuracy (dashed) over epochs. Both should increase and converge.

**How to read**:
- **LSTM** shows gradual convergence over ~22 epochs with training and validation curves tracking closely — good generalization.
- **GRU** stops very early at epoch 5, with validation loss plateauing quickly — the simpler architecture couldn't improve further on this data.

---

## 11. Libraries Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.2.1 | Data loading and manipulation (DataFrames) |
| `numpy` | 1.26.4 | Numerical operations, array handling |
| `scikit-learn` | 1.8.0 | All ML algorithms, metrics, preprocessing |
| `matplotlib` | 3.10.8 | Base plotting library |
| `seaborn` | 0.13.2 | Statistical visualizations (built on matplotlib) |
| `tensorflow` | 2.21.0 | Deep learning framework (LSTM, GRU models) |
| `keras` | 3.14.1 | High-level neural network API (included with TensorFlow) |

### scikit-learn Components Used

| Component | Import Path | Purpose |
|---|---|---|
| `train_test_split` | `sklearn.model_selection` | Splitting data into train and test sets |
| `LabelEncoder` | `sklearn.preprocessing` | Converting categorical strings to numbers |
| `StandardScaler` | `sklearn.preprocessing` | Normalizing features to mean=0, std=1 |
| `GaussianNB` | `sklearn.naive_bayes` | Naive Bayes classifier |
| `DecisionTreeClassifier` | `sklearn.tree` | Decision Tree classifier |
| `RandomForestClassifier` | `sklearn.ensemble` | Random Forest classifier |
| `LinearRegression` | `sklearn.linear_model` | Linear Regression (Simple & Multiple) |
| `LogisticRegression` | `sklearn.linear_model` | Logistic Regression classifier |
| `KNeighborsClassifier` | `sklearn.neighbors` | KNN classifier |
| `SVC` | `sklearn.svm` | Support Vector Machine classifier |
| `accuracy_score` | `sklearn.metrics` | Calculate accuracy |
| `precision_score` | `sklearn.metrics` | Calculate precision |
| `recall_score` | `sklearn.metrics` | Calculate recall |
| `f1_score` | `sklearn.metrics` | Calculate F1-score |
| `roc_auc_score` | `sklearn.metrics` | Calculate ROC-AUC |
| `confusion_matrix` | `sklearn.metrics` | Generate confusion matrix |
| `roc_curve` | `sklearn.metrics` | Generate ROC curve data points |

### TensorFlow/Keras Components Used (Homework 2)

| Component | Import Path | Purpose |
|---|---|---|
| `Sequential` | `tensorflow.keras.models` | Sequential model container |
| `LSTM` | `tensorflow.keras.layers` | Long Short-Term Memory RNN layer |
| `GRU` | `tensorflow.keras.layers` | Gated Recurrent Unit RNN layer |
| `Dense` | `tensorflow.keras.layers` | Fully connected layer |
| `Dropout` | `tensorflow.keras.layers` | Regularization layer (random neuron deactivation) |
| `EarlyStopping` | `tensorflow.keras.callbacks` | Stop training when validation loss plateaus |
| `Adam` | `tensorflow.keras.optimizers` | Adaptive Moment Estimation optimizer |
