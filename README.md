# Customer Lifetime Value: Segmentation, Revenue Forecasting & Churn Prediction

An end-to-end customer analytics pipeline built on transactional retail data (~4,300 customers, ~408K transactions). The project moves from raw data cleaning through unsupervised segmentation to two supervised models that predict future customer value and churn risk.

---

## Overview

Online retailers rarely treat every customer the same — but knowing who to prioritize requires more than intuition. This project answers three practical questions using only a customer's first 6 months of activity:

*   **Who are our customers, structurally?** → RFM-based clustering
*   **How much revenue will each customer generate in the next 6 months?** → Regression
*   **Will each customer come back at all?** → Classification

The three questions are deliberately split into separate models rather than one black box, so each can be evaluated (and acted on) independently — a segment label for marketing, a revenue number for forecasting, a churn probability for retention campaigns.

---

## Pipeline

*   **Phase 0** → Data Cleaning
*   **Phase 1** → RFM Clustering (customer segmentation)
*   **Phase 2** → Revenue Regression (predict next-6-month spend)
*   **Phase 3** → Churn Classification (predict next-6-month inactivity)

Each phase is a standalone notebook that reads the previous phase's output and writes its own artifact (cleaned data, trained model, or feature table) to disk.

| Notebook | Input | Output |
| :--- | :--- | :--- |
| `Phase_0_Data_Cleaning.ipynb` | raw transaction data | `df_success.csv` (validated, successful orders only) |
| `Phase_1_Clustering.ipynb` | `df_success.csv` | `customer_segmented.csv`, `clustering_model.pkl` |
| `Phase_2_Money_Regressor.ipynb` | `df_success.csv`, `customer.csv` | `regressor_model.pkl` |
| `Phase_3_Churn_Classification.ipynb` | `df_success.csv`, `customer.csv` | `churn_model.pkl`, `feature_importance_churn.csv` |

---

## Methodology

### Phase 0 — Data Cleaning
*   Parsed invoice dates, normalized types, dropped rows with missing Customer ID.
*   Filtered out cancelled/returned orders (negative-quantity and credit-note invoices) to isolate successfully fulfilled orders.
*   Derived `Amount = Quantity × Price` as the base monetary field used downstream.

### Phase 1 — RFM Clustering
*   Computed Recency, Frequency, Monetary (RFM) per customer.
*   Applied `log1p` transform to reduce skew/outlier influence before scaling (`StandardScaler`).
*   Selected K using four independent metrics — WCSS (elbow), Silhouette, Davies-Bouldin, and Calinski-Harabasz — rather than relying on a single heuristic.
*   Fit K-Means with K = 3, then translated statistical clusters into business-readable segments:

| Segment | Share | Recency (median) | Monetary (median) | Frequency (median) |
| :--- | :--- | :--- | :--- | :--- |
| **The Elite** | 19.1% | 10 days | $3,446 | 9 |
| **The Loyalists** | 41.0% | 38 days | $977 | 3 |
| **Need Attention** | 39.9% | 157 days | $262 | 1 |

*   Visualized cluster separation with PCA (2D projection).

### Phase 2 — Revenue Regression
*   **Framing:** using only each customer's first 6 months of transactions, predict total spend in the following 6 months (including customers who spend $0, i.e. churn).
*   **Engineered features:** recency, monetary, frequency, average order value, days between purchases, product diversity, and country-level spending benchmarks.
*   Applied `log1p` to the (heavily skewed) revenue target, trained Linear Regression / Random Forest / Gradient Boosting, then inverse-transformed predictions for evaluation.
*   Split train/test stratified on whether the customer churned (target = 0), so both sets preserve the same proportion of zero-revenue customers.

> **Best model: Random Forest**

| Metric | Score |
| :--- | :--- |
| **R²** | 0.77 |
| **MAE** | $870.68 |
| **RMSE** | $2,298.64 |

### Phase 3 — Churn Classification
*   **Target:** did the customer make zero purchases in the second 6-month window? (Churn rate in this dataset: 29.4%.)
*   Built a richer feature set than Phase 2 (13+ engineered features: purchase cadence statistics, spend variability, product diversity, etc.), all derived strictly from the first-6-month window.
*   Addressed class imbalance with `class_weight='balanced'` and stratified 5-fold CV.
*   Compared Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting on Accuracy / Precision / Recall / F1 / ROC-AUC.
*   Selected the operating threshold from the ROC curve (maximizing TPR − FPR) rather than defaulting to 0.5.

> **Best model: Logistic Regression**

| Metric | Score |
| :--- | :--- |
| **F1-Score** | 0.556 |
| **ROC-AUC** | 0.735 |
| **Recall** | 0.802 |
| **Precision** | 0.425 |

*Note: Recall was prioritized over precision by design — in a retention context, the cost of missing an about-to-churn customer is generally higher than the cost of a wasted retention email.*

---

## Key Findings
*   **Roughly 1 in 5 customers ("The Elite")** drives a disproportionate share of revenue, with recent, frequent, high-value purchases — a natural VIP/retention-priority target.
*   **Revenue in the next 6 months is reasonably predictable** from first-6-month behavior alone (R² 0.77), meaning early-lifecycle signals carry real forward-looking value.
*   **Nearly 30% of customers go fully inactive within a year.** The churn model catches ~80% of eventual churners at the chosen threshold, at the cost of flagging some active customers too — an intentional precision/recall trade-off for a retention use case.

---

## Tech Stack
`Python` · `pandas` / `numpy` · `scikit-learn` · `matplotlib` / `seaborn` · `joblib`

---

## Known Limitations / Next Steps
Being upfront about what this version doesn't yet handle well:
*   Country-level features in Phase 2 currently use full-period statistics rather than being recomputed strictly from the first-6-month window — a leakage risk being addressed in the next iteration by deriving country benchmarks only from the training window.
*   Linear Regression is included as a baseline but performs very poorly on this target (extremely skewed revenue distribution); tree-based models were far more robust and are used for all reported results.
*   Segmentation logic (Phase 1) is currently tuned for K=3; extending to a variable number of segments would need a more general labeling rule.
*   No hyperparameter tuning (GridSearch/Optuna) yet — current models use reasonable defaults, not optimized configurations.

---

## Repository Structure
```text
├── Phase_0_Data_Cleaning.ipynb
├── Phase_1_Clustering.ipynb
├── Phase_2_Money_Regressor.ipynb
├── Phase_3_Churn_Classification.ipynb
├── data/
│   ├── customer.csv
│   ├── country.csv
│   └── df_success.csv          (generated by Phase 0)
└── models/
    ├── clustering_model.pkl    (generated by Phase 1)
    ├── regressor_model.pkl     (generated by Phase 2)
    └── churn_model.pkl         (generated by Phase 3)
