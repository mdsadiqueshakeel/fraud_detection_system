# Credit Card Fraud Detection

## Project Overview

This project implements a machine learning solution to detect fraudulent credit card transactions. It begins with a logistic regression pipeline that achieves a ROC-AUC score of **0.972** on the held‑out test set, and further explores more powerful learners (Random Forest and XGBoost) which, once tuned, push performance above **0.978**.

All models support threshold optimization to minimize business losses by analyzing the trade‑off between fraud detection costs and false alarm costs.

## Problem Statement

Credit card fraud is a significant challenge for financial institutions, resulting in substantial financial losses. The project addresses the following objectives:

- **Detect fraudulent transactions** with high recall to minimize undetected fraud
- **Balance false positives** to reduce customer inconvenience
- **Optimize decision thresholds** based on business costs (fraud cost: ₹10,000 vs false alarm cost: ₹50)
- **Handle class imbalance** inherent in fraud detection datasets where fraud cases are rare

## Dataset Source

- **Dataset**: Credit Card Fraud Detection Dataset
- **File**: `data/creditcard.csv`
- **Format**: CSV with 30 features and 1 target variable (Class)
- **Class Distribution**: Highly imbalanced - majority legitimate transactions (Class 0) with rare fraud cases (Class 1)
- **Train-Test Split**: 80-20 with stratification to maintain fraud ratio across splits

## How to Run Project

### Prerequisites
Ensure you have the following Python packages installed:
- pandas
- scikit-learn
- joblib
- numpy

### Installation
```bash
pip install -r requirment.txt
```

### Training the Model
Navigate to the `src` directory and run:
```bash
python train.py
```
This will:
1. Load the credit card dataset from `data/creditcard.csv`
2. Split data into training (80%) and testing (20%) sets
3. Train a logistic regression model with standardized features
4. Save the trained pipeline to `models/logistic_model.pkl`

### Evaluating the Model
Run the evaluation script:
```bash
python evalute.py
```
This will:
1. Load the trained model
2. Generate confusion matrix and classification report
3. Calculate ROC-AUC score
4. Display threshold tuning analysis with business loss calculations

### Hyperparameter Tuning & Comparison
To try stronger models, two tuning scripts are provided:
```bash
python tune_rf.py          # searches Random Forest hyperparameters
python tune_xgb_clean.py   # searches XGBoost hyperparameters
```
Each script saves the best estimator under `models/`.
After tuning, compare all three algorithms using:
```bash
python final_model_comparison.py
```
which prints ROC‑AUC, optimal threshold, and resulting business loss for every candidate.

## Metrics Achieved

### Confusion Matrix
```
                Predicted: Legitimate    Predicted: Fraud
Actual: Legitimate    55,478              1,386
Actual: Fraud         8                   90
```

### Classification Report (Default Threshold: 0.5)
```
              Precision    Recall    F1-Score   Support
Class 0           1.00      0.98       0.99     56,864
Class 1           0.06      0.92       0.11        98

Accuracy                                0.98     56,962
Macro Avg         0.53      0.95       0.55     56,962
Weighted Avg      1.00      0.98       0.99     56,962
```

### Overall Performance
- **ROC-AUC Score**: 0.972 (Excellent)
- **Accuracy**: 98.0%
- **Fraud Recall**: 92% (catches 92 out of 98 fraudulent transactions)

## Threshold Optimization Explanation

### Concept
The model's default probability threshold of 0.5 may not be optimal for fraud detection. By adjusting the decision threshold, we can control the trade-off between precision (false positive rate) and recall (fraud detection rate).

### Business Loss Analysis
Each decision has associated costs:
- **Fraud Cost (False Negatives)**: ₹10,000 per missed fraud
- **False Alarm Cost (False Positives)**: ₹50 per incorrectly flagged legitimate transaction

### Threshold Tuning Results
| Threshold | Precision | Recall | Business Loss |
|-----------|-----------|--------|---------------|
| 0.10      | 0.0082    | 0.9490 | ₹615,750     |
| 0.20      | 0.0164    | 0.9388 | ₹335,750     |
| 0.30      | 0.0273    | 0.9184 | ₹240,050     |
| 0.40      | 0.0422    | 0.9184 | ₹182,250     |
| 0.50      | 0.0610    | 0.9184 | ₹149,300     |
| 0.60      | 0.0867    | 0.9082 | ₹136,850     |
| 0.70      | 0.1213    | 0.9082 | ₹122,250     |
| 0.80      | 0.1596    | 0.8878 | ₹132,900     |
| 0.90      | 0.2472    | 0.8878 | ₹123,250     |

### Key Insights

1. **Optimal Threshold**: **0.70** minimizes business loss at **₹122,250**
   - Achieves 90.82% fraud recall while maintaining reasonable false positive control
   - Represents the best balance between catching fraud and minimizing customer friction

2. **Trade-off Analysis**:
   - Lower thresholds (0.10-0.30) catch more fraud but generate excessive false alarms
   - Higher thresholds (0.80-0.90) reduce false alarms but miss more fraud
   - Threshold 0.70 provides the optimal cost-benefit balance

3. **Business Impact**:
   - Using threshold 0.70 instead of default 0.5 reduces business loss from ₹149,300 to ₹122,250
   - Saves approximately ₹27,050 per test set cycle through optimized fraud-cost trade-off

---

**Note**: This analysis assumes fraud cost of ₹10,000 and false alarm cost of ₹50. Adjust thresholds based on your institution's specific cost-benefit parameters.

## Random Forest Evaluation

The project also includes a Random Forest classifier (`models/random_forest.pkl`). Evaluation on the test set produced the following results. *(these figures are for the **baseline** RF model before hyperparameter tuning; see the Model Comparison section for tuned performance)*

### Confusion Matrix
```
[[56861     3]
 [   25    73]]
```

### Classification Report (Default Threshold: 0.5)
```
           precision    recall  f1-score   support

         0       1.00      1.00      1.00     56,864
         1       0.96      0.74      0.84        98

   accuracy                           1.00     56,962
   macro avg       0.98      0.87      0.92     56,962
weighted avg       1.00      1.00      1.00     56,962
```

### ROC-AUC Score
- **ROC-AUC Score**: 0.9529

### Threshold Tuning Results (Random Forest)
| Threshold | Precision | Recall | Business Loss |
|-----------|-----------|--------|---------------|
| 0.10      | 0.7414    | 0.8776 | ₹121,500     |
| 0.20      | 0.8571    | 0.8571 | ₹140,700     |
| 0.30      | 0.9205    | 0.8265 | ₹170,350     |
| 0.40      | 0.9518    | 0.8061 | ₹190,200     |
| 0.50      | 0.9605    | 0.7449 | ₹250,150     |
| 0.60      | 0.9726    | 0.7245 | ₹270,100     |
| 0.70      | 0.9701    | 0.6633 | ₹330,100     |
| 0.80      | 0.9667    | 0.5918 | ₹400,100     |
| 0.90      | 0.9583    | 0.4694 | ₹520,100     |

### Insights
- The Random Forest achieves strong precision for fraud detection while maintaining high overall accuracy.
- The lowest business loss occurs at the lower thresholds (0.10), driven by higher recall; however, business preference for fewer false alarms may shift the chosen threshold.
- Compare Random Forest and Logistic Regression results to pick a model and threshold aligned with your risk and cost preferences.

## XGBoost Evaluation

An XGBoost classifier (`models/xgb_model.pkl`) was also trained and evaluated. Results on the test set are below. *(again, this is the un‑tuned baseline – tuning further improves the metrics as summarized later)*

### Confusion Matrix
```
[[56850    14]
 [   19    79]]
```

### Classification Report (Default Threshold: 0.5)
```
           precision    recall  f1-score   support

         0       1.00      1.00      1.00     56,864
         1       0.85      0.81      0.83        98

   accuracy                           1.00     56,962
   macro avg       0.92      0.90      0.91     56,962
weighted avg       1.00      1.00      1.00     56,962
```

### ROC-AUC Score
- **ROC-AUC Score**: 0.9771

### Threshold Tuning Results (XGBoost)
| Threshold | Precision | Recall | Business Loss |
|-----------|-----------|--------|---------------|
| 0.10      | 0.7736    | 0.8367 | ₹161,200     |
| 0.20      | 0.8039    | 0.8367 | ₹161,000     |
| 0.30      | 0.8283    | 0.8367 | ₹160,850     |
| 0.40      | 0.8351    | 0.8265 | ₹170,800     |
| 0.50      | 0.8495    | 0.8061 | ₹190,700     |
| 0.60      | 0.8587    | 0.8061 | ₹190,650     |
| 0.70      | 0.8778    | 0.8061 | ₹190,550     |
| 0.80      | 0.8876    | 0.8061 | ₹190,500     |
| 0.90      | 0.9186    | 0.8061 | ₹190,350     |

### Feature Importance (Top 10)
```
   Feature  Importance
14     V14    0.614015
4       V4    0.085196
10     V10    0.028916
12     V12    0.027020
8       V8    0.024901
1       V1    0.024063
13     V13    0.021440
7       V7    0.020446
3       V3    0.020325
16     V16    0.016684
```

### Insights
- XGBoost has the highest ROC-AUC (0.977) among the models.
- Best business-loss observed at threshold 0.30: ₹160,850—higher than both logistic and random forest minima, but still strong recall.
- Feature importance indicates V14 as the dominant predictor.
- SHAP plots (generated by `evalute_xgb.py`) can provide deeper explainability.

*** End Patch
## Model Comparison

**Logistic Regression (baseline)**
- ROC-AUC: 0.972
- Default (0.5) fraud recall: 92% (high recall), fraud precision: 0.06 (low)
- Best observed business-loss: ₹122,250 at threshold 0.70

**Random Forest (tuned)**
- ROC-AUC: **0.9785**
- Optimal decision threshold: **0.20**
- Minimum business-loss on test set: **₹107,000**
- Hyperparameter search uses `tune_rf.py` and results are saved as `models/rf_tuned_clean.pkl`.

**XGBoost (tuned)**
- ROC-AUC: **0.9776**
- Optimal decision threshold: **0.75**
- Minimum business-loss on test set: **₹118,850**
- Hyperparameter search uses `tune_xgb_clean.py` and results are saved as `models/xgb_tuned_clean.pkl`.

**Recommendation**
1. The tuned Random Forest now delivers the lowest business loss (₹107K) along with the highest ROC‑AUC, making it the leading candidate under the current cost assumptions.
2. XGBoost still offers excellent discriminative power and may be preferred when explainability (via SHAP) or slightly higher precision is desired.
3. Logistic Regression remains a reliable, interpretable baseline and may be adequate if simplicity is prioritized.

Choose the model and threshold that align with your operational priorities—loss minimization, recall, precision, or explainability. Consider running `final_model_comparison.py` frequently after making changes to data or costs to re‑evaluate.

If you'd like, I can add a short script to compute and plot business loss vs threshold for all models and include the plot in the README.
