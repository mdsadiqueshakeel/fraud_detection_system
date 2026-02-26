# Credit Card Fraud Detection

## Project Overview

This project implements a machine learning solution to detect fraudulent credit card transactions. Using logistic regression with standardized features, the model achieves a ROC-AUC score of **0.972**, effectively identifying fraudulent transactions while maintaining a balance between precision and recall.

The solution includes threshold optimization to minimize business losses by analyzing the trade-off between fraud detection costs and false alarm costs.

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
