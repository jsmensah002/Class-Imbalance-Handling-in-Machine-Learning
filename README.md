Brief Overview:
- This project explores how class imbalance handling scaling, and threshold tuning collectively impact model performance with eight experimental setups:
- Not Optimized: Models trained on raw data without any hyperparameter tuning.
- Not Optimized but Scaled: Models trained on scaled features to observe the effect of normalization.
- Not Optimized Class Balanced: Models with SMOTE and threshold tuning applied to handle class imbalance.
- Not Optimized but Scaled Class Balanced: Scaled models with class imbalance handling applied.
- Optimized with Outliers Present: Models tuned with hyperparameter optimization, keeping all data including outliers.
- Optimized with Outliers Present Class Balanced: Optimized models with class imbalance handling applied.
- Optimized with Outliers Removed: Models tuned after removing outliers, showing the impact of noise reduction on performance.
- Optimized with Outliers Removed Class Balanced: Outlier removed models with class imbalance handling applied.

The full results compilation across all eight experiments is available in the TitanicClassificationReport Excel file.

Method:
- Studied correlation between inputs and output to validate feature relevance.
- Checked class distribution to determine whether imbalance handling was necessary.
- Applied SMOTE for oversampling and threshold tuning to address the 62% / 38% class split.
- Observed how scaling and class balancing affect each model differently.
- Models used were Logistic Regression (LogReg), Support Vector Classification (SVC), Random Forest Classifier (RFC), and XGBoost (XGB).

Model Selection:
- Across all eight experiments with four models per experiment, a total of 32 model results were compiled and assessed in the TitanicClassificationReport Excel file.
- The selection process followed a two stage approach. The first and most critical criterion was the train-test gap, ranked in ascending order. Avoiding overfitting was treated as the highest priority because a model that cannot generalise to unseen data has no real world value regardless of how well it performs on training data.
- Once models were filtered by generalisation stability, the second criterion was overall predictive performance across precision, recall, F1 score, and AUC-ROC. From this analysis, SVC from the 'Not Optimized but Scaled Class Balanced' experiment emerged as the clear winner. It not only recorded the fourth lowest train-test gap of 0.0003 across all 32 results, but also delivered the strongest balanced performance across every metric. 

Overall Best Model: Support Vector Classification (SVC) — 'Not Optimized but Scaled Class Balanced' Experiment :
- Train Score: 0.8205
- Test Score: 0.8202
- Train-Test Gap: 0.0003
- Precision: 0.8213
- Recall: 0.8073
- F1 Score:	0.8128
- AUC_ROC: 0.8631




