# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

df = pd.read_csv('Titanic.csv')
print(df)

print(df.isna().sum())
df = df.drop(columns=['deck'])
df = df.dropna(subset=['embarked', 'embark_town'])
df['age'] = df['age'].fillna(df['age'].median())

df['adult_male'] = df['adult_male'].astype(str).apply(lambda x: 1 if x == 'True' else 0)
df['alone']      = df['alone'].astype(str).apply(lambda x: 1 if x == 'True' else 0)

label_encoders = {}
for col in ['sex', 'class', 'who', 'embark_town']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

numerical   = df[['age', 'adult_male', 'alone', 'fare']]
categorical = df[['sex', 'class', 'who', 'embark_town']]
x = pd.concat([numerical, categorical], axis='columns')
y = df['survived']

print(x)
print(y)

# --- CLASS IMBALANCE CHECK ---
print("Class Distribution:")
print(df['survived'].value_counts())
print(df['survived'].value_counts(normalize=True) * 100)

# --- TRAIN TEST SPLIT ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- GRIDSEARCH ---
# LogReg and SVC: scaling inside Pipeline
# RFC and XGB: no scaling needed

# Logistic Regression (scaling inside pipeline)
log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000))
])
log_grid = GridSearchCV(
    log_pipe,
    {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__solver": ["lbfgs"],
        "logreg__class_weight": [None, 'balanced']
    },
    cv=5, scoring="accuracy", n_jobs=1
)
log_grid.fit(x_train, y_train)
print("Logistic Regression best params:", log_grid.best_params_)
best_logreg = log_grid.best_estimator_

# SVC (scaling inside pipeline)
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True))
])
svc_grid = GridSearchCV(
    svc_pipe,
    {
        "svc__kernel": ["linear", "rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
svc_grid.fit(x_train, y_train)
print("SVC best params:", svc_grid.best_params_)
best_svc = svc_grid.best_estimator_

# Random Forest (no scaling)
rfc_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
rfc_grid.fit(x_train, y_train)
print("Random Forest best params:", rfc_grid.best_params_)
best_rfc = rfc_grid.best_estimator_

# XGBoost (no scaling)
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42),
    {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
xgb_grid.fit(x_train, y_train)
print("XGBoost best params:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_

# --- EVALUATION ---
models = {
    'LogReg': best_logreg,
    'SVC':    best_svc,
    'RFC':    best_rfc,
    'XGB':    best_xgb
}

best_preds  = {}
best_probas = {}

for name, model in models.items():
    y_pred  = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    best_preds[name]  = y_pred
    best_probas[name] = y_proba

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"{name} Train Score: {model.score(x_train, y_train):.4f}")
    print(f"{name} Test Score:  {model.score(x_test, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")  

# ============================================================
# RESULTS EXPORTER — edit only the values in this section
# ============================================================
RESULTS_FILE = 'TitanicClassificationReport.xlsx'        # change 
EXPERIMENT   = 'Not Optimized with Outliers Present'      # change 

# QUICK REFERENCE — X_TRAIN and X_TEST per experiment:
# -------------------------------------------------------
# Not optimized                       → x_train            | x_test
# Not Optimized but scaled            → x_train            | x_test
# Optimized outliers present          → x_train            | x_test
# Optimized outliers absent           → x_train            | x_test_filled
# Imbalance handled not optimized     → x_train_resampled  | x_test
# Imbalance handled scaled            → x_train_resampled  | x_test
# Imbalance handled outliers present  → x_train_resampled  | x_test
# Imbalance handled outliers absent   → x_train_resampled  | x_test_filled
# -------------------------------------------------------
X_TRAIN = x_train                       # x_train_resampled if imbalance handled, # x_train_filled if outliers removed and class not balanced
Y_TRAIN = y_train                       # y_train_resampled if imbalance handled
X_TEST  = x_test                        # x_test_filled if outliers removed
# ============================================================

import os
from sklearn.metrics import precision_score, recall_score, f1_score

results = []
for name, model in models.items():  #if there are brackets () in the model dict, replace model with (model, x_tr)
    y_pred  = best_preds[name]
    y_proba = best_probas[name]

    results.append({
        'Experiment':  EXPERIMENT,
        'Model':       name,
        'Train Score': round(model.score(X_TRAIN, Y_TRAIN), 4),
        'Test Score':  round(model.score(X_TEST, y_test), 4),
        'Precision':   round(precision_score(y_test, y_pred, average='macro'), 4),
        'Recall':      round(recall_score(y_test, y_pred, average='macro'), 4),
        'F1':          round(f1_score(y_test, y_pred, average='macro'), 4),

        #for multiclass, auc-roc becomes 'AUC-ROC': round(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'), 4)
        'AUC-ROC':     round(roc_auc_score(y_test, y_proba), 4)
    })

new_df = pd.DataFrame(results)
if os.path.exists(RESULTS_FILE):
    existing = pd.read_excel(RESULTS_FILE)
    final_df = pd.concat([existing, new_df], ignore_index=True)
else:
    final_df = new_df

final_df.to_excel(RESULTS_FILE, index=False)
print(f"\nResults saved to {RESULTS_FILE}")