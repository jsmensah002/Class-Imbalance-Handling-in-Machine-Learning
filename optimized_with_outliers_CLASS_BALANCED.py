# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- LOAD AND PREPROCESS ---
df = pd.read_csv('Titanic.csv')

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

# --- STEP 1: SMOTE (before scaling) ---
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
sm = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)
print(f"After SMOTE:  {pd.Series(y_train_resampled).value_counts().to_dict()}")

# --- STEP 2: CLASS WEIGHTING ---
majority_count = y_train.value_counts()[0]
minority_count = y_train.value_counts()[1]
scale = majority_count / minority_count

# --- GRIDSEARCH ---
# LogReg and SVC: scaling handled inside Pipeline
# RFC and XGB: no scaling needed, use resampled data directly

# Logistic Regression (scaling inside pipeline)
log_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, class_weight='balanced'))
])
log_grid = GridSearchCV(
    log_pipe,
    {
        "logreg__C": [0.01, 0.1, 1, 10],
        "logreg__solver": ["lbfgs"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
log_grid.fit(x_train_resampled, y_train_resampled)
print("\nLogistic Regression best params:", log_grid.best_params_)
best_logreg = log_grid.best_estimator_

# SVC (scaling inside pipeline)
svc_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(class_weight='balanced', probability=True))
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
svc_grid.fit(x_train_resampled, y_train_resampled)
print("SVC best params:", svc_grid.best_params_)
best_svc = svc_grid.best_estimator_

# Random Forest (no scaling)
rfc_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    {
        "n_estimators": [100, 300],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt"]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
rfc_grid.fit(x_train_resampled, y_train_resampled)
print("Random Forest best params:", rfc_grid.best_params_)
best_rfc = rfc_grid.best_estimator_

# XGBoost (no scaling)
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, scale_pos_weight=scale),
    {
        "n_estimators": [100, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0]
    },
    cv=5, scoring="accuracy", n_jobs=1
)
xgb_grid.fit(x_train_resampled, y_train_resampled)
print("XGBoost best params:", xgb_grid.best_params_)
best_xgb = xgb_grid.best_estimator_

# --- STEP 3: THRESHOLD TUNING ---
# LogReg and SVC: pass x_test directly, pipeline scales internally
# RFC and XGB: pass x_test directly, no scaling needed
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
best_thresholds = {} 
best_preds  = {}
best_probas = {}

models = {
    'LogReg': (best_logreg, x_train_resampled),
    'SVC':    (best_svc,    x_train_resampled),
    'RFC':    (best_rfc,    x_train_resampled),
    'XGB':    (best_xgb,    x_train_resampled)
}

for name, (model, x_tr) in models.items():
    y_proba = model.predict_proba(x_test)[:, 1]

    print(f"\n{'='*55}")
    print(f"  {name} — Threshold Analysis")
    print(f"{'='*55}")

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        p  = precision_score(y_test, y_pred_temp)
        r  = recall_score(y_test, y_pred_temp)
        f1 = f1_score(y_test, y_pred_temp)
        print(f"Threshold: {threshold} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    best_thresholds[name] = best_threshold

    print(f"\nBest Threshold: {best_threshold}")
    y_pred = (y_proba >= best_threshold).astype(int)
    best_preds[name]  = y_pred
    best_probas[name] = y_proba

    print(f"{name} Train Score: {model.score(x_tr, y_train_resampled):.4f}")
    print(f"{name} Test Score:  {model.score(x_test, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")  

