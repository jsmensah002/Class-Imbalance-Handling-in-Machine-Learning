# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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

# --- SCALING (LogReg and SVC only) ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_resampled)
x_test_scaled  = scaler.transform(x_test)

# --- STEP 2: CLASS WEIGHTING ---
majority_count = y_train.value_counts()[0]
minority_count = y_train.value_counts()[1]
scale = majority_count / minority_count

# --- MODELS: scaled vs unscaled ---
scaled_models = {
    'LogReg': (LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000), x_train_scaled, x_test_scaled),
    'SVC':    (SVC(class_weight='balanced', random_state=42, probability=True), x_train_scaled, x_test_scaled),
}

unscaled_models = {
    'RFC':    (RandomForestClassifier(class_weight='balanced', random_state=42), x_train_resampled, x_test),
    'XGB':    (XGBClassifier(scale_pos_weight=scale, random_state=42), x_train_resampled, x_test),
}

all_models = {**scaled_models, **unscaled_models}

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
best_thresholds = {} 
best_preds  = {}
best_probas = {}

for name, (model, x_tr, x_te) in all_models.items():
    model.fit(x_tr, y_train_resampled)
    y_proba = model.predict_proba(x_te)[:, 1]

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
    print(f"{name} Test Score:  {model.score(x_te, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")  

