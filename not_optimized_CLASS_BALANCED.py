# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- LOAD AND PREPROCESS ---
df = pd.read_csv('Titanic.csv')

df = df.drop(columns=['deck'])
df = df.dropna(subset=['embarked', 'embark_town'])
df['age'] = df['age'].fillna(df['age'].median())

df['adult_male'] = df['adult_male'].astype(str).apply(lambda x: 1 if x == 'True' else 0)
df['alone']      = df['alone'].astype(str).apply(lambda x: 1 if x == 'True' else 0)

for col in ['sex', 'class', 'who', 'embark_town']:
    df[col] = LabelEncoder().fit_transform(df[col])

numerical   = df[['age', 'adult_male', 'alone', 'fare']]
categorical = df[['sex', 'class', 'who', 'embark_town']]
x = pd.concat([numerical, categorical], axis='columns')
y = df['survived']

# --- CLASS IMBALANCE CHECK ---
print("Class Distribution:")
print(df['survived'].value_counts())
print(df['survived'].value_counts(normalize=True) * 100)

# TRAIN TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# STEP 1: SMOTE
print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
sm = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = sm.fit_resample(x_train, y_train)
print(f"After SMOTE:  {pd.Series(y_train_resampled).value_counts().to_dict()}")

# STEP 2: CLASS WEIGHTING + STEP 3: THRESHOLD TUNING
majority_count = y_train.value_counts()[0]
minority_count = y_train.value_counts()[1]
scale = majority_count / minority_count

models = {
    'LogReg': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'SVC':    SVC(class_weight='balanced', random_state=42, probability=True),
    'RFC':    RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGB':    XGBClassifier(scale_pos_weight=scale, random_state=42)
}

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
best_thresholds = {} 
best_preds  = {}
best_probas = {}

for name, model in models.items():
    model.fit(x_train_resampled, y_train_resampled)
    y_proba = model.predict_proba(x_test)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    for threshold in thresholds:
        y_pred_temp = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    best_thresholds[name] = best_threshold
    y_pred = (y_proba >= best_threshold).astype(int)
    best_preds[name]  = y_pred
    best_probas[name] = y_proba

    print(f"\nBest Threshold: {best_threshold}")
    y_pred = (y_proba >= best_threshold).astype(int)

    print(f"{name} Train Score: {model.score(x_train_resampled, y_train_resampled):.4f}")
    print(f"{name} Test Score:  {model.score(x_test, y_test):.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")  

