# Always check for class imbalance and confusion matrix!!!!!

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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

print(df.select_dtypes(include='number').corr()['survived'].sort_values(ascending=False))

# --- TRAIN TEST SPLIT ---
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- SCALING (LogReg and SVC only) ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

# --- MODELS ---
scaled_models = {
    'LogReg': LogisticRegression(random_state=42, max_iter=1000),
    'SVC':    SVC(random_state=42, probability=True),
}

unscaled_models = {
    'RFC':    RandomForestClassifier(random_state=42),
    'XGB':    XGBClassifier(random_state=42)
}

all_models = {**scaled_models, **unscaled_models}

best_preds  = {}
best_probas = {}

for name, model in all_models.items():
    if name in scaled_models:
        model.fit(x_train_scaled, y_train)
        y_pred  = model.predict(x_test_scaled)
        y_proba = model.predict_proba(x_test_scaled)[:, 1]
        train_score = model.score(x_train_scaled, y_train)
        test_score  = model.score(x_test_scaled, y_test)
    else:
        model.fit(x_train, y_train)
        y_pred  = model.predict(x_test)
        y_proba = model.predict_proba(x_test)[:, 1]
        train_score = model.score(x_train, y_train)
        test_score  = model.score(x_test, y_test)

    best_preds[name]  = y_pred  
    best_probas[name] = y_proba  

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"{name} Train Score: {train_score:.4f}")
    print(f"{name} Test Score:  {test_score:.4f}")
    print(f"\n{name} Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"{name} AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")  

