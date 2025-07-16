import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv('data/dataset_med.csv')

if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)

df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')
df['treatment_duration'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df.drop(columns=['diagnosis_date', 'end_treatment_date'], inplace=True)

binary_cols = ['gender', 'family_history', 'hypertension', 'asthma', 'cirrhosis', 'other_cancer']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0, 'male': 1, 'female': 0})

one_hot_cols = ['country', 'cancer_stage', 'treatment_type', 'smoking_status']
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=True)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(model, 'models/lung_survival_model.pkl')
print("\n✅ Model saved to 'models/lung_survival_model.pkl'")

random_sample = X_test.sample(1, random_state=42)
prediction = model.predict(random_sample)[0]
print("Survived ✅" if prediction == 1 else "Did not survive ❌")