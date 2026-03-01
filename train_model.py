# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_csv(r"C:\Users\cgc\Desktop\hypertension_prediction\patient_data.csv")

# ---------------------------
# Convert range strings to numeric
# ---------------------------
def convert_range_to_float(val):
    """Convert 'min-max' string to midpoint float, or numeric string to float"""
    if isinstance(val, str) and '-' in val:
        parts = val.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except:
            return np.nan
    else:
        try:
            return float(val)
        except:
            return np.nan

# Convert relevant numeric columns
for col in ['Age','Systolic','Diastolic']:
    df[col] = df[col].apply(convert_range_to_float)

# ---------------------------
# Fill missing values
# ---------------------------
df.fillna(method='ffill', inplace=True)

# ---------------------------
# Features and targets
# ---------------------------
feature_cols = ['C','Age','History','Patient','TakeMedication','BreathShortness',
                'VisualChanges','NoseBleeding','Whendiagnoused','Systolic',
                'Diastolic','ControlledDiet']

X = df[feature_cols]
y_sev = df['Severity']
y_stage = df['Stages']

# ---------------------------
# Encode categorical features
# ---------------------------
categorical_cols = ['C','History','Patient','TakeMedication','BreathShortness',
                    'VisualChanges','NoseBleeding','Whendiagnoused','ControlledDiet']

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

# Replace categorical columns with encoded array
X_numeric = X.drop(columns=categorical_cols).to_numpy()
X_final = np.hstack([X_numeric, X_encoded])

# ---------------------------
# Train-test split (optional)
# ---------------------------
X_train, X_test, y_train_sev, y_test_sev, y_train_stage, y_test_stage = train_test_split(
    X_final, y_sev, y_stage, test_size=0.2, random_state=42
)

# ---------------------------
# Train models
# ---------------------------
severity_model = RandomForestClassifier(n_estimators=100, random_state=42)
stages_model = RandomForestClassifier(n_estimators=100, random_state=42)

severity_model.fit(X_train, y_train_sev)
stages_model.fit(X_train, y_train_stage)

# ---------------------------
# Save models and encoder
# ---------------------------
with open(r"C:\Users\cgc\Desktop\hypertension_prediction\severity_model.pkl", "wb") as f:
    pickle.dump(severity_model, f)

with open(r"C:\Users\cgc\Desktop\hypertension_prediction\stages_model.pkl", "wb") as f:
    pickle.dump(stages_model, f)

with open(r"C:\Users\cgc\Desktop\hypertension_prediction\encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("Models and encoder trained and saved successfully!")