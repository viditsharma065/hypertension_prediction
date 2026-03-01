# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# ---------------------------
# Load models and encoder
# ---------------------------
with open(r"C:\Users\cgc\Desktop\hypertension_prediction\severity_model.pkl", "rb") as f:
    severity_model = pickle.load(f)

with open(r"C:\Users\cgc\Desktop\hypertension_prediction\stages_model.pkl", "rb") as f:
    stages_model = pickle.load(f)

with open(r"C:\Users\cgc\Desktop\hypertension_prediction\encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

categorical_cols = ['C','History','Patient','TakeMedication','BreathShortness',
                    'VisualChanges','NoseBleeding','Whendiagnoused','ControlledDiet']

numeric_cols = ['Age','Systolic','Diastolic']

# ---------------------------
# Helper function
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

def prepare_input(form):
    """Prepare input features for prediction"""
    data = {}
    for col in categorical_cols + numeric_cols:
        data[col] = [form.get(col)]
    
    # Convert Age to numeric midpoint if it's a range
    data['Age'] = [convert_range_to_float(data['Age'][0])]
    
    # Ensure Systolic & Diastolic are numeric
    data['Systolic'] = [float(data['Systolic'][0])]
    data['Diastolic'] = [float(data['Diastolic'][0])]
    
    df_input = pd.DataFrame(data)

    # Encode categorical features
    X_encoded = encoder.transform(df_input[categorical_cols])
    
    # Combine numeric and encoded features
    X_numeric = df_input[numeric_cols].to_numpy()
    X_final = np.hstack([X_numeric, X_encoded])
    
    return X_final

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        X_input = prepare_input(request.form)
        
        severity_pred = severity_model.predict(X_input)[0]
        stages_pred = stages_model.predict(X_input)[0]
        
        return render_template("index.html", severity=severity_pred, stages=stages_pred)
    except Exception as e:
        return f"Error: {e}"

# ---------------------------
# Run Flask app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)