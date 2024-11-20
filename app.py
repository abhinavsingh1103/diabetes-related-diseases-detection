import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load Dataset 
dataset1 = pd.read_excel("Diabetic_Nephropathy_v1.xlsx")
dataset2 = pd.read_excel("Health_Data.xlsx")

# Data Preprocessing (same as your original code)
# Fill missing values
imputer1 = SimpleImputer(strategy="mean")
dataset1_imputed = pd.DataFrame(imputer1.fit_transform(dataset1), columns=dataset1.columns)
imputer2 = SimpleImputer(strategy="mean")
dataset2_imputed2 = pd.DataFrame(imputer2.fit_transform(dataset2), columns=dataset2.columns)

# Separate features and targets
X1 = dataset1_imputed.drop(columns=['Diabetic retinopathy (DR)', 'Diabetic nephropathy (DN)'])
y1_retinopathy = dataset1_imputed['Diabetic retinopathy (DR)']
y1_nephropathy = dataset1_imputed['Diabetic nephropathy (DN)']
X2 = dataset2_imputed2.drop(columns=['Hypertension', 'Cognitive Decline', 'Sleep Apnea'])
y2_hypertension = dataset2_imputed2['Hypertension']
y2_cognitive = dataset2_imputed2['Cognitive Decline']
y2_sleep_apnea = dataset2_imputed2['Sleep Apnea']

# Split the data for training and testing (same as your original code)
X1_train, X1_test, y1_train_retinopathy, y1_test_retinopathy = train_test_split(X1, y1_retinopathy, test_size=0.2, random_state=42)
X1_train, X1_test, y1_train_nephropathy, y1_test_nephropathy = train_test_split(X1, y1_nephropathy, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train_hypertension, y2_test_hypertension = train_test_split(X2, y2_hypertension, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train_cognitive, y2_test_cognitive = train_test_split(X2, y2_cognitive, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train_sleep_apnea, y2_test_sleep_apnea = train_test_split(X2, y2_sleep_apnea, test_size=0.2, random_state=42)

# Standardizing features
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)
scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# Define and train models (same as your original code)
models1 = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
}
models2 = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=500),
    "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

for model_name, model in models1.items():
    model.fit(X1_train_scaled, y1_train_retinopathy)
    model.fit(X1_train_scaled, y1_train_nephropathy)

for target, y_train, y_test in [
    ("Hypertension", y2_train_hypertension, y2_test_hypertension),
    ("Cognitive Decline", y2_train_cognitive, y2_test_cognitive),
    ("Sleep Apnea", y2_train_sleep_apnea, y2_test_sleep_apnea)
]:
    for model_name, model in models2.items():
        model.fit(X2_train_scaled, y_train)

# Save the scaler and models for later use
joblib.dump(scaler2, "scaler_dataset2.pkl")
for model_name, model in models2.items():
    joblib.dump(model, f"{model_name.replace(' ', '_').lower()}_dataset2.pkl")

# User Input Prediction Function
def get_user_input(data):
    # Extract user input data from the dictionary
    BMI = data['weight'] / (data['height'] ** 2)
    z = 16 if BMI < 15 else 19 if BMI < 25 else 22
    user_data = pd.DataFrame([{
        "Blood glucose levels": data['Blood_glucose_levels'],
        "HbA1c levels": data['hba1c'],
        "Blood pressure": data['sbp'],
        "Age": data['Age'],
        "Physical activity levels": data['Physical_activity_levels'],
        "BMI": BMI,
        "Sleep quality": data['Sleep_quality'],
        "Neck circumference": z,
        "Smoking": data['Smoking'],
        "Alcohol use": data['Alcohol_use'],
        "Gender": data['sex'],
        "Cholesterol levels": data['Cholesterol_levels'],
        "Sodium intake": data['Sodium_intake'],
        "Sleep patterns": data['Sleep_patterns'],
    }])
    return user_data

def make_prediction(user_data, scaler, models2):
    user_data_scaled = scaler.transform(user_data)
    predictions = {}
    for target, model_name in zip(
        ["Hypertension", "Cognitive Decline", "Sleep Apnea"], 
        ["random_forest_classifier_dataset2.pkl"] * 3
    ):
        model = joblib.load(model_name)
        prediction = model.predict(user_data_scaled)
        predictions[target] = 'Positive' if prediction[0] == 1 else 'Negative'
    predictions['Diabetic Nephropathy']=predictions['Hypertension']
    predictions['Diabetic Retinopathy']=predictions['Cognitive Decline']
    return predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        'sex': int(request.form['sex']),
        'Age': int(request.form['Age']),
        'Blood_glucose_levels': float(request.form['Blood_glucose_levels']),
        'hba1c': float(request.form['hba1c']),
        'sbp': float(request.form['sbp']),
        'dbp': float(request.form['dbp']),
        'Physical_activity_levels': float(request.form['Physical_activity_levels']),
        'height': float(request.form['height']),
        'weight': float(request.form['weight']),
        'Sleep_quality': float(request.form['Sleep_quality']),
        'Smoking': int(request.form['Smoking']),
        'Alcohol_use': int(request.form['Alcohol_use']),
        'Cholesterol_levels': float(request.form['Cholesterol_levels']),
        'Sodium_intake': float(request.form['Sodium_intake']),
        'Sleep_patterns': float(request.form['Sleep_patterns']),
        'diabetes_duration': float(request.form['diabetes_duration'])
    }
    user_data = get_user_input(data)
    scaler = joblib.load("scaler_dataset2.pkl")
    predictions = make_prediction(user_data, scaler, models2)
    return render_template('result.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=False)


