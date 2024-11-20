## Overview
This project is a Flask-based web application designed to predict multiple health conditions related to diabetes, such as **Diabetic Nephropathy**, **Diabetic Retinopathy**, **Hypertension**, **Cognitive Decline**, and **Sleep Apnea**. The application uses machine learning models trained on two datasets to provide predictions based on user input.

## Features
- **User-friendly interface**: A web-based form for users to input their data.
- **Multiple Predictions**: Predicts the likelihood of various diabetes-related health issues.
- **ML Models**: Utilizes Logistic Regression, Decision Tree, and Random Forest algorithms.
- **Scalable**: Trained models are saved using Joblib for reusability.

---

## Installation

### Prerequisites
1. **Python 3.7+**
2. **Pip** package manager
3. Necessary Python libraries (listed below)

### Libraries Required
Install the required libraries using:
```bash
pip install pandas numpy scikit-learn flask joblib openpyxl
```

### Files
- **Python File**: `app.py` (the main Flask application)
- **Datasets**: 
  - `Diabetic_Nephropathy_v1.xlsx`
  - `Health_Data.xlsx`
- **Trained Models**:
  - `scaler_dataset2.pkl`
  - `{model_name}_dataset2.pkl` (e.g., `random_forest_classifier_dataset2.pkl`)
- **Templates**:
  - `templates/index.html` (input form)
  - `templates/result.html` (output display)

---

## Running the Application

1. Clone the repository or download the project files.
2. Place the datasets and trained models in the project directory.
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/`.

---

## User Guide

### Input Fields
1. **Sex**: Enter `0` for Female, `1` for Male.
2. **Age**: Enter your age in years.
3. **Blood Glucose Levels**: Enter your blood glucose levels (mg/dL).
4. **HbA1c**: Enter your HbA1c levels (percentage).
5. **Systolic Blood Pressure (SBP)**: Enter SBP (mmHg).
6. **Diastolic Blood Pressure (DBP)**: Enter DBP (mmHg).
7. **Physical Activity Levels**: Activity score (0–10).
8. **Height**: Enter height in meters.
9. **Weight**: Enter weight in kilograms.
10. **Sleep Quality**: Sleep quality score (0–10).
11. **Smoking**: Enter `1` for smoker, `0` for non-smoker.
12. **Alcohol Use**: Enter `1` for yes, `0` for no.
13. **Cholesterol Levels**: Enter cholesterol levels (mg/dL).
14. **Sodium Intake**: Daily sodium intake (mg).
15. **Sleep Patterns**: Sleep pattern score (0–10).
16. **Diabetes Duration**: Duration of diabetes diagnosis (years).

### Output
- Displays predictions for:
  - **Hypertension**
  - **Cognitive Decline**
  - **Sleep Apnea**
  - **Diabetic Retinopathy**
  - **Diabetic Nephropathy**
- Each result is displayed as **Positive** (at risk) or **Negative** (not at risk).

---

## Project Workflow

### Data Preprocessing
1. Missing values in the datasets are handled using mean imputation.
2. Features are standardized using `StandardScaler`.

### Model Training
1. **Dataset 1**: Predicts *Diabetic Retinopathy* and *Diabetic Nephropathy*.
2. **Dataset 2**: Predicts *Hypertension*, *Cognitive Decline*, and *Sleep Apnea*.

### User Input
- User-provided data is preprocessed (e.g., BMI calculation, feature scaling) to match the trained models' input format.

### Prediction
- Models saved as `.pkl` files are loaded for inference.
- Outputs are displayed on the results page.

---

## Folder Structure
```
project/
│
├── app.py                     # Main application script
├── Diabetic_Nephropathy_v1.xlsx
├── Health_Data.xlsx
├── scaler_dataset2.pkl         # Saved scaler
├── random_forest_classifier_dataset2.pkl
├── templates/
│   ├── index.html              # Input form
│   ├── result.html             # Output display
└── README.md
```

---

## Future Improvements
1. Add a database for user data storage.
2. Implement advanced visualization of results.
3. Include more features and conditions for predictions.

---

## Contact
For any queries, please contact **[Your Name]** at **[Your Email]**.
