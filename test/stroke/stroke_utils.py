import os
import numpy as np
import pandas as pd


def load_stroke_data(file_path=None):
    if file_path is None:
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../test_data/Health/healthcare-dataset-stroke-data.csv'))

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Stroke dataset file not found at {file_path}")

    df = pd.read_csv(file_path)
    if 'stroke' not in df.columns:
        raise ValueError("Expected a 'stroke' target column in the dataset")

    df = df.copy()
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df = df.drop(columns=['id'])
    
    # Drop rows with missing BMI values
    df = df.dropna(subset=['bmi'])
    
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Expected categorical column '{col}' in stroke dataset")
    
    # Encode categorical variables with specified mappings
    gender_map = {'Male': 1, 'Female': 0, 'Other': 2}
    ever_married_map = {'Yes': 1, 'No': 0}
    work_type_map = {'children': 0, 'Never_worked': 1, 'Self-employed': 2, 'Private': 3, 'Govt_job': 4}
    residence_type_map = {'Rural': 0, 'Urban': 1}
    smoking_status_map = {'Unknown': 0, 'never smoked': 1, 'formerly smoked': 2, 'smokes': 3}
    
    df['gender'] = df['gender'].map(gender_map)
    df['ever_married'] = df['ever_married'].map(ever_married_map)
    df['work_type'] = df['work_type'].map(work_type_map)
    df['Residence_type'] = df['Residence_type'].map(residence_type_map)
    df['smoking_status'] = df['smoking_status'].map(smoking_status_map)
    
    df = df.dropna(subset=['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'stroke'])

    X = df.drop(columns=['stroke'])
    y = df['stroke'].astype(int)
    return X, y
