import pandas as pd
import numpy as np
import os

def load_student_data(subject='mat'):
    """
    Load and preprocess the student performance dataset.
    subject: 'mat' (Math) or 'por' (Portuguese)
    """
    if subject == 'mat':
        file_path = os.path.join(os.path.dirname(__file__), '../../test_data/Education/student+performance/student/student-mat.csv')
    elif subject == 'por':
        file_path = os.path.join(os.path.dirname(__file__), '../../test_data/Education/student+performance/student/student-por.csv')
    else:
        raise ValueError("subject must be 'mat' or 'por'")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # The dataset uses ';' as delimiter
    df = pd.read_csv(file_path, sep=';')
    
    # Coding variables as requested
    # School: GP 1, MS 0
    df['school'] = df['school'].map({'GP': 1, 'MS': 0})
    
    # sex: F 0, M 1
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    
    # address: U 0, R 1
    df['address'] = df['address'].map({'U': 0, 'R': 1})
    
    # famsize: LE3 0, GT3 1
    df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
    
    # Pstatus: T 0, A 1
    df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
    
    # Mjob and Fjob: other 0, at_home 1, services 2, teacher 3, health 4
    job_map = {'other': 0, 'at_home': 1, 'services': 2, 'teacher': 3, 'health': 4}
    df['Mjob'] = df['Mjob'].map(job_map)
    df['Fjob'] = df['Fjob'].map(job_map)
    
    # reason: other 0, home 1, reputation 2, course 3
    reason_map = {'other': 0, 'home': 1, 'reputation': 2, 'course': 3}
    df['reason'] = df['reason'].map(reason_map)
    
    # guardian: other 0, mother 1, father 2
    guardian_map = {'other': 0, 'mother': 1, 'father': 2}
    df['guardian'] = df['guardian'].map(guardian_map)

    # Binary variables not explicitly mentioned but often encoded
    # schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
    binary_map = {'yes': 1, 'no': 0}
    for col in ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']:
        if col in df.columns:
            df[col] = df[col].map(binary_map)

    # Cleaning: check for missing values
    if df.isnull().any().any():
        print(f"Warning: Missing values found in {subject} student data. Dropping rows.")
        df = df.dropna()
        
    return df

if __name__ == "__main__":
    df_mat = load_student_data('mat')
    print(f"Math student shape: {df_mat.shape}")
    print(df_mat.head())
    
    df_por = load_student_data('por')
    print(f"Portuguese student shape: {df_por.shape}")
    print(df_por.head())
