import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    # Define column names if not already present
    column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
                    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    if data.columns[0] != "Pregnancies":
        data = pd.read_csv(data, names=column_names)
    
    # Impute missing values with median
    for col in data.columns[:-1]:  # Exclude 'Outcome'
        data[col] = data[col].replace(0, data[col].median())
    
    # Split features and target
    X = data.drop("Outcome", axis=1).values
    y = data["Outcome"].values
    y_one_hot = np.eye(2)[y]  # Convert to one-hot encoding

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y_one_hot