import pandas as pd
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from src.utils.preprocess import preprocess_data

def train_svc_model(X_train, y_train):
    svc_model = SVC(probability=True)
    svc_model.fit(X_train, np.argmax(y_train, axis=1))
    return svc_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    return y_pred, y_proba, y_test_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SVC Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    # Load and preprocess data
    data = pd.read_csv(args.data)
    X, y_one_hot = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Train and evaluate SVC
    svc_model = train_svc_model(X_train, y_train)
    y_pred, y_proba, _ = evaluate_model(svc_model, X_test, y_test)

    # Save results
    np.save("results/svc_predictions.npy", y_pred)
    np.save("results/svc_proba.npy", y_proba)