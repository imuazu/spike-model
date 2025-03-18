import pandas as pd
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from src.utils.preprocess import preprocess_data

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    # Load and preprocess data
    data = pd.read_csv(args.data)
    X, y_one_hot = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Reshape for CNN
    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

    # Train CNN
    cnn_model = create_cnn_model((X_train.shape[1], 1))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, verbose=1)
    y_proba = cnn_model.predict(X_test_cnn)

    # Save results
    np.save("results/cnn_proba.npy", y_proba)