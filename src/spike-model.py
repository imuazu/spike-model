import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from src.utils.preprocess import preprocess_data

class SpikeBasedMetaLearningModel:
    def __init__(self, input_dim, output_dim, kernel_sigma=1.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        self.bias = np.random.randn(output_dim) * 0.01
        self.kernel_sigma = kernel_sigma

    def forward(self, x):
        spikes = np.dot(x, self.weights) + self.bias
        return self._softmax(spikes)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def _parzen_window(self, errors):
        n = len(errors)
        entropy = 0
        for i in range(n):
            distances = np.exp(-((errors[i] - errors) ** 2) / (2 * self.kernel_sigma ** 2))
            entropy += np.log(np.sum(distances) / n + 1e-10)
        return -entropy / n

    def _cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-10))

    def train(self, X, y, epochs=100, lr=0.01, lambda_entropy=0.1):
        print("Training Spike-Based Meta-Learning Model...")
        for epoch in tqdm(range(epochs), desc="Epochs"):
            total_loss = 0
            for i in range(len(X)):
                x = X[i]
                y_true = y[i]
                y_pred = self.forward(x)
                cross_entropy_loss = self._cross_entropy(y_true, y_pred)
                error_entropy = self._parzen_window(y_pred - y_true)
                loss = cross_entropy_loss + lambda_entropy * error_entropy
                grad_output = y_pred - y_true
                grad_weights = np.outer(x, grad_output)
                grad_bias = grad_output
                self.weights -= lr * grad_weights
                self.bias -= lr * grad_bias
                total_loss += loss
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        predictions = [self.forward(x) for x in X]
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        return np.array([self.forward(x) for x in X])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Spike-Based Meta-Learning Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    args = parser.parse_args()

    # Load and preprocess data
    data = pd.read_csv(args.data)
    X, y_one_hot = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Train model
    model = SpikeBasedMetaLearningModel(input_dim=X_train.shape[1], output_dim=2)
    model.train(X_train, y_train, epochs=args.epochs, lr=args.lr)

    # Save predictions or results as needed
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    np.save("results/spike_predictions.npy", y_pred)
    np.save("results/spike_proba.npy", y_proba)