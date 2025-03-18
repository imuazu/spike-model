import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, auc
)
import pandas as pd
import os

def plot_roc_curves(y_test_labels, y_proba_spike, y_proba_svc, y_proba_cnn, y_proba_resnet):
    plt.figure(figsize=(6, 5))
    for model_name, y_proba in zip(['Spike-Based', 'SVC', 'CNN', 'ResNet'], 
                                   [y_proba_spike, y_proba_svc, y_proba_cnn, y_proba_resnet]):
        fpr, tpr, _ = roc_curve(y_test_labels, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc(fpr, tpr):.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.savefig("results/roc_curves.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"results/confusion_matrix_{title.replace(' ', '_').lower()}.png")
    plt.close()

if __name__ == "__main__":
    # Load test labels (assuming saved from one of the model scripts)
    y_test = np.load("results/y_test_labels.npy", allow_pickle=True)

    # Load predictions and probabilities
    y_pred_spike = np.load("results/spike_predictions.npy")
    y_proba_spike = np.load("results/spike_proba.npy")
    y_pred_svc = np.load("results/svc_predictions.npy")
    y_proba_svc = np.load("results/svc_proba.npy")
    y_proba_cnn = np.load("results/cnn_proba.npy")
    y_proba_resnet = np.load("results/resnet_proba.npy")

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Compute metrics
    results_df = pd.DataFrame({
        'Model': ['Spike-Based', 'SVC', 'CNN', 'Resnet50'],
        'Accuracy': [
            accuracy_score(y_test, y_pred_spike),
            accuracy_score(y_test, y_pred_svc),
            accuracy_score(y_test, np.argmax(y_proba_cnn, axis=1)),
            accuracy_score(y_test, np.argmax(y_proba_resnet, axis=1))
        ],
        'Precision': [
            precision_score(y_test, y_pred_spike),
            precision_score(y_test, y_pred_svc),
            precision_score(y_test, np.argmax(y_proba_cnn, axis=1)),
            precision_score(y_test, np.argmax(y_proba_resnet, axis=1))
        ],
        'Recall': [
            recall_score(y_test, y_pred_spike),
            recall_score(y_test, y_pred_svc),
            recall_score(y_test, np.argmax(y_proba_cnn, axis=1)),
            recall_score(y_test, np.argmax(y_proba_resnet, axis=1))
        ],
        'F1 Score': [
            f1_score(y_test, y_pred_spike),
            f1_score(y_test, y_pred_svc),
            f1_score(y_test, np.argmax(y_proba_cnn, axis=1)),
            f1_score(y_test, np.argmax(y_proba_resnet, axis=1))
        ],
        'ROC AUC': [
            roc_auc_score(y_test, y_proba_spike[:, 1]),
            roc_auc_score(y_test, y_proba_svc[:, 1]),
            roc_auc_score(y_test, y_proba_cnn[:, 1]),
            roc_auc_score(y_test, y_proba_resnet[:, 1])
        ]
    })
    results_df.to_csv("results/performance_metrics.csv", index=False)
    print(results_df)

    # Plot results
    plot_roc_curves(y_test, y_proba_spike, y_proba_svc, y_proba_cnn, y_proba_resnet)
    plot_confusion_matrix(y_test, y_pred_spike, 'Spike-Based Model')
    plot_confusion_matrix(y_test, y_pred_svc, 'SVC Model')
    plot_confusion_matrix(y_test, np.argmax(y_proba_cnn, axis=1), 'CNN Model')
    plot_confusion_matrix(y_test, np.argmax(y_proba_resnet, axis=1), 'Resnet50 Model')