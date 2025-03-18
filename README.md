# spike-model
# Robust Diabetes Prediction with Spike-Based Meta-Learning

## Overview
 This repository contains the implementation of a novel **Spike-Based Continual Meta-Learning Model** combined with a **Minimum Entropy Criterion** for diabetes prediction, as described in the paper *"Towards Robust Diabetes Prediction: Leveraging Spike-Based Meta-Learning and Minimum Entropy for Improved Performance Over Traditional Machine Learning Models"* by Isah Muazu, İlker Etikan, and Fadi Al-Turjman. The model leverages Spiking Neural Networks (SNNs) and meta-learning to achieve superior performance over traditional machine learning models like SVC, CNN, and ResNet50, evaluated on the Pima Indians Diabetes dataset.

 Key features:
 - Achieves **99.14% accuracy** and **0.9999 ROC-AUC**, outperforming baseline models.
 - Incorporates energy-efficient SNNs with biological plausibility.
 - Uses minimum entropy regularization for robust generalization.
 - Provides interpretability via SHAP and ALE analysis, highlighting glucose as a critical predictor.

 ## Paper
 The full paper is available in this repository: [Paper PDF](docs/paper.pdf). Supplementary materials (figures and tables) are also included.

 ## Repository Structure
 ```
 spike-model/
 ├── data/                # Dataset files (Pima Indians Diabetes dataset)
 ├── src/                 # Source code for the spike-based model and baselines
 │   ├── spike_model.py   # Spike-Based Meta-Learning model implementation
 │   ├── baselines/       # SVC, CNN, ResNet50 implementations
 │   └── utils/           # Preprocessing and evaluation utilities
 ├── results/             # Figures, tables, and evaluation metrics
 ├── docs/                # Paper and additional documentation
 ├── requirements.txt     # Python dependencies
 └── README.md            # This file
 ```

 ## Prerequisites
 - Python 3.8+
 - Git
 - Basic familiarity with machine learning and neural networks

 ## Installation
 Clone the repository and set up the environment:

 ```bash
 git clone https://github.com/vmercel/spike-model.git
 cd spike-model
 pip install -r requirements.txt
 ```

 ## Dataset
 The model is trained and evaluated on the **Pima Indians Diabetes Dataset**, included in `data/pima-indians-diabetes.csv`. It contains 768 samples with 8 features (e.g., glucose, BMI) and a binary target (diabetes or not).

 Preprocessing steps (applied in `src/utils/preprocess.py`):
 - Impute missing values with median.
 - Standardize features to zero mean and unit variance.

 ## Usage
 ### Training the Spike-Based Model
 Run the main script to train and evaluate the model:

 ```bash
 python src/spike_model.py --data data/pima-indians-diabetes.csv --epochs 100 --lr 0.01
 ```
 - `--data`: Path to the dataset.
 - `--epochs`: Number of training epochs (default: 100).
 - `--lr`: Learning rate (default: 0.01).

 ### Evaluating Baseline Models
 Compare with SVC, CNN, and ResNet50:

 ```bash
 python src/baselines/svc.py --data data/pima-indians-diabetes.csv
 python src/baselines/cnn.py --data data/pima-indians-diabetes.csv
 python src/baselines/resnet50.py --data data/pima-indians-diabetes.csv
 ```

 ### Generating Results
 Reproduce figures (ROC curves, confusion matrices, SHAP plots) and tables:

 ```bash
 python src/utils/plot_results.py
 ```
 Results are saved in the `results/` directory.

 ## Model Architecture
 The Spike-Based Meta-Learning Model uses:
 - **Spiking Neural Networks (SNNs)**: Energy-efficient, biologically inspired neurons.
 - **SoftMax Activation**: Mimics neural spiking for interpretability.
 - **Minimum Entropy Regularization**: Reduces prediction uncertainty using Parzen window estimation.
 - **Gradient Descent**: Optimizes weights and biases with a combined loss function.

 See `src/spike_model.py` for implementation details.

 ## Results
 The model outperforms baselines across key metrics:

 | Model       | Accuracy | Precision | Recall | F1 Score | ROC AUC  |
 |-------------|----------|-----------|--------|----------|----------|
 | Spike-Based | 0.9914   | 0.9846    | 0.9846 | 0.9846   | 0.9999   |
 | SVC         | 0.9828   | 0.9841    | 0.9538 | 0.9688   | 0.9993   |
 | CNN         | 0.9907   | 0.9848    | 1.0000 | 0.9924   | 0.9997   |
 | ResNet50    | 0.9828   | 0.9420    | 1.0000 | 0.9701   | 0.9986   |

 - **ROC Curves**: See `results/roc_curves.png`.
 - **Confusion Matrices**: See `results/confusion_matrices/`.
 - **SHAP Analysis**: Glucose is the top predictor (`results/shap_summary.png`).

 ## Interpretability
 - **SHAP Summary**: Highlights glucose, BMI, and diabetes pedigree as key features.
 - **ALE Plots**: Show glucose’s sigmoidal effect on predictions.
 - **Feature Interactions**: Glucose and age interplay critically affects risk.

 ## Contributing
 Contributions are welcome! Please:
 1. Fork the repository.
 2. Create a feature branch (`git checkout -b feature/new-idea`).
 3. Commit changes (`git commit -m "Add new feature"`).
 4. Push to the branch (`git push origin feature/new-idea`).
 5. Open a Pull Request.

 ## Citation
 If you use this work, please cite:

 ```bibtex
 @article{muazu2023spike,
   title={Towards Robust Diabetes Prediction: Leveraging Spike-Based Meta-Learning and Minimum Entropy for Improved Performance Over Traditional Machine Learning Models},
   author={Muazu, Isah and Etikan, İlker and Al-Turjman, Fadi},
   journal={Scientific Report (in press)},
   year={202_...},
   publisher={springer}
 }
 ```

 ## Acknowledgments
 - Supported by the International Research Center for AI and IoT at Near East University.
 - Dataset sourced from the Pima Indians Diabetes Database.

 ## License
 This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 ## Contact
 - Isah Muazu: [20214918@std.neu.edu.tr](mailto:20214918@std.neu.edu.tr)
 - GitHub Issues: Report bugs or ask questions [here](https://github.com/vmercel/spike-model/issues).
