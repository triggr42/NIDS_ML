# AI Model

## Overview

This repository contains a machine learning model designed for classification tasks. The model implements several supervised learning algorithms, including Logistic Regression, K-Nearest Neighbors (KNN), and Decision Trees. The project also includes hyperparameter optimization using Optuna to improve model accuracy. The entire process covers data preprocessing, feature selection, model training, and evaluation.

## Dependencies

The following Python packages are required to run the model:

- `pandas`: For data manipulation and analysis.
- `seaborn`: For data visualization.
- `matplotlib`: For plotting graphs.
- `scikit-learn`: For machine learning algorithms and utilities.
- `optuna`: For hyperparameter optimization.
- `hashlib`: For computing MD5 and SHA-256 checksums.

You can install these dependencies via pip:

```bash
pip install pandas seaborn matplotlib scikit-learn optuna
```
## File Paths
The model requires access to specific CSV files for training and testing data. Ensure that your data files are located at the specified paths in the code before running the model.

Training Data: Path to the CSV file containing the training dataset.
Test Data: Path to the CSV file containing the test dataset.
Data Integrity Check
The model includes functionality to compute MD5 and SHA-256 checksums for the data files. This ensures that the data has not been altered or corrupted, helping to maintain data integrity throughout the modeling process.

## Data Preprocessing
The model performs several preprocessing steps:

- Loading Data: The training and test datasets are loaded from CSV files.
- Exploration and Cleaning: The data is explored to understand its structure and content. Missing values and duplicates are identified and handled appropriately.
- Visualization: The distribution of classes in the training set is visualized to check for balance.
- Label Encoding: Categorical features are converted into numerical values using label encoding.
- Feature Selection: Important features are selected using Recursive Feature Elimination (RFE).
- Scaling: Features are standardized to ensure they are on the same scale.
- Model Training and Evaluation
- Logistic Regression

## Training

- The model is trained using Logistic Regression and evaluated on both the training and test sets using metrics such as accuracy, confusion matrix, and classification report.

- K-Nearest Neighbors (KNN) with Optuna Tuning
Hyperparameter tuning for the KNN model is performed using Optuna, optimizing the n_neighbors parameter for better accuracy.

- Decision Tree with Optuna Tuning
The Decision Tree model is also optimized using Optuna, focusing on max_depth and max_features parameters.

## Performance Metrics
The performance of each model is assessed using:

- Confusion Matrix: To evaluate the correctness of predictions.
- Classification Report: To provide precision, recall, F1-score, and support for each class.
- Training and Test Accuracy: To measure the model's ability to generalize.

## Conclusion
This repository provides a comprehensive pipeline for building, tuning, and evaluating machine learning models for classification tasks. It includes all necessary steps from data loading to model evaluation, making it a useful template for similar projects.

## References
Dataset and guidance used: https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection

Big thanks to: Sampada Bhosale
