# Breast Cancer Tumor Classification with Decision Trees and Random Forests

This repository contains a machine learning project focused on classifying **breast cancer tumors** as **malignant** or **benign** using the **Wisconsin Breast Cancer dataset**. The project explores data characteristics, preprocessing impacts, and ensemble optimization with Random Forests.

> **Note:** The random seed value is fixed at 42 wherever applicable for reproducibility.

## Assignment Overview

1. **Load and Inspect Dataset**  
   - Load the Wisconsin Breast Cancer dataset from `sklearn.datasets`.  
   - Report dataset shape, target class distribution, and identify any class imbalance.

2. **Data Quality Check**  
   - Check for missing values and duplicate rows.  
   - Identify the top 5 features with the highest standard deviation.  
   - Present results with clear print messages.

3. **Decision Tree Classifiers**  
   - Split data using stratified 80/20 train/test split.  
   - Train two Decision Tree classifiers (`max_depth=3`) on the training set:  
     - One with **RobustScaler** preprocessing  
     - One without preprocessing  
   - Compare test set predictions and discuss observed differences.

4. **Decision Tree Analysis**  
   - Export one of the trained Decision Trees.  
   - Identify the **root node splitting feature** and the **total number of leaves**.

5. **Random Forest Variants with GridSearchCV**  
   Build and tune three Random Forest variants using **5-fold cross-validation** based on **ROC-AUC** metric:  
   
   - **a. Simple Random Forest (no preprocessing):**  
     - `n_estimators`: [50, 100]  
     - `max_depth`: [3, 5, 10]  
     
   - **b. Random Forest with PCA (Principal Component Analysis):**  
     - Apply `StandardScaler`  
     - `pca_n_components`: [10, 20, 30]  
     - `n_estimators`: [50, 100]  
     - `max_depth`: [3, 5, 10]  
     
   - **c. Random Forest with LLE (Locally Linear Embedding):**  
     - Apply `StandardScaler`  
     - `lle_n_components`: [10, 15]  
     - `lle_n_neighbors`: [5, 10, 15]  
     - `n_estimators`: [50, 100]  
     - `max_depth`: [3, 5, 10]  

   For each variant, report:  
   - Mean ROC-AUC (training set)  
   - ROC-AUC (test set)  
   - Accuracy (test set)  
   - Training time (seconds)  
   - Best hyperparameters  

   Identify the best performing model and provide a brief analysis comparing all three variants.

## Requirements

- Python 3.x  
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`  

## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/Breast-Cancer-Classification.git

2. Open the Jupyter Notebook:
   jupyter notebook Breast_Cancer_Classification.ipynb

3. Follow the notebook to load the dataset, preprocess features, train Decision Trees, build and tune Random Forest variants, and evaluate models.

## Outcome

This project provides a complete workflow for tumor classification, including exploratory data analysis, feature preprocessing, model training, hyperparameter tuning, ensemble methods, and performance evaluation. It also highlights the impact of preprocessing and dimensionality reduction on ensemble model performance.
