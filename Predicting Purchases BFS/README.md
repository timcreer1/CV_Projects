# 🛒 Black Friday Sales Analysis & Price Prediction

This project provides a comprehensive analysis of Black Friday sales data, focusing on understanding purchasing behavior and predicting purchase amounts using various machine learning and deep learning models. The project involves detailed exploratory data analysis, data preprocessing, and the development of regression, tree-based, and neural network models to predict sales, offering valuable insights into consumer behavior during the Black Friday sales period.

## 📄 Overview

The project employs data analysis and machine learning techniques to explore and predict purchasing patterns in Black Friday sales data. Key steps include data cleaning, exploratory data analysis, model building, and performance evaluation using metrics such as MAE, MSE, R2, and RMSE. The analysis highlights purchasing behavior across demographic groups and identifies the best model for predicting purchase amounts.

### 🔍 Key Components

- **Data**:  
  Contains the Black Friday sales data used for the analysis. Outputs generated during code execution will be stored here.

- **1. EDA**:  
  Loads and examines the dataset, identifies missing and unique values, and analyzes purchase distributions across demographic groups.

- **2. Data Cleaning**:  
  Performs data preprocessing, handles missing values, encodes categorical features, and assesses multicollinearity using VIF and Tolerance metrics.

- **3a. Linear Models**:  
  Builds and evaluates regression models (Bayesian Ridge, Linear Regression, Lasso, and SVR) with hyperparameter tuning and residual analysis.

- **3b. Tree-Based Models**:  
  Develops tree-based models (Decision Trees, Random Forests, XGBoost, Bagging Regressors) with scaling, hyperparameter tuning, and residual analysis.

- **3c. Neural Network Model**:  
  Constructs a neural network model with dense layers, batch normalization, and L1 regularization, including callbacks for early stopping and learning rate adjustments.

- **4. Results**:  
  Compares the performance of all models, visualizes evaluation metrics, and identifies the best-performing model for predictive accuracy.

## 🛠️ Requirements

Ensure the following packages are installed before running the code:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
