# 🛒 Black Friday Sales Analysis & Price Prediction

This project provides a comprehensive analysis of Black Friday sales data, focusing on understanding purchasing behaviour and predicting purchase amounts using various machine learning and deep learning models. The project involves detailed exploratory data analysis, data preprocessing, and the development of regression, tree-based, and neural network models to predict sales, offering valuable insights into consumer behaviour during the Black Friday sales period.

## 📄 Overview

The project uses data analysis and machine learning techniques to explore and predict purchasing patterns in Black Friday sales data. Key steps include data cleaning, exploratory data analysis, model building, and performance evaluation using metrics like MAE, MSE, R2, and RMSE. Each model undergoes comprehensive residual analysis, including Histogram of Residuals, Residual Plot (Residuals vs Predicted Values), QQ Plot to check normality, and Residuals vs. Fitted Plot to assess homoscedasticity. The analysis highlights purchasing behaviour across demographic groups and identifies the best model for accurate purchase predictions.

### 🔍 Key Components

- **Data**:  
  Contains the Black Friday sales data used for the analysis. Outputs generated during code execution will be stored here.

- **1. EDA**:  
  This code loads data from a CSV file, examines its structure, identifies missing and unique values, and analyses purchase distributions. It visualises total and average purchases across demographic groups (Gender, Marital Status, Occupation, City Stay Duration, Age) to highlight purchasing behaviour patterns.

- **2. Data Cleaning**:  
  This code performs data preprocessing and multicollinearity analysis on a purchase dataset by handling missing values, encoding categorical features, and examining correlations between variables using a heatmap. It calculates Variance Inflation Factor (VIF) and Tolerance to assess multicollinearity, drops low-correlation columns, and saves the cleaned data for further analysis.

- **3a. Linear Models**:  
  This code builds and evaluates regression models (Bayesian Ridge, Linear Regression, Lasso, and SVR) with hyperparameter tuning and residual analysis.

- **3b. Tree-Based Models**:  
  This code constructs tree-based models (Decision Trees, Random Forests, XGBoost, Bagging Regressors) with scaling, hyperparameter tuning, and residual analysis.

- **3c. Neural Network Model**:  
  This code builds a neural network model with dense layers, batch normalisation, and L1 regularisation, including callbacks for early stopping and learning rate adjustments.

- **4. Results**:  
  Compares the performance of all models, visualises evaluation metrics for easy comparison, and identifies the best-performing model for predictive accuracy.

## 🛠️ Requirements

Ensure the following packages are installed before running the code:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
