# Necessary imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import randint, uniform
import xgboost as xgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Import cleaned data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/cleaned_data.csv')

# Reducing the sample size and shuffling the data (uncomment if needed)
df = df.sample(frac=1).iloc[:10000]

# Preparing tree-based data and standardizing X values
Tree_y = df['Purchase'].values
Tree_X = scale(df.drop('Purchase', axis=1).values)  # Scaling features for improved performance
Tree_X_train, Tree_X_test, Tree_y_train, Tree_y_test = train_test_split(Tree_X, Tree_y, test_size=0.3, random_state=30)

# Displaying train and test shapes for verification
print(f"Training set shape: {Tree_X_train.shape}, Test set shape: {Tree_X_test.shape}")

# Hyperparameter ranges for each model
param_grid = {
    'dt': {
        'max_depth': randint(1, 30),
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    },
    'rf': {
        'n_estimators': randint(50, 500),
        'max_depth': randint(1, 30),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    },
    'xgb_model': {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': [2, 3, 5, 7],
        'subsample': uniform(0.5, 0.9),
        'colsample_bytree': uniform(0.5, 0.9),
        'gamma': uniform(0, 0.5)
    },
    'bag_reg': {
        'n_estimators': randint(10, 200),
        'max_samples': uniform(0.1, 0.9),
        'max_features': uniform(0.1, 0.9),
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }
}

# Models to be tuned
models = {
    'dt': DecisionTreeRegressor(),
    'rf': RandomForestRegressor(),
    'xgb_model': xgb.XGBRegressor(),
    'bag_reg': BaggingRegressor()
}

# Performing randomized search for hyperparameter tuning
scores = []
for model_name, model in tqdm(models.items(), desc="Tuning Models"):
    rs = RandomizedSearchCV(
        model, param_grid[model_name], cv=5, n_jobs=-1, n_iter=10, verbose=1, random_state=30
    )
    rs.fit(Tree_X_train, Tree_y_train)
    scores.append({
        'model': model_name,
        'best_score': rs.best_score_,
        'best_params': rs.best_params_
    })

# Displaying tuning results
T_results = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print("\nTuning Results:\n", T_results)

# Using the best model to make predictions
best_model_name = T_results.loc[T_results['best_score'].idxmax(), 'model']
best_model = models[best_model_name]
best_params = T_results.loc[T_results['best_score'].idxmax(), 'best_params']
best_model.set_params(**best_params)
best_model.fit(Tree_X_train, Tree_y_train)

# Make predictions on the test set
T_y_pred = best_model.predict(Tree_X_test).astype(int)

# Convert best parameters to a DataFrame and save
T_best_params_df = pd.DataFrame([{'Model': best_model_name, 'Best_Params': best_params}])
T_best_params_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/T_best_params_df.csv', index=False)
print("\nBest model parameters saved as 'T_best_params_df.csv' in the specified directory.")

# Evaluating the best model
T_metrics = {
    'MAE': mean_absolute_error(Tree_y_test, T_y_pred),
    'MSE': mean_squared_error(Tree_y_test, T_y_pred),
    'R2': r2_score(Tree_y_test, T_y_pred),
    'RMSE': np.sqrt(mean_squared_error(Tree_y_test, T_y_pred)),  # Changed to RMSE for better interpretation
    'MedAE': median_absolute_error(Tree_y_test, T_y_pred)
}

# Displaying metrics
T_metrics_df = pd.DataFrame(list(T_metrics.items()), columns=['Metric', 'R_Values'])
T_metrics_df["R_Values"] = T_metrics_df["R_Values"].round(2)
print("\nModel Evaluation Metrics:\n", T_metrics_df)

# Save the evaluation metrics DataFrame
T_metrics_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/T_metrics_df.csv', index=False)
print("\nTree results data saved as 'T_metrics_df.csv' in the specified directory.")

# Residual analysis
# Calculating residuals
residuals = Tree_y_test - T_y_pred

# 1. Histogram of Residuals
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# 2. Residual Plot (Residuals vs Predicted Values)
plt.figure(figsize=(10, 6))
plt.scatter(T_y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# 3. QQ Plot to Check Normality
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

# 4. Residuals vs. Fitted Plot (Checking Homoscedasticity)
plt.figure(figsize=(10, 6))
sns.residplot(x=T_y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.show()
