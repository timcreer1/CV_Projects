# Necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge, LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from scipy.stats import randint
from tqdm import tqdm

# Import cleaned data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/cleaned_data.csv')

# Reducing the sample size and shuffling the data
df = df.sample(frac=1).iloc[:10000]

# Splitting data into features and target
Reg_y = df['Purchase'].values
Reg_X = df.drop('Purchase', axis=1).values

# Splitting into train/test sets (70:30 ratio)
Reg_X_train, Reg_X_test, Reg_y_train, Reg_y_test = train_test_split(Reg_X, Reg_y, test_size=0.3, random_state=30)

# Models to be evaluated
models = {
    'bayes_ridge': BayesianRidge(),
    'lin_reg': LinearRegression(),
    'lasso': Lasso(),
    'svr': SVR()
}

# parameter grids for each model
# Corrected parameter grids for each model
param_distributions = {
    'lin_reg': {'fit_intercept': [True, False], 'copy_X': [True, False]},
    'bayes_ridge': {
        'max_iter': randint(100, 400),
        'alpha_1': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'alpha_2': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'lambda_1': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'lambda_2': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'fit_intercept': [True, False]
    },
    'lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random'],
        'max_iter': randint(100, 1000),
        'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'warm_start': [True, False]
    },
    'svr': {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.1, 1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4]
    }
}


# Performing the randomized search with cross-validation
scores = []
for model_name, model in tqdm(models.items(), desc="Tuning Models"):
    rs = RandomizedSearchCV(
        model, param_distributions[model_name], cv=5, n_jobs=-1, n_iter=10, verbose=1, random_state=30
    )
    rs.fit(Reg_X_train, Reg_y_train)
    scores.append({
        'model': model_name,
        'best_score': rs.best_score_,
        'best_params': rs.best_params_
    })

# Displaying the results
R_results = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print("\nModel Tuning Results:\n", R_results)

# Use the best model to make predictions on the test set
best_model_name = R_results.loc[R_results['best_score'].idxmax(), 'model']
best_model = models[best_model_name]
best_params = R_results.loc[R_results['best_score'].idxmax(), 'best_params']
best_model.set_params(**best_params)
best_model.fit(Reg_X_train, Reg_y_train)
R_y_pred = best_model.predict(Reg_X_test).astype(int)

# Convert best parameters to a DataFrame and save
R_best_params_df = pd.DataFrame([{'Model': best_model_name, 'Best_Params': best_params}])
R_best_params_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/R_best_params_df.csv', index=False)
print("\nBest model parameters saved as 'R_best_params_df.csv' in the specified directory.")

# Evaluating the model with various metrics
R_metrics = {
    'MAE': mean_absolute_error(Reg_y_test, R_y_pred),
    'MSE': mean_squared_error(Reg_y_test, R_y_pred),
    'R2': r2_score(Reg_y_test, R_y_pred),
    'RMSE': np.sqrt(mean_squared_error(Reg_y_test, R_y_pred)),
    'MedAE': median_absolute_error(Reg_y_test, R_y_pred)
}

# Displaying metrics
R_metrics_df = pd.DataFrame(list(R_metrics.items()), columns=['Metric', 'R_Values'])
R_metrics_df["R_Values"] = R_metrics_df["R_Values"].round(2)
print("\nModel Evaluation Metrics:\n", R_metrics_df)

# Save the evaluation metrics DataFrame
R_metrics_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/R_metrics_df.csv', index=False)
print("\nLinear results data saved as 'R_metrics_df.csv' in the specified directory.")



# Residual Analysis
# Calculating residuals
residuals = Reg_y_test - R_y_pred

# 1. Residual Plot
plt.figure(figsize=(10, 6))
plt.scatter(R_y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# 2. Histogram of Residuals
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

# 3. QQ Plot to Check Normality
plt.figure(figsize=(8, 6))
import scipy.stats as stats
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals')
plt.show()

# 4. Residuals vs. Fitted Plot (Checking Homoscedasticity)
plt.figure(figsize=(10, 6))
import seaborn as sns
sns.residplot(x=R_y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.show()
