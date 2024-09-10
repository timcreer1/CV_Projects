# Necessary imports
import pandas as pd
import matplotlib.pyplot as plt

# File paths for the saved metrics
nn_metrics_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/NN_metrics_df.csv'
t_metrics_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/T_metrics_df.csv'
r_metrics_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/R_metrics_df.csv'

# Load the metrics DataFrames
NN_metrics_df = pd.read_csv(nn_metrics_path)
T_metrics_df = pd.read_csv(t_metrics_path)
R_metrics_df = pd.read_csv(r_metrics_path)

# Add a 'Model' column to each DataFrame to identify the rename columns
NN_metrics_df['Model'] = 'Neural Network'
T_metrics_df['Model'] = 'Tree-Based'
R_metrics_df['Model'] = 'Regression'
NN_metrics_df = NN_metrics_df.rename(columns={'NN_Values': 'Value'})
T_metrics_df = T_metrics_df.rename(columns={'R_Values': 'Value'})
R_metrics_df = R_metrics_df.rename(columns={'R_Values': 'Value'})

# Combine the metrics into a single DataFrame
combined_metrics_df = pd.concat([NN_metrics_df, T_metrics_df, R_metrics_df], ignore_index=True)

# Display the combined DataFrame
print("\nCombined Metrics DataFrame:\n", combined_metrics_df)

# List of unique metrics
metrics = combined_metrics_df['Metric'].dropna().unique()

# Plot each metric for comparison across models
for metric in metrics:
    plt.figure(figsize=(10, 6))
    metric_data = combined_metrics_df[combined_metrics_df['Metric'] == metric]
    plt.bar(metric_data['Model'], metric_data['Value'], color=['blue', 'green', 'orange'])
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title(f'Comparison of {metric} Across Models')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Determine the best model based on each metric
best_metrics = {}
# Pivot the combined metrics DataFrame to structure it for comparison
pivot_df = combined_metrics_df.pivot(index='Metric', columns='Model', values='Value')

# Loop through each metric to determine the best model
for metric, row in pivot_df.iterrows():
    if metric == 'R2':
        best_model = row.idxmax()  # Higher is better for R2
    else:
        best_model = row.idxmin()  # Lower is better for MAE, MSE, RMSE, MedAE
    best_metrics[metric] = best_model

# Count how many times each model is the best
best_model_counts = pd.Series(best_metrics).value_counts()

# Identify the overall best model
overall_best_model = best_model_counts.idxmax()

# Display conclusion
print("\n**Conclusion**")
print(f"Based on the displayed metrics, the best performing model overall is: {overall_best_model}.")
print(f"The {overall_best_model} model performed best in the following metrics:")

# Display which metrics each model performed best in
for metric, model in best_metrics.items():
    print(f"- {metric}: {model}")

print("\nSummary:")
print(f"The {overall_best_model} model is recommended for further use and analysis due to its superior performance in the majority of key metrics.")
