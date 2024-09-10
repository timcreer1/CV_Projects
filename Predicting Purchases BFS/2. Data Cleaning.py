# Imports
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_colwidth = 200

# Import data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/BFS.csv')
print(df.head())

# Making all null values 0 and converting column to integers
df['Product_Category_2'] = df['Product_Category_2'].fillna(0).astype(int)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0).astype(int)

# Making categorical columns into integer encoded columns
columns_to_encode = ['Product_ID', 'User_ID', 'Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years',
                     'Product_Category_1', 'Product_Category_2', 'Product_Category_3']
df[columns_to_encode] = df[columns_to_encode].apply(lambda x: pd.factorize(x)[0])
df2 = df.copy()
print(df2.head())

# Viewing the correlation between variables
plt.figure(figsize=(8, 5))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Dropping specified columns as low correlation to purchase
df2.drop(columns=['User_ID', 'Product_ID', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status'], inplace=True)

# Calculate VIF
vif = [variance_inflation_factor(df2.values, i) for i in range(df2.shape[1])]
vif_df = pd.DataFrame({'variable': df2.columns, 'vif': vif})

# Calculate Tolerance
tolerance = 1 / vif_df['vif']
tolerance_df = pd.DataFrame({'variable': df2.columns, 'tolerance': tolerance})

# Display the results
result_table = pd.merge(vif_df, tolerance_df, on='variable')
print("\nVariance Inflation Factor and Tolerance:\n")
print(result_table)

# Interpretation
if (result_table['vif'] <= 5).all() and (result_table['tolerance'] >= 0.2).all():
    print("\nNo variables with VIF > 5 or Tolerance < 0.2 detected.")
else:
    print("\nSome variables have VIF > 5 or Tolerance < 0.2.")

# Save the cleaned DataFrame
df2.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/cleaned_data.csv', index=False)
print("\nCleaned data saved as 'cleaned_data.csv' in the specified directory.")