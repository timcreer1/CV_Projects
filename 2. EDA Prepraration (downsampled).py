import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import ast
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/matrix_downsampled.csv')
pd.set_option('display.max_columns', None)
print(df)
# Rename columns and drop unneeded ones
df.rename(columns={'Attribute_1': 'Species', 'Attribute_2': 'Country', 'Attribute_3': 'Age', 'Attribute_4': 'Status', 'Attribute_5': 'Experiment'}, inplace=True)
df.drop(df.columns[[0, 3, 5, 6, 7, 8]], axis=1, inplace=True)

# Update abbreviated terms to full names
species_code_mapping = {'AA': 'Anopheles arabiensis', 'AG': 'Anopheles gambiae', 'AC': 'Anopheles coluzzi'}
df['Species'] = df['Species'].replace(species_code_mapping)

# Species analysis
print('Total per species:')
print(df['Species'].value_counts())
print('\nPercentage split:')
print(df['Species'].value_counts(normalize=True) * 100)
print('\nTotal count of species:')
print(df['Species'].count())

# Drop rows with species 'Anopheles coluzzi'
df1 = df[df['Species'] != 'Anopheles coluzzi'].copy()

# Function to categorize age
def categorize_age(age):
    if isinstance(age, str) and age.endswith('D'):
        days = int(age[:-1])
        if 1 <= days <= 4:
            return '1-4 days'
        elif 5 <= days <= 10:
            return '5-10 days'
        elif 11 <= days <= 17:
            return '11-17 days'
    return 'Unknown'

df1.loc[:, 'Age'] = df1['Age'].apply(categorize_age)
df1 = df1[df1['Age'] != 'Unknown']

# Split the data into stratified training and test sets by age and species
train_df, test_df = train_test_split(df1, test_size=0.30, random_state=42, stratify=df1[['Species', 'Age']])
print("Training set shape:", train_df.shape)
print("Testing set shape:", test_df.shape)

# Graphs to show the split of the train and test
fig, axs = plt.subplots(2, 2, figsize=(14, 8), gridspec_kw={'wspace': 0.3, 'hspace': 0.5})

# Define a function to plot the bar charts
def plot_counts(ax, data, title, xlabel, palette):
    counts = data.value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=palette)
    for index, value in enumerate(counts.values):
        ax.text(index, value / 2, str(value), ha='center', va='center', color='white', fontsize=12)
    ax.set_title(title)
    ax.set_ylabel('Examples')
    ax.set_xlabel(xlabel)

# Plotting the data
plot_counts(axs[0, 0], train_df['Species'], '(a) Train', 'Species', 'Pastel1')
plot_counts(axs[0, 1], test_df['Species'], '(b) Test', 'Species', 'Pastel1')
plot_counts(axs[1, 0], train_df['Age'], '(c) Train', 'Age', 'Pastel2')
plot_counts(axs[1, 1], test_df['Age'], '(d) Test', 'Age', 'Pastel2')
plt.show()

print(df1.dtypes)
print(df1.head())

# Function to Z-normalize a list of values
def z_norm(spectrum):
    spectrum = np.array(spectrum)
    if spectrum.size == 0 or spectrum.std() == 0:
        return np.zeros_like(spectrum)
    return (spectrum - spectrum.mean()) / spectrum.std()

# Apply Z-normalization to the 'Spectrum' column
df1['Spectrum'] = df1['Spectrum'].apply(lambda x: z_norm(list(map(float, x.split(',')))))

# Function to round a list of values to 6 decimal places
def round_spectrum(spectrum, decimals=8):
    return [round(float(value), decimals) for value in spectrum]

# Apply rounding to the 'Spectrum' column
df1['Spectrum'] = df1['Spectrum'].apply(round_spectrum)

# Check for NaN or infinite values in Spectrum
print("Checking for NaN or infinite values in Spectrum column after normalization and rounding...")
if df1['Spectrum'].apply(lambda x: np.any(np.isnan(x) | np.isinf(x))).any():
    raise ValueError("Spectrum column contains NaN or infinite values")

# Save the DataFrame to a CSV file
output_file_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csv'
df1.to_csv(output_file_path, index=False)
print(f"Matrix has been saved to {output_file_path}.")

print(df1)
