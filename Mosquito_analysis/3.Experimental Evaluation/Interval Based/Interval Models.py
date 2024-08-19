import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sktime.classification.interval_based import TimeSeriesForestClassifier, CanonicalIntervalForest, DrCIF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csv')

# Randomly select 1000 rows from the DataFrame
# df = df.sample(n=1000, random_state=42)

# Convert the Spectrum column to lists
print("Converting Spectrum column to lists...")
df['Spectrum'] = df['Spectrum'].apply(ast.literal_eval)

# Encode age labels
print("Encoding age labels...")
label_encoder_age = LabelEncoder()
df['Age_encoded'] = label_encoder_age.fit_transform(df['Age'])

# Combine Spectrum and Age_encoded to create feature matrix X
print("Combining Spectrum and Age_encoded to create feature matrix X...")
X_spectrum = np.array(df['Spectrum'].tolist())
X_age = df['Age_encoded'].values.reshape(-1, 1)
X = np.hstack((X_spectrum, X_age))

# Convert X to the required format for sktime
print("Converting X to the required format for sktime...")
X = np.array([pd.Series(x) for x in tqdm(X, desc="Formatting X")])

# Assuming the 'Species' column is the target variable for species prediction
print("Encoding species labels...")
y_species = df['Species']
label_encoder_species = LabelEncoder()
y_species_encoded = label_encoder_species.fit_transform(y_species)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_species_encoded, test_size=0.30, random_state=42)

# Define a dictionary to hold the models and their configurations
print("Defining models...")
models = {
    "Time Series Forest (TSF)": TimeSeriesForestClassifier(n_estimators=50, random_state=42),
    "Canonical Interval Forest (CIF)": CanonicalIntervalForest(n_estimators=50, random_state=42),
    "Diverse Representation CIF (DrCIF)": DrCIF(n_estimators=50, random_state=42)
}

# DataFrame to store results
results = pd.DataFrame(columns=['Model', 'Task', 'Accuracy'])

# Train and evaluate each model for species prediction with progress tracking
for model_name, model in tqdm(models.items(), desc="Training and Evaluating Models", unit="model"):
    print(f"\nTraining {model_name} for Species Prediction...")

    # Initialize model-specific progress bar
    with tqdm(total=100, desc=f"{model_name} Progress", unit="%", position=1, leave=True) as model_pbar:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy (Species Prediction): {accuracy:.2f}")
        results = pd.concat(
            [results, pd.DataFrame({'Model': [model_name], 'Task': ['Species Prediction'], 'Accuracy': [accuracy]})],
            ignore_index=True)
        model_pbar.update(100)  # Update progress bar to 100% completion for this model

# Save the results to a CSV file
print("Saving results to CSV file...")
results.to_csv('/Users/creer/PycharmProjects/2024_Projects/venv/Mosquito_analysis/Data/model_results_interval_based.csv', index=False)
print("Model results have been saved to model_results_interval_based.csv")

# Display the results DataFrame
print("Displaying results...")
print(results)
