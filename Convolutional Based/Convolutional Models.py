import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csv')

# Testing on small sample not 51000 rows
#df = df.sample(n=100, random_state=42)

# Convert the Spectrum column to lists
print("Converting Spectrum column to lists...")
df['Spectrum'] = df['Spectrum'].apply(ast.literal_eval)

# Encode age labels
print("Encoding age labels...")
label_encoder_age = LabelEncoder()
df['Age_encoded'] = label_encoder_age.fit_transform(df['Age'])

# Combine Spectrum and Age_encoded to create feature matrix X
print("Combining Spectrum and Age_encoded to create feature matrix X...")
X_spectrum = np.array(df['Spectrum'].tolist(), dtype=np.float32)
X_age = df['Age_encoded'].values.reshape(-1, 1).astype(np.float32)
X = np.hstack((X_spectrum, X_age))

# Convert X to the required format for sktime
print("Converting X to the required format for sktime...")
X = pd.DataFrame({"ts": [pd.Series(x) for x in X]})

# Assuming the 'Species' column is the target variable for species prediction
print("Encoding species labels...")
y_species = df['Species']
label_encoder_species = LabelEncoder()
y_species_encoded = label_encoder_species.fit_transform(y_species)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_species_encoded, test_size=0.30, random_state=42)

# Verify the shapes before transformation
print(f"Shapes before transformation - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# Define the Rocket and MiniRocket transformers with reduced kernel sizes
print("Defining Rocket and MiniRocket transformers...")
rocket_transformer = Rocket(num_kernels=1000, random_state=42)
minirocket_transformer = MiniRocket(num_kernels=1000, random_state=42)

# Transform the data with progress tracking
print("Transforming data with Rocket and MiniRocket...")
with tqdm(total=4, desc="Data Transformation", unit="step") as pbar:
    X_train_rocket = rocket_transformer.fit_transform(X_train)
    pbar.update(1)
    X_test_rocket = rocket_transformer.transform(X_test)
    pbar.update(1)
    X_train_minirocket = minirocket_transformer.fit_transform(X_train)
    pbar.update(1)
    X_test_minirocket = minirocket_transformer.transform(X_test)
    pbar.update(1)

# Ensure transformed data is numpy arrays
X_train_rocket = np.asarray(X_train_rocket)
X_test_rocket = np.asarray(X_test_rocket)
X_train_minirocket = np.asarray(X_train_minirocket)
X_test_minirocket = np.asarray(X_test_minirocket)

# Check for infinite or very large values
def check_values(X, name):
    if np.isinf(X).any() or np.isnan(X).any():
        raise ValueError(f"{name} contains infinity or NaN values")
    if np.any(np.abs(X) > 1e10):
        raise ValueError(f"{name} contains values too large for dtype('float32')")

print("Checking for infinite or very large values in transformed data...")
check_values(X_train_rocket, "X_train_rocket")
check_values(X_test_rocket, "X_test_rocket")
check_values(X_train_minirocket, "X_train_minirocket")
check_values(X_test_minirocket, "X_test_minirocket")

# Normalize the data
print("Normalizing the data...")
scaler = StandardScaler()

# Function to normalize and check data
def normalize_and_check(X, scaler, name):
    X_normalized = scaler.fit_transform(X)
    check_values(X_normalized, name)
    return X_normalized

X_train_rocket = normalize_and_check(X_train_rocket, scaler, "X_train_rocket_normalized")
X_test_rocket = normalize_and_check(X_test_rocket, scaler, "X_test_rocket_normalized")
X_train_minirocket = normalize_and_check(X_train_minirocket, scaler, "X_train_minirocket_normalized")
X_test_minirocket = normalize_and_check(X_test_minirocket, scaler, "X_test_minirocket_normalized")

# Print the shapes of the datasets to verify consistency
print("Shapes of the datasets after transformation and normalization:")
print(f"X_train_rocket: {X_train_rocket.shape}, y_train: {y_train.shape}")
print(f"X_test_rocket: {X_test_rocket.shape}, y_test: {y_test.shape}")
print(f"X_train_minirocket: {X_train_minirocket.shape}, y_train: {y_train.shape}")
print(f"X_test_minirocket: {X_test_minirocket.shape}, y_test: {y_test.shape}")

# Define a dictionary to hold the classifiers and their configurations
print("Defining classifiers...")
classifiers = {
    "ROCKET": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    "MiniRocket": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
}

# DataFrame to store results
results = pd.DataFrame(columns=['Model', 'Task', 'Accuracy'])

# Function to train and evaluate a single classifier
def train_and_evaluate(classifier_name, X_train_transformed, X_test_transformed, y_train, y_test):
    print(f"\nTraining {classifier_name} for Species Prediction...")
    print(f"Shapes - X_train: {X_train_transformed.shape}, y_train: {y_train.shape}")
    classifier = classifiers[classifier_name]
    classifier.fit(X_train_transformed, y_train)
    y_pred = classifier.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{classifier_name} Accuracy (Species Prediction): {accuracy:.2f}")
    return pd.DataFrame({'Model': [classifier_name], 'Task': ['Species Prediction'], 'Accuracy': [accuracy]})

# Train and evaluate each classifier for species prediction with progress tracking
print("Training and evaluating classifiers...")
results_list = Parallel(n_jobs=-1, prefer="threads")(
    delayed(train_and_evaluate)(classifier_name, X_train_transformed, X_test_transformed, y_train, y_test)
    for classifier_name, X_train_transformed, X_test_transformed in tqdm([
        ("ROCKET", X_train_rocket, X_test_rocket),
        ("MiniRocket", X_train_minirocket, X_test_minirocket)
    ], desc="Training and Evaluating Classifiers", unit="classifier")
)

# Combine the results into a single DataFrame
print("Combining results...")
results = pd.concat(results_list, ignore_index=True)

# Save the results to a CSV file
print("Saving results to CSV file...")
results.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Convolutional Based/model_results_convolutional_based.csv', index=False)
print("Model results have been saved to model_results_interval_based.csv")

# Display the results DataFrame
print("Displaying results...")
print(results)
