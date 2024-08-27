import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sktime.transformations.panel.rocket import Rocket, MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
import torch

# Check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csv')

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

# Define the Rocket and MiniRocket transformers with specified parameters
print("Defining Rocket and MiniRocket transformers...")
rocket_transformer = Rocket(num_kernels=10000, random_state=42)
minirocket_transformer = MiniRocket(num_kernels=10000, max_dilations_per_kernel=32, features_per_kernel=4, random_state=42)

# Transform the data with Rocket and MiniRocket
print("Transforming data with Rocket and MiniRocket...")
X_train_rocket = rocket_transformer.fit_transform(X_train)
X_test_rocket = rocket_transformer.transform(X_test)
X_train_minirocket = minirocket_transformer.fit_transform(X_train)
X_test_minirocket = minirocket_transformer.transform(X_test)

# Define a dictionary to hold the classifiers and their configurations
print("Defining classifiers...")
classifiers = {
    "ROCKET": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    "MiniRocket": RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
}

# Function to train and evaluate a single classifier
def train_and_evaluate(classifier_name, X_train_transformed, X_test_transformed, y_train, y_test):
    print(f"\nTraining {classifier_name} for Species Prediction...")
    classifier = classifiers[classifier_name]
    classifier.fit(X_train_transformed, y_train)
    y_pred = classifier.predict(X_test_transformed)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{classifier_name} Accuracy (Species Prediction): {accuracy:.2f}")
    return pd.DataFrame({'Model': [classifier_name], 'Task': ['Species Prediction'], 'Accuracy': [accuracy]})

# Train and evaluate each classifier for species prediction
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
print("Model results have been saved.")

# Display the results DataFrame
print("Displaying results...")
print(results)
