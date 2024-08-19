import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('/Users/creer/PycharmProjects/2024_Projects/venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csv')

# Convert the Spectrum column to lists
df['Spectrum'] = df['Spectrum'].apply(ast.literal_eval)

# Encode age labels
label_encoder_age = LabelEncoder()
df['Age_encoded'] = label_encoder_age.fit_transform(df['Age'])

# Assuming the 'Species' column is the target variable for species prediction
X_spectrum = np.array(df['Spectrum'].tolist())
X_age = df['Age_encoded'].values.reshape(-1, 1)
X = np.hstack((X_spectrum, X_age))

y_species = df['Species']

# Encode species labels
label_encoder_species = LabelEncoder()
y_species_encoded = label_encoder_species.fit_transform(y_species)

# Split the data into training and testing sets
X_train, X_test, y_train_species, y_test_species = train_test_split(X, y_species_encoded, test_size=0.30, random_state=42)

# Initialize models based on the parameters in the screenshot
models = [
    ("K-Nearest Neighbors (KNN)", KNeighborsClassifier(n_neighbors=1, metric='manhattan')),
    ("Logistic Regression (LR)", LogisticRegression(C=5, penalty='l1', solver='liblinear')),
    ("Support Vector Machines (SVM)", SVC(C=5, kernel='linear')),
    ("Random Forest (RF)", RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=42)),
    ("XGBoost (XGB)", xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, gamma=0.1, max_depth=7, random_state=42))
]

# List to store results
results = []

# Train and evaluate each model
for name, model in models:
    model.fit(X_train, y_train_species)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test_species, y_pred)
    results.append([name, "Species Prediction", accuracy])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Task", "Accuracy"])
results_df

#save results
results_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Feature-based/model_results.csv', index=False)
