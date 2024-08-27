# 🦟 Time Series Mining Approaches for Malaria Vector Prediction on Mid-Infrared Spectroscopy Data

This project replicates a segment of a scientific research paper that investigates using mid-infrared spectroscopy (MIRS) data as input for efficient time series classification methods to predict the species and age of malaria mosquitoes.

## 🌐 Resources

- [GitHub Repository](https://github.com/SimonAB/DL-MIRS_Siria_et_al?tab=readme-ov-file)
- [Published Paper](https://datascience.codata.org/articles/10.5334/dsj-2024-025?fbclid=IwY2xjawEg7MFleHRuA2FlbQIxMAABHa4RHi_KrsCsxhLPxNcjZzFAJWPEO8EzEnopxWnPMZTsSpqe3Oe35ijmmQ_aem_nhPzwgnCNAqox8N4yqRwMg#B42)

## 📄 Overview

The project evaluates 14 algorithms across four time series mining approaches: feature-based, interval-based, convolutional-based, and deep learning methods. The focus is on predicting mosquito species using raw data after preprocessing, as computational limitations prevented age prediction and the use of different data feature sets.

### 🔍 Key Components

- **Data**:  
  Contains a link to the original dataset and notes for the dataset.

- **1. Extracting Data**:  
  The code processes mosquito spectral data files by:
  - Ensuring consistency in file naming conventions
  - Reading and downsampling spectral data
  - Storing the processed data in a pandas DataFrame

- **2. EDA Preparation (downsampled)**:  
  This step involves:
  - Renaming and filtering columns
  - Mapping species codes to full names
  - Categorising age and splitting the data into training and testing sets
  - Visualising the distribution of species and age
  - Applying Z-normalisation and rounding to the spectra
  - Checking for invalid values and saving the cleaned data

- **3. Experimental Evaluation**:  
  This folder contains four different time series mining approaches, with models evaluated using accuracy metrics:

  - **Convolutional-Based**:  
    The code preprocesses the data by encoding labels, applies Rocket and MiniRocket transformations, and trains RidgeClassifierCV models for species prediction.
  
  - **Deep Learning-Based**:  
    The code preprocesses the data by converting the spectral data from strings to lists, encodes the labels, and prepares the input features, reshaping them to fit the input requirements of CNN models (ResNet, InceptionTime, FCN, and Time-CNN). Early stopping is applied during training.
  
  - **Feature-Based**:  
    The code encodes labels, splits the data into training and testing sets, and trains and evaluates classification models using KNN, Logistic Regression, SVM, Random Forest, and XGBoost with specified parameters.
  
  - **Interval-Based**:  
    The code preprocesses the data by converting the spectral data from strings to lists, encoding mosquito age as numerical labels, and prepares the input features for use with interval-based time series classifiers (Time Series Forest, Canonical Interval Forest, and Diverse Representation CIF).

- **4. Conclusion**:  
  The code loads the evaluation results of the four models, concatenates them into a single DataFrame, and visualises the accuracy of each model across categories using bar charts.

## 🛠️ Requirements

Ensure the following packages are installed before running the code:

```bash
pip install numpy scipy pandas sktime torch tensorflow xgboost matplotlib seaborn tqdm
