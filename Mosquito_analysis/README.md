# Time Series Mining Approaches for Malaria Vector Prediction on Mid-Infrared Spectroscopy Data Project
This project involves replicating a segment of a scientific research paper that investigates the use of mid-infrared spectroscopy (MIRS) data as input for efficient time series classification methods to predict the species and age of malaria mosquitoes.

## Resources
- [GitHub repository](https://github.com/SimonAB/DL-MIRS_Siria_et_al?tab=readme-ov-file)
- [Published Paper](https://datascience.codata.org/articles/10.5334/dsj-2024-025?fbclid=IwY2xjawEg7MFleHRuA2FlbQIxMAABHa4RHi_KrsCsxhLPxNcjZzFAJWPEO8EzEnopxWnPMZTsSpqe3Oe35ijmmQ_aem_nhPzwgnCNAqox8N4yqRwMg#B42)

## Overview
The project evaluates 14 algorithms across four time series mining approaches: feature-based, interval-based, convolutional-based, and deep learning methods. The focus is on predicting mosquito species using the raw data after preprocessing, as computational limitations prevented age prediction and the use of different data feature sets.

**1. Extracting Data**: The code processes mosquito spectral data files by ensuring consistency in file naming conventions, reading and downsampling spectral data, and storing the processed data in a pandas DataFrame.

**2. EDA Preparation (downsampled)**: The code renames and filters columns, maps species codes to full names, categorises age, splits the data into training and testing sets, visualises the distribution of species and age, applies Z-normalisation and rounding to the spectra, checks for invalid values, and saves the cleaned data.

**3. Experimental Evaluation**: This folder contains four different time series mining approaches, with models evaluated using accuracy metrics:
- **Convolutional-Based**: The code preprocesses the data by encoding labels, applies Rocket and MiniRocket transformations, and then trains RidgeClassifierCV models for species prediction.
- **Deep Learning-Based**: The code preprocesses the data by converting the spectral data from strings to lists, encodes the labels, and then prepares the input features, reshaping them to fit the input requirements of the CNN models ResNet, InceptionTime, FCN, and Time-CNN. Early stopping is used on the models during training.
- **Feature-Based**: The code encodes labels, splits the data into training and testing sets, and trains and evaluates classification models using KNN, Logistic Regression, SVM, Random Forest, and XGBoost with specified parameters.
- **Interval-Based**: The code preprocesses the data by converting the spectral data from strings to lists and encoding the mosquito age as numerical labels, and then prepares the input features by combining the spectral data with the encoded age, reshaping them for use with the interval-based time series classifiers Time Series Forest, Canonical Interval Forest, and Diverse Representation CIF.

**4. Conclusion**: The code loads 4 model evaluation results, concatenates them into a single DataFrame, and then visualises the accuracy of each model across these categories in a series of bar charts.

## Requirements
Before running the code in this project, make sure you have installed the following packages:

* numpy
* scipy
* os
* pathlib
* pandas
* zipfile
* time
* tqdm
* matplotlib
* seaborn
* re
* ast
* sklearn
* sktime
* joblib
* torch
* xgboost
* tensorflow

You can install these packages using pip or conda. The code was run using python 3.11
