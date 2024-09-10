# Necessary imports
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, mean_squared_log_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import cleaned data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/cleaned_data.csv')

# Reducing the sample size and shuffling the data
#df = df.sample(frac=1).iloc[:10000]

# Split the data into training and test sets
NN_y = df['Purchase'].values
NN_X = df.drop('Purchase', axis=1).values

# Scaling the features
scaler = MinMaxScaler()
NN_X = scaler.fit_transform(NN_X)

# Splitting data into training and testing sets
NN_X_train, NN_X_test, NN_y_train, NN_y_test = train_test_split(NN_X, NN_y, test_size=0.2, random_state=30)

# Define the model architecture
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(NN_X_train.shape[1],), kernel_regularizer=l1(0.0005)))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.0005)))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.0005)))
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mean_squared_error')

# Define callbacks for early stopping and learning rate reduction
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.6, verbose=1, mode='min', min_lr=0.0001)

# Start timer
start_time = time.time()

# Train the model
history = model.fit(NN_X_train, NN_y_train, epochs=100, batch_size=64,
                    validation_data=(NN_X_test, NN_y_test),
                    callbacks=[early_stop, reduce_lr])

# Stop timer and calculate elapsed time
elapsed_time = time.time() - start_time

# Plot the training loss
model_loss = pd.DataFrame(history.history)
model_loss.plot()
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Evaluate the model
NN_y_pred = model.predict(NN_X_test)
NN_y_pred = NN_y_pred.flatten()  # Flatten predictions to ensure correct format

# Calculating evaluation metrics
NN_metrics = {
    'MAE': mean_absolute_error(NN_y_test, NN_y_pred),
    'MSE': mean_squared_error(NN_y_test, NN_y_pred),
    'R2': r2_score(NN_y_test, NN_y_pred),
    'RMSE': np.sqrt(mean_squared_error(NN_y_test, NN_y_pred)),  # Changed to RMSE for proper interpretation
    'MedAE': median_absolute_error(NN_y_test, NN_y_pred)
}

# Displaying metrics
NN_metrics_df = pd.DataFrame(list(NN_metrics.items()), columns=['Metric', 'NN_Values'])
NN_metrics_df["NN_Values"] = NN_metrics_df["NN_Values"].round(2)

#Save the model
NN_metrics_df.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/NN_metrics_df.csv', index=False)
print("\nNuerak Network results data saved as 'NN_metrics_df.csv' in the specified directory.")


print(model.summary())
print(NN_metrics_df)
# Print elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))

# Residual analysis for Neural Network Model
# Calculating residuals
NN_residuals = NN_y_test - NN_y_pred

# 1. Histogram of Residuals
plt.figure(figsize=(8, 5))
plt.hist(NN_residuals, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals - Neural Network')
plt.show()

# 2. Residual Plot (Residuals vs Predicted Values)
plt.figure(figsize=(10, 6))
plt.scatter(NN_y_pred, NN_residuals, alpha=0.7, edgecolors='k')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values - Neural Network')
plt.show()

# 3. QQ Plot to Check Normality
plt.figure(figsize=(8, 6))
import scipy.stats as stats
stats.probplot(NN_residuals, dist="norm", plot=plt)
plt.title('QQ Plot of Residuals - Neural Network')
plt.show()

# 4. Residuals vs. Fitted Plot (Checking Homoscedasticity)
plt.figure(figsize=(10, 6))
import seaborn as sns
sns.residplot(x=NN_y_pred, y=NN_residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values - Neural Network')
plt.axhline(0, linestyle='--', color='red', linewidth=1)
plt.show()




