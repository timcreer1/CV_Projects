import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

# Load data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csvjh')

# Randomly select 100 rows from the DataFrame
#df = df.sample(n=10)

# Efficiently apply literal_eval
df['Spectrum'] = df['Spectrum'].apply(ast.literal_eval)
df['Age_encoded'] = LabelEncoder().fit_transform(df['Age'])

# Prepare features and labels
spectrum_list = df['Spectrum'].tolist()
age_encoded = df['Age_encoded'].values

def prepare_data(spectrum_list, age_encoded):
    n = len(spectrum_list)
    m = len(spectrum_list[0]) + 1  # Assuming all spectra have the same length
    X = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        X[i, :-1] = spectrum_list[i]
        X[i, -1] = age_encoded[i]
    return X

X = prepare_data(spectrum_list, age_encoded)
y = LabelEncoder().fit_transform(df['Species'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Add channel and width dimensions for CNN input
X_train = X_train[..., np.newaxis, np.newaxis]
X_test = X_test[..., np.newaxis, np.newaxis]

# Define models
def create_model(input_shape, model_type):
    model = models.Sequential()
    if model_type == 'ResNet':
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        for filters in [128, 256, 512]:
            x = layers.Conv2D(filters, (3, 1), activation='relu', padding='same')(x)
            x = layers.Conv2D(filters, (3, 1), activation='relu', padding='same')(x)
            x = layers.MaxPooling2D((2, 1))(x)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
        model = models.Model(inputs, outputs)
    elif model_type == 'InceptionTime':
        model.add(layers.Conv2D(32, (3, 1), activation='relu', padding='same', input_shape=input_shape))
        for filters in [32, 64, 128, 256]:
            model.add(layers.Conv2D(filters, (3, 1), activation='relu', padding='same'))
            model.add(layers.Conv2D(filters, (3, 1), activation='relu', padding='same'))
            model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))
    elif model_type == 'FCN':
        model.add(layers.Conv2D(128, (8, 1), activation='relu', input_shape=input_shape, padding='same'))
        model.add(layers.Conv2D(256, (5, 1), activation='relu', padding='same'))
        model.add(layers.Conv2D(128, (3, 1), activation='relu', padding='same'))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))
    elif model_type == 'Time-CNN':
        model.add(layers.Conv2D(6, (7, 1), activation='sigmoid', input_shape=input_shape, padding='valid'))
        model.add(layers.Conv2D(12, (7, 1), activation='sigmoid', padding='valid'))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(len(np.unique(y_train)), activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create models
models_dict = {name: create_model(X_train.shape[1:], name) for name in ['ResNet', 'InceptionTime', 'FCN', 'Time-CNN']}

# Prepare results DataFrame
results = pd.DataFrame(columns=['Model', 'Task', 'Accuracy'])

class CustomEarlyStopping(Callback):
    def __init__(self, patience=20, min_delta=0):
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_weights = None
        self.wait = 0
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_loss")
        if current is None:
            return

        if np.less(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)
                print(f"\nEarly stopping at epoch {epoch + 1} with best val_loss: {self.best}")

# Train and evaluate models
for model_name, model in tqdm(models_dict.items(), desc="Training and Evaluating Models", unit="model"):
    print(f"\nTraining {model_name} for Species Prediction...")
    with tqdm(total=100, desc=f"{model_name} Training Progress", unit="epoch", position=1, leave=True) as model_pbar:
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=0,
                            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: model_pbar.update(1))])
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy (Species Prediction): {accuracy:.2f}")
    results = pd.concat([results, pd.DataFrame({'Model': [model_name], 'Task': ['Species Prediction'], 'Accuracy': [accuracy]})], ignore_index=True)

# Save and display results
results.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Deep Learning Based/model_results_convolutional_based.csv', index=False)
print("Model results have been saved to model_results_deep_learning_based.csv")
print(results)
