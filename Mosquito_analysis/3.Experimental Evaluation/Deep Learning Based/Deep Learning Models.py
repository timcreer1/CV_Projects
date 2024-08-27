import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Load and preprocess data
df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/cleaned_matrix_downsampled.csvjh')
df['Spectrum'] = df['Spectrum'].apply(ast.literal_eval)
df['Age_encoded'] = LabelEncoder().fit_transform(df['Age'])
X = np.array([spectrum + [age] for spectrum, age in zip(df['Spectrum'], df['Age_encoded'])])
y = LabelEncoder().fit_transform(df['Species'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train = X_train[..., np.newaxis, np.newaxis]
X_test = X_test[..., np.newaxis, np.newaxis]

# Model creation functions
def create_resnet(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(128, (8, 1), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (5, 1), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 1))(x)

    for _ in range(3):  # Residual blocks
        shortcut = x
        x = layers.Conv2D(128, (8, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (5, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
        x = layers.add([x, shortcut])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(output_shape, activation='softmax')(x)
    return models.Model(inputs, outputs)

def create_inceptiontime(input_shape, output_shape):
    inputs = layers.Input(shape=input_shape)
    for _ in range(5):  # Inception modules for ensemble
        x = layers.Conv2D(32, (40, 1), activation='relu', padding='same')(inputs)
        x = layers.Conv2D(32, (40, 1), activation='relu', padding='same')(x)
        x = layers.Conv2D(32, (40, 1), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        shortcut = x
        x = layers.Conv2D(32, (40, 1), activation='relu', padding='same')(x)
        x = layers.add([x, shortcut])

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(output_shape, activation='softmax')(x)
    return models.Model(inputs, outputs)

def create_fcn(input_shape, output_shape):
    model = models.Sequential([
        layers.Conv2D(128, (8, 1), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(256, (5, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 1), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(output_shape, activation='softmax')
    ])
    return model

def create_time_cnn(input_shape, output_shape):
    model = models.Sequential([
        layers.Conv2D(6, (7, 1), activation='sigmoid', padding='valid', input_shape=input_shape),
        layers.Conv2D(12, (7, 1), activation='sigmoid', padding='valid'),
        layers.AveragePooling2D(pool_size=(3, 1)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(output_shape, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

# Create models
models_dict = {
    'ResNet': create_resnet(X_train.shape[1:], len(np.unique(y_train))),
    'InceptionTime': create_inceptiontime(X_train.shape[1:], len(np.unique(y_train))),
    'FCN': create_fcn(X_train.shape[1:], len(np.unique(y_train))),
    'Time-CNN': create_time_cnn(X_train.shape[1:], len(np.unique(y_train)))
}

# Custom Early Stopping Callback
class CustomEarlyStopping(callbacks.Callback):
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
results = pd.DataFrame(columns=['Model', 'Task', 'Accuracy'])
for model_name, model in tqdm(models_dict.items(), desc="Training and Evaluating Models", unit="model"):
    print(f"\nTraining {model_name} for Species Prediction...")
    history = model.fit(
        X_train, y_train,
        epochs=2000 if model_name != 'InceptionTime' else 1500,  # Adjust epochs as per screenshot
        validation_data=(X_test, y_test),
        verbose=0,
        callbacks=[CustomEarlyStopping(patience=20)]
    )
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy (Species Prediction): {accuracy:.2f}")
    results = results.append({'Model': model_name, 'Task': 'Species Prediction', 'Accuracy': accuracy}, ignore_index=True)

# Save and display results
results.to_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/3.Experimental Evaluation/Deep Learning Based/model_results_convolutional_based.csv', index=False)
print("Model results have been saved to model_results_convolutional_based.csv")
print(results)
