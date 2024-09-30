import time
from tqdm import tqdm # Cool progress bar

import numpy as np
import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf

from keras import backend as K
from tensorflow.keras import layers, models, metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt
import seaborn as sns

from utilities import *
from utilities import plot_training_history

SEED = 458
RNG = np.random.default_rng(SEED) 


# Import the mnist dataset
datasets = load_mnist(verbose=0)
X_train, y_train = datasets['X_train'], datasets['y_train']
X_val,   y_val   = datasets['X_val'],   datasets['y_val']
X_test,  y_test  = datasets['X_test'],  datasets['y_test']

X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0).astype('int32')

del datasets, X_val, y_val # Good to reduce uneccesary RAM usage



# Normalize the pixels to values [0,1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float') / 255.0


# One-hot encoding
y_train_encoded = to_categorical(y_train, num_classes=10)
y_test_encoded = to_categorical(y_test, num_classes=10)


fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()



# Iterate through each class (from 0 to 9)
for i in range(10):
    # Find the first sample where the encoded label has a '1' at the i-th position
    img = X_train[np.argmax(y_train_encoded, axis=1) == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='viridis')
    ax[i].set_title(f"Label: {i}")

# Remove x and y ticks for all subplots
for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

plt.tight_layout()
plt.show()



# Confirming that our pixels are values [0,1]
print("Min value in X_train:", X_train.min())
print("Max value in X_train:", X_train.max())

unique_labels = np.unique(X_train)
print("Unique labels in y_train:", unique_labels)



# Created a f1 metric function with K from keras
def f1_metric(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_score = 2*(precision*recall) / (precision+recall+K.epsilon())

    return f1_score




# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_encoded, test_size=1/6, random_state=SEED)


# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(30, activation='sigmoid'),
    layers.Dense(10, activation='sigmoid')
])


# Compile the model
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['accuracy', f1_metric])


# Train the model
history = model.fit(
    X_train_split, y_train_split,
    epochs=5,
    batch_size=10,
    validation_data=(X_val_split, y_val_split))



# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_encoded, test_size=1/6, random_state=SEED)

lr = 0.001

# Define our model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='sigmoid'),
    layers.Dense(10, activation='softmax')
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=lr),
                loss = 'categorical_crossentropy',
                metrics=['accuracy', f1_metric])


# Train the model
history = model.fit(X_train_split, y_train_split,
                    epochs=20,
                    batch_size=32,
                    validation_data=(X_val_split, y_val_split))


plot_training_history(history)




from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=1/6, random_state=SEED)


# Reshape the datasets
X_train_reshaped = X_train_split.reshape(-1, 28 * 28)
X_val_reshaped = X_val_split.reshape(-1, 28 * 28)
X_test_reshaped = X_test.reshape(-1, 28 * 28)


# Create a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


# Train the model
clf.fit(X_train_reshaped, y_train_split)


# Predict on the test set
y_test_pred = clf.predict(X_test_reshaped)


# Compute metrics
acc = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, average='macro')
report = classification_report(y_test, y_test_pred)

print(f"Accuracy: {acc * 100:.2f}%")
print(f"Macro F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(report)


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Display the confusion matrix
plt.figure(figsize=(10, 10))
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix")
plt.show()
