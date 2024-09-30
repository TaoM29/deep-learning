import time
from tqdm import tqdm 

import numpy as np
import pandas as pd
import h5py

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from keras import backend as K
import tensorflow.keras as ks

from keras.utils import to_categorical
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout, Add, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from keras.regularizers import l2

SEED = 458
RNG = np.random.default_rng(SEED) 

from utilities import *




# Loading the dataset
datasets = load_mnist(verbose=0)
X_train, y_train = datasets['X_train'], datasets['y_train']
X_val,   y_val   = datasets['X_val'],   datasets['y_val']
X_test,  y_test  = datasets['X_test'],  datasets['y_test']

X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0).astype('int32')

del datasets, X_val, y_val # Good to reduce uneccesary RAM usage




# Reshape data to account for color channel
X_train = np.expand_dims(X_train, -1)
X_test  = np.expand_dims(X_test, -1)

# Normalizing input between [0,1]
X_train = X_train.astype("float32")/np.max(X_train)
X_test  = X_test.astype("float32")/np.max(X_test)

# Converting targets from numbers to categorical format
y_train = ks.utils.to_categorical(y_train, len(np.unique(y_train)))
y_test  = ks.utils.to_categorical(y_test, len(np.unique(y_test)))


# Check the shape of our data
print('Training:', X_train.shape, y_train.shape)
print('Test Set:', X_test.shape, y_test.shape )


# Check that our training values have been normalized
print("Min value in X_train:", X_train.min())
print("Max value in X_train:", X_train.max())
print('Unique values in X_train:\n', np.unique(X_train))




# Visualize each label
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

# Iterate through each class (from 0 to 9)
for i in range(10):
    # Find the first sample where the encoded label has a '1' at the i-th position
    img = X_train[np.argmax(y_train, axis=1) == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='viridis')
    ax[i].set_title(f"Label: {i}")

# Remove x and y ticks for all subplots
for axis in ax:
    axis.set_xticks([])
    axis.set_yticks([])

plt.tight_layout()
plt.show()



# F1-score implementation with K from keras
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())

    return f1_val




# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=1/6, random_state=SEED)


# Defining the LeNet CNN architecture
def leNet_model():
    model = Sequential()

    model.add(Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    return model


model = leNet_model()



# Compile the model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', f1_score])


# Training our model
history = model.fit(X_train_split, y_train_split,
                    batch_size = 128,
                    epochs = 40,
                    validation_data = (X_val_split, y_val_split))


plot_training_history(history)


# Evalute our model on the test data
results = model.evaluate(X_test, y_test)

print(f"Test Loss: {results[0]:.4f} ")
print(f"Test Accuracy: {results[1]:.4f} ")
print(f"Test F1 Score: {results[2]:.4f} ")



dataset_path = './student_dataset_CIFAR10.h5'

with h5py.File(dataset_path,'r') as f:
    print('Datasets in file:', list(f.keys()))
    X_train = np.asarray(f['X_train'])
    y_train = np.asarray(f['y_train'])
    X_test  = np.asarray(f['X_test'])
    print('Nr. train images: %i'%(X_train.shape[0]))
    print('Nr. test images: %i'%(X_test.shape[0]))

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Normalizing input between [0,1]
X_train = X_train.astype("float32")/np.max(X_train)
X_test  = X_test.astype("float32")/np.max(X_test)


# One hot encoding
y_train_encoded = to_categorical(y_train, num_classes = 10)


# Visualization for each label
def plot_sample_per_label(images, labels, label_names):
    # Assuming labels is a 1D array with class indices
    unique_labels = np.unique(labels)

    plt.figure(figsize=(15, 5))
    for i, label in enumerate(unique_labels):
        # Find the index of the first occurrence of this label
        idx = np.where(labels == label)[0][0]

        plt.subplot(1, len(unique_labels), i+1)
        plt.imshow(images[idx])
        plt.title(label_names[label])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

plot_sample_per_label(X_train, y_train, label_names)


# Visalization of the distribution
unique, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(12, 7))
sns.barplot(x=unique, y=counts, palette="viridis")
plt.title('Distribution of labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.xticks(ticks=range(len(unique)), labels=label_names)
plt.tight_layout()
plt.show()


# F1-score implementation with K from keras
def custom_f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision*recall) / (precision+recall+K.epsilon())

    return f1_val


from keras.utils import get_custom_objects

get_custom_objects().update({"custom_f1_score": custom_f1_score})


# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train_encoded, test_size=1/6, random_state=SEED)


from tensorflow.keras.regularizers import l2


# Defining our architecture, Inspiration from VGG
def cifar10_model():
    model = Sequential()

    # Convolutional Layers
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same' ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same' ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same' ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same' ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3,3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3,3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.6))


    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))
    model.add(Dense(10, activation='softmax'))

    return model


model = cifar10_model()


opt = Adam(learning_rate=0.001)

# Compile the model
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy', 'custom_f1_score'])



# Saving our best model
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')


# Stop when not improving accuracy
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience=10, verbose=1, restore_best_weights=True)


# Train the model
history = model.fit(X_train_split, y_train_split,
                    epochs=40,
                    batch_size = 32,
                    validation_data=(X_val_split, y_val_split),
                    callbacks=[checkpoint, early_stopping],
                    verbose=1)

plot_training_history(history)


from keras.models import load_model

best_model = load_model('best_model.h5', custom_objects={'custom_f1_score': custom_f1_score})

loss, accuracy, f1_score_val = best_model.evaluate(X_val_split, y_val_split)

print(f"Validation loss: {loss:.4f} ")
print(f"Validation accuracy: {accuracy:.4f} ")
print(f"Validation f1 score: {f1_score_val:.4f} ")

# Prediction with our best model
prediction = best_model.predict(X_test)
flat_prediction = np.argmax(prediction, axis=1) # Flatten softmax predictions
submissionDF = pd.DataFrame()
submissionDF['ID'] = range(len(flat_prediction)) # The submission csv file must have an row index column called 'ID'
submissionDF['Prediction'] = flat_prediction
submissionDF.to_csv('submission.csv', index=False) # Remember to store the dataframe to csv without the nameless index column.
