# IMPORT LIBRARIES
import tensorflow.keras as keras
import pandas as pd
import tensorflow as tf
import numpy as np
import PIL
import PIL.Image
import sklearn
import scipy
import pathlib
import time
import os
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds

celeba_bldr = tfds.builder('celeb_a')
celeba_bldr.download_and_prepare()
celeba = celeba_bldr.as_dataset(shuffle_files=False)

celeba_train_path = '/content/drive/MyDrive/train./content/CelebA_test'
celeba_validation_path = '/content/drive/MyDrive/validation./content/CelebA_validation'

celeba_train = tf.data.Dataset.load(celeba_train_path)
celeba_validation = tf.data.Dataset.load(celeba_validation_path)


# Checking both datasets structure
for data in celeba_train.take(1):
    print(data.keys())


    attributes = data['attributes']
    images = data['image']
    landmarks = data['landmarks']


for data in celeba_validation.take(1):
    print(data.keys())


    attributes = data['attributes']
    images = data['image']
    landmarks = data['landmarks']



BATCH_SIZE = 32
img_w, img_h = 64,64

# Precrocessing function 
def preprocess(example, size=(img_w, img_h), mode='train'):
  image = example['image']
  if mode == 'train':
    image_resized = tf.image.resize(image, size=size)

    return image_resized/255.0, image_resized/255.0
  

# Prepare the training dataset
train_ds = celeba_train.map(lambda x: preprocess(x))
train_ds = train_ds.batch(BATCH_SIZE)

# Prepare the validation dataset
validation_ds = celeba_validation.map(lambda x: preprocess(x))
validation_ds = validation_ds.batch(BATCH_SIZE)



# Visualize a few images from the datasets
def show_batch(dataset, num_images=9):

    plt.figure(figsize=(10, 10))

    for images, _ in dataset.take(1):
        for i in range(num_images):

            ax = plt.subplot(3, 3, i + 1)

            plt.imshow(images[i].numpy())
            plt.axis("off")


show_batch(train_ds)
show_batch(validation_ds)



from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_shape, latent_dim):


    input_img = Input(shape=input_shape, name='Image_input')

    flat_img = Flatten(name='Image_as_vector')(input_img)

    encoded = Dense(latent_dim, activation='relu', name='Encode')(flat_img)

    decoded = Dense(input_shape[0] * input_shape[1] * input_shape[2], activation='sigmoid', name='Decoder')(encoded)

    output_img = Reshape(input_shape, name='Output')(decoded)

    autoencoder = Model(input_img, output_img)

    return autoencoder


input_shape = (64, 64, 3)
latent_dim = 64
autoencoder = build_autoencoder(input_shape, latent_dim)


# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the autoencoder model
autoencoder.compile(optimizer=optimizer,
                    loss='binary_crossentropy')

# Train the autoencoder model
autoencoder.fit(train_ds,
                epochs=5,
                validation_data=validation_ds,
                callbacks=[model_checkpoint_callback])




# Create a numpy array for my validation images
validation_images = []

for images, _ in validation_ds:
    validation_images.append(images.numpy())

validation_np = np.concatenate(validation_images, axis=0)


# Generate a random index for selecting an image from the validation set
random_img = np.random.randint(0, len(validation_np))

# Count the number of files in the temporary directory
len_check = len(os.listdir('./tmp/'))


fig, ax = plt.subplots(1, len_check + 1, figsize=(15, 15))

# Display the original image
ax[0].imshow(validation_np[random_img])
ax[0].axis('off')
ax[0].set_title('Original')

# Load weights and display reconstructed images
for i, weights in enumerate(sorted(os.listdir('./tmp/'), reverse=True), 1):
    epoch_model = tf.keras.models.clone_model(autoencoder)
    epoch_model.load_weights('./tmp/' + weights)

    # Ensure the image is correctly shaped for prediction
    img_to_predict = validation_np[random_img].reshape(1, *validation_np[random_img].shape)
    epoch_pred = epoch_model.predict(img_to_predict, verbose=0)

    ax[i].imshow(np.clip(epoch_pred[0], 0, 1))
    ax[i].axis('off')
    ax[i].set_title('Epoch: {}'.format(i))




# Define input shape and latent dimensions for the autoencoder
input_shape       = (64, 64, 3)
latent_dimensions = [8, 64, 128]

# Lists to store models, training history, and training time
model_list        = []
train_history     = []
train_time        = []

# Loop through each latent dimension and train the autoencoder
for latent_dim in latent_dimensions:
    print(f"Training model with latent dimension: {latent_dim}")

    # Build and compile the autoencoder model
    autoencoder = build_autoencoder(input_shape, latent_dim)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

    # Train the model and record the training time
    start_time = time.time()
    history = autoencoder.fit(train_ds, epochs=5, validation_data=validation_ds)
    end_time = time.time()

    # Store the trained model, training history, and training time
    model_list.append(autoencoder)
    train_history.append(history.history)
    train_time.append(end_time - start_time)





# Generate a random index to select an image from the validation dataset
random_image = np.random.randint(0, len(validation_np))
PSNR_list = []

# Set up the figure for displaying images
fig, axs = plt.subplots(1, len(latent_dimensions) + 1, figsize=(15, 5))
axs[0].imshow(validation_np[random_image])
axs[0].set_title('Original')
axs[0].axis('off')

# Loop through each latent dimension and reconstruct the image
for i, latent_dim in enumerate(latent_dimensions):
    model = model_list[i]

    # Reconstruct the image
    x_pred = model.predict(validation_np[random_image].reshape(1, *validation_np[random_image].shape))

    # Display the reconstructed image
    axs[i + 1].imshow(x_pred[0])
    axs[i + 1].set_title(f'Latent dim: {latent_dim}')
    axs[i].axis('off')
    axs[i + 1].axis('off')

    # Calculate and append the PSNR value
    PSNR_list.append(tf.image.psnr(x_pred[0], validation_np[random_image], max_val=1.0).numpy())

# Show the plot
plt.show()






# Define the X-axis positions for plotting
X_axis = np.arange(len(latent_dimensions))

# Assign PSNR values collected during reconstruction
psnr_values = PSNR_list

# Extract training losses from training history
train_losses = [train_history[i]['loss'][0] for i in range(len(train_history))]

# Store training times for each model
train_times = train_time



# Training loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.bar(X_axis, train_losses, align='center', alpha=0.7)
plt.xticks(X_axis, latent_dimensions)
plt.xlabel('Latent Dimension')
plt.ylabel('Training Loss')
plt.title('Training Loss for Different Latent Dimensions')


# PSNR values
plt.subplot(1, 3, 2)
plt.bar(X_axis, psnr_values, align='center', alpha=0.7)
plt.xticks(X_axis, latent_dimensions)
plt.xlabel('Latent Dimension')
plt.ylabel('PSNR Value')
plt.title('PSNR Values for Different Latent Dimensions')


# Training time
plt.subplot(1, 3, 3)
plt.bar(X_axis, train_times, align='center', alpha=0.7)
plt.xticks(X_axis, latent_dimensions)
plt.xlabel('Latent Dimension')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time for Different Latent Dimensions')

plt.tight_layout()
plt.show()




import tensorflow as tf
from tensorflow.keras import layers, models


# Encoder function
def encoder(input_shape, hidden_dims, latent_dim, model_name='encoder'):

    # INITIATE INPUT STRUCTURE
    inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
    x = inputs


    # CONVOLUTIONAL LAYERS
    for h_dim in hidden_dims:
        x = layers.Conv2D(h_dim, kernel_size=3, strides=2, padding="same", activation="relu", name=f'Conv_dim{h_dim}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)


    # FLATTEN
    x = layers.Flatten(name='Flatten_1d_vector')(x)


    # VARIATIONAL LAYERS
    z_mean = layers.Dense(latent_dim, name='Z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='Z_log_var')(x)

    return models.Model(inputs, [z_mean, z_log_var], name=model_name)




# Decoder function
def decoder(input_shape, hidden_dims, latent_dim, model_name='decoder'):

    # INPUT FROM LATENT SPACE
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='decoder_input')


    # DETERMINE SIZE NEEDED FOR FIRST DENSE LAYER
    initial_dim_x = input_shape[0] // (2 ** len(hidden_dims))
    initial_dim_y = input_shape[1] // (2 ** len(hidden_dims))
    x = layers.Dense(initial_dim_x * initial_dim_y * hidden_dims[-1], activation='relu', name='Dense_for_reshape')(latent_inputs)
    x = layers.Reshape((initial_dim_x, initial_dim_y, hidden_dims[-1]), name='Reshape_to_3d')(x)


    # DE-CONVOLUTIONAL LAYERS
    for h_dim in hidden_dims[::-1]:

        x = layers.Conv2DTranspose(h_dim, kernel_size=3, strides=2, padding="same", activation="relu", name=f'ConvTransose_dim{h_dim}')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)


    # OUTPUT: SAME AS ORIGINAL INPUT DIMENSIONS
    outputs = layers.Conv2DTranspose(input_shape[2], kernel_size=3, activation="sigmoid", padding="same", name='Decoder_output')(x)

    return models.Model(latent_inputs, outputs, name=model_name)



input_shape = (64, 64, 3)
hidden_dims = [512, 256, 128]
latent_dim = 64





# Definitation of a reprarametrization layer
class reparameterization_layer(tf.keras.layers.Layer):
    """
    Input: Mean Vector, Variance Vector (from the variational layers of the Encoder - see illustration above)
    Output: Z = μ + σ⊙ε     where ε ~ 𝒩(0,1) and σ = exp(z_var/2)
    """
    def call(self,z_inputs):
        z_mean, z_std = z_inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_std) * epsilon, z_mean, z_std




import tensorflow as tf
from tensorflow.keras import layers, models





# Encoder function
def encoder(input_shape, hidden_dims, latent_dim, model_name='encoder'):

    inputs = tf.keras.Input(shape=input_shape, name='encoder_input')
    x = inputs

    for h_dim in hidden_dims:
        x = layers.Conv2D(h_dim, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = reparameterization_layer()([z_mean, z_log_var])

    model = models.Model(inputs, [z_mean, z_log_var, z], name=model_name)

    return model




# Decoder function
def decoder(input_shape, hidden_dims, latent_dim, model_name='decoder'):

    latent_inputs = tf.keras.Input(shape=(latent_dim,), name='decoder_input')

    x = layers.Dense(hidden_dims[-1] * (input_shape[0] // 2**len(hidden_dims)) * (input_shape[1] // 2**len(hidden_dims)), activation='relu')(latent_inputs)
    x = layers.Reshape((input_shape[0] // 2**len(hidden_dims), input_shape[1] // 2**len(hidden_dims), hidden_dims[-1]))(x)

    for h_dim in hidden_dims[::-1]:
        x = layers.Conv2DTranspose(h_dim, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    outputs = layers.Conv2DTranspose(input_shape[2], kernel_size=3, activation="sigmoid", padding="same", name='decoder_output')(x)

    return models.Model(latent_inputs, outputs, name=model_name)


input_shape = (64, 64, 3)
hidden_dims = [512, 256, 128]
latent_dim = 256


enc = encoder(input_shape, hidden_dims, latent_dim)
dec = decoder(input_shape, hidden_dims, latent_dim)






class VAE(tf.keras.Model):
    """
    Input: encoder, decoder
    1. Track reconstruction loss, kl divergence, and total loss
    2. Update gradients
    Output: loss
    """
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rec_loss_tracker = keras.metrics.Mean(name="rec_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.tot_loss_tracker = keras.metrics.Mean(name="total_loss")

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z[0])
        return reconstructed


    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as grad:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z[0])
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = grad.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.tot_loss_tracker.update_state(total_loss)
        self.rec_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)


        return {
            "loss": self.tot_loss_tracker.result(),
            "reconstruction_loss": self.rec_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    


# Build, compile and train a model
vae = VAE(encoder=enc, decoder=dec)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

history = vae.fit(train_ds,
                  epochs=5,
                  batch_size=64,
                  validation_data=validation_ds)


# Reconstruction of sample by sampling the latent space
data_subset = np.stack(list(validation_ds.take(1))).squeeze(0)[0]
reconstruced_image = vae.decoder(vae.encoder(data_subset)[0])


# Sample from a standard normal distribution
z_sample = np.random.normal(size=(1, latent_dim))

# Generate the image using the decoder
generated_image = vae.decoder.predict(z_sample)

# Assuming the generated image is in the right format and scale
plt.imshow(generated_image[0, :, :, :])
plt.title("Generated Image")
plt.axis('off')
plt.show()




BATCH_SIZE = 32
img_w, img_h = 64,64


# New, extended preprocess function
def preprocess(example, att = 'Smiling', size=(img_w, img_h), mode='attributes'):
    image = example['image']
    if mode == 'train':
        image_resized = tf.image.resize(image, size=size)
        return image_resized / 255.0, image_resized / 255.0

    elif mode == 'attributes':
      img = example['image']
      attri = example['attributes']
      if attri[att]:
        label = 1
      else:
        label = 0
      img = tf.image.resize(img, size=size) / 255.0
      return img, label


# Map the preprocessing function to the CelebA training dataset
train_att = celeba_train.map(lambda x: preprocess(x))

# Batch the dataset
train_att = train_att.batch(BATCH_SIZE)

# Take the first 20 batches from the dataset
train_att = train_att.take(20)





# Sorting relevant images
img_with = []
img_without = []
for images, labels in train_att:
    for i in range(BATCH_SIZE):
      if tf.reduce_any(tf.equal(labels[i], 1)):
         img_with.append(images[i])
      else:
         img_without.append(images[i])
img_with, img_without = np.array(img_with), np.array(img_without)


# Designing the latent space with desired attributes
_, z_mean_with, _ = vae.encoder(img_with)
attribute_vector = tf.reduce_mean(z_mean_with, axis=0, keepdims=True)

# Ensure attribute_vector has the shape (1, 256)
print("Shape of attribute_vector:", attribute_vector.shape)




# Parameters
n_examples = 5
beta = 20

# Obtain the original latent encoding from the VAE encoder
og_enc = vae.encoder(img_without)[0]

# Adjust the latent encoding by adding a scaled attribute vector
att_enc = og_enc + beta * attribute_vector

# Set up the figure and axes for displaying images
f, axs = plt.subplots(3, n_examples, figsize=(16, 6))



# Set titles
axs[0, n_examples // 2].set_title("Original images")
axs[1, n_examples // 2].set_title("Reconstructed images")
axs[2, n_examples // 2].set_title("Images with added attribute")

for j in range(n_examples):

    # Original image without attribute
    axs[0, j].imshow(img_without[j])

    # Reconstruction orignal latent space
    axs[1, j].imshow(vae.decoder(og_enc)[j])

    # Reconstructed conditioned latent space
    axs[2, j].imshow(vae.decoder(att_enc)[j])

    for ax in axs[:, j]:
       ax.axis('off')

plt.tight_layout()





# Uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=5):
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


# Interpolation function
def create_interpolated_samples():
    num_steps = 8
    step_size = 1.0 / num_steps

    # Initialize sample_a and sample_b
    sample_a, sample_b = None, None

    # Retrieve data that correspond to the class indices
    for img_batch, label_batch in train_att:
        img_batch_np = img_batch.numpy()
        label_batch_np = label_batch.numpy()
        for i, label in enumerate(label_batch_np):
            if sample_a is None and label == 1:
                sample_a = img_batch_np[i]
            if sample_b is None and label == 0:
                sample_b = img_batch_np[i]
            if sample_a is not None and sample_b is not None:
                break

        if sample_a is not None and sample_b is not None:
            break

    # Check if both samples were found
    if sample_a is None or sample_b is None:
        raise ValueError("Could not find required samples")

    # Process samples
    sample_a = tf.reshape(sample_a, [1, img_h, img_w, 3])
    sample_b = tf.reshape(sample_b, [1, img_h, img_w, 3])

    # Encode both images to get latent representations
    z_a, z_b = vae.encoder(sample_a)[0].numpy(), vae.encoder(sample_b)[0].numpy()

    # Interpolate in latent space
    interpolated_zs = interpolate_points(z_a, z_b, n_steps=num_steps)

    # Decode interpolated points
    out_imgs = [vae.decoder.predict(z.reshape(1, -1)) for z in interpolated_zs]

    return out_imgs