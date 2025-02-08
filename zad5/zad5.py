import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout, Lambda
from tensorflow.keras.regularizers import l1
from tensorflow.keras import backend as K, layers
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras

(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

def show_decoded_vs_og(decoded_imgs, n=10, og_images=x_test):
    plt.figure(figsize=(20, 4))
    
    for i in range(n):
        # Display original image
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(og_images[i].reshape(28, 28), cmap='gray')
        ax.set_title("Original")
        ax.axis('off')
    
        # Display reconstructed image
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
        ax.set_title("Reconstructed")
        ax.axis('off')
    
    plt.suptitle("Original vs Reconstructed Images")
    plt.tight_layout()
    plt.show()

input_img = Input(shape=(28, 28, 1))
flat_img = Flatten()(input_img)

# Simple autoencoder
encoded = Dense(16, activation='relu')(flat_img)
decoded = Dense(784, activation='sigmoid')(encoded)
decoded_img = Reshape((28, 28, 1))(decoded)

autoencoder = Model(input_img, decoded_img)
autoencoder.compile(optimizer=Adam(), loss='mse')

history_simple = autoencoder.fit(
    x_train, x_train, 
    epochs=10, 
    batch_size=256, 
    validation_split=0.2,
)
decoded_imgs = autoencoder.predict(x_test)
show_decoded_vs_og(decoded_imgs)

# Sparse autoencoder with L1 regularization
encoded_sparse = Dense(32, activation='relu', activity_regularizer=l1(1e-6))(flat_img)
decoded_sparse = Dense(784, activation='sigmoid')(encoded_sparse)
decoded_img_sparse = Reshape((28, 28, 1))(decoded_sparse)

sparse_autoencoder = Model(input_img, decoded_img_sparse)
sparse_autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

history_sparse = sparse_autoencoder.fit(
    x_train, x_train, 
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
)
decoded_imgs = sparse_autoencoder.predict(x_test)
show_decoded_vs_og(decoded_imgs)

# Deep fully-connected autoencoder
encoded_deep = Dense(256, activation='relu')(flat_img)
encoded_deep = Dense(16, activation='relu')(encoded_deep)
decoded_deep = Dense(256, activation='relu')(encoded_deep)
decoded_deep = Dense(784, activation='sigmoid')(decoded_deep)
decoded_img_deep = Reshape((28, 28, 1))(decoded_deep)

deep_autoencoder = Model(input_img, decoded_img_deep)
deep_autoencoder.compile(optimizer=Adam(), loss='mse')

history_deep = deep_autoencoder.fit(
    x_train, x_train, 
    epochs=10, 
    batch_size=256, 
    validation_split=0.2,
)
decoded_imgs = deep_autoencoder.predict(x_test)
show_decoded_vs_og(decoded_imgs)

# Convolutional autoencoder
class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),  # (28, 28, 1) -> (14, 14, 16)
            layers.BatchNormalization(),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),   # (14, 14, 16) -> (7, 7, 8)
            layers.BatchNormalization(),
        ])
    
        self.decoder = keras.Sequential([
            layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),  # (7, 7, 8) -> (14, 14, 8)
            layers.BatchNormalization(),
            layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2), # (14, 14, 8) -> (28, 28, 16)
            layers.BatchNormalization(),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')                    # (28, 28, 16) -> (28, 28, 1)
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

conv_autoencoder = Denoise()
conv_autoencoder.compile(optimizer='adam', loss='mse')

history_conv = conv_autoencoder.fit(x_train, x_train,
    epochs=10,
    shuffle=True,
    validation_data=(x_test, x_test),
    batch_size=256
)
decoded_imgs = conv_autoencoder.predict(x_test)
show_decoded_vs_og(decoded_imgs)

# Denoising problem
noise_factor = 0.3
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)
show_decoded_vs_og(x_test_noisy)

from tensorflow.keras.optimizers import Adam
conv_autoencoder = Denoise()
conv_autoencoder.compile(optimizer='adam', loss='mse')

history_denoising = conv_autoencoder.fit(
    x_train_noisy, x_train, 
    epochs=10, 
    batch_size=256, 
    validation_split=0.2,
)
decoded_imgs = conv_autoencoder.predict(x_test_noisy)
show_decoded_vs_og(decoded_imgs, og_images=x_test_noisy)

# VAE - does not work
# original_dim = 28 * 28
# intermediate_dim = 64
# latent_dim = 2

# inputs = keras.Input(shape=(original_dim,))
# h = layers.Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = layers.Dense(latent_dim)(h)
# z_log_sigma = layers.Dense(latent_dim)(h)

# from keras import backend as K

# def sampling(args):
    # z_mean, z_log_sigma = args
    # epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              # mean=0., stddev=0.1)
    # return z_mean + K.exp(z_log_sigma) * epsilon

# z = layers.Lambda(sampling, output_shape=(2,))([z_mean, z_log_sigma])

# # Create encoder
# encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# # Create decoder
# latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
# x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = layers.Dense(original_dim, activation='sigmoid')(x)
# decoder = keras.Model(latent_inputs, outputs, name='decoder')

# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = keras.Model(inputs, outputs, name='vae_mlp')

# reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
# reconstruction_loss *= original_dim
# # kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
# # kl_loss = K.sum(kl_loss, axis=-1)
# # kl_loss *= -0.5
# # vae_loss = K.mean(reconstruction_loss + kl_loss)
# # vae.add_loss(vae_loss)
# vae.add_loss(reconstruction_loss)
# vae.compile(optimizer='adam')

# history_vae = vae.fit(x_train, x_train,
        # epochs=10,
        # batch_size=32,
        # validation_data=(x_test, x_test))
# decoded_imgs = vae.predict(x_test)
# show_decoded_vs_og(decoded_imgs)
