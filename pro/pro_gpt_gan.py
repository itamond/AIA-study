import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image



# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (_, _) = mnist.load_data()

# Load handwriting data
handwriting_data = np.load('handwriting_data.npy')
x_train = np.concatenate([x_train, handwriting_data])
y_train = np.concatenate([y_train, np.zeros(handwriting_data.shape[0]) + 10])

# Normalize pixel values
x_train = x_train.astype('float32') / 255

# Define generator model
generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Reshape((7, 7, 256)),
    tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
])

# Define discriminator model
discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1)
])

# Define loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define training loop
BATCH_SIZE = 256
EPOCHS = 50
SAVE_INTERVAL = 5
IMAGE_SIZE = 28
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    # Save generated images every SAVE_INTERVAL epochs
    if (EPOCHS + 1) % SAVE_INTERVAL == 0:
        save_images(generated_images, EPOCHS + 1)

def save_images(images, epoch):
    # Rescale images to 0-255 range and convert to uint8
    images = (images * 127.5 + 127.5).numpy().astype(np.uint8)
    
    # Create grid of images
    grid_size = int(np.sqrt(images.shape[0]))
    img_grid = np.zeros((grid_size * IMAGE_SIZE, grid_size * IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Fill grid with images
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        img_grid[row*IMAGE_SIZE:(row+1)*IMAGE_SIZE, col*IMAGE_SIZE:(col+1)*IMAGE_SIZE] = img
    
    # Save grid image
    img_path = os.path.join(SAVE_DIR, f"generated_images_{epoch:04d}.png")
    Image.fromarray(img_grid).save(img_path)