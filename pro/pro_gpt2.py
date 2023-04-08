import tensorflow as tf
import numpy as np
import cv2

# Load Mnist images
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load handwriting images
def load_handwriting_images():
    images = []
    for i in range(1, 11001):
        img = cv2.imread(f'path/to/handwriting/image_{i}.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        images.append(img)
    return np.array(images)

handwriting_images = load_handwriting_images()

# Normalize pixel values and reshape images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (-1, 28, 28, 1))
x_test = np.reshape(x_test, (-1, 28, 28, 1))
handwriting_images = handwriting_images.astype('float32') / 255
handwriting_images = np.reshape(handwriting_images, (-1, 28, 28, 1))

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Predict numbers in handwriting images
predictions = model.predict(handwriting_images)
predicted_numbers = np.argmax(predictions, axis=1)

# Convert predicted numbers to characters and save images
for i, number in enumerate(predicted_numbers):
    char = chr(number + 48) # convert to ASCII code
    img = handwriting_images[i] * 255 # convert back to 0-255 scale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # convert to color image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, char, (5, 25), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(f'path/to/output/image_{i}.jpg', img)