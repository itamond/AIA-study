import os
from PIL import Image
import numpy as np



data_dir = "C:/project_mnist/numbers/mnist_png/Hnd"
image_size = (28, 28) # Resize images to 28x28 pixels

images = []
labels = []


for label in range(10):
    folder_name = f"sample{label}"
    folder_path = os.path.join(data_dir, folder_name)

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".png"):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).resize(image_size).convert("L") # Convert to grayscale
            images.append(image)
            labels.append(label)
            
            
          

# Normalize pixel values to [0, 1]
images = [np.array(image) / 255.0 for image in images]

# Convert image data to numpy arrays
images = np.array(images)
labels = np.array(labels)



from sklearn.model_selection import train_test_split

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)



import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), 
                           activation='relu', 
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, 
                    train_labels, 
                    epochs=3,
                    validation_data=(val_images, val_labels))

test_loss, test_acc = model.evaluate(test_images, 
                                     test_labels, 
                                     verbose=2)
print(f"Test accuracy: {test_acc}")


def calculate(expression):
    # Extract the numbers and operator from the expression
    a, op, b = expression.split()

    # Convert the numbers to image arrays and normalize pixel values
    a_img = np.array(Image.fromarray(np.uint8(a)).resize(image_size)) / 255.0
    b_img = np.array(Image.fromarray(np.uint8(b)).resize(image_size)) / 255.0

    # Reshape the images to match the model's input shape
    a_img = np.reshape(a_img, (1, 28, 28, 1))
    b_img = np.reshape(b_img, (1, 28, 28, 1))

    # Pass the images through the model to get the predicted output
    a_pred = np.argmax(model.predict(a_img))
    b_pred = np.argmax(model.predict(b_img))

    # Perform the operation and return the result
    if op == "+":
        return f"{a} + {b} = {a_pred + b_pred}"
    elif op == "-":
        return f"{a} - {b} = {a_pred - b_pred}"
    elif op == "*":
        return f"{a} * {b} = {a_pred * b_pred}"
    elif op == "/":
        return f"{a} / {b} = {a_pred / b_pred}"
    else:
        return "Invalid operator"
    
    
    
from PIL import ImageDraw, ImageFont

def save_image(expression, result, filename):
    # Create a new image with the expression and result overlaid
    img = Image.new('RGB', (500, 50), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 20)
    draw.text((10, 10), f"{expression} = {result}", font=font, fill=(0, 0, 0))

    # Save the image to disk
    img.save(filename)
    
expression = "3 + 4"
result = calculate(expression)
filename = "C:/Users/bitcamp/result.png"
save_image(expression, result, filename)