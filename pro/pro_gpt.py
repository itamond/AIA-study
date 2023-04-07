import cv2
import numpy as np
import os
from tensorflow import keras

mnist_dir = 'C:/project_mnist/numbers/mnist_png/Hnd/'
handwriting_dir = 'C:/project_mnist/numbers/chars74k_png/Hnd/'

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

input_shape = (28, 28, 1)

model = tf.keras.models.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


def predict_numbers(img):
    # Convert image to grayscale and threshold it
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the image
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # Predict the numbers in each contour
    predicted_numbers = []
    for contour in contours:
        # Extract bounding box for contour
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:
            continue

        # Extract ROI from image
        roi = img_gray[y:y+h, x:x+w]

        # Resize ROI to 28x28 pixels
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize pixel values to range [0, 1]
        roi_normalized = roi_resized.astype('float32') / 255.0

        # Reshape image to match input shape of MNIST model
        roi_reshaped = np.reshape(roi_normalized, (1, 28, 28, 1))

        # Predict number using MNIST model
        number = np.argmax(mnist_model.predict(roi_reshaped))

        predicted_numbers.append(number)

    return predicted_numbers

# Define function to perform arithmetic operations
def calculate(a, b, operator):
    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        return a / b

# Define paths to input images
img_a_path = 'input_images/image_a.jpg'
img_b_path = 'input_images/image_b.jpg'

# Load input images
img_a = cv2.imread(img_a_path)
img_b = cv2.imread(img_b_path)

# Predict numbers in input images
numbers_a = predict_numbers(img_a)
numbers_b = predict_numbers(img_b)

# Convert predicted numbers to integers
a = int(''.join(map(str, numbers_a)))
b = int(''.join(map(str, numbers_b)))

# Calculate results of arithmetic operations
result_add = calculate(a, b, '+')
result_subtract = calculate(a, b, '-')
result_multiply = calculate(a, b, '*')
result_divide = calculate(a, b, '/')

# Define function to save image of formula and result
def save_image_formula_and_result(a, b, operator, result):
    # Define font and text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    # Create blank image
    img = np.zeros((100, 600, 3), np.uint8)

    # Write formula and result on image
    formula
    



이 프로젝트의 내용을 종합하여 코드를 만들어줘.
코드는 OpenCV를 활용하고 다음 내용이 적용되어야 해.
1. datasets = mnist 이미지 60000장, 손글씨 이미지 10000장
2. a와 b라는 숫자 사진 두장을 입력하면 각 사진의 숫자를 예측하여 출력
3. 사칙연산이 가능한 계산기 함수 생성
4. a와 b로 출력된 숫자를 3번의 계산기 함수에 입력
5. 계산식과 결과를 이미지로 저장    
Mnist = C:\project_mnist\numbers\mnist_png\Hnd
손글씨 = C:\project_mnist\numbers\chars74k_png\Hnd