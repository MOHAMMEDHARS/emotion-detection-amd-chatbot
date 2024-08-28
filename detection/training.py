import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os

# Function to load and preprocess images
def load_images_and_labels(data_dir):
    images = []
    labels = []
    emotions = ["angry", "disgusted", "fearful", "happy", "sad", "surprised", "neutral"]
    for label, emotion in enumerate(emotions):
        emotion_dir = os.path.join(data_dir, emotion)
        for filename in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (48, 48))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
data_dir = r"C:\Users\mhars\Downloads\archive (7)\train"
images, labels = load_images_and_labels(data_dir)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes for 7 emotions
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('emotion_detection_model.h5')
