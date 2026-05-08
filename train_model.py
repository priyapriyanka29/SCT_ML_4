import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Dataset folder
data_dir = "leapGestRecog"

X = []
y = []

# Gesture labels
gesture_map = {
    "01_palm": 0,
    "02_l": 1,
    "03_fist": 2,
    "04_fist_moved": 3,
    "05_thumb": 4,
    "06_index": 5,
    "07_ok": 6,
    "08_palm_moved": 7,
    "09_c": 8,
    "10_down": 9
}

# Loop through person folders
for person in os.listdir(data_dir):

    # Skip extra folder
    if person == "leapGestRecog":
        continue

    person_path = os.path.join(data_dir, person)

    if not os.path.isdir(person_path):
        continue

    # Loop through gesture folders
    for gesture in os.listdir(person_path):

        gesture_path = os.path.join(person_path, gesture)

        if not os.path.isdir(gesture_path):
            continue

        # Get label
        label = gesture_map[gesture]

        # Read images
        for img_name in os.listdir(gesture_path):

            img_path = os.path.join(gesture_path, img_name)

            img = cv2.imread(img_path)

            # Skip invalid images
            if img is None:
                continue

            # Resize image
            img = cv2.resize(img, (64, 64))

            X.append(img)
            y.append(label)

# Convert to numpy arrays
X = np.array(X) / 255.0
y = to_categorical(y, num_classes=10)

print("Dataset Loaded Successfully")
print("Total Images:", len(X))

# CNN Model
model = Sequential([

    Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(64, 64, 3)
    ),

    MaxPooling2D(2, 2),

    Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X, y, epochs=5)

# Save model
model.save("gesture_model.h5")

print("Model Trained Successfully")