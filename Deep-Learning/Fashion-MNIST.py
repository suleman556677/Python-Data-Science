# Fashion MNIST Classification using TensorFlow / Keras

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 2. Normalize images (0-255 -> 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 -> 784
    keras.layers.Dense(128, activation='relu'),  # Hidden layer
    keras.layers.Dense(10, activation='softmax') # Output layer (10 classes)
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
model.fit(train_images, train_labels, epochs=10)

# 6. Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"\n Test Accuracy: {test_acc*100:.2f}%")

# 7. Make predictions
predictions = model.predict(test_images)

# Function to plot image + prediction
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% (True: {})".format(
        class_names[predicted_label],
        100*np.max(predictions_array),
        class_names[true_label]),
        color=color
    )

# 8. Visualize first 15 test images with predictions
num_rows = 3
num_cols = 5
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_rows*num_cols):
    plt.subplot(num_rows, num_cols, i+1)
    plot_image(i, predictions[i], test_labels, test_images)
plt.tight_layout()
plt.show()

# Example single prediction
index = 0
print("\n Example Prediction:")
print("Predicted:", class_names[np.argmax(predictions[index])])
print("Actual:", class_names[test_labels[index]])
