from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

# This loads the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# This will normalise the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    #This creates a basic neural network
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#This will train the model
model.fit(x_train, y_train, epochs=5)

#This will evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)



#This will get a prediction for the test set
predictions = model.predict(x_test)

#This function plots the images and predictions
def plot_image(i):
    prediction = np.argmax(predictions[i])
    actual = y_test[i]
    color = 'green' if prediction == actual else 'red'

    plt.figure(figsize=(4, 4))
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {prediction} | Actual: {actual}", color=color)
    plt.axis('off')
    plt.show()


#This plots a dew test images with predictions
for i in range(5):
    plot_image(i)


