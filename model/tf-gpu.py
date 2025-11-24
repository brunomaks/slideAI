import tensorflow as tf
import platform
from tensorflow import keras

print(f"Platform: {platform.system()}")
print(f"TensorFlow version: {tf.__version__}")
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Simple model to test stability
model = keras.Sequential([
    keras.layers.Dense(1024, activation='relu', input_shape=(784,)),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Generate dummy data
import numpy as np
x_train = np.random.random((60000, 784))
y_train = np.random.randint(10, size=(60000,))

# Train for many epochs to test stability
history = model.fit(x_train, y_train, 
                    epochs=100,
                    batch_size=128,
                    verbose=1)
