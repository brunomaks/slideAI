import tensorflow as tf
import platform
from tensorflow.keras import layers, models

IMG_SIZE = (50, 50)
BATCH_SIZE = 32
EPOCHS = 10

print(f"Platform: {platform.system()}")
print(f"TensorFlow version: {tf.__version__}")
print("GPU available: ", tf.config.list_physical_devices('GPU'))

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    'images/train',
    shuffle=False,
    color_mode='grayscale',
    validation_split=0.1,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
	'images/train',
    shuffle=False,
    color_mode='grayscale',
    validation_split=0.1,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

num_classes = len(train_data.class_names)  # Access here before prefetch

# for performance optimization: preload the next batch while the current batch is being processed
train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

#    layers.Conv2D(128, (3, 3), activation='relu'),
#    layers.BatchNormalization(),
#    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

test_loss, test_acc = model.evaluate(val_data, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('gesture_cnn_model.h5')