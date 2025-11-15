import tensorflow as tf

loaded_model = tf.keras.models.load_model('gesture_cnn_model.h5')
loaded_model.summary()


test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'images/test',
    color_mode='grayscale',
    image_size=(50, 50),
    batch_size=32
)

predictions = loaded_model.predict(test_data)
for i, prediction in enumerate(predictions):
    predicted_class = tf.argmax(prediction).numpy()
    print(f"Image {i}: Predicted class {predicted_class}")

    