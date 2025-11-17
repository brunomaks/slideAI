import tensorflow as tf
from PIL import Image
import numpy as np

loaded_model = tf.keras.models.load_model('gesture_cnn_model.h5')
loaded_model.summary()


# test_data = tf.keras.preprocessing.image_dataset_from_directory(
#     'images/test',
#     color_mode='grayscale',
#     image_size=(50, 50),
#     batch_size=32
# )

# predictions = loaded_model.predict(test_data)
# for i, prediction in enumerate(predictions):
#     predicted_class = tf.argmax(prediction).numpy()
#     print(f"Image {i}: Predicted class {predicted_class}")


img = Image.open('pavlo/ok.jpg').convert('L')

img_array = np.array(img)

img_array = np.expand_dims(img_array, axis=-1) # (50, 50, 1)

img_array = img_array.reshape(1, 50, 50, 1)

print(img_array.shape)

prediction = loaded_model.predict(img_array)

for i in range(0, prediction.shape[1]):
    print(f"Class: {i}: {prediction[0][i]:.10f}")

predicted_class = np.argmax(prediction)
print(f"Predicted class {predicted_class}")


