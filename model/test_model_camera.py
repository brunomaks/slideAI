import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time

# loaded_model = tf.keras.models.load_model('gesture_cnn_model.h5')
# loaded_model.summary()

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


loaded_model = tf.keras.models.load_model('gesture_cnn_model.h5')
loaded_model.summary()

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("Could not open the camera")
    exit()

fps_counter = 0
fps_time = time.time()
frame_count = 0
skip_frames = 3  # Process every 3rd frame for better performance

lower = np.array([0, 0, 0])      # lower HSV bound
upper = np.array([180, 255, 55])   # upper HSV bound
    

try:
    while True:
        ret, frame = cap.read()
        frame_flipped = cv2.flip(frame, 1)
        
        if not ret:
            print("Error: Can't receive frame")
            break

        frame_count += 1
        if frame_count % (skip_frames) != 0:
            continue

        # preprocessing
        frame_resized = cv2.resize(frame_flipped, (50, 50), interpolation=cv2.INTER_AREA)
        frame_hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        frame_hsv = cv2.inRange(frame_hsv, lower, upper)
        frame_array = frame_hsv.reshape(1, 50, 50, 1)

        start_time = time.time()
        prediction = loaded_model.predict(frame_array)
        inference_time = time.time() - start_time

        predicted_class = np.argmax(prediction)

        cv2.putText(frame_flipped, str(predicted_class), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame_flipped, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        cv2.imshow('Processed frame', frame_hsv)
        
        cv2.imshow('Webcam Feed', frame_flipped)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Stopping")
finally:
    cap.release()
    cv2.destroyAllWindows()



img = Image.open('real-images/mask_five.png')

img_array = np.array(img)

img_array = np.expand_dims(img_array, axis=-1) # (50, 50, 1)

img_array = img_array.reshape(1, 50, 50, 1)

print(img_array.shape)

prediction = loaded_model.predict(img_array)

for i in range(0, prediction.shape[1]):
    print(f"Class: {i}: {prediction[0][i]:.10f}")

predicted_class = np.argmax(prediction)
print(f"Predicted class {predicted_class}")
