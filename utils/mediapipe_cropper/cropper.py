import numpy as np
import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
import tqdm

MARGIN = 20  # pixels

def main():
    args = parse_args()
    process_images(args.input, args.output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest = "input", default = "input/", help="Path to input folder")
    parser.add_argument("-o", "--output", dest = "output", default = "output/", help="Path to output folder")
    return parser.parse_args()

def process_images(input_path, output_path):
    # STEP 0: Resolve detector path.
    detector_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')

    # STEP 1: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path=detector_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # STEP 2: Iterate through images in the input folder.
    labels = [l for l in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, l))]

    for label in labels:
        label_input_path = os.path.join(input_path, label)
        label_output_path = os.path.join(output_path, label)

        os.makedirs(label_output_path, exist_ok=True) # make sure dir exists

        tqdm.tqdm.write(f"Processing directory: {label_input_path}")

        files = [f for f in os.listdir(label_input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for file in tqdm.tqdm(files, unit="img"):
    # STEP 3: Load the input image.
            image = mp.Image.create_from_file(os.path.join(label_input_path, file))

    # STEP 4: Detect hand landmarks from the input image.
            detection_result = detector.detect(image)

    # STEP 5: Save cropped image
            # slice to get rid of alpha channel if present
            save_detected_hand(image.numpy_view()[:,:,:3], detection_result, os.path.join(label_output_path, file))

def save_detected_hand(rgb_image, detection_result, save_file_path):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    if not hand_landmarks_list:
        fname = os.path.basename(save_file_path)
        print(f"No hands detected in the image: {fname}. Skipping.")
        return

    if len(hand_landmarks_list) > 1:
        fname = os.path.basename(save_file_path)
        print(f"Multiple hands detected in the image: {fname}. Skipping.")
        return

    # Extract the first and only detected hand
    hand_landmarks = hand_landmarks_list[0]

    # Get the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    box_x_min = int(min(x_coordinates) * width) - MARGIN
    box_y_min = int(min(y_coordinates) * height) - MARGIN
    box_x_max = int(max(x_coordinates) * width) + MARGIN
    box_y_max = int(max(y_coordinates) * height) + MARGIN

    # Ensure the bounding box is within the image dimensions
    box_x_min = max(box_x_min, 0)
    box_y_min = max(box_y_min, 0)
    box_x_max = min(box_x_max, width)
    box_y_max = min(box_y_max, height)

    if box_x_min >= box_x_max or box_y_min >= box_y_max:
      print(f"Invalid bounding box dimensions for {os.path.basename(save_file_path)}. Skipping this hand.")
      return

    # Crop the image by the bounding box
    cropped_img = annotated_image[box_y_min:box_y_max, box_x_min:box_x_max]

    if cropped_img.size == 0:
        print("Empty crop, skipping...")
        return

    # Save cropped image
    to_save_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_file_path, to_save_rgb)

if __name__ == "__main__":
    main()