import numpy as np
import cv2
import argparse
import os 
import tqdm

def main():
    args = parse_args()
    process_images(args.input, args.size, args.output)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", dest = "size", default = 94, help="Output image size (width and height)", type=int)
    parser.add_argument("-i", "--input", dest = "input", default = "./input/", help="Path to input folder")
    parser.add_argument("-o", "--output", dest = "output", default = "./output/", help="Path to output folder")
    return parser.parse_args()

def process_images(input_path, size, output_path):
    # STEP 1: Iterate through images in the input folder.
    labels = [l for l in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, l))]

    for label in labels:
        label_input_path = os.path.join(input_path, label)
        label_output_path = os.path.join(output_path, label)

        os.makedirs(label_output_path, exist_ok=True) # make sure dir exists

        tqdm.tqdm.write(f"Processing directory: {label_input_path}")

        files = [f for f in os.listdir(label_input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for file in tqdm.tqdm(files, unit="img"):

    # STEP 2: Load the input image. 
            input_file_path = os.path.join(label_input_path, file)
            image = cv2.imread(input_file_path)

    # STEP 3: Resize and save the image
            output_file_path = os.path.join(label_output_path, file)
            resize_and_save(image, size, output_file_path)


def resize_and_save(img, size, save_file_path):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # compute padding
    top = (size - new_h) // 2
    bottom = size - new_h - top
    left = (size - new_w) // 2
    right = size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )
        
    # Save cropped image
    cv2.imwrite(save_file_path, padded)

if __name__ == "__main__":
    main()
