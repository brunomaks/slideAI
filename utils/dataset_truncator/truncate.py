import zipfile
import os
import shutil

def truncate_dataset(zip_path, output_zip_path, keep_fraction=0.5):
    """
    Truncate a gesture dataset by keeping only a fraction of images per gesture.
    
    Args:
        zip_path: Path to input .zip file
        output_zip_path: Path for output .zip file
        keep_fraction: Fraction of images to keep (0.5 = half)
    """
    # Create temporary extraction directory
    temp_dir = "temp_dataset_extract"
    output_dir = "temp_dataset_truncated"

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        for gesture_folder in os.listdir(temp_dir):
            gesture_path = os.path.join(temp_dir, gesture_folder)
            
            if not os.path.isdir(gesture_path):
                continue
            
            print(f"Processing gesture: {gesture_folder}")
            
            image_files = sorted([f for f in os.listdir(gesture_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            
            num_to_keep = int(len(image_files) * keep_fraction)
            images_to_keep = image_files[:num_to_keep]
            
            print(f"  Total images: {len(image_files)}, Keeping: {num_to_keep}")
            
            output_gesture_path = os.path.join(output_dir, gesture_folder)
            os.makedirs(output_gesture_path, exist_ok=True)
            
            for img_file in images_to_keep:
                src = os.path.join(gesture_path, img_file)
                dst = os.path.join(output_gesture_path, img_file)
                shutil.copy2(src, dst)
        
        print(f"\nCreating truncated zip file: {output_zip_path}")
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)
        
        print(f"\nDone! Truncated dataset saved to: {output_zip_path}")
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

if __name__ == "__main__":
    input_zip = "dataset.zip"
    output_zip = "dataset_half.zip"
    
    truncate_dataset(input_zip, output_zip, keep_fraction=0.5)