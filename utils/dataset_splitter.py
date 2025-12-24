import os
import shutil
import glob

def split_dataset():
    # Define paths
    # Using absolute paths as observed in the user's workspace
    # The user requested to process 'Top_view' folder. 
    # Based on directory structure, the images are in 'train'.
    base_dir = r"g:\work\Part\Virtual_fence\Datasets\Top_view"
    source_dir = os.path.join(base_dir, "train")
    dest_dir = os.path.join(base_dir, "vali")

    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {dest_dir}")

    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    else:
        print(f"Directory '{dest_dir}' already exists.")

    # Get all image files (assuming jpg for now, can be expanded)
    # Using sorted() to ensure deterministic order which is crucial for "every 8th"
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    # Check for multiple extensions just in case
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, ext)))
    
    # Sort the files to ensure we pick images in a consistent sequence order
    image_files.sort()
    
    if not image_files:
        print("No image files found in source directory.")
        return

    print(f"Found {len(image_files)} images. Starting move operation...")

    count_moved = 0
    
    # Iterate through images and pick every 8th one (indices 0, 8, 16, ...)
    for i, img_path in enumerate(image_files):
        if i % 8 == 0:
            # Construct paths
            img_filename = os.path.basename(img_path)
            dst_img_path = os.path.join(dest_dir, img_filename)
            
            # Handle corresponding annotation file
            # Assuming annotation has same basename but .txt extension
            base_name, _ = os.path.splitext(img_filename)
            txt_filename = base_name + ".txt"
            src_txt_path = os.path.join(source_dir, txt_filename)
            dst_txt_path = os.path.join(dest_dir, txt_filename)
            
            # Move image
            try:
                shutil.move(img_path, dst_img_path)
                # print(f"Moved image: {img_filename}")
            except Exception as e:
                print(f"Error moving image {img_filename}: {e}")
                continue

            # Move annotation if it exists
            if os.path.exists(src_txt_path):
                try:
                    shutil.move(src_txt_path, dst_txt_path)
                    # print(f"Moved annotation: {txt_filename}")
                except Exception as e:
                    print(f"Error moving annotation {txt_filename}: {e}")
            else:
                print(f"Warning: Annotation file not found for {img_filename}")
            
            count_moved += 1

    print(f"Operation complete. Moved {count_moved} pairs of images and annotations to '{dest_dir}'.")

if __name__ == "__main__":
    split_dataset()
