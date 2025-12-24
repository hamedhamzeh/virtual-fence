import os
import shutil
import glob

def organize_dataset():
    base_dir = r"g:\work\Part\Virtual_fence\Datasets\Top_view"
    subsets = ['train', 'test', 'val']
    
    # Extensions to identify images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for subset in subsets:
        subset_dir = os.path.join(base_dir, subset)
        
        if not os.path.exists(subset_dir):
            print(f"Directory not found: {subset_dir}. Skipping...")
            continue
            
        print(f"Processing {subset} directory: {subset_dir}")
        
        images_dir = os.path.join(subset_dir, "images")
        labels_dir = os.path.join(subset_dir, "labels")
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # List all files in the directory
        files = os.listdir(subset_dir)
        
        moved_images = 0
        moved_labels = 0
        
        for filename in files:
            file_path = os.path.join(subset_dir, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if ext in image_extensions:
                shutil.move(file_path, os.path.join(images_dir, filename))
                moved_images += 1
            elif ext == '.txt':
                shutil.move(file_path, os.path.join(labels_dir, filename))
                moved_labels += 1
                
        print(f"  Moved {moved_images} images to 'images' folder.")
        print(f"  Moved {moved_labels} labels to 'labels' folder.")

    print("Organization complete.")

if __name__ == "__main__":
    organize_dataset()
