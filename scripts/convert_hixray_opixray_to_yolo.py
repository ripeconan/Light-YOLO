import os
import shutil
from PIL import Image # Used to read image dimensions

# --- Configuration Parameters ---
# Root directory of your datasets (e.g., parent directory containing 'HiXray' and 'OPIXray')
DATASET_ROOT = '/data' 

# Name of the dataset to be processed (e.g., 'HiXray' or 'OPIXray')
CURRENT_DATASET_NAME = 'HiXray' # <--- Modify to 'OPIXray' or 'HiXray'

# Define original image and label subdirectory names for TRAIN and TEST splits
# IMPORTANT: Adjust these based on your actual extracted folder names
ORIGINAL_TRAIN_IMAGES_SUBDIR_NAME = 'train_image'     # e.g., 'train/train_image'
ORIGINAL_TRAIN_LABELS_SUBDIR_NAME = 'train_annotation' # e.g., 'train/train_annotation'
ORIGINAL_TEST_IMAGES_SUBDIR_NAME = 'test_image'       # e.g., 'test/test_image'
ORIGINAL_TEST_LABELS_SUBDIR_NAME = 'test_annotation'   # e.g., 'test/test_annotation'

# Target YOLO format image and label folder names (standardized to 'images' and 'labels')
TARGET_IMAGES_SUBDIR_NAME = 'images'
TARGET_LABELS_SUBDIR_NAME = 'labels'
# Original labels directory will be renamed to 'original_labels' for backup after conversion
BACKUP_LABELS_SUBDIR_NAME = 'original_labels'

# Class list (needs to be manually created or extracted from dataset info)
# Ensure the order matches your desired YOLO class_id (0, 1, 2...)
CLASSES = [
    "Folding_Knife", "Multi-tool_Knife", "Scissor", "Straight_Knife", "Utility_Knife", # OPIXray classes
    "Portable_Charger_1", "Portable_Charger_2", "Mobile_Phone", "Laptop", "Tablet", 
    "Cosmetic", "Water", "Nonmetallic_Lighter" # HiXray classes
]

# --- Helper Function: Bounding Box Conversion ---
def convert_bbox_to_yolo(img_width, img_height, xmin, ymin, xmax, ymax):
    # Converts pixel coordinates to YOLO normalized format
    x_center = ((xmin + xmax) / 2.0) / img_width
    y_center = ((ymin + ymax) / 2.0) / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return (x_center, y_center, width, height)

# --- Main Processing Logic for a Dataset Split (train/test) ---
def process_dataset_split(split_name, dataset_root, dataset_name, classes):
    print(f"\n--- Processing {dataset_name}/{split_name} split ---")

    split_path = os.path.join(dataset_root, dataset_name, split_name)
    if not os.path.exists(split_path):
        print(f"Error: Split directory '{split_path}' not found. Skipping.")
        return

    # Dynamically select original subdirectory names based on split_name
    if split_name == 'train':
        original_images_subdir = ORIGINAL_TRAIN_IMAGES_SUBDIR_NAME
        original_labels_subdir = ORIGINAL_TRAIN_LABELS_SUBDIR_NAME
    elif split_name == 'test':
        original_images_subdir = ORIGINAL_TEST_IMAGES_SUBDIR_NAME
        original_labels_subdir = ORIGINAL_TEST_LABELS_SUBDIR_NAME
    else:
        print(f"Error: Unknown split name '{split_name}'. Expected 'train' or 'test'. Skipping.")
        return

    original_images_path = os.path.join(split_path, original_images_subdir)
    original_labels_path = os.path.join(split_path, original_labels_subdir)
    
    target_images_path = os.path.join(split_path, TARGET_IMAGES_SUBDIR_NAME)
    target_labels_path = os.path.join(split_path, TARGET_LABELS_SUBDIR_NAME)
    backup_labels_path = os.path.join(split_path, BACKUP_LABELS_SUBDIR_NAME)

    # 1. Folder Renaming/Organization
    # Rename original images folder
    if os.path.exists(original_images_path) and not os.path.exists(target_images_path):
        print(f"Renaming '{original_images_path}' to '{target_images_path}'...")
        os.rename(original_images_path, target_images_path)
    elif not os.path.exists(target_images_path):
        print(f"Warning: Original images directory '{original_images_path}' not found for renaming. Assuming target '{target_images_path}' is already set up.")
    else:
        print(f"Images directory '{target_images_path}' already exists or '{original_images_path}' was already renamed.")

    # Rename original labels folder (for backup)
    if os.path.exists(original_labels_path) and not os.path.exists(backup_labels_path):
        print(f"Renaming original labels '{original_labels_path}' to '{backup_labels_path}' for backup...")
        os.rename(original_labels_path, backup_labels_path)
    elif not os.path.exists(backup_labels_path):
        print(f"Warning: Original labels directory '{original_labels_path}' not found for renaming. Assuming target '{backup_labels_path}' is already set up.")
    else:
        print(f"Backup labels directory '{backup_labels_path}' already exists or '{original_labels_path}' was already renamed.")

    # Create new target labels folder
    os.makedirs(target_labels_path, exist_ok=True)
    print(f"Ensured target labels directory '{target_labels_path}' exists.")

    # 2. Annotation File Format Conversion
    if not os.path.exists(backup_labels_path):
        print(f"Error: No backup labels directory found at '{backup_labels_path}'. Cannot proceed with conversion for {split_name}.")
        return

    original_txt_files = [f for f in os.listdir(backup_labels_path) if f.endswith('.txt')]
    
    processed_count = 0
    skipped_count = 0

    for txt_file_name in original_txt_files:
        original_txt_path = os.path.join(backup_labels_path, txt_file_name)
        yolo_txt_path = os.path.join(target_labels_path, txt_file_name)

        with open(original_txt_path, 'r') as f_in, open(yolo_txt_path, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) != 6:
                    print(f"Warning: Skipping malformed line in {txt_file_name}: '{line.strip()}' (Expected 'img_filename class_name xmin ymin xmax ymax')")
                    skipped_count += 1
                    continue
                
                img_filename_from_txt = parts[0]
                class_name = parts[1]
                xmin, ymin, xmax, ymax = map(float, parts[2:])

                img_full_path = os.path.join(target_images_path, img_filename_from_txt)
                if not os.path.exists(img_full_path):
                    print(f"Warning: Image '{img_full_path}' not found for annotation '{txt_file_name}'. Skipping object '{class_name}'.")
                    skipped_count += 1
                    continue
                
                try:
                    img = Image.open(img_full_path)
                    img_width, img_height = img.size
                except Exception as e:
                    print(f"Error reading image '{img_full_path}': {e}. Skipping object '{class_name}'.")
                    skipped_count += 1
                    continue

                if class_name not in classes:
                    print(f"Warning: Class '{class_name}' not found in CLASSES list. Skipping object in {txt_file_name}.")
                    skipped_count += 1
                    continue

                class_id = classes.index(class_name)

                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, xmin, ymin, xmax, ymax)
                
                f_out.write(f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            processed_count += 1
    
    print(f"Processed {processed_count} label files. Skipped {skipped_count} objects due to warnings/errors.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Create classes.txt file
    dataset_base_path = os.path.join(DATASET_ROOT, CURRENT_DATASET_NAME)
    classes_txt_path = os.path.join(dataset_base_path, 'classes.txt')
    
    if not os.path.exists(dataset_base_path):
        print(f"Error: Dataset base directory '{dataset_base_path}' not found. Please check DATASET_ROOT and CURRENT_DATASET_NAME.")
    else:
        with open(classes_txt_path, 'w') as f:
            for cls_name in CLASSES:
                f.write(cls_name + '\n')
        print(f"Created classes.txt at: {classes_txt_path}")

        # 2. Process train and test splits
        process_dataset_split('train', DATASET_ROOT, CURRENT_DATASET_NAME, CLASSES)
        process_dataset_split('test', DATASET_ROOT, CURRENT_DATASET_NAME, CLASSES)

    print("\nAll conversions attempted. Please manually verify the output directory structure and label files.")
    print(f"Target structure example (train): {os.path.join(dataset_base_path, 'train', TARGET_IMAGES_SUBDIR_NAME)} and {os.path.join(dataset_base_path, 'train', TARGET_LABELS_SUBDIR_NAME)}")
    print(f"Target structure example (test): {os.path.join(dataset_base_path, 'test', TARGET_IMAGES_SUBDIR_NAME)} and {os.path.join(dataset_base_path, 'test', TARGET_LABELS_SUBDIR_NAME)}")
