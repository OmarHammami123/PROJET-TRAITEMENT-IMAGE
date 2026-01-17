import os
import shutil
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path("mammography-classification")
DATA_DIR = BASE_DIR / "data"
CSV_DIR = BASE_DIR / "dataset/csv"
# We move them to a separate folder so they don't interfere with classification training
# but are kept safe for potential future use (segmentation).
DEST_DIR = DATA_DIR / "separated_data" 

# Ensure destination directories exist
(DEST_DIR / "masks").mkdir(parents=True, exist_ok=True)
(DEST_DIR / "crops").mkdir(parents=True, exist_ok=True)
(DEST_DIR / "unknown").mkdir(parents=True, exist_ok=True)

def load_metadata():
    print("Loading metadata...")
    try:
        mass_train = pd.read_csv(CSV_DIR / "mass_case_description_train_set.csv")
        mass_test = pd.read_csv(CSV_DIR / "mass_case_description_test_set.csv")
        calc_train = pd.read_csv(CSV_DIR / "calc_case_description_train_set.csv")
        calc_test = pd.read_csv(CSV_DIR / "calc_case_description_test_set.csv")
        
        all_meta = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)
        print(f"✅ Loaded metadata for {len(all_meta)} cases")
        return all_meta
    except Exception as e:
        print(f"❌ Could not load CSVs: {e}")
        return pd.DataFrame()

def identify_image_type(filename, metadata):
    # Extract UID from filename (format: UID_original-name.jpg)
    # The filename format from organize_dataset.py was likely preserving the UID at the start
    parts = filename.rsplit('_', 1)
    if len(parts) != 2:
        return "Unknown"
        
    folder_uid = parts[0]
    
    # Find matches in metadata
    matches = metadata[
        metadata['image file path'].str.contains(folder_uid, na=False) |
        metadata['cropped image file path'].str.contains(folder_uid, na=False) |
        metadata['ROI mask file path'].str.contains(folder_uid, na=False)
    ]
    
    if matches.empty:
        return "Unknown"
    
    row = matches.iloc[0]
    
    # Priority check to determine type
    # We check if the UID is part of the specific path in the CSV
    if folder_uid in str(row['ROI mask file path']):
        return "ROI Mask"
    elif folder_uid in str(row['cropped image file path']):
        return "Cropped Image"
    elif folder_uid in str(row['image file path']):
        return "Full Mammogram"
    
    return "Ambiguous"

def clean_dataset():
    metadata = load_metadata()
    if metadata.empty:
        print("Aborting: No metadata found.")
        return

    stats = {"Moved Masks": 0, "Moved Crops": 0, "Moved Unknown": 0, "Kept Full": 0}

    # Iterate through train, val, test
    for split in ["train", "val", "test"]:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            continue
            
        print(f"\nCleaning {split} set...")
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir(): continue
            
            print(f"  Processing {class_dir.name}...")
            images = list(class_dir.glob("*.jpg"))
            
            for img_path in images:
                img_type = identify_image_type(img_path.name, metadata)
                
                if img_type == "ROI Mask":
                    shutil.move(str(img_path), str(DEST_DIR / "masks" / img_path.name))
                    stats["Moved Masks"] += 1
                elif img_type == "Cropped Image":
                    shutil.move(str(img_path), str(DEST_DIR / "crops" / img_path.name))
                    stats["Moved Crops"] += 1
                elif img_type == "Unknown" or img_type == "Ambiguous":
                    shutil.move(str(img_path), str(DEST_DIR / "unknown" / img_path.name))
                    stats["Moved Unknown"] += 1
                else:
                    stats["Kept Full"] += 1

    print("\nCleaning Complete!")
    print(f"Summary: {stats}")
    print(f"Cleaned training data (Full Mammograms) remains in {DATA_DIR}")
    print(f"Separated Masks and Crops are in {DEST_DIR}")

if __name__ == "__main__":
    clean_dataset()
