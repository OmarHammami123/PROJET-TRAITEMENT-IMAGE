#!/usr/bin/env python3
"""
Dataset organization script for CBIS-DDSM mammography dataset.
This script reorganizes the dataset into train/val/test splits with all 3 pathology classes.
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def organize_cbis_dataset(dataset_root, output_root, test_size=0.2, val_size=0.1):
    """
    Organize CBIS-DDSM dataset into train/val/test structure with 3 classes.
    
    Args:
        dataset_root: Path to your current 'dataset' folder
        output_root: Where to create the organized dataset
        test_size: Fraction for test set
        val_size: Fraction for validation set (from remaining after test split)
    """
    
    # Paths
    csv_dir = Path(dataset_root) / "csv"
    jpeg_dir = Path(dataset_root) / "jpeg"
    output_dir = Path(output_root)
    
    print("🔍 Loading CSV files...")
    
    # Load both mass and calc datasets
    try:
        mass_train = pd.read_csv(csv_dir / "mass_case_description_train_set.csv")
        mass_test = pd.read_csv(csv_dir / "mass_case_description_test_set.csv")
        calc_train = pd.read_csv(csv_dir / "calc_case_description_train_set.csv")
        calc_test = pd.read_csv(csv_dir / "calc_case_description_test_set.csv")
        
        print(f"✅ Loaded {len(mass_train)} mass train cases")
        print(f"✅ Loaded {len(mass_test)} mass test cases")
        print(f"✅ Loaded {len(calc_train)} calc train cases")
        print(f"✅ Loaded {len(calc_test)} calc test cases")
        
        # Combine all data
        all_data = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)
        
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return
    
    print(f"\n📊 Total cases: {len(all_data)}")
    print("📊 Pathology distribution:")
    for pathology, count in all_data['pathology'].value_counts().items():
        print(f"   {pathology}: {count}")
    
    # Create a mapping of patient folders to images
    print("\n🔍 Mapping images to labels...")
    
    image_data = []
    jpeg_folders = list(jpeg_dir.glob("*"))
    
    for folder in jpeg_folders:
        if folder.is_dir():
            # Get images in this folder
            images = list(folder.glob("*.jpg"))
            folder_name = folder.name
            
            # Find corresponding entries in CSV
            # Look for entries that match this folder pattern
            matching_entries = all_data[
                all_data['image file path'].str.contains(folder_name, na=False) |
                all_data['cropped image file path'].str.contains(folder_name, na=False)
            ]
            
            if not matching_entries.empty:
                # Use the most common pathology for this patient/folder
                pathology = matching_entries['pathology'].mode().iloc[0]
                patient_id = matching_entries['patient_id'].iloc[0]
                
                for img_path in images:
                    image_data.append({
                        'image_path': img_path,
                        'label': pathology,
                        'patient_id': patient_id,
                        'folder': folder_name
                    })
    
    if not image_data:
        print("❌ No matching images found!")
        return
    
    print(f"✅ Mapped {len(image_data)} images to labels")
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(image_data)
    
    # Print label distribution
    label_counts = df['label'].value_counts()
    print(f"\n📊 Image Label Distribution:")
    for label, count in label_counts.items():
        print(f"   {label}: {count}")
    
    # Split by patient to avoid data leakage
    unique_patients = df['patient_id'].unique()
    
    # Split patients into train/temp, then temp into val/test
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=test_size + val_size, random_state=42, 
        stratify=df.groupby('patient_id')['label'].first()[unique_patients]
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=test_size/(test_size + val_size), random_state=42,
        stratify=df.groupby('patient_id')['label'].first()[temp_patients]
    )
    
    # Create split assignments
    df['split'] = 'train'
    df.loc[df['patient_id'].isin(val_patients), 'split'] = 'val'
    df.loc[df['patient_id'].isin(test_patients), 'split'] = 'test'
    
    # Print split statistics
    print(f"\n📊 Dataset Splits:")
    split_stats = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(split_stats)
    
    # Create directory structure for all 3 classes
    print(f"\n📁 Creating directory structure at {output_dir}...")
    
    # Map pathology labels to folder names
    label_mapping = {
        'BENIGN': 'benign',
        'MALIGNANT': 'malignant', 
        'BENIGN_WITHOUT_CALLBACK': 'benign_without_callback'
    }
    
    for split in ['train', 'val', 'test']:
        for label, folder_name in label_mapping.items():
            target_dir = output_dir / split / folder_name
            target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy images to appropriate folders
    print("\n📋 Copying images...")
    
    copied_count = 0
    for _, row in df.iterrows():
        source_path = row['image_path']
        split = row['split']
        label = row['label']
        
        # Convert label to folder name
        folder_name = label_mapping[label]
        
        # Create unique filename to avoid conflicts
        patient_folder = row['folder']
        original_name = source_path.name
        new_name = f"{patient_folder}_{original_name}"
        
        target_path = output_dir / split / folder_name / new_name
        
        try:
            shutil.copy2(source_path, target_path)
            copied_count += 1
            
            if copied_count % 100 == 0:
                print(f"   Copied {copied_count}/{len(df)} images...")
                
        except Exception as e:
            print(f"   ❌ Error copying {source_path}: {e}")
    
    print(f"\n✅ Successfully organized {copied_count} images!")
    print(f"📁 Dataset organized in: {output_dir}")
    
    # Print final structure
    print(f"\n📊 Final Dataset Structure:")
    for split in ['train', 'val', 'test']:
        print(f"   {split}/")
        for label, folder_name in label_mapping.items():
            count = len(list((output_dir / split / folder_name).glob("*.jpg")))
            print(f"     {folder_name}/: {count} images")

    # Cleanup source
    print("\n🗑️ Cleaning up source dataset...")
    try:
        if jpeg_dir.exists():
            print(f"   Removing {jpeg_dir}...")
            shutil.rmtree(jpeg_dir)
            print("   ✅ Source JPEG folder removed")
        else:
            print("   ⚠️ Source JPEG folder not found")
            
        # Optional: Remove the csv folder if you want to clean everything
        # if csv_dir.exists():
        #     shutil.rmtree(csv_dir)
            
        print("✅ Cleanup complete. Duplicate data removed.")
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

if __name__ == "__main__":
    # Configuration
    DATASET_ROOT = "dataset"  # Your current dataset folder
    OUTPUT_ROOT = "data"      # Where to create organized dataset
    
    print("🚀 CBIS-DDSM Dataset Organization (3-Class)")
    print("=" * 50)
    
    organize_cbis_dataset(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        test_size=0.15,  # 15% for test
        val_size=0.15    # 15% for validation
    )
    
    print("\n🎯 Next Steps:")
    print("1. Check the 'data' folder for your organized dataset")
    print("2. Decide if you want 3-class or binary classification:")
    print("   - 3-class: benign, malignant, benign_without_callback")
    print("   - Binary: combine benign classes OR focus on malignant vs all benign")
    print("3. Update your config.yaml accordingly")
    print("4. Start implementing your dataset.py module")
