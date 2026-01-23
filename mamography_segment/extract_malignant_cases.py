#!/usr/bin/env python3
"""
Extract malignant cases for segmentation training.
This script identifies patient folders with 2 images (original + mask) 
and organizes them for segmentation training.
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def identify_malignant_folders(dataset_root):
    """
    Identify patient folders that contain 2 images (malignant cases with masks).
    
    Args:
        dataset_root: Path to the dataset folder containing csv/ and jpeg/
        
    Returns:
        List of malignant patient folder paths
    """
    jpeg_dir = Path(dataset_root) / "jpeg"
    
    if not jpeg_dir.exists():
        print(f"❌ JPEG directory not found: {jpeg_dir}")
        return []
    
    malignant_folders = []
    benign_folders = []
    
    print("🔍 Analyzing patient folders...")
    
    # Check each patient folder
    for patient_folder in jpeg_dir.iterdir():
        if patient_folder.is_dir():
            # Count image files in this folder
            image_files = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.jpeg")) + list(patient_folder.glob("*.png"))
            
            if len(image_files) == 2:
                malignant_folders.append(patient_folder)
                print(f"  📁 Malignant (2 images): {patient_folder.name}")
            elif len(image_files) == 1:
                benign_folders.append(patient_folder)
                print(f"  📁 Benign (1 image): {patient_folder.name}")
            else:
                print(f"  ⚠️ Unusual folder (has {len(image_files)} images): {patient_folder.name}")
    
    print(f"\n📊 Summary:")
    print(f"  Malignant cases (2 images): {len(malignant_folders)}")
    print(f"  Benign cases (1 image): {len(benign_folders)}")
    
    return malignant_folders

def organize_malignant_cases(malignant_folders, output_root, test_size=0.2, val_size=0.1):
    """
    Organize malignant patient folders into train/val/test splits for segmentation.
    Each patient folder contains both the original image and mask.
    
    Args:
        malignant_folders: List of malignant patient folder paths
        output_root: Where to create the organized dataset
        test_size: Fraction for test set
        val_size: Fraction for validation set
    """
    
    if not malignant_folders:
        print("❌ No malignant cases found to organize!")
        return
    
    output_dir = Path(output_root)
    
    # Create directory structure (just train/val/test, no malignant subdirectory)
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Split folders into train/val/test
    # First split: train vs temp (val+test)
    train_folders, temp_folders = train_test_split(
        malignant_folders, 
        test_size=test_size + val_size, 
        random_state=42
    )
    
    # Second split: val vs test
    if temp_folders:
        val_folders, test_folders = train_test_split(
            temp_folders, 
            test_size=test_size/(test_size + val_size), 
            random_state=42
        )
    else:
        val_folders, test_folders = [], []
    
    # Organize the splits
    splits = {
        'train': train_folders,
        'val': val_folders, 
        'test': test_folders
    }
    
    print(f"\n📁 Organizing malignant patient folders...")
    total_copied = 0
    
    for split_name, folders in splits.items():
        print(f"\n  {split_name.upper()} ({len(folders)} patient folders):")
        split_dir = output_dir / split_name
        
        for patient_folder in folders:
            # Verify this folder has exactly 2 images
            image_files = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.jpeg")) + list(patient_folder.glob("*.png"))
            
            if len(image_files) != 2:
                print(f"    ⚠️ Skipping {patient_folder.name} - has {len(image_files)} images")
                continue
            
            # Copy entire patient folder to split directory
            dest_patient_dir = split_dir / patient_folder.name
            
            try:
                # Copy entire folder with all its contents
                shutil.copytree(patient_folder, dest_patient_dir, dirs_exist_ok=True)
                
                print(f"    ✅ {patient_folder.name}: copied folder with {len(image_files)} images")
                total_copied += 1
                
            except Exception as e:
                print(f"    ❌ Error copying {patient_folder.name}: {e}")
    
    print(f"\n✅ Successfully organized {total_copied} malignant patient folders!")
    
    # Print final structure
    print(f"\n📊 Final Dataset Structure:")
    for split in ['train', 'val', 'test']:
        split_dir = output_dir / split
        patient_folders = [d for d in split_dir.iterdir() if d.is_dir()]
        total_images = sum(len(list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))) 
                          for folder in patient_folders)
        print(f"  {split}/: {len(patient_folders)} patient folders, {total_images} total images")

def main():
    """Main function to organize malignant cases for segmentation."""
    
    # Configuration
    DATASET_ROOT = "dataset"  # Your dataset folder with csv/ and jpeg/
    OUTPUT_ROOT = "data"      # Where to create organized dataset for segmentation
    
    print("🩻 Malignant Case Extraction for Segmentation Training")
    print("=" * 60)
    
    # Step 1: Identify malignant folders (2 images each)
    malignant_folders = identify_malignant_folders(DATASET_ROOT)
    
    if not malignant_folders:
        print("\n❌ No malignant cases found!")
        print("Please check that:")
        print("1. The dataset folder exists and contains jpeg/ subdirectory")
        print("2. Malignant patient folders contain exactly 2 images each")
        return
    
    # Step 2: Organize malignant cases into train/val/test
    organize_malignant_cases(
        malignant_folders=malignant_folders,
        output_root=OUTPUT_ROOT,
        test_size=0.15,   # 15% for test
        val_size=0.15     # 15% for validation  
    )
    
    print(f"\n🎯 Next Steps:")
    print("1. Check the 'data' folder for your organized segmentation dataset")
    print("2. Run the segmentator_trainer.ipynb notebook to start training")
    print("3. The script automatically pairs images with their masks")

if __name__ == "__main__":
    main()