#!/usr/bin/env python3
"""
Mammography Segmentation Training Script
Train a UNet model for segmenting malignant cases in mammography images.
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import json

class PatientFolderDataset(Dataset):
    def __init__(self, split_dir, image_size=(256, 256)):
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.patient_folders = []
        
        # Find all patient folders in the split directory
        if self.split_dir.exists():
            self.patient_folders = [f for f in self.split_dir.iterdir() if f.is_dir()]
        
        # Validate each patient folder has exactly 2 images
        self.valid_patients = []
        for patient_folder in self.patient_folders:
            image_files = list(patient_folder.glob("*.jpg")) + list(patient_folder.glob("*.jpeg")) + list(patient_folder.glob("*.png"))
            
            if len(image_files) == 2:
                # Identify by filename pattern: 1- prefix for mammography, 2- prefix for mask
                mammography_img = None
                mask_img = None
                
                for img_file in image_files:
                    if img_file.name.startswith('1-'):
                        mammography_img = img_file
                    elif img_file.name.startswith('2-'):
                        mask_img = img_file
                
                if mammography_img and mask_img:
                    self.valid_patients.append((mammography_img, mask_img))
        
        print(f"Found {len(self.valid_patients)} valid patient cases in {split_dir}")
        
    def __len__(self):
        return len(self.valid_patients)
    
    def __getitem__(self, idx):
        original_path, mask_path = self.valid_patients[idx]
        
        # Load and preprocess original image
        image = Image.open(original_path).convert('L')  # Convert to grayscale
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load and preprocess mask
        mask = Image.open(mask_path).convert('L')  # Convert to grayscale
        mask = mask.resize(self.image_size, Image.Resampling.NEAREST)  # Use nearest for masks
        mask = np.array(mask, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        mask = (mask > 0.5).astype(np.float32)  # Binarize
        
        # Convert to tensors and add channel dimension
        image = torch.tensor(image).unsqueeze(0)  # Shape: (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0)    # Shape: (1, H, W)
        
        return {"image": image, "mask": mask}

def create_dataloaders(data_dir, batch_size=4):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data directory containing train/val/test folders
        batch_size: Batch size for dataloaders
        
    Returns:
        train_loader, val_loader (or None if not found)
    """
    # Define paths for train and validation splits
    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"
    
    # Create datasets
    train_dataset = None
    val_dataset = None
    
    if train_dir.exists():
        train_dataset = PatientFolderDataset(train_dir)
    else:
        print("Training directory not found:", train_dir.absolute())
    
    if val_dir.exists():
        val_dataset = PatientFolderDataset(val_dir)
    else:
        print("Validation directory not found:", val_dir.absolute())
    
    # Create data loaders with GPU optimizations
    train_loader = None
    val_loader = None
    
    # Set num_workers and pin_memory for GPU efficiency
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()
    
    if train_dataset and len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        print(f"Training loader created with {len(train_dataset)} samples")
    else:
        print("No training data found! Please run extract_malignant_cases.py first.")
    
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
        print(f"Validation loader created with {len(val_dataset)} samples")
    else:
        print("No validation data found! Please run extract_malignant_cases.py first.")
    
    return train_loader, val_loader

def create_model(device):
    """
    Create and initialize the UNet model.
    
    Args:
        device: Device to place the model on
        
    Returns:
        model, criterion, optimizer, dice_metric
    """
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    
    return model, criterion, optimizer, dice_metric

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: UNet model
        train_loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Clear GPU cache periodically to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return epoch_loss / len(train_loader)

def validate_epoch(model, val_loader, dice_metric, device):
    """
    Validate for one epoch.
    
    Args:
        model: UNet model
        val_loader: Validation dataloader
        dice_metric: Dice metric calculator
        device: Device
        
    Returns:
        Average dice score for the epoch
    """
    model.eval()
    dice_metric.reset()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            dice_metric(y_pred=preds, y=masks)
    
    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()
    
    return val_dice

def train_model(data_dir="data", num_epochs=20, batch_size=4, save_dir="saved_models"):
    """
    Main training function.
    
    Args:
        data_dir: Path to data directory
        num_epochs: Number of training epochs
        batch_size: Batch size
        save_dir: Directory to save the best model
    """
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_segmentation_model.pth")
    
    print(f"Mammography Segmentation Training")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Save directory: {save_dir}")
    print()
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    
    # Check if we have data before starting training
    if train_loader is None or val_loader is None:
        print("Cannot start training without data!")
        print("\nNext steps:")
        print("1. Run extract_malignant_cases.py to organize malignant patient folders")
        print("2. Check that the data directory structure is correct")
        print("3. Ensure patient folders contain exactly 2 images each (original + mask)")
        return
    
    # Create model
    model, criterion, optimizer, dice_metric = create_model(device)
    
    print(f"Starting training with {len(train_loader.dataset)} training samples and {len(val_loader.dataset)} validation samples")
    print()
    
    # Training loop
    best_dice = 0.0
    training_logs = {
        'train_loss': [],
        'val_dice': [],
        'epochs': []
    }
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_dice = validate_epoch(model, val_loader, dice_metric, device)
        
        # Print GPU memory usage if available
        gpu_memory_info = ""
        if torch.cuda.is_available():
            gpu_memory_info = f" | GPU Mem: {torch.cuda.memory_allocated()/1e6:.0f}MB"
        
        print(f"Train Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}{gpu_memory_info}")
        
        # Log metrics
        training_logs['train_loss'].append(train_loss)
        training_logs['val_dice'].append(val_dice)
        training_logs['epochs'].append(epoch + 1)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (Dice: {best_dice:.4f})")
        
        print()
    
    print("Training complete!")
    print(f"Best Dice score: {best_dice:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    # Save training logs
    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"Training logs saved to: {log_path}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train mammography segmentation model")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Path to data directory (default: data)")
    parser.add_argument("--epochs", type=int, default=20, 
                       help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=4, 
                       help="Batch size (default: 4)")
    parser.add_argument("--save_dir", type=str, default="saved_models", 
                       help="Directory to save models (default: saved_models)")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()