#!/usr/bin/env python3
"""
Mammography Ground Truth Visualization
Display original mammography images with their ground truth masks overlayed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse
import os
import random
import cv2

def preprocess_image(image_path, target_size=None):
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        target_size: Optional target size for resizing
        
    Returns:
        Image array
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image_array = np.array(image, dtype=np.float32)
    
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    if image_array.max() > 1:
        image_array = image_array / 255.0
    
    return image_array

def load_ground_truth_mask(mask_path, target_size=None):
    """
    Load and preprocess the ground truth mask.
    
    Args:
        mask_path: Path to the mask file
        target_size: Optional target size for resizing
        
    Returns:
        Ground truth mask as numpy array
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('L')
    
    if target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    
    mask_array = np.array(mask, dtype=np.float32)
    
    # Normalize and binarize
    if mask_array.max() > 1:
        mask_array = mask_array / 255.0
    mask_array = (mask_array > 0.5).astype(np.float32)
    
    return mask_array

def create_mask_overlay(original_image, gt_mask, alpha=0.6):
    """
    Create an overlay visualization of the ground truth mask on the original image.
    
    Args:
        original_image: Original mammography image
        gt_mask: Ground truth binary mask
        alpha: Transparency for overlay
        
    Returns:
        RGB image with overlay
    """
    # Ensure images are the same size
    if original_image.shape != gt_mask.shape:
        # Resize mask to match original image
        gt_mask = cv2.resize(gt_mask.astype(np.uint8), 
                           (original_image.shape[1], original_image.shape[0]), 
                           interpolation=cv2.INTER_NEAREST).astype(np.float32)
    
    # Create RGB version of the original image
    rgb_image = np.stack([original_image, original_image, original_image], axis=-1)
    
    # Create mask overlay
    overlay = rgb_image.copy()
    
    # Add ground truth mask in red
    overlay[gt_mask > 0, 0] = 1.0  # Red channel
    overlay[gt_mask > 0, 1] = 0.0  # Green channel
    overlay[gt_mask > 0, 2] = 0.0  # Blue channel
    
    # Blend original image with overlay
    result = alpha * overlay + (1 - alpha) * rgb_image
    
    return np.clip(result, 0, 1)

def get_test_images(test_dir, num_images=4):
    """
    Get random test images from the test directory.
    
    Args:
        test_dir: Path to test directory
        num_images: Number of images to select
        
    Returns:
        List of (original_image_path, mask_image_path) tuples
    """
    test_path = Path(test_dir)
    patient_folders = [f for f in test_path.iterdir() if f.is_dir()]
    
    valid_patients = []
    for patient_folder in patient_folders:
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
                valid_patients.append((mammography_img, mask_img))
    
    # Randomly select the requested number of patients
    if len(valid_patients) < num_images:
        print(f"Warning: Only {len(valid_patients)} valid patients found, using all available.")
        return valid_patients
    
    selected = random.sample(valid_patients, num_images)
    return selected

def calculate_mask_statistics(gt_mask):
    """
    Calculate statistics for the ground truth mask.
    
    Args:
        gt_mask: Ground truth binary mask
        
    Returns:
        Dictionary containing mask statistics
    """
    total_pixels = gt_mask.size
    mask_pixels = np.sum(gt_mask > 0)
    mask_percentage = (mask_pixels / total_pixels) * 100
    
    return {
        'mask_pixels': int(mask_pixels),
        'total_pixels': int(total_pixels),
        'mask_percentage': mask_percentage
    }

def plot_ground_truth_overlays(test_images, save_dir="results", show_plots=True):
    """
    Create and save ground truth mask overlay visualizations.
    
    Args:
        test_images: List of (original_image_path, mask_image_path) tuples
        save_dir: Directory to save plots
        show_plots: Whether to display plots on screen
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Mammography Images with Ground Truth Masks\n(Red: Malignant Regions)', 
                 fontsize=16, fontweight='bold')
    
    all_stats = []
    
    for idx, (original_path, mask_path) in enumerate(test_images):
        # Load original image
        original_image = preprocess_image(original_path)
        
        # Load ground truth mask
        gt_mask = load_ground_truth_mask(mask_path)
        
        if gt_mask is None:
            print(f"Warning: Could not load mask for image {idx+1}")
            continue
        
        # Calculate mask statistics
        stats = calculate_mask_statistics(gt_mask)
        all_stats.append(stats)
        
        # Create overlay visualization
        overlay = create_mask_overlay(original_image, gt_mask, alpha=0.6)
        
        # Plot original image
        axes[0, idx].imshow(original_image, cmap='gray')
        axes[0, idx].set_title(f'Original Image {idx+1}', fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot overlay
        axes[1, idx].imshow(overlay)
        axes[1, idx].set_title(f'With Ground Truth Mask {idx+1}\n'
                              f'Malignant: {stats["mask_percentage"]:.1f}% of image', 
                              fontweight='bold')
        axes[1, idx].axis('off')
        
        # Add mask statistics
        axes[1, idx].text(0.02, 0.02, 
                         f'Mask pixels: {stats["mask_pixels"]:,}', 
                         transform=axes[1, idx].transAxes, 
                         verticalalignment='bottom', fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='red', alpha=0.6, label='Malignant Regions (Ground Truth)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'ground_truth_overlays.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Ground truth overlay visualization saved to: {plot_path}")
    
    # Print summary statistics
    if all_stats:
        print("\n" + "="*50)
        print("GROUND TRUTH MASK STATISTICS")
        print("="*50)
        
        avg_percentage = np.mean([s['mask_percentage'] for s in all_stats])
        total_mask_pixels = sum([s['mask_pixels'] for s in all_stats])
        total_pixels = sum([s['total_pixels'] for s in all_stats])
        
        print(f"Average malignant area: {avg_percentage:.2f}% per image")
        print(f"Total malignant pixels: {total_mask_pixels:,}")
        print(f"Total pixels analyzed: {total_pixels:,}")
        
        print("\nIndividual Image Statistics:")
        for i, stats in enumerate(all_stats):
            print(f"Image {i+1}: {stats['mask_percentage']:.1f}% malignant "
                  f"({stats['mask_pixels']:,} pixels)")
        
        print("="*50)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_individual_overlays(test_images, save_dir="results", show_plots=True):
    """
    Create individual overlay images for each test case.
    
    Args:
        test_images: List of (original_image_path, mask_image_path) tuples
        save_dir: Directory to save plots
        show_plots: Whether to display plots on screen
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, (original_path, mask_path) in enumerate(test_images):
        # Load original image
        original_image = preprocess_image(original_path)
        
        # Load ground truth mask
        gt_mask = load_ground_truth_mask(mask_path)
        
        if gt_mask is None:
            print(f"Warning: Could not load mask for image {idx+1}")
            continue
        
        # Calculate mask statistics
        stats = calculate_mask_statistics(gt_mask)
        
        # Create overlay visualization
        overlay = create_mask_overlay(original_image, gt_mask, alpha=0.6)
        
        # Create individual plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Mammography Case {idx+1} - Ground Truth Analysis', 
                     fontsize=14, fontweight='bold')
        
        # Plot original image
        ax1.imshow(original_image, cmap='gray')
        ax1.set_title('Original Mammography', fontweight='bold')
        ax1.axis('off')
        
        # Plot overlay
        ax2.imshow(overlay)
        ax2.set_title(f'With Ground Truth Mask\nMalignant: {stats["mask_percentage"]:.1f}% of image', 
                     fontweight='bold')
        ax2.axis('off')
        
        # Add detailed statistics
        stats_text = f'''Mask Statistics:
• Malignant pixels: {stats["mask_pixels"]:,}
• Total pixels: {stats["total_pixels"]:,}
• Malignant area: {stats["mask_percentage"]:.2f}%'''
        
        fig.text(0.02, 0.02, stats_text, fontsize=12, verticalalignment='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # Add legend
        legend_elements = [
            patches.Patch(color='red', alpha=0.6, label='Malignant Regions')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = os.path.join(save_dir, f'ground_truth_case_{idx+1}.png')
        plt.savefig(individual_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Individual case {idx+1} saved to: {individual_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize mammography ground truth masks")
    parser.add_argument("--test_dir", type=str, default="data/test",
                       help="Path to test data directory")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--num_images", type=int, default=4,
                       help="Number of test images to visualize (default: 4)")
    parser.add_argument("--show", action="store_true", default=False,
                       help="Display plots on screen")
    parser.add_argument("--individual", action="store_true", default=False,
                       help="Create individual plots for each case")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible image selection")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("Mammography Ground Truth Mask Visualization")
    print("="*43)
    
    # Get test images
    print(f"Loading test images from: {args.test_dir}")
    test_images = get_test_images(args.test_dir, args.num_images)
    if not test_images:
        print("No valid test images found!")
        return
    
    print(f"Found {len(test_images)} valid test cases")
    
    # Create visualizations
    print(f"Creating ground truth overlay visualizations...")
    plot_ground_truth_overlays(test_images, args.save_dir, args.show)
    
    if args.individual:
        print(f"Creating individual case visualizations...")
        create_individual_overlays(test_images, args.save_dir, args.show)
    
    print("\nVisualization complete!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()