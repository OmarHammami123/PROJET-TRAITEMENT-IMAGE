#!/usr/bin/env python3
"""
Mammography Segmentation Prediction Visualization
Load the trained UNet model and visualize predictions with mask overlays on test images.
"""

import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse
import os
import random
from matplotlib.colors import LinearSegmentedColormap
import cv2

def load_model(model_path, device):
    """
    Load the trained UNet model.
    
    Args:
        model_path: Path to the saved model weights
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    # Load the trained weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        return None
    
    return model

def preprocess_image(image_path, image_size=(256, 256)):
    """
    Preprocess an image for model inference.
    
    Args:
        image_path: Path to the image file
        image_size: Target size for resizing
        
    Returns:
        Preprocessed image tensor and original image array
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    original_image = np.array(image)
    
    # Resize for model input
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    # Convert to tensor and add batch and channel dimensions
    image_tensor = torch.tensor(image_array).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    return image_tensor, original_image

def predict_mask(model, image_tensor, device):
    """
    Generate a prediction mask for an input image.
    
    Args:
        model: Trained UNet model
        image_tensor: Preprocessed image tensor
        device: Device for inference
        
    Returns:
        Predicted mask as numpy array
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        
        # Apply sigmoid and threshold to get binary mask
        pred_mask = torch.sigmoid(output) > 0.7
        pred_mask = pred_mask.squeeze().cpu().numpy()
        
        # Get confidence scores as well
        confidence = torch.sigmoid(output).squeeze().cpu().numpy()
        
    return pred_mask, confidence

def load_ground_truth_mask(mask_path, image_size=(256, 256)):
    """
    Load and preprocess the ground truth mask.
    
    Args:
        mask_path: Path to the mask file
        image_size: Target size for resizing
        
    Returns:
        Ground truth mask as numpy array
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize(image_size, Image.Resampling.NEAREST)
    mask_array = np.array(mask, dtype=np.float32) / 255.0
    mask_array = (mask_array > 0.5).astype(np.float32)
    
    return mask_array

def create_overlay_visualization(original_image, pred_mask, confidence=None, gt_mask=None, 
                                alpha=0.6, original_size=None):
    """
    Create an overlay visualization of the mask on the original image.
    
    Args:
        original_image: Original mammography image
        pred_mask: Predicted binary mask
        confidence: Confidence scores (optional)
        gt_mask: Ground truth mask (optional)
        alpha: Transparency for overlay
        original_size: Original image size for proper scaling
        
    Returns:
        RGB image with overlay
    """
    # Resize mask to match original image size if needed
    if original_size and pred_mask.shape != original_size:
        pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                              (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
        if confidence is not None:
            confidence = cv2.resize(confidence, 
                                  (original_size[1], original_size[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        if gt_mask is not None:
            gt_mask = cv2.resize(gt_mask.astype(np.uint8), 
                               (original_size[1], original_size[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Normalize original image to [0, 1]
    if original_image.max() > 1:
        original_image = original_image.astype(np.float32) / 255.0
    
    # Create RGB version of the original image
    rgb_image = np.stack([original_image, original_image, original_image], axis=-1)
    
    # Create mask overlay
    overlay = rgb_image.copy()
    
    # Add predicted mask in red
    overlay[pred_mask > 0, 0] = 1.0  # Red channel
    overlay[pred_mask > 0, 1] = 0.0  # Green channel
    overlay[pred_mask > 0, 2] = 0.0  # Blue channel
    
    # If ground truth is available, add it in green where it doesn't overlap with prediction
    if gt_mask is not None:
        # Green for ground truth areas not predicted
        gt_only = (gt_mask > 0) & (pred_mask == 0)
        overlay[gt_only, 0] = 0.0  # Red channel
        overlay[gt_only, 1] = 1.0  # Green channel
        overlay[gt_only, 2] = 0.0  # Blue channel
        
        # Yellow for overlapping areas (red + green = yellow)
        overlap = (gt_mask > 0) & (pred_mask > 0)
        overlay[overlap, 0] = 1.0  # Red channel
        overlay[overlap, 1] = 1.0  # Green channel
        overlay[overlap, 2] = 0.0  # Blue channel
    
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

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate segmentation metrics.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dictionary containing metrics
    """
    if gt_mask is None:
        return {}
    
    # Flatten masks
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    
    # Calculate intersection and union
    intersection = np.sum(pred_flat * gt_flat)
    union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
    
    # Calculate metrics
    dice = (2.0 * intersection) / (np.sum(pred_flat) + np.sum(gt_flat)) if (np.sum(pred_flat) + np.sum(gt_flat)) > 0 else 0
    iou = intersection / union if union > 0 else 0
    
    # Sensitivity (Recall) and Specificity
    tp = intersection
    fn = np.sum(gt_flat) - intersection
    fp = np.sum(pred_flat) - intersection
    tn = len(gt_flat) - tp - fn - fp
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def plot_predictions(test_images, model, device, save_dir="results", 
                    show_plots=True, image_size=(256, 256)):
    """
    Create and save prediction visualizations.
    
    Args:
        test_images: List of (original_image_path, mask_image_path) tuples
        model: Trained UNet model
        device: Device for inference
        save_dir: Directory to save plots
        show_plots: Whether to display plots on screen
        image_size: Size for model input
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Mammography Segmentation Predictions\n(Red: Predicted, Green: Ground Truth Only, Yellow: Overlap)', 
                 fontsize=16, fontweight='bold')
    
    all_metrics = []
    
    for idx, (original_path, mask_path) in enumerate(test_images):
        # Load and preprocess the image
        image_tensor, original_image = preprocess_image(original_path, image_size)
        
        # Load ground truth mask
        gt_mask = load_ground_truth_mask(mask_path, image_size)
        
        # Make prediction
        pred_mask, confidence = predict_mask(model, image_tensor, device)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        # Create overlay visualization
        overlay = create_overlay_visualization(
            original_image, pred_mask, confidence, gt_mask, 
            alpha=0.6, original_size=(original_image.shape[0], original_image.shape[1])
        )
        
        # Plot original image
        axes[0, idx].imshow(original_image, cmap='gray')
        axes[0, idx].set_title(f'Original Image {idx+1}', fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot overlay
        axes[1, idx].imshow(overlay)
        
        # Create title with metrics
        title = f'Prediction {idx+1}'
        if metrics:
            title += f'\nDice: {metrics["dice"]:.3f}, IoU: {metrics["iou"]:.3f}'
        
        axes[1, idx].set_title(title, fontweight='bold')
        axes[1, idx].axis('off')
        
        # Add confidence information if available
        if confidence is not None:
            max_conf = np.max(confidence[pred_mask > 0]) if np.any(pred_mask > 0) else 0
            axes[1, idx].text(0.02, 0.98, f'Max Conf: {max_conf:.3f}', 
                             transform=axes[1, idx].transAxes, 
                             verticalalignment='top', fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Add legend
    legend_elements = [
        patches.Patch(color='red', alpha=0.6, label='Predicted Mask'),
        patches.Patch(color='green', alpha=0.6, label='Ground Truth Only'),
        patches.Patch(color='yellow', alpha=0.6, label='Overlap (TP)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'segmentation_predictions.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Prediction visualization saved to: {plot_path}")
    
    # Print summary metrics
    if all_metrics and any(metrics for metrics in all_metrics):
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        
        valid_metrics = [m for m in all_metrics if m]
        if valid_metrics:
            avg_dice = np.mean([m['dice'] for m in valid_metrics])
            avg_iou = np.mean([m['iou'] for m in valid_metrics])
            avg_sensitivity = np.mean([m['sensitivity'] for m in valid_metrics])
            avg_specificity = np.mean([m['specificity'] for m in valid_metrics])
            
            print(f"Average Dice Score: {avg_dice:.4f}")
            print(f"Average IoU: {avg_iou:.4f}")
            print(f"Average Sensitivity: {avg_sensitivity:.4f}")
            print(f"Average Specificity: {avg_specificity:.4f}")
            
            print("\nIndividual Results:")
            for i, metrics in enumerate(valid_metrics):
                print(f"Image {i+1}: Dice={metrics['dice']:.3f}, IoU={metrics['iou']:.3f}, "
                      f"Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}")
        
        print("="*50)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_confidence_heatmap(test_images, model, device, save_dir="results", 
                             show_plots=True, image_size=(256, 256)):
    """
    Create confidence heatmap visualizations.
    
    Args:
        test_images: List of (original_image_path, mask_image_path) tuples
        model: Trained UNet model
        device: Device for inference
        save_dir: Directory to save plots
        show_plots: Whether to display plots on screen
        image_size: Size for model input
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    fig.suptitle('Prediction Confidence Heatmaps', fontsize=16, fontweight='bold')
    
    # Custom colormap for confidence
    colors = ['black', 'blue', 'green', 'yellow', 'red']
    confidence_cmap = LinearSegmentedColormap.from_list('confidence', colors)
    
    for idx, (original_path, mask_path) in enumerate(test_images):
        # Load and preprocess the image
        image_tensor, original_image = preprocess_image(original_path, image_size)
        
        # Make prediction
        pred_mask, confidence = predict_mask(model, image_tensor, device)
        
        # Plot original image
        axes[0, idx].imshow(original_image, cmap='gray')
        axes[0, idx].set_title(f'Original Image {idx+1}', fontweight='bold')
        axes[0, idx].axis('off')
        
        # Resize confidence to match original image size
        if confidence.shape != original_image.shape:
            confidence_resized = cv2.resize(confidence, 
                                          (original_image.shape[1], original_image.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
        else:
            confidence_resized = confidence
        
        # Plot confidence heatmap
        im = axes[1, idx].imshow(confidence_resized, cmap=confidence_cmap, vmin=0, vmax=1)
        axes[1, idx].set_title(f'Confidence Map {idx+1}\nMax: {np.max(confidence_resized):.3f}, '
                              f'Mean: {np.mean(confidence_resized):.3f}', fontweight='bold')
        axes[1, idx].axis('off')
        
        # Add colorbar for the first image
        if idx == 0:
            cbar = plt.colorbar(im, ax=axes[1, idx], shrink=0.8)
            cbar.set_label('Confidence', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # Save the plot
    heatmap_path = os.path.join(save_dir, 'confidence_heatmaps.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Confidence heatmap saved to: {heatmap_path}")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Visualize mammography segmentation predictions")
    parser.add_argument("--model_path", type=str, default="saved_models/best_segmentation_model.pth",
                       help="Path to saved model weights")
    parser.add_argument("--test_dir", type=str, default="data/test",
                       help="Path to test data directory")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="Directory to save plots")
    parser.add_argument("--num_images", type=int, default=4,
                       help="Number of test images to visualize (default: 4)")
    parser.add_argument("--show", action="store_true", default=False,
                       help="Display plots on screen")
    parser.add_argument("--confidence", action="store_true", default=False,
                       help="Also create confidence heatmaps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible image selection")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print("Mammography Segmentation Prediction Visualization")
    print("="*52)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, device)
    if model is None:
        return
    
    # Get test images
    print(f"Loading test images from: {args.test_dir}")
    test_images = get_test_images(args.test_dir, args.num_images)
    if not test_images:
        print("No valid test images found!")
        return
    
    print(f"Found {len(test_images)} valid test cases")
    
    # Create visualizations
    print(f"Creating prediction visualizations...")
    plot_predictions(test_images, model, device, args.save_dir, args.show)
    
    if args.confidence:
        print(f"Creating confidence heatmaps...")
        create_confidence_heatmap(test_images, model, device, args.save_dir, args.show)
    
    print("\nVisualization complete!")
    print(f"Results saved to: {args.save_dir}")

if __name__ == "__main__":
    main()