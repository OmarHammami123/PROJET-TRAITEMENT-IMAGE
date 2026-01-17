"""
Model evaluation script for mammography classification
"""
import yaml
import torch
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR))

from src.data.preprocessing import get_transforms
from src.data.dataset import create_dataloaders
from src.models.efficientnet import create_model
from src.utils.metrics import (
    calculate_metrics, plot_confusion_matrix, 
    plot_roc_curves, print_metrics_summary
)


def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: cuda or cpu
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n🔍 Evaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Collect predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob, class_names)
    
    return metrics, y_true, y_pred, y_prob


def main():
    # Load config
    config_path = SCRIPT_DIR / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Create data loaders
    print("📦 Loading test data...")
    transforms = get_transforms(cfg)
    loaders, sizes, class_names = create_dataloaders(cfg, transforms)
    test_loader = loaders['test']
    print(f"   Test set: {sizes['test']} images")
    print(f"   Classes: {class_names}\n")
    
    # Create model
    print("🧠 Creating model...")
    model = create_model(cfg, device=device)
    
    # Load best checkpoint
    checkpoint_path = Path(cfg['training']['checkpoint_dir']) / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print("   Please train the model first using train.py")
        return
    
    print(f"📥 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"   Checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Best validation loss: {checkpoint['val_loss']:.4f}\n")
    
    # Evaluate model
    metrics, y_true, y_pred, y_prob = evaluate_model(
        model, test_loader, device, class_names
    )
    
    # Print results
    print_metrics_summary(metrics, class_names)
    
    # Create results directory
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Plot confusion matrix
    print("\n📊 Generating visualizations...")
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names,
        save_path=results_dir / 'confusion_matrix.png'
    )
    
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names,
        save_path=results_dir / 'confusion_matrix_normalized.png',
        normalize=True
    )
    
    # Plot ROC curves
    plot_roc_curves(
        y_true, y_prob, class_names,
        save_path=results_dir / 'roc_curves.png'
    )
    
    # Save metrics to file
    metrics_file = results_dir / 'test_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MAMMOGRAPHY CLASSIFICATION - TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: {cfg['model']['name']}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n")
        f.write(f"Test samples: {sizes['test']}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision (macro): {metrics['precision_macro']:.4f}\n")
        f.write(f"Recall (macro): {metrics['recall_macro']:.4f}\n")
        f.write(f"F1-Score (macro): {metrics['f1_macro']:.4f}\n")
        if 'roc_auc_macro' in metrics:
            f.write(f"ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(metrics['classification_report'])
    
    print(f"✅ Metrics saved to {metrics_file}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Results saved in: {results_dir}")
    print(f"   - confusion_matrix.png")
    print(f"   - confusion_matrix_normalized.png")
    print(f"   - roc_curves.png")
    print(f"   - test_metrics.txt")
    print()


if __name__ == '__main__':
    main()
