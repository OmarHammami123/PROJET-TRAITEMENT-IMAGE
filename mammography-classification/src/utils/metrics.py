"""
Metrics utilities for model evaluation
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    Calculate comprehensive classification metrics
    
    Args:
        y_true: True labels (numpy array or tensor)
        y_pred: Predicted labels (numpy array or tensor)
        y_prob: Prediction probabilities (optional, for ROC-AUC)
        class_names: List of class names
    
    Returns:
        Dictionary of metrics
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    if class_names:
        for i, name in enumerate(class_names):
            metrics[f'precision_{name}'] = precision_per_class[i]
            metrics[f'recall_{name}'] = recall_per_class[i]
            metrics[f'f1_{name}'] = f1_per_class[i]
    
    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2:  # Multi-class
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:  # Binary
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            pass
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # Classification report (string)
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return metrics


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=False):
    """
    Plot confusion matrix with seaborn heatmap
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure (optional)
        normalize: Whether to normalize values
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    Plot ROC curves for multi-class classification
    
    Args:
        y_true: True labels (1D array)
        y_prob: Prediction probabilities (n_samples, n_classes)
        class_names: List of class names
        save_path: Path to save figure (optional)
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.cpu().numpy()
    
    n_classes = len(class_names)
    
    # Binarize labels for multi-class ROC
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {auc:.3f})')
    
    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ ROC curves saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation metrics over epochs
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Path to save figure (optional)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_metrics_summary(metrics, class_names=None):
    """
    Print formatted metrics summary
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("📊 EVALUATION METRICS SUMMARY")
    print("="*60)
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision: {metrics['precision_macro']:.4f}")
    print(f"   Recall:    {metrics['recall_macro']:.4f}")
    print(f"   F1-Score:  {metrics['f1_macro']:.4f}")
    
    if 'roc_auc_macro' in metrics:
        print(f"   ROC-AUC:   {metrics['roc_auc_macro']:.4f}")
    
    if class_names:
        print(f"\n📋 Per-Class Performance:")
        for name in class_names:
            print(f"\n   {name.upper()}:")
            print(f"      Precision: {metrics[f'precision_{name}']:.4f}")
            print(f"      Recall:    {metrics[f'recall_{name}']:.4f}")
            print(f"      F1-Score:  {metrics[f'f1_{name}']:.4f}")
    
    print("\n" + "="*60)
    print("📄 Classification Report:")
    print("="*60)
    print(metrics['classification_report'])
    print("="*60 + "\n")
